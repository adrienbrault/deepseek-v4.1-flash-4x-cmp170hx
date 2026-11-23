#!/usr/bin/env python3
"""Check plugin imports and subclass contracts without importing torch or vLLM."""
import argparse
import ast
import functools
from pathlib import Path
import tomllib


@functools.lru_cache(maxsize=None)
def tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def module_path(root, module):
    base = root.joinpath(*module.split("."))
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()), None)


def statements(nodes):
    for node in nodes:
        yield node
        # Descend through module/class conditionals and optional imports, never
        # through function bodies when identifying a module or class binding.
        if isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.ExceptHandler)):
            for field in ("body", "orelse", "handlers", "finalbody"):
                yield from statements(getattr(node, field, ()))


def bindings(nodes):
    for node in statements(nodes):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                yield alias.asname or alias.name.split(".")[0], node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    yield target.id, node


def imports(source):
    for node in ast.walk(tree(source)):
        if isinstance(node, ast.ImportFrom) and not node.level:
            if node.module and node.module.split(".")[0] == "vllm":
                for alias in node.names:
                    yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "vllm":
                    yield alias.name, None


def resolve(root, module, name, seen=frozenset()):
    if module in seen:
        return None
    seen = seen | {module}
    path = module_path(root, module)
    if path is None:
        return None
    if name is None:
        return path
    result = dict(bindings(tree(path).body)).get(name) or module_path(root, module + "." + name)
    if result is not None:
        return result
    for node in statements(tree(path).body):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
            if node.level:
                parent = package[:len(package) + 1 - node.level]
                origin = ".".join(parent + ([node.module] if node.module else []))
            else:
                origin = node.module
            result = resolve(root, origin, name, seen)
            if result is not None:
                return result
    return None


def signature(node):
    args = node.args
    return (
        tuple(arg.arg for arg in args.posonlyargs),
        tuple(arg.arg for arg in args.args),
        tuple(arg.arg for arg in args.kwonlyargs),
        len(args.defaults),
        tuple(value is not None for value in args.kw_defaults),
        args.vararg is not None,
        args.kwarg is not None,
    )


# Compare against the prior upstream contract as well as checking the strict
# MoE call shape. Plugin methods may specialize a permissive base signature.
BASES = {
    "Exl3Config": ("vllm.model_executor.layers.quantization.base_config", "QuantizationConfig"),
    "Exl3MoEMethod": ("vllm.model_executor.layers.fused_moe.fused_moe_method_base", "FusedMoEMethodBase"),
    "Exl3EmbeddingMethod": ("vllm.model_executor.layers.quantization.base_config", "QuantizeMethodBase"),
    "Exl3LinearMethod": ("vllm.model_executor.layers.linear", "LinearMethodBase"),
    "Exl3StandaloneNemotronLoader": ("vllm.model_executor.model_loader.default_loader", "DefaultModelLoader"),
}


# These wrappers are applied by the general plugin to plugin-owned objects.
# Check that both ends still exist after replaying the image layers.
HOOKS = {
    "k78_compat": ("__init__",),
    "tp_geometry_compat": ("_resolve_tp_geometry",),
    "mixed_k_guard": ("_try_prescan_trellis_shapes",),
    "physical_k_compat": ("build_exl3_fused_state",),
    "deepseek_v41": ("_native_moe_dimensions_supported", "get_quant_method"),
    "runtime_policy": ("_native_moe_max_rows",),
    "uva_offload": ("process_weights_after_loading",),
    "bf16_madv_compat": ("_make_weight_loader",),
}


def methods(cls):
    if not isinstance(cls, ast.ClassDef):
        return {}
    return {name: node for name, node in bindings(cls.body)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def check(plugin, target, pin=None):
    failures = []
    classes = {}
    for source in sorted(plugin.rglob("*.py")):
        for module, name in sorted(set(imports(source)), key=str):
            ok = resolve(target, module, name) is not None
            symbol = module if name is None else module + "." + name
            print(f"{source.relative_to(plugin)} import {symbol} {'OK' if ok else 'MISSING'}")
            if not ok:
                failures.append(symbol)
        classes.update((node.name, node) for node in tree(source).body
                       if isinstance(node, ast.ClassDef) and node.name in BASES)
    for derived, (module, base) in BASES.items():
        if derived not in classes:
            failures.append(derived)
            continue
        new = methods(resolve(target, module, base))
        old = methods(resolve(pin, module, base)) if pin else {}
        if not new:
            failures.append(f"{module}.{base}")
        if derived in ("Exl3MoEMethod", "Exl3LinearMethod"):
            ancestor = BASES["Exl3EmbeddingMethod"]
            new = methods(resolve(target, *ancestor)) | new
            if pin:
                old = methods(resolve(pin, *ancestor)) | old
        for name, override in methods(classes[derived]).items():
            if name in old and (name not in new or signature(old[name]) != signature(new[name])):
                if name not in new or signature(override) != signature(new[name]):
                    failures.append(f"{derived}.{name}: upstream call contract changed")
            if name in new:
                print(f"{derived}.{name} overrides {base}.{name}")
        if derived == "Exl3MoEMethod" and "apply" in new:
            if signature(methods(classes[derived])["apply"]) != signature(new["apply"]):
                failures.append("Exl3MoEMethod.apply: call shape differs")
    exl3 = tree(plugin / "exl3.py")
    local_names = set(dict(bindings(exl3.body)))
    local_names.update(name for cls in classes.values() for name in methods(cls))
    for compat, names in HOOKS.items():
        wrapper = tree(plugin / (compat + ".py"))
        writer_names = {
            node.attr for node in ast.walk(wrapper)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
        }
        writer_names.update(
            node.args[1].value for node in ast.walk(wrapper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "setattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        )
        for name in names:
            if name not in writer_names or name not in local_names:
                failures.append(f"{compat}: monkeypatch target {name} drift")
    op_names = {
        item.value.value
        for node in ast.walk(exl3) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "direct_register_custom_op"
        for item in node.keywords if item.arg == "op_name" and isinstance(item.value, ast.Constant)
    }
    if op_names != {"exl3_linear_forward", "exl3_ngram_lookup", "exl3_ngram_lookup_out"}:
        failures.append("EXL3 custom op registration drift")
    registrations = {
        "vllm.model_executor.layers.quantization": "register_quantization_config",
        "vllm.model_executor.model_loader": "register_model_loader",
        "vllm.utils.torch_utils": "direct_register_custom_op",
        "vllm.plugins": "DEFAULT_PLUGINS_GROUP",
    }
    for module, name in registrations.items():
        if resolve(target, module, name) is None:
            failures.append(f"{module}.{name}: registration seam missing")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True, help="vllm_exl3 package directory")
    parser.add_argument("--vllm-root", type=Path, required=True, help="parent of vllm/")
    parser.add_argument("--pin-root", type=Path, help="parent of the previous vllm/")
    parser.add_argument("--pyproject", type=Path, required=True)
    args = parser.parse_args()
    with args.pyproject.open("rb") as handle:
        entries = tomllib.load(handle)["project"]["entry-points"]["vllm.general_plugins"]
    if entries.get("vllm_exl3") != "vllm_exl3.bf16_madv_compat:register":
        raise SystemExit("Unexpected vllm_exl3 general plugin entry point")
    failures = check(args.plugin, args.vllm_root, args.pin_root)
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS plugin imports, overrides, and registration seams")


if __name__ == "__main__":
    main()
