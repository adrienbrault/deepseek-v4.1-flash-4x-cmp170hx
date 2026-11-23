#!/usr/bin/env python3
"""Audit overlay imports, entry points, private hooks and pin targets without CUDA."""
import argparse
import ast
import configparser
import json
from pathlib import Path
import re
import sys

KIT = Path(__file__).resolve().parents[1]
WORK = KIT.parent / "work"
sys.path.insert(0, str(KIT / "tools"))
from derive_argv import PLUGIN_ALLOWLIST
from install_overlays import OLD_SITE, whole_paths
from overlay_sources import Sources, dotted, parse, signature
from repin_overlays import pin_targets

LOGGER_NAMES = {
    "vllm.dsv41_attn_audit", "vllm.dsv41_stack", "vllm.dsv41_draft_argmax",
}
UPSTREAM = {"vllm", "vllm_exl3"}
GROUP = "vllm.general_plugins"


def external(module):
    return module.split(".")[0] in UPSTREAM


def references(source, roots):
    """Find direct imports, literal import targets and attributes of imported aliases."""
    nodes = list(ast.walk(parse(source)))
    aliases = {}
    refs = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and not node.level and node.module and external(node.module):
            for name in node.names:
                if name.name == "*":
                    raise ValueError(f"Wildcard upstream import: {source}:{node.lineno}")
                ref = f"{node.module}:{name.name}"
                refs.add(("import", ref))
                resolved = roots.lookup(ref)
                module = f"{node.module}.{name.name}"
                aliases[name.asname or name.name] = (module, "") if (
                    resolved and isinstance(resolved.node, ast.Module)) else (node.module, name.name)
        elif isinstance(node, ast.Import):
            for name in node.names:
                if external(name.name):
                    refs.add(("import", name.name))
                    aliases[name.asname or name.name] = (name.name, "")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (node.value not in LOGGER_NAMES
                    and re.fullmatch(r"vllm(?:_exl3)?(?:\.[a-z_][a-z_0-9]*)+", node.value)):
                refs.add(("module literal", node.value))
    for node in nodes:
        if not isinstance(node, ast.Attribute):
            continue
        parts = dotted(node).split(".")
        if len(parts) < 2:
            continue
        # For unaliased dotted imports use the longest matching module prefix.
        for stop in range(len(parts) - 1, 0, -1):
            alias = ".".join(parts[:stop])
            if alias not in aliases:
                continue
            module, member = aliases[alias]
            # The first member is statically discoverable. Lower attributes may
            # belong to Python objects obtained at runtime (e.g. current_platform).
            path = ".".join(x for x in (member, parts[stop]) if x)
            if (parts[stop] not in {"__dict__", "__file__", "__name__", "__package__"}
                    and not resolved_external_value(roots, module, member)):
                refs.add(("imported attribute", f"{module}:{path}"))
            break
    return refs


def resolved_external_value(roots, module, member):
    """A deferred or opaque binding whose type requires an explicit contract."""
    target = roots.lookup(f"{module}:{member}") if member else None
    return target and not isinstance(target.node, (ast.Module, ast.ClassDef))


def local_classes(source):
    for node in ast.walk(parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.bases:
            yield f"{node.name} bases {', '.join(ast.unparse(x) for x in node.bases)}"
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{node.name}.{method.name}({', '.join(signature(method))})"


def entry_points(bundle, plugin_entry, roots):
    entries = dict(plugin_entry)
    for path in sorted(bundle.rglob("entry_points.txt")):
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read(path)
        for group in cfg.sections():
            if group != GROUP:
                raise ValueError(f"Unexpected overlay entry group {group}: {path}")
            for name, ref in cfg[group].items():
                if name in entries:
                    raise ValueError(f"Duplicate overlay entry point: {name}")
                entries[name] = ref
    expected = set(PLUGIN_ALLOWLIST) | {"dsv41_stack_audit", "dsv41_draft_argmax", "dsv41_r7h_switch"}
    if set(entries) != expected:
        raise ValueError(f"Entry point set differs: {set(entries) ^ expected}")
    for name, ref in entries.items():
        node = roots.lookup(ref)
        if node is None or signature(node.node) != []:
            raise ValueError(f"Unresolved plugin registration {name}: {ref}")
        print(f"{GROUP} {name} = {ref} ({'allowlisted' if name in PLUGIN_ALLOWLIST else 'inactive'})")


def find_export(row, exports):
    dest = row["dest"]
    path = exports / f"{row['n']:03d}" / Path(dest).name
    if not path.exists():
        raise ValueError(f"Missing mount export: {path}")
    return path


def check(bundle, pristine, stage1, effective, served, strict):
    errors = []
    refs = json.loads((bundle / "contracts.json").read_text())
    for name in refs:
        if not (bundle / name).is_file():
            errors.append(f"contract file vanished: {name}")
    pin_refs = pin_targets(bundle)
    for source in sorted(bundle.rglob("*.py")):
        name = source.relative_to(bundle).as_posix()
        print(f"\n{name}")
        for detail in local_classes(source):
            print(f"  class contract {detail}")
        uses = references(source, effective)
        for kind, symbols in refs.get(name, {}).items():
            if kind == "signature":
                continue
            uses.update((kind, ref) for ref in symbols)
        uses.update(("pin", ref) for ref in pin_refs.get(name, ()))
        for kind, ref in sorted(uses):
            if kind == "measurement-only hook" and served is None:
                print(f"  {kind} {ref} (checked offline against the packet)")
                continue
            if ref.startswith(("vllm.triton_utils.tl:", "vllm.triton_utils.triton:")):
                # Attributes of the third-party Triton modules re-exported by vllm.triton_utils: not vLLM source.
                # The installed image resolves them at import (check_image_cpu.py), so the first image build
                # (2026-11-19) must not fail on them here.
                print(f"  {kind} {ref} (third-party Triton attribute)")
                continue
            old = served.lookup(ref) if served else None
            current = effective.lookup(ref)
            fields = []
            for label, roots in (("pristine", pristine), ("stage1", stage1),
                                 ("effective", effective)):
                if roots is not None:
                    fields.append(f"{label}={'OK' if roots.lookup(ref) else 'MISSING'}")
            if old or current:
                before = signature(old.node) if old else None
                after = signature(current.node) if current else None
                if before != after:
                    fields.append(f"signature {before} → {after}")
                elif after is not None:
                    fields.append(f"signature {after}")
            print(f"  {kind} {ref} {' '.join(fields)}")
            if not current:
                errors.append(f"{name}: {kind}: {ref}")
        # Signature assertions cover names the hooks pass or select explicitly.
        for ref, parameters in refs.get(name, {}).get("signature", {}).items():
            target = effective.lookup(ref)
            if target and signature(target.node) != parameters:
                errors.append(f"{name}: signature changed for {ref}: {signature(target.node)}")
    if strict and errors:
        raise ValueError("Unresolved overlay references:\n" + "\n".join(errors))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-overlay-root", type=Path, default=WORK / "ported-overlays")
    parser.add_argument("--plugin-root", type=Path, default=WORK / "plugin-replay/src")
    parser.add_argument("--image", action="store_true", help="check the installed image's source files")
    args = parser.parse_args()
    bundle = KIT / "overlays"
    packages = tuple(p for p in sorted(bundle.iterdir()) if p.is_dir() and p.name != "rootfs")
    local = (*packages, bundle / "rootfs/opt/qv", bundle / "rootfs/opt/stack-code",
             bundle / "rootfs/opt/prefill-bitwise")
    if args.image:
        import importlib.metadata
        from install_overlays import locate_site
        roots = Sources(locate_site("vllm"), locate_site("vllm_exl3"), *local)
        check(bundle, None, None, roots, None, True)
        entries = importlib.metadata.entry_points(group=GROUP, name="vllm_exl3")
    else:
        qv = KIT.parents[1] / "rounds/quick-verify-r1"
        local = (*local, qv, qv / "overlay")
        exports = WORK / "overlays"
        manifest = json.loads((exports / "manifest.json").read_text())
        exported_vllm = {row["dest"].removeprefix(OLD_SITE): find_export(row, exports)
                         for row in manifest if row["exported"] and
                         row["dest"].startswith(OLD_SITE + "vllm") and row["dest"].endswith(".py")}
        pristine = Sources(WORK / "v034", args.plugin_root, *local)
        stage1 = Sources(WORK / "ported", args.plugin_root, *local)
        roots = Sources(args.whole_overlay_root, WORK / "ported", args.plugin_root, *local)
        served = Sources(WORK / "served", *local, mounted=exported_vllm)
        errors = check(bundle, pristine, stage1, roots, served, False)
        missing_files = [name for name in whole_paths(manifest)
                         if not (args.whole_overlay_root / name).is_file()]
        entries = []
        parser_old = configparser.ConfigParser(interpolation=None)
        parser_old.read(WORK / "served/vllm_exl3-0.5.0.dist-info/entry_points.txt")
        entries.extend(parser_old[GROUP].items())
        if errors or missing_files:
            if missing_files:
                print("\nWaiting for ported whole files:\n" + "\n".join(missing_files), file=sys.stderr)
            if errors:
                print("\nUnresolved overlay references:\n" + "\n".join(errors), file=sys.stderr)
            status = 1
        else:
            status = 0
    if args.image:
        entries = [(ep.name, ep.value) for ep in entries]
        if entries != [("vllm_exl3", "vllm_exl3.bf16_madv_compat:register")]:
            raise ValueError("Installed EXL3 entry point differs")
        status = 0
    entry_points(bundle, entries, roots)
    return status


if __name__ == "__main__":
    sys.exit(main())
