#!/usr/bin/env python3
"""Recompute overlay identity pins from final installed source, without importing it."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

from install_overlays import locate_site, sha, write_source
from overlay_sources import Sources, parse
from pprint import pformat

KIT = Path(__file__).resolve().parents[1]
PACKAGE = {
    "dispatch": "dsv41_dispatch_fuse",
    "depth": "dsv41_depth",
    "draft": "dsv41_draft_argmax",
    "rowstats": "dsv41_rowstats",
    "wkv": "dsv41_wkv_replica",
    "attention": "dsv41_attn_port",
    "ablit": "dsv41_ablit_r7",
}
FILES = {
    "dispatch": ("hooks.py", "PINS"),
    "draft": ("hooks.py", "PINS"),
    "rowstats": ("hooks.py", "PINS"),
    "wkv": ("hooks.py", "PINS"),
}
WKV_CHANGED = {
    "apply_mxfp8_marlin_linear", "_marlin_gemm", "CudaCommunicator.all_gather",
    "Worker.load_model",
}
ROOTFS_INDEXER = "rootfs/opt/qv/indexer_runtime.py"


def assignment(path, name):
    matches = [node for node in parse(path).body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} assignment: {path}")
    return matches[0]


def source_map(path, name):
    value = assignment(path, name).value
    return {
        ast.literal_eval(assignment(path, key.id).value) if isinstance(key, ast.Name)
        else ast.literal_eval(key): ast.literal_eval(pin)
        for key, pin in zip(value.keys, value.values, strict=True)
    }


def project_paths(bundle):
    for key, (file, _) in FILES.items():
        package = PACKAGE[key]
        yield key, bundle / package / package / file


def pin_targets(bundle):
    refs = {path.relative_to(bundle).as_posix(): set(source_map(path, FILES[key][1]))
            for key, path in project_paths(bundle)}
    wkv = bundle / PACKAGE["wkv"] / PACKAGE["wkv"] / "reviewed.py"
    refs[wkv.relative_to(bundle).as_posix()] = {
        f"{module}:{symbol}" for module, symbol in source_map(wkv, "REVIEWED_SOURCES")
    }
    attention = bundle / PACKAGE["attention"] / PACKAGE["attention"]
    refs[(attention / "contracts.py").relative_to(bundle).as_posix()] = set(
        json.loads((attention / "metadata-contract.json").read_text()))
    dense = bundle / PACKAGE["ablit"] / PACKAGE["ablit"] / "dense.py"
    refs[dense.relative_to(bundle).as_posix()] = {
        "vllm.lora.layers.base_linear", "dsv41_ablit_r7.fused_rank1"
    }
    refs[ROOTFS_INDEXER] = {"dsv41_attn_audit", "dsv41_attn_port.topk_glue"}
    return refs


def replace(path, name, value):
    node = assignment(path, name)
    lines = path.read_text().splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [f"{name} = {pformat(value, sort_dicts=False)}\n"]
    write_source(path, "".join(lines).encode())
    parse.cache_clear()


def module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_digest(source):
    node = ast.parse(textwrap.dedent(source)).body[0]
    node.decorator_list = []
    if ast.get_docstring(node) is not None:
        node.body.pop(0)
    # On 3.13+ the default skips empty fields; 3.12 emitted them.
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return hashlib.sha256(ast.dump(node, include_attributes=False, **options).encode()).hexdigest()


def rewrite(site, plugin_site, qv):
    roots = Sources(site, plugin_site)

    def lookup(ref):
        node = roots.lookup(ref)
        if node is None:
            raise ValueError(f"Cannot re-pin unresolved image source: {ref}")
        return node

    changes = {}

    def update(path, key, value):
        before = ast.get_source_segment(path.read_text(), assignment(path, key).value)
        changes[str(path) + ":" + key] = dict(incumbent=before, candidate=value)
        replace(path, key, value)

    # The file pins identify the reviewed candidate's exact module bytes. Values
    # are materialized after both image patch series and the whole-file handoff.
    for key, (file, variable) in FILES.items():
        path = site / PACKAGE[key] / file
        update(path, variable, {module: sha(lookup(module).path)
                                for module in source_map(path, variable)})
    path = site / PACKAGE["wkv"] / "reviewed.py"
    module = module_from_path(path, "_r18_wkv_digest")
    reviewed = source_map(path, "REVIEWED_SOURCES")
    pins = {(mod, sym): module.digest(lookup(f"{mod}:{sym}").source())
            for mod, sym in reviewed}
    changed = {sym for (mod, sym), before in reviewed.items() if pins[mod, sym] != before}
    if changed != WKV_CHANGED:
        raise ValueError(f"Unexpected reviewed WKV source changes: {changed ^ WKV_CHANGED}")
    update(path, "REVIEWED_SOURCES", pins)
    # Represent tuple keys as strings in the provenance JSON.
    changes[str(path) + ":REVIEWED_SOURCES"]["candidate"] = {
        f"{mod}:{sym}": digest for (mod, sym), digest in pins.items()
    }

    attention = site / PACKAGE["attention"]
    contract = module_from_path(attention / "contracts.py", "_r18_attention_contract")
    topk = "vllm.model_executor.layers.indexer_topk:SparseIndexerTopk."
    contract.validate_sources(lookup(topk + "forward").source(), lookup(topk + "_row_ends").source())
    path = attention / "metadata-contract.json"
    old = json.loads(path.read_text())
    pins = {ref: metadata_digest(lookup(ref).source()) for ref in old}
    if pins != old:
        raise ValueError("SWA metadata meaning changed beyond the reviewed port")
    changes[str(path)] = dict(incumbent=old, candidate=pins)


    dense = site / PACKAGE["ablit"] / "dense.py"
    update(dense, "R11_SHA256", sha(lookup("vllm.lora.layers.base_linear").path))
    expected = ast.literal_eval(assignment(dense, "FUSED_SHA256").value)
    if sha(lookup("dsv41_ablit_r7.fused_rank1").path) != expected:
        raise ValueError("Fixed fused rank-1 identity changed")
    audit_path = qv / "indexer_runtime.py"
    audit_sha = sha(lookup("dsv41_attn_audit").path)
    if audit_sha != ast.literal_eval(assignment(audit_path, "AUDIT_SHA256").value):
        raise ValueError("Indexer audit wrapper identity changed")
    changes[str(audit_path) + ":AUDIT_SHA256"] = dict(candidate=audit_sha)
    changes[str(dense) + ":FUSED_SHA256"] = dict(candidate=expected)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=KIT / "runtime/overlay-pins.json")
    args = parser.parse_args()
    changes = rewrite(locate_site("vllm"), locate_site("vllm_exl3"), Path("/opt/qv"))
    args.receipt.write_text(json.dumps(changes, indent=2) + "\n")


if __name__ == "__main__":
    main()
