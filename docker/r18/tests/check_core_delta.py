#!/usr/bin/env python3
"""Check the installed Python delta and emit reproducible four-tree diffs."""
import argparse
import base64
import csv
from difflib import unified_diff
import hashlib
import json
from pathlib import Path
import subprocess

KIT = Path(__file__).resolve().parents[1]
WORK = KIT.parent / "work"


def in_scope(rel):
    path = Path(rel)
    return (path.parts[0] == "vllm" and path.suffix == ".py"
            and "vllm_flash_attn" not in path.parts and rel != "vllm/_version.py")


def files(tree):
    return {p.relative_to(tree).as_posix(): p.read_bytes() for p in (tree / "vllm").rglob("*.py")
            if in_scope(p.relative_to(tree).as_posix())}


def wheel_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def diff(old, new, old_label, new_label, rel):
    return "".join(unified_diff((old or b"").decode().splitlines(keepends=True),
                                (new or b"").decode().splitlines(keepends=True),
                                f"{old_label}/{rel}", f"{new_label}/{rel}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, help="write each three-way merge and its paired diffs")
    args = parser.parse_args()
    pin, served, release, port = [files(WORK / name) for name in ("pin", "served", "v034", "ported")]
    metadata = json.loads((KIT / "patches/vllm/series.json").read_text())
    paths = set(metadata["reviewed_paths"])
    changed = {name for name, data in served.items() if pin.get(name) != data}
    if changed != paths:
        raise SystemExit(f"Unexpected served delta: extra={sorted(changed - paths)}, missing={sorted(paths - changed)}")
    dist, = (WORK / "served").glob("vllm-*.dist-info")
    with (dist / "RECORD").open() as file:
        record = {name: digest for name, digest, *_ in csv.reader(file) if in_scope(name)}
    missing_pin = pin.keys() - served.keys()
    missing_matches = {name for name in missing_pin if record.get(name) == wheel_hash(pin[name])}
    if missing_matches != missing_pin:
        raise SystemExit(f"Pin paths missing from export and unmatched by RECORD: {sorted(missing_pin - missing_matches)}")
    record_changed = {name for name, data in served.items() if record.get(name) != wheel_hash(data)}
    if record_changed != paths:
        raise SystemExit(f"Export/RECORD delta differs: {sorted(record_changed ^ paths)}")
    touched = {name for name in release.keys() | port.keys() if release.get(name) != port.get(name)}
    folded = {"vllm/v1/core/kv_cache_planning.py"}
    if touched != paths - folded:
        raise SystemExit(f"Unexpected ported Python delta: {sorted(touched ^ (paths - folded))}")

    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
    exact = set()
    for name in sorted(paths):
        a, b, c, d = (tree.get(name) for tree in (pin, served, release, port))
        if a == c:
            merge = b
        else:
            result = subprocess.run(
                ["git", "merge-file", "-p", "--diff3", "-L", "v034", "-L", "pin", "-L", "served",
                 *[str(WORK / tree / name) for tree in ("v034", "pin", "served")]],
                capture_output=True,
            )
            if not 0 <= result.returncode <= 127 or result.stderr:
                raise SystemExit(result.stderr.decode(errors="replace"))
            merge = result.stdout
        if merge == d:
            exact.add(name)
        if args.audit_dir:
            stem = args.audit_dir / name
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.with_suffix(".diff").write_text(
                diff(a, b, "pin", "served", name) + diff(c, d, "v034", "ported", name)
                + diff(merge, d, "merge", "ported", name)
            )
            if merge != d:
                stem.with_suffix(".diff3").write_bytes(merge)
    print(f"PASS {len(served)} exported core files, {len(paths)} changed paths, "
          f"{len(missing_matches)} omitted pin paths match RECORD")
    print(f"RECORD-only vendor paths: {len(record.keys() - served.keys() - pin.keys())}")
    print(f"Exact mechanical merges: {len(exact)}. Review remaining paths in CORE-AUDIT.md.")
    if args.audit_dir:
        print(f"Four-tree hunk report: {args.audit_dir}")


if __name__ == "__main__":
    main()
