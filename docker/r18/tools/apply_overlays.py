#!/usr/bin/env python3
"""Bake the packet overlays into the stage-1 packages."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

SERIES = Path(__file__).resolve().parents[1] / "patches/overlays"
PACKAGES = ("vllm", "vllm_exl3")


def load_series():
    entries = json.loads((SERIES / "series.json").read_text())
    order = [entry["patch"] for entry in entries]
    if not order or order != sorted(path.name for path in SERIES.glob("*.patch")):
        raise ValueError("Overlay patch order or inventory differs from series.json")
    return entries


def check_files(roots, files, phase):
    for rel, record in files.items():
        package = rel.split("/", 1)[0]
        path = roots[package] / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != record[phase]:
            raise ValueError(f"Overlay {phase} SHA-256 mismatch for {rel}: {actual}")


def apply(roots):
    for entry in load_series():
        patch = SERIES / entry["patch"]
        sections = re.split(r"(?m)(?=^diff --git )", patch.read_text())
        paths = []
        patches = {package: [] for package in PACKAGES}
        for section in filter(None, sections):
            header = re.fullmatch(r"diff --git a/((vllm|vllm_exl3)/\S+\.py) b/\1",
                                  section.splitlines()[0])
            if header is None:
                raise ValueError(f"Unexpected overlay header in {patch.name}")
            rel, package = header.groups()
            paths.append(rel)
            patches[package].append(section)
        files = entry["files"]
        if len(paths) != len(files) or set(paths) != files.keys():
            raise ValueError(f"Overlay paths differ from series.json for {patch.name}")
        check_files(roots, files, "before")
        for mode in (["--dry-run"], []):
            for package, sections in patches.items():
                if sections:
                    subprocess.run(
                        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", *mode],
                        input="".join(sections), text=True, cwd=roots[package], check=True,
                    )
        check_files(roots, files, "after")
        for rel in files:
            path = roots[rel.split("/", 1)[0]] / rel
            for cache in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
                cache.unlink()
        print("APPLIED", patch.name, entry["purpose"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path,
                        help="directory containing both stage-1 packages")
    args = parser.parse_args()
    if args.package_root is None:
        roots = {
            package: Path(importlib.util.find_spec(package).origin).resolve().parent.parent
            for package in PACKAGES
        }
    else:
        roots = dict.fromkeys(PACKAGES, args.package_root.resolve())
    apply(roots)


if __name__ == "__main__":
    main()
