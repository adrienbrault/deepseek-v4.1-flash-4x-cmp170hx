#!/usr/bin/env python3
"""Replay plugin patches and compare the complete f760 Python package."""
from configparser import ConfigParser
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib

from check_plugin_symbols import check

KIT = Path(__file__).resolve().parents[1]
ROUND = KIT.parent
R17 = ROUND.parent / "r17/kit"
WORK = ROUND / "work"


def python_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.py")}


def apply(tree, patch):
    for mode in (["--dry-run"], []):
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", *mode, "-i", str(patch)],
            cwd=tree, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )


def check_replay():
    served = WORK / "served/vllm_exl3"
    actual = python_files(served)
    dist, = sorted(served.parent.glob("vllm_exl3-*.dist-info"))
    with (dist / "RECORD").open() as record:
        installed = {
            path.removeprefix("vllm_exl3/") for path, *_ in csv.reader(record)
            if path.startswith("vllm_exl3/") and path.endswith(".py")
        }
    if installed - actual.keys():
        raise ValueError(f"Export missing installed paths: {sorted(installed - actual.keys())}")
    config = ConfigParser()
    config.read(dist / "entry_points.txt")
    expected_entries = dict(config["vllm.general_plugins"])

    series = json.loads((KIT / "series.json").read_text())["vllm_exl3"]
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "vendor"
        shutil.copytree(R17 / "vendor/vllm-exl3", source)
        for name in series["inherited"]:
            apply(source, R17 / "patches/vllm-exl3" / name)
        replay, port = series["r18"]
        apply(source, KIT / "patches/vllm-exl3" / replay)
        package = source / "src/vllm_exl3"
        replayed = python_files(package)
        changed = sorted(path for path in actual.keys() & replayed.keys()
                         if actual[path] != replayed[path])
        missing = sorted(actual.keys() - replayed.keys())
        extra = sorted(replayed.keys() - actual.keys())
        if changed or missing or extra:
            raise ValueError(f"Plugin replay differs: changed={changed}, missing={missing}, extra={extra}")
        print(f"PASS installed f760 plugin replay matches all {len(actual)} Python files")

        apply(source, KIT / "patches/vllm-exl3" / port)
        ported = python_files(package)
        manifest = json.loads((KIT / "plugin-patched-files.json").read_text())
        expected_changes = {
            path.removeprefix("src/vllm_exl3/"): digest for path, digest in manifest.items()
        }
        if ported.keys() != actual.keys():
            raise ValueError("v0.34 patch changes the Python file inventory")
        changes = {
            path: hashlib.sha256(contents).hexdigest() for path, contents in ported.items()
            if contents != actual[path]
        }
        if changes != expected_changes:
            raise ValueError("v0.34 patch differs from its two-file ABI manifest")

        with (source / "pyproject.toml").open("rb") as handle:
            entries = tomllib.load(handle)["project"]["entry-points"]["vllm.general_plugins"]
        if entries != expected_entries:
            raise ValueError("Ported entry points differ from the installed metadata")
        failures = check(package, WORK / "v034", WORK / "pin")
        if failures:
            raise ValueError("\n".join(failures))
        print("PASS v0.34 imports, entry point, overrides and registration seams")


if __name__ == "__main__":
    try:
        check_replay()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stdout.decode(errors="replace")) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
