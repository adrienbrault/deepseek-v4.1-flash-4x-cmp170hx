"""Apply the checked series to the located package, never a hard-coded site path."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
kind = sys.argv[1]
assert kind in ("vllm", "vllm-exl3")
if kind == "vllm":
    if len(sys.argv) == 4 and sys.argv[2] == "--package-root":
        cwd = Path(sys.argv[3]).resolve()
    else:
        assert len(sys.argv) == 2
        package = Path(importlib.util.find_spec("vllm").origin).parent
        cwd = package.parent
    for rel, expected in json.loads((ROOT / "base-files.json").read_text()).items():
        path = cwd / rel
        if expected is None:
            assert not path.exists(), f"new path already exists: {rel}"
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, rel
else:
    cwd = ROOT / "vendor/vllm-exl3"
patches = sorted((ROOT / "patches" / kind).glob("*.patch"))
assert patches
for path in patches:
    contents = path.read_text()
    if kind == "vllm":
        # Wheels contain vllm/, not the source checkout's tests/benchmarks.
        # Keep the full format-patch exports intact for source consumers.
        sections = re.split(r"(?=^diff --git )", contents, flags=re.MULTILINE)
        contents = "".join(section for section in sections
                           if re.match(r"diff --git a/vllm/\S+ b/vllm/\S+\n", section))
        if not contents:
            print("SOURCE-ONLY", path.name)
            continue
    for mode in (["--dry-run"], []):
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", *mode,
                        ], input=contents, text=True, cwd=cwd, check=True)
    print("APPLIED", path.name)
if kind == "vllm":
    for rel, expected in json.loads((ROOT / "patched-files.json").read_text()).items():
        assert hashlib.sha256((cwd / rel).read_bytes()).hexdigest() == expected, rel
    for rel in json.loads((ROOT / "patched-files.json").read_text()):
        path = cwd / rel
        if path.suffix == ".py":
            for cache in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
                cache.unlink()
    print("PASS exact patched package contents")
else:
    for rel, expected in json.loads((ROOT / "plugin-patched-files.json").read_text()).items():
        assert hashlib.sha256((cwd / rel).read_bytes()).hexdigest() == expected, rel
    print("PASS exact patched plugin contents")
