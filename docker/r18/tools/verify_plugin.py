"""Verify the pristine pinned source before applying the sole native ABI patch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "vendor/plugin-source.json").read_text())
assert manifest["revision"] == "3c60e6b45b5ec9e845b07aa70b4f974819010ffe"
for rel, digest in manifest["sha256"].items():
    assert hashlib.sha256((root / "vendor/vllm-exl3" / rel).read_bytes()).hexdigest() == digest, rel
print("PASS: vllm-exl3 source 3c60e6b")
