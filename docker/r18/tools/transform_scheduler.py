#!/usr/bin/env python3
"""Generate a v0.34 serving scheduler from the installed, patched image source."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile

from prefill_share import add_prefill_share

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def transform(source, fair_r2=False, *, prefill_share=False):
    with tempfile.TemporaryDirectory() as temp:
        candidate = Path(temp) / "scheduler.py"
        candidate.write_bytes(source.read_bytes())
        stages = ("r7g", "r7h") if fair_r2 else ("r7g",)
        for stage in stages:
            patch = RUNTIME / f"{stage}-scheduler.patch"
            subprocess.run(["patch", "--batch", "--forward", "--fuzz=0",
                            str(candidate), str(patch)], check=True, stdout=subprocess.PIPE)
        result = candidate.read_bytes()
    if prefill_share:
        if not fair_r2:
            raise ValueError("Prefill sharing requires the r7h scheduler transform")
        result = add_prefill_share(result.decode(), RUNTIME).encode()
        compile(result, str(source), "exec")
    compile(result, str(source), "exec", ast.PyCF_ONLY_AST)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="default: installed vLLM scheduler")
    parser.add_argument("--fair-r2", action="store_true", help="apply r7h policy on top of r7g")
    parser.add_argument("--prefill-share", action="store_true", help="include the opt-in half-share path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source
    if source is None:
        spec = importlib.util.find_spec("vllm")
        source = Path(spec.origin).parent / "v1/core/sched/scheduler.py"
    result = transform(source, args.fair_r2, prefill_share=args.prefill_share)
    with args.output.open("xb") as handle:
        handle.write(result)
    print(f"{hashlib.sha256(result).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
