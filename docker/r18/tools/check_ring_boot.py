#!/usr/bin/env python3
"""Match all four loaded-workspace gate logs against the standalone PASS."""
import argparse
import json
import re
from pathlib import Path


def check(logs, receipt):
    if receipt["event"] != "PASS" or not receipt["rows"]:
        raise ValueError("missing standalone true-ring PASS")
    rows = receipt["rows"]
    shapes = sorted({1, 6, 24, 48} | set(rows))
    calls = (5 * len(shapes)) | 1
    messages = {
        "binary": f"Opera true-ring native verified: count9/wire8, sha256={receipt['binary']},",
        "policy": f"exact rows{'/'.join(map(str, rows))}, extra={receipt['extra']}",
        "workspace": f"Opera true-ring loaded-workspace gate PASS: eager{len(shapes)}, "
                     f"graph{5 * calls}calls, rows{'/'.join(map(str, shapes))}",
    }
    verified, passed = set(), set()
    for line in logs.splitlines():
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        rank = re.search(r"\(Worker_TP([0-3])\s+pid=\d+\)", line)
        if rank is None:
            continue
        if messages["binary"] in line and messages["policy"] in line:
            verified.add(int(rank[1]))
        if line.rstrip().endswith(messages["workspace"]):
            passed.add(int(rank[1]))
    if verified != set(range(4)) or passed != set(range(4)):
        raise ValueError(f"Ring activation incomplete: native ranks {verified}, workspace ranks {passed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    with args.measurements.open() as results:
        for line in results:
            receipt = json.loads(line)
    check(args.logs.read_text(errors="replace"), receipt)
    print("PASS r18 true-ring loaded workspace on all four ranks")
