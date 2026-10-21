#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize Chrome Torch traces: DIRECTORY --output SUMMARY.json.

Durations are Chrome trace microseconds. Kernel interval coverage is the union
across all streams in each file, not GPU utilization. CPU times are inclusive;
no CPU self time or exclusive device attribution is inferred.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

MAX_FILES = 32
MAX_TRACE_BYTES = 256 * 1024 * 1024
COMMUNICATION = re.compile(r"nccl|all.?reduce|pcie_ipc|all.?gather", re.I)


def trace_name(path):
    name = path.name.lower()
    return ((name.startswith("torchtrace") or name.endswith((".pt.trace.json", ".pt.trace.json.gz")))
            and name.endswith((".json", ".json.gz")))


def discover(directory, output):
    files = []
    for path in Path(directory).rglob("*"):
        if path.is_file() and trace_name(path) and path.resolve() != Path(output).resolve():
            files.append(path)
            if len(files) > MAX_FILES:
                raise ValueError(f"more than {MAX_FILES} trace files; refusing partial selection")
    return sorted(files)


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def merged_length(intervals):
    end = None
    total = 0.0
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total


def top(rows):
    return [{"name": name, "summed_us": value[0], "count": value[1]}
            for name, value in sorted(rows.items(), key=lambda item: (-item[1][0], item[0]))[:15]]


def summarize_file(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as source:
        raw = source.read(MAX_TRACE_BYTES + 1)
    if len(raw) > MAX_TRACE_BYTES:
        raise ValueError(f"trace exceeds {MAX_TRACE_BYTES} uncompressed bytes")
    data = json.loads(raw)
    del raw
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    kernels, cpu = {}, {}
    intervals = []
    processes = []
    kernel_streams = {}
    kernel_count = cpu_count = communication_count = 0
    communication_us = 0.0
    invalid_complete = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name", "<unnamed>")
        if event.get("ph") == "M" and name == "process_name":
            args = event.get("args", {})
            processes.append({"pid": event.get("pid"), "name": args.get("name") if isinstance(args, dict) else None})
        if event.get("ph") != "X":
            continue
        categories = str(event.get("cat", "")).split(",")
        categories = {category.strip() for category in categories}
        if not categories.intersection({"kernel", "cpu_op"}):
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if not number(ts) or not number(dur) or dur < 0 or not number(ts + dur):
            invalid_complete += 1
            continue
        name = str(name)
        if "kernel" in categories:
            stream = {"pid": event.get("pid"), "tid": event.get("tid")}
            kernel_streams[json.dumps(stream, sort_keys=True)] = stream
            kernel_count += 1
            stats = kernels.setdefault(name, [0.0, 0])
            stats[0] += dur
            stats[1] += 1
            intervals.append((ts, ts + dur))
            if COMMUNICATION.search(name):
                communication_count += 1
                communication_us += dur
        if "cpu_op" in categories:
            cpu_count += 1
            stats = cpu.setdefault(name, [0.0, 0])
            stats[0] += dur
            stats[1] += 1
    total = sum(value[0] for value in kernels.values())
    span = max(hi for _, hi in intervals) - min(lo for lo, _ in intervals) if intervals else 0.0
    union = merged_length(intervals)
    return {"file": str(path), "rank_name_from_filename": path.name,
            "process_names": processes, "kernel_streams": list(kernel_streams.values()),
            "kernel_count": kernel_count,
            "total_kernel_us": total, "kernel_span_us": span,
            "union_kernel_us": union,
            "union_kernel_coverage_fraction": union / span if span else None,
            "coverage_definition": "kernel interval union across all streams divided by kernel span; not device utilization",
            "communication_kernel_us": communication_us,
            "communication_kernel_count": communication_count,
            "communication_summed_kernel_fraction": communication_us / total if total else None,
            "top_kernels": top(kernels), "cpu_op_count": cpu_count,
            "top_cpu_ops_inclusive": top(cpu), "invalid_complete_events": invalid_complete}


def summarize(directory, output):
    report = {"status": "error", "units": "microseconds", "traces": [], "errors": [],
              "gpu_trace_count": 0, "zero_kernel_trace_count": 0,
              "limits": {"files": MAX_FILES, "uncompressed_bytes_per_file": MAX_TRACE_BYTES},
              "interpretation": "Each file is independent. Overlapping kernels can inflate summed durations. Union coverage is not device utilization; CPU op durations are inclusive, not self time."}
    try:
        files = discover(directory, output)
    except Exception as exc:
        report["errors"].append(str(exc))
        return report
    for path in files:
        try:
            trace = summarize_file(path)
            report["traces"].append(trace)
            if trace["kernel_count"]:
                report["gpu_trace_count"] += 1
            else:
                report["zero_kernel_trace_count"] += 1
        except Exception as exc:
            report["errors"].append({"file": str(path), "error": str(exc)})
    if not report["gpu_trace_count"]:
        report["errors"].append("no GPU kernel traces found; frontend/CPU-only traces do not satisfy GPU profiling")
    report["status"] = "complete" if not report["errors"] else "error"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.directory, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gpu_trace_count": report["gpu_trace_count"],
                      "output": str(args.output)}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
