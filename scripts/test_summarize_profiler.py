# SPDX-License-Identifier: MIT
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name("summarize-profiler.py")
spec = importlib.util.spec_from_file_location("summarize_profiler", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def event(name, ts, dur, cat="kernel", tid=1):
    return {"name": name, "ts": ts, "dur": dur, "cat": cat, "ph": "X", "pid": 1, "tid": tid}


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "torchtrace-summary.json"

    def trace(self, name, events):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"traceEvents": events})
        if name.endswith(".gz"):
            with gzip.open(path, "wt") as out:
                out.write(text)
        else:
            path.write_text(text)
        return path

    def test_normal_gzip_union_communication_and_inclusive_cpu(self):
        events = [event("gemm", 100, 10), event("ncclAllReduce", 105, 20, tid=2),
                  event("gemm", 140, 10), event("aten::parent", 100, 50, "cpu_op"),
                  event("aten::child", 110, 20, "cpu_op"),
                  {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "rank0"}},
                  {"ph": "B", "cat": "kernel", "name": "ignore", "ts": 0}]
        self.trace("nested/torchtrace-rank0.json.gz", events)
        self.trace("worker1.1.pt.trace.json", [event("pcie_ipc_reduce", 0, 5)])
        self.trace("torchtrace-frontend.json", [event("aten::frontend", 0, 20, "cpu_op")])
        self.output.write_text("not a trace; must be excluded")
        report = m.summarize(self.root, self.output)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["gpu_trace_count"], 2)
        self.assertEqual(report["zero_kernel_trace_count"], 1)
        trace = next(t for t in report["traces"] if "rank0" in t["file"])
        self.assertEqual(trace["kernel_count"], 3)
        self.assertEqual(trace["kernel_streams"], [{"pid": 1, "tid": 1}, {"pid": 1, "tid": 2}])
        self.assertEqual(trace["total_kernel_us"], 40)
        self.assertEqual(trace["kernel_span_us"], 50)
        self.assertEqual(trace["union_kernel_us"], 35)
        self.assertEqual(trace["union_kernel_coverage_fraction"], .7)
        self.assertEqual(trace["communication_kernel_us"], 20)
        self.assertEqual(trace["communication_summed_kernel_fraction"], .5)
        self.assertEqual(trace["process_names"], [{"pid": 1, "name": "rank0"}])
        self.assertEqual(trace["top_kernels"][0], {"name": "gemm", "summed_us": 20, "count": 2})
        self.assertEqual(sum(row["summed_us"] for row in trace["top_cpu_ops_inclusive"]), 70)

    def test_no_kernels_is_explicit_error_and_cli_nonzero(self):
        self.trace("torchtrace-cpu.json.gz", [event("aten::cpu", 0, 1, "cpu_op")])
        report = m.summarize(self.root, self.output)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["gpu_trace_count"], 0)
        self.assertIn("no GPU", report["errors"][-1])
        result = subprocess.run(["python3", str(MODULE), str(self.root), "--output", str(self.output)], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(self.output.read_text())["status"], "error")

    def test_limits_and_invalid_durations(self):
        self.trace("torchtrace-rank0.json", [event("ok", 0, 1), event("negative", 0, -1),
                                             event("nan", 1, float("nan"))])
        report = m.summarize(self.root, self.output)
        self.assertEqual(report["traces"][0]["kernel_count"], 1)
        self.assertEqual(report["traces"][0]["invalid_complete_events"], 2)
        with patch.object(m, "MAX_TRACE_BYTES", 5):
            report = m.summarize(self.root, self.output)
            self.assertEqual(report["status"], "error")
            self.assertIn("exceeds", report["errors"][0]["error"])
        with patch.object(m, "MAX_FILES", 1):
            self.trace("torchtrace-rank1.json", [event("ok", 0, 1)])
            report = m.summarize(self.root, self.output)
            self.assertEqual(report["status"], "error")
            self.assertIn("more than", report["errors"][0])

    def test_top15_and_communication_names(self):
        events = [event(f"kernel{i}", i * 20, i + 1) for i in range(20)]
        events += [event(name, 1000, 2) for name in ("all_reduce", "all-gather", "PCIE_IPC", "NCCL")]
        self.trace("torchtrace-all.json", events)
        trace = m.summarize(self.root, self.output)["traces"][0]
        self.assertEqual(len(trace["top_kernels"]), 15)
        self.assertEqual(trace["top_kernels"][0]["name"], "kernel19")
        self.assertEqual(trace["communication_kernel_count"], 4)
        self.assertEqual(trace["communication_kernel_us"], 8)


if __name__ == "__main__":
    unittest.main()
