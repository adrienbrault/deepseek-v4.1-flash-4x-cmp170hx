"""CPU-only telemetry filtering, failure and owned-process lifecycle tests."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('experiment-telemetry.py')
spec = importlib.util.spec_from_file_location('telemetry', SCRIPT)
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


def wait_for(predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError('timed out waiting for test condition')


def process_running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


class TelemetryTests(unittest.TestCase):
    def test_metric_filter_excludes_histograms_and_keeps_selected_labels(self):
        values = telemetry.filter_metrics('''# HELP ignored
vllm:num_requests_running{model_name="m"} 4
vllm:kv_cache_usage_perc 0.5
vllm:generation_tokens_total{engine="0"} 700
vllm:spec_decode_num_accepted_tokens_total 12
vllm:spec_decode_acceptance_rate NaN
vllm:spec_decode_latency_bucket{le="2"} 20
vllm:spec_decode_latency_sum 30
vllm:time_to_first_token_seconds_bucket{le="2"} 10
python_gc_objects_collected_total 88
broken line
''')
        self.assertEqual(len(values), 5)
        self.assertEqual(values[0]['labels'], '{model_name="m"}')
        self.assertEqual(values[0]['value'], 4)
        self.assertEqual(values[-1]['value'], 'NaN')
        json.dumps(values, allow_nan=False)

    def test_only_online_unique_policies_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'online').write_text('0-1,3\n')
            policy = root/'cpufreq'/'policy0'
            policy.mkdir(parents=True)
            (policy/'scaling_cur_freq').write_text('2000000')
            for cpu in (0, 1, 3):
                (root/f'cpu{cpu}').mkdir()
                (root/f'cpu{cpu}'/'cpufreq').symlink_to(policy)
            got = telemetry.active_cpu_policies(root)
            self.assertEqual(len(got), 1)
            self.assertEqual(next(iter(got.values()))['scaling_cur_freq'], '2000000')

    def test_missing_commands_and_endpoint_are_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = telemetry.Collector(tmp, 0.1, 'http://127.0.0.1:1')
            with patch.object(telemetry.shutil, 'which', return_value=None):
                c.start_tools()
            c.sample()
            c.stop_tools()
            self.assertTrue(all(x['status'] == 'missing' for x in c.status['commands']))
            sample = json.loads((Path(tmp)/'server-metrics.jsonl').read_text())
            self.assertEqual(sample['status'], 'error')
            self.assertTrue((Path(tmp)/'heartbeat.json').exists())
            self.assertTrue((Path(tmp)/'system.jsonl').exists())

    def test_actual_process_group_cleanup_term_and_int(self):
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=stop_signal), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bindir = root/'bin'
                bindir.mkdir()
                fake = '''#!/usr/bin/env python3
import os, pathlib, signal, subprocess, sys, time
name=pathlib.Path(sys.argv[0]).name
log=pathlib.Path(os.environ['FAKE_PID_LOG'])
with log.open('a') as f:f.write(str(os.getpid())+'\\n')
if name=='iostat':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:time.sleep(.1)
if name=='pidstat':
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'])
    with log.open('a') as f:f.write(str(child.pid)+'\\n')
    def end(*a):
        child.wait(timeout=3)
        sys.exit(0)
    signal.signal(signal.SIGTERM,end)
else:
    # Nonzero tool failures must be recorded without stopping collection.
    sys.exit(7)
while True:time.sleep(.1)
'''
                for tool in ('iostat', 'pidstat', 'nvidia-smi'):
                    p = bindir/tool
                    p.write_text(fake)
                    p.chmod(0o755)
                env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                           FAKE_PID_LOG=str(root/'pids'))
                output = root/'output'
                process = subprocess.Popen([sys.executable, str(SCRIPT), str(output),
                                            '--interval', '0.1', '--url', 'http://127.0.0.1:1'], env=env)
                try:
                    wait_for(lambda: (root/'pids').exists() and
                             len((root/'pids').read_text().splitlines()) >= 4)
                    wait_for(lambda: (output/'heartbeat.json').exists())
                    time.sleep(0.2)
                    process.send_signal(stop_signal)
                    self.assertEqual(process.wait(timeout=7), 0)
                    pids = [int(x) for x in (root/'pids').read_text().splitlines()]
                    wait_for(lambda: not any(process_running(pid) for pid in pids))
                    status = json.loads((output/'collector-status.json').read_text())
                    self.assertTrue(status['stopped'])
                    self.assertTrue(all(x['returncode'] is not None for x in status['commands']))
                    gpu = next(x for x in status['commands'] if x['tool'] == 'nvidia-smi')
                    self.assertEqual(gpu['returncode'], 7)
                    self.assertTrue((output/'meta.json').exists())
                    self.assertTrue((output/'start.json').exists())
                    self.assertTrue((output/'end.json').exists())
                    self.assertTrue(json.loads((output/'heartbeat.json').read_text())['stopped'])
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=7)
                    # Exact test-owned PIDs only; cleanup remains safe on assertion failure.
                    if (root/'pids').exists():
                        for pid in map(int, (root/'pids').read_text().splitlines()):
                            if process_running(pid):
                                os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main(verbosity=2)
