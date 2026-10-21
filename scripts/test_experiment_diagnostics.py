"""Exercise diagnostic phase ordering and cleanup against a real HTTP fixture."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).with_name('experiment-diagnostics.py'))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticTests(unittest.TestCase):
    def test_cpu_inventory_includes_workers_engine_and_api_only(self):
        listing = ('PID COMMAND\n11 VLLM::Worker_TP0\n12 VLLM::Worker_TP1\n'
                   '13 VLLM::Worker_TP2\n14 VLLM::Worker_TP3\n15 VLLM::EngineCore\n'
                   '16 /usr/bin/python3 /usr/local/bin/vllm serve /model\n'
                   '17 /usr/bin/python3 resource_tracker\n')
        with patch.object(diagnostics.subprocess, 'check_output', return_value=listing):
            self.assertEqual(diagnostics.workers('model'), [11, 12, 13, 14])
            self.assertEqual(diagnostics.workers('model', controls=True), [15, 16])

    def run_case(self, ready=True, fail_start=False, fail_stop=False):
        events = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                payload = {'data': [{'id': 'test-model'}]} if self.path == '/v1/models' else {
                    'paths': {'/start_profile': {}, '/stop_profile': {}} if ready else {}}
                self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(payload).encode())
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                events.append((self.path, payload))
                self.send_response(422 if ((fail_start and self.path == '/start_profile') or
                    (fail_stop and self.path == '/stop_profile')) else 200)
                self.end_headers()
                self.wfile.write(json.dumps({'usage': {'completion_tokens': payload.get('max_tokens', 0)}}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        old_signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with tempfile.TemporaryDirectory() as folder:
                process = Mock(pid=999999, returncode=0)
                process.poll.return_value = 0
                with patch.object(sys, 'argv', ['experiment-diagnostics.py', folder, '--url',
                      f'http://127.0.0.1:{server.server_port}']), \
                     patch.object(diagnostics, 'workers', return_value=[11, 12, 13, 14]), \
                     patch.object(diagnostics.subprocess, 'check_output', return_value='[]'), \
                     patch.object(diagnostics.subprocess, 'Popen', return_value=process), \
                     patch.object(diagnostics.os, 'killpg'), \
                     patch.object(diagnostics.subprocess, 'run', return_value=Mock(returncode=0)):
                    diagnostics.main()
                meta = json.loads((Path(folder) / 'diagnostics.json').read_text())
                process.wait.assert_called()
                return events, meta
        finally:
            server.shutdown(); server.server_close(); thread.join()
            for sig, handler in old_signals.items(): signal.signal(sig, handler)

    def test_profile_phases_and_unscored_metadata(self):
        events, meta = self.run_case()
        self.assertEqual(meta['status'], 'completed')
        starts = [p for path, p in events if path == '/start_profile']
        self.assertEqual([p['profile_prefix'] for p in starts], ['c1', 'c4-mixed'])
        self.assertTrue(all(p['max_iterations'] == 12 for p in starts))
        self.assertEqual(sum(path == '/stop_profile' for path, _ in events), 2)
        self.assertEqual(sum(path == '/v1/completions' for path, _ in events), 6)
        self.assertIn('excluded from benchmark', meta['classification'])
        self.assertTrue(meta['safe_for_scored_work'])

    def test_start_failure_still_stops_both_sessions(self):
        events, meta = self.run_case(fail_start=True)
        self.assertEqual(meta['status'], 'partial')
        self.assertEqual(sum(path == '/stop_profile' for path, _ in events), 2)
        self.assertEqual(len(meta['errors']), 2)

    def test_absent_profiler_records_partial_cpu_only(self):
        events, meta = self.run_case(ready=False)
        self.assertEqual(meta['status'], 'partial')
        self.assertFalse(any('profile' in path for path, _ in events))
        self.assertFalse(meta['gpu_profiler_ready'])

    def test_stop_failure_prevents_further_requests_and_is_unsafe(self):
        events, meta = self.run_case(fail_stop=True)
        self.assertFalse(meta['safe_for_scored_work'])
        self.assertEqual(sum(path == '/start_profile' for path, _ in events), 1)
        self.assertEqual(sum(path == '/stop_profile' for path, _ in events), 2)
        self.assertEqual(len(meta['phases']), 1)


if __name__ == '__main__':
    unittest.main()
