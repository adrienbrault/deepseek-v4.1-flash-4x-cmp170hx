"""Exercise the warm-up HTTP requests against an in-process fake server."""
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler
import io
import json
from pathlib import Path
import socket
import sys
from threading import Barrier, Thread
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "tools"))
import warmup


class WarmupTest(unittest.TestCase):
    def test_sizes_cover_the_served_dispatch_boundaries(self):
        argv = json.loads((KIT / "runtime/serve-argv.json").read_text())
        budget = int(argv[argv.index("--max-num-batched-tokens") + 1])
        window = int(argv[argv.index("--block-size") + 1])
        sizes = warmup.PREFILL_SIZES
        for low, high in ((7, 15), (16, 32), (33, 42), (64, 64),
                          (window, window), (1024, budget - 1)):
            self.assertEqual(sum(low <= n <= high for n in sizes), 1)
        self.assertEqual(sizes[-1], budget + window + 1)
        self.assertEqual(len(sizes), 7)
        self.assertEqual(warmup.MIXED_SIZE * 2, budget)

    def test_posts_cold_prompts_and_runs_mixed_pair_concurrently(self):
        requests = []
        workers = []
        mixed = Barrier(2, timeout=3)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers["Content-Type"], body))
                size = len(body["prompt"])
                if body["max_tokens"] > 1:
                    mixed.wait()
                if size == 8 and body["model"] == warmup.BASE_MODEL:
                    self.send_error(503)
                    return
                if size == 16 and body["model"] == warmup.BASE_MODEL:
                    size += 1
                reply = json.dumps({"usage": {
                    "prompt_tokens": size, "completion_tokens": body["max_tokens"],
                }}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *_):
                pass

        # Real HTTP over local socket pairs also works in sandboxes that
        # disallow binding a TCP listener.
        class Connection(warmup.http.client.HTTPConnection):
            def connect(self):
                self.sock, peer = socket.socketpair()
                self.sock.settimeout(self.timeout)

                def serve():
                    with peer:
                        Handler(peer, ("local test", 0), None)

                worker = Thread(target=serve)
                workers.append(worker)
                worker.start()

        log = io.StringIO()
        with patch.object(warmup.http.client, "HTTPConnection", Connection), redirect_stdout(log):
            warmup.warmup(8001)
        for worker in workers:
            worker.join()

        models = (warmup.BASE_MODEL, warmup.ABLITERATED_MODEL)
        single = [(model, size) for size in warmup.PREFILL_SIZES for model in models]
        self.assertEqual(len(requests), len(single) + 2)
        bodies = [body for _, _, body in requests]
        self.assertEqual([(b["model"], len(b["prompt"])) for b in bodies[:-2]], single)
        self.assertCountEqual([b["model"] for b in bodies[-2:]], models)
        self.assertEqual({len(b["prompt"]) for b in bodies[-2:]}, {warmup.MIXED_SIZE})
        self.assertEqual([b["max_tokens"] for b in bodies], [1] * len(single) + [16, 16])
        self.assertEqual(len({b["cache_salt"] for b in bodies}), len(bodies))
        for path, content_type, body in requests:
            self.assertEqual((path, content_type), ("/v1/completions", "application/json"))
            self.assertTrue(body["cache_salt"])
            self.assertTrue(all(isinstance(token, int) for token in body["prompt"]))
            self.assertFalse(body["add_special_tokens"])
            self.assertEqual(body["temperature"], 0)
            self.assertTrue(body["ignore_eos"])
        self.assertIn("HTTP 503", log.getvalue())
        self.assertIn("server prompt count differs", log.getvalue())
        self.assertIn("ok=14 failed=2", log.getvalue())

    def test_skip_and_exhausted_budget(self):
        with patch.dict(warmup.os.environ, WARMUP="0"), patch.object(sys, "argv", ["warmup.py"]):
            with patch.object(warmup, "warmup") as run, redirect_stdout(io.StringIO()):
                warmup.main()
            run.assert_not_called()
        log = io.StringIO()
        with patch.object(warmup.http.client, "HTTPConnection") as connect, redirect_stdout(log):
            self.assertFalse(warmup.warm_request(
                8001, warmup.BASE_MODEL, 8, 1, "single", deadline=0,
            ))
        connect.assert_not_called()
        self.assertIn("WARNING", log.getvalue())
        self.assertIn("budget exhausted", log.getvalue())
