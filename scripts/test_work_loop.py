#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name("work-loop.py")
spec = importlib.util.spec_from_file_location("work_loop", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class WorkLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.queue = self.root / "queue.json"
        self.state = self.root / "state.json"
        self.logs = self.root / "logs"
        self.write([])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, jobs, units=None):
        m.atomic_json(self.queue, {"version": 1, "jobs": jobs, "wait_units": units or []})

    def job(self, name, body, timeout=5):
        script = self.root / (name + ".sh")
        script.write_text(body)
        return {"id": name, "script": str(script), "env": {"SCREEN_RESTORE_GLM": "1"}, "timeout_s": timeout}

    def loop(self):
        return m.WorkLoop(self.queue, self.state, self.logs, .02)

    def test_success_failure_continue_and_resume(self):
        record = self.root / "record"
        jobs = [self.job("one", f'echo "one:$SCREEN_RESTORE_GLM" >> {record}\n'),
                self.job("bad", f'echo bad >> {record}\nexit 7\n'),
                self.job("three", f'echo three >> {record}\n')]
        self.write(jobs)
        loop = self.loop()
        for _ in jobs:
            loop.pass_once()
        self.assertEqual(record.read_text().splitlines(), ["one:0", "bad", "three"])
        self.assertEqual([loop.state["jobs"][j["id"]]["status"] for j in jobs], ["completed", "failed", "completed"])
        self.assertEqual(loop.state["jobs"]["bad"]["returncode"], 7)
        resumed = self.loop()
        resumed.pass_once()
        resumed.pass_once()
        self.assertEqual(resumed.state["status"], "needs_plan")
        self.assertEqual(len(record.read_text().splitlines()), 3)
        self.assertEqual((self.logs / "worker.log").read_text().count("needs_plan"), 1)

    def test_restart_running_is_interrupted_but_next_executes(self):
        jobs = [self.job("old", "exit 99\n"), self.job("next", "exit 0\n")]
        self.write(jobs)
        m.atomic_json(self.state, {"version": 1, "jobs": {"old": {"id": "old", "status": "running"}}})
        loop = self.loop()
        loop.pass_once()
        self.assertEqual(loop.state["jobs"]["old"]["status"], "interrupted")
        self.assertEqual(loop.state["jobs"]["next"]["status"], "completed")

    def test_receipt_survives_crash_before_aggregate_state(self):
        self.write([self.job("done", "exit 99\n")])
        m.atomic_json(self.state, {"version": 1, "jobs": {"done": {"id": "done", "status": "running"}}})
        m.atomic_json(self.logs / "done.json", {"id": "done", "status": "completed", "returncode": 0})
        loop = self.loop()
        loop.pass_once()
        self.assertEqual(loop.state["jobs"]["done"]["status"], "completed")
        self.assertFalse((self.logs / "done.log").exists())

    def test_external_unit_and_fresh_queue_after_empty(self):
        first = self.job("discard", "exit 99\n")
        second = self.job("fresh", "exit 0\n")
        self.write([first], ["exact.service"])
        loop = self.loop()
        with patch.object(m.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "active\n", "")) as call:
            loop.pass_once()
            self.assertEqual(loop.state["status"], "waiting_external")
            self.assertEqual(call.call_args.args[0], ["systemctl", "show", "exact.service", "--property=ActiveState", "--value"])
        self.assertFalse(loop.state["jobs"])
        self.write([])
        loop.pass_once()
        self.assertEqual(loop.state["status"], "needs_plan")
        self.write([second])
        loop.pass_once()
        self.assertEqual(loop.state["jobs"]["fresh"]["status"], "completed")
        self.assertNotIn("discard", loop.state["jobs"])

    def alive(self, pid):
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            return text.split(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    def test_timeout_cleans_child_group_and_continues(self):
        pidfile = self.root / "child.pid"
        self.write([self.job("slow", f'sleep 30 &\necho $! > {pidfile}\nwait\n', .12),
                    self.job("after", "exit 0\n")])
        loop = self.loop()
        loop.pass_once()
        child = int(pidfile.read_text())
        self.assertFalse(self.alive(child))
        self.assertEqual(loop.state["jobs"]["slow"]["reason"], "timeout")
        loop.pass_once()
        self.assertEqual(loop.state["jobs"]["after"]["status"], "completed")

    def test_sigterm_reaps_owned_group_and_lock_excludes_second_worker(self):
        pidfile = self.root / "child.pid"
        self.write([self.job("sleep", f'sleep 30 &\necho $! > {pidfile}\nwait\n')])
        cmd = ["python3", str(MODULE), "--queue", str(self.queue), "--state", str(self.state),
               "--log-dir", str(self.logs), "--lock", str(self.root / "lock"), "--poll-seconds", ".02"]
        worker = subprocess.Popen(cmd)
        try:
            end = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < end:
                time.sleep(.02)
            self.assertTrue(pidfile.exists())
            other = subprocess.run(cmd + ["--once"], capture_output=True, timeout=3)
            self.assertEqual(other.returncode, 1)
            worker.send_signal(signal.SIGTERM)
            worker.wait(timeout=5)
            self.assertFalse(self.alive(int(pidfile.read_text())))
            self.assertEqual(json.loads(self.state.read_text())["jobs"]["sleep"]["status"], "interrupted")
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
