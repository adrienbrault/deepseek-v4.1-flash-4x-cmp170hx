#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Durable serial experiment queue. --once runs at most one job, then returns.

Run under systemd KillMode=control-group; each script must also hold the GPU
flock. Restarted 'running' jobs are interrupted, never retried automatically.
'completed' means wrapper exit zero, not a speed or quality verdict.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

TERMINAL = {"completed", "failed", "interrupted"}
BUSY = {"active", "activating", "deactivating", "reloading"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class WorkLoop:
    def __init__(self, queue, state, log_dir, poll_seconds=2):
        self.queue_path = Path(queue)
        self.state_path = Path(state)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.stop = False
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"version": 1, "jobs": {}}
        # Per-job receipt is committed first. Recover it if a crash occurred
        # before the aggregate state write, so a finished job cannot rerun.
        for path in self.log_dir.glob("*.json"):
            receipt = json.loads(path.read_text())
            job_id = receipt["id"]
            if path.name != job_id + ".json":
                raise ValueError("receipt filename/id mismatch")
            self.state["jobs"][job_id] = receipt
        for job_id, receipt in self.state["jobs"].items():
            if receipt["status"] == "running":
                receipt.update(status="interrupted", finished_at=now(), reason="worker restarted; prior service cgroup must be stopped")
                self.receipt(job_id, receipt)
        self.save()

    def save(self):
        self.state["heartbeat_at"] = now()
        atomic_json(self.state_path, self.state)

    def receipt(self, job_id, receipt):
        self.state["jobs"][job_id] = receipt
        atomic_json(self.log_dir / (job_id + ".json"), receipt)
        self.save()

    def status(self, value, **detail):
        changed = self.state.get("status") != value
        self.state.update(status=value, **detail)
        if changed:
            with (self.log_dir / "worker.log").open("a") as out:
                out.write(f"{now()} {value}\n")
        self.save()

    def blocked(self, units):
        for unit in units:
            result = subprocess.run(["systemctl", "show", unit, "--property=ActiveState", "--value"],
                                    capture_output=True, text=True, timeout=10)
            # Failure/unknown state is never permission to overlap another unit.
            if result.returncode or result.stdout.strip() not in {"inactive", "failed"}:
                return unit
        return None

    def terminate(self, child):
        # Only this worker's explicitly created child process group is touched.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # Kill remaining descendants too, including a child ignoring SIGTERM.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    def execute(self, job):
        job_id = job["id"]
        receipt = {"id": job_id, "status": "running", "started_at": now(),
                   "script": job["script"], "timeout_s": job["timeout_s"],
                   "log": str(self.log_dir / (job_id + ".log"))}
        self.receipt(job_id, receipt)
        env = dict(os.environ, **job.get("env", {}))
        env["SCREEN_RESTORE_GLM"] = "0"
        child = None
        start = time.monotonic()
        reason = None
        try:
            with Path(receipt["log"]).open("ab", buffering=0) as out:
                child = subprocess.Popen(["bash", job["script"]], env=env, stdout=out,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                receipt["child_pid"] = child.pid
                self.receipt(job_id, receipt)
                while child.poll() is None:
                    self.save()
                    if self.stop or time.monotonic() - start >= job["timeout_s"]:
                        reason = "stopped" if self.stop else "timeout"
                        self.terminate(child)
                        break
                    time.sleep(self.poll_seconds)
                code = child.wait()
                # A wrapper may exit while leaving descendants: do not let those
                # overlap the next queued job, regardless of wrapper exit status.
                self.terminate(child)
            receipt.update(returncode=code, status="interrupted" if reason == "stopped" else
                           "failed" if reason or code else "completed")
            if reason:
                receipt["reason"] = reason
        except Exception as exc:
            if child is not None:
                self.terminate(child)
            receipt.update(status="failed", reason=str(exc), returncode=child.returncode if child else None)
        finally:
            receipt.update(finished_at=now(), elapsed_s=time.monotonic() - start)
            self.receipt(job_id, receipt)

    def pass_once(self):
        try:
            queue = json.loads(self.queue_path.read_text())
            if queue.get("version") != 1 or not isinstance(queue.get("jobs"), list):
                raise ValueError("queue must have version 1 and jobs list")
            jobs = queue["jobs"]
            ids = set()
            for job in jobs:
                job_id = job["id"]
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id) or job_id in ids:
                    raise ValueError("invalid or duplicate job id")
                ids.add(job_id)
                if not isinstance(job["script"], str) or not Path(job["script"]).is_absolute():
                    raise ValueError("job script must be absolute")
                if not isinstance(job["timeout_s"], (int, float)) or job["timeout_s"] <= 0:
                    raise ValueError("positive timeout_s required")
                if not isinstance(job.get("env", {}), dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in job.get("env", {}).items()):
                    raise ValueError("env must map strings to strings")
            blocker = self.blocked(queue.get("wait_units", []))
        except Exception as exc:
            self.status("queue_error", error=str(exc))
            return
        if blocker:
            self.status("waiting_external", wait_unit=blocker)
            return
        for job in jobs:
            if self.stop:
                return
            if self.state["jobs"].get(job["id"], {}).get("status") in TERMINAL:
                continue
            self.status("running", current_job=job["id"])
            self.execute(job)
            return
        self.status("needs_plan", current_job=None)

    def run(self, once=False):
        while not self.stop:
            self.pass_once()
            if once:
                break
            time.sleep(self.poll_seconds)
        if self.stop:
            self.status("stopped", current_job=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "state", "log-dir", "lock"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    lock = Path(args.lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "another work-loop owns the lock\n")
        loop = WorkLoop(args.queue, args.state, args.log_dir, args.poll_seconds)
        def stop(signum, frame):
            loop.stop = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        loop.run(args.once)


if __name__ == "__main__":
    main()
