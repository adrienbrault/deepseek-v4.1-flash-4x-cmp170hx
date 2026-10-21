#!/usr/bin/env python3
"""Read-only experiment telemetry; failures are receipts, never benchmark failures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

GPU_QUERY = ('timestamp,index,utilization.gpu,memory.used,power.draw,'
             'clocks.current.sm,clocks.current.memory,temperature.gpu,pstate')
SELECTED = {'num_requests_running', 'num_requests_waiting', 'num_requests_swapped',
            'gpu_cache_usage_perc', 'cpu_cache_usage_perc', 'kv_cache_usage_perc',
            'prompt_tokens_total', 'generation_tokens_total', 'num_preemptions_total',
            'prefix_cache_hits_total', 'prefix_cache_queries_total',
            'gpu_prefix_cache_hits_total', 'gpu_prefix_cache_queries_total'}
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def filter_metrics(payload):
    result = []
    for line in payload.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        short = name.removeprefix('vllm:')
        if short not in SELECTED and not (
            short.startswith('spec_decode_') and not short.endswith(('_bucket', '_sum', '_count'))
        ):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        result.append({'name': name, 'labels': labels or '',
                       'value': number if math.isfinite(number) else value})
    return result


def active_cpu_policies(root=Path('/sys/devices/system/cpu')):
    """Only online CPUs: offline cpufreq policies may reject reads with EBUSY."""
    online = set()
    for span in (root / 'online').read_text().strip().split(','):
        bounds = span.split('-')
        online.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    policies = {}
    for cpu in sorted(online):
        policy = (root / f'cpu{cpu}' / 'cpufreq').resolve()
        if str(policy) in policies:
            continue
        values = {}
        for key in ('scaling_cur_freq', 'scaling_min_freq', 'scaling_max_freq',
                    'cpuinfo_cur_freq', 'scaling_governor'):
            try:
                values[key] = (policy / key).read_text().strip()
            except OSError as error:
                values[key] = {'error': type(error).__name__}
        policies[str(policy)] = values
    return policies


def system_sample():
    result = {'utc': utc()}
    try:
        result['cpu_policies'] = active_cpu_policies()
    except (OSError, ValueError) as error:
        result['cpu_policies_error'] = str(error)
    for filename, selected in (
        ('vmstat', {'pgfault', 'pgmajfault', 'pswpin', 'pswpout', 'pgpgin', 'pgpgout'}),
        ('meminfo', {'MemAvailable:', 'MemFree:', 'Cached:', 'SwapFree:', 'Dirty:', 'Writeback:'}),
        ('stat', {'cpu', 'procs_running', 'procs_blocked', 'ctxt'}),
    ):
        try:
            result[filename] = {parts[0]: parts[1:] for line in Path('/proc', filename).read_text().splitlines()
                                if (parts := line.split()) and parts[0] in selected}
        except OSError as error:
            result[filename + '_error'] = str(error)
    return result


class Collector:
    def __init__(self, output, interval=2.0, url='http://127.0.0.1:8001'):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.url = url.rstrip('/') + '/metrics'
        self.stopping = False
        self.children = []
        self.streams = []
        self.status = {'started_utc': utc(), 'pid': os.getpid(), 'hostname': socket.gethostname(),
                       'interval_seconds': interval, 'metrics_url': self.url, 'commands': []}

    def receipt(self):
        for entry, process in self.children:
            entry['returncode'] = process.poll()
            if entry['returncode'] is not None:
                entry['status'] = 'exited'
        atomic_json(self.output / 'collector-status.json', self.status)

    def start_tools(self):
        interval = str(max(1, round(self.interval)))
        commands = [
            ('iostat', ['iostat', '-dx', '-y', interval]),
            ('pidstat', ['pidstat', '-dru', '-h', '-p', 'ALL', interval]),
            ('nvidia-smi', ['nvidia-smi', '--query-gpu=' + GPU_QUERY,
                            '--format=csv', '-l', interval]),
        ]
        for name, argv in commands:
            entry = {'tool': name, 'argv': argv}
            self.status['commands'].append(entry)
            if shutil.which(argv[0]) is None:
                entry['status'] = 'missing'
                continue
            stream = None
            try:
                stream = (self.output / (name + '.log')).open('ab', buffering=0)
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.streams.append(stream)
                self.children.append((entry, process))
                entry.update(status='started', pid=process.pid, pgid=process.pid)
            except OSError as error:
                if stream is not None:
                    stream.close()
                entry.update(status='spawn_error', error=str(error))
        self.receipt()

    def stop_tools(self):
        # Signal exact process groups we created, even if their leaders exited.
        for _, process in self.children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2.0
        for _, process in self.children:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for entry, process in self.children:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                entry['reap_error'] = 'timeout'
        for stream in self.streams:
            stream.close()
        self.status['ended_utc'] = utc()
        self.status['stopped'] = True
        self.receipt()

    def sample(self):
        sample = {'utc': utc()}
        try:
            with urllib.request.urlopen(self.url, timeout=min(1.0, self.interval)) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                if len(payload) > 4 * 1024 * 1024:
                    raise ValueError('metrics response exceeded 4 MiB')
            sample.update(status='ok', metrics=filter_metrics(payload.decode('utf-8', errors='replace')))
        except Exception as error:
            sample.update(status='error', error=str(error))
        append_json(self.output / 'server-metrics.jsonl', sample)
        append_json(self.output / 'system.jsonl', system_sample())
        atomic_json(self.output / 'heartbeat.json', {'utc': utc(), 'pid': os.getpid(), 'stopping': self.stopping})
        self.receipt()

    def run(self):
        previous = {}
        def stop(_signum, _frame):
            self.stopping = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stop)
        atomic_json(self.output / 'meta.json', dict(self.status, collector_version=1,
                    gpu_query=GPU_QUERY, selected_metrics=sorted(SELECTED)))
        atomic_json(self.output / 'start.json', self.status)
        try:
            self.start_tools()
            while not self.stopping:
                started = time.monotonic()
                self.sample()
                while not self.stopping and time.monotonic() - started < self.interval:
                    time.sleep(min(0.1, self.interval))
        finally:
            self.stop_tools()
            atomic_json(self.output / 'end.json', self.status)
            atomic_json(self.output / 'heartbeat.json', {'utc': utc(), 'pid': os.getpid(), 'stopped': True})
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0.1:
        parser.error('--interval must be at least 0.1 seconds')
    Collector(args.output_dir, args.interval, args.url).run()


if __name__ == '__main__':
    main()
