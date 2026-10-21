#!/usr/bin/env python3
"""Bounded, unscored CPU/GPU diagnosis on the existing loaded model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def request(base, path, payload=None, timeout=45):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def workers(container, controls=False):
    text = subprocess.check_output(['docker', 'top', container, '-eo', 'pid,args'], text=True, timeout=10)
    return [int(p[0]) for line in text.splitlines()[1:]
            if len(p := line.strip().split(maxsplit=1)) == 2
            and (bool(re.fullmatch(r'VLLM::Worker_TP[0-9]+', p[1])) if not controls
                 else p[1] == 'VLLM::EngineCore' or 'vllm serve ' in p[1])]


def stop_child(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--container', default='dsv41-1017')
    parser.add_argument('--prompt-doc', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'started_at': timestamp(), 'classification': 'diagnostic only; excluded from benchmark scores',
              'status': 'running', 'phases': [], 'errors': [], 'capture_iterations': 12}
    perf = None
    perf_log = None
    active = False
    started = time.monotonic()
    def interrupted(signum, frame):
        raise InterruptedError('diagnostic interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        model = request(args.url, '/v1/models')['data'][0]['id']
        config = json.loads(subprocess.check_output(
            ['docker', 'inspect', args.container, '--format', '{{json .Config.Cmd}}'], text=True, timeout=10))
        result['container_command'] = config
        pids = workers(args.container)
        result['worker_pids'] = pids
        try:
            if len(pids) != 4:
                raise ValueError(f'expected four TP workers, found {pids}')
            cpu_pids = sorted(set(pids + workers(args.container, controls=True)))
            result['cpu_sample_pids'] = cpu_pids
            perf_log = (args.output / 'cpu-perf-record.txt').open('w')
            perf = subprocess.Popen(['perf', 'record', '-F', '99', '-e', 'cycles', '--call-graph', 'fp',
                '-p', ','.join(map(str, cpu_pids)), '-o', str(args.output / 'cpu-perf.data'), '--', 'sleep', '12'],
                stdout=perf_log, stderr=subprocess.STDOUT, start_new_session=True)
        except (OSError, ValueError) as exc:
            result['errors'].append('CPU sample: ' + str(exc))
        document = args.prompt_doc.read_text() if args.prompt_doc else (
            'def total(values):\n    return sum(values)\n' * 400)
        def completion(index, long=False, tokens=128):
            code = document[index * 400:index * 400 + (12000 if long else 260)]
            prompt = f'Explain what this code does, including its edge cases. Sample {index}.\n{code}'
            response = request(args.url, '/v1/completions', {
                'model': model, 'prompt': prompt, 'max_tokens': tokens, 'ignore_eos': True,
                'temperature': 0, 'seed': 15})
            return {'usage': response.get('usage'), 'index': index, 'long_prefill': long}
        # Warm the existing model before diagnostic capture; no score is computed.
        completion(0, tokens=16)
        routes = request(args.url, '/openapi.json').get('paths', {})
        ready = '/start_profile' in routes and '/stop_profile' in routes
        result['gpu_profiler_ready'] = ready
        if not ready:
            result['errors'].append('GPU profiler was not enabled at this model boot')
        for label, count in [('c1', 1), ('c4-mixed', 4)]:
            phase = {'label': label, 'started_at': timestamp(), 'requests': []}
            try:
                if ready:
                    active = True
                    request(args.url, '/start_profile', {'profile_prefix': label,
                        'delay_iterations': 0, 'max_iterations': 12}, timeout=60)
                with ThreadPoolExecutor(max_workers=count) as pool:
                    jobs = [pool.submit(completion, i + (0 if count == 1 else 4),
                                        count == 4 and i == 3) for i in range(count)]
                    phase['requests'] = [f.result() for f in jobs]
            except InterruptedError:
                raise
            except Exception as exc:
                phase['error'] = str(exc)
                result['errors'].append(label + ': ' + str(exc))
            finally:
                if active:
                    try:
                        request(args.url, '/stop_profile', {}, timeout=60)
                        active = False
                    except Exception as exc:
                        result['errors'].append('stop_profile: ' + str(exc))
            phase['finished_at'] = timestamp()
            result['phases'].append(phase)
            if active:
                break  # Never issue more requests while stopping is uncertain.
    except Exception as exc:
        result['errors'].append(str(exc))
    finally:
        if active:
            try:
                request(args.url, '/stop_profile', {}, timeout=60)
                active = False
            except Exception as exc:
                result['errors'].append('cleanup stop_profile: ' + str(exc))
        stop_child(perf)
        if perf_log:
            perf_log.close()
        if perf is not None:
            result['perf_returncode'] = perf.returncode
            if perf.returncode == 0:
                with (args.output / 'cpu-perf-report.txt').open('w') as out:
                    report = subprocess.run(['perf', 'report', '--stdio', '--no-children',
                        '--sort', 'dso,symbol', '-i', str(args.output / 'cpu-perf.data')],
                        stdout=out, stderr=subprocess.STDOUT, timeout=20)
                if report.returncode:
                    result['errors'].append(f'perf report exit {report.returncode}')
            else:
                result['errors'].append(f'perf record exit {perf.returncode}')
        result['elapsed_s'] = round(time.monotonic() - started, 3)
        result['finished_at'] = timestamp()
        result['safe_for_scored_work'] = not active
        result['status'] = 'partial' if result['errors'] else 'completed'
        (args.output / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
