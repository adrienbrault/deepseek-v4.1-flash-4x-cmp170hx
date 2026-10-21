# SPDX-License-Identifier: MIT
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

def restore(snapshot):
    if snapshot.exists():
        settings = json.loads(snapshot.read_text())
        for path in settings:
            p = Path(path)
            assert p.name == 'scaling_max_freq' and p.parent.parent == Path('/sys/devices/system/cpu/cpufreq')
        errors = []
        for path, value in settings.items():
            try:
                Path(path).write_text(value)
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            raise OSError('; '.join(errors))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore', type=Path)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.restore:
        restore(a.restore)
        return
    assert a.snapshot and a.out
    policies = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_max_freq'))
    assert policies
    originals = {str(f): f.read_text() for f in policies}
    cap = 1600000
    assert all((int(v) > cap for v in originals.values()))
    assert all((int((f.parent / 'scaling_min_freq').read_text()) <= cap for f in policies))
    a.snapshot.write_text(json.dumps(originals, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('serve_bench', Path(__file__).with_name('serve_bench.py'))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    request_args = SimpleNamespace(url='http://127.0.0.1:8001', model='deepseek-v4.1-flash-1017')
    result = {'status': 'RUNNING', 'cap_khz': cap, 'original_max_khz': sorted(set(map(int, originals.values()))), 'seed': 15, 'temperature': 0, 'completion_tokens': 400, 'ignore_eos': True, 'measurements': []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.out.write_text(json.dumps(result, indent=2) + '\n')

    def interrupted(sig, frame):
        raise SystemExit(128 + sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)

    def prompt(i):
        return f'Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with a short docstring. Nonce {150000 + i}.'
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: bench.request(request_args, prompt(i), 32), range(4)))
        for index, state in enumerate(('normal', 'limited', 'limited', 'normal', 'normal', 'limited')):
            for f in policies:
                f.write_text(originals[str(f)] if state == 'normal' else str(cap))
            time.sleep(1)
            for concurrency in (1, 4):
                samples, stop = ([], threading.Event())

                def sample():
                    while not stop.is_set():
                        samples.append([float(line.split(':')[1]) for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('cpu MHz')])
                        stop.wait(0.5)
                sampler = threading.Thread(target=sample)
                sampler.start()
                started = time.monotonic()
                try:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(lambda i: bench.request(request_args, prompt(i), 400), range(concurrency)))
                finally:
                    stop.set()
                    sampler.join()
                r = {'round': index, 'state': state, 'concurrency': concurrency, 'requests': rows, 'batch_wall_s': time.monotonic() - started, 'aggregate_decode_tok_s': sum((x['decode_tps'] for x in rows)), 'cpu_mhz_samples': samples}
                result['measurements'].append(r)
                save()
                print(json.dumps({k: r[k] for k in ('round', 'state', 'concurrency', 'aggregate_decode_tok_s')}), flush=True)
        result['summary'] = {str(c): {s: statistics.mean((x['aggregate_decode_tok_s'] for x in result['measurements'] if x['concurrency'] == c and x['state'] == s)) for s in ('normal', 'limited')} for c in (1, 4)}
        result['status'] = 'PASS'
    finally:
        restore(a.snapshot)
        result['restored'] = all((f.read_text().strip() == originals[str(f)].strip() for f in policies))
        save()
    print(json.dumps(result['summary']), flush=True)
if __name__ == '__main__':
    main()
