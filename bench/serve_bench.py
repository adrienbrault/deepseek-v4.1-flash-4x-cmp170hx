# SPDX-License-Identifier: MIT
import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def request(a, prompt, tokens):
    payload = {'model': a.model, 'prompt': prompt, 'max_tokens': tokens, 'temperature': 0, 'seed': 15, 'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(a.url.rstrip('/') + '/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first = usage = None
    done = False
    text = []
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            data = raw[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            assert 'error' not in event, event
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                piece = choice.get('text') or ''
                if piece:
                    first = first or time.monotonic()
                    text.append(piece)
    ended = time.monotonic()
    assert done and first and usage and (usage['completion_tokens'] == tokens), (done, usage)
    return {'ttft_seconds': first - started, 'wall_seconds': ended - started, 'completion_tokens': tokens, 'prompt_tokens': usage['prompt_tokens'], 'decode_tps': (tokens - 1) / (ended - first) if tokens > 1 else None, 'output_sha256': hashlib.sha256(''.join(text).encode()).hexdigest()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--model', default='deepseek-v4.1-flash-1017')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--runs', default=8, type=int)
    p.add_argument('--tokens', default=400, type=int)
    p.add_argument('--prefill-lengths', nargs='+', type=int, default=[8192, 32768, 137216])
    p.add_argument('--prefill-runs', type=int, default=3)
    a = p.parse_args()
    assert a.runs >= 8 and a.tokens > 1 and (a.prefill_runs >= 1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'FAIL', 'runs': a.runs, 'completion_tokens': a.tokens, 'seed': 15, 'decode': {}, 'prefill': {}}

    def save():
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    try:
        for concurrency in (1, 4, 8):

            def batch(run, concurrency=concurrency):
                prompts = [f'Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with a short docstring. Nonce {150000 + run * 8 + i}.' for i in range(concurrency)]
                begin = time.monotonic()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    requests = list(pool.map(lambda prompt: request(a, prompt, a.tokens), prompts))
                elapsed = time.monotonic() - begin
                return {'requests': requests, 'aggregate_decode_tps': sum((x['decode_tps'] for x in requests)), 'batch_wall_tps': concurrency * a.tokens / elapsed}
            batch(-1)
            measurements = [batch(run) for run in range(a.runs)]
            result['decode'][str(concurrency)] = {'measurements': measurements, 'mean_aggregate_decode_tps': statistics.mean((x['aggregate_decode_tps'] for x in measurements)), 'stdev_aggregate_decode_tps': statistics.stdev((x['aggregate_decode_tps'] for x in measurements))}
            save()
        for length in a.prefill_lengths:
            measurements = []
            for run in range(a.prefill_runs):
                rng = random.Random(150000 + length + run)
                measurement = request(a, [rng.randrange(1000, 20000) for _ in range(length)], 1)
                assert measurement['prompt_tokens'] == length
                measurement['prefill_tps_proxy'] = length / measurement['ttft_seconds']
                measurements.append(measurement)
            result['prefill'][str(length)] = {'measurements': measurements, 'mean_prefill_tps_proxy': statistics.mean((x['prefill_tps_proxy'] for x in measurements))}
            save()
        result['status'] = 'PASS'
    finally:
        save()
    print('PASS benchmark', a.output)
if __name__ == '__main__':
    main()
