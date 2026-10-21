#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh loaded-model prefill screen. TTFT rates are end-to-end proxies, not kernel rates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import secrets
import statistics
import threading
import time
import urllib.request

WORDS = 'the room light blue green red small large clear warm cold window floor door table book chair water air tree road field cloud rain sun wind stone glass paper simple quiet bright dark soft hard open close fast slow'.split()
METRIC = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def http_json(base, route, body, timeout):
    request = urllib.request.Request(base + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def bounded_timeout(timeout, deadline):
    remaining = deadline-time.monotonic() if deadline is not None else timeout
    if remaining <= 0: raise TimeoutError("screen budget exhausted")
    return min(timeout, remaining)


def fresh_prompt(target, base, model, timeout, deadline=None):
    # Randomness begins immediately after the unavoidable chat-template prefix;
    # >128 random bytes precede the word body so no full KV block can repeat.
    lead = secrets.token_hex(128) + '\n'
    rng = random.Random(secrets.randbits(128))
    words = [rng.choice(WORDS) for _ in range(target)]
    text = lead + ' '.join(words) + '\nReply with one word.'
    calibration = []
    for _ in range(3):
        result = http_json(base, '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False}, bounded_timeout(timeout, deadline))
        count = result['count']
        if not isinstance(count, int) or count <= 0: raise ValueError('invalid tokenize count')
        calibration.append(count)
        if abs(count-target) < max(32, target*.015): break
        length = max(1, round(len(words) * (target-180) / max(1, count-180)))
        if length > len(words): words += [rng.choice(WORDS) for _ in range(length-len(words))]
        words = words[:length]
        text = lead + ' '.join(words) + '\nReply with one word.'
    return text, calibration


def stream_one(base, model, text, timeout, barrier=None, deadline=None):
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'chat_template_kwargs': {'thinking': False}, 'max_tokens': 1,
            'ignore_eos': True, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}
    request = urllib.request.Request(base+'/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    if barrier: barrier.wait(timeout=10)
    start = time.monotonic(); first = None; usage = None; output = ''; terminal_only = False
    with urllib.request.urlopen(request, timeout=bounded_timeout(timeout, deadline)) as response:
        for raw in response:
            bounded_timeout(timeout, deadline)
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            payload = line[5:].strip()
            if payload == '[DONE]': break
            data = json.loads(payload)
            if data.get('error'): raise RuntimeError(str(data['error']))
            usage = data.get('usage') or usage
            for choice in data.get('choices', []):
                delta = choice.get('delta') or {}
                piece = (delta.get('content') or '') + (delta.get('reasoning_content') or delta.get('reasoning') or '')
                if first is None and (piece or choice.get('finish_reason') is not None):
                    first = time.monotonic(); terminal_only = not bool(piece)
                output += piece
    end = time.monotonic()
    if first is None or usage is None: raise ValueError('stream lacks first output event or usage')
    prompt = usage.get('prompt_tokens'); generated = usage.get('completion_tokens')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    if not isinstance(prompt, int) or prompt <= 0 or generated != 1:
        raise ValueError('expected positive prompt count and exactly one completion token: '+str(usage))
    if not isinstance(cached, int) or cached < 0: raise ValueError('cached token count missing/invalid; cannot certify cold input')
    return {'prompt_tokens': prompt, 'cached_tokens': cached, 'completion_tokens': generated,
            'ttft_s': first-start, 'elapsed_s': end-start,
            'input_tokens_per_ttft_s_e2e_proxy': prompt/(first-start),
            'first_event_terminal_only': terminal_only, 'output': output,
            'start_monotonic': start, 'first_monotonic': first, 'end_monotonic': end,
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'usage': usage}


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        match = METRIC.match(line)
        if not match: continue
        name, labels, value = match.groups()
        if not name.startswith('vllm:'): continue
        try: value = float(value)
        except ValueError: continue
        if math.isfinite(value): result[(name, labels or '')] = value
    return result


def metric_delta(before, after):
    values = []
    for (name, labels), value in sorted(after.items()):
        if not any(word in name for word in ('prefill', 'time_to_first_token', 'queue', 'prompt_tokens', 'prefix_cache', 'request_success')): continue
        if (name, labels) not in before: continue
        delta = value-before[(name, labels)]
        values.append({'name': name, 'labels': labels, 'before': before[(name, labels)], 'after': value,
                       'delta': delta, 'counter_reset': delta < 0})
    sums = [r for r in values if r['name'] == 'vllm:request_prefill_time_seconds_sum' and not r['counter_reset']]
    counts = [r for r in values if r['name'] == 'vllm:request_prefill_time_seconds_count' and not r['counter_reset']]
    count = sum(r['delta'] for r in counts)
    return {'series': values, 'server_request_prefill_observation_count': count if counts else None,
            'server_request_prefill_mean_s': sum(r['delta'] for r in sums)/count if sums and count > 0 else None,
            'server_request_prefill_sum_s': sum(r['delta'] for r in sums) if sums else None,
            'note': 'Server request-prefill histogram observations, when present; overlapping requests may sum overlapping time. Not GPU kernel time. Other traffic contaminates these deltas.'}


def snapshot(base, path, timeout):
    try:
        with urllib.request.urlopen(base+'/metrics', timeout=timeout) as response: text=response.read(16*1024*1024).decode()
        path.write_text(text)
        return parse_metrics(text), None
    except Exception as error:
        path.write_text('# metrics unavailable: '+repr(error)+'\n')
        return {}, repr(error)


def run_batch(base, model, target, concurrency, timeout, retries=2, deadline=None):
    attempts = []
    for attempt in range(retries+1):
        prompts = [fresh_prompt(target, base, model, timeout, deadline) for _ in range(concurrency)]
        barrier = threading.Barrier(concurrency)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(stream_one, base, model, text, timeout, barrier, deadline) for text, _ in prompts]
            rows = [f.result() for f in futures]
        for row, (_, calibration) in zip(rows, prompts): row['text_tokenize_calibration']=calibration
        cold = all(row['cached_tokens'] == 0 for row in rows)
        attempts.append({'attempt': attempt, 'cold': cold, 'requests': rows})
        if cold:
            first_start = min(r['start_monotonic'] for r in rows)
            last_first = max(r['first_monotonic'] for r in rows)
            total = sum(r['prompt_tokens'] for r in rows)
            return {'target_text_tokens': target, 'concurrency': concurrency, 'status': 'COLD',
                    'attempts': attempts, 'prompt_tokens_total': total,
                    'ttft_median_s': statistics.median(r['ttft_s'] for r in rows),
                    'ttft_max_s': max(r['ttft_s'] for r in rows),
                    'batch_first_output_span_s': last_first-first_start,
                    'aggregate_input_tok_s_e2e_proxy': total/(last_first-first_start),
                    'actual_prompt_tokens_min': min(r['prompt_tokens'] for r in rows),
                    'actual_prompt_tokens_max': max(r['prompt_tokens'] for r in rows)}
    return {'target_text_tokens': target, 'concurrency': concurrency, 'status': 'REJECTED_CACHED', 'attempts': attempts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--model', default='deepseek-v4.1-flash-1017')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=270)
    parser.add_argument('--request-timeout', type=float, default=75)
    parser.add_argument('--include-32k', action='store_true')
    args=parser.parse_args()
    if args.budget_seconds<=0 or args.request_timeout<=0: parser.error('timeouts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    base=args.base_url.rstrip('/'); start=time.monotonic()
    result={'started_utc': datetime.now(timezone.utc).isoformat(), 'model': args.model,
            'protocol': 'fresh synthetic random-leading text; thinking=false; max_tokens=1; no model restart',
            'interpretation': 'Input tokens / client TTFT and aggregate batch first-output span are end-to-end proxies including HTTP, tokenization, scheduling, queueing, prefill and first decode. Not pure kernel prefill.',
            'batches': [], 'status': 'RUNNING'}
    def save():
        temporary=args.output/'summary.json.tmp'; temporary.write_text(json.dumps(result,indent=2)+'\n'); temporary.replace(args.output/'summary.json')
    save()
    try:
        plan=[(target,c) for target in (2048,8192) for c in (1,4,8)]
        if args.include_32k: plan.append((32768,1))
        for target,c in plan:
            remaining=args.budget_seconds-(time.monotonic()-start)
            if remaining < 15:
                result['status']='BUDGET_EXHAUSTED'; break
            stem=f't{target}-c{c}'
            before,be=snapshot(base,args.output/(stem+'-before.prom'),min(10,remaining))
            batch=run_batch(base,args.model,target,c,min(args.request_timeout,max(1,remaining-5)), deadline=start+args.budget_seconds-5)
            after,ae=snapshot(base,args.output/(stem+'-after.prom'),10)
            batch['metrics']=metric_delta(before,after);batch['metrics_errors']=[x for x in (be,ae) if x]
            metrics=batch['metrics']; total_s=metrics['server_request_prefill_sum_s']
            clean=(not batch['metrics_errors'] and metrics['server_request_prefill_observation_count']==c
                   and len(batch['attempts'])==1 and total_s is not None and total_s>0
                   and not any(r['counter_reset'] for r in metrics['series']))
            batch['server_prefill_input_tok_s_per_request_time_normalized']=(
                batch['prompt_tokens_total']/total_s if clean and batch['status']=='COLD' else None)
            batch['server_rate_note']='Input tokens divided by summed server request-prefill seconds; per-request-time-normalized, NOT aggregate throughput. Only emitted for one cold attempt and histogram count equal to concurrency.'
            result['batches'].append(batch);save()
            print(json.dumps(batch),flush=True)
            if batch['status']!='COLD': raise RuntimeError('could not obtain zero-cache batch')
        else: result['status']='COMPLETED'
    except Exception as error:
        result['status']='FAILED';result['error']=repr(error);raise
    finally:
        result['elapsed_s']=time.monotonic()-start;save()
    if result['status']!='COMPLETED': raise SystemExit(2)


if __name__=='__main__': main()
