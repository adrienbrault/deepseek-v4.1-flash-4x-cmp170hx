# SPDX-License-Identifier: MIT
import json, os, random, sys, threading, time, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else '8001'
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 400
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
kind = sys.argv[5] if len(sys.argv) > 5 else 'prose'
if 'DECODE_PROBE_SEED' in os.environ:
    random.seed(int(os.environ['DECODE_PROBE_SEED']))
CODE = 'def parse_duration(text):\n    """Parse an ISO 8601 duration such as P3DT4H5M into seconds."""\n    import re\n    m = re.fullmatch(r"P(?:(\\d+)D)?(?:T(?:(\\d+)H)?(?:(\\d+)M)?(?:(\\d+)S)?)?", text)\n    if not m:\n        raise ValueError(f"bad duration: {text}")\n    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())\n    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds\n'

def prompt():
    nonce = random.randrange(10 ** 9)
    if kind.startswith('long'):
        target = int(kind[4:-1]) * 1000
        blocks = []
        while len(blocks) * 110 < target:
            blocks.append(f'# block {len(blocks)} nonce {nonce}\n' + CODE.replace('parse_duration', f'parse_duration_{len(blocks)}'))
        return '\n'.join(blocks) + '\nHow many functions are defined above? Answer with one number.'
    if kind == 'code':
        return f'Reply with only the code, no thinking. Rewrite this function with the regex compiled once at module level, keeping everything else identical. Then repeat the function three more times, renamed parse_duration_2, _3, _4. Nonce {nonce}.\n\n```python\n{CODE}```'
    return f'Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with a short docstring. Nonce {nonce}.'

def one(results, i, text_prompt):
    body = {'model': 'deepseek-v4.1-flash-1017', 'messages': [{'role': 'user', 'content': text_prompt}], 'max_tokens': max_tokens, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}, 'ignore_eos': True}
    if 'DECODE_PROBE_SEED' in os.environ:
        body['seed'] = int(os.environ['DECODE_PROBE_SEED'])
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    usage = None
    text = ''
    for raw in urllib.request.urlopen(req, timeout=1800):
        line = raw.decode().strip()
        if not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        d = json.loads(line[6:])
        usage = d.get('usage') or usage
        for c in d.get('choices', []):
            delta = c.get('delta', {})
            s = (delta.get('content') or '') + (delta.get('reasoning_content') or delta.get('reasoning') or '')
            if s and first is None:
                first = time.time()
            text += s
    t1 = time.time()
    n = usage['completion_tokens']
    results[i] = (first - t0, n, (n - 1) / (t1 - first), usage['prompt_tokens'], text)
warmup = int(os.environ.get('DECODE_PROBE_WARMUP', '0'))
for run in range(-warmup, runs):
    results = [None] * conc
    prompts = [prompt() for i in range(conc)]
    threads = [threading.Thread(target=one, args=(results, i, prompts[i])) for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rates = [r[2] for r in results]
    assert all((r[1] == max_tokens for r in results)), results
    if run >= 0 and os.environ.get('DECODE_PROBE_OUT'):
        with open(os.environ['DECODE_PROBE_OUT'], 'a') as out:
            out.write(json.dumps({'run': run, 'concurrency': conc, 'kind': kind, 'max_tokens': max_tokens, 'temperature': 0, 'seed': os.environ.get('DECODE_PROBE_SEED'), 'aggregate_decode_tok_s': sum(rates), 'requests': [{'ttft_s': r[0], 'completion_tokens': r[1], 'decode_tok_s': r[2], 'prompt_tokens': r[3]} for r in results]}) + '\n')
    print(f"run {run} {kind} c{conc}: per-stream decode {' / '.join((f'{x:.2f}' for x in rates))} tok/s, aggregate {sum(rates):.2f} tok/s, ttft {' / '.join((f'{r[0]:.2f}' for r in results))} s, completion {results[0][1]} tokens, prompt {results[0][3]} tokens" + (f', prefill {results[0][3] / results[0][0]:.0f} tok/s' if kind.startswith('long') else ''))
    print('  ' + repr(results[0][4][:300]))
