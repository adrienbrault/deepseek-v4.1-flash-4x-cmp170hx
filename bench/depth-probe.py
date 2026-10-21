# SPDX-License-Identifier: MIT
import json, os, pathlib, sys, time, urllib.request
port, model, src, tokens, nq, out = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
chars = tokens * 3
parts, n = ([], 0)
for p in sorted(pathlib.Path(src).rglob('*')):
    if p.suffix in ('.py', '.sh', '.patch', '.md') and p.is_file():
        t = p.read_text(errors='replace')
        parts.append(f'===== {p.relative_to(src)} =====\n{t}\n')
        n += len(t)
        if n >= chars:
            break
doc = ''.join(parts)[:chars]
if 'DEPTH_PROBE_DOC' in os.environ:
    doc = pathlib.Path(os.environ['DEPTH_PROBE_DOC']).read_text()[:chars]
QUESTIONS = ['Write a new Python module, with tests, that validates every JSON manifest these tools read and reports each problem with file and key. Think it through, then give the full code.', 'Find the three most fragile pieces of shell in the code above and rewrite each robustly. Explain briefly, then give the full rewritten functions.', 'Design and implement a small CLI that runs any of these gate tools with a timeout and writes a JSON receipt. Give the complete code.', 'Write an HTML page with inline JavaScript that renders a voxel pagoda in a garden with three.js-like logic you write yourself (no libraries). Give the full file.']
url = f'http://127.0.0.1:{port}/v1/chat/completions'
with open(out, 'a') as f:
    for i in range(nq):
        body = {'model': model, 'max_tokens': 1500, 'stream': True, 'stream_options': {'include_usage': True}, 'reasoning_effort': 'high', 'messages': [{'role': 'system', 'content': 'You are a coding agent. The repository follows.'}, {'role': 'user', 'content': doc + '\n\n' + QUESTIONS[i % len(QUESTIONS)]}]}
        if 'DEPTH_PROBE_SEED' in os.environ:
            body['seed'] = int(os.environ['DEPTH_PROBE_SEED'])
        req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        t0 = time.time()
        first = last = None
        usage = {}
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b'data:') or line == b'data: [DONE]':
                    continue
                d = json.loads(line[5:])
                if d.get('usage'):
                    usage = d['usage']
                ch = d.get('choices') or []
                if ch and ((ch[0].get('delta') or {}).get('content') or (ch[0].get('delta') or {}).get('reasoning_content') or (ch[0].get('delta') or {}).get('reasoning')):
                    now = time.time()
                    first = first or now
                    last = now
        ct = usage.get('completion_tokens', 0)
        rec = {'q': i, 'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': ct, 'ttft_s': round(first - t0, 2) if first else None, 'decode_tok_s': round((ct - 1) / (last - first), 1) if first and last and (last > first) else None}
        print(json.dumps(rec), flush=True)
        f.write(json.dumps(rec) + '\n')
