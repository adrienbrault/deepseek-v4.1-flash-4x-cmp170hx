# SPDX-License-Identifier: MIT
import json
import sys
import urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else '8001'
BASE = f'http://127.0.0.1:{PORT}/v1/'

def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]

def repeated(text, n=4):
    w = text.split()
    grams = [' '.join(w[i:i + 4]) for i in range(len(w) - 3)]
    return any((grams.count(g) >= n for g in set(grams)))
MODEL = json.load(urllib.request.urlopen(BASE + 'models'))['data'][0]['id']
RAW = [('raw-paris', 'The capital of France is', lambda t: 'Paris' in t), ('raw-apollo', 'In 1969, Apollo 11', lambda t: any((k in t for k in ('Armstrong', 'moon', 'Moon', 'lunar')))), ('raw-fib', 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n', lambda t: 'fibonacci(n - 1)' in t.replace('n-1', 'n - 1') or 'a, b = b, a + b' in t or 'a, b = b, a+b' in t)]
CHAT = [('chat-mult', 'What is 17*23? Answer with the number only.', lambda t: '391' in t), ('chat-canberra', 'Name the capital of Australia in one word.', lambda t: 'Canberra' in t), ('chat-reverse', 'Reverse the letters of the word ELEPHANT. Answer with the reversed word only.', lambda t: 'TNAHPELE' in t.upper())]
fails = 0
for name, prompt, ok in RAW:
    text = post('completions', {'model': MODEL, 'prompt': prompt, 'max_tokens': 64, 'temperature': 0})['text']
    good = ok(text) and (not repeated(text))
    fails += not good
    print(json.dumps({'case': name, 'pass': good, 'text': text[:200]}), flush=True)
for name, prompt, ok in CHAT:
    msg = post('chat/completions', {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 1024, 'temperature': 0})['message']
    text = msg.get('content') or ''
    good = ok(text) and (not repeated(text))
    fails += not good
    print(json.dumps({'case': name, 'pass': good, 'text': text[:200]}), flush=True)
print(json.dumps({'quality': 'PASS' if not fails else 'FAIL', 'failed': fails}), flush=True)
sys.exit(1 if fails else 0)
