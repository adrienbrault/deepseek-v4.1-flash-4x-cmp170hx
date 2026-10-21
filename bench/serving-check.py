# SPDX-License-Identifier: MIT
import base64
import json
import os
import random
import re
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path
BASE = 'http://127.0.0.1:8001/v1/'
MODEL = 'deepseek-v4.1-flash-1017'

def ask(messages, limit):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': limit, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}}
    req = urllib.request.Request(BASE + 'chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        r = json.load(response)
    return (time.monotonic() - started, r['choices'][0]['message']['content'], r['usage'])
nonce = random.randrange(10 ** 9)
doc = '\n'.join((f'Record {i} (batch {nonce}): sensor {i % 17} read {i * 7919 % 1000} at minute {i}.' for i in range(600)))
messages = [{'role': 'user', 'content': doc + '\n\nWhich sensor read the highest value at minute 599? Answer briefly.'}]
cold, answer, usage = ask(messages, 160)
warm, cached_answer, warm_usage = ask(messages, 160)
cached = (warm_usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
result = {'prefix': {'cold_s': round(cold, 3), 'warm_s': round(warm, 3), 'prompt_tokens': usage['prompt_tokens'], 'cached_tokens': cached, 'same_text': answer == cached_answer, 'answer': answer, 'cached_answer': cached_answer}}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
for text in (answer, cached_answer):
    assert re.search('\\bsensor\\s*4\\b', text, re.I) and re.search('\\b481\\b', text), result
assert warm < cold * 0.5, result
result['prefix']['cache_reporting_pass'] = cached > 0.8 * usage['prompt_tokens']
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
if not result['prefix']['cache_reporting_pass']:
    assert os.environ.get('SCREEN_ALLOW_CACHE_REPORTING_DEFECT') == '1', result
    print('CACHE_REPORTING_FAIL: timing confirms reuse; performance screen only', flush=True)

def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 4294967295)
raw = b''.join((b'\x00' + b'\xff\x00\x00' * 64 for _ in range(64)))
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
image_url = 'data:image/png;base64,' + base64.b64encode(png).decode()
_, answer, _ = ask([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': 'What color is this image? One word.'}]}], 20)
result['vision'] = {'answer': answer, 'pass': 'red' in answer.lower()}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
assert result['vision']['pass'], result
print(json.dumps(result), flush=True)
