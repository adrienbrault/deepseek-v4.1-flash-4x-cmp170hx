#!/usr/bin/env python3
"""Run the pinned MiaAI prose protocol used by GNmac on an already loaded model.

--stage downloads the AGPL-3.0 upstream benchmark, verifies its git blob, and
caches it separately. Normal runs never download or change the upstream script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import urllib.request

REPO = 'MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks'
COMMIT = '6278ecb'
BLOB = 'd9b24cf29a56c6e215144b267bc874e428b9dcb4'
URL = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/tests/bench_decode.py'


def verify(data):
    actual = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
    if actual != BLOB:
        raise ValueError(f'benchmark source mismatch: {actual}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    source = Path(__file__).resolve().parent / 'mia-cache' / 'bench_decode.py'
    if args.stage:
        with urllib.request.urlopen(URL, timeout=30) as response:
            data = response.read(256 * 1024)
        verify(data)
        source.parent.mkdir(exist_ok=True)
        temporary = source.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(source)
        print(f'STAGED {BLOB}')
        return
    if args.output is None:
        ap.error('--output is required for a run')
    verify(source.read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'repo': REPO, 'commit': COMMIT, 'git_blob': BLOB,
                  'url': URL, 'licence': 'AGPL-3.0',
                  'protocol': 'prose, c1, temperature0, thinking off, 5x400-token cap',
                  'comparison': 'GNmac protocol; different hardware, no claim of equal conditions'}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    benchmark = runpy.run_path(str(source), run_name='mia_benchmark')
    globals_ = benchmark['main'].__globals__
    globals_['BASE'] = 'http://127.0.0.1:8001'
    globals_['MODEL'] = 'deepseek-v4.1-flash-1017'
    sys.argv = ['bench_decode.py', '--phase', 'c1-mia-prose',
                '--out', str(args.output / 'prose.json'), '--runs', '5',
                '--max-tokens', '400', '--concurrency', '1', '--skip-coherence']
    rc = benchmark['main']()
    result = json.loads((args.output / 'prose.json').read_text())
    if rc or result.get('any_nan') or result.get('health_code_after') != 200:
        raise RuntimeError('MiaAI prose probe failed; retain output for diagnosis')


if __name__ == '__main__':
    main()
