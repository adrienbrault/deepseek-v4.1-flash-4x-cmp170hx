#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify exported source/patch hashes and syntax without importing the runtime."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for name in ('docs/SOURCE-INVENTORY.json', 'patches/manifest.json'):
        for row in json.loads((ROOT / name).read_text()):
            f = (ROOT / row['file']).resolve()
            if not f.is_relative_to(ROOT):
                raise ValueError('source path escapes package')
            if hashlib.sha256(f.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('source digest mismatch: ' + row['file'])
            count += 1
    files = list((ROOT / 'docker/overlays').rglob('*.py'))
    files += [ROOT / 'scripts/serve.py', ROOT / 'bench/plot.py']
    for f in files:
        ast.parse(f.read_text(), filename=str(f.relative_to(ROOT)))
    print(f'Source package: {count} hash records and {len(files)} Python modules PASS')
    print('Image build, patch application, native compilation and GPU execution: NOT RUN')


if __name__ == '__main__':
    main()
