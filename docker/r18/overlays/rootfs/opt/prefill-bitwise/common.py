# SPDX-License-Identifier: MIT
"""Packet policy; no torch, host queries, or import hooks."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent
ITEMS = (6, 7, '7b', 11, 12)
SWITCHES = {i: f'DSV41_BITWISE_PREFILL_{str(i).upper()}' for i in ITEMS}
IMAGE = 'sha256:f760136967d6c4cac9f4de317e2b9a19a170d27519ddeb0ba9aad3e420ec3c27'


def enabled(env, item):
    value = env.get(SWITCHES[item], '0')
    if value not in ('0', '1'):
        raise ValueError(f'{SWITCHES[item]} must be 0 or 1')
    return value == '1'


def eligible(item, rows, tc=None, prefill=False):
    if item == 6:
        return bool(tc and tc.get('v2') and 43 <= rows <= 64)
    if item in (7, '7b'):
        # Do not take over >cap subchunking or any coop/decode path.
        minimum = 128 if item == '7b' else 65
        return bool(tc and tc.get('prefill') and prefill and minimum <= rows <= min(tc['cap'], 2048))
    if item == 12:
        return prefill and 65 <= rows <= 2048
    return False


def gains(samples):
    """Ten adjacent pairs in five ABBA quartets; every pair must win."""
    if len(samples) != 10:
        raise ValueError('exactly five ABBA quartets (ten pairs) required')
    result = []
    for a, b in samples:
        if not all(math.isfinite(x) and x > 0 for x in (a, b)):
            raise ValueError('invalid event duration')
        result.append(100 * (a - b) / a)
    return result, all(x >= 1 for x in result)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_pins():
    pins = json.loads((HERE / 'source-pins.json').read_text())
    for rel, want in pins.items():
        path = ROUNDS / rel
        if digest(path) != want:
            raise RuntimeError(f'source changed: {rel}; explicit requalification required')
    return pins


def verify_capture(capture, item):
    if capture.get('schema') != 'bitwise-prefill-capture/1' or capture.get('item') != (7 if item == '7b' else item):
        raise ValueError('wrong capture schema/item')
    if capture.get('image') != IMAGE or capture.get('incumbent') != 'current-r7a':
        raise ValueError('capture must be from current-r7a/f760')
    if not capture.get('cases'):
        raise ValueError('no real cases captured')
    if any(not c.get('origin') for c in capture['cases']):
        raise ValueError('missing real-input origin')


if __name__ == '__main__':
    verify_pins()
    print('PREFILL_PINS_OK')
