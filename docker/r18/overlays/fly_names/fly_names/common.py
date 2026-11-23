# SPDX-License-Identifier: MIT
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
ROUNDS = HERE.parent


def need(ok, why):
    if not ok:
        raise ValueError('fly-names: ' + why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path).resolve()
    need(path.is_relative_to(HERE / 'out'), 'output outside packet/out')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.new')
    with open(tmp, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    tmp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


@contextlib.contextmanager
def sibling(packet, file, bare=()):
    saved, paths = {k: sys.modules.get(k) for k in bare}, list(sys.path)
    try:
        sys.path.insert(0, str(ROUNDS / packet))
        for key in bare:
            sys.modules.pop(key, None)
        yield load(ROUNDS / packet / file, '_fly_' + packet.replace('-', '_') + '_' + Path(file).stem)
    finally:
        sys.path[:] = paths
        for key, obj in saved.items():
            sys.modules.pop(key, None)
            if obj is not None:
                sys.modules[key] = obj


def sources():
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and 'out' not in p.relative_to(HERE).parts
            and '__pycache__' not in p.parts and p.suffix in ('.py', '.sh', '.json')}


def pins():
    for name, expected in read(HERE / 'source-pins.json').items():
        path = (ROUNDS / name).resolve()
        need(path.is_relative_to(ROUNDS.parent) and sha(path) == expected, 'source drift: ' + name)


def ablit():
    sys.path.insert(0, str(ROUNDS / 'ablit-r7-r1'))
    from ablit_r7 import planner
    return planner
