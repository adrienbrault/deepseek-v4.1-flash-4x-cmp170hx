# SPDX-License-Identifier: MIT
"""Packet-local I/O and isolated imports. Importing opens no socket/device."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
ROUNDS = P.parent
IMAGE = 'sha256:f760136967d6c4cac9f4de317e2b9a19a170d27519ddeb0ba9aad3e420ec3c27'
TORCH_GIT = '08187d9e0fba026dc8217405802ab5381dc88d90'
ROWS = (3, 4, 6, 8, 9, 12, 15, 16, 18, 21, 24)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path).resolve()
    need(path.is_relative_to(P/'out'), 'packet output confinement')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def source_map():
    return {str(f.relative_to(P)): sha(f) for f in sorted(P.rglob('*')) if f.is_file()
            and not {'out', '__pycache__'} & set(f.relative_to(P).parts)}


def pins():
    deps = read(P/'dependency-pins.json')
    need(all(sha(ROOT/p) == h for p, h in deps.items()), 'pinned dependency drift')
    return digest(dict(own=source_map(), dependencies=deps))


def source_key():
    return pins()


@contextmanager
def module(packet, name):
    # Siblings intentionally have conflicting bare imports. Preserve their
    # identities, including this packet's common module, across every import.
    saved, paths = dict(sys.modules), sys.path[:]
    directory = ROUNDS/packet
    for key in list(sys.modules):
        if (directory/(key+'.py')).is_file():
            sys.modules.pop(key)
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location('_live2_'+name.removesuffix('.py'), directory/name)
        obj = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = obj
        spec.loader.exec_module(obj)
        yield obj
    finally:
        sys.path[:] = paths
        for key in set(sys.modules)-set(saved):
            sys.modules.pop(key, None)
        sys.modules.update(saved)


def registration():
    return read(P/'registration.json')
