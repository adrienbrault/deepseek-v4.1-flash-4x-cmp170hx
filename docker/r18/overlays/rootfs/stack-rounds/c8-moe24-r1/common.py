# SPDX-License-Identifier: MIT
"""Read-only shared imports and a deliberately narrow native-call selector."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
R = P.parent
IMAGE = 'sha256:f760136967d6c4cac9f4de317e2b9a19a170d27519ddeb0ba9aad3e420ec3c27'
SWITCH = 'DSV41_FIXED640_ROWS24'
def need(ok, why):
    if not ok: raise ValueError(why)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, obj): Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m
@contextmanager
def trim():
    names = ('packet', 'captures'); saved = {n: sys.modules.pop(n, None) for n in names}
    paths = sys.path[:]
    try:
        sys.path.insert(0, str(R/'moe-trim-r1'))
        t = load(R/'moe-trim-r1/packet.py', 'packet'); sys.modules['packet'] = t
        yield t, load(R/'moe-trim-r1/build.py', '_rows24_builder'), load(R/'moe-trim-r1/bench.py', '_rows24_bench')
    finally:
        sys.path[:] = paths
        for n, m in saved.items():
            sys.modules.pop(n, None)
            if m is not None: sys.modules[n] = m
def pins():
    for rel, value in read(P/'source-pins.json').items(): need(sha(R/rel) == value, 'dependency drift: '+rel)
    with trim() as (t, _, __): t.pins()
def select(incumbent, sibling, args, env=None, *, role=None, requests=None):
    value = (os.environ if env is None else env).get(SWITCH, '0')
    need(value in ('0', '1'), SWITCH+' must be 0 or 1')
    if value == '0' or role != 'verify' or requests != 8: return incumbent
    # Role/request witnesses come from the caller, never inferred from padding.
    # Even a c4 fill-in drafter with 24 rows must retain the original module.
    eligible = (len(args) == 36 and tuple(args[0].shape) == (24,5120) and args[5] == 5120 and
                tuple(args[1].shape) == tuple(args[2].shape) == (24,6) and
                args[18:24] == (3,3,3,False,True,0) and args[25] is True and
                str(args[28].dtype) == 'torch.float16' and args[29].shape[-1] in (512,640) and
                args[31].shape[-1] == args[33].shape[-1] == 5120)
    return sibling if value == '1' and role == 'verify' and requests == 8 and eligible else incumbent
def call(incumbent, sibling, args, env=None, **context):
    return select(incumbent, sibling, args, env, **context).exl3_moe_coop(*args)
def source_key():
    pins()
    return hashlib.sha256(json.dumps({str(f.relative_to(P)): sha(f) for f in sorted(P.rglob('*'))
        if f.is_file() and not {'out', '__pycache__'} & set(f.relative_to(P).parts)}, sort_keys=True).encode()).hexdigest()
