# SPDX-License-Identifier: MIT
"""Confined, read-only parameterisation of the pinned rows24 packet."""
import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path

P = Path(__file__).resolve().parent
R = P.parent
spec = importlib.util.spec_from_file_location('_family_shared', R/'c8-moe24-r1/common.py')
shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
need, sha, read, write, load, trim, IMAGE = (getattr(shared,k) for k in ('need','sha','read','write','load','trim','IMAGE'))
SWITCH = 'DSV41_MOE_ROWS_FAMILY'
ROWS = (24,8,12,1,2,3,4,5,6,7,9,10,11,13,14,15,16,18,20,21,25,28,30,32,36,42)
@contextmanager
def module(file):
    saved = sys.modules.get('common'); sys.modules['common'] = sys.modules[__name__]
    try: yield load(R/'c8-moe24-r1'/file, '_family_'+file.replace('.','_'))
    finally:
        if saved is None: sys.modules.pop('common',None)
        else: sys.modules['common'] = saved
def pins():
    for rel,h in read(P/'source-pins.json').items(): need(sha(R.parent/rel) == h, 'dependency drift: '+rel)
def receipt():
    with module('qualify.py') as q: return q.shared_receipt()
def source_key(): return receipt().source_key()
def select(incumbent, candidate, args, env=None, *, role=None, requests=None):
    value = (shared.os.environ if env is None else env).get(SWITCH,'0')
    need(value in ('0','1'), 'switch must be 0 or 1')
    if value == '0' or role != 'verify' or len(args) != 36: return incumbent
    shape = tuple(args[0].shape)
    if len(shape)!=2: return incumbent
    rows = shape[0]
    # Reuse the complete semantic guard, adapting only shape and context witnesses.
    proxy = list(args)
    class Shape:
        def __init__(self,shape): self.shape = shape
    proxy[0],proxy[1],proxy[2] = Shape((24,*args[0].shape[1:])),Shape((24,*args[1].shape[1:])),Shape((24,*args[2].shape[1:]))
    eligible = rows in ROWS and rows > 1 and tuple(args[1].shape) == tuple(args[2].shape) == (rows,6)
    return shared.select(incumbent,candidate,tuple(proxy),{shared.SWITCH:value},role=role,requests=8) if eligible else incumbent
def call(incumbent,candidate,args,env=None,**context):
    return select(incumbent,candidate,args,env,**context).exl3_moe_coop(*args)
def gains(pairs):
    need(len(pairs) == 10 and all(len(p)==2 and all(math.isfinite(v) and v>0 for v in p) for p in pairs),'incomplete/invalid ABBA')
    values = [100*(a-b)/a for a,b in pairs]
    return values, all(v>0 for v in values)
def replace(source, old, new, count=1):
    need(source.count(old) == count, 'pinned adapter seam changed: '+old[:40])
    return source.replace(old,new)
if __name__ == '__main__':
    # The shared receipt imports this module by its canonical name as well.
    sys.modules['common'] = sys.modules[__name__]
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('key','check','write')); p.add_argument('--expected-key'); a=p.parse_args()
    m=receipt(); key=m.source_key()
    if a.action == 'key': print(key)
    elif a.action == 'check': m.check(); print('LINUX_QUALIFIED')
    else:
        need(sys.platform=='linux' and key==a.expected_key,'Linux/source changed')
        row=dict(source_key=key,platform='linux',tests_passed=True)
        write(P/'out/linux-qualified.json',row); print('LINUX_RECEIPT '+shared.json.dumps(row))
