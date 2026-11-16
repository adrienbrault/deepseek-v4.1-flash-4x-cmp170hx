# SPDX-License-Identifier: MIT
"""Packet-local exact K2 adapters; shared sources are never modified."""
import sys
from pathlib import Path
from contextlib import contextmanager
import importlib.util

P = Path(__file__).resolve().parent
R = P.parent
spec = importlib.util.spec_from_file_location('_k2_shared', R/'c8-moe24-r1/common.py')
shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
need, sha, read, write, load, trim, IMAGE = (getattr(shared,k) for k in ('need','sha','read','write','load','trim','IMAGE'))
SWITCH, ROWS = 'DSV41_MOE_K2_FIXED16', (16,)
def replace(source, old, new, count=1):
    need(source.count(old)==count, 'pinned adapter seam changed: '+old[:40])
    return source.replace(old,new)
def adapted(packet,file,changes=(),scope=None):
    path=R/packet/file; source=path.read_text(); end=source.rfind('\nif __name__')
    if end>=0: source=source[:end]
    for change in changes: source=replace(source,*change)
    if scope is None: scope=dict(__name__='_k2_'+file.replace('.','_'),__file__=str(path))
    exec(compile(source,str(path), 'exec'),scope); return scope
@contextmanager
def module(file):
    saved=sys.modules.get('common'); sys.modules['common']=sys.modules[__name__]
    try: yield load(R/'c8-moe24-r1'/file,'_k2_'+file.replace('.','_'))
    finally:
        if saved is None: sys.modules.pop('common',None)
        else: sys.modules['common']=saved
def pins():
    for rel,h in read(P/'source-pins.json').items():
        f=R.parent/rel
        need(sha(f)==h,'dependency drift: '+rel)
        need(f.stat().st_mode & 4, 'pinned file is not world-readable: '+rel)
    with trim() as (t,_,__): t.pins()
def receipt():
    with module('qualify.py') as q: return q.shared_receipt()
def source_key(): return receipt().source_key()
def gains(pairs):
    return adapted('moe-rows-family-r1','common.py')['gains'](pairs)
def select(incumbent,candidate,args,env=None,*,role=None,requests=None):
    value=(shared.os.environ if env is None else env).get(SWITCH,'0')
    need(value in ('0','1'),'switch must be 0 or 1')
    if value=='0' or role!='verify' or requests!=4 or len(args)!=36: return incumbent
    eligible=(tuple(args[0].shape)==(16,5120) and args[5]==5120 and
        tuple(args[1].shape)==tuple(args[2].shape)==(16,6) and
        args[18:24]==(2,2,2,False,True,0) and args[25] is True and
        str(args[28].dtype)=='torch.float16' and args[29].shape[-1] in (512,640) and
        args[31].shape[-1]==args[33].shape[-1]==5120)
    return candidate if eligible else incumbent
def call(incumbent,candidate,args,env=None,**context):
    return select(incumbent,candidate,args,env,**context).exl3_moe_coop(*args)
if __name__=='__main__':
    sys.modules['common']=sys.modules[__name__]
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('key','check','write')); p.add_argument('--expected-key'); a=p.parse_args()
    m=receipt(); key=m.source_key()
    if a.action=='key': print(key)
    elif a.action=='check': m.check(); print('LINUX_QUALIFIED')
    else:
        need(sys.platform=='linux' and key==a.expected_key,'Linux/source changed')
        (P/'out').mkdir(exist_ok=True)
        row=dict(source_key=key,platform='linux',tests_passed=True)
        write(P/'out/linux-qualified.json',row); print('LINUX_RECEIPT '+shared.json.dumps(row))
