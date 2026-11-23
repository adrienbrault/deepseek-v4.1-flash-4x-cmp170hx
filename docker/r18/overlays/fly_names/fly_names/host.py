# SPDX-License-Identifier: MIT
"""One model boot using QV's composer, controls, audit and terminal path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .common import HERE, ROUNDS, ablit, load, need, read, sibling, sources, write
from .planner import check, compose


def harness():
    # Rebind only output journals in this process. Sibling files remain read-only.
    # QV's standalone CLI is deliberately not called: it writes its own out/.
    sys.path.insert(0, str(ROUNDS/'quick-verify-r1'))
    import qv_host, qv_workload
    qv_host.write = write
    qv_workload.write = write
    return qv_host, qv_workload


def hardware(run, stage, label):
    need(run.resolve().is_relative_to(HERE/'out'),'run outside packet/out')
    host, _ = harness()
    host.hardware(run, stage, label)


def run(prepared, run, token, name):
    from .workload import control, diagnostic, hard_gate, quality, speed
    need(run.resolve().is_relative_to(HERE/'out'),'run outside packet/out')
    cfg = check(prepared)
    host,q = harness()
    p = host.protocol(config=cfg)
    (run/'live/force').mkdir(parents=True,mode=0o700)
    session = run.name
    write(run/'live/control.json',dict(session=session,generation=0,arm='off',effective='off'))
    control(run, 1, 'bootstrap')
    final, paths = compose(prepared,run,token,name,session)
    write(run/'argv-private.json',final); write(run/'runtime-paths.json',paths)
    write(run/'registration.json',dict(config=cfg,sources=sources(),budget_s=1190,
          baseline='current-r7a native standard OFF',measurement='packet FLy workload over unchanged QV APIs',
          gate='OFF diagnostic first; barrier + forced K3 identity before timing; ABBA 2 pairs, >=1% both pairs'))
    cid = host.command(['docker',*final],timeout=60).strip()  # exactly one run; no retry/replacement
    actual = host.inspect(cid)
    plan = dict(name=name,container_id=actual['Id'],container_started=actual['State']['StartedAt'])
    write(run/'plan.json',plan)
    deadline = time.monotonic()+300
    while True:
        host.lifetime(plan)
        try:
            import urllib.request
            with urllib.request.urlopen(q.BASE+'/v1/models',timeout=2) as response:
                models = json.load(response)
            need({r['id'] for r in models['data']} == {ablit().RAW,ablit().ADAPT,ablit().RAW+'-fast',ablit().ADAPT+'-fast'},
                 'four exact served aliases required')
            break
        except (OSError,ValueError):
            need(time.monotonic()<deadline,'boot readiness deadline')
            time.sleep(1)
    write(run/'models.json',models)
    (run/'container-id.txt').write_text(actual['Id']+'\n')
    (run/'container-started.txt').write_text(actual['State']['StartedAt']+'\n')
    pin = load(ROUNDS.parent/'serve-current/pin_workers.py','_fly_pin_workers')
    pin.apply(run)  # serving's own mechanism; journal only in this run
    receipt = read(run/'worker-pinning.json')
    observed = host.command(['nvidia-smi','--query-gpu=uuid,serial','--format=csv,noheader,nounits'])
    serial_by_uuid = {u.strip():s.strip() for u,s in (line.split(',') for line in observed.splitlines())}
    serials = [serial_by_uuid[receipt['workers'][str(rank)]['gpu_uuid']] for rank in range(4)]
    need(serials == read(prepared/'prepared.json')['serials'],'TP card order changed from capture receipt')
    host.runtime_audit(run,plan)
    generation, worker_pids = 0,None
    def switch(arm):
        nonlocal generation,worker_pids
        generation += 1
        host.lifetime(plan)
        control(run, 1, f'switch-{generation}')
        acks = host.switch(run,dict(generation=generation,arm=arm,effective=arm),cfg,session,p)
        pids = {r['who']:r['pid'] for r in acks}
        need(worker_pids is None or pids==worker_pids,'worker lifetime changed')
        worker_pids=pids
        # QV warms K3 for its switch receipt. Timing uses adaptive width;
        # the correctness workload explicitly forces K3 after this boundary.
        q.force(run,None)
    try:
        diagnosed = diagnostic(run,q,switch,ablit().RAW)
    except Exception as error:
        if not (run/'diagnostic.json').exists():
            write(run/'diagnostic.json',dict(status='INCOMPLETE',error=str(error),timing_permitted=False))
        raise
    identity = hard_gate(run,q,switch,ablit().RAW,ablit().ADAPT)
    need(identity['status']=='PASS','identity gate did not pass')
    measured = speed(run,q,switch,ablit().RAW,ablit().ADAPT)
    quality(run,q,switch,ablit().RAW,ablit().ADAPT)
    q.flush(run); host.runtime_audit(run,plan); pin.require(run)
    check(prepared)
    log=host.command(['docker','logs',name],timeout=20)
    (run/'boot.log').write_text(log)
    need('FLY_NAMES_INSTALLED' in log,'FLy activation missing')
    host.finish(run)
    write(run/'workload.json',dict(status='COMPLETE',diagnostic=diagnosed,identity=identity,speed=measured,promotion_eligible=False))


def seal(run,success):
    elapsed=time.time()-float((run/'started').read_text())
    complete=success and elapsed<=1190 and (run/'workload.json').exists() and (run/'cards-post.json').exists() and (run/'hardware-deltas.json').exists()
    report=dict(status='PASS' if complete else 'INCOMPLETE',elapsed_s=elapsed,limit_s=1190,
                serving_after=False,teardown_confirmed=(run/'cards-post.json').exists(),sources=sources())
    write(run/'report.json',report)
    if complete:
        for cell in read(run/'speed.json'):
            print('FLY_NAMES_VERDICT',cell['name'],f'c{cell["c"]}',cell['verdict'],cell['pair_gains_pct'])
    else:
        print('FLY_NAMES_VERDICT INCOMPLETE identity/timing not qualified')
    need(complete,'run incomplete, failed gate or exceeded 20-minute budget')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    a=sub.add_parser('run')
    for name in ('prepared','run'):a.add_argument(name,type=Path)
    for name in ('token','name'):a.add_argument(name)
    a=sub.add_parser('hardware');a.add_argument('run',type=Path);a.add_argument('stage');a.add_argument('label')
    a=sub.add_parser('seal');a.add_argument('run',type=Path);a.add_argument('success',type=int)
    a=parser.parse_args()
    if a.action=='run':run(a.prepared,a.run,a.token,a.name)
    elif a.action=='hardware':hardware(a.run,a.stage,a.label)
    else:seal(a.run,bool(a.success))


if __name__=='__main__':main()
