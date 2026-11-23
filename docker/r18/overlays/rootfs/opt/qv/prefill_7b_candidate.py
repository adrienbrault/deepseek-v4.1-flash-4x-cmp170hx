# SPDX-License-Identifier: MIT
"""Extend the qualified eager-only 7b adapter and r7d's identity owner."""
import importlib.util
import json
import os
from pathlib import Path
import stack_identity_owner as identity
import borrowed_workspace as borrowed

owner=None
counts={'off':0,'on':0}
rows={}
select_scheduler=identity.select_scheduler
audit_scheduler=identity.audit_scheduler

def enabled():
    value=os.environ.get('DSV41_PREFILL_7B_R7D','0')
    if value not in ('0','1'): raise ValueError('7b boot switch must be 0/1')
    return value=='1'

def install():
    global owner
    enabled()
    s=importlib.util.spec_from_file_location('_r7d_prefill_owner','/opt/qv/prefill_7b_owner.py')
    owner=importlib.util.module_from_spec(s);s.loader.exec_module(owner)
    owner.admission=lambda:identity
    owner.bind=lambda runner:borrowed.bind(runner,owner)
    # Reuses its pre-capture binding, 48-row graph fence and forward-context
    # guard. Native candidate switches in that helper remain default OFF;
    # this packet alone selects the fixed per-boot arm on the QV worker seam.
    owner.install()
    import dsv41_qv as qv
    original=qv.wrap_worker
    def wrap(cls):
        original(cls)
        if not getattr(cls.profile_run,'_7b_borrowed',False):
            cls.profile_run=borrowed.wrap_profile(cls.profile_run)
    qv.wrap_worker=wrap
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    capture,replay=CudaGraphManager.capture,CudaGraphManager.run_fullgraph
    def capturing(manager,*args,**kwargs):
        layer=owner.binding.layers[0]
        with layer.capture_scope(manager):return capture(manager,*args,**kwargs)
    def replayed(manager,*args,**kwargs):
        layer=owner.binding.layers[0]
        layer.check_stream()
        borrowed.require(layer.lock.acquire(blocking=False),'overlapping graph replay')
        try:return replay(manager,*args,**kwargs)
        finally:layer.lock.release()
    CudaGraphManager.capture,CudaGraphManager.run_fullgraph=capturing,replayed

def select_worker(runner,arm):
    identity.select_worker(runner,arm)  # QV admission channel stays OFF
    if not getattr(runner,'_7b_graphs_proven',False):
        owner.select_worker(runner,'on' if enabled() else 'off')
        proof=runner._prefill_runtime_proof
        draft=getattr(getattr(runner,'speculator',None),'query_cudagraph_manager',None)
        # Initialization creates/drops managers during memory profiling. Cache
        # only the completed real graph inventory; capture/dispatch/replay
        # fences remain installed. Avoid a model/module scan and redundant
        # control-file read on every timed decode step.
        runner._7b_graphs_proven=(proof.get('captured',0)>0 and draft is not None and bool(draft.graphs))
    else:
        owner.binding.select(enabled(),prefill=False)
        owner.events.clear()
    if getattr(runner,'_7b_profiling',False):
        # profile_run forces eager in DP dispatch and bypasses manager.dispatch.
        # Supply the exact metadata that real >=128-row eager dispatch supplies.
        owner.binding.select(enabled(),prefill=True)
        for layer in owner.binding.layers:
            layer.profiling=True
            layer.execution_mode='NONE'

def audit_worker(runner,arm):
    value=identity.audit_worker(runner,arm)
    if getattr(runner,'_7b_profiling',False):
        # Profiling proves the allocation/native ABI, never scored exposure.
        owner.events.clear()
        return value
    events=owner.events
    if events:
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        torch.cuda.synchronize()  # completed eager launches, not selection ACKs
        for e in events:
            if e['execution']!='eager' or not 128<=e['rows']<=2048:
                raise ValueError('7b executed outside registered eager domain')
            counts[e['arm']]+=1
            key=str(e['rows']);rows[key]=rows.get(key,0)+1
        from dsv41_moe_stack import stack_plugin
        b=stack_plugin.binding
        receipt=dict(pid=os.getpid(),on=enabled(),calls=dict(counts),rows=dict(rows),
                     graph=runner._prefill_runtime_proof,
                     stack=dict(installed=b is not None,targets=len(stack_plugin.targets),
                                capture_calls=0 if b is None else b.calls,
                                by_kind={} if b is None else b.by_kind))
        receipt['workspace']=dict(borrowed=True,new_persistent_bytes=0,
                                 native_identity=sorted(owner.binding.layers[0].verified))
        path=Path('/qv')/f'prefill-7b-rank{get_tensor_model_parallel_rank()}.json'
        tmp=path.with_suffix('.new');tmp.write_text(json.dumps(receipt));tmp.replace(path)
    return value
