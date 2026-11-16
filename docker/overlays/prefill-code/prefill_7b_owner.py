# SPDX-License-Identifier: MIT
"""7b alone. Extend QV's owning class wrapper; no additional import finder."""
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys

from graph_policy import manager_proof, mode_name, require, require_decode_only, runner_proof

ROOT = Path('/qv')
binding = None
active = None
events = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def admission():
    global _admission
    if '_admission' not in globals():
        _admission = load('/opt/qv/r7b_identity.py', '_prefill_admission')
    return _admission


def guard_forward():
    import torch
    from vllm.forward_context import get_forward_context
    require(not torch.compiler.is_compiling(), '7b called by torch compiler')
    require(mode_name(get_forward_context().cudagraph_runtime_mode) == 'NONE',
            '7b called in graph forward context')


def bind(runner):
    global binding
    if binding is not None:
        require(getattr(runner, '_prefill_7b', None) is binding, 'second runner/hook owner')
        return
    require(all(os.environ.get('DSV41_BITWISE_PREFILL_' + k, '0') == '0'
                for k in ('6', '7', '7B', '11', '12')), 'all candidate env switches must default OFF')
    # Isolate the sibling's bare common/layers imports from other plugins.
    names = ('common', 'layers', 'prefetch')
    saved = {n: sys.modules.pop(n, None) for n in names}
    paths = sys.path[:]
    try:
        sys.path.insert(0, '/opt/prefill-bitwise')
        runtime = load('/opt/prefill-bitwise/runtime.py', '_prefill_binding')
        binding = runtime.Binding(runner, '7b')
    finally:
        sys.path[:] = paths
        for n, old in saved.items():
            sys.modules.pop(n, None)
            if old is not None:
                sys.modules[n] = old
    runner._prefill_7b = binding
    layer = binding.layers[0]
    layer.execution_guard = guard_forward
    layer.witness = lambda arm, rows: events.append(dict(arm=arm, rows=rows, execution='eager'))
    print('PREFILL_7B_BOUND before_capture=1 item6=BLOCKED', flush=True)


def install():
    import dsv41_qv as qv
    original = qv.wrap_worker

    def wrap(cls):
        original(cls)
        if getattr(cls.load_model, '_prefill_7b', False):
            return
        old_load, old_capture = cls.load_model, cls.capture_model

        @functools.wraps(old_load)
        def loaded(self, *args, **kwargs):
            result = old_load(self, *args, **kwargs)
            bind(self)  # r7a and native r7b have finished; before graph warmup
            return result

        @functools.wraps(old_capture)
        def captured(self, *args, **kwargs):
            require(binding is not None, 'hook not bound before capture')
            profile_only = bool(kwargs.get('profile_only', False))
            runner_proof(self, graphs_required=not profile_only)
            result = old_capture(self, *args, **kwargs)
            if not profile_only:
                runner_proof(self)  # drafter manager now exists and has captured
            return result
        loaded._prefill_7b = True
        cls.load_model, cls.capture_model = loaded, captured
    qv.wrap_worker = wrap

    # All target/drafter managers inherit these methods. FULL replay bypasses
    # Python ops, so fence its descriptor BEFORE replay as well as the hook.
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    dispatch, replay, capture = (CudaGraphManager.dispatch, CudaGraphManager.run_fullgraph,
                                 CudaGraphManager.capture)

    @functools.wraps(capture)
    def capturing(self, *args, **kwargs):
        manager_proof(self)
        return capture(self, *args, **kwargs)

    @functools.wraps(dispatch)
    def dispatched(self, num_reqs, num_tokens, *args, **kwargs):
        require_decode_only(self)
        desc = dispatch(self, num_reqs, num_tokens, *args, **kwargs)
        require(mode_name(desc.cg_mode) in ('NONE', 'FULL'), 'non-decode dispatch mode')
        require(mode_name(desc.cg_mode) == 'NONE' or desc.num_tokens <= 48,
                'graph dispatch exceeds decode ceiling')
        require(desc.num_tokens < 128 or mode_name(desc.cg_mode) == 'NONE',
                '128+ rows selected a graph')
        if binding is not None:
            for layer in binding.layers:
                layer.prefill = num_tokens >= 128
                layer.execution_mode = mode_name(desc.cg_mode)
        return desc

    @functools.wraps(replay)
    def replayed(self, desc, *args, **kwargs):
        require_decode_only(self)
        require(0 < desc.num_tokens <= 48, 'over-ceiling graph replay forbidden')
        require(mode_name(desc.cg_mode) == 'FULL' and
                (desc.uniform_token_count is not None or desc.max_query_len is not None),
                'non-decode graph replay forbidden')
        return replay(self, desc, *args, **kwargs)
    CudaGraphManager.dispatch, CudaGraphManager.run_fullgraph = dispatched, replayed
    CudaGraphManager.capture = capturing


def select_scheduler(scheduler, arm):
    admission().select_scheduler(scheduler, arm)


def audit_scheduler(scheduler, arm):
    admission().audit_scheduler(scheduler, arm)
    return '7b:scheduler'


def select_worker(runner, arm):
    global active
    require(binding is not None, 'pre-capture binding was bypassed')
    # QV selects on every execute_model, including profile_run's dummy forward, which
    # runs before initialize_kv_cache creates the graph managers (11-14 boots of 13:30
    # and 13:42 Paris). Scored rows must still carry both proofs: prefill_metrics.executed.
    ready = (runner.cudagraph_manager is not None and runner.speculator is not None and
             runner.speculator.query_cudagraph_manager is not None)
    runner._prefill_runtime_proof = runner_proof(runner, graphs_required=ready)
    active = json.loads((ROOT / 'control.json').read_text())
    binding.select(arm == 'on', prefill=False)
    events.clear()


def audit_worker(runner, arm):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    # Complete the eager work before publishing an actual-execution witness.
    # Symmetric OFF/ON; never counts an ACK as candidate execution.
    if events:
        torch.cuda.synchronize()
    rank = get_tensor_model_parallel_rank()
    row = dict(control=active, rank=rank, pid=os.getpid(), events=events[:],
               graph=runner._prefill_runtime_proof)
    with (ROOT / f'7b-rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    return '7b:' + arm
