# SPDX-License-Identifier: MIT
"""Single target graph bank. Reuse rung-2 Binding and its exact native selectors."""
import functools
import inspect
import json
import os
import weakref
from pathlib import Path

binding = None
owner = None
targets = weakref.WeakSet()
entrypoints = ()

def record_capture():
    """One receipt after startup capture, never synchronous I/O per worker step."""
    if 'DSV41_QV_SESSION' not in os.environ: return
    from vllm.distributed import get_tensor_model_parallel_rank
    b = binding
    row = dict(pid=os.getpid(), on=os.environ.get('DSV41_MOE_STACK') == '1',
               installed=b is not None, single_bank_targets=len(targets),
               native_capture_calls=0 if b is None else b.calls,
               by_kind={} if b is None else b.by_kind)
    path = Path('/qv')/f'stack-rank{get_tensor_model_parallel_rank()}.json'
    tmp = path.with_suffix('.new'); tmp.write_text(json.dumps(row)); tmp.replace(path)
    if not row['installed'] or row['native_capture_calls'] == 0:
        print('WARN_ONLY STACK capture coverage not yet observed', flush=True)

def install_capture_only():
    """Keep the measured Binding, but leave no STACK callable in eager MoE."""
    global entrypoints
    import vllm_exl3.coop_kernel as coop
    import vllm_exl3.exl3 as exl3
    from stack_runtime import install
    sites = ((coop, 'execute'), (exl3, 'apply_exl3_fused_moe'))
    original = tuple(getattr(module, name) for module, name in sites)
    try:
        value = install()  # authenticates the same measured native bytes/ABI
        entrypoints = tuple((module, name, getattr(module, name)) for module, name in sites)
        return value
    finally:
        for (module, name), fn in zip(sites, original): setattr(module, name, fn)

def wrap(cls):
    if getattr(cls.load_model, '_moe_stack_single', False): return
    loaded, capture = cls.load_model, cls.capture_model
    @functools.wraps(loaded)
    def load(self, *a, **kw):
        global binding
        value = loaded(self, *a, **kw)
        if binding is not None: raise ValueError('STACK second model load')
        binding = install_capture_only()
        return value
    @functools.wraps(capture)
    def captured(self, *a, **kw):
        global owner
        profile = inspect.signature(capture).bind(self,*a,**kw).arguments.get('profile_only',False)
        saved = owner
        if not profile: owner = self.cudagraph_manager
        try:
            value = capture(self,*a,**kw)
            if not profile: record_capture()
            return value
        finally: owner = saved
    load._moe_stack_single = True
    cls.load_model, cls.capture_model = load, captured

def register():
    switch = os.environ.get('DSV41_MOE_STACK','0')
    if switch not in ('0','1'): raise ValueError('STACK switch')
    if switch == '0': return  # absolutely no imports/native allocation/hooks off
    if os.environ.get('VLLM_EXL3_COOP_KERNEL') != '1':
        raise ValueError('STACK needs VLLM_EXL3_COOP_KERNEL=1')
    import sys
    sys.path.insert(0,'/opt/stack-code')
    sys.path.insert(0,'/opt/qv')
    import dsv41_lora_argmax.hooks as hooks
    if getattr(hooks.patch,'_moe_stack_single',False): return
    original = hooks.patch
    @functools.wraps(original)
    def patch(module):
        original(module)
        if module.__name__ == 'vllm.v1.worker.gpu.model_runner': wrap(module.GPUModelRunner)
    patch._moe_stack_single = True
    hooks.patch = patch; hooks.install()
    target = sys.modules.get('vllm.v1.worker.gpu.model_runner')
    if target is not None: wrap(target.GPUModelRunner)
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager as Manager
    old = Manager.capture
    @functools.wraps(old)
    def capture(manager, create_forward_fn, *a, **kw):
        if manager is not owner: return old(manager,create_forward_fn,*a,**kw)
        targets.add(manager)
        def scoped(d, warmup):
            fn = create_forward_fn(d,warmup=warmup)
            from stack_runtime import kind
            if kind(d) is None: return fn
            def forward(mode):
                saved = binding.on, binding.descriptor
                originals = tuple(getattr(module, name) for module, name, _ in entrypoints)
                binding.on, binding.descriptor = True, d
                try:
                    for module, name, candidate in entrypoints: setattr(module, name, candidate)
                    return fn(mode)
                finally:
                    for (module, name, _), original in zip(entrypoints, originals): setattr(module, name, original)
                    binding.on, binding.descriptor = saved
            return forward
        return old(manager,scoped,*a,**kw)
    Manager.capture = capture
    # Keep the incumbent descriptor set and dispatch, including padded/varlen
    # replays. Wrappers exist only during eligible target capture forwards
    # (including their warmup); all eager calls use the exact incumbent objects.
