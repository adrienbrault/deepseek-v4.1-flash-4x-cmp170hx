# SPDX-License-Identifier: MIT
"""Single-bank daily hook; compose through the incumbent argmax import owner."""
import functools
import os
import sys

TARGET = 'vllm.v1.worker.gpu.model_runner'
binding = None

def wrap(cls):
    old = cls.load_model
    if getattr(old, '_indexer_confirm3', False): return
    @functools.wraps(old)
    def loaded(self, *args, **kw):
        global binding
        if binding is not None: raise ValueError('bounded indexer: second runner/load')
        value = old(self, *args, **kw)
        from indexer_runtime import install
        binding = install()
        if binding is None: raise ValueError('bounded indexer ON not installed')
        return value
    loaded._indexer_confirm3 = True
    cls.load_model = loaded

def register():
    arm = os.environ.get('DSV41_INDEXER_BOUNDED', '0')
    if arm not in ('0','1'): raise ValueError('bounded indexer: switch')
    if arm == '0': return  # no imports, allocations, native load or hooks
    if os.environ.get('DSV41_ATTN_TOPK_GLUE') != '1':
        raise ValueError('bounded indexer needs DSV41_ATTN_TOPK_GLUE=1')
    sys.path.insert(0, '/opt/qv')
    import dsv41_lora_argmax.hooks as owner
    if getattr(owner.patch, '_indexer_confirm3', False): return
    original = owner.patch
    @functools.wraps(original)
    def patch(module):
        original(module)
        if module.__name__ == TARGET: wrap(module.GPUModelRunner)
    patch._indexer_confirm3 = True
    owner.patch = patch
    # Use the incumbent's existing finder; do not introduce a competing loader.
    owner.install()
    if TARGET in sys.modules: wrap(sys.modules[TARGET].GPUModelRunner)
