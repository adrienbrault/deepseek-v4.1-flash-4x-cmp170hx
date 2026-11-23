# SPDX-License-Identifier: MIT
"""Plain serving: existing hook chain, static ON, no control-file/per-step I/O."""
import functools
import os
import sys

def register():
    switch=os.environ.get('DSV41_FLY_SERVING','0')
    if switch not in ('0','1'):raise ValueError('FLy serving switch must be 0/1')
    if switch=='0' or os.environ.get('DSV41_QV')=='1':return
    from fly_names import runtime as fly
    fly.install()
    fly.cell_control=lambda:dict(record=False,diagnostic=False,admission=0)
    from dsv41_lora_argmax import hooks
    if getattr(hooks.patch,'_r7g_fly',False):return
    def wrap(cls):
        if getattr(cls.execute_model,'_r7g_fly',False):return
        original=cls.execute_model
        @functools.wraps(original)
        def execute(runner,*args,**kwargs):
            fly.select_worker(runner,'on')
            return original(runner,*args,**kwargs)
        execute._r7g_fly=True;cls.execute_model=execute
    original=hooks.patch
    @functools.wraps(original)
    def patch(module):
        original(module)
        if module.__name__==fly.CONFIG['runner']:wrap(module.GPUModelRunner)
    patch._r7g_fly=True;hooks.patch=patch;hooks.install()
    target=sys.modules.get(fly.CONFIG['runner'])
    if target is not None:wrap(target.GPUModelRunner)
