# SPDX-License-Identifier: MIT
"""Serving bootstrap for the exact measured adapter, using the incumbent hook.

QV owns the adapter unchanged during acceptance. Plain serving supplies only
its worker-install seam and inert control metadata; no measurement collector,
admission barrier, scheduler override or second import finder is installed.
"""
import functools
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

class FixedControl:
    def __truediv__(self,name):
        if name!='control.json':raise ValueError('unexpected serving control read')
        return self
    def read_text(self):return json.dumps(dict(effective='off'))

def install_adapter(adapter):
    # install() changes only qv.wrap_worker, then wraps the graph manager.
    # Keep the measured code intact and retain the resulting class wrapper.
    if 'dsv41_qv' in sys.modules:raise ValueError('unexpected QV owner in plain serving')
    shim=SimpleNamespace(wrap_worker=lambda cls:None)
    sys.modules['dsv41_qv']=shim
    try:adapter.install()
    finally:sys.modules.pop('dsv41_qv',None)
    adapter.owner.ROOT=FixedControl()  # active metadata is unused without audit_worker
    return shim.wrap_worker

def register():
    switch=os.environ.get('DSV41_PREFILL_7B_R7D','0')
    if switch not in ('0','1'):raise ValueError('7b boot switch must be 0/1')
    if switch=='0' or os.environ.get('DSV41_QV')=='1':return
    sys.path.insert(0,'/opt/qv')
    adapter=importlib.import_module('prefill_7b_candidate')
    import dsv41_lora_argmax.hooks as hooks
    if getattr(hooks.patch,'_prefill_7b_serving',False):return
    worker_install=install_adapter(adapter)
    def wrap(cls):
        if getattr(cls.execute_model,'_prefill_7b_serving',False):return
        worker_install(cls)
        original=cls.execute_model
        @functools.wraps(original)
        def execute(runner,*args,**kwargs):
            adapter.select_worker(runner,'off')
            return original(runner,*args,**kwargs)
        execute._prefill_7b_serving=True
        cls.execute_model=execute
    original=hooks.patch
    @functools.wraps(original)
    def patch(module):
        original(module)
        if module.__name__=='vllm.v1.worker.gpu.model_runner':wrap(module.GPUModelRunner)
    patch._prefill_7b_serving=True
    hooks.patch=patch;hooks.install()
    target=sys.modules.get('vllm.v1.worker.gpu.model_runner')
    if target is not None:wrap(target.GPUModelRunner)
