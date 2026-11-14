# SPDX-License-Identifier: AGPL-3.0-only
"""Default-off, privately compiled native coop kernel; original extension stays installed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ASSETS=Path('/opt/dsv41-coop-kernel')
STOCK_SHA='cc7d6d1b53dd48fc6714147d913b4a8a06cb071ce50aac525a3a79769e205fcc'
_NATIVE=None

def eligible(args):
    x,sel=args[:2]
    return (x.is_cuda and str(x.dtype)=='torch.float16' and x.dim()==2 and x.shape[1]==5120
            and 1<=x.shape[0]<=42 and sel.dim()==2 and sel.shape[1]==6
            and x.shape[0]*sel.shape[1]<=256 and args[5]==5120
            and args[18]==args[19]==args[20] and args[18] in (2,3,4)
            and args[25] is True and args[29].shape[-1] in (512,640)
            and args[31].shape[-1]==5120
            and os.environ.get('EXL3_MOE_COOP_WIDE','-1') in ('','-1','1')
            and os.environ.get('VLLM_DSV41_TBO','0')=='0'
            and os.environ.get('EXL3_MOE_COOP_ROWINV','1')=='1'
            and os.environ.get('EXL3_MOE_COOP_KSPLIT','0') in ('','0','1'))

def load():
    global _NATIVE
    if _NATIVE is not None:return _NATIVE
    import torch
    from vllm.logger import init_logger
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Opera coop must initialize during warmup before graph capture')
    if torch.cuda.get_device_capability()!=(8,0):raise RuntimeError('Opera coop requires gated SM80')
    stock=importlib.util.find_spec('exllamav3_ext')
    if stock is None or hashlib.sha256(Path(stock.origin).read_bytes()).hexdigest()!=STOCK_SHA:
        raise RuntimeError('Opera coop stock native extension differs from qualified base')
    receipt=json.loads((ASSETS/'build-manifest.json').read_text());library=ASSETS/'opera_coop_ext.so'
    if str(torch.__version__)!=receipt['torch_version'] or receipt['stock_native_sha256']!=STOCK_SHA:
        raise RuntimeError('Opera coop Torch/native ABI does not match build receipt')
    if hashlib.sha256(library.read_bytes()).hexdigest()!=receipt['library_sha256']:
        raise RuntimeError('Opera coop private binary fingerprint changed')
    spec=importlib.util.spec_from_file_location('opera_coop_ext',library)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.opera_coop_variant_v1()!=1:raise RuntimeError('Opera coop private native probe mismatch')
    init_logger("vllm." + __name__).info('Opera coop native kernel enabled: branchfree/MMA-swap, I512 wide-down, I640 stock128; sha256=%s',receipt['library_sha256'])
    _NATIVE=module
    return module

def execute(*args):
    if not eligible(args):
        import exllamav3_ext
        return exllamav3_ext.exl3_moe_coop(*args)
    return load().exl3_moe_coop(*args)
