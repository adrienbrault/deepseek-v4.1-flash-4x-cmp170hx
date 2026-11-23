# SPDX-License-Identifier: MIT
"""Only replace token topk_glue.run; reuse the sealed, unmodified k1b DSO.

No host lens reads, lens expansion, copies, masking or additional store. An
unsupported ABI calls the incumbent with the exact original argument objects.
"""
import functools
import importlib
import importlib.util
import os
from pathlib import Path
from types import CodeType, FunctionType
from indexer_common import ROWS, need, read, sha

SWITCH = 'DSV41_INDEXER_BOUNDED'
LIVE = 'DSV41_INDEXER_BOUNDED_LIVE'
NAME = 'indexer_bounded_k1b'
# Registered served wrapper; keep its source in dependency-pins/deploy inventory.
AUDIT_MODULE = 'dsv41_attn_audit'
AUDIT_SHA256 = 'ad75c3358e2c739c6a5d473b9664b4a21cbb6c8d9429b98c4d5f7faa90849689'


def authenticate_glue(glue, expected_sha256):
    """Authenticate every wrapper, then the sealed innermost token function.

    wraps() copies __module__ from the original, so a wrapper's defining
    module is identified by its globals and registered code, not __module__.
    Return the module attribute unchanged: Binding installs above this chain.
    """
    original = current = glue.run
    seen = set()
    while True:
        need(isinstance(current, FunctionType) and id(current) not in seen,
             'invalid/cyclic token glue chain')
        seen.add(id(current))
        need(not getattr(current, '_indexer_bounded', False), 'duplicate token hook')
        if not hasattr(current, '__wrapped__'):
            need(current.__module__ == glue.__name__ and current.__globals__.get('__name__') == glue.__name__
                 and sha(current.__code__.co_filename) == expected_sha256, 'token glue source/hook drift')
            return original
        need(current.__globals__.get('__name__') == AUDIT_MODULE
             and getattr(current, '_r25_audit', False) is True
             and sha(current.__code__.co_filename) == AUDIT_SHA256, 'unknown token glue wrapper')
        audit = importlib.import_module(AUDIT_MODULE)
        need(current.__globals__ is vars(audit) and sha(audit.__file__) == AUDIT_SHA256
             and any(current.__code__ is code for code in audit.observe.__code__.co_consts
                     if isinstance(code, CodeType)), 'unknown token glue wrapper')
        closure = dict(zip(current.__code__.co_freevars, (c.cell_contents for c in current.__closure__)))
        need(closure.get('original') is current.__wrapped__ and closure.get('after') is None
             and closure.get('switch') == 'DSV41_ATTN_TOPK_GLUE', 'unknown token glue wrapper')
        current = current.__wrapped__


def native(directory, seal_path, expected_seal):
    need(sha(seal_path) == expected_seal, 'native seal changed')
    seal = read(seal_path)
    need(seal['schema'] == 'r18-indexer-native/1', 'native seal schema')
    need(sha(directory/'build-receipt.json') == seal['receipt_sha256'], 'native receipt changed')
    name = seal['binary']
    need(Path(name).name == name and name.startswith(NAME) and name.endswith('.so'), 'native basename')
    need(sha(directory/name) == seal['binary_sha256'], 'native binary changed')
    receipt = read(directory/'build-receipt.json')
    need(name == NAME+'.so' and set(receipt['artifacts']) == {name, 'build.ninja'} and
         all(sha(directory/f) == h for f,h in receipt['artifacts'].items()), 'native build artifacts changed')
    import torch
    from native_build import check_r18_environment
    check_r18_environment(receipt, torch)
    spec = importlib.util.spec_from_file_location(NAME, directory/name)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    need(callable(obj.run), 'native ABI')
    return obj


def supported(t, x, lens, next_n, out, k):
    # These are precisely k1b's qualified native preconditions, narrowed to
    # the observed token site. Nonzero-offset views/strided outputs fall back.
    nr = x.shape[0] if x.ndim == 2 else 0
    return (nr in ROWS and k == 512 and type(next_n) is int and next_n == 1
            and x.is_cuda and x.dtype == t.float32 and x.shape == (nr, 262144)
            and x.is_contiguous() and x.storage_offset() == 0
            and lens.shape == (nr, 1) and lens.dtype == t.int32
            and lens.device == x.device and lens.is_contiguous() and lens.storage_offset() == 0
            and out.shape == (nr, 512) and out.dtype == t.int32 and out.device == x.device
            and out.is_contiguous() and out.storage_offset() == 0)


class Binding:
    def __init__(self, torch, op, device):
        self.t, self.op, self.on = torch, op, False
        self.buffers, self.calls, self.fallbacks = {}, 0, 0
        self.eligible_calls = 0
        for nr in ROWS:
            self.buffers[nr] = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (
                ((nr, 512), torch.float32), ((nr, 512), torch.int64),
                ((nr, 256, 256), torch.int16), ((nr, 2), torch.int32),
                ((2, nr, 256), torch.int32)))
        self.bytes = sum(b.numel()*b.element_size() for row in self.buffers.values() for b in row)

    def install(self, glue):
        need(not getattr(glue.run, '_indexer_bounded', False), 'duplicate token hook')
        original = glue.run

        @functools.wraps(original)
        def run(x, lens, next_n, out, k):
            if self.on and x.ndim == 2 and x.shape[0] in ROWS and x.shape[1] == 262144 and k == 512 and next_n == 1:
                self.eligible_calls += 1
            if self.on and supported(self.t, x, lens, next_n, out, k):
                values, ids, hist, state, counts = self.buffers[x.shape[0]]
                self.op.run(x, lens, values, ids, out, hist, state, counts)
                self.calls += 1
                return None  # original glue writes caller storage and returns None
            self.fallbacks += 1
            return original(x, lens, next_n, out, k)
        run._indexer_bounded = True
        glue.run = run


def install(environ=None):
    env = os.environ if environ is None else environ
    need(env.get(SWITCH, '0') in ('0', '1') and env.get(LIVE, '0') in ('0', '1'), 'invalid switch')
    if env.get(SWITCH, '0') == env.get(LIVE, '0') == '0':
        return None  # no torch/glue import, allocation, launch, DSO load or hook
    import torch
    import dsv41_attn_port.topk_glue as glue
    from vllm.platforms import current_platform
    need(current_platform.is_cuda() and current_platform.is_device_capability(80), 'requires SM80')
    authenticate_glue(glue, read('/opt/qv/native-seal.json')['glue_sha256'])
    op = native(Path('/opt/indexer-native'), Path('/opt/qv/native-seal.json'),
                Path('/opt/qv/native-seal.sha256').read_text().strip())
    binding = Binding(torch, op, torch.device('cuda', torch.cuda.current_device()))
    binding.on = env.get(SWITCH, '0') == '1'
    binding.install(glue)
    return binding
