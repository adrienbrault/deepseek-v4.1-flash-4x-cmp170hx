# SPDX-License-Identifier: AGPL-3.0-only
# The coop branch below reproduces vllm-exl3's apply_exl3_fused_moe (vcruz305/vllm-exl3 3c60e6b + plugin
# patches, as served by moe-gap-r2 stage-r18, exl3.py sha256 8468e2ec...): same predicates, same
# preamble order, same scratch allocation and the same 36 positional coop arguments. Only the prep that
# builds xh/sel/rw/ctr/out changes (prep.py). Early-coop idea: GNmac-lab bbaf367 (see NOTICE.md). The
# meta-path hook, shadow counters and drain follow ../../rowstats-r1 and ../../comm-r1 (MIT).
"""Import-time hook on vllm_exl3.exl3: one-launch dispatch prep on the coop path; refuses anything else.

``apply_exl3_fused_moe`` is replaced on the module object. Its only callers (``apply_exl3_experts``
and its own prefill slicing) resolve it through the module globals at call time, so nothing is edited.

The wrapper takes the fused path only when every stock predicate of the coop branch holds, evaluated
host-side from shapes, dtypes and layer attributes (``plan``): coop on, uniform-K, gated, no expert
map, ``tokens * topk <= 256`` (<= 42 rows at top-6), no fat expert possible (``tokens <=
FAT_EXPERT_THRESHOLD``), uniform codebook flags, hidden and intermediate multiples of 128, and the
coop op present. Inputs must also be contiguous, 16-byte aligned CUDA tensors of the dtypes the
kernel was warmed for. Everything else (43-64 rows: TC-MoE V2; prefill; any other layer) calls the
stock function unchanged. On the fused path the stock preamble still runs first in stock order
(TC hook, then the native backend attempt), then the coop kernel receives the prep's tensors.

The first call (vLLM's profile run, eager) is the READY point: it refuses an inert arm (coop off) and
compiles the kernel for the observed dtypes, so no compile happens inside a graph capture.

Verification boots only (DSV41_DISPATCH_FUSE_SHADOW=1): every fused call first runs the stock function
with the coop entry swapped for a recorder (the stock prep runs, the kernel does not), then the fused
prep, and counts on the device, inside the captured graph, the elements of xh/sel/rw/ctr/out whose
bits differ, separately for eager calls and graph replays, plus a per-row-count call histogram for
the fused path and for every passthrough call (V2 at 43-64 rows). Non-tensor arguments, scratch
geometry and pointer-table identity are compared host-side. A nonce written to
DSV41_DISPATCH_FUSE_DRAIN_FILE makes the next eager call synchronize and log
``DSV41_DISPATCH_FUSE_FINAL`` (drain.py).
"""
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys

from vllm.logger import init_logger

from . import prep as prep_mod

EXL3 = 'vllm_exl3.exl3'
# The exact source whose coop branch this wrapper reproduces (moe-gap-r2 stage-r18 overlay).
PINS = {
    EXL3: '8468e2ec0108937adcf6f17ed9c22db97e68665457b61a5b5b228fdba862f918',
}
SHADOW_ENV = 'DSV41_DISPATCH_FUSE_SHADOW'
DRAIN_ENV = 'DSV41_DISPATCH_FUSE_DRAIN_FILE'
DRAIN_FILE_RE = re.compile(r'/tmp/[A-Za-z0-9._-]{1,64}')
TOKEN_RE = re.compile(r'[A-Za-z0-9]{8,64}')
MAX_ROWS = 64                     # histogram buckets 0..64; bucket 65 = more than 64 rows
FIELDS = ('calls', 'rows', 'bad_xh', 'bad_sel', 'bad_rw', 'bad_ctr', 'bad_out')
X_DTYPES = ('bfloat16', 'float16')
ID_DTYPES = ('int32', 'int64')
W_DTYPES = ('float32', 'bfloat16', 'float16')
PTR_KEYS = ('gate_trellis', 'gate_suh', 'gate_svh', 'up_trellis', 'up_suh', 'up_svh',
            'down_trellis', 'down_suh', 'down_svh')
# Positions in the 36-argument coop call.
A_XH, A_SEL, A_RW, A_CTR, A_OUT = 0, 1, 2, 32, 33
A_PTRS = range(6, 15)
logger = init_logger('vllm.' + __name__)


class Refused(RuntimeError):
    pass


def shadow_config():
    """(shadow on, drain file or None) for a verification boot; refuses half settings."""
    raw = os.environ.get(SHADOW_ENV, '0')
    drain = os.environ.get(DRAIN_ENV, '') or None
    if raw not in ('0', '1'):
        raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED {SHADOW_ENV}={raw!r} must be 0 or 1')
    if drain is not None and (raw != '1' or not DRAIN_FILE_RE.fullmatch(drain) or '..' in drain):
        raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED {DRAIN_ENV} needs {SHADOW_ENV}=1 and a /tmp/<name> path')
    return raw == '1', drain


def read_token(path):
    try:
        with open(path) as f:
            token = f.read(80).strip()
    except OSError:
        return None
    return token if TOKEN_RE.fullmatch(token) else None


def source_sha(name):
    """sha256 of a module's source file, located without executing it (its parent package is imported)."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED missing module {name}')
    with open(spec.origin, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_sources(sha=source_sha):
    for name, want in PINS.items():
        got = sha(name)
        if got != want:
            raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED {name} sha256 {got} != pinned {want}')


def dtype_name(dtype):
    return str(dtype).rsplit('.', 1)[-1]


def plan(m, x2d, ids, weights, layer, inners, expert_map, *, import_ext, device_ok):
    """The stock coop-branch predicates, host-only: a dict for the fused path, else the reason (str)."""
    if not m._COOP:
        return 'coop-off'
    if x2d.dim() != 2 or ids.dim() != 2 or tuple(weights.shape) != tuple(ids.shape):
        return 'geometry'
    tokens, hidden = (int(v) for v in x2d.shape)
    topk = int(ids.shape[-1])
    if int(ids.shape[0]) != tokens or tokens < 1 or topk < 1:
        return 'geometry'
    if tokens * topk > prep_mod.COOP_MAX_ROUTES:
        return 'rows'                      # 43-64 rows: TC-MoE V2; prefill; or the stock path
    if tokens > m.FAT_EXPERT_THRESHOLD or tokens > m.TEMP_ROWS_FUSED:
        return 'fat'
    if getattr(layer, '_exl3_mixedk_unified', False) or getattr(layer, '_exl3_mixed_store', None) is not None:
        return 'mixed-k'
    if getattr(layer, '_exl3_gateless', False):
        return 'gateless'
    if expert_map is not None:
        return 'expert-map'
    ptrs = getattr(layer, '_exl3_ptrs', None)
    temps = getattr(layer, '_exl3_fused_temps', None)
    if not ptrs or temps is None:
        return 'no-pointer-tables'         # stock raises; let it
    flags = getattr(layer, '_exl3_codebook_flags', m._MCG_CODEBOOK_FLAGS)
    mcg, mul1 = bool(flags[0]), bool(flags[1])
    inter_dim = int(temps[2].shape[-1])
    if not (all(bool(flags[i]) == mcg and bool(flags[i + 1]) == mul1 for i in range(0, 6, 2))
            and hidden % 128 == 0 and inter_dim % 128 == 0):
        return 'codebook-geometry'
    if (dtype_name(x2d.dtype) not in X_DTYPES or dtype_name(ids.dtype) not in ID_DTYPES
            or dtype_name(weights.dtype) not in W_DTYPES):
        return 'dtype'
    if not (x2d.is_contiguous() and ids.is_contiguous() and weights.is_contiguous()):
        return 'layout'
    if not device_ok(x2d, ids, weights):
        return 'device'
    try:
        ext = import_ext()
    except Exception:
        return 'no-extension'              # stock: Python loop
    if not hasattr(ext, 'exl3_moe_coop'):
        return 'no-coop-op'
    return dict(tokens=tokens, hidden=hidden, topk=topk, inter_dim=inter_dim, n_exp=len(inners), ptrs=ptrs,
                k=int(getattr(layer, '_exl3_k', 4)), mcg=mcg, mul1=mul1, ext=ext)


def coop_args(m, p, xh, sel, rw, ctr, out, limit, torch):
    """The stock coop call's 36 positional arguments, scratch allocated exactly as stock does."""
    tokens, hidden, inter_dim, k = p['tokens'], p['hidden'], p['inter_dim'], p['k']
    slots = tokens * p['topk']
    smax = 4 * slots
    dev = xh.device
    had_g = torch.empty((slots, hidden), dtype=torch.float16, device=dev)
    had_u = torch.empty_like(had_g)
    gu_g = torch.empty((smax, 1, inter_dim), dtype=torch.float16, device=dev)
    gu_u = torch.empty_like(gu_g)
    act_out = torch.empty_like(gu_g)
    d_out = torch.empty((smax, 1, hidden), dtype=torch.float32, device=dev)
    ptrs = p['ptrs']
    return (xh, sel, rw, 0, int(p['n_exp']), hidden,
            *(ptrs[key] for key in PTR_KEYS),
            None, None, None,
            k, k, k, p['mcg'], p['mul1'], m.MOE_ACT_SILU,
            float(limit) if (limit is not None and limit > 0) else 0.0,
            True,
            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out,
            None, None)


def coop_entry(m, ext):
    """(object, attribute) the stock branch calls: the private coop wrapper or the extension op."""
    if m._COOP_KERNEL:
        return importlib.import_module(m.__package__ + '.coop_kernel'), 'execute'
    return ext, 'exl3_moe_coop'


def record_stock(m, ext, call):
    """Run the stock function with its coop entry swapped for a recorder: the stock prep runs, the
    kernel does not. Returns the recorded argument tuple, or None if the stock took another path."""
    seen = []

    def recorder(*args):
        seen.append(args)
    target, attr = coop_entry(m, ext)
    saved = getattr(target, attr)
    setattr(target, attr, recorder)
    try:
        call()
    finally:
        setattr(target, attr, saved)
    return seen[0] if len(seen) == 1 else None


def meta_diff(stock, fused):
    """Argument positions whose host-visible metadata differ (tensor geometry, scalars, table identity)."""
    if len(stock) != len(fused):
        return ['arity']
    bad = []
    for i, (a, b) in enumerate(zip(stock, fused)):
        if i in A_PTRS:
            if a is not b:
                bad.append(i)
        elif hasattr(a, 'shape') or hasattr(b, 'shape'):
            if not (hasattr(a, 'shape') and hasattr(b, 'shape')) or (
                    tuple(a.shape), a.dtype, tuple(a.stride()), a.device) != (
                    tuple(b.shape), b.dtype, tuple(b.stride()), b.device):
                bad.append(i)
        elif type(a) is not type(b) or a != b:
            bad.append(i)
    return bad


def bits(t):
    import torch
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def bucket(rows):
    return min(int(rows), MAX_ROWS + 1)


def hist(row):
    """'rows:count,...' of the nonzero buckets (65 = more than 64 rows), '-' when empty."""
    return ','.join(f'{r}:{c}' for r, c in enumerate(row) if c) or '-'


class Shadow:
    """Verification-only device counters, allocated at the READY call (eager, before any capture)."""

    def __init__(self, zeros):
        self.counters = zeros(2, len(FIELDS))      # [0] eager calls, [1] graph replays (served decode)
        self.fused_rows = zeros(2, MAX_ROWS + 2)   # fused calls per row count
        self.pass_rows = zeros(2, MAX_ROWS + 2)    # stock passthrough calls per row count (V2 at 43-64)
        self.bad_meta = 0                          # host-side, per call site (eager + capture)
        self.bad_path = 0

    def compare(self, stock, fused, tokens, capturing, log):
        import torch
        cap = int(capturing)
        if stock is None:
            self.bad_path += 1
            log('path', 'stock did not reach the coop branch for a fused-eligible call')
            return
        diff = meta_diff(stock, fused)
        if diff:
            self.bad_meta += 1
            log('meta', f'argument metadata differ at positions {diff}')
            if set(diff) & {A_XH, A_SEL, A_RW, A_CTR, A_OUT}:
                return
        bad = torch.stack([
            (bits(stock[A_XH]) != bits(fused[A_XH])).sum(),
            (stock[A_SEL] != fused[A_SEL]).sum(),
            (bits(stock[A_RW]) != bits(fused[A_RW])).sum(),
            (stock[A_CTR] != fused[A_CTR]).sum(),
            (bits(stock[A_OUT]) != bits(fused[A_OUT])).sum(),
        ])
        # Recorded into the graph when capturing, so [1] grows only on replay.
        self.counters[cap, 0].add_(1)
        self.counters[cap, 1].add_(tokens)
        self.counters[cap, 2:].add_(bad)
        self.fused_rows[cap, bucket(tokens)].add_(1)


class State:
    def __init__(self, rank, shadow=None, drain_file=None):
        self.rank = rank
        self.shadow = shadow
        self.drain_file = drain_file
        self.drained = None
        self.drain_stamp = None
        self.announced = set()
        self.warmed = set()
        self.noted = set()


def ready(m, b, x2d, ids, weights, capturing):
    rank = b['tp_rank']()
    if capturing:
        raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED rank={rank} first MoE call inside a graph capture '
                      '(no eager point to compile the kernel)')
    if not m._COOP:
        # The layer was requested explicitly: an inert arm would waste a screen.
        raise Refused(f'DSV41_DISPATCH_FUSE_REFUSED rank={rank} VLLM_EXL3_COOP is off (no coop path to fuse)')
    shadow, drain = shadow_config()
    st = State(rank, Shadow(b['zeros']) if shadow else None, drain)
    key = (x2d.dtype, ids.dtype, weights.dtype)
    if (dtype_name(key[0]) in X_DTYPES and dtype_name(key[1]) in ID_DTYPES and dtype_name(key[2]) in W_DTYPES):
        b['prewarm'](x2d.device, *key)
        st.warmed.add(key)
    m._dsv41_dispatch_fuse = st
    logger.info('DSV41_DISPATCH_FUSE_READY rank=%d coop_kernel=%d fat_threshold=%d warmed=%s shadow=%d',
                rank, int(bool(m._COOP_KERNEL)), int(m.FAT_EXPERT_THRESHOLD),
                '/'.join(dtype_name(d) for d in key) if st.warmed else '-', int(shadow))
    return st


def drain(st, sync):
    """Verification drain: a new nonce in the drain file -> device sync -> one FINAL line."""
    sh = st.shadow
    try:
        stamp = os.stat(st.drain_file).st_mtime_ns
    except OSError:
        return
    if stamp == st.drain_stamp:
        return
    st.drain_stamp = stamp
    token = read_token(st.drain_file)
    if token is None or token == st.drained:
        return
    st.drained = token
    sync()
    eager, graph = sh.counters.cpu().tolist()
    fused, passed = sh.fused_rows.cpu().tolist(), sh.pass_rows.cpu().tolist()
    logger.info('DSV41_DISPATCH_FUSE_FINAL rank=%d token=%s calls=%d rows=%d bad_xh=%d bad_sel=%d bad_rw=%d '
                'bad_ctr=%d bad_out=%d bad_meta=%d bad_path=%d fused_rows=%s pass_rows=%s eager=%d:%d:%s',
                st.rank, token, *graph, sh.bad_meta, sh.bad_path, hist(fused[1]), hist(passed[1]),
                eager[0], sum(eager[2:]), hist(fused[0]))


def wrap(original, *, module, b):
    m = module

    def note(st, kind, rows, capturing, detail):
        key = (kind, rows, capturing, detail)
        if key not in st.noted:
            st.noted.add(key)
            logger.warning('DSV41_DISPATCH_FUSE_%s rank=%d rows=%d capturing=%d %s',
                           kind, st.rank, rows, int(capturing), detail)

    @functools.wraps(original)
    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit=None):
        import torch
        capturing = bool(b['capturing']())
        st = getattr(m, '_dsv41_dispatch_fuse', None)
        if st is None:
            st = ready(m, b, x2d, ids, weights, capturing)
        sh = st.shadow
        if sh is not None and not capturing and st.drain_file:
            drain(st, b['sync'])
        rows = int(x2d.shape[0]) if x2d.dim() else 0
        p = plan(m, x2d, ids, weights, layer, inners, expert_map,
                 import_ext=b['import_ext'], device_ok=b['device_ok'])
        if isinstance(p, str):
            if p not in ('rows', 'coop-off'):
                note(st, 'SKIP', rows, capturing, 'reason=' + p)
            if sh is not None:
                sh.pass_rows[int(capturing), bucket(rows)].add_(1)
            return original(x2d, ids, weights, layer, inners, expert_map, limit)
        key = (x2d.dtype, ids.dtype, weights.dtype)
        if key not in st.warmed:
            if capturing:
                note(st, 'COLD', rows, capturing, 'key=' + '/'.join(dtype_name(d) for d in key))
                if sh is not None:
                    sh.pass_rows[1, bucket(rows)].add_(1)
                return original(x2d, ids, weights, layer, inners, expert_map, limit)
            b['prewarm'](x2d.device, *key)
            st.warmed.add(key)
        # The stock preamble, in stock order (host-only for these shapes: both return None).
        tc_out = importlib.import_module(m.__package__ + '.tc_moe_hook').try_apply(
            x2d, ids, weights, layer, expert_map, limit)
        if tc_out is not None:
            return tc_out
        if m.get_moe_kernel_backend() == 'native':
            try:
                native_out = m._apply_native_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
            except Exception as exc:
                native_out = None
                layer._exl3_native_error = repr(exc)
                getattr(m.logger, 'warning_once', m.logger.warning)(
                    'Native EXL3 MoE dispatch failed in fused entry point; '
                    'falling back to ExLlamaV3/Python: %s', exc)
            if native_out is not None:
                layer._exl3_last_apply = 'native'
                return native_out
        tokens = p['tokens']
        key_seen = (tokens, capturing)
        if key_seen not in st.announced:   # once per row count, separately for eager and graph capture
            st.announced.add(key_seen)
            logger.info('DSV41_DISPATCH_FUSE_ACTIVE rank=%d rows=%d capturing=%d', st.rank, tokens, int(capturing))
        stock = None
        if sh is not None:
            stock = record_stock(m, p['ext'], lambda: original(x2d, ids, weights, layer, inners, expert_map, limit))
        n_ctr = prep_mod.counter_len(tokens, p['topk'], p['hidden'], p['inter_dim'])
        sel, rw, xh, out, ctr = b['prep'](x2d, ids, weights, p['n_exp'], n_ctr)
        args = coop_args(m, p, xh, sel, rw, ctr, out, limit, torch)
        if sh is not None:
            sh.compare(stock, args, tokens, capturing,
                       lambda kind, detail: note(st, 'MISMATCH', tokens, capturing, f'kind={kind} {detail}'))
        target, attr = coop_entry(m, p['ext'])
        getattr(target, attr)(*args)
        return out
    return apply_exl3_fused_moe


def _cuda_aligned(*tensors):
    first = tensors[0]
    return all(t.is_cuda and t.device == first.device and t.data_ptr() % 16 == 0 for t in tensors)


def _vllm_bindings():
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    return dict(
        tp_rank=get_tensor_model_parallel_rank,
        zeros=lambda *shape: torch.zeros(*shape, dtype=torch.int64,
                                         device=torch.device('cuda', torch.cuda.current_device())),
        sync=torch.cuda.synchronize,
        capturing=torch.cuda.is_current_stream_capturing,
        prep=prep_mod.fused_prep,
        prewarm=prep_mod.prewarm,
        device_ok=_cuda_aligned,
        import_ext=lambda: importlib.import_module('exllamav3_ext'),
    )


def patch(module, bindings=None):
    if getattr(module, '_dsv41_dispatch_fuse_patched', False):
        return
    b = bindings or _vllm_bindings()
    module.apply_exl3_fused_moe = wrap(module.apply_exl3_fused_moe, module=module, b=b)
    module._dsv41_dispatch_fuse_patched = True


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != EXL3:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('missing hook target ' + fullname)
        loader = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s) if hasattr(loader, 'create_module') else None

            def exec_module(self, module):
                loader.exec_module(module)
                patch(module)
        spec.loader = Loader()
        return spec


def install():
    shadow_config()
    check_sources()
    if EXL3 in sys.modules:
        patch(sys.modules[EXL3])
    elif not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    logger.info('DSV41_DISPATCH_FUSE_INSTALLED pid=%d shadow=%d', os.getpid(), int(shadow_config()[0]))
