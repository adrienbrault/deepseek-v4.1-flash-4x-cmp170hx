# SPDX-License-Identifier: MIT
"""Import-time hook on the stage-r18 row-mask bridge; refuses anything it was not built for.

Four module functions of ``vllm.utils.dsv41_row_mask`` are wrapped. Callers resolve them through
the module object at call time (``_dsv41_row_mask.record_top1(...)``), so no caller is edited:

* ``configure_drafter`` (draft model load, before any capture): after the stock body, refuse
  unless the row mask is on, the loaded policy's entropy weight is exactly 0 at every position
  and no host-width ``p_hat`` is produced (``DSV41_DEPTH_HOST`` off: the decision then writes
  integers only, see lean.py); log ``DSV41_ROWSTATS_READY``. The decision is a boot-time constant.
* ``record_top1`` (every Markov step inside the DSpark draft graph, c1 under cap 1): the lean
  write of ``lean.record``. A graph captured for a request count replays what it captured.
* ``publish`` and ``begin_target``: verification boots only (below); otherwise the stock body.

Verification boots only (DSV41_ROWSTATS_SHADOW=1):

* ``record_top1`` first runs the stock body on the same logits, keeps its row in a shadow buffer,
  then the lean write, and counts on the device, inside the captured graph, rows whose top1 bits
  differ and rows whose entropy NaN classification differs.
* ``publish`` runs the stock body a second time into shadow slot buffers, from the stock rows.
* ``begin_target`` (eager), after the stock body computed this step's decision from the served
  (lean) statistics, reruns the same decision (``compute_row_mask``: the CUDA kernel on the GPU)
  on scratch outputs from the stock statistics and counts ``live_len`` and ``row_live``
  mismatches: the integer decision outputs themselves, not just a NaN pattern.
* A nonce written to DSV41_ROWSTATS_DRAIN_FILE makes the next eager ``begin_target`` synchronize
  and log ``DSV41_ROWSTATS_FINAL`` (drain.py).
"""
import copy
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys

from vllm.logger import init_logger

from . import lean

ROW_MASK = 'vllm.utils.dsv41_row_mask'
SPECULATOR = 'vllm.v1.worker.gpu.spec_decode.dspark.speculator'
DECISION = 'dsv41_depth.decision'
POLICY = 'dsv41_depth.policy'
# The exact sources the equivalence argument in lean.py reads (moe-gap-r2 stage-r18 overlay).
PINS = {
    ROW_MASK: '50abe80da1499f61cbc43388564a709897180ffb6f6a951b34e952dabe861e2c',
    SPECULATOR: 'fa7df49bf473ef44c18c83d75d2c8ebe5adeaa43a76912a800521d16b5d31828',
    DECISION: 'fc4c8c4fa261123b15741373c35632ca6e7a00f9065dded35137fe4a7f6cfa72',
    POLICY: 'decdf7bc2a059823571503a8b6edf358dd19c14db5d01174c8bf2e73760fe4e3',
}
WRAPPED = ('configure_drafter', 'record_top1', 'publish', 'begin_target')
SHADOW_ENV = 'DSV41_ROWSTATS_SHADOW'
DRAIN_ENV = 'DSV41_ROWSTATS_DRAIN_FILE'
DRAIN_FILE_RE = re.compile(r'/tmp/[A-Za-z0-9._-]{1,64}')
TOKEN_RE = re.compile(r'[A-Za-z0-9]{8,64}')
ENT = 'ent'
logger = init_logger('vllm.' + __name__)


class Refused(RuntimeError):
    pass


def shadow_config():
    """(shadow on, drain file or None) for a verification boot; refuses half settings."""
    raw = os.environ.get(SHADOW_ENV, '0')
    drain = os.environ.get(DRAIN_ENV, '') or None
    if raw not in ('0', '1'):
        raise Refused(f'DSV41_ROWSTATS_REFUSED {SHADOW_ENV}={raw!r} must be 0 or 1')
    if drain is not None and (raw != '1' or not DRAIN_FILE_RE.fullmatch(drain) or '..' in drain):
        raise Refused(f'DSV41_ROWSTATS_REFUSED {DRAIN_ENV} needs {SHADOW_ENV}=1 and a /tmp/<name> path')
    return raw == '1', drain


def read_token(path):
    try:
        with open(path) as f:
            token = f.read(80).strip()
    except OSError:
        return None
    return token if TOKEN_RE.fullmatch(token) else None


def source_sha(name):
    """sha256 of a module's source file, located without executing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise Refused(f'DSV41_ROWSTATS_REFUSED missing module {name}')
    with open(spec.origin, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_sources(sha=source_sha):
    for name, want in PINS.items():
        got = sha(name)
        if got != want:
            raise Refused(f'DSV41_ROWSTATS_REFUSED {name} sha256 {got} != pinned {want}')


def ent_weights_zero(state, features):
    """Every position's entropy weight is exactly zero (either sign). Reads the device copy once."""
    column = state.weights[:, features.index(ENT)]
    return bool((column == 0).all().item())


class Shadow:
    """Verification-only buffers, allocated at draft-model load (before any capture)."""

    def __init__(self, speculator, state, zeros):
        import torch
        # int64 [2, 4] = (rows compared, top1 bit mismatches, entropy NaN-classification
        # mismatches, rows whose stock entropy is non-finite); [0] eager calls (warmups, uncaptured
        # sizes), [1] CUDA-graph replays only, i.e. served traffic.
        self.counters = zeros(2, 4)
        # int64 [3] = (requests decided, live_len mismatches, row_live rows that differ), counted
        # eagerly at begin_target over every uniform masked step.
        self.decisions = zeros(3)
        self.stats = torch.zeros_like(speculator._dsv41_row_stats)   # stock rows, batch order
        self.conf_by_slot = torch.ones_like(state.conf_by_slot)       # shadow publish target only
        self.stats_by_slot = torch.zeros_like(state.stats_by_slot)    # stock rows, slot order
        self.ref_slots = torch.zeros_like(state.stats_by_slot)        # this step's reference input
        self.live_len = torch.zeros_like(state.live_len)
        self.row_live = torch.zeros_like(state.row_live)


class Lean:
    def __init__(self, rank, shadow=None, drain_file=None):
        self.rank = rank
        self.shadow = shadow
        self.drain_file = drain_file
        self.drained = None
        self.announced = set()

    @property
    def counters(self):
        return None if self.shadow is None else self.shadow.counters


def wrap_configure(original, *, module, tp_rank, zeros, features):
    @functools.wraps(original)
    def configure(speculator, model):
        original(speculator, model)
        rank = tp_rank()
        state = module.get_state()
        if state is None:
            # The layer was requested explicitly: an inert arm would waste a screen.
            raise Refused(f'DSV41_ROWSTATS_REFUSED rank={rank} row mask is off (nothing to make lean)')
        if not ent_weights_zero(state, features):
            raise Refused(f'DSV41_ROWSTATS_REFUSED rank={rank} policy uses the entropy feature')
        if state.p_hat_width:
            # p_hat is the decision's only floating-point output; the contract covers integers only.
            raise Refused(f'DSV41_ROWSTATS_REFUSED rank={rank} host-width p_hat is on (DSV41_DEPTH_HOST=1)')
        shadow, drain = shadow_config()
        module._dsv41_rowstats = Lean(rank, Shadow(speculator, state, zeros) if shadow else None, drain)
        logger.info('DSV41_ROWSTATS_READY rank=%d ent_weight=0 uses_stats=%d row_max_batch=%d shadow=%d',
                    rank, int(bool(state.uses_stats)), state.row_max_batch, int(shadow))
    return configure


def wrap_record(original, *, module, capturing, record=lean.record):
    @functools.wraps(original)
    def record_top1(speculator, logits, step):
        st = getattr(module, '_dsv41_rowstats', None)
        if st is None:
            return original(speculator, logits, step)
        n = logits.shape[0]
        if not module.features_enabled(n):
            return None
        key = (n, bool(capturing()))
        if key not in st.announced:  # once per count, separately for warmup/eager and graph capture
            st.announced.add(key)
            logger.info('DSV41_ROWSTATS_ACTIVE rank=%d num_reqs=%d capturing=%d', st.rank, n, int(key[1]))
        sh = st.shadow
        if sh is None:
            return record(speculator, logits, step)
        # Shadow: the stock statistics first, on the same logits, then the lean write over them.
        original(speculator, logits, step)
        out = speculator._dsv41_row_stats
        ref = sh.stats[:n, :, step]
        ref.copy_(out[:n, :, step])
        record(speculator, logits, step)
        new = out[:n, :, step]
        bad_top1 = (_bits(new[:, 0]) != _bits(ref[:, 0])).sum()
        bad_term = (new[:, 1].isnan() != ref[:, 1].isnan()).sum()
        nonfinite = (~ref[:, 1].isfinite()).sum()
        rows = bad_top1.new_full((), n)
        # Recorded into the graph when capturing, so [1] grows only on replay.
        sh.counters[int(key[1])].add_(_stack((rows, bad_top1, bad_term, nonfinite)))
        return None
    return record_top1


def wrap_publish(original, *, module):
    """Shadow only: the stock publish again, from the stock rows into the shadow slot buffers.

    Runs inside the draft graph like the stock call; the swapped attributes are read by the stock
    body at call (capture) time, so the replay writes the shadow buffers, never the served ones."""
    @functools.wraps(original)
    def publish(speculator, num_reqs):
        result = original(speculator, num_reqs)
        st = getattr(module, '_dsv41_rowstats', None)
        state = module.get_state()
        if st is None or st.shadow is None or state is None:
            return result
        proxy = copy.copy(state)
        proxy.conf_by_slot = st.shadow.conf_by_slot
        proxy.stats_by_slot = st.shadow.stats_by_slot
        served = speculator._dsv41_row_stats
        module._STATE, speculator._dsv41_row_stats = proxy, st.shadow.stats
        try:
            original(speculator, num_reqs)
        finally:
            module._STATE, speculator._dsv41_row_stats = state, served
        return result
    return publish


def compare_decision(sh, state, batch, dummy_run):
    """After the stock begin_target: the same decision from the stock statistics, compared.

    The served ``stats_by_slot`` holds the lean rows of every slot published since its last reset;
    the shadow slot buffer the stock rows of the same publishes, but it never sees the stock resets
    (``reset_slot``, ``after_propose``, ``begin_step``'s stale/pending fills). A slot position is
    taken from the shadow only where its top1 bits equal the served ones (top1 is identical in both
    writes; a reset zeroes it, and a published top1 is >= 1/vocab or NaN, never +0.0); elsewhere
    the served value, which a reset made identical in both runs. Integer outputs compared only."""
    import torch
    from dsv41_depth import decision
    if not state.active:            # this step's stock begin_step did not run the decision
        return
    k = decision.uniform_num_drafts(state, batch, dummy_run)
    if not k:
        return
    n = batch.num_reqs
    ref = sh.ref_slots
    ref.copy_(state.stats_by_slot)
    published = _bits(sh.stats_by_slot[:, 0]) == _bits(state.stats_by_slot[:, 0])
    ref[:, 1] = torch.where(published, sh.stats_by_slot[:, 1], state.stats_by_slot[:, 1])
    scratch = copy.copy(state)
    scratch.stats_by_slot, scratch.live_len, scratch.row_live = ref, sh.live_len, sh.row_live
    decision.compute_row_mask(scratch, batch.idx_mapping, n, k, None)
    bad_len = (sh.live_len[:n] != state.live_len[:n]).sum()
    bad_rows = (sh.row_live != state.row_live).sum()
    sh.decisions.add_(_stack((bad_len.new_full((), n), bad_len, bad_rows)))


def wrap_begin(original, *, module, capturing, sync):
    """Verification drain: a new nonce in the drain file -> device sync -> one FINAL line."""
    @functools.wraps(original)
    def begin_target(batch, dummy_run):
        st = getattr(module, '_dsv41_rowstats', None)
        sh = None if st is None or capturing() else st.shadow
        if sh is not None and st.drain_file:
            token = read_token(st.drain_file)
            if token is not None and token != st.drained:
                st.drained = token
                sync()
                eager, graph = sh.counters.cpu().tolist()
                logger.info('DSV41_ROWSTATS_FINAL rank=%d token=%s rows=%d bad_top1=%d bad_term=%d '
                            'nonfinite=%d decisions=%d bad_live_len=%d bad_row_live=%d eager=%d:%d:%d:%d',
                            st.rank, token, *graph, *sh.decisions.cpu().tolist(), *eager)
        result = original(batch, dummy_run)
        state = module.get_state()
        if sh is not None and state is not None:
            compare_decision(sh, state, batch, dummy_run)
        return result
    return begin_target


def _bits(t):
    import torch
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _stack(values):
    import torch
    return torch.stack(values)


def _vllm_bindings():
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from dsv41_depth.policy import FEATURES
    return dict(
        tp_rank=get_tensor_model_parallel_rank,
        zeros=lambda *shape: torch.zeros(*shape, dtype=torch.int64,
                                         device=torch.device('cuda', torch.cuda.current_device())),
        sync=torch.cuda.synchronize,
        capturing=torch.cuda.is_current_stream_capturing,
        features=FEATURES,
    )


def patch(module, bindings=None):
    if getattr(module, '_dsv41_rowstats_patched', False):
        return
    b = bindings or _vllm_bindings()
    module.configure_drafter = wrap_configure(module.configure_drafter, module=module, tp_rank=b['tp_rank'],
                                              zeros=b['zeros'], features=b['features'])
    module.record_top1 = wrap_record(module.record_top1, module=module, capturing=b['capturing'])
    module.publish = wrap_publish(module.publish, module=module)
    module.begin_target = wrap_begin(module.begin_target, module=module, capturing=b['capturing'], sync=b['sync'])
    module._dsv41_rowstats_patched = True


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != ROW_MASK:
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
    if ROW_MASK in sys.modules:
        patch(sys.modules[ROW_MASK])
    elif not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    logger.info('DSV41_ROWSTATS_INSTALLED pid=%d shadow=%d', os.getpid(), int(shadow_config()[0]))
