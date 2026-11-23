# SPDX-License-Identifier: MIT
"""Import-time hook on the DSpark speculator; refuses anything it was not built for.

Only ``DSparkSpeculator._sample_sequential`` and ``load_draft_model`` are
wrapped. The sharded path is chosen per call by a pure function of
``num_reqs`` (and boot-time constants), so a CUDA graph captured for a
request count always replays the path it was captured with:

  sharded  <=>  eligible model  and  num_reqs >= MIN_REQS
               and not dsv41_row_mask.features_enabled(num_reqs)

Verification boots only (DSV41_DRAFT_ARGMAX_SHADOW=1): every sharded call first
runs the stock dense sampler on the same inputs and counts, on the device and
inside the captured graph, rows whose drafts or confidences differ bitwise. A
nonce written to DSV41_DRAFT_ARGMAX_DRAIN_FILE makes the next eager
``propose`` synchronize and log ``DSV41_DRAFT_ARGMAX_FINAL`` (drain.py).

The calibrated row mask (cap 1 in today's stack) needs full-row softmax
statistics at c1 (``record_top1``), so those request counts keep the stock
gather. Everything else (Markov head, confidence head, draft_tokens layout,
row-mask publish) is the stock code path.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import re
import sys

TARGET = 'vllm.v1.worker.gpu.spec_decode.dspark.speculator'
ROW_MASK = 'vllm.utils.dsv41_row_mask'
# The exact sources whose dense greedy branch shard.sample_sequential mirrors.
PINS = {
    TARGET: 'fa7df49bf473ef44c18c83d75d2c8ebe5adeaa43a76912a800521d16b5d31828',
    ROW_MASK: '50abe80da1499f61cbc43388564a709897180ffb6f6a951b34e952dabe861e2c',
}
SWITCH = 'DSV41_DRAFT_SHARD_ARGMAX'
MIN_REQS_ENV = 'DSV41_DRAFT_SHARD_ARGMAX_MIN_REQS'
SHADOW_ENV = 'DSV41_DRAFT_ARGMAX_SHADOW'
DRAIN_ENV = 'DSV41_DRAFT_ARGMAX_DRAIN_FILE'
DRAIN_FILE_RE = re.compile(r'/tmp/[A-Za-z0-9._-]{1,64}')
TOKEN_RE = re.compile(r'[A-Za-z0-9]{8,64}')
logger = logging.getLogger('vllm.dsv41_draft_argmax')


class Refused(RuntimeError):
    pass


def min_reqs():
    raw = os.environ.get(MIN_REQS_ENV, '2')
    if not raw.isdigit() or not 1 <= int(raw) <= 4096:
        raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED {MIN_REQS_ENV}={raw!r} must be an integer >= 1')
    return int(raw)


def shadow_config():
    """(shadow on, drain file or None) for a verification boot; refuses half settings."""
    raw = os.environ.get(SHADOW_ENV, '0')
    drain = os.environ.get(DRAIN_ENV, '') or None
    if raw not in ('0', '1'):
        raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED {SHADOW_ENV}={raw!r} must be 0 or 1')
    if drain is not None and (raw != '1' or not DRAIN_FILE_RE.fullmatch(drain) or '..' in drain):
        raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED {DRAIN_ENV} needs {SHADOW_ENV}=1 and a /tmp/<name> path')
    return raw == '1', drain


def read_token(path):
    try:
        with open(path) as f:
            token = f.read(80).strip()
    except OSError:
        return None
    return token if TOKEN_RE.fullmatch(token) else None


def source_sha(name):
    """sha256 of a module's source file, located without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED missing module {name}')
    with open(spec.origin, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_sources(sha=source_sha):
    for name, want in PINS.items():
        got = sha(name)
        if got != want:
            raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED {name} sha256 {got} != pinned {want}')


def ineligible(spec, model, tp_size, vllm_config):
    """Reason the sharded path cannot reproduce this speculator, or None."""
    head = getattr(model, 'lm_head', None)
    lp = getattr(model, 'logits_processor', None)
    parallel = getattr(vllm_config, 'parallel_config', None)
    checks = (
        (tp_size > 1, 'tensor parallel size 1'),
        (getattr(parallel, 'pipeline_parallel_size', 0) == 1, 'pipeline parallel'),
        (getattr(vllm_config, 'lora_config', 0) is None, 'LoRA configured'),
        (getattr(spec, 'draft_logits', 0) is None, 'probabilistic drafting (draft_logits cache)'),
        (getattr(spec, '_draft_topk', 0) is None, 'top-k candidate drafting'),
        (getattr(spec, 'distributed_draft', True) is False, 'VLLM_TP_DISTRIBUTED_SAMPLING draft path'),
        (getattr(spec, 'acceptance_estimator', 0) is None
         and not getattr(spec, 'use_acceptance_estimator', True), 'acceptance estimator reads draft logits'),
        (getattr(spec, 'draft_watermarker', 0) is None, 'draft watermarker'),
        (getattr(model, 'sample_sequential', None) is None, 'model-level sample_sequential'),
        (getattr(model, 'draft_id_to_target_id', None) is None, 'reduced draft vocabulary'),
        (hasattr(model, 'compute_draft_logits_local'), 'no compute_draft_logits_local'),
        (head is not None and getattr(head, 'tp_size', None) == tp_size, 'draft head TP differs'),
        (head is not None and getattr(head, 'num_added_embeddings', 1) == 0, 'added vocabulary'),
        (head is not None and getattr(head, 'org_vocab_size', None) == getattr(spec, 'vocab_size', None),
         'draft head vocabulary differs'),
        (lp is not None and getattr(lp, 'org_vocab_size', None) == getattr(spec, 'vocab_size', None),
         'draft/target vocabulary size differ'),
        (lp is not None and getattr(lp, 'soft_cap', 1) is None and getattr(lp, 'scale', 0) == 1.0,
         'logits scale/soft cap'),
        (getattr(spec, 'vocab_size', 1 << 31) < (1 << 31), 'vocabulary too large'),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    return None


class State:
    def __init__(self, rank, minimum, features_enabled, confidence_enabled, counters=None, drain_file=None):
        self.rank = rank
        self.min_reqs = minimum
        self.features_enabled = features_enabled
        self.confidence_enabled = confidence_enabled
        self.announced = set()
        # Shadow mode: int64 [2, max_num_reqs + 1, 3] = (rows compared, token mismatches,
        # confidence mismatches) per request count; [0] eager calls (warmup, uncaptured
        # sizes), [1] CUDA-graph replays only, i.e. served traffic. Allocated before capture.
        self.counters = counters
        self.drain_file = drain_file
        self.drained = None

    def use_for(self, num_reqs):
        return num_reqs >= self.min_reqs and not self.features_enabled(num_reqs)


def wrap_load(original, *, tp_info, row_mask, prewarm, zeros):
    @functools.wraps(original)
    def load(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        rank, size = tp_info()
        reason = ineligible(self, model, size, self.vllm_config)
        if reason is not None:
            # The layer was requested explicitly: an inert arm would waste a screen.
            raise Refused(f'DSV41_DRAFT_ARGMAX_REFUSED rank={rank} {reason}')
        shadow, drain = shadow_config()
        counters = zeros(2, self.max_num_reqs + 1, 3) if shadow else None
        st = State(rank, min_reqs(), row_mask.features_enabled, row_mask.confidence_enabled, counters, drain)
        head = model.lm_head
        # JIT both Triton kernels for this rank's specialization now, never inside a graph capture.
        prewarm(head.num_embeddings_per_partition, head.shard_indices.org_vocab_start_index,
                model.logits_processor.org_vocab_size, size)
        self._dsv41_draft_argmax = st
        logger.info('DSV41_DRAFT_ARGMAX_READY rank=%d min_reqs=%d vocab=%d start=%d width=%d shadow=%d',
                    rank, st.min_reqs, model.logits_processor.org_vocab_size,
                    head.shard_indices.org_vocab_start_index,
                    head.num_embeddings_per_partition, int(shadow))
        return model
    return load


def wrap_sample_sequential(original, *, all_gather, capturing):
    from . import shard

    @functools.wraps(original)
    def run(self, num_reqs, head_hidden):
        st = getattr(self, '_dsv41_draft_argmax', None)
        if st is None or not st.use_for(num_reqs):
            return original(self, num_reqs, head_hidden)
        key = (num_reqs, bool(capturing()))
        if key not in st.announced:  # once per count, separately for warmup/eager and graph capture
            st.announced.add(key)
            logger.info('DSV41_DRAFT_ARGMAX_ACTIVE rank=%d num_reqs=%d capturing=%d',
                        st.rank, num_reqs, int(key[1]))
        use_confidence = self.use_confidence_head and st.confidence_enabled(num_reqs)
        if st.counters is None:
            return shard.sample_sequential(self, num_reqs, head_hidden, all_gather=all_gather,
                                           use_confidence=use_confidence)
        # Shadow: stock dense sampler first, on the same inputs, then the sharded one over it.
        original(self, num_reqs, head_hidden)
        ref_tokens = self.draft_tokens[:num_reqs].clone()
        ref_conf = self.draft_token_confidence_probs[:num_reqs].clone() if use_confidence else None
        shard.sample_sequential(self, num_reqs, head_hidden, all_gather=all_gather, use_confidence=use_confidence)
        bad_tokens = (self.draft_tokens[:num_reqs] != ref_tokens).sum()
        if use_confidence:  # bitwise: compare the raw bits, so NaN == NaN and -0.0 != +0.0
            bad_conf = (_bits(self.draft_token_confidence_probs[:num_reqs]) != _bits(ref_conf)).sum()
        else:
            bad_conf = bad_tokens.new_zeros(())
        rows = bad_tokens.new_full((), num_reqs * self.num_speculative_steps)
        # Recorded into the graph when capturing, so the [1] rows grow only on replay.
        st.counters[int(key[1]), num_reqs].add_(_stack((rows, bad_tokens.to(rows.dtype), bad_conf.to(rows.dtype))))
    return run


def _bits(t):
    import torch
    return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _stack(values):
    import torch
    return torch.stack(values)


def wrap_propose(original, *, capturing, sync):
    """Verification drain: a new nonce in the drain file -> device sync -> one FINAL line."""
    @functools.wraps(original)
    def propose(self, *args, **kwargs):
        st = getattr(self, '_dsv41_draft_argmax', None)
        if st is not None and st.drain_file and st.counters is not None and not capturing():
            token = read_token(st.drain_file)
            if token is not None and token != st.drained:
                st.drained = token
                sync()
                eager, graph = st.counters.cpu().tolist()
                counts = ','.join(f'{n}:{r}:{t}:{c}' for n, (r, t, c) in enumerate(graph) if r)
                e = [sum(row[j] for row in eager) for j in range(3)]
                logger.info('DSV41_DRAFT_ARGMAX_FINAL rank=%d token=%s rows=%d bad_tokens=%d bad_conf=%d '
                            'counts=%s eager=%d:%d:%d',
                            st.rank, token, sum(r[0] for r in graph), sum(r[1] for r in graph) + e[1],
                            sum(r[2] for r in graph) + e[2], counts or '-', *e)
        return original(self, *args, **kwargs)
    return propose


def _vllm_bindings():
    import torch
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)
    import vllm.utils.dsv41_row_mask as row_mask

    def prewarm(width, start, vocab, tp):
        from .kernels import local_pair_cuda, pick_pairs_cuda
        dev = torch.device('cuda', torch.cuda.current_device())
        n = min(width, max(0, vocab - start))  # the width shard.sample_sequential passes
        local_pair_cuda(torch.zeros(1, n, dtype=torch.bfloat16, device=dev), start, vocab)
        pick_pairs_cuda(torch.zeros(1, 2 * tp, dtype=torch.int64, device=dev), vocab)
        torch.cuda.synchronize(dev)
    return dict(
        prewarm=prewarm,
        zeros=lambda *shape: torch.zeros(*shape, dtype=torch.int64,
                                         device=torch.device('cuda', torch.cuda.current_device())),
        sync=torch.cuda.synchronize,
        tp_info=lambda: (get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()),
        all_gather=lambda pair: tensor_model_parallel_all_gather(pair, dim=1),
        capturing=torch.cuda.is_current_stream_capturing,
        row_mask=row_mask,
    )


def patch(module, bindings=None):
    if getattr(module, '_dsv41_draft_argmax_patched', False):
        return
    b = bindings or _vllm_bindings()
    cls = module.DSparkSpeculator
    cls.load_draft_model = wrap_load(cls.load_draft_model, tp_info=b['tp_info'], row_mask=b['row_mask'],
                                     prewarm=b['prewarm'], zeros=b['zeros'])
    cls.propose = wrap_propose(cls.propose, capturing=b['capturing'], sync=b['sync'])
    cls._sample_sequential = wrap_sample_sequential(
        cls._sample_sequential, all_gather=b['all_gather'], capturing=b['capturing'])
    module._dsv41_draft_argmax_patched = True


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
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
    min_reqs()
    shadow_config()
    check_sources()
    if TARGET in sys.modules:
        patch(sys.modules[TARGET])
    elif not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    logger.info('DSV41_DRAFT_ARGMAX_INSTALLED pid=%d min_reqs=%d', os.getpid(), min_reqs())
