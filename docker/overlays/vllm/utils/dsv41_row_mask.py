# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated row-mask bridge for our K5 engine; see NOTICE and RESULTS.md."""
import os
from contextlib import contextmanager
import torch
from dsv41_depth import policy as calibrated

SWITCH = "VLLM_DSV41_MOE_ROW_MASK"
_STATE = None
_TARGET_ENABLED = True
_CAPTURE_KEY = None
_GRAPH_AUDIT = {}


def features_enabled(num_reqs):
    """Scalar shape decision; capture uses the descriptor's padded requests."""
    state = _STATE
    return state is not None and (not state.row_max_batch or num_reqs <= state.row_max_batch)


def confidence_enabled(num_reqs):
    # Preserve native confidence use when this entire experimental policy is off.
    return _STATE is None or features_enabled(num_reqs)


def graph_compatible(desc, num_reqs):
    """Never pad a live masked batch into an unmasked capture, or vice versa."""
    if _STATE is None or not _STATE.row_max_batch:
        return True
    return desc.num_reqs is not None and features_enabled(desc.num_reqs) == features_enabled(num_reqs)


def target_enabled(layer):
    # This Python decision executes at capture time, never at FULL replay.
    return _TARGET_ENABLED and getattr(layer, "_dsv41_mask_target", False)


@contextmanager
def capture_scope(manager, desc):
    global _TARGET_ENABLED, _CAPTURE_KEY
    previous = _TARGET_ENABLED, _CAPTURE_KEY
    if _STATE is not None and _STATE.row_max_batch:
        if desc.num_reqs is None or desc.cg_mode.name != "FULL" or desc.num_ubatches != 1:
            raise ValueError("off-above-cap requires FULL request-keyed graphs without ubatching")
        _TARGET_ENABLED = features_enabled(desc.num_reqs)
        _CAPTURE_KEY = (id(manager), desc)
        _GRAPH_AUDIT[_CAPTURE_KEY] = {"routes": 0, "draft": None}
    try:
        yield
    finally:
        _TARGET_ENABLED, _CAPTURE_KEY = previous


def capture_forward(manager, desc, forward):
    """Scope both warmup and real capture to an immutable graph descriptor."""
    def run(mode):
        with capture_scope(manager, desc):
            return forward(mode)
    return run


def draft_features(num_reqs):
    enabled = features_enabled(num_reqs)
    if _CAPTURE_KEY is not None:
        desc = _CAPTURE_KEY[1]
        if desc.num_reqs != num_reqs:
            raise ValueError("drafter request count differs from capture descriptor")
        _GRAPH_AUDIT[_CAPTURE_KEY]["draft"] = enabled
    return enabled


def check_off_above_cap(runner, profile_only=False):
    """Fail before readiness if target/draft captures do not prove exclusion.

    Inspect actual captures, not just planned shapes. Profiling intentionally
    captures only a sample and must not emit a qualification banner.
    """
    state = _STATE
    if state is None or not state.row_max_batch or profile_only:
        return
    from vllm.config.compilation import CUDAGraphMode
    target = runner.cudagraph_manager
    draft = runner.speculator
    managers = [value for value in vars(draft).values()
                if hasattr(value, "graphs") and hasattr(value, "dispatch")]
    if not target.graphs or not managers:
        raise ValueError("off-above-cap needs actual target and draft FULL captures")
    for manager, role in [(target, "target")] + [(m, "draft") for m in managers]:
        if not manager.graphs:
            raise ValueError("off-above-cap found an uncaptured draft manager")
        for desc in manager.graphs:
            record = _GRAPH_AUDIT.get((id(manager), desc))
            if record is None:
                raise ValueError("off-above-cap capture audit missing")
            admitted = features_enabled(desc.num_reqs)
            if role == "target" and bool(record["routes"]) != admitted:
                raise ValueError("off-above-cap target routes differ from descriptor")
            if role == "draft" and record["draft"] != admitted:
                raise ValueError("off-above-cap draft features differ from descriptor")
        widths = (2, 3, 4, 6) if role == "target" else (draft.num_query_per_req,)
        for n in range(1, min(8, state.max_num_reqs) + 1):
            for width in widths:
                # DFlash variants use uniform or varlen dispatch. Check both;
                # every selected FULL graph must have the same cap decision.
                found = False
                for uniform in (width, None):
                    desc = manager.dispatch(n, n * width, uniform, 0)
                    if desc.cg_mode == CUDAGraphMode.FULL:
                        found = True
                        if desc not in manager.graphs or not graph_compatible(desc, n):
                            raise ValueError("off-above-cap dispatch selects wrong graph")
                if not found:
                    raise ValueError("off-above-cap graph coverage missing")
    from vllm.logger import init_logger
    from vllm.distributed import get_tp_group
    init_logger("vllm.utils.dsv41_row_mask").info(
        "DSV41 off above cap verified: row_max_batch=%d rank=%d target=excluded draft=excluded reset=excluded",
        state.row_max_batch, get_tp_group().rank_in_group)


def get_state():
    return _STATE


def mark_target(moe_runner, is_target):
    if _STATE is None:
        return
    if getattr(getattr(moe_runner, "moe_config", None), "is_sequence_parallel", False):
        raise ValueError("row-mask does not support sequence-parallel token rows")
    # FusedMoEFactory returns a runner; quant_method.apply receives its
    # RoutedExperts child, not that runner. Mark exactly the apply owner.
    routed = getattr(moe_runner, "routed_experts", None)
    if routed is None:
        raise ValueError("row-mask requires the RoutedExperts MoE runner contract")
    routed._dsv41_mask_target = is_target


def init_worker(runner):
    global _STATE
    path = os.environ.get(SWITCH, "")
    if not path:
        return
    if _STATE is not None:
        raise ValueError("row-mask worker initialized twice")
    c = runner.vllm_config
    p, spec = c.parallel_config, c.speculative_config
    if (spec is None or not spec.use_dspark() or spec.draft_sample_method != "greedy"
            or getattr(spec, "rejection_sample_method", "standard") != "standard"
            or p.tensor_parallel_size != 4 or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1 or p.enable_expert_parallel or p.use_ubatching
            or getattr(p, "decode_context_parallel_size", 1) != 1
            or getattr(p, "prefill_context_parallel_size", 1) != 1
            or getattr(p, "enable_batch_sharded_sampling", False)
            or c.quant_config is None or c.quant_config.get_name() != "exl3"
            or c.model_config.hf_config.hidden_size != 5120):
        raise ValueError("row-mask requires DSpark greedy drafts/standard rejection, EXL3 TP4/PP1/DP1, no EP/CP/ubatching/batch-sharded sampling")
    if os.environ.get("VLLM_DSV41_PREFILL_OVERLAP", "0") != "0" or os.environ.get("DSV41_PREFILL_OVERLAP", "0") != "0":
        raise ValueError("row-mask prefill overlap is not qualified")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") != "0":
        raise ValueError("row-mask batch-invariant mode is not qualified")
    if runner.num_speculative_steps != 5:
        raise ValueError("calibrated port is qualified only for our K5 drafter")
    _STATE = calibrated.init_row_mask_state(
        max_num_reqs=runner.max_num_reqs, max_num_tokens=runner.max_num_tokens,
        max_draft=runner.num_speculative_steps, device=runner.device,
        verify_drafts=[0] + [runner.num_speculative_steps] * runner.max_num_reqs)
    from vllm.logger import init_logger
    init_logger(__name__).info("DSV41 calibrated row mask on: K5 confidence+top1 host=%s row_max_batch=%d",
                               os.environ.get("DSV41_DEPTH_HOST", "0"), _STATE.row_max_batch)


def reset_slot(slot):
    if _STATE is not None:
        # A previous over-cap draft invalidated all features. Defer the device
        # writes to re-entry, keeping request admission above the cap free too.
        if _STATE.row_max_batch:
            _STATE.pending_slots.add(slot)
            return
        _STATE.conf_by_slot[slot].fill_(1.)
        _STATE.stats_by_slot[slot].zero_()


def configure_drafter(speculator, model):
    if _STATE is None:
        return
    if speculator._draft_topk is not None or speculator.distributed_draft:
        raise ValueError("calibrated features require full-vocabulary greedy DSpark logits")
    if speculator.enable_adaptive_verification:
        raise ValueError("device adaptive verification is not supported; host AV is separate")
    if model.model.confidence_head is None:
        raise ValueError("checkpoint confidence head is required")
    # The custom sampler can bypass _sample_logits and leave top1 unwritten.
    if getattr(model, "sample_sequential", None) is not None:
        raise ValueError("custom sequential draft sampler is not qualified for calibrated features")
    speculator.use_confidence_head = True
    speculator._dsv41_row_stats = torch.zeros(
        speculator.max_num_reqs, 2, _STATE.max_draft,
        dtype=torch.float32, device=speculator.device)


def record_top1(speculator, logits, step):
    if not features_enabled(logits.shape[0]):
        return
    # Exactly the upstream statistics: full logits, Markov bias included,
    # before temperature. Entropy is carried even though its shipped weight is 0.
    x = logits.float()
    n = x.shape[0]
    out = speculator._dsv41_row_stats
    out[:n, 0, step] = torch.exp(x.amax(dim=-1) - torch.logsumexp(x, dim=-1))
    probs = torch.softmax(x, dim=-1)
    out[:n, 1, step] = -torch.special.xlogy(probs, probs).sum(dim=-1)


def publish(speculator, num_reqs):
    state = _STATE
    if not features_enabled(num_reqs):
        return
    conf = speculator.draft_token_confidence_probs
    stats = speculator._dsv41_row_stats
    width = speculator.num_speculative_steps
    if conf.is_cuda:
        from dsv41_depth.confidence import publish_confidences
        publish_confidences(conf, speculator.sample_idx_mapping, num_reqs, width,
                            state.conf_by_slot, stats, state.stats_by_slot)
    else:
        for r in range(num_reqs):
            slot = int(speculator.sample_idx_mapping[r * width])
            if 0 <= slot < state.max_num_reqs:
                state.conf_by_slot[slot].zero_()
                state.stats_by_slot[slot].zero_()
                state.conf_by_slot[slot, :width] = conf[r, :width]
                state.stats_by_slot[slot, :, :width] = stats[r, :, :width]


def begin_target(batch, dummy_run):
    global _TARGET_ENABLED
    if _STATE is None:
        return
    # Eager path only. FULL captures set this independently from their desc.
    _TARGET_ENABLED = features_enabled(batch.num_reqs)
    from dsv41_depth.decision import begin_step
    begin_step(_STATE, batch, dummy_run)


def mask_routes(ids, weights, layer):
    state = _STATE
    if state is None or not target_enabled(layer):
        return ids, weights
    if _CAPTURE_KEY is not None:
        _GRAPH_AUDIT[_CAPTURE_KEY]["routes"] += 1
    rows = ids.shape[0]
    if rows > state.max_num_tokens:
        raise ValueError("row-mask buffer too small")
    # Do not branch on state.active here: captured graphs must always read
    # the fixed row_live address, even when captured with every row live.
    # Capture-time selection has already excluded over-cap descriptors.
    if ids.is_cuda:
        from vllm.utils.dsv41_row_mask_kernels import mask_routes_cuda
        return mask_routes_cuda(state, ids, weights)
    dead = state.row_live[:rows, None] == 0
    return ids.masked_fill(dead, -1), weights.masked_fill(dead, 0)


def cap_drafts(logits_indices, invalid_drafts):
    if _STATE is None or not _STATE.active:
        return invalid_drafts
    masked = _STATE.row_live.index_select(0, logits_indices).eq(0)
    return masked if invalid_drafts is None else masked.logical_or_(invalid_drafts)


def after_propose(batch, num_drafts):
    # A zero-draft proposal does not publish a new row. Clear its slots so a
    # later verify cannot read a feature vector from an older round.
    if _STATE is not None and not features_enabled(batch.num_reqs):
        _STATE.features_stale = True
        return
    if _STATE is not None and num_drafts == 0:
        slots = batch.idx_mapping[:batch.num_reqs].long()
        _STATE.conf_by_slot.index_fill_(0, slots, 1.)
        _STATE.stats_by_slot.index_fill_(0, slots, 0.)
