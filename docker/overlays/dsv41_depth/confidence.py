# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot publication adapted from Morrowmake v1.0.2; see NOTICE."""
import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _publish_kernel(
    conf_ptr,
    conf_stride,
    stats_ptr,
    stats_stride,
    slot_ptr,
    slot_stride,
    out_ptr,
    out_stride,
    stats_out_ptr,
    stats_out_stride,
    num_slots,
    width,
    max_draft,
    stats_width,
    STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    slot = tl.load(slot_ptr + req * slot_stride)
    if (slot >= 0) & (slot < num_slots):
        cols = tl.arange(0, BLOCK)
        conf = tl.load(conf_ptr + req * conf_stride + cols, mask=cols < width, other=0.0)
        tl.store(out_ptr + slot * out_stride + cols, conf, mask=cols < max_draft)
        if STATS:
            src = stats_ptr + req * stats_stride + cols
            dst = stats_out_ptr + slot * stats_out_stride + cols
            top1 = tl.load(src, mask=cols < width, other=0.0)
            ent = tl.load(src + stats_width, mask=cols < width, other=0.0)
            tl.store(dst, top1, mask=cols < max_draft)
            tl.store(dst + max_draft, ent, mask=cols < max_draft)


def publish_confidences(
    confidences: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    num_reqs: int,
    width: int,
    conf_by_slot: torch.Tensor,
    stats: torch.Tensor | None = None,
    stats_by_slot: torch.Tensor | None = None,
) -> None:
    """Write ``confidences[r, :width]`` (zeros past it) into
    ``conf_by_slot[slot_r]`` for each request r < ``num_reqs`` whose slot
    ``slot_r = sample_idx_mapping[r * width]`` is not -1; with ``stats``
    ([>= num_reqs, 2, >= width], contiguous last two dims) likewise
    ``stats[r, :, :width]`` into ``stats_by_slot[slot_r]``.

    ``confidences``: [>= num_reqs, >= width] float32, unit column stride;
    ``conf_by_slot``: [num_slots, max_draft] float32, contiguous;
    ``stats_by_slot``: [num_slots, 2, max_draft] float32, contiguous."""
    if num_reqs == 0:
        return
    max_draft = conf_by_slot.shape[1]
    assert confidences.stride(1) == 1 and conf_by_slot.stride(1) == 1
    assert width <= max_draft
    use_stats = stats is not None
    if use_stats:
        assert stats_by_slot is not None and stats_by_slot.shape[1:] == (2, max_draft)
        assert stats.stride(2) == 1 and stats.stride(1) == stats.shape[2]
        assert stats.shape[2] >= width
    _publish_kernel[(num_reqs,)](
        confidences,
        confidences.stride(0),
        stats if use_stats else confidences,
        stats.stride(0) if use_stats else 0,
        sample_idx_mapping,
        width,
        conf_by_slot,
        conf_by_slot.stride(0),
        stats_by_slot if use_stats else conf_by_slot,
        stats_by_slot.stride(0) if use_stats else 0,
        conf_by_slot.shape[0],
        width,
        max_draft,
        stats.shape[2] if use_stats else 0,
        STATS=use_stats,
        BLOCK=triton.next_power_of_2(max_draft),
    )
