# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse metadata/masking around the SAME torch.topk, including tie choices.

Adaptation of the selection-pipeline fusion idea, not a transplant of their
canonical radix selector: ours keeps score order and PyTorch's tie selection.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mask(logits, lens, masked, WIDTH: tl.constexpr, N: tl.constexpr,
          LS: tl.constexpr, ROW_LENS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    if ROW_LENS:
        end = tl.load(lens + row)
    else:
        end = tl.maximum(tl.load(lens + row // N) - N + 1 + row % N, 0)
    value = tl.load(logits + row * LS + col, col < WIDTH, other=float('-inf'))
    value = tl.where(col >= end, float('-inf'), value)
    tl.store(masked + row * WIDTH + col, value, col < WIDTH)


@triton.jit
def _store(columns, lens, out, K: tl.constexpr, N: tl.constexpr,
           CS: tl.constexpr, OS: tl.constexpr, ROW_LENS: tl.constexpr,
           B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    if ROW_LENS:
        end = tl.load(lens + row)
    else:
        end = tl.maximum(tl.load(lens + row // N) - N + 1 + row % N, 0)
    col = tl.load(columns + row * CS + i, i < K, other=-1)
    tl.store(out + row * OS + i, tl.where(i < end, col, -1).to(tl.int32), i < K)


def supported(logits, seq_lens, next_n, out, k):
    return (logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2
            and 0 < logits.shape[0] <= 48 and logits.stride(1) == 1
            and type(next_n) is int and 0 < next_n <= 8
            and seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
            and seq_lens.device == logits.device
            and (seq_lens.numel() == logits.shape[0]
                 or seq_lens.numel() * next_n == logits.shape[0])
            and 0 < k <= 2048 and k <= logits.shape[1]
            and out.shape == (logits.shape[0], k) and out.dtype == torch.int32
            and out.stride(1) == 1 and out.device == logits.device)


def run(logits, seq_lens, next_n, out, k):
    rows, width = logits.shape
    masked = torch.empty((rows, width), dtype=logits.dtype, device=logits.device)
    per_row = seq_lens.numel() == rows
    _mask[(rows, triton.cdiv(width, 256))](
        logits, seq_lens, masked, width, next_n, logits.stride(0), per_row, 256)
    # Identical shape, dtype, values, sorted=True and API as the reference.
    columns = masked.topk(k, dim=-1).indices
    _store[(rows,)](columns, seq_lens, out, k, next_n,
                   columns.stride(0), out.stride(0), per_row,
                   triton.next_power_of_2(k), num_warps=4)
