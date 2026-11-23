# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions Copyright 2026 Morrowmake; adapted for this packet.
"""Two fixed-shape Triton reductions; no data-dependent CPU synchronization.

Verbatim copy of lora-argmax-r1/overlay/dsv41_lora_argmax/kernels.py
(sha256 26e8ecea921bfb64c0bb1924f4dcf19aa111e611fc228fa36ecc824b024c3c90),
the target-argmax kernels live in today's stack, vendored so this layer does
not depend on that plugin being mounted.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _local(X, O, stride, width: tl.constexpr, start: tl.constexpr,
           vocab: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    valid = (col < width) & (start + col < vocab)
    x = tl.load(X + row * stride + col, valid, other=0).to(tl.float32)
    bits = x.to(tl.int32, bitcast=True)
    bits = tl.where(bits == -2147483648, 0, bits)
    nan = (bits & 0x7fffffff) > 0x7f800000
    wide = bits.to(tl.int64)
    key = tl.where(wide >= 0, wide + 2147483648, -1 - wide)
    key = tl.where(nan, 4294967295, key)
    key = tl.where(valid, key, -1)
    best = tl.max(key, 0)
    idx = tl.min(tl.where(valid & (key == best), col + start, vocab), 0)
    tl.store(O + row * 2, best)
    tl.store(O + row * 2 + 1, idx)


@triton.jit
def _pick(P, O, stride, tp: tl.constexpr, vocab: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, B)
    key = tl.load(P + row * stride + rank * 2, rank < tp, other=-1)
    idx = tl.load(P + row * stride + rank * 2 + 1, rank < tp, other=vocab)
    best = tl.max(key, 0)
    winner = tl.min(tl.where(key == best, idx, vocab), 0)
    tl.store(O + row, winner)


def local_pair_cuda(local, start, vocab):
    if local.stride(1) != 1 or not 1 <= local.shape[1] <= 65536:
        raise ValueError('CUDA shard width/stride unsupported')
    out = torch.empty((local.shape[0], 2), dtype=torch.int64, device=local.device)
    _local[(local.shape[0],)](local, out, local.stride(0), local.shape[1], start,
                            vocab, triton.next_power_of_2(local.shape[1]), num_warps=8)
    return out


def pick_pairs_cuda(pairs, vocab):
    tp = pairs.shape[1] // 2
    out = torch.empty(pairs.shape[0], dtype=torch.int64, device=pairs.device)
    _pick[(pairs.shape[0],)](pairs, out, pairs.stride(0), tp, vocab,
                           triton.next_power_of_2(tp), num_warps=1)
    return out
