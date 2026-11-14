# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions Copyright 2026 Morrowmake; adapted for this packet.
"""Exact first-index argmax; one integer (value-key, global index) per rank.

The IEEE ordering key preserves NaNs, signed zero and subnormals without any
floating comparison. Integer collectives avoid encoding token IDs in BF16.
Only FP16/BF16/FP32 logits are accepted (conversion to FP32 is exact).
"""
import torch


def keys(x):
    bits = x.float().contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits == -2147483648, 0, bits)
    nan = (bits & 0x7fffffff) > 0x7f800000
    key = torch.where(bits >= 0, bits + 2147483648, -1 - bits)
    return torch.where(nan, 4294967295, key)


def local_pair(local, start, vocab):
    rows, width = local.shape
    n = min(width, max(0, vocab - start))
    if local.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('argmax requires FP16/BF16/FP32 logits')
    if local.is_cuda:
        from .kernels import local_pair_cuda
        return local_pair_cuda(local, start, vocab)
    if n == 0:
        return torch.tensor([-1, vocab], dtype=torch.int64).repeat(rows, 1)
    k = keys(local[:, :n])
    best, i = k.max(dim=1)  # first occurrence on ties
    return torch.stack((best, i + start), dim=1)


def pick_pairs(pairs, vocab):
    """pairs: [rows, 2*TP], rank-major (key, index) columns."""
    if pairs.is_cuda:
        from .kernels import pick_pairs_cuda
        return pick_pairs_cuda(pairs, vocab)
    k, i = pairs[:, 0::2], pairs[:, 1::2]
    best = k.max(dim=1, keepdim=True).values
    return torch.where(k == best, i, vocab).min(dim=1).values


def shard_tokens(local, start, vocab, all_gather):
    if local.ndim != 2 or not 0 <= start or vocab < 1:
        raise ValueError('invalid vocabulary geometry')
    if local.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=local.device)
    return pick_pairs(all_gather(local_pair(local, start, vocab)), vocab)


def argmax_rows(model, hidden):
    from vllm.distributed import tensor_model_parallel_all_gather
    # Called on the inner text class only; VL forwards here exactly once.
    head = model.lm_head
    local = model.compute_logits_local(hidden)
    vocab = model.logits_processor.org_vocab_size
    start = head.shard_indices.org_vocab_start_index
    tokens = shard_tokens(local, start, vocab,
                          lambda pair: tensor_model_parallel_all_gather(pair, dim=1))
    # Preserve the stock sampler/rejection/AV code and its row layout. No full
    # logits are exchanged. The dense synthetic rows still cost local writes.
    out = local.new_full((local.shape[0], vocab), float('-inf'))
    return out.scatter_(1, tokens[:, None], 0.0)
