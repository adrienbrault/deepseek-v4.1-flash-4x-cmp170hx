# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions Copyright 2026 Morrowmake; adapted for this packet.
"""Greedy DSpark Markov drafting over the vocabulary-parallel head, exactly.

Stock (gathered) path, per draft block of ``num_reqs * K`` rows:
  base = all_gather(lm_head_local(norm(h)))          # [num_reqs*K, V] bf16
  for i in 0..K-1: tok_i = argmax(base[:, i] + markov_bias(tok_{i-1}))
The gather is the NCCL ``AllGather_RING_LL`` inside the FULL draft graph.

Sharded path (this module), same block:
  local = lm_head_local(norm(h))                     # [num_reqs*K, V/TP]
  for i: x = local[:, i] + markov_bias(tok_{i-1})[:, own columns]
         pair = (ordering key, global index) of x's first maximum   # int64 [rows, 2]
         tok_i = pick(all_gather(pair))              # [rows, 2*TP] int64
Per column the addition is the same tensor op on the same values (the Markov
head is replicated, ``disable_tp=True``), so every x element equals the stock
row's element bit for bit. The key orders values as torch.argmax does (NaN
above everything, -0.0 == +0.0, first index on ties; integer comparisons only)
and shards hold increasing global columns, so the lowest index among equal
best keys is the stock first-maximum. The collective moves int64s: exact.
"""
import torch


def keys(x):
    """IEEE-order int64 key of FP16/BF16/FP32 values (conversion to FP32 is exact)."""
    bits = x.float().contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits == -2147483648, 0, bits)
    nan = (bits & 0x7fffffff) > 0x7f800000
    key = torch.where(bits >= 0, bits + 2147483648, -1 - bits)
    return torch.where(nan, 4294967295, key)


def local_pair(local, start, vocab):
    """[rows, 2] int64 (best key, global index of its first occurrence)."""
    rows, width = local.shape
    if local.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('argmax requires FP16/BF16/FP32 logits')
    if local.is_cuda:
        from .kernels import local_pair_cuda
        return local_pair_cuda(local, start, vocab)
    n = min(width, max(0, vocab - start))
    if n == 0:
        return torch.tensor([-1, vocab], dtype=torch.int64).repeat(rows, 1)
    best, i = keys(local[:, :n]).max(dim=1)  # first occurrence on ties
    return torch.stack((best, i + start), dim=1)


def pick_pairs(pairs, vocab):
    """pairs: [rows, 2*TP] rank-major (key, index) columns -> [rows] int64."""
    if pairs.is_cuda:
        from .kernels import pick_pairs_cuda
        return pick_pairs_cuda(pairs, vocab)
    k, i = pairs[:, 0::2], pairs[:, 1::2]
    best = k.max(dim=1, keepdim=True).values
    return torch.where(k == best, i, vocab).min(dim=1).values


def shard_tokens(local, start, vocab, all_gather):
    """Global first-argmax token per row of a vocabulary shard."""
    if local.ndim != 2 or start < 0 or vocab < 1:
        raise ValueError('invalid vocabulary geometry')
    if local.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=local.device)
    return pick_pairs(all_gather(local_pair(local, start, vocab)), vocab)


def sample_sequential(spec, num_reqs, head_hidden, *, all_gather, use_confidence):
    """Sharded twin of DSparkSpeculator._sample_sequential's greedy dense branch.

    Mirrors moe-gap-r2 stage-r18 speculator.py (sha256 fa7df49b..., pinned in
    hooks.py) with ``distributed_draft`` False and ``draft_logits`` None, except
    that the base logits stay sharded and each step's argmax is exchanged as
    pairs. ``use_confidence`` is the caller's
    ``use_confidence_head and confidence_enabled(num_reqs)``.
    """
    model = spec.model
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    local = model.compute_draft_logits_local(sample_hidden)
    width = local.shape[-1]
    local = local.view(num_reqs, n_spec, width)
    head = model.lm_head
    start = head.shard_indices.org_vocab_start_index
    vocab = model.logits_processor.org_vocab_size
    n = min(width, max(0, vocab - start))  # padding columns are never read
    confidence_markov_embeds = []
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        markov_embed = model.markov_embed(prev)
        if use_confidence:
            confidence_markov_embeds.append(markov_embed)
        bias = model.markov_bias(markov_embed)
        logits_i = local[:, i, :n] + bias[:, start:start + n]
        draft_sampled_i = model.map_draft_to_target(
            shard_tokens(logits_i, start, vocab, all_gather))
        spec.draft_tokens[:num_reqs, i] = draft_sampled_i
        prev = draft_sampled_i
    if use_confidence:
        confidence = model.compute_confidence(
            sample_hidden,
            torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
        )
        spec.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)
