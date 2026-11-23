# SPDX-License-Identifier: AGPL-3.0-only
# Allocation and arguments follow vllm-exl3's apply_exl3_fused_moe coop branch (moe-gap-r2 stage-r18
# exl3.py sha256 8468e2ec..., lines 2555-2656); the one-launch idea is GNmac-lab bbaf367's coop prep.
"""The fused coop dispatch prep: what the coop kernel reads, built by one launch.

Stock (stage-r18 ``apply_exl3_fused_moe``, coop-eligible decode rows) builds, before the coop call:

    local  = map_topk_to_local(ids, n_exp, None)      # where(ids<0 | ids>=n_exp, n_exp, ids)
    sel_c  = local.reshape(tokens, topk).to(int64)     # coop arg 1
    rw_c   = weights.reshape(-1).to(float16).reshape(tokens, topk)   # coop arg 2
    xh     = x2d.contiguous().half()                   # coop arg 0
    out    = torch.zeros(tokens, hidden, float32)      # coop arg 33 (returned)
    ctr    = torch.zeros(n_ctr, int32)                 # coop arg 32
    + flat_token (arange/repeat_interleave), expert_count (zeros/ones/scatter_add), counts > fat,
      fat_route zeros: computed and never read on the coop branch.

``fused_prep`` produces the same five tensors (same shapes, dtypes and contiguous strides) with one
Triton launch; ``emulate`` is the same per-element computation in torch, used where Triton is absent
(CPU tests). Each element is either copied, mapped through the same integer predicate, cast once with
round-to-nearest-even (bf16 -> fp32 is exact, so bf16 -> fp16 has one rounding on both sides), or a
constant zero.
"""

COOP_MAX_ROUTES = 256   # stage-r18: tokens * topk <= 256 (coop slot scratch)


def counter_len(tokens, topk, hidden, inter_dim):
    """The coop counter length, the stock expression."""
    smax = 4 * tokens * topk
    return smax * (inter_dim // 128) + tokens * (hidden // 128) + 2 * smax + 3


def allocate(x2d, topk, n_ctr):
    import torch
    tokens, hidden = x2d.shape
    dev = x2d.device
    return dict(
        xh=torch.empty((tokens, hidden), dtype=torch.float16, device=dev),
        out=torch.empty((tokens, hidden), dtype=torch.float32, device=dev),
        sel=torch.empty((tokens, topk), dtype=torch.int64, device=dev),
        rw=torch.empty((tokens, topk), dtype=torch.float16, device=dev),
        ctr=torch.empty(n_ctr, dtype=torch.int32, device=dev),
    )


def fused_prep(x2d, ids, weights, n_exp, n_ctr):
    """(sel, rw, xh, out, ctr) from one Triton launch. Contiguous, 16-byte aligned inputs only."""
    from .kernel import launch
    t = allocate(x2d, ids.shape[-1], n_ctr)
    launch(x2d, ids, weights, t['xh'], t['out'], t['sel'], t['rw'], t['ctr'], int(n_exp))
    return t['sel'], t['rw'], t['xh'], t['out'], t['ctr']


def _to_half(t):
    """Triton's ``.to(tl.float16)``: a no-op on fp16 (bits kept), bf16 via fp32 (exact), one RNE rounding."""
    import torch
    if t.dtype == torch.float16:
        return t
    return t.to(torch.float32).to(torch.float16)


def emulate(x2d, ids, weights, n_exp, n_ctr):
    """The kernel's per-element computation in torch (no Triton): the CPU-test model of fused_prep."""
    import torch
    t = allocate(x2d, ids.shape[-1], n_ctr)
    t['xh'].copy_(_to_half(x2d.reshape(-1)).view(x2d.shape))
    t['out'].zero_()
    i = ids.reshape(-1).to(torch.int64)
    t['sel'].view(-1).copy_(torch.where((i < 0) | (i >= n_exp), torch.full_like(i, n_exp), i))
    t['rw'].view(-1).copy_(_to_half(weights.reshape(-1)))
    t['ctr'].zero_()
    return t['sel'], t['rw'], t['xh'], t['out'], t['ctr']


def prewarm(device, x_dtype, ids_dtype, w_dtype, prep=fused_prep):
    """Compile/load the kernel for one dtype key before any graph capture (one binary for all sizes)."""
    import torch
    x = torch.zeros((1, 128), dtype=x_dtype, device=device)
    ids = torch.zeros((1, 6), dtype=ids_dtype, device=device)
    w = torch.zeros((1, 6), dtype=w_dtype, device=device)
    prep(x, ids, w, 1, counter_len(1, 6, 128, 128))
    if x.is_cuda:
        torch.cuda.synchronize(device)
