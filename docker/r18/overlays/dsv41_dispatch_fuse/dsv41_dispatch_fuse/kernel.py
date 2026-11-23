# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from GNmac-lab/deepseek-v41-flash-1017-cmp170hx bbaf3672914eb1bbbca11de685be5f0c150f577a
# (patches/vllm-exl3/patch_exl3_coop_prep.py, as rebased in ../../gnmac-coop-r2 coop_prep.py).
# Changes: the routing ids get vllm-exl3's map_topk_to_local sentinel (invalid -> n_exp) instead of a
# raw cast, and the fp32 MoE output is zeroed in the same launch (stock torch.zeros), so every tensor the
# coop kernel reads equals the stock prep bit for bit. Sizes are not specialized: one binary for all rows.
"""The one-launch dispatch prep kernel (CUDA, or the Triton interpreter). Imported lazily."""
from vllm.triton_utils import tl, triton

BLOCK = 1024


@triton.jit(do_not_specialize=['n_x', 'n_routes', 'n_exp', 'n_ctr'])
def _dispatch_prep(x_ptr, xh_ptr, out_ptr, n_x,
                   ids_ptr, sel_ptr, w_ptr, rw_ptr, n_routes, n_exp,
                   ctr_ptr, n_ctr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mx = offs < n_x
    x = tl.load(x_ptr + offs, mask=mx, other=0.0)
    tl.store(xh_ptr + offs, x.to(tl.float16), mask=mx)              # stock x2d.contiguous().half()
    tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=mx)   # stock torch.zeros(...)
    mr = offs < n_routes
    ids = tl.load(ids_ptr + offs, mask=mr, other=0).to(tl.int64)
    sel = tl.where((ids < 0) | (ids >= n_exp), n_exp, ids)          # stock map_topk_to_local(.., None)
    tl.store(sel_ptr + offs, sel.to(tl.int64), mask=mr)
    w = tl.load(w_ptr + offs, mask=mr, other=0.0)
    tl.store(rw_ptr + offs, w.to(tl.float16), mask=mr)              # stock weights...to(float16)
    tl.store(ctr_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < n_ctr)  # stock torch.zeros


def launch(x, ids, weights, xh, out, sel, rw, ctr, n_exp):
    n_x, n_routes, n_ctr = x.numel(), ids.numel(), ctr.numel()
    grid = (triton.cdiv(max(n_x, n_routes, n_ctr), BLOCK),)
    _dispatch_prep[grid](x, xh, out, n_x, ids, sel, weights, rw, n_routes, n_exp, ctr, n_ctr,
                         BLOCK=BLOCK, num_warps=4)
