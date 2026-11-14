# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device kernels adapted from Morrowmake's DSpark row-mask port; see NOTICE."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _routes(IDS, W, LIVE, OI, OW, ROWS: tl.constexpr, TOP: tl.constexpr,
            IS: tl.constexpr, WS: tl.constexpr, IC: tl.constexpr,
            WC: tl.constexpr, B: tl.constexpr):
    flat = tl.program_id(0) * B + tl.arange(0, B)
    r, c = flat // TOP, flat % TOP
    ok = r < ROWS
    live = tl.load(LIVE + r, mask=ok, other=1) != 0
    ids = tl.load(IDS + r * IS + c * IC, mask=ok, other=-1)
    w = tl.load(W + r * WS + c * WC, mask=ok, other=0.)
    tl.store(OI + flat, tl.where(live, ids, -1), mask=ok)
    tl.store(OW + flat, tl.where(live, w, 0.), mask=ok)


def mask_routes_cuda(state, ids, weights):
    oi = torch.empty(ids.shape, dtype=ids.dtype, device=ids.device)
    ow = torch.empty(weights.shape, dtype=weights.dtype, device=weights.device)
    n, k = ids.shape
    _routes[(triton.cdiv(n * k, 128),)](
        ids, weights, state.row_live, oi, ow, n, k,
        ids.stride(0), weights.stride(0), ids.stride(1), weights.stride(1), 128,
        num_warps=4)
    return oi, ow
