# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One SM80 kernel for OUR legacy row arithmetic, including scale 0/255.

Inspired by Morrowmake ops/sm80_engram_dequant.py (see NOTICE). Deliberately
does not adopt its E8M0 corrections: ours bitcasts scale_byte << 23, so 0 is
zero and 255 is infinity. Changing that would change the baseline model.
No hardware FP8 conversion is used. Finite BF16 output must be bit-identical;
NaN positions must agree but NaN payload/sign is outside this contract.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _legacy_engram_dequant_kernel(w, s, out, N: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(w + x, x < N, other=0).to(tl.int32)
    exponent = (b >> 3) & 15
    mantissa = b & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal | ((b & 128) << 24)
    # E4M3 subnormals are exactly mantissa * 2^-9, including signed zero.
    sub = (mantissa.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True)
    sub = sub | ((b & 128) << 24)
    value = tl.where(exponent == 0, sub, normal).to(tl.float32, bitcast=True)
    value = tl.where((b & 127) == 127, float("nan"), value)
    scale_b = tl.load(s + x // 32, x < N, other=0).to(tl.int32)
    scale = (scale_b << 23).to(tl.float32, bitcast=True)
    # Keep FP32 subnormal products, matching torch rather than FTZ arithmetic.
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[value, scale],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(out + x, product.to(tl.bfloat16), x < N)


def dequant(values, scales, out):
    n = values.shape[0]
    if (values.dtype != torch.uint8 or scales.dtype != torch.uint8
            or out.dtype != torch.bfloat16 or values.shape != (n, 256)
            or scales.shape != (n, 8) or out.shape != (n, 256)
            or not all(t.is_contiguous() and t.is_cuda for t in (values, scales, out))
            or values.device != scales.device or out.device != values.device):
        raise ValueError("EG dequant requires contiguous same-device CUDA u8 [N,256]/[N,8], BF16 out")
    if n:
        _legacy_engram_dequant_kernel[(triton.cdiv(n * 256, 256),)](values, scales, out, n * 256, 256)
