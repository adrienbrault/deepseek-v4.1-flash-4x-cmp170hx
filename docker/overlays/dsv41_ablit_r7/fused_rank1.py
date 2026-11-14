# SPDX-License-Identifier: MIT
"""Isolated candidate only: one launch, no scratch, allocation or host scalar read.

Triton 3.8.0 in f760. Import this module only when the GPU dependency is needed.
The launch grid depends on tensor shapes, never on the contents of LoRA metadata.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rank1(X, Y, A, B, Indices,
           SX0: tl.constexpr, SX1: tl.constexpr,
           SY0: tl.constexpr, SY1: tl.constexpr,
           SA: tl.constexpr, SB: tl.constexpr, SI: tl.constexpr,
           N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    # Uniform within a CTA. In an all-raw batch every CTA takes this cheap
    # path: one metadata read, no activation/weight/output load and NO store.
    if tl.load(Indices + row * SI) == 0:
        k = tl.arange(0, 2048)
        x = tl.load(X + row * SX0 + k * SX1).to(tl.float32)
        a = tl.load(A + k * SA).to(tl.float32)
        # BF16 products are exact in FP32 (normal finite model values).
        # This FP32 tree reduction is NOT cuBLAS's unspecified reduction tree.
        t = tl.sum(x * a, axis=0)
        n = tile * BLOCK + tl.arange(0, BLOCK)
        b = tl.load(B + n * SB, n < N, other=0).to(tl.float32)
        # Match the SERVED numerics: the helper runs inside vLLM's torch.compile'd
        # forward, where Inductor keeps the BF16 delta in FP32 and rounds old + t*b
        # once (compile_probe.py 2026-11-13: bitwise to compiled-unfused at decode
        # sizes; the earlier double rounding matched only the eager helper). No FMA.
        old = tl.load(Y + row * SY0 + n * SY1, n < N, other=0).to(tl.float32)
        updated = (old + t * b).to(tl.bfloat16)
        tl.store(Y + row * SY0 + n * SY1, updated, n < N)


def dense_rank1(x, output, a, b, token_lora_indices):
    """Same in-place signature and supported shapes as the r11 dense helper.

    Raw rows (any index other than slot 0) are never loaded or stored, including
    NaN payloads and signed zero. Adapted rows are numerically, not bitwise,
    equivalent; see HOW-TO-RUN.md for the fixed benchmark tolerance.
    """
    assert x.ndim == output.ndim == 2
    m = x.shape[0]
    assert output.shape == (m, 5120) and x.shape[1] == 2048
    assert token_lora_indices.ndim == 1 and m <= token_lora_indices.shape[0]
    assert a.shape == (1, 1, 1, 2048) and b.shape == (1, 1, 5120, 1)
    assert x.dtype == output.dtype == a.dtype == b.dtype == torch.bfloat16
    assert token_lora_indices.dtype in (torch.int32, torch.int64)
    assert x.device == output.device == a.device == b.device == token_lora_indices.device
    assert x.is_cuda
    # All validation is shape/dtype/device-only, so replay requires no host sync.
    _rank1[(m, triton.cdiv(5120, 256))](
        x, output, a, b, token_lora_indices,
        x.stride(0), x.stride(1), output.stride(0), output.stride(1),
        a.stride(3), b.stride(2), token_lora_indices.stride(0),
        N=5120, BLOCK=256, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return output
