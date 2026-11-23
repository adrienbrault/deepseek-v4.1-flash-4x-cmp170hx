# SPDX-License-Identifier: MIT
"""Fixed r7a dense binding; importing this module needs no torch/Triton/CUDA."""
import hashlib
from pathlib import Path

R11_SHA256 = 'e7152d0f8f93f73986074605b10a29339e7baee232db36c122d7684e4dbf7a78'
FUSED_SHA256 = '5e4188e153b177faf85dfbb386a5d7e3ac5844b834eae467a2b8f192ddc0bce5'
TRITON_VERSION = '3.8.0'


def authenticate(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('r7a dense source drift: ' + str(path))


def install():
    """Called after engine model construction, before profiling/capture.

    Change only the global resolved by r11's one dense wo_b call site. Its
    assertions and every non-dense/Punica path remain the original code.
    No CUDA allocation or JIT compilation occurs until the first forward.
    """
    from vllm.lora.layers import base_linear
    authenticate(base_linear.__file__, R11_SHA256)
    authenticate(Path(__file__).with_name('fused_rank1.py'), FUSED_SHA256)
    try:
        from . import fused_rank1
    except ImportError as error:
        raise RuntimeError('r7a requires the fused Triton dense implementation; import failed') from error
    if fused_rank1.triton.__version__ != TRITON_VERSION:
        raise ValueError('r7a requires Triton ' + TRITON_VERSION)
    current = base_linear._dsv41_dense_rank1
    if current is not fused_rank1.dense_rank1:
        if (current.__module__ != base_linear.__name__
                or current.__name__ != '_dsv41_dense_rank1'
                or current.__globals__ is not base_linear.__dict__):
            raise ValueError('r7a dense binding was replaced before installation')
        base_linear._dsv41_dense_rank1 = fused_rank1.dense_rank1
    return fused_rank1


def receipt(rank):
    return f'DSV41_ABLIT_R7_DENSE rank={rank} impl=fused-triton sha={FUSED_SHA256[:12]}'


def require_receipts(log):
    # Require the complete value, not a prefix of a wrong/longer SHA or rank.
    for rank in range(4):
        if not any(line.endswith(receipt(rank)) for line in log.splitlines()):
            raise ValueError('r7a fused dense audit missing on TP rank ' + str(rank))
