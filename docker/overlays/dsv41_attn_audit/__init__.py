# SPDX-License-Identifier: MIT
"""Observe successful candidate dispatch once per switch per worker process."""
import functools
import os


def _rank():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else -1


def _log(message, *args):
    from vllm.logger import init_logger
    init_logger('vllm.dsv41_attn_audit').info(message, *args)


def observe(module, function, switch, successful, after=None):
    original = getattr(module, function)
    if getattr(original, '_r25_audit', False):
        return
    logged = False

    @functools.wraps(original)
    def call(*args, **kwargs):
        nonlocal logged
        result = original(*args, **kwargs)
        if not logged and successful(result):
            _log('DSV41_ATTN_ACTIVE switch=%s rank=%d pid=%d', switch, _rank(), os.getpid())
            logged = True
        if after is not None:
            after(result, args)
        return result

    call._r25_audit = True
    setattr(module, function, call)


def row_cap_observer(swa_meta, default):
    """2026-11-09: with DSV41_ATTN_SWA_FUSED_MAX_ROWS set (swa_meta.MAX_ROWS != default), log
    the cap at the first fused build, then count fused builds above the default per shape
    (rows, requests, causal mode, width) and log the count at 1, 2, 4, 8, ... and at every
    multiple of 256: a higher logged count proves builds of that shape after a lower one, and
    any 256 builds of a shape cross a logged count (review attn-r1/r2 P2)."""
    if swa_meta.MAX_ROWS == default:
        return None
    counts = {}
    capped = False

    def after(result, args):
        nonlocal capped
        if result is None:
            return
        if not capped:
            _log('DSV41_ATTN_SWA_ROWS max_rows=%d rank=%d pid=%d', swa_meta.MAX_ROWS, _rank(), os.getpid())
            capped = True
        rows = result.num_decode_tokens
        if rows <= default:
            return
        # build(builder, common_prefix_len, common_attn_metadata, ...): the metadata says the mode.
        key = (rows, result.num_decodes, int(not args[2].causal), result.decode_swa_width)
        count = counts[key] = counts.get(key, 0) + 1
        if count & (count - 1) == 0 or count % 256 == 0:
            _log('DSV41_ATTN_SWA_WIDE rows=%d reqs=%d noncausal=%d width=%d count=%d max_rows=%d rank=%d pid=%d',
                 *key, count, swa_meta.MAX_ROWS, _rank(), os.getpid())
    return after


def register():
    flags = ('DSV41_ATTN_SWA_FUSED', 'DSV41_ATTN_TOPK_GLUE')
    selected = {k: os.environ.get(k, '0') for k in flags}
    if any(v not in ('0', '1') for v in selected.values()):
        raise ValueError('invalid attention audit switches')
    if '1' not in selected.values():
        return False
    # Entry-point iteration order is not a qualification assumption. Register
    # the original (idempotent) plugin here before wrapping its dispatched code.
    import dsv41_attn_port
    dsv41_attn_port.register()
    if selected[flags[0]] == '1':
        from dsv41_attn_port import swa_meta
        from dsv41_attn_port.config import SWA_MAX_ROWS_DEFAULT
        observe(swa_meta, 'build', flags[0], lambda result: result is not None,
                row_cap_observer(swa_meta, SWA_MAX_ROWS_DEFAULT))
    if selected[flags[1]] == '1':
        from dsv41_attn_port import topk_glue
        observe(topk_glue, 'run', flags[1], lambda result: True)
    return True
