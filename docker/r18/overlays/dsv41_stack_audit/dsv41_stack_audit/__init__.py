# SPDX-License-Identifier: MIT
"""Startup audit only; no policy or tensor changes."""
import os


def register():
    from vllm.logger import init_logger
    log = init_logger('vllm.dsv41_stack')
    required={'DSV41_DENSE_LORA':'0','DSV41_TARGET_ARGMAX':'1','DSV41_EG_DEQUANT':'1',
              'DSV41_EG_PREFETCH':'1','DSV41_EG_ROW_CACHE_MB':'16'}
    if any(os.environ.get(k)!=v for k,v in required.items()):
        raise ValueError('stack audit requires base-only + target argmax')
    log.info('[vllm.dsv41_stack] DSV41_STACK_READY base-only argmax=1 engram=1')
    log.info('[vllm.dsv41_stack] DSV41_STACK_CONFIG splitk=%s rowmask=%s lookup=%s live=%s',
             os.environ.get('VLLM_SM80_SPARSE_DECODE','0'),
             'cap1' if os.environ.get('VLLM_DSV41_MOE_ROW_MASK') else 'off',
             os.environ.get('VLLM_DSV41_PROMPT_LOOKUP','0'),os.environ.get('DSV41_LIVE','0'))
    return True
