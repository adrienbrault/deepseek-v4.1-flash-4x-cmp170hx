# SPDX-License-Identifier: MIT
import os


def register():
    required = {'DSV41_DENSE_LORA': '1', 'DSV41_ABLIT_R7': '1', 'DSV41_TARGET_ARGMAX': '1',
                'DSV41_EG_DEQUANT': '1', 'DSV41_EG_PREFETCH': '1', 'DSV41_EG_ROW_CACHE_MB': '16'}
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError('r7a stack audit requires dense LoRA + r7a + target argmax + engram 1/1/16')
    from .draft import install as install_draft
    from .hooks import install
    install_draft()
    install()
    from vllm.logger import init_logger
    log = init_logger('vllm.dsv41_stack')
    log.info('[vllm.dsv41_stack] DSV41_STACK_READY_R7A dense-lora=1 ablit-r7=1 argmax=1 engram=1')
    log.info('[vllm.dsv41_stack] DSV41_STACK_CONFIG splitk=%s rowmask=%s lookup=%s live=%s',
             os.environ.get('VLLM_SM80_SPARSE_DECODE', '0'),
             'cap1' if os.environ.get('VLLM_DSV41_MOE_ROW_MASK') else 'off',
             os.environ.get('VLLM_DSV41_PROMPT_LOOKUP', '0'), os.environ.get('DSV41_LIVE', '0'))
    return True
