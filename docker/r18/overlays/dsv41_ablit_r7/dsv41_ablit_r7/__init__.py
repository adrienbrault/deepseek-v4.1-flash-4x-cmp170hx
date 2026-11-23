# SPDX-License-Identifier: MIT
import os


def register():
    switch = os.environ.get('DSV41_ABLIT_R7', '0')
    if switch == '0':
        return False
    if switch != '1':
        raise ValueError('DSV41_ABLIT_R7 must be 0 or 1')
    from .hooks import install
    install()
    draft = os.environ.get('DSV41_DRAFT_SHARD_ARGMAX', '0')
    if draft != '0':
        from .draft import install as install_draft
        install_draft()
    from vllm.logger import init_logger
    log = init_logger('vllm.dsv41_stack')
    engram = all(os.environ.get(key) == value for key, value in {
        'DSV41_EG_DEQUANT': '1', 'DSV41_EG_PREFETCH': '1', 'DSV41_EG_ROW_CACHE_MB': '16'
    }.items())
    log.info('[vllm.dsv41_stack] DSV41_STACK_READY_R7A dense-lora=1 ablit-r7=1 argmax=%s engram=%d',
             os.environ.get('DSV41_TARGET_ARGMAX', '0'), engram)
    log.info('[vllm.dsv41_stack] DSV41_STACK_CONFIG splitk=%s rowmask=%s lookup=%s live=%s',
             os.environ.get('VLLM_SM80_SPARSE_DECODE', '0'),
             'cap1' if os.environ.get('VLLM_DSV41_MOE_ROW_MASK') else 'off',
             os.environ.get('VLLM_DSV41_PROMPT_LOOKUP', '0'), os.environ.get('DSV41_LIVE', '0'))
    return True
