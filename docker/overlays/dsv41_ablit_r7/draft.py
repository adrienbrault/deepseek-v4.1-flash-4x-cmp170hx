# SPDX-License-Identifier: MIT
"""Scope the frozen draft guard's LoRA exception to an unchanged draft head.

Only wo_b is adapted in the target; DSpark's head and processor must stay plain.
All original draft eligibility and source checks still run. No shared file changes.
"""
import functools
import os


class ConfigView:
    lora_config = None

    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        return getattr(self.config, name)


def extend_ineligible(original):
    @functools.wraps(original)
    def ineligible(spec, model, tp_size, config):
        cfg = config.lora_config
        if cfg is None:
            return original(spec, model, tp_size, config)
        if (os.environ.get('DSV41_ABLIT_R7') != '1' or os.environ.get('DSV41_DENSE_LORA') != '1'
                or tp_size != 4 or set(cfg.target_modules or ()) != {'wo_b'}
                or cfg.max_loras != 1 or cfg.max_lora_rank != 1 or str(cfg.lora_dtype) != 'torch.bfloat16'):
            return 'r7a draft requires one bf16 rank-1 wo_b adapter at TP4'
        if (type(model.lm_head).__name__ != 'ParallelLMHead'
                or type(model.logits_processor).__name__ != 'LogitsProcessor'
                or any('WithLoRA' in type(m).__name__ for _, m in model.named_modules())):
            return 'r7a draft head/processor/model must be unwrapped'
        reason = original(spec, model, tp_size, ConfigView(config))
        if reason is None:
            print(f'DSV41_DRAFT_ARGMAX_R7A_READY rank={model.lm_head.tp_rank} head=plain wo_b_only=1', flush=True)
        return reason
    return ineligible


def install():
    import dsv41_draft_argmax
    from dsv41_draft_argmax import hooks
    if not getattr(hooks.ineligible, '_r7a_extension', False):
        hooks.ineligible = extend_ineligible(hooks.ineligible)
        hooks.ineligible._r7a_extension = True
    if not dsv41_draft_argmax.register():
        raise ValueError('r7a requires sharded draft argmax')
