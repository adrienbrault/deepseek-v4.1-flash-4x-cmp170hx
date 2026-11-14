# SPDX-License-Identifier: MIT
"""Extend the existing argmax hook; no competing MetaPathFinder.

The r11 dense wo_b call site is bound to the authenticated fused Triton kernel.
Only wo_b is LoRA-wrapped. Head/processor/topology/sampler eligibility remains
the original check, with a read-only runner view after a model audit.
"""
import functools
from contextvars import ContextVar
import inspect
import os
import re
import sys
import numpy as np

FALLBACK = ContextVar('r7a_sampled_spec_evidence', default=None)


def audit(runner):
    cfg = runner.lora_config
    if cfg is None or set(cfg.target_modules or ()) != {'wo_b'} or cfg.max_loras != 1 or cfg.max_lora_rank != 1:
        raise ValueError('r7a requires one rank-1 wo_b adapter')
    if str(cfg.lora_dtype) != 'torch.bfloat16':
        raise ValueError('r7a requires bf16 LoRA')
    lm = getattr(runner.model, 'language_model', runner.model)
    if (type(lm).__name__ != 'DeepseekV41LLMForCausalLM'
            or type(lm.lm_head).__name__ != 'ParallelLMHead'
            or type(lm.logits_processor).__name__ != 'LogitsProcessor'
            or lm.lm_head.num_added_embeddings != 0):
        raise ValueError('r7a head/processor/vocabulary is wrapped or unknown')
    names = []
    for name, module in runner.model.named_modules():
        if 'WithLoRA' not in type(module).__name__:
            continue
        match = re.search(r'(?:^|\.)layers\.(\d+)\.attn\.wo_b$', name)
        if (type(module).__name__ != 'RowParallelLinearWithLoRA' or not match
                or module.tp_size != 4 or module._enable_aux_cuda_stream):
            raise ValueError('r7a unexpected LoRA wrapper: ' + name)
        names.append(int(match[1]))
    # Layers 0/1 are wrapped but have zero adapter weights (checkpoint is 2..39).
    if sorted(names) != list(range(40)):
        raise ValueError('r7a expected precisely forty wo_b wrappers')
    runner._dsv41_r7a_audited = True
    print('DSV41_ABLIT_R7_AUDIT rank=' + str(lm.lm_head.tp_rank) + ' wrappers=40 head=plain', flush=True)


class RunnerView:
    lora_config = None

    def __init__(self, runner):
        self.runner = runner

    def __getattr__(self, key):
        return getattr(self.runner, key)


def extend_eligible(original):
    @functools.wraps(original)
    def eligible(runner, *args):
        if runner.lora_config is None:
            return original(runner, *args)
        if not getattr(runner, '_dsv41_r7a_audited', False):
            return False
        return original(RunnerView(runner), *args)
    return eligible


def wrap_load(original):
    @functools.wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        audit(self)
        from .dense import install, receipt
        install()  # Engine only; before either profiling or real graph capture.
        lm = getattr(self.model, 'language_model', self.model)
        print(receipt(lm.lm_head.tp_rank), flush=True)
        from .graphs import configure
        configure(self)
        return result
    return load


def wrap_sample(original):
    @functools.wraps(original)
    def sample(self, hidden, batch, grammar):
        sampled_spec = (batch.num_draft_tokens > 0 and
                        np.any(self.sampler.sampling_states.temperature.np[batch.idx_mapping_np] != 0))
        token = FALLBACK.set((self, batch) if sampled_spec else None)
        try:
            return original(self, hidden, batch, grammar)
        finally:
            FALLBACK.reset(token)
    return sample


def wrap_logits(original):
    @functools.wraps(original)
    def logits(self, *args, **kwargs):
        from dsv41_lora_argmax.hooks import ACTIVE
        probe = FALLBACK.get()
        # The original wrapped method selects gathered logits only with ACTIVE=None.
        gathered = probe is not None and ACTIVE.get() is None
        result = original(self, *args, **kwargs)
        if probe is not None:
            if not gathered or result.shape[-1] != self.logits_processor.org_vocab_size:
                raise ValueError('r7a sampled speculation bypassed gathered logits')
            runner, batch = probe
            ids = runner.lora_state.lora_ids[batch.idx_mapping_np]
            seen = getattr(runner, '_r7a_fallback_seen', set())
            for i, request_id in enumerate(batch.req_ids):
                # Log only the bounded diagnostic requests, correlated by API id.
                # f760 InputProcessor appends an internal 8-hex request suffix.
                match = re.fullmatch(r'(chatcmpl-r7a-fallback-\d+-[01])(?:-[0-9a-f]{8})?', request_id)
                if not match or match[1] in seen:
                    continue
                request_id = match[1]
                mode = 'raw' if ids[i] == 0 else 'adapted'
                print(f'DSV41_ABLIT_R7_FALLBACK rank={self.lm_head.tp_rank} mode={mode} sampled_spec=gathered request_id={request_id}', flush=True)
                seen.add(request_id)
            runner._r7a_fallback_seen = seen
        return result
    return logits


def wrap_available(original):
    @functools.wraps(original)
    def available(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        print(f'DSV41_ABLIT_R7_KV rank={self.rank} available_bytes={result}', flush=True)
        return result
    return available


def wrap_capture(original):
    @functools.wraps(original)
    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not kwargs.get('profile_only', False):
            from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
            tokens, _ = get_kv_cache_capacity(self.vllm_config, self.kv_cache_config)
            lm = getattr(self.model, 'language_model', self.model)
            print(f'DSV41_ABLIT_R7_MEMORY rank={lm.lm_head.tp_rank} tokens={tokens} graph_bytes={result}', flush=True)
        return result
    return capture


def install():
    if os.environ.get('DSV41_DENSE_LORA') != '1' or os.environ.get('VLLM_LORA_ENABLE_DUAL_STREAM', '0') != '0':
        raise ValueError('r7a dense update required; auxiliary stream forbidden')
    from dsv41_lora_argmax import hooks as base, eligibility
    if getattr(base, '_r7a_extension', False):
        return
    base.install()
    # Extend the SAME finder/callback, including the worker's CPU memory query.
    base.TARGETS.add('vllm.v1.worker.gpu_worker')
    # sampled-argmax saves the old greedy gate in this field. Preserve either
    # registration order; sampled-spec stays ineligible in its own gate.
    key = '_sampled_original_eligible' if hasattr(eligibility, '_sampled_original_eligible') else 'eligible'
    setattr(eligibility, key, extend_eligible(getattr(eligibility, key)))
    previous = base.patch

    def patch(module):
        previous(module)
        if module.__name__ == 'vllm.v1.worker.gpu.model_runner' and not getattr(module, '_r7a_load_hook', False):
            module.GPUModelRunner.load_model = wrap_load(module.GPUModelRunner.load_model)
            module.GPUModelRunner.sample = wrap_sample(module.GPUModelRunner.sample)
            module.GPUModelRunner.capture_model = wrap_capture(module.GPUModelRunner.capture_model)
            module._r7a_load_hook = True
        if module.__name__ == 'vllm.models.deepseek_v41.nvidia.model' and not getattr(module, '_r7a_logits_hook', False):
            cls = module.DeepseekV41LLMForCausalLM
            cls.compute_logits = wrap_logits(cls.compute_logits)
            module._r7a_logits_hook = True
        if module.__name__ == 'vllm.v1.worker.gpu_worker' and not getattr(module, '_r7a_memory_hook', False):
            module.Worker.determine_available_memory = wrap_available(module.Worker.determine_available_memory)
            module._r7a_memory_hook = True
    base.patch = patch
    for name in base.TARGETS & sys.modules.keys():
        patch(sys.modules[name])
    base._r7a_extension = True
    print('DSV41_ABLIT_R7_READY dense=fused-triton argmax=audited-whole-batch', flush=True)
