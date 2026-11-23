# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions Copyright 2026 Morrowmake; adapted for this packet.
"""Host-state eligibility; unsupported batches retain original gathered logits."""
import numpy as np


def eligible(r, batch, grammar, sampler_type, rejection_type):
    p = r.parallel_config
    if (r.lora_config is not None or r.batch_sharder is not None or grammar is not None
            or p.tensor_parallel_size != 4 or p.pipeline_parallel_size != 1
            or p.prefill_context_parallel_size != 1 or p.decode_context_parallel_size != 1):
        return False
    if batch.num_reqs == 0 or batch.has_structured_output_reqs:
        return False
    sampler, rejection = r.sampler, r.rejection_sampler
    if type(sampler) is not sampler_type or (rejection is not None and type(rejection) is not rejection_type):
        return False
    if (sampler.compute_nans or sampler.return_sampling_mask or sampler.trace_replay_state is not None
            or len(sampler.logits_processors) != 3):
        return False
    idx = batch.idx_mapping_np
    if (sampler.get_logprobs_dims(idx) is not None or np.any(sampler.needs_logits_processing[idx])
            or np.any(sampler.distributed_fallback[idx])):
        return False
    s = sampler.sampling_states
    if not (np.all(s.temperature.np[idx] == 0) and np.all(s.top_k.np[idx] == s.vocab_size)
            and np.all(s.top_p.np[idx] == 1)):
        return False
    if rejection is not None and (rejection.use_block_verification or rejection.fly_window_size
            or rejection.synthetic_conditional_rates is not None or rejection.watermark_key is not None):
        return False
    lm = getattr(r.model, 'language_model', r.model)
    # No head/processor wrapper, added vocabulary, head permutation or unknown
    # model is allowed. In particular never bypass an lm_head LoRA wrapper.
    if type(lm).__name__ != 'DeepseekV41LLMForCausalLM':
        return False
    head, processor = lm.lm_head, lm.logits_processor
    if (type(head).__name__ != 'ParallelLMHead' or type(processor).__name__ != 'LogitsProcessor'
            or head.tp_size != 4 or head.num_added_embeddings != 0 or processor.logits_as_input):
        return False
    si = head.shard_indices
    width = head.num_embeddings_per_partition
    if (si.org_vocab_start_index != head.tp_rank * width or not 1 <= width <= 65536
            or not 1 <= processor.org_vocab_size < 2**24):
        return False
    spec = r.speculator
    if spec is not None:
        # Reject alternate/speculative distributions. DSpark's proposal and
        # confidence head themselves are never wrapped or changed.
        if r.speculative_config is None or not r.speculative_config.use_dspark():
            return False
        draft_logits = getattr(spec, 'draft_logits', None)
        if draft_logits is not None and draft_logits.values.shape[-1] < processor.org_vocab_size:
            return False
    return True
