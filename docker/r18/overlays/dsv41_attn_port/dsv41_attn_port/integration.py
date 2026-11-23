# SPDX-License-Identifier: Apache-2.0
"""Narrow hooks: no model weights, KV writes, projections or LoRA replaced."""
import functools
import os
from vllm.platforms import current_platform
from .config import swa_max_rows


def sm80():
    return current_platform.is_cuda() and current_platform.is_device_capability(80)


def install(selected):
    if selected['DSV41_ATTN_SWA_FUSED']:
        from vllm.models.deepseek_v41.amd.rocm import (
            DeepseekV41ROCMAiterSparseSWAMetadataBuilder as Builder,
            DeepseekV41ROCMAiterSparseSWAMetadata as Metadata,
        )
        from . import swa_meta
        swa_meta.MAX_ROWS = swa_max_rows(os.environ)
        if not getattr(Builder.build, '_attn_port', False):
            from .contracts import validate_metadata
            validate_metadata()
            original = Builder.build

            @functools.wraps(original)
            def build(self, common_prefix_len, common_attn_metadata,
                      fast_build=False, replay_start=None):
                if sm80() and self.device.type == 'cuda':
                    result = swa_meta.build(self, common_prefix_len, common_attn_metadata,
                                            fast_build, replay_start, Metadata)
                    if result is not None:
                        return result
                return original(self, common_prefix_len, common_attn_metadata,
                                fast_build, replay_start)
            build._attn_port = True
            Builder.build = build

    if selected['DSV41_ATTN_TOPK_GLUE']:
        from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
        from .contracts import validate_topk
        from . import topk_glue
        if not getattr(SparseIndexerTopk.forward, '_attn_port', False):
            # Fail before a model arm; never guess at an unexported source revision.
            validate_topk(SparseIndexerTopk)
            original_topk = SparseIndexerTopk.forward

            @functools.wraps(original_topk)
            def forward(self, logits, seq_lens, next_n, topk_indices,
                        topk_tokens, max_seq_len):
                if (self._backend == 'torch' and sm80()
                    and topk_glue.supported(logits, seq_lens, next_n,
                                           topk_indices, topk_tokens)):
                    return topk_glue.run(logits, seq_lens, next_n, topk_indices, topk_tokens)
                return original_topk(self, logits, seq_lens, next_n, topk_indices,
                                     topk_tokens, max_seq_len)
            forward._attn_port = True
            SparseIndexerTopk.forward = forward
    from vllm.logger import init_logger
    init_logger('vllm.' + __name__).info('DSV41 attention port installed (default-off packet): %s swa_max_rows=%s',
                                         selected, swa_max_rows(os.environ) if selected['DSV41_ATTN_SWA_FUSED'] else 'n/a')
    return True
