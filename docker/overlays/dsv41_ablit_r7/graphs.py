# SPDX-License-Identifier: MIT
"""V1's unspecialized LoRA policy on constructed, served r7 V2 objects.

Only graph selection changes. Request IDs, real active counts and Punica's
runtime token/slot mapping remain upstream's, including -1 for raw rows.
The stage-r18 manager's row-mask cap compatibility and capture scopes still
run; shared LoRA selection must never override their padding/eager decisions.
"""
import functools


def configure(runner):
    cfg = runner.lora_config
    if cfg is None or runner.compilation_config.cudagraph_specialize_lora is not False:
        return
    if getattr(runner, '_r7a_shared_lora_graphs', False):
        return
    case = cfg.max_loras + 1
    if runner.cudagraph_manager is not None or runner.lora_capture_cases != [0, case]:
        raise ValueError('r7a unexpected V2 LoRA construction/capture order')

    # load_model has returned; neither the profiling nor real KV manager exists.
    # Both later constructions consume this list before _init_candidates().
    runner.lora_capture_cases = [case]
    original = runner.initialize_kv_cache

    @functools.wraps(original)
    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        manager = runner.cudagraph_manager
        if (manager.lora_capture_cases != [case] or manager.graphs
                or manager._graphs_captured
                or any(d.num_active_loras != case
                       for ds in manager._capture_descs.values() for d in ds)
                or any(key[1] != case for key in manager._candidates)):
            raise ValueError('r7a unexpected V2 LoRA graph candidates')
        resolve = manager._resolve_effective_loras

        @functools.wraps(resolve)
        def effective(num_active_loras):
            # Includes zero, mixed, adapted-only and dummy batches. Preserve
            # upstream's shape/CG-mode checks and eager fallback in dispatch().
            return resolve(case)

        @functools.wraps(manager.captured_token_counts)
        def token_counts():
            # Upstream filters case zero here. AV profiling needs the sizes of
            # the shared graph after that case has been removed.
            return sorted({d.num_tokens for d in manager.graphs
                           if d.num_active_loras == case})

        manager._resolve_effective_loras = effective
        manager.captured_token_counts = token_counts
        lm = getattr(runner.model, 'language_model', runner.model)
        print(f'DSV41_ABLIT_R7_GRAPHS rank={lm.lm_head.tp_rank} cases={case} dispatch_zero={case}', flush=True)
        return result

    runner.initialize_kv_cache = initialize
    runner._r7a_shared_lora_graphs = True
