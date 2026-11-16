# SPDX-License-Identifier: MIT
"""Fail-closed, CPU-only graph policy proof, also used on constructed runners."""
import json

SIZES = list(range(1, 11)) + [12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 32, 36, 42, 48]
GRAPH_MODE = 'FULL_DECODE_ONLY'


def require(ok, message):
    if not ok:
        raise RuntimeError('7b EAGER_ONLY: ' + message)


def mode_name(value):
    return getattr(value, 'name', str(value).rsplit('.', 1)[-1])


def argv_proof(argv):
    def option(key, default=None):
        values = [arg.split('=', 1)[1] for arg in argv if arg.startswith(key + '=')]
        values += [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == key]
        require(len(values) <= 1, 'duplicate option ' + key)
        return values[0] if values else default

    require('--enforce-eager' not in argv and option('--enforce-eager', 'false') == 'false',
            'enforce-eager changed')
    require(option('--compilation-config') is None and option('-cc') is None,
            'opaque compilation config')
    sizes = json.loads(option('-cc.cudagraph_capture_sizes', '[]'))
    require(sizes and sizes == sorted(set(sizes)) and
            all(type(n) is int and 0 < n <= 48 for n in sizes) and max(sizes) == 48,
            'capture sizes must retain incumbent 48-row ceiling')
    maximum = int(option('-cc.max_cudagraph_capture_size', str(max(sizes))))
    require(maximum == 48, 'maximum graph batch size differs')
    graph_mode = option('-cc.cudagraph_mode')
    require(graph_mode == GRAPH_MODE, 'incumbent graph mode differs; require ' + GRAPH_MODE)
    compile_mode = option('-cc.mode', 'default')
    require(compile_mode in ('default', '0', '3', 'NONE', 'VLLM_COMPILE'),
            'stock/unknown torch compilation could hide Python switching')
    require(option('-cc.compile_sizes', '[]') == '[]', 'static compile sizes unsupported')
    spec = json.loads(option('--speculative-config', '{}'))
    require(int(option('--max-num-seqs', '0')) == 8 and
            spec.get('method') == 'dspark' and spec.get('num_speculative_tokens') == 5 and
            spec.get('enable_adaptive_verification') is False,
            'decode bound must remain 8 * (5 + 1) = 48 rows')
    require(int(option('--max-num-batched-tokens', '0')) == 2048, 'prefill chunk cap differs')
    return dict(capture_sizes=sizes, max_graph_rows=maximum, graph_mode=graph_mode,
                enforce_eager=False, compilation_mode=compile_mode, compile_sizes=[],
                max_decode_rows=48, min_7b_rows=128, max_prefill_rows=2048,
                conclusion='128+ rows absent from graph candidate keys; dispatch returns NONE')


def require_decode_only(manager):
    require(mode_name(manager.cudagraph_mode) == GRAPH_MODE and
            mode_name(manager.compilation_config.cudagraph_mode) == GRAPH_MODE,
            'effective graph mode differs; require ' + GRAPH_MODE)


def manager_proof(manager):
    require_decode_only(manager)
    cfg = manager.compilation_config
    require(max(cfg.cudagraph_capture_sizes) == cfg.max_cudagraph_capture_size == 48,
            'effective capture config changed')
    descriptors = [d for ds in manager._capture_descs.values() for d in ds]
    require(descriptors and all(0 < d.num_tokens <= 48 for d in descriptors),
            'capture descriptor exceeds decode ceiling')
    require(all(d.num_tokens <= 48 for d in manager.graphs), 'captured graph exceeds ceiling')
    require(manager._candidates and all(key[0] <= 48 for key in manager._candidates),
            'graph dispatch candidate exceeds ceiling')
    inventory = descriptors + list(manager.graphs) + [
        d for candidates in manager._candidates.values() for d in candidates]
    require(all(mode_name(d.cg_mode) == 'FULL' and 0 < d.num_tokens <= 48 and
                (d.uniform_token_count is not None or d.max_query_len is not None)
                for d in inventory), 'non-decode graph descriptor')
    return dict(graph_mode=GRAPH_MODE, max_graph_rows=48,
                descriptors=len(descriptors), captured=len(manager.graphs))


def runner_proof(runner, graphs_required=True):
    require(not runner.model_config.enforce_eager, 'live enforce_eager differs')
    require(mode_name(runner.compilation_config.cudagraph_mode) == GRAPH_MODE,
            'live graph mode differs; require ' + GRAPH_MODE)
    require(mode_name(runner.compilation_config.mode) != 'STOCK_TORCH_COMPILE',
            'whole-model compilation is unsupported')
    from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule
    require(not has_compiled_submodule(runner.get_model()), 'compiled submodule may hide switch')
    require(runner.lora_config is not None and runner.model_state.replay_cudagraphs is None,
            'inner decoder-replay graphs must remain disabled by served LoRA config')
    require(runner.speculator is not None and runner.speculator.num_query_per_req <= 6,
            'draft query bound changed')
    draft = runner.speculator.query_cudagraph_manager
    target = runner.cudagraph_manager
    if not graphs_required and (target is None or draft is None):
        # vLLM order: load_model, profile_run's dummy forward (no graph managers),
        # the graph-memory profile (minimal KV init, a profile_only capture, then the
        # drafter manager is dropped), initialize_kv_cache (both managers), real capture.
        # 11-14 boots of 13:30 and 13:42 Paris failed here. Existing managers are still
        # proved; every capture is fenced by manager_proof; scored witnesses must carry
        # both managers' proofs (prefill_metrics.executed).
        proof = dict(graph_mode=GRAPH_MODE, graphs='not-yet-created')
        if target is not None:
            proof.update(manager_proof(target))
        if draft is not None:
            manager_proof(draft)
        return proof
    require(target is not None, 'target graph manager absent')
    require(draft is not None, 'draft graph manager absent')
    manager_proof(draft)
    return dict(manager_proof(runner.cudagraph_manager),
                compilation_mode=mode_name(runner.compilation_config.mode),
                enforce_eager=False, active_compiled_submodule=False,
                inner_decoder_replay=False,
                draft_max_rows=max(d.num_tokens for ds in draft._capture_descs.values() for d in ds))
