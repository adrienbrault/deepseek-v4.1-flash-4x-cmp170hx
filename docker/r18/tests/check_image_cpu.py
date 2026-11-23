#!/usr/bin/env python3
"""Import the assembled image and exercise its exact serve parser without CUDA."""
import argparse
from functools import partial
import hashlib
import importlib
from importlib.metadata import entry_points
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from apply_overlays import check_files, load_series
from transform_scheduler import transform
from derive_argv import PLUGIN_ALLOWLIST
from install_overlays import sha
from prefill_share import MODE_ENV


def module_for(root, path):
    path = path.relative_to(root).with_suffix("")
    parts = path.parts[:-1] if path.name == "__init__" else path.parts
    return ".".join(parts)


def import_patches(root, paths, package):
    for rel in sorted(paths):
        path = root / rel
        if path.suffix == ".py":
            module = importlib.import_module(module_for(package, path))
            if Path(module.__file__).resolve() != path.resolve():
                raise ValueError(f"shadowed patched module: {module.__name__}")
            print("IMPORT", module.__name__)


def compile_serving_scheduler(source):
    for fair_r2 in (False, True):
        result = transform(source, fair_r2)
        module = ModuleType("vllm.v1.core.sched._r18_scheduler_check")
        module.__package__ = "vllm.v1.core.sched"
        sys.modules[module.__name__] = module
        # Exec framework definitions to verify dependencies and decorators
        # without constructing a scheduler or touching model/GPU state.
        try:
            exec(compile(result, str(source), "exec"), module.__dict__)
            assert hasattr(module.Scheduler, "_fair_init")
            assert hasattr(module.Scheduler, "_fair_has_long_prefill") is fair_r2
        finally:
            del sys.modules[module.__name__]
        print("scheduler", "r7h" if fair_r2 else "r7g", hashlib.sha256(result).hexdigest())


def check_prefill_share(source, installed):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    assert installed.read_bytes() == transform(source, fair_r2=True, prefill_share=True)
    assert "_install_prefill_share" in Scheduler.__init__.__code__.co_names
    # The served AsyncScheduler inherits Scheduler.schedule. Exercise selection
    # on that class without allocating caches or initializing a device.
    for settings, expected in (
        ({}, Scheduler.schedule),
        ({MODE_ENV: "off"}, Scheduler.schedule),
        ({MODE_ENV: "half"}, Scheduler._schedule_with_prefill_share),
        ({MODE_ENV: "0.25"}, Scheduler.schedule),
    ):
        state = AsyncScheduler.__new__(AsyncScheduler)
        with patch.dict(os.environ, settings, clear=True):
            state._install_prefill_share()
        assert state.schedule.__func__ is expected
        if settings.get(MODE_ENV) == "half":
            assert state._small_clipped_steps == state._small_mixed_clipped_steps == 0
    print("PASS prefill share defaults off and installs half on the served async scheduler")


def check_native(expect_ring):
    import torch
    import exllamav3_ext
    import vllm_exl3_c

    assert callable(exllamav3_ext.exl3_moe_coop)
    assert callable(vllm_exl3_c.p2b_fused_moe)
    coop = Path("/opt/dsv41-coop-kernel")
    manifest = json.loads((coop / "build-manifest.json").read_text())
    binary = coop / "opera_coop_ext.so"
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest["library_sha256"]
    assert manifest["torch_version"] == torch.__version__
    assert hashlib.sha256(Path(exllamav3_ext.__file__).read_bytes()).hexdigest() == manifest["stock_native_sha256"]
    spec = importlib.util.spec_from_file_location("opera_coop_ext", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.opera_coop_variant_v1() == 1
    from vllm.distributed.device_communicators import opera_true_ring

    receipt = ROOT / "ring-helper-build.json"
    ring_root = opera_true_ring.ASSETS
    if expect_ring is not None:
        assert receipt.is_file() == (expect_ring == "1"), "wrong true-ring image build"
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        assert sha(Path(opera_true_ring.__file__)) == record["installed_sha256"]
        assert sha(ring_root / "manifest.json") == record["manifest_sha256"]
        manifest, policy = opera_true_ring.read_assets()
        extra, _ = opera_true_ring.read_extra()
        # The same startup verifier checks all stock hashes, the ABI, native
        # library identity and exports. Loading it does not create a workspace.
        opera_true_ring.load_native(manifest)
        print("PASS true-ring manifest and CPU load", manifest["native_library_sha256"],
              "rows", sorted(set(extra) | {int(row) for row in policy["ring_configs"]}))
    else:
        assert os.environ.get("DSV41_TRUE_RING", "0") == "0"
        assert not (ring_root / "native.so").exists()
        assert not (ring_root / "manifest.json").exists()
        print("true-ring off; using stock FlashInfer PCIe IPC")


def check_runtime_native():
    import ctypes
    import torch
    from build_runtime_native import INDEXER, PREFILL, QV, ROW, STACK, check_stack_binary
    sys.path.insert(0, str(QV))
    from indexer_runtime import native

    assert ctypes.CDLL(str(ROW)).row_store_gap_revision() == 1
    assert sha(ROW) == json.loads(ROW.with_name("build.json").read_text())["binary_sha256"]
    torch.ops.load_library(str(PREFILL))
    assert callable(torch.ops._dsv41_mm_try.moe_out_prefill_try)
    assert callable(torch.ops._dsv41_mm_try.moe_workspace_size_prefill_try)
    assert sha(PREFILL) == json.loads(PREFILL.with_name("build.json").read_text())["binary_sha256"]
    assert callable(native(INDEXER, QV / "native-seal.json",
                           (QV / "native-seal.sha256").read_text().strip()).run)
    seal = STACK / "seal.json"
    assert sha(seal) == (STACK / "seal.sha256").read_text().strip()
    for name, digest in json.loads(seal.read_text())["binaries"].items():
        path = STACK / name
        assert sha(path) == digest
        check_stack_binary(path, path.stem)


def check_decoder_replay(engine):
    import numpy as np
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model
    from vllm.models.deepseek_v41.nvidia import model_state

    assert engine.enable_lora
    assert engine.enable_prefix_caching
    assert engine.swa_bounded_replay
    # The mounted checkpoint is absent at build time. Use its layer and window
    # geometry with the real gate and the settings resolved from the serve argv.
    cut, end_layer, window = 20, 40, 128
    model = SimpleNamespace(
        start_layer=0, end_layer=end_layer, use_sequence_parallel=False,
        config=SimpleNamespace(num_hidden_layers=end_layer, sliding_window=window,
                               engram_layer_ids=(1, 14)),
        layers={cut: SimpleNamespace(attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(
            bounded_replay=engine.swa_bounded_replay)))},
    )
    model._decoder_replay_supported = partial(DeepseekV4Model._decoder_replay_supported, model)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=engine.pipeline_parallel_size,
            data_parallel_size=engine.data_parallel_size,
            prefill_context_parallel_size=engine.prefill_context_parallel_size,
            decode_context_parallel_size=engine.decode_context_parallel_size,
            use_ubatching=engine.enable_dbo,
        ),
        compilation_config=engine.compilation_config,
        lora_config=SimpleNamespace(target_modules=engine.lora_target_modules),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=window, layer_types=["sliding_attention"]),
        )),
    )
    assert DeepseekV4Model._decoder_replay_enabled(model, config, cut)

    class ReplaySelected(Exception):
        pass

    def init_state(state, *_):
        state.vllm_config = config
        state.max_num_reqs, state.max_num_tokens = 2, 576

    def check_kept_input(batch, slot_mappings, replay_start, kept_lens, num_padded):
        assert kept_lens.tolist() == [64, window]
        assert num_padded == 64 + window
        raise ReplaySelected  # From here the real path launches the row gather.

    lora_context = object()
    wrapper = SimpleNamespace(make_token_replay_context=lambda: lora_context)
    layers = DecoderReplayLayers(
        window, None, None, set(), "swa", get_lora_wrapper=lambda: wrapper,
    )
    model.decoder_replay_layers = layers
    model.token_lookback_depth = 0
    with (
        patch.object(model_state.DefaultModelState, "__init__", init_state),
        patch.object(model_state.DefaultModelState, "add_request"),
        patch.object(model_state, "UvaBufferPool"),
    ):
        state = model_state.DeepseekV41ModelState(config, model, None, torch.device("cpu"))
        state.add_request(0, SimpleNamespace(
            lora_request=None, replay_start=0, sampling_params=None,
        ))
        state.add_request(1, SimpleNamespace(
            lora_request=object(), replay_start=0, sampling_params=None,
        ))
    assert layers.lora_context is lora_context
    state._kept_input_batch = check_kept_input
    batch = SimpleNamespace(
        num_reqs=2, idx_mapping_np=np.array([0, 1]), num_tokens_after_padding=576,
        query_start_loc_np=np.array([0, 64, 576]),
        seq_lens_cpu_upper_bound=torch.tensor([64, 512]),
        is_prefilling_np=np.array([True, True]), pcp=None,
    )
    args = (None, None, None, None, None)
    try:
        state._prepare_replay_batch(batch, CUDAGraphMode.NONE, *args)
    except ReplaySelected:
        pass
    else:
        raise AssertionError("served eager LoRA step did not select replay")
    layers.replay_batch = object()
    state._prepare_replay_batch(batch, CUDAGraphMode.FULL, *args)
    assert layers.replay_batch is None
    print("PASS decoder SWA bounded replay ON for eager adapter steps with served LoRA argv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-argv", type=Path, default=ROOT / "runtime/serve-argv.json")
    parser.add_argument("--expect-ring", choices=("0", "1"))
    options = parser.parse_args()

    import torch
    import vllm
    import vllm_exl3

    assert not torch.cuda.is_initialized()
    from vllm.plugins import load_general_plugins

    (entry,) = entry_points(group="vllm.general_plugins", name="vllm_exl3")
    assert entry.value == "vllm_exl3.bf16_madv_compat:register"
    allowed = os.environ["VLLM_PLUGINS"].split(",")
    assert tuple(allowed) == PLUGIN_ALLOWLIST
    selected = [ep.name for ep in entry_points(group="vllm.general_plugins")
                if ep.name in allowed]
    assert len(selected) == len(allowed) and set(selected) == set(allowed)
    print("General plugin registration order:", ", ".join(selected))
    load_general_plugins()
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.config import LoadConfig
    from vllm_exl3 import exl3
    from vllm_exl3.models.loader_shim import Exl3StandaloneNemotronLoader

    assert get_quantization_config("exl3") is exl3.Exl3Config
    assert isinstance(get_model_loader(LoadConfig(load_format="exl3_standalone_nemotron")),
                      Exl3StandaloneNemotronLoader)
    assert exl3._exl3_register_custom_ops()
    package = Path(vllm.__file__).resolve().parent
    import_patches(package.parent, json.loads((ROOT / "patched-files.json").read_text()),
                   package.parent)
    plugin = Path(vllm_exl3.__file__).resolve().parent
    changed = json.loads((ROOT / "plugin-patched-files.json").read_text())
    import_patches(plugin.parent, [name.removeprefix("src/") for name in changed], plugin.parent)
    overlay_roots = {"vllm": package.parent, "vllm_exl3": plugin.parent}
    files = {rel: info for entry in load_series() for rel, info in entry["files"].items()}
    check_files(overlay_roots, files, "after")
    for name, root in overlay_roots.items():
        import_patches(root, [rel for rel in files if rel.startswith(name + "/")], root)
    check_native(options.expect_ring)
    scheduler = package / "v1/core/sched/scheduler.py"
    source = ROOT / "runtime/scheduler-image.py"
    compile_serving_scheduler(source)
    check_prefill_share(source, scheduler)
    check_runtime_native()

    # A build container has no NVIDIA driver, so NVML fails and vLLM resolves UnspecifiedPlatform, whose empty
    # device type makes the serve parser raise "Failed to infer device type" (first image build, 2026-11-19).
    # The served host is CUDA: select the non-NVML CUDA platform for this driverless parser check only.
    import vllm.platforms as platforms
    if not platforms.current_platform.device_type:
        from vllm.platforms.cuda import NonNvmlCudaPlatform
        platforms._current_platform = NonNvmlCudaPlatform()
        print("NOTE driverless container: CUDA platform selected for the parser check")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.generate.base.protocol import PerRequestMetrics
    from vllm.entrypoints.serve.engine.protocol import UsageInfo
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    argv = json.loads(options.serve_argv.read_text())
    if argv[:2] == ["vllm", "serve"]:
        argv = argv[2:]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    validate_parsed_serve_args(args)
    assert args.enable_per_request_metrics is True
    assert args.per_request_spec_decode_metrics == "summary"
    assert args.disable_log_stats is False
    assert {"prefill_tokens", "prefill_tokens_per_second", "decode_tokens_per_second"} <= (
        PerRequestMetrics.model_fields.keys()
    )
    usage_timing = {
        "prompt_tokens_per_sec": 256.0,
        "completion_tokens_per_sec": 32.0,
        "prompt_time": 0.5,
        "completion_time": 0.25,
        "total_time": 1.0,
    }
    assert usage_timing.keys() <= UsageInfo.model_fields.keys()
    usage = json.loads(UsageInfo(**usage_timing).model_dump_json())
    assert {name: usage[name] for name in usage_timing} == usage_timing
    assert not usage_timing.keys() & json.loads(UsageInfo().model_dump_json()).keys()
    if args.model_tag is not None:
        args.model = args.model_tag  # Same precedence as ServeSubcommand.cmd.
    # The model bind mount is absent in a build container; with HF offline vLLM would then treat the path as a
    # repository id. Keep the path as given for this parser check (first image build, 2026-11-19).
    if not Path(args.model).exists():
        import vllm.engine.arg_utils as arg_utils
        arg_utils.get_model_path = lambda model, revision=None: model
        print("NOTE model path absent: kept as given for the parser check")
    engine = AsyncEngineArgs.from_cli_args(args)
    assert engine.model == "/model-view"
    assert args.host == "127.0.0.1" and args.port == 8001
    assert args.default_chat_template_kwargs == {"drop_thinking": False}
    assert len(args.lora_modules) == 1
    assert args.lora_modules[0].base_model_name == "deepseek-v4.1-flash-1017"
    assert engine.block_size == 128
    assert engine.gpu_memory_utilization == 0.95
    assert engine.enable_extensible_kv_cache is False
    assert engine.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
    assert engine.compilation_config.cudagraph_capture_sizes
    assert engine.compilation_config.cudagraph_specialize_lora is False
    assert engine.attention_config.sparse_mla_shard_q is False
    assert engine.attention_config.indexer_logits_dtype == "float32"
    assert engine.attention_config.indexer_sparse_logits is False
    assert engine.speculative_config["method"] == "dspark"
    assert engine.speculative_config["num_speculative_tokens"] == 5
    assert engine.speculative_config["enable_adaptive_verification"] is False
    assert engine.per_request_spec_decode_metrics == "summary"
    from vllm.entrypoints.generate.base.serving import build_spec_decoding_metrics
    from vllm.entrypoints.openai.responses.context import ConversationContext
    from vllm.v1.metrics.stats import RequestSpecDecodeMetrics

    accumulator = RequestSpecDecodeMetrics.new(5)
    accumulator.observe(num_draft_tokens=5, num_accepted=2)
    output = SimpleNamespace(outputs=[SimpleNamespace(spec_decode_metrics=accumulator)])
    summary = build_spec_decoding_metrics(output)
    assert summary is not None
    assert summary.acceptance_histogram == [0, 0, 1, 0, 0, 0]
    assert summary.mean_acceptance_length == 3 and summary.draft_acceptance_rate == 0.4
    assert ConversationContext.request_spec_decode_metrics is None
    check_decoder_replay(engine)
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    from vllm.tokenizers.registry import TokenizerRegistry

    ReasoningParserManager.get_reasoning_parser(args.reasoning_parser)
    ToolParserManager.get_tool_parser(args.tool_call_parser)
    TokenizerRegistry.load_tokenizer_cls(args.tokenizer_mode)
    assert not torch.cuda.is_initialized()
    print("PASS CPU import, plugins, native registration, scheduler transforms, serve parser")


if __name__ == "__main__":
    main()
