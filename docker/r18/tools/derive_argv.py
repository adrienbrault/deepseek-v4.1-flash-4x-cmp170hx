#!/usr/bin/env python3
"""Derive r18 test argv from a sealed Docker run argument array."""
import argparse
import json
from pathlib import Path
import re

from prefill_share import MODE_ENV, MODES

DOCKER_VALUES = set("""
--name --network --ipc --shm-size --gpus --cap-add --ulimit --memory
--memory-swap --runtime --cgroupns --log-driver --env -e --volume -v --mount
""".split())
IMAGE_DEFAULTS = set("""
PATH NVARCH NVIDIA_REQUIRE_CUDA NV_CUDA_CUDART_VERSION CUDA_VERSION LD_LIBRARY_PATH
NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES DEBIAN_FRONTEND CUDA_HOME
CRIU_CUDA_PLUGIN_DIR UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE UV_PYTHON_INSTALL_DIR
UV_CACHE_DIR UV_COMPILE_BYTECODE UV_OVERRIDE PIP_NO_INDEX PIP_DISABLE_PIP_VERSION_CHECK
UV_OFFLINE MAX_JOBS EXL3_EXT_INCLUDE VLLM_USAGE_SOURCE VLLM_BUILD_COMMIT
VLLM_BUILD_PIPELINE VLLM_BUILD_URL VLLM_IMAGE_TAG
""".split())
FAIR = {
    "VLLM_DSV41_FAIR_PREFILL": "1024",
    "VLLM_DSV41_FAIR_PREFILL_SPLIT": "2",
    "VLLM_DSV41_FAIR_PREFILL_MIN_TOKENS": "2048",
}
DATA_TARGETS = {
    "/model-view", "/official", "/configs", "/configs-hav/policy.json",
    "/lora/deepseek-v4.1-flash-1017-abliterated", "/dsv41-sparse/sparse.json",
    "/opt/dsv41-calibrated-depth.json",
}
FRESH_TARGETS = {
    "/root/.cache": "cache",
    "/root/.tilelang": "tilelang",
    "/root/.triton": "triton",
    "/results": "results",
}
BAKED_NATIVE_SEALS = ("DSV41_INDEXER_NATIVE_SEAL_SHA256", "DSV41_STACK_NATIVE_SEAL")
PLUGIN_ALLOWLIST = (
    "vllm_exl3", "dsv41_lora_argmax", "dsv41_attn_port", "dsv41_attn_audit",
    "dsv41_rowstats", "dsv41_wkv_replica", "dsv41_dispatch_fuse", "dsv41_ablit_r7",
    "dsv41_indexer_bounded", "dsv41_moe_stack", "dsv41_prefill_7b", "dsv41_fly_names",
)
REDUCED_ENV = dict.fromkeys((
    "DSV41_TARGET_ARGMAX", "DSV41_ABLIT_R7", "DSV41_DRAFT_SHARD_ARGMAX",
    "DSV41_EG_DEQUANT", "DSV41_EG_PREFETCH", "DSV41_EG_WKV_REPLICA",
    "DSV41_EG_ROW_CACHE_MB", "DSV41_ATTN_SWA_FUSED", "DSV41_ATTN_TOPK_GLUE",
    "DSV41_ROWSTATS_LEAN", "DSV41_DISPATCH_FUSE", "DSV41_INDEXER_BOUNDED",
    "DSV41_MOE_STACK", "DSV41_PREFILL_7B_R7D", "DSV41_FLY_SERVING",
    "VLLM_EXL3_COOP_KERNEL",
), "0") | {"VLLM_DSV41_MOE_ROW_MASK": ""}


def split_run(argv):
    """Read only the Docker grammar present in the captured launcher."""
    if argv[:1] == ["docker"]:
        argv = argv[1:]
    if argv[:2] == ["run", "--pull=never"]:
        argv = [argv[0], *argv[2:]]
    if argv[:1] != ["run"] or any(not isinstance(x, str) for x in argv):
        raise ValueError("Expected a Docker run argument array")
    options = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-d":
            options.append((arg, None))
            i += 1
        elif arg in DOCKER_VALUES and i + 1 < len(argv):
            options.append((arg, argv[i + 1]))
            i += 2
        elif re.fullmatch(r"sha256:[0-9a-f]{64}", arg):
            return options, argv[i + 1:]
        else:
            raise ValueError(f"Unexpected Docker option before sealed image at index {i}")
    raise ValueError("Missing sealed image ID")


def serve_args(argv):
    args = list(argv)
    if not args or args[0].startswith("-"):
        raise ValueError("Expected the positional model argument")
    for flag in ("--host", "--port", "--attention-config", "--gpu-memory-utilization"):
        if args.count(flag) != 1:
            raise ValueError(f"Expected one {flag}")
    args[args.index("--host") + 1] = "127.0.0.1"
    args[args.index("--port") + 1] = "8001"
    if args[args.index("--gpu-memory-utilization") + 1] != "0.95":
        raise ValueError("Expected the sealed 0.95 GPU memory setting")
    index = args.index("--attention-config") + 1
    attention = json.loads(args[index])
    attention.update(sparse_mla_shard_q=False, indexer_logits_dtype="float32",
                     indexer_sparse_logits=False)
    args[index] = json.dumps(attention, separators=(",", ":"))
    if any(x.split("=")[0] in (
        "--block-size", "--enable-extensible-kv-cache", "--no-enable-extensible-kv-cache",
    ) for x in args):
        raise ValueError("Expected sealed argv without explicit v0.34 KV settings")
    if any(
        arg.partition("=")[0].replace("_", "-") == "--disable-log-stats" for arg in args
    ):
        raise ValueError("--enable-per-request-metrics requires removing --disable-log-stats")
    return args + [
        "--block-size", "128", "--no-enable-extensible-kv-cache",
        "--enable-per-request-metrics", "--per-request-spec-decode-metrics", "summary",
    ]


def mount_parts(option, value):
    if option in ("-v", "--volume"):
        source, target, mode = value.split(":")
        return source, target, mode
    kind, source, target, mode = value.split(",")
    if kind != "type=bind" or not source.startswith("source=") or not target.startswith("target="):
        raise ValueError("Expected a sealed bind mount")
    return source.removeprefix("source="), target.removeprefix("target="), mode


def derive(options, serve, image_id, runtime_root, profile, *, reduced=False, ring=False,
           prefill_share="off", fair_off=False, fair_min_tokens=None):
    """Keep data and writable caches as binds. The kit owns package code."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Use the r18 image ID from docker image inspect")
    if not runtime_root.is_absolute():
        raise ValueError("Runtime root must be absolute")
    if profile not in ("r7g", "r7h"):
        raise ValueError("Unknown serving profile")
    if prefill_share not in MODES:
        raise ValueError("Unknown prefill share mode")
    if fair_min_tokens is not None and (fair_min_tokens < 0 or profile != "r7h"):
        raise ValueError("Fair minimum requires a nonnegative token count and the r7h profile")
    root = runtime_root / image_id.removeprefix("sha256:")
    result = ["run", "--pull=never"]
    plugins_seen = False
    ring_shapes = None
    for flag, value in options:
        if value is None:
            result.append(flag)
            continue
        if flag == "--name":
            value = f"dsv41-r18-{profile}-test"
        elif flag in ("-e", "--env"):
            key, eq, old_value = value.partition("=")
            if not eq:
                raise ValueError("Sealed env must contain a value")
            if key == "DSV41_TRUE_RING_SHAPES":
                if ring_shapes is not None:
                    raise ValueError("Duplicate captured ring shape list")
                ring_shapes = old_value
                continue
            if key in IMAGE_DEFAULTS or key == "VLLM_USE_V2_MODEL_RUNNER":
                continue
            if key in BAKED_NATIVE_SEALS:
                continue  # Read the image-owned native seal identities.
            if key in (MODE_ENV, "VLLM_SERVER_DEV_MODE"):
                continue  # Select test-only switches explicitly for each boot.
            if key == "VLLM_PLUGINS":
                if plugins_seen or tuple(old_value.split(",")) != PLUGIN_ALLOWLIST:
                    raise ValueError("Unexpected captured plugin allowlist")
                plugins_seen = True
            if key in FAIR:
                if profile == "r7g":
                    continue
                if key == "VLLM_DSV41_FAIR_PREFILL_MIN_TOKENS" and old_value == "<redacted>":
                    old_value = FAIR[key]
                if old_value != FAIR[key]:
                    raise ValueError(f"Unexpected r7h policy value for {key}")
                if key == "VLLM_DSV41_FAIR_PREFILL_MIN_TOKENS":
                    if fair_min_tokens is not None:
                        old_value = str(fair_min_tokens)
                elif fair_off:
                    old_value = "0"
            if key == "DSV41_TRUE_RING":
                continue
            value = f"{key}={REDUCED_ENV.get(key, old_value) if reduced else old_value}"
        elif flag in ("-v", "--volume", "--mount"):
            source, target, mode = mount_parts(flag, value)
            if target in FRESH_TARGETS:
                source = root / FRESH_TARGETS[target]
            elif target not in DATA_TARGETS:
                continue
            value = (f"type=bind,source={source},target={target},{mode}" if flag == "--mount"
                     else f"{source}:{target}:{mode}")
        result += [flag, value]
    if not plugins_seen:
        raise ValueError("Missing captured plugin allowlist")
    result += [
        "-e", f"DSV41_TRUE_RING={int(ring)}",
        "-e", "FLASHINFER_WORKSPACE_BASE=/root/.cache",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor",
        "-e", "CUDA_CACHE_PATH=/root/.cache/cuda",
    ]
    if ring:
        if not ring_shapes:
            raise ValueError("Ring launch requires the captured exact shape list")
        result += ["-e", f"DSV41_TRUE_RING_SHAPES={ring_shapes}"]
    if prefill_share != "off":
        result += ["-e", f"{MODE_ENV}={prefill_share}"]
    return result + [image_id, *serve]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_argv", type=Path)
    parser.add_argument("--serve-only", action="store_true", help="emit server arguments for CPU qualification")
    parser.add_argument("--profile", choices=("r7g", "r7h"), default="r7g")
    parser.add_argument("--image-id")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--reduced", action="store_true", help="start with optional runtime hooks off")
    parser.add_argument("--ring", action="store_true", help="enable the image-built true ring on the captured rows")
    parser.add_argument("--prefill-share", choices=MODES, default="off")
    parser.add_argument("--fair-off", action="store_true", help="disable the r7h cap and split")
    parser.add_argument("--fair-min-tokens", type=int, help="override the r7h minimum remaining tokens")
    args = parser.parse_args()
    try:
        options, sealed_serve = split_run(json.loads(args.sealed_argv.read_text()))
        serve = serve_args(sealed_serve)
        if args.serve_only:
            result = serve
        else:
            if not all((args.image_id, args.runtime_root)):
                parser.error("Full argv requires image ID and runtime root")
            result = derive(options, serve, args.image_id, args.runtime_root,
                            args.profile, reduced=args.reduced, ring=args.ring,
                            prefill_share=args.prefill_share, fair_off=args.fair_off,
                            fair_min_tokens=args.fair_min_tokens)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
