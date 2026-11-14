# SPDX-License-Identifier: Apache-2.0
"""Curtis White's DeepSeek TC-MoE hook, adapted to the r7 plugin.

Kernels: Morrowmake; DeepSeek port and original hook: Curtis White
(CurtisAccelerate). See tc_moe/NOTICE.md and codec.cuh for provenance.
"""

import logging
import math
import os

import torch

logger = logging.getLogger("vllm.exl3.tc_moe")
_OPS = None
_WORKSPACES = {}
HIDDEN, EXPERTS, TOPK = 5120, 384, 6


def settings():
    enabled = os.getenv("DSV41_TC_MOE", "0") == "1"
    v2 = enabled and os.getenv("VLLM_EXL3_TC_MOE_V2", "1") == "1"
    prefill = enabled and os.getenv("VLLM_EXL3_TC_MOE_PREFILL", "1") == "1"
    cap = int(os.getenv("VLLM_EXL3_TC_PREFILL_ROWS", "2048"))
    if not 65 <= cap <= 2048:
        raise ValueError("VLLM_EXL3_TC_PREFILL_ROWS must be 65..2048")
    return v2, prefill, cap


def select_path(rows, v2, prefill, cap):
    if v2 and 43 <= rows <= 64:
        return "v2"
    if prefill and rows >= 65:
        # The native prefill op subchunks internally at cap, on one stream.
        return "prefill"
    return None


def _log_once(layer, reason):
    reasons = getattr(layer, "_exl3_tc_reasons", None)
    if reasons is None:
        reasons = layer._exl3_tc_reasons = set()
    if reason not in reasons:
        reasons.add(reason)
        name = (getattr(layer, "layer_name", None)
                or getattr(layer, "prefix", None)
                or getattr(layer, "_exl3_layer_idx", "?"))
        logger.info("EXL3 TC MoE layer=%s fallback to coop/stock: %s", name, reason)


def reject(layer, inners):
    if len(inners) != EXPERTS:
        return f"local experts {len(inners)} (need {EXPERTS})"
    if (getattr(layer, "_exl3_mixed_k", False)
            or getattr(layer, "_exl3_mixedk_unified", False)
            or getattr(layer, "_exl3_mixed_store", None) is not None):
        return "heterogeneous K within a layer"
    if getattr(layer, "_exl3_hidden_size", None) != HIDDEN:
        return "hidden width"
    inter = getattr(layer, "_exl3_intermediate_local", None)
    if inter not in (512, 640):
        return f"intermediate {inter} (need 512/640)"
    act = getattr(getattr(layer, "moe_config", None), "activation", None)
    if getattr(act, "value", act) not in (None, "silu"):
        return "activation (need gated silu)"
    if tuple(getattr(layer, "_exl3_codebook_flags", ())) != (False, True) * 3:
        return "codebook (need mul1 on all projections)"
    first = inners[0].get("gate")
    if first is None:
        return "gateless"
    device = first.trellis.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        return "device (need sm_80 CUDA)"
    bits = first.trellis.shape[-1] // 8
    if bits not in (4, 6):
        return f"bits_x2 {bits} (need K2/K3)"
    for pack in inners:
        for name in ("gate", "up", "down"):
            linear = pack.get(name)
            if linear is None:
                return f"missing {name}"
            if getattr(linear, "bias", None) is not None:
                return f"{name} bias"
            n_in, n_out = (inter, HIDDEN) if name == "down" else (HIDDEN, inter)
            expected = (n_in // 16, n_out // 16, bits * 8)
            t, a, b = linear.trellis, linear.suh, linear.svh
            if tuple(t.shape) != expected or t.dtype != torch.int16:
                return f"{name} trellis geometry/K/dtype"
            if a.numel() != n_in or b.numel() != n_out:
                return f"{name} scale lengths"
            if a.dtype != torch.float16 or b.dtype != torch.float16:
                return f"{name} scale dtype"
            for tensor in (t, a, b):
                if (tensor.device != device or not tensor.is_contiguous()
                        or tensor.data_ptr() % 16):
                    return f"{name} device/layout/alignment"
    return None


def _ops():
    global _OPS
    if _OPS is None:
        torch.ops.load_library(os.getenv(
            "VLLM_EXL3_TC_MOE_SO", "/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so"))
        _OPS = torch.ops._dsv41_tc_moe_C
    return _OPS


def try_build(layer, inners):
    """Bind final arena views after load; never repack or expand weights."""
    layer._exl3_tc = None
    if os.getenv("DSV41_TC_MOE", "0") != "1":
        return
    try:
        v2, prefill, cap = settings()
        reason = reject(layer, inners)
        if reason or not (v2 or prefill):
            _log_once(layer, reason or "both paths off")
            return
        ops = _ops()
        device = inners[0]["gate"].trellis.device
        inter = int(layer._exl3_intermediate_local)
        bits = int(inners[0]["gate"].trellis.shape[-1] // 8)
        ptrs, meta = [], []
        for pack in inners:
            pp, mm = [], []
            for name in ("gate", "up", "down"):
                linear = pack[name]
                pp.append([v.data_ptr() for v in
                           (linear.trellis, linear.suh, linear.svh)])
                dims = (inter, HIDDEN) if name == "down" else (HIDDEN, inter)
                mm.append([*dims, bits, 2])
            ptrs.append(pp)
            meta.append(mm)
        # One H=5120,I=640 workspace per device; all routed layers run in order.
        key = (device.index, v2, prefill, cap)
        workspace = _WORKSPACES.get(key)
        if workspace is None:
            nv = ops.moe_workspace_size_v2(64, HIDDEN, 640, TOPK, EXPERTS) if v2 else 0
            np = (ops.moe_workspace_size_prefill(cap, HIDDEN, 640, TOPK, EXPERTS)
                  if prefill else 0)
            workspace = torch.empty(max(nv, np), dtype=torch.float32, device=device)
            _WORKSPACES[key] = workspace
        tc = dict(ops=ops, ptrs=torch.tensor(ptrs, dtype=torch.int64, device=device),
                  meta=torch.tensor(meta, dtype=torch.int64, device=device),
                  emap=torch.arange(EXPERTS, dtype=torch.int32, device=device),
                  workspace=workspace, intermediate=inter, bits_x2=bits,
                  v2=v2, prefill=prefill, cap=cap,
                  owners=inners)
        # Populate per-kernel occupancy/shared-memory statics before graph capture.
        for path, rows in (("v2", 48), ("prefill", 65)):
            if not tc[path]:
                continue
            x = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.zeros((rows, TOPK), dtype=torch.int64, device=device)
            weights = torch.full((rows, TOPK), 1 / TOPK,
                                 dtype=torch.float32, device=device)
            out = torch.empty_like(x)
            getattr(ops, "moe_out_" + path)(
                x, ids, weights, tc["ptrs"], tc["meta"], tc["emap"], inter,
                bits, 10.0, out, workspace)
        torch.cuda.current_stream(device).synchronize()
        layer._exl3_tc = tc
        _log_once(layer, "rows 1..42 retain coop; TC begins at 43")
        logger.info("EXL3 TC MoE ready layer=%s I=%s K=%s v2=%s prefill=%s "
                    "cap=%s shared_ws=%.1fMiB (Curtis White / Morrowmake)",
                    getattr(layer, "prefix", getattr(layer, "layer_name", "?")),
                    inter, bits // 2, v2, prefill, cap,
                    workspace.numel() * 4 / 1048576)
    except Exception as exc:
        _log_once(layer, f"setup failed: {exc}")


def try_apply(x, ids, weights, layer, expert_map, limit):
    tc = getattr(layer, "_exl3_tc", None)
    if tc is None:
        return None
    if x.dim() != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        _log_once(layer, "activation geometry/dtype (need BF16)")
        return None
    if (ids.dim() != 2 or tuple(ids.shape) != (x.shape[0], TOPK)
            or tuple(weights.shape) != tuple(ids.shape)
            or ids.dtype not in (torch.int32, torch.int64)):
        _log_once(layer, "routing geometry/dtype (need top-6)")
        return None
    emap = tc["emap"] if expert_map is None else expert_map
    if (not x.is_cuda or any(t.device != x.device for t in
                            (ids, weights, emap, tc["ptrs"]))
            or emap.dim() != 1 or emap.numel() != EXPERTS
            or emap.dtype not in (torch.int32, torch.int64)):
        _log_once(layer, "routing device/map")
        return None
    if limit is not None and not math.isfinite(float(limit)):
        _log_once(layer, "nonfinite clamp")
        return None
    path = select_path(int(x.shape[0]), tc["v2"], tc["prefill"], tc["cap"])
    if path is None:
        return None
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    # Do not catch launch failures: CUDA errors cannot safely become coop outputs.
    getattr(tc["ops"], "moe_out_" + path)(
        x.contiguous(), ids.contiguous(), weights.float().contiguous(),
        tc["ptrs"], tc["meta"], emap.contiguous(), tc["intermediate"],
        tc["bits_x2"], float(limit or 0), out, tc["workspace"])
    return out
