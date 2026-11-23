# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD rows only: retain the original TP gather and WKV on their caller stream.

Scheduling/lifetime pattern adapted from Morrowmake's sm80_engram_prefetch.py.
No EXL3 projection or other shared scratch runs on this stream. One permanent
stream per Store serializes ALL capacities, eager calls and captured replays.
Supported contract: one target forward in flight, DP1, no SP/DBO, eager or FULL.
"""
import os


def initialize(layer, config):
    layer._gap_stream = None
    layer._gap_done = None
    requested = os.environ.get("DSV41_EG_PREFETCH", "0")
    if requested not in ("0", "1"):
        raise ValueError("DSV41_EG_PREFETCH must be 0 or 1")
    if requested == "0":
        return
    table = layer.embed_tokens
    parallel = config.parallel_config
    mode = config.compilation_config.cudagraph_mode
    if (not table._ssd or table.dp_size != 1 or table.cpu_offload
            or layer.use_sequence_parallel or parallel.use_ubatching
            or getattr(parallel, "num_ubatches", 1) > 1
            or not config.use_v2_model_runner
            or os.environ.get("DSV41_ENGRAM_SSD_ASYNC", "0") != "0"
            or (mode is not None and mode.has_piecewise_cudagraphs())):
        raise ValueError("EG prefetch requires SSD, DP1, V2, no SP/DBO/old async, eager or FULL")
    import torch
    store = table.ssd_table
    if store is None:
        raise ValueError("SSD Store must be attached before initializing prefetch")
    stream = getattr(store, "_gap_lookup_stream", None)
    if stream is None:
        stream = torch.cuda.Stream(device=layer.staged_rows.device)
        store._gap_lookup_stream = stream
    layer._gap_stream = stream
    layer._gap_done = torch.cuda.Event()
    print(f"EG_PREFETCH layer={layer.layer_hash_index} rows-only DP1 FULL/eager", flush=True)


def prepare(layer, hashes, rows):
    import torch
    current = torch.cuda.current_stream(hashes.device)
    side = layer._gap_stream
    side.wait_stream(current)
    hashes.record_stream(side)
    rows.record_stream(side)
    with torch.cuda.stream(side):
        layer.embed_tokens.lookup(hashes, rows)
        layer._gap_done.record(side)


def join(layer):
    if layer._gap_done is not None:
        import torch
        torch.cuda.current_stream().wait_event(layer._gap_done)
