#!/usr/bin/env python3
"""Qualify the sealed r18 binary on the served configurations in one TP4 run."""
import argparse
from datetime import timedelta
from pathlib import Path
import os
import statistics
import traceback

import torch
import torch.distributed as dist
from flashinfer.comm import PcieIpcAllReduceWorkspace
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig
from vllm.distributed.device_communicators import opera_true_ring as ring

import ring_gate as gate


class GateWorkspace:
    """Let the original oracle label private configs with the policy ID."""

    def __init__(self, workspace, policy):
        self.ring = ring.RingWorkspace(workspace, policy, IpcLaunchConfig)

    def __getattr__(self, name):
        return getattr(self.ring, name)

    def all_reduce(self, inp, *, out, config):
        if int(config.variant) == gate.RING_VARIANT:
            config = None  # Exercise the exact-row serving dispatch.
        return self.ring.all_reduce(inp, out=out, config=config)


def qualify(out):
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    if dist.get_world_size() != 4:
        raise RuntimeError("r18 ring qualification requires TP4")
    gate.CPU_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    rows = ring.requested_rows()
    if not rows:
        raise RuntimeError("r18 ring qualification requires explicit served rows")
    manifest, policy = ring.read_assets()
    table, extra_sha = ring.effective_table(rows, policy)
    ring.initialize(world_size=4, hidden_dim=gate.HIDDEN, dtype=torch.bfloat16,
                    max_numel=48 * gate.HIDDEN, tbo_stream=None)
    identity = {"binary": manifest["native_library_sha256"], "extra": extra_sha, "table": table}
    peers = [None] * 4
    dist.all_gather_object(peers, identity, group=gate.CPU_GROUP)
    if any(other != identity for other in peers):
        raise RuntimeError("r18 qualification ranks disagree on ring assets")
    gate.emit(out, {"event": "native_candidate_verified", **identity})
    workspace = PcieIpcAllReduceWorkspace(
        group=dist.group.WORLD, max_numel=48 * gate.HIDDEN, dtype=torch.bfloat16,
        tune_batches=rows, tune_cache=str(out / "stock-tune.json"),
    )
    if workspace.memop_supported:
        raise RuntimeError("unexpected SM120 publication on the SM80 qualification workspace")
    workspace.tune([gate.HIDDEN], dtype=torch.bfloat16, tune_group=gate.CPU_GROUP)
    stock = workspace.prepare([(n, gate.HIDDEN) for n in rows], dtype=torch.bfloat16)
    bridge = GateWorkspace(workspace, dict(policy, ring_configs=table))
    selected = {}
    for n in rows:
        base = stock[n, gate.HIDDEN]
        if base is None or int(base.variant) >= gate.RING_VARIANT:
            raise RuntimeError(f"row {n} lacks an SM80 stock control")
        private = IpcLaunchConfig(**table[str(n)])
        gate.check_config(bridge, base, n, out)
        gate.check_config(bridge, private, n, out)
        selected[n] = (base, private)
        timings = {"stock": [], "ring": []}
        for turn in range(2):
            for label, config in (("stock", base), ("ring", private),
                                  ("ring", private), ("stock", base)):
                sample = gate.timing(bridge, config, n)
                timings[label].append(sample["median_us"])
                gate.emit(out, dict(sample, event="paired", row=n, tag=label, turn=turn))
        gate.emit(out, {"event": "shape_result", "row": n, "ring_config": table[str(n)],
                        "stock_config": gate.record_config(base),
                        **{f"{label}_us": statistics.median(values)
                           for label, values in timings.items()}})
    gate.SHAPES = rows
    gate.mixed_shape_check(bridge, selected, out)
    torch.cuda.synchronize()
    gate.barrier()
    workspace.destroy()
    gate.barrier()
    gate.emit(out, {"event": "PASS", "rows": rows, **identity})
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    try:
        qualify(parser.parse_args().out)
    except BaseException:
        # Avoid releasing shared IPC slabs while a peer may be in a kernel.
        traceback.print_exc()
        os._exit(1)
