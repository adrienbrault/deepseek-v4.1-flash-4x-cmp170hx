# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline sparse-decode policy. No device reads or tuning during graph capture."""

import functools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SparseDecodeConfig:
    block_h: int = 16
    block_k: int = 32
    warps: int = 4
    stages: int = 1
    splits: int = 1
    decode: str = "bits"
    legacy: bool = False
    pv_precision: str = "bf16"
    rope_round: str = "fused"

    def __post_init__(self):
        if (
            any(
                type(value) is not int
                for value in (
                    self.block_h,
                    self.block_k,
                    self.warps,
                    self.stages,
                    self.splits,
                )
            )
            or self.block_h not in (16, 32)
            or self.block_k not in (16, 32, 64)
            or self.warps not in (4, 8)
            or self.stages not in (1, 2)
            or not 1 <= self.splits <= 16
            or self.decode not in ("alu", "bits", "lut")
            or type(self.legacy) is not bool
            or self.pv_precision not in ("bf16", "bf16x2", "fp32")
            or self.rope_round not in ("fused", "bf16")
            or (self.legacy and (self.pv_precision != "bf16" or self.rope_round != "fused"))
        ):
            raise ValueError("Invalid SM80 sparse decode configuration")

    def to_dict(self):
        return asdict(self)


def shape_key(rows: int, heads: int, main_capacity: int, topk: int) -> str:
    # Capacities come from allocation shapes, never indptr contents. Ragged
    # lengths can change on replay without changing the compiled policy.
    return f"{rows}:{heads}:{main_capacity}:{topk}"


@functools.cache
def sparse_decode_enabled() -> bool:
    return os.getenv("VLLM_SM80_SPARSE_DECODE", "0") == "1"


@functools.cache
def load_table() -> dict:
    path = os.getenv("VLLM_SM80_SPARSE_CONFIG")
    if not path:
        return {}
    table = json.loads(Path(path).read_text())
    if table.get("version") != 1 or not isinstance(table.get("configs"), dict):
        raise ValueError("Invalid SM80 sparse decode table")
    for config in table["configs"].values():
        SparseDecodeConfig(**config)
    return table


def validate_device(table: dict, device: dict) -> None:
    recorded = table.get("device")
    if table and (
        recorded is None
        or any(recorded.get(k) != device[k] for k in ("name", "sm_count", "capability"))
    ):
        raise ValueError("SM80 sparse decode table belongs to a different GPU SKU")


def select_config(
    rows: int,
    heads: int,
    main_capacity: int,
    topk: int,
    sm_count: int,
    table: dict,
) -> SparseDecodeConfig:
    key = shape_key(rows, heads, main_capacity, topk)
    if key in table.get("configs", {}):
        return SparseDecodeConfig(**table["configs"][key])
    # Seed only: the operator replaces it with graph timings on their SKU.
    base = max(1, rows * ((heads + 15) // 16))
    if base >= sm_count:
        return SparseDecodeConfig()
    tiles = max(1, (main_capacity + topk + 31) // 32)

    def cost(splits):
        waves = (base * splits + max(1, sm_count) - 1) // max(1, sm_count)
        iterations = sum(
            ((length + splits - 1) // splits + 31) // 32
            for length in (main_capacity, topk)
        )
        return waves * (iterations + 0.15) + 0.01 * splits

    splits = min(range(1, min(16, tiles) + 1), key=cost)
    return SparseDecodeConfig(splits=splits)
