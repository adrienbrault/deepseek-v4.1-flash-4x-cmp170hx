"""Fail before plugin compilation if the base lacks Curtis's ExLlamaV3 ABI."""

import importlib.metadata
import json
from pathlib import Path

import torch
import exllamav3
import exllamav3_ext
for build_dependency in ("setuptools", "wheel", "ninja"):
    importlib.import_module(build_dependency)

headers = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
for name in ("util.h", "util.cuh", "quant/exl3_gemv_kernel.cuh", "exl3_dq.cuh",
             "hadamard_inner.cuh", "ptx.cuh"):
    if not (headers / name).is_file() and not (headers / "quant" / name).is_file():
        raise SystemExit(f"ExLlamaV3 development header missing: {headers / name}")
for name in ("exl3_moe", "exl3_moe_coop", "exl3_moe_max_concurrency", "had_r_128"):
    if not hasattr(exllamav3_ext, name):
        raise SystemExit(f"ExLlamaV3 extension lacks required symbol: {name}")
if int(importlib.metadata.version("setuptools").split(".")[0]) < 77:
    raise SystemExit("setuptools >=77 required by pinned plugin")
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "exllamav3": importlib.metadata.version("exllamav3"),
                  "extension": exllamav3_ext.__file__, "headers": str(headers),
                  "required_symbols": "present"}, indent=2))
