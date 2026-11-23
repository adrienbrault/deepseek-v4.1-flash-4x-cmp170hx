"""Build-time import and public API gate; does not initialize a GPU."""
import inspect
import json
from pathlib import Path

import vllm_exl3
vllm_exl3.register()
from vllm_exl3 import exl3
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

assert exl3._VLLM_AVAILABLE
assert get_quantization_config("exl3") is exl3.Exl3Config
required = tuple(inspect.signature(FusedMoEMethodBase.apply).parameters)
assert tuple(inspect.signature(exl3.Exl3MoEMethod.apply).parameters) == required
config = exl3.Exl3Config.from_config(json.loads(
    (Path(__file__).resolve().parents[1] / "quantization_config.json").read_text()
))
assert config.weight_block_size == [32, 32]
print("PASS quant registration, fused-MoE apply signature, source scale geometry")
