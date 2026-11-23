"""Fresh-process ABI/registration check; no CUDA context or GPU required."""
import importlib.util
import hashlib
import json
from pathlib import Path

import torch

root = Path("/opt/dsv41-tc-moe")
spec = importlib.util.spec_from_file_location(
    "_dsv41_tc_moe_C", root / "_dsv41_tc_moe_C.abi3.so")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
info = module.build_info()
receipt = json.loads((root / "build.json").read_text())
assert receipt["torch"] == torch.__version__
assert receipt["sha256"] == hashlib.sha256(
    (root / "_dsv41_tc_moe_C.abi3.so").read_bytes()).hexdigest()
# Some nightly headers omit the Python dev/local suffix. The receipt pins the
# full server build; also verify the compiled major/minor and native ABIs.
assert info["torch_version"].split(".")[:2] == torch.__version__.split(".")[:2]
assert info["cuda_version"] == 13000
assert info["cxx11_abi"] == int(torch._C._GLIBCXX_USE_CXX11_ABI)
ops = torch.ops._dsv41_tc_moe_C
for name in ("moe_out_v2", "moe_out_prefill"):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "_dsv41_tc_moe_C::" + name, "CUDA")
sizes = dict(v2=ops.moe_workspace_size_v2(64, 5120, 640, 6, 384),
             prefill=ops.moe_workspace_size_prefill(2048, 5120, 640, 6, 384))
print(json.dumps(dict(abi=info, workspace_float32_elements=sizes)))
