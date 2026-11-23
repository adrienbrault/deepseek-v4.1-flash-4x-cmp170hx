"""Check the pinned v0.34 release image before compiling local extensions."""
import importlib.metadata as md
from pathlib import Path
import re
import shutil
import sysconfig
import json
import platform
import site
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
assert re.fullmatch(r"vllm/vllm-openai:v0\.34\.0@sha256:[a-f0-9]{64}", sys.argv[1]), (
    "Set VLLM_BASE to the operator-verified linux/amd64 v0.34.0 manifest digest"
)
assert platform.machine() == "x86_64", platform.machine()
assert md.version("vllm").split("+")[0] == "0.34.0", md.version("vllm")
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == "2.14.0+cu130", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert md.version("flashinfer-python") == "0.7.3"
for tool in ("g++", "nvcc", "patch", "ldd"):
    assert shutil.which(tool), f"release image lacks required build tool: {tool}"
for wheel in sorted((ROOT / "offline").glob("*.whl")):
    assert not wheel.name.lower().startswith(("torch-", "vllm-", "vllm_exl3-", "exllamav3-")), wheel
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], check=True)
for dep in ("setuptools", "wheel", "ninja"):
    md.version(dep)
assert int(md.version("setuptools").split(".")[0]) >= 77
from torch.utils.cpp_extension import CUDA_HOME

assert CUDA_HOME, "CUDA toolkit required"
include = Path(CUDA_HOME) / "include"
for package_root in site.getsitepackages():
    for folder in Path(package_root).glob("nvidia/*/include"):
        for header in folder.iterdir():
            dest = include / header.name
            if not dest.exists():
                dest.symlink_to(header)
for header in ("cuda_runtime.h", "cusparse.h", "cublas_v2.h"):
    assert (include / header).is_file(), f"missing CUDA 13.0 header: {header}"
receipt = {
    "base": sys.argv[1],
    "python": sys.version,
    "soabi": sysconfig.get_config_var("SOABI"),
    "libc": platform.libc_ver(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "packages": {name: md.version(name) for name in (
        "vllm", "flashinfer-python", "apache-tvm-ffi", "triton", "tilelang",
        "setuptools", "wheel", "ninja"
    )},
    "toolchain": {tool: subprocess.check_output([tool, "--version"], text=True).strip()
                  for tool in ("g++", "nvcc", "ldd")},
}
(ROOT / "build-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
