#!/usr/bin/env python3
"""CPU build and sealing for the r18 runtime native assets."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

from install_overlays import locate_site, sha

KIT = Path(__file__).resolve().parents[1]
SOURCE = KIT / "native/runtime"
INDEXER = Path("/opt/indexer-native")
STACK = Path("/opt/stack-native")
ROW = Path("/opt/dsv41-engram-gap/librow_store.so")
PREFILL = Path("/opt/prefill-bitwise/out/lib/_dsv41_mm_try.abi3.so")
QV = Path("/opt/qv")
NAME = "indexer_bounded_k1b"
TORCH_LIBRARIES = ("libc10.so", "libc10_cuda.so", "libtorch_cpu.so",
                   "libtorch_cuda.so", "libtorch_python.so")


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(path.read_text())


def identity(torch):
    return dict(torch_version=str(torch.__version__), torch_git=torch.version.git_version,
                cuda_version=torch.version.cuda, python_soabi=sysconfig.get_config_var("SOABI"),
                cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))


def torch_hashes(torch):
    root = Path(torch.__file__).parent / "lib"
    return {name: sha(root / name) for name in TORCH_LIBRARIES}


def build_indexer(torch, libraries):
    from torch.utils.cpp_extension import load

    source = SOURCE / "indexer"
    old = read(source / "incumbent-build.json")
    seal = read(source / "incumbent-seal.json")
    # Old receipt is used for source/flag provenance. The old wheel/header
    # receipt and GPU qualification never transfer to a rebuild.
    if sha(source / "incumbent-build.json") != seal["receipt_sha256"]:
        raise ValueError("Indexer's incumbent build receipt changed")
    files = {name: sha(source / name) for name in ("bounded.cpp", "bounded.cu", "sort.inc")}
    if files != seal["source_sha256"] or files != old["sources"]:
        raise ValueError("Indexer's sealed kernel sources changed")
    cccl = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "targets/x86_64-linux/include/cccl"
    INDEXER.mkdir()
    obj = load(name=NAME, sources=[str(source / n) for n in ("bounded.cpp", "bounded.cu")],
               build_directory=str(INDEXER), extra_cflags=old["cflags"],
               extra_cuda_cflags=old["cuda_flags"], extra_include_paths=[str(cccl)],
               verbose=True)
    lib = Path(obj.__file__)
    if lib != INDEXER / (NAME + ".so") or not callable(obj.run):
        raise ValueError("Unexpected indexer extension path or ABI")
    receipt = INDEXER / "build-receipt.json"
    write(receipt, dict(schema="r18-indexer-build/1", abi=identity(torch), sources=files,
                        torch_libraries=libraries, arch=os.environ["TORCH_CUDA_ARCH_LIST"],
                        cflags=old["cflags"], cuda_flags=old["cuda_flags"],
                        cub_header_sha256=sha(cccl / "cub/version.cuh"),
                        artifacts={p.name: sha(p) for p in (lib, INDEXER / "build.ninja")}))
    seal_path = QV / "native-seal.json"
    glue = locate_site("dsv41_attn_port") / "dsv41_attn_port/topk_glue.py"
    write(seal_path, dict(schema="r18-indexer-native/1", gpu_qualification="PENDING",
                          binary=lib.name, binary_sha256=sha(lib), receipt_sha256=sha(receipt),
                          glue_sha256=sha(glue), source_sha256=files,
                          parent_seal_sha256=sha(source / "incumbent-seal.json")))
    (QV / "native-seal.sha256").write_text(sha(seal_path) + "\n")
    for path in INDEXER.iterdir():
        if path.name not in (lib.name, "build.ninja", receipt.name):
            path.unlink()


def build_rowstore():
    source = SOURCE / "row_store.cpp"
    ROW.parent.mkdir()
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                    "-fPIC", "-shared", "-pthread", str(source), "-o", str(ROW)],
                   check=True)
    if ctypes.CDLL(str(ROW)).row_store_gap_revision() != 1:
        raise ValueError("Unexpected row-store gap ABI")
    write(ROW.with_name("build.json"), dict(source_sha256=sha(source), binary_sha256=sha(ROW)))


def build_prefill(torch):
    source = SOURCE / "prefill"
    script = source / "build_try.py"
    build = source / "build"
    subprocess.run([sys.executable, str(script), "--out", str(PREFILL.parent),
                    "--build-dir", str(build), "--verbose"], check=True)
    shutil.rmtree(build)
    write(PREFILL.with_name("build.json"), dict(
        abi=identity(torch), binary_sha256=sha(PREFILL),
        sources={p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()},
        gpu_qualification="PENDING"))


def check_stack_binary(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.opera_coop_variant_v1() != 1 or not callable(module.exl3_moe_coop):
        raise ValueError(f"Unexpected stack extension ABI: {name}")


def copy_stack(torch, libraries):
    source = SOURCE / "stack"
    old = read(source / "incumbent-seal.json")
    if (old["schema"] != "moe-stack-native/1" or not old["no_rebuild"]
            or set(old["binaries"]) != {"opera_rows_family.so", "opera_k2_fixed16.so"}):
        raise ValueError("Unexpected measured stack bundle")
    for build in old["builds"].values():
        if build["torch"] != torch.__version__ or build["cuda"] != torch.version.cuda:
            raise ValueError("Measured stack Torch/CUDA version differs")
    STACK.mkdir()
    for name, digest in old["binaries"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Measured stack binary changed: {name}")
        path = STACK / name
        shutil.copyfile(source / name, path)
        check_stack_binary(path, path.stem)
    seal = STACK / "seal.json"
    write(seal, dict(schema="r18-stack-native/1", gpu_qualification="PENDING", binaries=old["binaries"],
                     abi=identity(torch), torch_libraries=libraries, no_rebuild=True,
                     parent_seal_sha256=sha(source / "incumbent-seal.json")))
    (STACK / "seal.sha256").write_text(sha(seal) + "\n")
    for name in old["binaries"]:
        (source / name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.0":
        raise ValueError("Expected SM80 build")
    libraries = torch_hashes(torch)
    build_rowstore()
    build_indexer(torch, libraries)
    build_prefill(torch)
    copy_stack(torch, libraries)
    if torch.cuda.is_initialized():
        raise ValueError("Runtime native build initialized CUDA")


if __name__ == "__main__":
    main()
