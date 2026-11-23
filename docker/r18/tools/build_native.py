#!/usr/bin/env python3
"""Build the native packet layers within the digest-pinned v0.34 image."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path("/opt/dsv41")
COOP = Path("/opt/dsv41-coop-kernel")
RING = ROOT / "opera-true-ring"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_root(name):
    spec = importlib.util.find_spec(name)
    return Path(spec.origin).parent


def verify(root, files):
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise ValueError(f"native source drift: {root / name}")


def build_coop():
    source = COOP / "SOURCE-MANIFEST.json"
    original = json.loads(source.read_text())
    verify(COOP / "source", original["native_source_files"])
    verify(COOP / "source", json.loads((COOP / "GENERATED-MANIFEST.json").read_text()))
    shutil.copy2(source, COOP / "INCUMBENT-SOURCE-MANIFEST.json")
    stock = Path(importlib.util.find_spec("exllamav3_ext").origin)
    source.write_text(json.dumps(dict(original, native_stock_sha256=sha(stock)), indent=2) + "\n")
    env = dict(os.environ, EXLLAMA_EXT_COMPRESS="require")
    env.pop("EXLLAMA_EXT_LINEINFO", None)
    subprocess.run([sys.executable, str(COOP / "build_component.py"), "--root", str(COOP),
                    "--out", str(COOP)], check=True, env=env)
    actual = json.loads((COOP / "build-manifest.json").read_text())
    old = json.loads((COOP / "incumbent-build.json").read_text())
    for key in ("cuda_flags", "arch", "sources"):
        if actual[key] != old[key]:
            raise ValueError(f"Current-r7b native build contract drift: {key}")
    if any(actual["source_sha256"].get(name) != digest
           for name, digest in old["source_sha256"].items()):
        raise ValueError("Current-r7b native source drift")
    shutil.rmtree(COOP / "build")


def patch_rowstore():
    packet = ROOT / "native/rowstore"
    manifest = json.loads((packet / "MANIFEST.json").read_text())
    site = package_root("vllm").parent
    actual = sha(site / manifest["base_file"])
    # A complete core port may already include the native donor patch.
    if actual != manifest["patched_sha256"]:
        subprocess.run([sys.executable, str(packet / "apply.py"), str(site)], check=True)
    print("PASS served row-store source", manifest["patched_sha256"])


RING_CUDA_SOURCE = "csrc/pcie_ipc_all_reduce.cu"
RING_COMM_HEADERS = tuple(
    f"include/flashinfer/comm/{name}.cuh" for name in (
        "pcie_ipc_all_reduce", "pcie_ipc_ce_ring", "pcie_ipc_ce_memop", "pcie_ipc_ce_sm120",
    )
)


def apply_ring_patch(patch, directory):
    # GNU patch in the image uses the same strict context check as the core series.
    for mode in (["--dry-run"], []):
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", *mode],
                       input=patch.read_bytes(), cwd=directory, check=True)


def prepare_ring_sources(fi_root):
    # The audit lock describes the two exported wheels' reviewed interface.
    # Record actual installed hashes after matching the known 0.7.3 bytes.
    audited = json.loads((RING / "flashinfer-073.json").read_text())
    stock = {rel: sha(fi_root / rel) for rel in audited}
    changed = [rel for rel in audited if stock[rel] != audited[rel]]
    if changed:
        raise ValueError(f"installed FlashInfer differs from the audited 0.7.3 sources: {changed}")
    for rel in (RING_CUDA_SOURCE, *RING_COMM_HEADERS):
        destination = RING / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fi_root / "data" / rel, destination)
    apply_ring_patch(RING / "flashinfer-073.patch", RING)
    return stock


def install_ring_helper(manifest_path):
    helper = package_root("vllm") / "distributed/device_communicators/opera_true_ring.py"
    original = sha(helper)
    served = RING / "served/opera_true_ring.py"
    inputs = json.loads((RING / "served-inputs.json").read_text())
    if sha(served) != inputs["050/opera_true_ring.py"]:
        raise ValueError("served true-ring helper differs from mount 050")
    apply_ring_patch(RING / "served-helper.patch", served.parent)
    text = served.read_text()
    anchor = "IMAGE_BUILD_MANIFEST_SHA256 = None"
    if text.count(anchor) != 1:
        raise ValueError("true-ring manifest seal anchor drift")
    helper.write_text(text.replace(anchor, f"IMAGE_BUILD_MANIFEST_SHA256 = {sha(manifest_path)!r}"))
    for cache in helper.parent.glob("__pycache__/opera_true_ring.*.pyc"):
        cache.unlink()
    (ROOT / "ring-helper-build.json").write_text(json.dumps({
        "core_port_sha256": original,
        "served_sha256": inputs["050/opera_true_ring.py"],
        "manifest_sha256": sha(manifest_path),
        "installed_sha256": sha(helper),
    }, indent=2) + "\n")


def build_ring():
    version = importlib.metadata.version("flashinfer-python")
    if version != "0.7.3" or os.environ.get("FLASHINFER_CUDA_ARCH_LIST") != "8.0":
        raise ValueError("r18 true ring requires FlashInfer 0.7.3 and an SM80-only build")
    fi_root = package_root("flashinfer")
    stock = prepare_ring_sources(fi_root)
    import flashinfer.comm.pcie_ipc_ar as ar
    from flashinfer.jit.core import gen_jit_spec
    import torch

    original = json.loads((RING / "incumbent-manifest.json").read_text())
    verify(RING, {"policy.json": original["files"]["policy.json"]})
    inputs = json.loads((RING / "served-inputs.json").read_text())
    extra = {rel.removeprefix("051/"): digest for rel, digest in inputs.items()
             if rel.startswith("051/")}
    verify(RING, extra)
    if ar.get_pcie_ipc_comm_module.cache_info().currsize or not callable(ar.gen_pcie_ipc_comm_module):
        raise ValueError("FlashInfer PCIe IPC factory must be importable and cold")
    if int(ar.IpcLaunchConfig(8, 256, 8).variant) != 8:
        raise ValueError("FlashInfer does not support an explicit private launch ID")

    jit_name = "dsv41_r18_opera_true_ring_073"
    link_dir = os.environ["LIBRARY_PATH"].split(os.pathsep, 1)[0]
    spec = gen_jit_spec(jit_name, [RING / RING_CUDA_SOURCE],
                        extra_include_paths=[fi_root / "data/csrc"],
                        extra_ldflags=["-L" + link_dir, "-lcuda"])
    if spec.aot_artifacts:
        raise ValueError("unexpected AOT artifact for r18 true ring")
    spec.build_and_load()
    lib = Path(spec.jit_library_path).resolve()
    if Path(spec.get_library_path()).resolve() != lib:
        raise ValueError("FlashInfer resolved something other than the r18 JIT result")
    shutil.copy2(lib, RING / "native.so")

    private = (RING_CUDA_SOURCE, *RING_COMM_HEADERS, "policy.json",
               "incumbent-manifest.json", *extra)
    manifest = RING / "manifest.json"
    manifest.write_text(json.dumps({
        "files": {rel: sha(RING / rel) for rel in private},
        "stock_fi_sources": stock,
        "incumbent_native_sha256": original["native_library_sha256"],
        "native_library_sha256": sha(RING / "native.so"),
        "flashinfer_version": version,
        "tvm_ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "r18_jit_name": jit_name,
        "arch": "8.0",
        "r18_hardware_gate": "PENDING",
    }, indent=2) + "\n")
    install_ring_helper(manifest)
    shutil.rmtree(RING / "served")
    print("PASS true ring compile and link")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("coop", "rowstore", "ring"))
    args = parser.parse_args()
    {"coop": build_coop, "rowstore": patch_rowstore, "ring": build_ring}[args.component]()


if __name__ == "__main__":
    main()
