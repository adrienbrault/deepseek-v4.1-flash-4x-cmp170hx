#!/usr/bin/env python3
"""Install runtime packages and generate the serving scheduler before sealing."""
import hashlib
import importlib.util
from pathlib import Path
import shutil

from transform_scheduler import transform

KIT = Path(__file__).resolve().parents[1]
OLD_SITE = "/usr/local/lib/python3.12/dist-packages/"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
PRIVATE_RING = "vllm/distributed/device_communicators/opera_true_ring.py"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def locate_site(package):
    return Path(importlib.util.find_spec(package).origin).parent.parent


def write_source(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    for cached in target.parent.glob(f"__pycache__/{target.stem}.*.pyc"):
        cached.unlink()


def whole_paths(manifest):
    """Inventory the captured package mounts covered by the file patch series."""
    return sorted(
        target for m in manifest if m["exported"] and m["dest"].startswith(OLD_SITE)
        for target in [m["dest"].removeprefix(OLD_SITE)]
        if target.split("/")[0] in ("vllm", "vllm_exl3")
        and target not in (SCHEDULER, PRIVATE_RING)
    )


def install():
    vllm_site = locate_site("vllm")
    scheduler = vllm_site / SCHEDULER
    untransformed = KIT / "runtime/scheduler-image.py"
    if untransformed.exists():
        raise ValueError("The image scheduler was already transformed")
    shutil.copyfile(scheduler, untransformed)
    # r7g strips the fair settings. The same scheduler selects r7h fair
    # accounting and the half-share path through independent boot settings.
    candidate = transform(untransformed, fair_r2=True, prefill_share=True)
    write_source(scheduler, candidate)
    print("Scheduler SHA-256", sha(untransformed), "→", sha(scheduler))

    bundle = KIT / "overlays"
    for package in sorted(bundle.iterdir()):
        if package.is_dir() and package.name != "rootfs":
            for path in sorted(package.iterdir()):
                dest = vllm_site / path.name
                if dest.exists():
                    raise ValueError(f"Overlay package already installed: {dest}")
                shutil.copytree(path, dest)
    shutil.copytree(bundle / "rootfs", "/", dirs_exist_ok=True)


if __name__ == "__main__":
    install()
