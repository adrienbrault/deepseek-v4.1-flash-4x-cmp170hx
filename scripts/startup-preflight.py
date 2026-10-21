"""Reuse successful startup checks for unchanged image, checkpoint and host boot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

CHECKS = [('preflight.json', '/opt/dsv41/tools/preflight.py', False),
          ('runtime.txt', '/opt/dsv41/tools/check_runtime.py', True),
          ('cuda-ssd.txt', '/opt/dsv41/tests/cuda_ssd.py', True),
          ('cuda-kernels.txt', '/opt/dsv41/tests/cuda_kernels.py', True),
          ('cuda-marlin.txt', '/opt/dsv41/tests/cuda_marlin.py', True)]


def checkpoint_identity(root):
    root = Path(root).resolve(strict=True)
    files = set(root.glob('*.json'))
    index = root / 'model.safetensors.index.json'
    weights = json.loads(index.read_text())['weight_map']
    for name in set(weights.values()):
        candidate = (root / name).resolve(strict=True)
        if candidate.parent != root:
            raise ValueError('checkpoint shard must be inside checkpoint directory')
        files.add(candidate)
    result = []
    for p in sorted(files):
        st = p.stat()
        row = [str(p.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        if p.suffix == '.json':
            row.append(hashlib.sha256(p.read_bytes()).hexdigest())
        result.append(row)
    return result


def fingerprint(image_id, model, official, launcher):
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,serial,pci.bus_id,name,driver_version,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    if len(hardware.splitlines()) != 4:
        raise ValueError('expected four GPU identity rows')
    boot = Path('/proc/sys/kernel/random/boot_id')
    d = {'version': 1, 'image': image_id, 'hardware': hardware,
         'kernel': platform.release(), 'boot': boot.read_text().strip() if boot.exists() else platform.node(),
         'model': checkpoint_identity(model), 'official': checkpoint_identity(official),
         'helper': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'launcher': hashlib.sha256(Path(launcher).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def run_checks(image, model, official, out):
    for filename, script, gpu in CHECKS:
        cmd = ['docker', 'run', '--rm', '--pull=never', '--network', 'none']
        if gpu:
            cmd += ['--gpus', 'all']
        cmd += ['--entrypoint', 'python3']
        if filename == 'preflight.json':
            cmd += ['-v', f'{model}:/model-view:ro', '-v', f'{official}:/official:ro']
        cmd += [image, script]
        if filename == 'preflight.json':
            cmd += ['--model', '/model-view', '--official', '/official']
        if filename == 'cuda-ssd.txt':
            cmd += ['--graphs']
        with (out / filename).open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print((out / filename).read_text(), end='', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('image', 'image-id', 'model', 'official', 'results', 'cache', 'launcher'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    key = fingerprint(a.image_id, a.model, a.official, a.launcher)
    cached = Path(a.cache) / 'startup-preflight' / key
    hit = False
    try:
        manifest = json.loads((cached / 'receipt.json').read_text())
        hit = manifest['key'] == key and manifest['status'] == 'PASS' and all(
            hashlib.sha256((cached / n).read_bytes()).hexdigest() == manifest['files'][n] for n, _, _ in CHECKS)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if hit and not a.force:
        for name, _, _ in CHECKS:
            shutil.copy2(cached / name, out / name)
        mode = 'HIT'
    else:
        # A forced recheck that fails must invalidate a previous success too.
        if cached.exists():
            shutil.rmtree(cached)
        run_checks(a.image, a.model, a.official, out)
        if key != fingerprint(a.image_id, a.model, a.official, a.launcher):
            raise RuntimeError('preflight inputs changed during checks')
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cached.parent) as temp:
            stage = Path(temp) / 'receipt'
            stage.mkdir()
            for name, _, _ in CHECKS:
                shutil.copy2(out / name, stage / name)
            receipt = {'status': 'PASS', 'key': key, 'files': {n: hashlib.sha256((stage / n).read_bytes()).hexdigest() for n, _, _ in CHECKS}}
            (stage / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            if cached.exists():
                shutil.rmtree(cached)
            os.replace(stage, cached)
        mode = 'MISS'
    report = {'status': 'PASS', 'cache': mode, 'key': key, 'seconds': round(time.monotonic() - started, 3)}
    (out / 'startup-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
