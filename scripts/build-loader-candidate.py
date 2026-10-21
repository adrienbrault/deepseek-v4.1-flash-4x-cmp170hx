"""Build a loading-only child image from exact int-r17; preserve its compiler caches."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

BASE = 'sha256:cae174260cb6d6c9ab113ecb9db179bf20180662e4e16860252f9b8000bdb916'
EXPECTED = {'exl3.py': '5f8b22b99948c36f6e1f1a8c406ad21a1225da8c38854a8b59922f314d2aa35d',
            'dspark.py': 'c12b0b9151d3bc56232afc1dbd8c3a8a6f20db18273cadba2d6d0bdb3a2c78f9'}
MODULES = {'exl3.py': 'vllm_exl3/exl3.py', 'dspark.py': 'vllm/models/deepseek_v41/nvidia/dspark.py'}
SITE = '/usr/local/lib/python3.12/dist-packages/'
H = Path(__file__).resolve().parent
out = H / 'loader-candidate'
context = out / 'image'
context.mkdir(parents=True, exist_ok=True)
patches = {'exl3.py': H.parent / 'patches/vllm-exl3/0008-load-reclaim-fast.patch', 'dspark.py': H.parent / 'patches/vllm/0026-filter-dspark-unused-shards.patch'}
for name, module in MODULES.items():
    source = subprocess.check_output(['docker', 'run', '--rm', '--pull=never', '--network', 'none',
                                      '--entrypoint', 'cat', BASE, SITE + module])
    assert hashlib.sha256(source).hexdigest() == EXPECTED[name], name
    work = out / ('plugin' if name == 'exl3.py' else 'vllm')
    rel = Path('src/vllm_exl3/exl3.py') if name == 'exl3.py' else Path('vllm/models/deepseek_v41/nvidia/dspark.py')
    path = work / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    (work / 'tests').mkdir(exist_ok=True)
    path.write_bytes(source)
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], cwd=work,
                   input=patches[name].read_bytes(), check=True)
    shutil.copy2(path, context / name)
(context / 'Dockerfile').write_text('FROM ' + BASE + '\n' + ''.join(
    f'COPY {name} {SITE}{module}\n' for name, module in MODULES.items()))
subprocess.run(['docker', 'build', '--pull=false', '--network=none', '-t', 'dsv41:loader-r1', str(context)], check=True)
base = json.loads(subprocess.check_output(['docker','image','inspect',BASE]))[0]
new = json.loads(subprocess.check_output(['docker','image','inspect','dsv41:loader-r1']))[0]
assert new['RootFS']['Layers'][:len(base['RootFS']['Layers'])] == base['RootFS']['Layers']
# The only child filesystem changes are the two reviewed loading-only Python files.
# Compiler libraries and Triton/TileLang kernel source remain byte-identical.
cache = Path(os.environ['DSV41_CACHE_DIR']) / 'compiler'
old_dir, new_dir = cache / BASE.removeprefix('sha256:'), cache / new['Id'].removeprefix('sha256:')
if old_dir.exists() and not new_dir.exists():
    shutil.copytree(old_dir, new_dir)
manifest = {'base_image':BASE, 'candidate_image':new['Id'], 'changed_modules':MODULES,
            'files':{n:hashlib.sha256((context/n).read_bytes()).hexdigest() for n in MODULES},
            'patches':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in patches.items()},
            'compiler_cache_seeded_from':BASE}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
