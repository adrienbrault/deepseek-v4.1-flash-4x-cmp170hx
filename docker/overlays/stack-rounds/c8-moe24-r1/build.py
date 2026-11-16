# SPDX-License-Identifier: MIT
"""CPU-only sibling build; materialize the cumulative daily through its builder."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from common import *

def materialize(out):
    with trim() as (t, builder, _):
        t.pins(); root, native = builder.materialize(out, 'fixed640')
    before = {str(f.relative_to(native)): sha(f) for f in native.rglob('*') if f.is_file()}
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(P/'0001-rows24.patch')], cwd=native,
                   check=True, stdout=subprocess.DEVNULL)
    after = {k: sha(native/k) for k in before}
    need([k for k in before if before[k] != after[k]] == ['quant/opera_coop.cu'], 'unexpected native change')
    original = (native/'quant/opera_coop.cu').read_text().replace('p.bsz == 24', 'p.bsz == 16')
    need(hashlib.sha256(original.encode()).hexdigest() == before['quant/opera_coop.cu'], 'change exceeds row guard')
    write(root/'proof.json', dict(before=before, after=after, change='row guard only; all arithmetic bodies identical'))
    return root, native

def incumbent(root, expected, proof):
    need(sha(root/'build.json') == expected, 'r7b build receipt differs from authenticated serving plan')
    b = read(root/'build.json'); s = read(root/'source-proof.json')
    need(b['schema'] == 'current-r7b-build/1' and b['status'] == 'PASS' and b['arm'] == 'fixed640'
         and b['image'] == IMAGE and b['cuda_initialized'] is False, 'not the qualified r7b build')
    need(sha(root/'source-proof.json') == b['source_proof_sha256'] and s['after'] == proof['before'], 'r7b source differs')
    for f, v in b['assets'].items():
        path = (root/'assets'/f).resolve()
        need(path.is_relative_to((root/'assets').resolve()) and sha(path) == v, 'r7b asset drift')
    return root/'assets/opera_coop_ext.so'

def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-only', action='store_true'); a = p.parse_args(); pins()
    root, native = materialize(a.out)
    if a.source_only: return
    import torch
    from torch.utils.cpp_extension import load as compile_extension
    need(not torch.cuda.is_initialized(), 'CPU-only build required')
    os.environ.update(TORCH_CUDA_ARCH_LIST='8.0', MAX_JOBS='2', EXLLAMA_EXT_COMPRESS='0')
    with trim() as (t, _, __): flags = t.load(t.BASE/'cuda_flags.py', '_rows24_flags').cuda_cflags()
    sources = ['opera_coop_binding.cpp', 'quant/opera_coop.cu'] + [f'quant/comp_units/opera_coop_inst_k{k}.cu' for k in (2,3,4)]
    (root/'build').mkdir()
    flags += ['-Dopera_coop_ns=opera_rows24_ns', '-Xptxas=-v', '-lineinfo']
    mod = compile_extension(name='opera_rows24', sources=[str(native/f) for f in sources],
        extra_include_paths=[str(native)], extra_cflags=['-O3'], extra_cuda_cflags=flags,
        build_directory=str(root/'build'), verbose=True)
    shutil.copy2(mod.__file__, root/'opera_rows24.so')
    need(not torch.cuda.is_initialized(), 'build initialized CUDA')
    write(root/'build.json', dict(status='BUILT', image=IMAGE, key=source_key(), torch=str(torch.__version__),
          cuda=torch.version.cuda, library_sha256=sha(root/'opera_rows24.so'), proof_sha256=sha(root/'proof.json'), flags=flags))
if __name__ == '__main__': main()
