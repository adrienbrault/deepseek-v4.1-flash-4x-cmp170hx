# SPDX-License-Identifier: MIT
"""Authenticate k1b's compilation inputs, never its whole-packet cache key.

Host preparation checks the preserved receipt, current sources and artifacts.
The mandatory image dry and native loader also check the recorded image files.
Importing this module does not import Torch, compile, or initialize CUDA.
"""
import math
from indexer_common import ROOT, IMAGE, TORCH_GIT, module, need, read, sha
from pathlib import Path

NAME = 'indexer_bounded_k1b.so'
HEADER_PREFIX = b'#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n'
HEADER_SUFFIX = (b'\n#else\n#error "This file should not be included when either TORCH_STABLE_ONLY '
                 b'or TORCH_TARGET_VERSION is defined."\n'
                 b'#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n')


def hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def hash_map(value):
    return isinstance(value, dict) and bool(value) and all(hash_value(v) for v in value.values())


def check_build(directory):
    directory = Path(directory)
    packet = directory.parents[1]
    r = read(directory/'build-receipt.json')
    need(r['schema'] == 'indexer-bounded-k1b-build/1', 'native build schema')
    need(r['image'] == IMAGE and r['torch_git'] == TORCH_GIT and
         hash_value(r['libtorch_cuda_sha256']) and r['cub_version'] == 300001 and
         r['arch'] == '8.0' and r['platform'] == 'linux' and r['cuda_was_initialized'] is False,
         'native build image/torch/CUB/arch contract')
    with module('indexer-bounded-k1b', 'common.py') as k1b:
        need(r['cflags'] == k1b.CFLAGS and r['cuda_flags'] == k1b.CUDA_FLAGS, 'native build flags')
    # sort.inc is a compile input even if an older/incomplete receipt omits it.
    # Missing or extra sources are refused; no hash is inferred for a receipt.
    need(r['sources'] == {f:sha(packet/f) for f in ('bounded.cpp', 'bounded.cu', 'sort.inc')},
         'native build sources (including required sort.inc)')
    need(r['registration_sha256'] == sha(packet/'registration.json'), 'native build registration')
    need(r['torch_build_source_hashes'] == read(packet/'torch-build-pins.json'), 'native Torch build sources')
    headers = {k:v for k,v in read(ROOT/'rounds/indexer-bounded-r1/torch-source-pins.json').items()
               if k.endswith(('.cuh', '.h'))}
    need(bool(headers) and r['normalized_source_headers'] == headers, 'native normalized build headers')
    includes = r['include_hashes']
    need(hash_map(includes) and all(isinstance(k, str) and len(Path(k).parts) > 1 and
         Path(k).parts[0] in ('torch', 'cccl') and not Path(k).is_absolute() and
         '..' not in Path(k).parts and str(Path(k)) == k for k in includes), 'native include hashes')
    need(set(r['source_headers']) == set(headers) and hash_map(r['source_headers']) and all(
         r['source_headers'][rel] == includes.get('torch/'+rel.removeprefix('aten/src/'))
         for rel in headers), 'native installed build headers')
    # Preserve the historical source key only as the identity of the original
    # compile claim. Recomputing it would include post-build RESULTS.md edits.
    attempt = r['compile_attempt']
    need(attempt == read(directory/'attempt.json') == read(directory.parent/'compile-attempt.json') and
         attempt['attempt'] == 1 and attempt['date'] == '2026-11-15' and attempt['cap_s'] == 300 and
         isinstance(attempt['start_epoch'], (int, float)) and math.isfinite(attempt['start_epoch']) and
         attempt['start_epoch'] > 0 and hash_value(r['source_key']) and
         attempt['source_key'] == r['source_key'], 'native compile attempt receipt')
    need(set(r['artifacts']) == {NAME, 'build.ninja'} and hash_map(r['artifacts']) and all(
         sha(directory/name) == h for name,h in r['artifacts'].items()), 'native build artifact hash')
    return r, directory/NAME


def check_environment(receipt, torch, roots=None):
    """Check actual image inputs before loading a DSO; no CUDA API is called."""
    need(torch.version.git_version == TORCH_GIT, 'runtime torch revision')
    package = Path(torch.__file__).parent
    need(sha(package/'lib/libtorch_cuda.so') == receipt['libtorch_cuda_sha256'], 'runtime libtorch CUDA hash')
    need(sha(package/'utils/cpp_extension.py') ==
         receipt['torch_build_source_hashes']['torch/utils/cpp_extension.py'], 'runtime Torch build source')
    if roots is None:
        from torch.utils.cpp_extension import include_paths
        roots = {'torch': Path(include_paths()[0]),
                 'cccl': Path('/usr/local/cuda/targets/x86_64-linux/include/cccl')}
    actual = {label+'/'+str(f.relative_to(root)):sha(f) for label,root in roots.items()
              for f in sorted(root.rglob('*')) if f.is_file()}
    need(actual == receipt['include_hashes'], 'runtime include hashes')
    import hashlib
    for rel, expected in receipt['normalized_source_headers'].items():
        f = roots['torch']/rel.removeprefix('aten/src/')
        data = f.read_bytes()
        need(sha(f) == receipt['source_headers'][rel] and data.startswith(HEADER_PREFIX) and
             data.endswith(HEADER_SUFFIX) and hashlib.sha256(data[len(HEADER_PREFIX):-len(HEADER_SUFFIX)]).hexdigest()
             == expected, 'runtime normalized build header')


def check_r18_environment(receipt, torch):
    import sysconfig
    package = Path(torch.__file__).parent
    actual = dict(torch_version=str(torch.__version__), torch_git=torch.version.git_version,
                  cuda_version=torch.version.cuda, python_soabi=sysconfig.get_config_var('SOABI'),
                  cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    need(actual == receipt['abi'], 'r18 runtime ABI differs')
    need(all(sha(package/'lib'/name) == digest
             for name,digest in receipt['torch_libraries'].items()), 'r18 Torch library bytes differ')
