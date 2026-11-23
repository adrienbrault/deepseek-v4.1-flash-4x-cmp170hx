# SPDX-License-Identifier: MIT
"""Import-time hooks; sealed source, DSpark proposals and AV are untouched."""
from contextvars import ContextVar
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import re
import sys

MODULES = tuple(f'layers.{i}.attn.wo_b' for i in range(2, 40))
ACTIVE = ContextVar('dsv41_argmax_target', default=None)
LOADING = ContextVar('dsv41_merged_loading', default=False)
_manifest = None
TARGETS = {
    'vllm.model_executor.layers.linear',
    'vllm.models.deepseek_v41.nvidia.model',
    'vllm.models.deepseek_v41.nvidia.vl_model',
    'vllm.v1.worker.gpu.model_runner',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def manifest():
    global _manifest
    if _manifest is not None:
        return _manifest
    value = os.environ.get('DSV41_MERGED_WOB', '')
    if not value:
        return None
    root = Path(value)
    m = json.loads((root / 'manifest.json').read_text())
    if (m.get('schema') != 'dsv41-merged-wob-v1' or m.get('status') != 'BUILT_NOT_GPU_VALIDATED'
            or tuple(m.get('modules', ())) != MODULES or m.get('format') not in ('bf16', 'fp8')
            or [x.get('module') for x in m.get('layers', [])] != list(MODULES)):
        raise ValueError('incomplete/unsupported merged sidecar')
    expected_files = {module + '.safetensors' for module in MODULES}
    if set(m['files_sha256']) != expected_files:
        raise ValueError('sidecar file set differs')
    for row in m['layers']:
        if row['file'] != row['module'] + '.safetensors' or row['shape'] != [5120, 8192]:
            raise ValueError('sidecar production shape/name differs')
    for name, sha in m['files_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('sidecar hash differs: ' + name)
    _manifest = root, m
    return _manifest


def merged_weights(weights, root, m):
    from safetensors import safe_open
    import torch
    expected = {module + '.' + suffix for module in m['modules'] for suffix in ('weight', 'scale')}
    source_hashes = {key: sha for row in m.get('layers', []) for key, sha in row['source_tensor_sha256'].items()}
    seen = set()
    for name, value in weights:
        if name in expected:
            if name in seen:
                raise ValueError('duplicate checkpoint tensor: ' + name)
            seen.add(name)
            # Startup-only CPU read: refuse a different/doubly-merged base.
            # This never runs during forward or graph replay.
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest() != source_hashes.get(name):
                raise ValueError('source checkpoint tensor differs: ' + name)
            module, suffix = name.rsplit('.', 1)
            if suffix == 'scale' and m['format'] == 'bf16':
                continue
            with safe_open(str(root / (module + '.safetensors')), framework='pt', device='cpu') as f:
                replacement = f.get_tensor(name)
            yield name, replacement
        else:
            yield name, value
    if seen != expected:
        raise ValueError('checkpoint does not match merged wo_b tensor set')


def wrap_load(original):
    @functools.wraps(original)
    def load(self, weights, *args, **kwargs):
        packet = manifest()
        if packet is None or LOADING.get():
            return original(self, weights, *args, **kwargs)
        token = LOADING.set(True)
        try:
            result = original(self, merged_weights(weights, *packet), *args, **kwargs)
            print('DSV41_MERGED_WOB loaded format=' + packet[1]['format'] + ' modules=38', flush=True)
            return result
        finally:
            LOADING.reset(token)
    return load


def wrap_linear(original):
    signature = inspect.signature(original)
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        packet = manifest()
        if packet is not None and packet[1]['format'] == 'bf16':
            bound = signature.bind(self, *args, **kwargs)
            prefix = bound.arguments.get('prefix', '')
            match = re.fullmatch(r'(?:language_model\.)?model\.(layers\.\d+\.attn\.wo_b)', prefix)
            if match and match[1] in packet[1]['modules']:
                if bound.arguments.get('input_size') != 8192 or bound.arguments.get('output_size') != 5120:
                    raise ValueError('merged wo_b construction shape differs')
                bound.arguments['quant_config'] = None
                return original(*bound.args, **bound.kwargs)
        return original(self, *args, **kwargs)
    return init


def wrap_logits(original):
    @functools.wraps(original)
    def compute(self, hidden, *args, **kwargs):
        if ACTIVE.get() is self:
            if args or kwargs:
                raise ValueError('unknown target compute_logits signature')
            from .argmax import argmax_rows
            return argmax_rows(self, hidden)
        return original(self, hidden, *args, **kwargs)
    return compute


def wrap_sample(original):
    @functools.wraps(original)
    def sample(self, hidden_states, input_batch, grammar_output):
        from vllm import envs
        from vllm.v1.worker.gpu.sample.sampler import Sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
        from .eligibility import eligible
        # Never activate both vocabulary-sampling implementations together.
        on = os.environ.get('DSV41_TARGET_ARGMAX', '0') == '1'
        if on and getattr(envs, 'VLLM_TP_DISTRIBUTED_SAMPLING', False):
            raise ValueError('disable VLLM_TP_DISTRIBUTED_SAMPLING with DSV41_TARGET_ARGMAX')
        lm = getattr(self.model, 'language_model', self.model)
        use = on and eligible(self, input_batch, grammar_output, Sampler, RejectionSampler)
        if use and not getattr(self, '_dsv41_argmax_announced', False):
            print('DSV41_TARGET_ARGMAX active rank=' + str(lm.lm_head.tp_rank), flush=True)
            self._dsv41_argmax_announced = True
        token = ACTIVE.set(lm if use else None)
        try:
            return original(self, hidden_states, input_batch, grammar_output)
        finally:
            ACTIVE.reset(token)
    return sample


def wrap_load_model(original):
    @functools.wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        packet = manifest()
        if packet is not None:
            if self.lora_config is not None:
                raise ValueError('merged arm must disable vLLM LoRA')
            lm = getattr(self.model, 'language_model', self.model)
            audit = []
            for name, module in lm.named_modules():
                if name.removeprefix('model.') in packet[1]['modules']:
                    tensors = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
                    audit.append({'module': name, 'class': type(module).__name__,
                                  'method': type(module.quant_method).__name__,
                                  'tensors': [{'name': n, 'dtype': str(t.dtype), 'shape': list(t.shape),
                                               'bytes': t.numel() * t.element_size()} for n, t in tensors]})
            if len(audit) != 38:
                raise ValueError('merged arm did not construct exactly 38 target wo_b modules')
            if packet[1]['format'] == 'bf16' and any(x['method'] != 'UnquantizedLinearMethod' for x in audit):
                raise ValueError('merged BF16 wo_b still has a quantized method')
            # Keep each line below PIPE_BUF so concurrent rank stdout cannot
            # split large per-layer JSON documents into one another.
            signatures = {}
            for layer in audit:
                for t in layer['tensors']:
                    key = (t['name'], t['dtype'], tuple(t['shape']))
                    signatures[key] = signatures.get(key, 0) + t['bytes']
            summary = {'rank': lm.lm_head.tp_rank, 'layer_count': len(audit),
                       'modules': [x['module'] for x in audit],
                       'methods': sorted({x['method'] for x in audit}),
                       'tensor_bytes': sum(signatures.values()),
                       'tensor_signatures': [{'name': k[0], 'dtype': k[1], 'shape': k[2], 'total_bytes': v}
                                             for k, v in signatures.items()]}
            print('DSV41_MERGED_AUDIT ' + json.dumps(summary), flush=True)
        return result
    return load


def patch(module):
    if getattr(module, '_dsv41_lora_argmax_patched', False):
        return
    name = module.__name__
    if name == 'vllm.model_executor.layers.linear' and manifest() is not None:
        module.RowParallelLinear.__init__ = wrap_linear(module.RowParallelLinear.__init__)
    elif name in ('vllm.models.deepseek_v41.nvidia.model', 'vllm.models.deepseek_v41.nvidia.vl_model'):
        cls = getattr(module, 'DeepseekV41LLMForCausalLM' if name.endswith('.model') else 'DeepseekV41ForCausalLM')
        if manifest() is not None:
            cls.load_weights = wrap_load(cls.load_weights)
        if name.endswith('.model') and os.environ.get('DSV41_TARGET_ARGMAX', '0') == '1':
            cls.compute_logits = wrap_logits(cls.compute_logits)
    elif name == 'vllm.v1.worker.gpu.model_runner':
        if os.environ.get('DSV41_TARGET_ARGMAX', '0') == '1':
            module.GPUModelRunner.sample = wrap_sample(module.GPUModelRunner.sample)
        if manifest() is not None:
            module.GPUModelRunner.load_model = wrap_load_model(module.GPUModelRunner.load_model)
    module._dsv41_lora_argmax_patched = True


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('missing overlay target ' + fullname)
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s) if hasattr(loader, 'create_module') else None
            def exec_module(self, module):
                loader.exec_module(module)
                patch(module)
        spec.loader = Loader()
        return spec


def install():
    if any(isinstance(f, Finder) for f in sys.meta_path):
        return
    manifest()  # CPU-only, all ranks validate before model construction
    # Other general plugins (EXL3) may have imported linear already. Patching
    # its class object still affects references held by model modules. General
    # plugin discovery runs before worker/model construction.
    for name in TARGETS & sys.modules.keys():
        patch(sys.modules[name])
    sys.meta_path.insert(0, Finder())
