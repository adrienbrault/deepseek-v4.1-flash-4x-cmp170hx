# SPDX-License-Identifier: MIT
"""The image functions the replicated wkv was reviewed against, pinned by source.

Equivalence argument (f760 image, sources in ../../in/wkv-replica-sources.json):

* Stock ``ColumnParallelLinear.forward`` with ``gather_output`` and no bias computes
  ``quant_method.apply(self, x, None)`` on this rank's shard, then
  ``tensor_model_parallel_all_gather(out, dim=-1)``. Every all-gather route
  (``GroupCoordinator.all_gather`` -> custom op ``all_gather`` or
  ``_all_gather_out_place`` -> ``CudaCommunicator.all_gather`` (PyNCCL) or
  ``DeviceCommunicatorBase.all_gather``) stacks the ranks' tensors in rank order, then
  ``reshape -> movedim(0, -1) -> reshape``: byte for byte ``torch.cat(shards, dim=-1)``.
  The one lossy route, the INT8 prefill gather, is refused by config.py.
  Adoption's own gathers call the same communicator directly into preallocated outputs:
  ``PyNcclCommunicator.all_gather(out, inp)`` (the call inside ``CudaCommunicator.all_gather``),
  or, when PyNCCL is disabled, ``dist.all_gather_into_tensor`` on the device group (the call
  inside ``DeviceCommunicatorBase.all_gather``). Both fill ``out`` rank-major.
  The processed tensors travel as raw uint8 bytes: Marlin's weight is int32 and its scales are
  ``float8_e8m0fnu`` (``mxfp8_marlin_process_scales``), which ``ncclDataTypeEnum`` cannot carry.
  Byte views of the typed storage move them unchanged; Marlin reads the typed views.
* The served wkv is ``ModelOptLinearMethod`` (MXFP8 static weight / dynamic activation,
  ``FormatScheme``) with ``MarlinMxfp8LinearKernel``. Its apply reads exactly
  ``layer.weight``, ``layer.weight_scale``, ``output_size_per_partition`` and
  ``input_size_per_partition``; weights are repacked once at load
  (``prepare_mxfp8_layer_for_marlin``) and only read afterwards. Atomic-add reduce is
  off for N >= 2048 (6400 per rank), so Marlin's split-K reduction has a fixed order
  that depends only on inputs, shapes and the SM count. Its lock buffer is keyed by
  stream and reset after every call, so ordered calls on one stream reuse it safely.
* Hence rank r running rank j's shard GEMM (same input bytes, same processed shard
  bytes, same shapes, same SM count) reproduces rank j's output exactly, and the
  concatenation equals the gathered output. The verify boot checks that claim
  bitwise inside the served graphs (replica.py shadow mode).

Digests are of the AST with decorators and docstring removed (engram-early-r1's scheme;
its 13 pins reappear here unchanged). Any change makes adoption refuse.
"""
import ast
import hashlib
import importlib
import inspect
import json
import textwrap

# (module, qualname) -> digest, computed inside the f760 image (Python 3.12.3), 2026-11-09.
REVIEWED_SOURCES = {
    ('vllm.model_executor.layers.quantization.modelopt', 'ModelOptLinearMethod.apply'):
        'd5cb5244e4104dc74afa92f45e5b597918811350084af9011fe68544b2ef7655',
    ('vllm.model_executor.layers.quantization.modelopt', 'FormatScheme.apply'):
        '529cca0ed33cf8caf35fc113d64985ea59065b773abe7e3c658f2ce9e5b8bb5f',
    ('vllm.model_executor.kernels.linear.mxfp8.marlin', 'MarlinMxfp8LinearKernel.apply_weights'):
        '6b15d946256d41efdcb1323051381360da35f1d8e31315a2f6f355a0f6d4fda6',
    ('vllm.model_executor.kernels.linear.mxfp8.marlin', 'MarlinMxfp8LinearKernel.process_weights_after_loading'):
        '543d69778e4764ff126c332c79e9fd6e55e86c4812eca1826a1f106fc1562195',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils_fp8', 'apply_mxfp8_marlin_linear'):
        '94843314c6c821d152348ba6fb07f3e62587e9eecec052a9d9fec8f13bfbb173',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils_fp8', 'prepare_mxfp8_layer_for_marlin'):
        '472c2af3b4ec5f9f51319711a04c849931e492bd0f52eafc36e79d3095b80088',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', '_marlin_gemm'):
        '83afd95ddeb3d0d133a60a0282f5a5e99e3195cf0487e3a98608dcf73fabfc66',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', 'get_marlin_workspace'):
        '6659a57ce7f0a83d4e4e467134bee060f1c971aaf0f74b9e9ad03751c859ae4f',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', 'should_use_atomic_add_reduce'):
        'fb6019529165cd1b1ad9a3f77b783bd77e4909267e0217fabc393ae3a977650e',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', 'marlin_pad_dim'):
        'f8fd714ad18bc72f37786c926f956adc0401c36039e379e8b03ffea1634fae93',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', 'marlin_unpad_output'):
        'e1d63f2da2eb4c69ba6ecf6df50b3919d8427063fc3c23edecb8f40c15d6fac0',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils', 'marlin_repacked_nk'):
        '2332ade3c76d4565e5f87b302bface48f5dd605585586915103d8fe7b9ff2a89',
    ('vllm.model_executor.layers.linear', 'ColumnParallelLinear.forward'):
        'c3349ce51b9bfc90771a5addf5cc27b28476ce2ecf5ec9b3b5f0f71b76f2c156',
    ('vllm.distributed.communication_op', 'tensor_model_parallel_all_gather'):
        '496927f5ecc51ee3ffe630c26928c1cad7349914a3e72f97aaa01b7d9fd2b56a',
    ('vllm.distributed.parallel_state', 'GroupCoordinator.all_gather'):
        '46fc6accb8569bffe9f3d9a385c13765b727301f14652a7f3bf5e0a357d96a0b',
    ('vllm.distributed.parallel_state', 'GroupCoordinator._all_gather_out_place'):
        '286975ac0e9163429ba58416fddf14b7a24a7d844b406fe9cfafaede2151d134',
    ('vllm.distributed.parallel_state', 'all_gather'):
        '5770a85c6159f530e8ca117a5d6370db7a46d4db02231bf69c451439ac4e4693',
    ('vllm.distributed.device_communicators.base_device_communicator', 'DeviceCommunicatorBase.all_gather'):
        'ccb791bb546c633f7c33713d8ac4fd4f1fb1391aecf1443bb7c60d0a082b0033',
    ('vllm.distributed.device_communicators.cuda_communicator', 'CudaCommunicator.all_gather'):
        '35d047e5c83a6c9c9a55992ead94651da354e8e45a69826ed2b598d000246046',
    ('vllm.distributed.device_communicators.pynccl', 'PyNcclCommunicator.all_gather'):
        '53e36ef1de610b68730f3b01b9469cb1992ef7aef1e5d772fc37f621951e7da7',
    ('vllm.distributed.device_communicators.pynccl_wrapper', 'ncclDataTypeEnum'):
        'd1a77bf14782b2f69c1340dea4e8440ed40cbc3c9148f12443336c290a0e9f98',
    ('vllm.model_executor.layers.quantization.utils.marlin_utils_fp8', 'mxfp8_marlin_process_scales'):
        '77a61d1a77cf3959ec81af0de5c8958c0cbecf13ff8f8255dc30db6947feb324',
    ('vllm.v1.worker.gpu_worker', 'Worker.load_model'):
        '388a730ff324186eb6f26e3711ef7122ec63abf54835011c89ea10b2b0eb1441',
}


def digest(source):
    node = ast.parse(textwrap.dedent(source)).body[0]
    node.decorator_list = []
    if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)

    def canonical(value):
        if isinstance(value, ast.AST):
            return dict(kind=type(value).__name__, fields={k: canonical(v)
                        for k, v in ast.iter_fields(value) if v is not None and v != []})
        if isinstance(value, list):
            return [canonical(v) for v in value]
        return value
    data = json.dumps(canonical(node), sort_keys=True, separators=(',', ':'), default=repr)
    return hashlib.sha256(data.encode()).hexdigest()


def resolve(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split('.'):
        obj = getattr(obj, part)
    return inspect.unwrap(obj)


def source_problem(pins=None, lookup=resolve):
    """None if every pinned function still has its reviewed source, else the first difference."""
    for (module, qualname), want in (REVIEWED_SOURCES if pins is None else pins).items():
        try:
            got = digest(inspect.getsource(lookup(module, qualname)))
        except Exception as exc:  # missing module/attribute/source: not the reviewed image
            return f'{qualname} unreadable ({type(exc).__name__})'
        if got != want:
            return f'{qualname} source differs from the reviewed f760 source'
    return None
