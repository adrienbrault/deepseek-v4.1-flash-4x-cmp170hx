# SPDX-License-Identifier: MIT
"""Import-time hook: build the Engram wkv replicas right after ``Worker.load_model``.

Only ``vllm.v1.worker.gpu_worker.Worker.load_model`` is wrapped. After the stock body
(weights loaded and processed into Marlin layout, before memory profiling, KV
allocation and graph capture) every TP worker calls ``replica.adopt_model`` on its
runner's model, in lockstep, so the KV pool is sized with the replica resident.

The plugin refuses anything it was not built for, loudly: an inert arm would waste a
screen. At install, the Engram source must be the stack's engram-gap-r1 overlay (file
sha256, located without importing it); at adoption, every rank must pass the
eligibility and source review (replica.py, reviewed.py) or all ranks raise.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

from vllm.logger import init_logger

from .config import ConfigError

TARGET = 'vllm.v1.worker.gpu_worker'
ENGRAM = 'vllm.models.deepseek_v41.common.engram'
# engram-gap-r1/out/overlay/vllm/models/deepseek_v41/common/engram.py, the file the stack
# mounts (same pin as engram-early-r1): Engram.forward feeds wkv the gathered rows of all
# heads, ``kv = self.wkv(self.embed(hash_ids).flatten(-2))``, unless an EngramBatch is set.
PINS = {ENGRAM: 'b31af87c1ccd5ef17636a94c7231e1b2b33c6c6fa8ec7863e10788caa358202b'}
logger = init_logger('vllm.' + __name__)


def source_sha(name):
    """sha256 of a module's source file, located without executing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED missing module {name}')
    with open(spec.origin, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_sources(sha=source_sha):
    for name, want in PINS.items():
        got = sha(name)
        if got != want:
            raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {name} sha256 {got} != pinned {want}')


def wrap_load(original, cfg, adopt, ops):
    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            adopt(self.model_runner.get_model(), cfg, ops())
        except Exception as exc:
            logger.warning('%s', exc)
            raise
        return result
    return load_model


def patch(module, cfg, adopt=None, ops=None):
    if getattr(module, '_dsv41_wkv_replica_patched', False):
        return
    if adopt is None or ops is None:
        from . import replica
        adopt, ops = adopt or replica.adopt_model, ops or replica.vllm_ops
    module.Worker.load_model = wrap_load(module.Worker.load_model, cfg, adopt, ops)
    module._dsv41_wkv_replica_patched = True


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, cfg):
        self.cfg = cfg

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('missing hook target ' + fullname)
        loader, cfg = spec.loader, self.cfg

        class Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s) if hasattr(loader, 'create_module') else None

            def exec_module(self, module):
                loader.exec_module(module)
                patch(module, cfg)
        spec.loader = Loader()
        return spec


def install(cfg):
    check_sources()
    if TARGET in sys.modules:
        patch(sys.modules[TARGET], cfg)
    elif not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder(cfg))
    logger.info('DSV41_EG_WKV_REPLICA_INSTALLED pid=%d min_rows=%d max_rows=%d shadow=%d',
                os.getpid(), cfg['min_rows'], cfg['max_rows'], int(cfg['shadow']))
