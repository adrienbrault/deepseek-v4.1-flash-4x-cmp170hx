# SPDX-License-Identifier: MIT
"""Measurement-only fair-cap transaction on EngineCore's idle utility loop.

Ring graphs and workers are never changed. No polling or schedule-path I/O.
The serving planner rejects this plugin, its environment and its mounts.
"""
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import time


def need(ok, message):
    if not ok:
        raise ValueError('r7f switch: '+message)


def transact(core, original, *args, **kwargs):
    need(not getattr(core, '_r7f_poisoned', False), 'poisoned; teardown required')
    root = Path(os.environ['R7F_SWITCH_DIR'])
    control = root/'control.json'
    if not control.exists():
        return original(core, *args, **kwargs)
    value = json.loads(control.read_text())
    need(set(value) == {'session', 'generation', 'enabled', 'threshold', 'requested_cap'}, 'control keys')
    need(value['session'] == os.environ['R7F_SWITCH_SESSION'], 'session mismatch')
    need(type(value['generation']) is int and value['generation'] > getattr(core, '_r7f_generation', 0),
         'stale generation')
    need(type(value['enabled']) is bool, 'enabled must be boolean')
    need(type(value['threshold']) is int and value['threshold'] == 2048, 'threshold must be 2048 for this packet')
    need(type(value['requested_cap']) is int and value['requested_cap'] in (1024, 512, 256), 'unregistered cap')
    s = core.scheduler
    need(type(s.fair_min_tokens) is int and s.fair_min_tokens == 2048, 'unexpected threshold state')
    need(not s.has_unfinished_requests() and not s.running and not s.waiting and
         not getattr(core, 'batch_queue', None), 'requires idle, drained engine')
    need((s.fair_cap, s.fair_split) in ((0, 0), (1024, 2), (512, 2), (256, 2)), 'unexpected fair state')
    core._r7f_poisoned = True
    # Native cache reset must succeed before mutating scheduler host attributes.
    # On any partial failure admission/scheduling remain poisoned until teardown.
    need(original(core, *args, **kwargs) is True, 'native reset failed')
    s.fair_cap, s.fair_split = (value['requested_cap'], 2) if value['enabled'] else (0, 0)
    s.fair_min_tokens = value['threshold']
    core._r7f_generation = value['generation']
    ack = dict(value, cap=s.fair_cap, split=s.fair_split, minimum=s.fair_min_tokens, pid=os.getpid(), unix_ns=time.time_ns())
    target = root/f"ack-{value['generation']}.json"
    tmp = target.with_suffix('.new')
    with tmp.open('x') as f:
        json.dump(ack, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(target)
    core._r7f_poisoned = False
    return True


def patch(module):
    cls = module.EngineCore
    if getattr(cls, '_r7f_switch_installed', False):
        return
    original = cls.reset_prefix_cache
    need(list(inspect.signature(original).parameters) == ['self', 'reset_running_requests', 'reset_connector'],
         'EngineCore.reset_prefix_cache API drift')
    @functools.wraps(original)
    def reset(self, *args, **kwargs):
        return transact(self, original, *args, **kwargs)
    cls.reset_prefix_cache = reset
    for name in ('step', 'step_with_batch_queue', 'add_request'):
        old = getattr(cls, name)
        def guarded(self, *args, __old=old, **kwargs):
            need(not getattr(self, '_r7f_poisoned', False), 'poisoned; teardown required')
            return __old(self, *args, **kwargs)
        setattr(cls, name, guarded)
    cls._r7f_switch_installed = True


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'vllm.v1.engine.core':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        need(spec and spec.loader, 'missing EngineCore')
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec) if hasattr(loader, 'create_module') else None
            def exec_module(self, module):
                loader.exec_module(module); patch(module)
        spec.loader = Loader()
        return spec


def register():
    need(os.environ.get('R7F_SWITCH_DIR') == '/r7f-control' and
         len(os.environ.get('R7F_SWITCH_SESSION', '')) == 32, 'measurement envelope missing')
    if 'vllm.v1.engine.core' in sys.modules:
        patch(sys.modules['vllm.v1.engine.core'])
    if not any(isinstance(x, Finder) for x in sys.meta_path):
        sys.meta_path.insert(0, Finder())
