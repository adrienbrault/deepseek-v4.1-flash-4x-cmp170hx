# SPDX-License-Identifier: MIT
"""Explicit binding for a future QV post-construction adapter.

This is not a QV adapter: current QV cannot express the prefill byte/metric gate.
Call prepare once before scored work, select at drained boundaries, then
after_execute on each scheduler output. No sys.meta_path or sitecustomize.
"""
from types import MethodType
from layers import MoeSchedule, WkvPrefill
from prefetch import EngramPlan


class Binding:
    def __init__(self, runner, item, model_dir=None):
        self.item, self.restores, self.layers = item, [], []
        self.prefetch = None
        if item == 6:
            raise RuntimeError('item 6 needs two pre-captured graph families; use isolated layer until retrofit')
        if item in (7, '7b'):
            import torch
            # Our engine's exl3.py (image and the stage-r18 overlay current-r7b mounts) does
            # `from .tc_moe_hook import try_apply` at every call, so the module attribute is the
            # served seam. The exl3 hook accessor named in Morrowmake's recipe does not exist here
            # (13:20 Paris 11-14 boot: AttributeError on all four workers).
            import vllm_exl3.tc_moe_hook as hook
            old = getattr(hook, 'try_apply', None)
            if not callable(old):
                raise RuntimeError('incumbent TC hook missing')
            schedule = MoeSchedule(item, old)
            schedule.prepare(torch.device('cuda', torch.cuda.current_device()))
            self.layers.append(schedule)
            self.restores.append(lambda: setattr(hook, 'try_apply', old))
            hook.try_apply = schedule
        elif item == 11:
            if model_dir is None:
                raise ValueError('explicit model directory required')
            self.prefetch = EngramPlan(runner, model_dir)
        elif item == 12:
            from dsv41_wkv_replica import replica
            model = runner.get_model()
            model = model.unwrap() if hasattr(model, 'unwrap') else model
            for layer in model.modules():
                state = getattr(layer, replica.STATE, None)
                if state is None or not hasattr(state, 'shards') or not hasattr(layer, 'quant_method'):
                    continue
                # Bind only already-adopted WKV objects, preserving incumbent decode.
                had_local, local = 'forward' in layer.__dict__, layer.__dict__.get('forward')
                old = layer.forward
                wrapper = WkvPrefill(lambda _, x, old=old: old(x), replica)
                self.layers.append(wrapper)
                layer.forward = MethodType(lambda obj, x, w=wrapper: w(obj, x), layer)
                def restore(layer=layer, had_local=had_local, local=local):
                    if had_local:
                        layer.forward = local
                    else:
                        del layer.forward
                self.restores.append(restore)
            if len(self.layers) != 2:
                self.close()
                raise RuntimeError('expected both incumbent Engram WKV replicas')
        else:
            raise ValueError('unknown layer')

    def select(self, on, *, prefill):
        # Prefill is explicit worker metadata, never inferred from concurrency.
        for layer in self.layers:
            layer.select(on, prefill=prefill)
        if self.prefetch is not None:
            self.prefetch.select(on)

    def after_execute(self, scheduler_output):
        if self.prefetch is not None:
            self.prefetch.after_execute(scheduler_output)

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()
