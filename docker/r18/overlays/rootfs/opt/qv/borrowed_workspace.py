# SPDX-License-Identifier: MIT
"""Storage/lifetime adapter for the unchanged, measured MoeSchedule('7b')."""
from contextlib import contextmanager
import functools
import threading


def require(ok, why):
    if not ok:
        raise RuntimeError('7b workspace: ' + why)


def schedule_type(base):
    class BorrowedSchedule(base):
        def __init__(self, item, incumbent, runner, **kwargs):
            super().__init__(item, incumbent, **kwargs)
            require(item == '7b', 'only eager item 7b may borrow')
            require(getattr(runner, 'ubatch_runner', None) is None, 'microbatch streams unsupported')
            model = runner.get_model()
            model = model.unwrap() if hasattr(model, 'unwrap') else model
            self.tcs = [m._exl3_tc for m in model.modules() if getattr(m, '_exl3_tc', None) is not None]
            require(self.tcs, 'no prepared target TC layers')
            self.thread = threading.get_ident()
            self.lock = threading.Lock()
            self.streams = {}
            self.buffers = {}
            self.storage_signatures = {}
            self.capturing = False
            self.profiling = False
            self.verified = set()

        def prepare(self, device):
            # This replaces ONLY prepare's torch.empty, before capture/KV sizing.
            # The native op lays its regions out inside this prefix view itself.
            n = self.ops.moe_workspace_size_prefill_try(2048, 5120, 640, 6, 384)
            for tc in self.tcs:
                ws = tc['workspace']
                require(ws.device == device and ws.is_cuda and ws.dtype == self.torch.float32
                        and ws.dim() == 1 and ws.is_contiguous() and ws.data_ptr() % 16 == 0,
                        'incumbent storage device/dtype/layout/alignment')
                require(tc['cap'] == 2048 and tc['prefill'] and ws.numel() >= n,
                        'incumbent workspace too small or wrong cap')
                require(device not in self.buffers or self.buffers[device] is ws,
                        'target layers do not share one incumbent workspace')
                self.buffers[device] = ws
            self.workspaces[device] = self.buffers[device].narrow(0, 0, n)
            self.storage_signatures[device] = (self.buffers[device].data_ptr(), self.buffers[device].numel())
            self.streams[device] = self.torch.cuda.current_stream(device).cuda_stream

        def check_thread(self):
            require(threading.get_ident() == self.thread, 'second launch thread')

        def check_stream(self):
            self.check_thread()
            if not self.capturing:
                for device, stream in self.streams.items():
                    require(self.torch.cuda.current_stream(device).cuda_stream == stream,
                            'eager/replay stream changed')

        @contextmanager
        def capture_scope(self, manager):
            self.check_stream()
            require(not self.capturing and getattr(manager, 'ubatch_runner', None) is None,
                    'nested or multistream capture')
            require(all(getattr(d, 'num_ubatches', 1) == 1
                        for ds in manager._capture_descs.values() for d in ds), 'microbatch descriptor')
            # vLLM captures on a private stream. Complete prior eager/replay
            # work before entry, then complete capture warmups before return.
            # No synchronization is added to scored eager calls or replay.
            self.torch.cuda.synchronize()
            self.capturing = True
            try:
                yield
            finally:
                self.capturing = False
                self.torch.cuda.synchronize()

        def __call__(self, x, ids, weights, layer, expert_map, limit):
            self.check_stream()
            require(self.lock.acquire(blocking=False), 'overlapping hook/replay launch')
            selected = self.on
            try:
                tc = getattr(layer, '_exl3_tc', None)
                target = any(tc is row for row in self.tcs)
                if target:
                    device = tc['workspace'].device
                    require(tc['workspace'] is self.buffers.get(device), 'incumbent workspace replaced')
                    require((tc['workspace'].data_ptr(), tc['workspace'].numel()) == self.storage_signatures[device],
                            'incumbent workspace resized or rebased')
                else:
                    # A drafter/other model retains its own incumbent buffers.
                    self.on = False
                key = None if tc is None else (tc['intermediate'], tc['bits_x2'])
                check = (target and self.on and self.prefill and self.profiling
                         and 128 <= x.shape[0] <= 2048 and key not in self.verified)
                if check:
                    # Native-library identity on real profile inputs, before KV
                    # sizing. Move reference to CPU and drop its GPU output
                    # before candidate execution: no extra live GPU activation.
                    reference = self.incumbent(x, ids, weights, layer, expert_map, limit)
                    require(reference is not None, 'native reference unavailable')
                    reference_cpu = reference.cpu()
                    del reference
                    before = self.calls['on']
                out = super().__call__(x, ids, weights, layer, expert_map, limit)
                if check:
                    require(self.calls['on'] == before + 1, 'profile fell back to incumbent')
                    require(out.shape == reference_cpu.shape and out.dtype == reference_cpu.dtype
                            and self.torch.equal(out.cpu().contiguous().view(self.torch.uint8),
                                                 reference_cpu.contiguous().view(self.torch.uint8)),
                            'native library output bytes differ')
                    self.verified.add(key)
                return out
            finally:
                self.on = selected
                self.lock.release()
    return BorrowedSchedule


def bind(runner, owner):
    """Use the shared Binding and exact shared launch body, replacing storage."""
    import os
    import sys
    require(owner.binding is None or getattr(runner, '_prefill_7b', None) is owner.binding,
            'second runner/hook owner')
    if owner.binding is not None:
        return
    require(all(os.environ.get('DSV41_BITWISE_PREFILL_' + k, '0') == '0'
                for k in ('6', '7', '7B', '11', '12')), 'legacy switches must be OFF')
    names = ('common', 'layers', 'prefetch')
    saved = {n: sys.modules.pop(n, None) for n in names}
    paths = sys.path[:]
    try:
        sys.path.insert(0, '/opt/prefill-bitwise')
        runtime = owner.load('/opt/prefill-bitwise/runtime.py', '_borrowed_binding')
        cls = schedule_type(runtime.MoeSchedule)
        runtime.MoeSchedule = lambda item, incumbent: cls(item, incumbent, runner)
        owner.binding = runtime.Binding(runner, '7b')
    finally:
        sys.path[:] = paths
        for n, old in saved.items():
            sys.modules.pop(n, None)
            if old is not None:
                sys.modules[n] = old
    runner._prefill_7b = owner.binding
    layer = owner.binding.layers[0]
    layer.execution_guard = owner.guard_forward
    layer.witness = lambda arm, rows: owner.events.append(dict(arm=arm, rows=rows, execution='eager'))
    print('PREFILL_7B_BOUND before_capture=1 borrowed=1 new_persistent_bytes=0 item6=BLOCKED', flush=True)


def wrap_profile(method):
    @functools.wraps(method)
    def profiled(runner, *args, **kwargs):
        require(not getattr(runner, '_7b_profiling', False), 'nested profile')
        runner._7b_profiling = True
        try:
            result = method(runner, *args, **kwargs)
            for layer in runner._prefill_7b.layers:
                require(layer.verified == {(tc['intermediate'], tc['bits_x2']) for tc in layer.tcs},
                        'profile did not verify every target TC configuration')
            return result
        finally:
            runner._7b_profiling = False
            for layer in runner._prefill_7b.layers:
                layer.profiling = False
                layer.prefill = False
    profiled._7b_borrowed = True
    return profiled
