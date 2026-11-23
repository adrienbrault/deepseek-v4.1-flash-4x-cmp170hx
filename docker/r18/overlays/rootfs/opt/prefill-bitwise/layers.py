# SPDX-License-Identifier: Apache-2.0
"""0001/0002/0004: explicit post-construction layers, disabled by default.

No meta-path finder; caller owns drained switch boundaries and graph families.
TC launch ABI differs between incumbent and the two upstream schedules.
See NOTICE.md. Imports of torch and serving modules are lazy.
"""
import math
import os
from common import HERE, eligible, enabled


class Selection:
    def __init__(self, item):
        self.item = item
        self.on = enabled(os.environ, item)
        self.prefill = False
        self.calls = {'off': 0, 'on': 0}
        self.last = None

    def select(self, on, *, prefill, capturing=False):
        if capturing:
            raise RuntimeError('select only outside graphs at drained boundaries')
        self.on, self.prefill = bool(on), bool(prefill)

    def used(self, arm):
        self.calls[arm] += 1
        self.last = arm


class MoeSchedule(Selection):
    """Compose after the exact incumbent TC hook; unsupported inputs keep it."""
    def __init__(self, item, incumbent, torch_api=None, ops=None):
        if item not in (6, 7, '7b'):
            raise ValueError('TC items are 6, 7 and 7b')
        super().__init__(item)
        if torch_api is None:
            import torch as torch_api
        self.torch = torch_api
        self.incumbent = incumbent
        tag = 'v2try' if item == 6 else 'try'
        if ops is None:
            torch_api.ops.load_library(str(HERE / 'out' / 'lib' / f'_dsv41_mm_{tag}.abi3.so'))
            ops = getattr(torch_api.ops, f'_dsv41_mm_{tag}')
        self.ops = ops
        self.workspaces = {}
        self.witness = None
        self.execution_mode = 'NONE'
        self.execution_guard = None

    def prepare(self, device):
        """Before capture; independent workspace for this schedule/family."""
        path, rows = ('v2', 64) if self.item == 6 else ('prefill', 2048)
        n = getattr(self.ops, f'moe_workspace_size_{path}_try')(rows, 5120, 640, 6, 384)
        self.workspaces[device] = self.torch.empty(n, dtype=self.torch.float32, device=device)

    def __call__(self, x, ids, weights, layer, expert_map, limit):
        t, tc = self.torch, getattr(layer, '_exl3_tc', None)
        # Even OFF must refuse an eligible graph call: it would make a later
        # switch ineffective. Replay itself does not execute this Python hook;
        # the adapter also fences the manager's dispatch and replay entry.
        domain = self.item == '7b' and eligible('7b', int(x.shape[0]), tc, True)
        if domain and (t.cuda.is_current_stream_capturing() or self.execution_mode != 'NONE'):
            raise RuntimeError('7b EAGER_ONLY: graph capture/replay is forbidden')
        if domain and self.execution_guard is not None:
            self.execution_guard()
        emap = None if tc is None else (tc['emap'] if expert_map is None else expert_map)
        ok = (tc is not None and x.dim() == 2 and x.shape[1] == 5120 and x.is_cuda
              and x.dtype == t.bfloat16 and tuple(ids.shape) == (x.shape[0], 6)
              and tuple(weights.shape) == tuple(ids.shape) and ids.dtype in (t.int32, t.int64)
              and tc['intermediate'] in (512, 640) and tc['bits_x2'] in (4, 6)
              and emap.dim() == 1 and emap.numel() == 384 and emap.dtype in (t.int32, t.int64)
              and all(v.device == x.device for v in (ids, weights, emap, tc['ptrs'], tc['meta']))
              and (limit is None or math.isfinite(float(limit))))
        if not self.on or not ok or not eligible(self.item, int(x.shape[0]), tc, self.prefill):
            self.used('off')
            out = self.incumbent(x, ids, weights, layer, expert_map, limit)
            if domain and self.prefill and self.witness is not None:
                self.witness('off', int(x.shape[0]))
            return out
        ws = self.workspaces.get(x.device)
        if ws is None:
            raise RuntimeError('prepare candidate workspace before capture/execution')
        out = t.empty_like(x, memory_format=t.contiguous_format)
        path = 'v2' if self.item == 6 else 'prefill'
        getattr(self.ops, f'moe_out_{path}_try')(
            x.contiguous(), ids.contiguous(), weights.float().contiguous(),
            tc['ptrs'], tc['meta'], emap.contiguous(), tc['intermediate'], tc['bits_x2'],
            out, ws, float(limit or 0))
        self.used('on')  # launch succeeded; never turn CUDA failures into fallback
        if domain and self.witness is not None:
            self.witness('on', int(x.shape[0]))
        return out


class WkvPrefill(Selection):
    """Extend already-adopted replica shards; leave 8..48 decode policy intact."""
    def __init__(self, incumbent_forward, replica_module):
        super().__init__(12)
        self.incumbent = incumbent_forward
        self.replica = replica_module

    def __call__(self, layer, x):
        st = layer.__dict__.get(self.replica.STATE)
        if not self.on or not eligible(12, x.shape[0], prefill=self.prefill):
            self.used('off')
            return self.incumbent(layer, x)
        if st is None or len(st.shards) != 4:
            raise RuntimeError('requires existing byte-qualified TP4 WKV replica adoption')
        if st.runtime.ops.capturing():
            raise RuntimeError('prefill extension is eager only; dual graph families required')
        out = self.replica.replicated(layer.quant_method, st.shards, x, st.runtime.ops.cat)
        self.used('on')
        return out
