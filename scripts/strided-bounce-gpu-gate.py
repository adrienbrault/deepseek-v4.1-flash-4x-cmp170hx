#!/usr/bin/env python3
"""Exercise the exact image's trellis gather and pinned H2D ownership on CUDA."""
import json
import os
from types import SimpleNamespace

import torch
import vllm_exl3.exl3 as exl3

assert torch.cuda.device_count() == 4
os.environ['VLLM_EXL3_MOE_TP_ALIGN'] = '128'
checks = 0
for device in range(4):
    with torch.cuda.device(device):
        exl3._BOUNCE_STATE = {'bufs': [None, None], 'evt': [None, None], 'slot': 0, 'cap': 0}
        for flag in ('0', '1'):
            os.environ['VLLM_EXL3_LOAD_STRIDED_BOUNCE'] = flag
            pending = []
            for shape in ((16, 32, 32), (40, 72, 32), (16, 32, 32), (16, 32, 32)):
                for dtype in (torch.int16, torch.int32):
                    for shard, proj in (('w1', 'gate'), ('w2', 'down'), ('w3', 'up')):
                        count = shape[0] * shape[1] * shape[2]
                        source = (torch.arange(count, dtype=torch.int32) % 32000).to(dtype).reshape(shape)
                        old = exl3.shard_exl3_row if shard == 'w2' else exl3.shard_exl3_col
                        expected = old(source, 'trellis', device, 4, aligned=True).to(torch.int16).clone()
                        selected = (exl3._trellis_load_tp_view(source, shard, device, 4)
                                    if flag == '1' else old(source, 'trellis', device, 4, aligned=True))
                        arena = torch.empty((1, *expected.shape), dtype=torch.int16, device=f'cuda:{device}')
                        layer = SimpleNamespace(
                            _exl3_trellis_eid_index={proj: {0: (tuple(expected.shape), 0)}},
                            _exl3_trellis_arena_plan={proj: {tuple(expected.shape): {'arena': arena, 'eid_to_idx': {0: 0}}}},
                            _exl3_trellis_filled=set())
                        exl3._direct_fill_trellis_slot(layer, proj, 0, selected)
                        source.fill_(123)
                        del selected, source
                        pending.append((arena, expected))
            torch.cuda.synchronize(device)
            for arena, expected in pending:
                assert torch.equal(arena[0].cpu(), expected), (device, flag, tuple(expected.shape))
                checks += 1
            del pending
        for event in exl3._BOUNCE_STATE['evt']:
            if event is not None:
                event.synchronize()
        exl3._BOUNCE_STATE = {'bufs': [None, None], 'evt': [None, None], 'slot': 0, 'cap': 0}
print(json.dumps({'status': 'PASS', 'devices': 4, 'checks': checks,
                  'coverage': 'flag parity, uneven aligned TP, strided gather, dtype conversion, source lifetime, buffer reuse/growth'}))
