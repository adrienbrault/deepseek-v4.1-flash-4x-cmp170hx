# SPDX-License-Identifier: MIT
"""QV adapter: reuse FLy admission, witness FINAL output, leave workers intact.

No import finder. QV calls select_scheduler on the constructed instance. Only
finite identity cells write receipts; timing uses the stock AddRequest path.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def require(ok, message):
    if not ok: raise ValueError('r7b identity: '+message)


def root():
    return Path(os.environ['VLLM_DSV41_ADAPTIVE_VERIFY_FORCE_FILE']).parent.parent


def control():
    c = json.loads((root()/'r7b-cell.json').read_text())
    require(type(c['enabled']) is bool and type(c['generation']) is int, 'cell mode/generation')
    return c


def install():
    pass  # The owning QV hook wraps constructed scheduler/worker objects.


class Observer:
    def __init__(self, scheduler):
        path = os.environ['DSV41_R7B_ADMISSION']
        spec = importlib.util.spec_from_file_location('_r7b_shared_admission', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.scheduler = scheduler
        self.released, self.active, self.step, self.policy_generation = [], {}, 0, None
        self.last_control, self.sequence = None, 0
        original_add, original_schedule = scheduler.add_request, scheduler.schedule
        def add_original(req, *args, **kw):
            original_add(req, *args, **kw)
            self.released.append(req)
        self.gate = module.Admission(add_original, control)

        def add(request, *args, **kw):
            c = self.gate.check()
            if not c['enabled']:
                require(not self.gate.pending, 'disabled with pending admission')
                return original_add(request, *args, **kw)
            require(not self.released, 'second group before first scheduler step')
            match = re.search(r'(?:^|-)flycell-([0-9a-f]{16})-row([0-7])(?:-|$)', request.request_id)
            require(match is not None, 'request has no cell row ID')
            require(match[1] == c['request_group'] and int(match[2]) < c['admission'], 'stale/wrong cell row ID')
            ids = getattr(request, 'prompt_token_ids', None)
            require(isinstance(ids,(list,tuple)) and ids and all(type(t) is int and t >= 0 for t in ids),
                    'original integer prompt tokens required')
            self.gate(request, *args, **kw)

        def schedule(*args, **kw):
            c = self.gate.check()
            output = original_schedule(*args, **kw)
            self.step += 1
            av = getattr(scheduler, 'dsv41_av', None)
            require(av is not None, 'adaptive verify state missing')
            forced = av._force_value
            if not c['enabled']:
                require(not self.released and not self.gate.pending, 'policy release with pending admission')
                if self.policy_generation != c['generation']:
                    self.record(dict(kind='policy', control=c, step=self.step, forced=forced,
                                     pending=0, admission='passthrough'))
                    self.policy_generation = c['generation']
                return output
            tokens = dict(output.num_scheduled_tokens)
            if not tokens: return output
            require(forced == 3, 'forced verify K3 not active')
            if self.released:
                ids = [r.request_id for r in self.released]
                new = [r.req_id for r in output.scheduled_new_reqs]
                require(list(tokens) == ids and new == ids, 'first scheduler step split/reordered admission')
                self.active = {key:i for i,key in enumerate(ids)}
                hashes = [hashlib.sha256(json.dumps(list(r.prompt_token_ids), separators=(',',':')).encode()).hexdigest()
                          for r in self.released]
                self.record(dict(kind='admission', control=c, step=self.step, request_ids=ids,
                                 new_request_ids=new, prompt_token_sha256=hashes))
                self.released = []
            require(set(tokens) <= set(self.active), 'foreign scheduler traffic')
            spec = output.scheduled_spec_decode_tokens or {}
            self.record(dict(kind='step', control=c, step=self.step, request_ids=list(tokens),
                             tokens=list(tokens.values()), widths=[len(spec.get(r) or ()) for r in tokens],
                             forced=forced, k=scheduler.dsv41_av_k))
            return output
        scheduler.add_request, scheduler.schedule = add, schedule

    def record(self, row):
        if row['control'] != self.last_control:
            self.sequence = 0
            self.last_control = dict(row['control'])
        self.sequence += 1
        row['sequence'] = self.sequence
        with (root()/'r7b-receipts.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


def select_scheduler(scheduler, arm):
    require(arm in ('off','on'), 'invalid QV arm')
    if not hasattr(scheduler, '_r7b_identity'):
        scheduler._r7b_identity = Observer(scheduler)
    scheduler._r7b_identity.gate.check()


def audit_scheduler(scheduler, arm):
    scheduler._r7b_identity.gate.check()
    return 'r7b:admission'


def select_worker(runner, arm):
    require(arm in ('off','on'), 'invalid QV arm')


def audit_worker(runner, arm):
    return 'r7b:unchanged'
