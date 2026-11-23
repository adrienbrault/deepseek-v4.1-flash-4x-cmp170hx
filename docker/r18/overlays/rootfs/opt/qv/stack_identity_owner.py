# SPDX-License-Identifier: MIT
"""Keep the r7b canonical identity barrier; audit the fixed per-boot arm."""
import json
import os
from pathlib import Path
import r7b_identity as identity

def install(): pass  # the SAME daily plugin installs before graph capture
select_scheduler = identity.select_scheduler
audit_scheduler = identity.audit_scheduler

def select_worker(runner, arm):
    identity.select_worker(runner, arm)
    if arm != 'off': raise ValueError('confirmation QV channel must remain unchanged/off')

def audit_worker(runner, arm):
    import dsv41_indexer_bounded as plugin
    from vllm.distributed import get_tensor_model_parallel_rank
    on = os.environ['DSV41_INDEXER_BOUNDED'] == '1'
    b = plugin.binding
    # Per execute: no eligible call may fall back. Positive coverage (calls > 0 on
    # all four ranks) is a whole-boot property checked by boot.py from the final
    # rank receipts; vLLM's first profile_run executes before any eligible call.
    if on and (b is None or not b.on or b.calls != b.eligible_calls):
        raise ValueError('missing bounded graph capture or eligible fallback')
    if not on and b is not None: raise ValueError('OFF has bounded binding')
    row = dict(pid=os.getpid(), on=on, bounded_calls=0 if b is None else b.calls,
               eligible_calls=0 if b is None else b.eligible_calls,
               workspace_bytes=0 if b is None else b.bytes)
    path = Path('/qv')/f'bounded-rank{get_tensor_model_parallel_rank()}.json'
    tmp = path.with_suffix('.new'); tmp.write_text(json.dumps(row)); tmp.replace(path)
    return 'r7b:unchanged'
