# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge the exact Morrowmake stale-width rule into our host-AV hooks."""
import os
from .config import row_max_batch
from .host_rule import configured, _Rounds, choose_width


class HostWidthController:
    def __init__(self, scheduler, av, hw):
        self.scheduler, self.av, self.hw = scheduler, av, hw
        self.rounds = _Rounds(hw)
        self.checked = False
        self.row_max_batch = row_max_batch()

    def _check(self):
        s = self.scheduler
        if s.num_spec_tokens != 5 or s.dynamic_sd_lookup is not None:
            raise ValueError("host width requires fixed K5 draft production, no dynamic SD")
        if s.vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("host width requires PP1")
        capture = s.vllm_config.compilation_config.cudagraph_capture_sizes
        if capture:
            bs = [b for b in self.hw.batch_sizes if b <= s.scheduler_config.max_num_seqs]
            missing = {b * (k + 1) for b in bs for k in self.hw.k_choices} - set(capture)
            if missing:
                raise ValueError("host width has uncaptured token sizes: " + str(sorted(missing)))
        self.checked = True

    def begin_step(self, running):
        self.rounds.padded.clear()
        # AV may cap K; fixed5 host arm never calls the acceptance AV rule.
        cap = self.av.begin_step(running) if self.av else self.scheduler.num_spec_tokens
        if not self.checked:
            self._check()
        b = len(running)
        # Above the cap retain the incoming AV/fixed width as well as all rows.
        if self.row_max_batch and b > self.row_max_batch:
            return cap
        if b not in self.hw.batch_sizes or any(r.is_prefill_chunk for r in running):
            return cap
        if self.av and self.av._forced() is not None:
            return cap
        on_hand = [len(r.spec_token_ids) for r in running if r.spec_token_ids]
        if on_hand:
            cap = min(cap, min(on_hand))
        estimates = self.rounds.estimates(running)
        k = min(self.hw.fallback_k, cap) if estimates is None else choose_width(self.hw, b, estimates, cap)
        for request in running:
            if len(request.spec_token_ids) > k:
                request.spec_token_ids = request.spec_token_ids[:k]
        self.rounds.record(0 if estimates is None else k)
        return k

    def mark_padded(self, req_id):
        self.rounds.padded.add(req_id)
        if self.av:
            self.av.mark_padded(req_id)

    def end_step(self, output):
        self.rounds.scheduled(output)
        if self.av:
            self.av.end_step(output)

    def observed(self, scheduler_output, model_runner_output):
        self.rounds.observed(scheduler_output, model_runner_output)

    def take_padded(self, output):
        return self.av.take_padded(output) if self.av else frozenset()

    def observe(self, *args, **kwargs):
        if self.av:
            self.av.observe(*args, **kwargs)

    def forget(self, req_id):
        self.rounds.by_req.pop(req_id, None)
        if self.av:
            self.av.forget(req_id)


# Loggers here use the vllm.* namespace: only it has vLLM's INFO handler (2026-11-08: the host-width banner below
# was silently dropped under "dsv41_depth.scheduler" and failed the activation check).
def compose(scheduler, av):
    if os.environ.get("DSV41_DEPTH_HOST", "0") != "1":
        return av
    hw = configured()
    if hw is None:
        raise ValueError("host width requires the calibrated row policy")
    from vllm.logger import init_logger
    init_logger("vllm." + __name__).info("DSV41 calibrated host width on: lag=%s K=%s AV=%s",
                               hw.lag, hw.k_choices, av is not None)
    return HostWidthController(scheduler, av, hw)


def observed(controller, scheduler_output, model_runner_output):
    if isinstance(controller, HostWidthController):
        controller.observed(scheduler_output, model_runner_output)
