"""Generate the opt-in r3 scheduling path from the r18 fair scheduler."""
from textwrap import indent

MODE_ENV = "VLLM_DSV41_PREFILL_SHARE"
MODES = ("off", "half")


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Prefill share scheduler seam changed: {old.strip()}")
    return source.replace(old, new)


def targeted_loops(loops):
    # vLLM still owns budget checks, allocation, admission and request state.
    # Each pass rebuilds scheduled_loras from all running requests, including
    # any waiter admitted by an earlier pass, before checking new admissions.
    # The new seams allow a pass to select a request without moving its peers.
    loops = replace(loops, "            request = self.running[req_index]\n", """            request = self.running[req_index]
            if (request.request_id in num_scheduled_tokens
                    or (target is not None and request is not target)
                    or (mixed and request.is_prefill_chunk and request is not primary)):
                req_index += 1
                continue
""")
    loops = replace(loops, "\n            # Multi-module MTP:", """
            if max_tokens is not None:
                num_new_tokens = min(num_new_tokens, max_tokens)

            # Multi-module MTP:""")
    loops = replace(loops, "                    # The request cannot be scheduled.\n                    # Preempt", """                    if short:
                        # Short work never preempts the primary to obtain KV.
                        break

                    # The request cannot be scheduled.
                    # Preempt""")
    loops = replace(loops, "                    # A deferred free will not help", """                    if primary is not None and preempted_req.request_id in num_scheduled_tokens:
                        # A prior pass already owns these rows and block references.
                        break

                    # A deferred free will not help""")
    loops = replace(loops, """        if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:
            if self.cache_aware_window and not defer_prefills:
""", """        if (not mixed and not preempted_reqs
                and self._pause_state != PauseState.PAUSED_ALL
                and (target is None or target_queue is not None)):
            if target is None and self.cache_aware_window and not defer_prefills:
""")
    loops = replace(loops, "                request_to_skip = from_queue.pop_request()", """                if target is not None:
                    # Keep a rejected target at its original queue position.
                    raise _SmallAdmissionRejected
                request_to_skip = from_queue.pop_request()""")
    # Leave the upstream load-only branch intact for ordinary passes. Target
    # visits require a free slot and rows and cannot initiate a connector load.
    loops = replace(loops, "                if not loads_only:\n", """                if target is not None:
                    if (loads_only or target.request_id in num_scheduled_tokens
                            or target not in target_queue):
                        break
                    request_queue = target_queue
                elif not loads_only:
""")
    loops = replace(loops, "                request = request_queue.peek_request()",
                    "                request = request_queue.peek_request() if target is None else target")
    loops = replace(loops, "                    # Pad new decode requests", """                    if target is not None:
                        # Reclassify using the actual post-replay cache hit.
                        if short != (num_new_tokens <= _SMALL_PREFILL_MAX):
                            break
                        if max_tokens is not None:
                            request_token_budget = min(request_token_budget, max_tokens)

                    # Pad new decode requests""")
    loops = replace(loops, "                        and not prefill_scheduled\n",
                    "                        and not (prefill_scheduled or (short and prefill_admitted))\n")
    loops = replace(loops, "                request = request_queue.pop_request()\n", """                if target is None:
                    request = request_queue.pop_request()
                else:
                    request_queue.remove_requests((request,))
""")
    return replace(loops, """                request.num_computed_tokens = num_computed_tokens
                self._set_kv_fetch_stage(request, None)
""", """                request.num_computed_tokens = num_computed_tokens
                if target is not None and target is primary:
                    primary_replay = num_replay_tokens
                self._set_kv_fetch_stage(request, None)
""")


def add_prefill_share(source, runtime):
    init = "        self._fair_init()\n"
    source = replace(source, init, init + "        self._install_prefill_share()\n")
    method = "    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n"
    start = source.index(method)
    end = source.index("    def _build_kv_connector_meta(", start)
    prefix, step = source[:start], source[start:end]
    run_start = step.index("        # First, schedule the RUNNING requests.")
    run_end = step.index("            # DP prefill balancing: on a step that admitted prefills (release),",
                         run_start)
    finish = step.index("        # Check if the scheduling constraints are satisfied.", run_end)

    passes = """        primary = None
        primary_replay = 0

        def schedule_pass(target=None, target_queue=None, max_tokens=None, short=False, mixed=False):
            nonlocal token_budget, input_budget, encoder_compute_budget, drafts_needed
            nonlocal prefill_scheduled, prefill_admitted, has_sync_kv_loads, primary_replay
"""
    passes += indent(targeted_loops(step[run_start:run_end]), "    ")
    passes += """        def visit(request, queue, limit=None, short=False):
            try:
                schedule_pass(request, queue, limit, short)
            except _SmallAdmissionRejected:
                pass

"""
    dispatch = (runtime / "prefill-share-dispatch.txt").read_text()
    methods = (runtime / "prefill-share-methods.txt").read_text().replace("@MODE_ENV@", MODE_ENV)
    opt_in = step[:run_start].replace("def schedule(", "def _schedule_with_prefill_share(", 1)
    # Decode-only steps need neither the planner nor targeted-pass closures.
    step_header, body = opt_in.split("\n", 1)
    opt_in = step_header + """
        if not (self.waiting or self.kv_holding_waiting or self._inflight_prefills):
            return Scheduler.schedule(self, throttle_prefills)

""" + body
    opt_in += passes + dispatch + step[finish:]
    cls = "class Scheduler(SchedulerInterface):"
    prefix = replace(prefix, cls, "_SMALL_PREFILL_MAX = 8192\n"
                     '_SCHEDULE_PROBE_PREFIX = "chatcmpl-dsv41-sched-"\n\n\n'
                     "class _SmallAdmissionRejected(Exception):\n    pass\n\n\n" + cls)
    tail = source[end:]
    update = "    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:\n"
    trace = (runtime / "schedule-trace-methods.txt").read_text()
    tail = replace(tail, update, trace + update + """        if any(req_id.startswith(_SCHEDULE_PROBE_PREFIX)
               for req_id in scheduler_output.num_scheduled_tokens):
            self._trace_schedule_step(scheduler_output)
""")
    tail = replace(tail, "        self.finished_req_ids.add(request_id)\n", """        if request_id.startswith(_SCHEDULE_PROBE_PREFIX):
            self._trace_schedule_end(request)
        self.finished_req_ids.add(request_id)
""")
    return prefix + step + methods + opt_in + tail
