# SPDX-License-Identifier: MIT
"""Extend the incumbent finder; bypass defaults and verify only opted-in spans.

The pinned f760 native RNG is a function of seed, position and token, rather
than a mutable generator (accept-sampled-r1's same-input diagnostic precedent).
Default rows always retain the unmodified standard invocation. Homogeneous
fast chunks need only the FLy invocation; evidence is explicitly untimed.
"""
from contextvars import ContextVar
import functools
import importlib
import importlib.util
import ast
import json
import os
from pathlib import Path
import sys

from .policy import KEY, names, request_name, row_flags, stamp

ACTIVE = ContextVar('fly_names_sample', default=None)
VERIFY = ContextVar('fly_names_verify', default=None)
CONFIG = None


def discover(root, classes):
    """Resolve protocol modules from installed definitions, not guessed paths."""
    found = {name: [] for name in classes}
    for path in sorted((root / 'entrypoints').rglob('*.py')):
        source = path.read_text()
        if not any('class ' + name + '(' in source or 'class ' + name + ':' in source for name in classes):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name in found:
                found[node.name].append('vllm.' + '.'.join(path.relative_to(root).with_suffix('').parts))
    if any(len(v) != 1 for v in found.values()):
        raise ValueError('FLy API definition missing/ambiguous: ' + repr(found))
    return {k: v[0] for k, v in found.items()}


def wrap_protocol(cls, mapping):
    original = cls.to_sampling_params
    if getattr(original, '_fly_names', False):
        return
    @functools.wraps(original)
    def convert(self, *args, **kw):
        params = original(self, *args, **kw)
        # None is the API's existing default-base spelling.
        return stamp(params, self.model or CONFIG['raw'], mapping)
    convert._fly_names = True
    cls.to_sampling_params = convert


def wrap_models(cls, mapping):
    original = cls.init_static_loras
    if getattr(original, '_fly_names', False):
        return
    @functools.wraps(original)
    async def initialize(self, *args, **kw):
        result = await original(self, *args, **kw)
        adapted = CONFIG['adapted']
        if adapted not in self.lora_requests or adapted + '-fast' in self.lora_requests:
            raise ValueError('FLy adapter missing/alias already exists')
        # The SAME object/int_id, no additional adapter load or graph case.
        self.lora_requests[adapted + '-fast'] = self.lora_requests[adapted]
        return result
    initialize._fly_names = True
    cls.init_static_loras = initialize
    show = cls.show_available_models
    @functools.wraps(show)
    async def available(self, *args, **kw):
        result = await show(self, *args, **kw)
        # The native dictionary now contains two references to the same LoRA.
        # Replace its duplicate model card with a card for the opt-in alias.
        adapted = CONFIG['adapted']
        cards = [r for r in result.data if r.id == adapted]
        if len(cards) != 2:
            raise ValueError('FLy expected shared adapter cards')
        cards[-1].id = adapted + '-fast'
        cards[-1].parent = CONFIG['raw'] + '-fast'
        return result
    cls.show_available_models = available


def wrap_runner(cls):
    if getattr(cls.add_requests, '_fly_names', False):
        return
    original, remove, sample = cls.add_requests, cls._remove_request, cls.sample
    gather = cls.gather_batch_req_state
    @functools.wraps(original)
    def add(self, output):
        # f760 calls this even with no new requests, before its zero-token
        # execute_model return (after finished/preempted request removal).
        # Keep missing metadata distinct from a proved empty step.
        self._fly_scheduled_tokens = getattr(output, 'total_num_scheduled_tokens', None)
        if not output.scheduled_new_reqs:
            return original(self, output)
        # Initialize BEFORE original: it calls our removal seam for streaming
        # updates. Never retain a second, stale set across those removals.
        if not hasattr(self, '_fly_slots'):
            self._fly_slots, self._fly_inputs, self._fly_fast_slots = {}, {}, set()
        pending = [(r.req_id, request_name(r.sampling_params, CONFIG['mapping'],
                    CONFIG['raw'], CONFIG['adapted'], r.lora_request))
                   for r in output.scheduled_new_reqs]
        result = original(self, output)
        metadata, inputs = self._fly_slots, self._fly_inputs
        for (req, name), data in zip(pending, output.scheduled_new_reqs):
            slot = int(self.req_states.req_id_to_index[req])
            if slot in metadata:
                raise ValueError('FLy slot reused without removal')
            metadata[slot] = name
            inputs[slot] = data.sampling_params
            if name != CONFIG['mapping'][name]:
                self._fly_fast_slots.add(slot)
        self._fly_active = (getattr(self, '_fly_enabled', False) and bool(self._fly_fast_slots)) or getattr(self, '_fly_evidence', False)
        # Written by the actual request transport seam, after incumbent handling.
        self._fly_transport_used = getattr(self, '_fly_transport_marker', 'fly:transport:on' if getattr(self, '_fly_enabled', False) else 'fly:transport:off')
        return result
    add._fly_names = True
    @functools.wraps(remove)
    def delete(self, req):
        slot = self.req_states.req_id_to_index.get(req)
        result = remove(self, req)
        if slot is not None:
            getattr(self, '_fly_slots', {}).pop(int(slot), None)
            getattr(self, '_fly_inputs', {}).pop(int(slot), None)
            getattr(self, '_fly_fast_slots', set()).discard(int(slot))
            self._fly_active = ((getattr(self, '_fly_enabled', False) and bool(getattr(self, '_fly_fast_slots', ())))
                                or getattr(self, '_fly_evidence', False))
        return result
    @functools.wraps(sample)
    def sampled(self, hidden, batch, grammar):
        # No array walk/allocation, ContextVar, tensor operation, or recorder on
        # the common path. The flag is maintained only at admission/removal and
        # arm changes. Diagnostic evidence is an explicit, untimed exception.
        if not getattr(self, '_fly_active', False):
            return sample(self, hidden, batch, grammar)
        # A live fast request may be unscheduled this step (e.g. chunked
        # prefill). A single host set-intersection check handles that case
        # without converting slots, validating names or building context.
        if (not getattr(self, '_fly_evidence', False)
                and self._fly_fast_slots.isdisjoint(batch.idx_mapping_np)):
            return sample(self, hidden, batch, grammar)
        slots = [int(x) for x in batch.idx_mapping_np]
        flags = row_flags(slots, self._fly_slots, self._fly_enabled, CONFIG['mapping'])
        if not any(flags) and not getattr(self, '_fly_evidence', False):
            return sample(self, hidden, batch, grammar)
        if getattr(self, '_fly_cell', {}).get('diagnostic'):
            effective(self, batch, slots, 'sample-entry')
        token = ACTIVE.set((self, batch, dict(zip(slots, flags))))
        try:
            return sample(self, hidden, batch, grammar)
        finally:
            ACTIVE.reset(token)
    @functools.wraps(gather)
    def gathered(self, *args, **kw):
        result = gather(self, *args, **kw)
        # This real input-transport seam also executes during dummy graph
        # capture, when execute_model deliberately skips add_requests.
        self._fly_transport_used = getattr(self, '_fly_transport_marker', 'fly:transport:off')
        return result
    cls.add_requests, cls._remove_request, cls.sample, cls.gather_batch_req_state = add, delete, sampled, gathered


def wrap_verifier(module):
    cls = module.RejectionSampler
    if getattr(cls._verify, '_fly_names', False):
        return
    original, native = cls._verify, module.rejection_sample
    import inspect
    signature = inspect.signature(original)
    native_sig = inspect.signature(native)
    slot_argument = list(signature.parameters).index('idx_mapping_np') - 1
    if 'draft_logits_cache' not in native_sig.parameters:
        raise ValueError('FLy expected the v0.34 persistent draft-logits cache ABI')
    native_positions = {name:i for i,(name,p) in enumerate(native_sig.parameters.items())
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)}
    fly_options = dict(fly_window_size=1, fly_entropy_threshold=.3, fly_entropy_top_k=3)
    @functools.wraps(original)
    def verify(self, *args, **kw):
        state = ACTIVE.get()
        if state is None:
            return original(self, *args, **kw)
        if (self.fly_window_size != 0 or self.use_block_verification or self.synthetic_conditional_rates is not None
                or getattr(self, 'enable_adaptive_verification', False)):
            raise ValueError('FLy requires incumbent standard verifier')
        runner, batch, flags_by_slot = state
        # The installed signature is resolved once; no per-chunk bind object.
        indices = kw['idx_mapping_np'] if 'idx_mapping_np' in kw else args[slot_argument]
        slots = [int(i) for i in indices]
        flags = [flags_by_slot[i] for i in slots]
        if not any(flags) and not getattr(runner, '_fly_evidence', False):
            return original(self, *args, **kw)
        token = VERIFY.set((runner, batch, slots, flags))
        try:
            return original(self, *args, **kw)
        finally:
            VERIFY.reset(token)
    verify._fly_names = True
    @functools.wraps(native)
    def rejection(*args, **kw):
        context = VERIFY.get()
        if context is None:
            return native(*args, **kw)  # exact incumbent args and outputs
        runner, batch, slots, flags = context
        if getattr(runner, '_fly_cell', {}).get('diagnostic'):
            from dsv41_lora_argmax.hooks import ACTIVE as TARGET
            effective(runner, batch, slots, 'target-argmax' if TARGET.get() is not None else 'gathered')
        window_index = native_positions.get('fly_window_size', len(args))
        window = args[window_index] if window_index < len(args) else kw.get('fly_window_size', 0)
        if window != 0:
            raise ValueError('global FLy conflicts with per-row selection')
        evidence = getattr(runner, '_fly_evidence', False)
        # Homogeneous fast traffic needs ONE native call, not standard + FLy.
        if all(flags) and not evidence:
            options = dict(kw)
            positional = args
            for name,value in fly_options.items():
                index = native_positions.get(name, len(args))
                if index < len(args):
                    if positional is args: positional = list(args)
                    positional[index] = value
                else: options[name] = value
            return native(*positional, **options)
        bound = native_sig.bind(*args, **kw)
        bound.apply_defaults()
        standard = native(*args, **kw)
        result = standard
        fly = standard
        if any(flags):
            # Evidence keeps standard outputs; production can replace selected
            # integer rows in the fresh native output allocation in place.
            if evidence:
                result = tuple(x.clone() for x in standard)
            fly_selected(native, bound, batch, slots, flags, result)
            fly = result
        if evidence:
            record(runner, batch, slots, flags, standard, fly, result, bound.arguments['cu_num_logits'])
        return result
    cls._verify, module.rejection_sample = verify, rejection


def fly_selected(native, bound, batch, slots, flags, result):
    """Slice contiguous opted-in spans, preserving persistent RNG/draft indices.

    CPU boundaries come from the real InputBatch. No CUDA reads or vocab-sized
    gather/copy. Chunk offsets are relative to the _verify call, not the batch.
    Adaptive verification is off on this pinned serving stack.
    """
    arguments = bound.arguments
    if all(flags):
        arguments.update(fly_window_size=1, fly_entropy_threshold=.3, fly_entropy_top_k=3)
        output = native(*bound.args, **bound.kwargs)
        for dst, src in zip(result, output): dst.copy_(src)
        return
    batch_offsets = batch.cu_num_logits_np
    lengths = {int(slot): int(b)-int(a) for slot,a,b in
               zip(batch.idx_mapping_np,batch_offsets,batch_offsets[1:])}
    if len(set(slots)) != len(slots):
        raise ValueError('FLy repeated verifier persistent slot')
    # A sampler shard can reorder rows *again* after runner.sample. Derive the
    # chunk layout by persistent slot, never by a compact/global row position.
    offsets = [0]
    for slot in slots: offsets.append(offsets[-1]+lengths[slot])
    if offsets[-1] != arguments['target_logits'].shape[0]:
        raise ValueError('FLy chunk logits/CPU boundaries differ')
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        end = i+1
        while end < len(flags) and flags[end]: end += 1
        lo, hi = offsets[i], offsets[end]
        # BoundArguments must be separate for each span; never mutate the
        # original/default call or slice arrays indexed by persistent slots.
        # In v0.34 draft_logits_cache includes a persistent-slot row mapping.
        # Pass the same cache and mapping through, even for sharded logits.
        selected = type(bound)(bound.signature, dict(arguments))
        for key in ('target_logits','draft_sampled','pos','expanded_idx_mapping','expanded_local_pos',
                    'contexts','watermarking_skip_mask'):
            value = selected.arguments.get(key)
            if value is not None: selected.arguments[key] = value[lo:hi]
        selected.arguments['idx_mapping'] = arguments['idx_mapping'][i:end]
        selected.arguments['cu_num_logits'] = arguments['cu_num_logits'][i:end+1] - lo
        selected.arguments.update(fly_window_size=1, fly_entropy_threshold=.3, fly_entropy_top_k=3)
        output = native(*selected.args, **selected.kwargs)
        for dst, src in zip(result, output): dst[i:end].copy_(src)
        i = end


def record(runner, batch, slots, flags, standard, fly, result, cu):
    """Per-request executed-path evidence, no prompt/output text or token IDs."""
    from vllm.distributed import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    s, f, n = (x[1].detach().cpu().tolist() for x in (standard, fly, result))
    offsets = cu.detach().cpu().tolist()
    widths = [b-a-1 for a, b in zip(offsets, offsets[1:])]
    request_ids = {int(slot): req for slot, req in zip(batch.idx_mapping_np, batch.req_ids)}
    rows = [dict(request_id=request_ids[slot], slot=slot, name=runner._fly_slots[slot],
                 enabled=runner._fly_enabled, fly=flag, standard_count=a, fly_count=b,
                 selected_count=c, draft_width=width, rescued_drafts=b-a if flag else 0)
            for slot, flag, a, b, c, width in zip(slots, flags, s, f, n, widths)]
    if len(rows) != len(slots):
        raise ValueError('FLy incomplete row counter data')
    with open('/qv/fly-rank' + str(rank) + '.jsonl', 'a') as stream:
        stream.write(json.dumps(dict(rows=rows)) + '\n')


def select_worker(runner, arm):
    if arm not in ('off', 'on'):
        raise ValueError('FLy invalid arm')
    # Both witnesses belong to THIS execute, including the cached-arm path.
    runner._fly_transport_used = None
    runner._fly_scheduled_tokens = None
    changed = getattr(runner, '_fly_arm', None) != arm
    if not changed and not CONFIG.get('control_each_step', False):
        return
    wrap_runner(type(runner))  # real constructed class; no second import finder
    runner._fly_arm = arm
    runner._fly_enabled = arm == 'on'
    runner._fly_transport_marker = 'fly:transport:' + arm
    # The finite legacy diagnostic changes cells without switching arms. The
    # production/fast runner reads control only at explicit arm transitions.
    if changed or CONFIG.get('control_each_step', False):
        runner._fly_cell = cell_control()
    runner._fly_evidence = bool(runner._fly_cell.get('record') or runner._fly_cell.get('diagnostic'))
    runner._fly_active = ((runner._fly_enabled and bool(getattr(runner, '_fly_fast_slots', ())))
                          or runner._fly_evidence)


def cell_control():
    return json.loads(Path('/qv/fly-control.json').read_text())


def select_scheduler(scheduler, arm):
    from .admission import install
    install(scheduler, cell_control)


def effective(runner, batch, slots, path):
    from vllm.distributed import get_tensor_model_parallel_rank
    observations = effective_rows(runner, batch, slots, path)
    rank = get_tensor_model_parallel_rank()
    with open('/qv/fly-diagnostic-rank' + str(rank) + '.jsonl', 'a') as stream:
        for row in observations:
            stream.write(json.dumps(dict(rank=rank, **row), allow_nan=False) + '\n')


def effective_rows(runner, batch, slots, path):
    """Observe real f760 persistent slots and staged GPU sampler state.

    Layout: fastloop-r1/in/f760-capture-sources/.../gpu/model_runner.py,
    gpu/lora_utils.py; SamplingStates fields are used by incumbent eligibility.py.
    This runs only in the finite diagnostic, never in a timing window.
    """
    lora_module = importlib.import_module(CONFIG['lora'])
    sampling = runner.sampler.sampling_states
    requests = {int(slot): req for slot, req in zip(batch.idx_mapping_np, batch.req_ids)}
    # Copy first, then index by the ORIGINAL persistent slots. No compact-index
    # assumption; these arrays may themselves be views of larger allocations.
    temps = sampling.temperature.gpu.detach().cpu().tolist()
    seeds = sampling.seeds.gpu.detach().cpu().tolist()
    verifier_seeds = runner.sampler.sampling_seeds.detach().cpu().tolist()
    observations = []
    for slot in slots:
        params = runner._fly_inputs[slot]
        adapter = runner.lora_state.lora_requests.get(requests[slot])
        fields = {key: getattr(params, key) for key in
                  ('temperature', 'seed', 'top_p', 'top_k', 'min_p', 'min_tokens', 'max_tokens',
                   'ignore_eos', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'logprobs')}
        fields['all_stop_token_ids'] = sorted(params.all_stop_token_ids)
        fields['sampling_type'] = params.sampling_type.name
        observations.append(dict(cell=runner._fly_cell['cell'], request_id=requests[slot], slot=slot,
            name=runner._fly_slots[slot], params=fields, path=path,
            lora_id=int(runner.lora_state.lora_ids[slot]), no_lora_id=int(lora_module.NO_LORA_ID),
            adapter=None if adapter is None else dict(name=adapter.lora_name, id=adapter.lora_int_id),
            temperature_cpu=float(sampling.temperature.np[slot]), temperature_gpu=float(temps[slot]),
            seed_cpu=int(sampling.seeds.np[slot]), seed_gpu=int(seeds[slot]),
            verifier_seed_gpu=int(verifier_seeds[slot]),
            top_k_cpu=int(sampling.top_k.np[slot]), top_p_cpu=float(sampling.top_p.np[slot]),
            logprobs_mode=runner.sampler.logprobs_mode,
            needs_logits_processing=bool(runner.sampler.needs_logits_processing[slot]),
            batch_slots=[int(x) for x in batch.idx_mapping_np],
            batch_request_ids=list(batch.req_ids), num_draft_tokens=int(batch.num_draft_tokens)))
    return observations


def audit_worker(runner, arm):
    value = runner._fly_transport_used
    count = getattr(runner, '_fly_scheduled_tokens', None)
    empty = type(count) is int and count == 0
    if runner._fly_arm != arm or (value != runner._fly_transport_marker and not empty):
        raise ValueError('FLy request transport did not execute')
    # QV's registered path denotes the selected policy. On an empty execute
    # there is no transport to witness; never fabricate _fly_transport_used.
    return runner._fly_transport_marker


def install():
    global CONFIG
    config = json.loads(os.environ['DSV41_FLY_NAMES'])
    config['mapping'] = names(config['raw'], config['adapted'])
    if CONFIG is not None:
        if CONFIG != config:
            raise ValueError('FLy configuration changed after install')
        return
    CONFIG = config
    from dsv41_lora_argmax import hooks as base
    base.install()
    layout = discover(Path(importlib.util.find_spec('vllm').origin).parent,
                      ('CompletionRequest', 'ChatCompletionRequest', 'OpenAIServingModels'))
    for name in ('CompletionRequest', 'ChatCompletionRequest'):
        wrap_protocol(getattr(importlib.import_module(layout[name]), name), config['mapping'])
    wrap_models(getattr(importlib.import_module(layout['OpenAIServingModels']), 'OpenAIServingModels'), config['mapping'])
    targets = {config['runner'], config['rejection']}
    base.TARGETS.update(targets)
    previous = base.patch
    def patch(module):
        previous(module)
        if module.__name__ == config['runner']:
            wrap_runner(module.GPUModelRunner)
        elif module.__name__ == config['rejection']:
            wrap_verifier(module)
    base.patch = patch
    for name in targets & sys.modules.keys():
        patch(sys.modules[name])
    print('FLY_NAMES_INSTALLED protocol_layout=' + json.dumps(layout, sort_keys=True), flush=True)
