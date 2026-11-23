# SPDX-License-Identifier: MIT
"""Stock-vs-lean equivalence checks shared by the CPU tests and the in-image preflight.

Nothing here runs in serving. ``row_mask`` is the stage-r18 ``dsv41_row_mask`` module (its
``record_top1`` unwrapped), ``policy``/``decision`` the stage-r18 ``dsv41_depth`` modules.
Integer decision outputs (``live_len``, ``row_live``) and ``top1`` are compared on raw bits.
Two comparisons are deliberately weaker, and say so in their names: the entropy column and
``p_hat`` are compared by NaN classification plus raw bits on the non-NaN entries
(``finite_bits_and_nan_class``: NaN sign and payload are NOT compared; -0.0 != +0.0 is). The
contract this checks (lean.py) needs no more: every float the decision computes reaches its
integer outputs only through ordered comparisons, which are false for a NaN of any sign or
payload, and the served stack produces no ``p_hat``.
"""
from types import SimpleNamespace

from . import lean

VOCAB = 129280   # DeepSeek-V4.1 vocabulary: the served full-row width
STEPS = 5        # DSpark K5 Markov steps


def bits(t):
    import torch
    return t.contiguous().view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def finite_bits_and_nan_class(a, b):
    """Same NaN positions, and raw-bit equality everywhere else (so +-0 and +-inf are told apart).

    NaN sign and payload are not compared: this is not raw-bit equality on NaN entries."""
    nan = a.isnan()
    return bool((nan == b.isnan()).all()) and bool((bits(a[~nan]) == bits(b[~nan])).all())


def nan_bits_equal(a, b):
    """Raw-bit equality of the NaN entries both tensors share (informational only)."""
    nan = a.isnan() & b.isnan()
    return bool((bits(a[nan]) == bits(b[nan])).all())


# NaN encodings exercised as logits: -qNaN, qNaN with payload, +sNaN, -sNaN (per dtype width).
NAN_PATTERNS = {2: (0xffc0, 0x7fc1, 0x7f81, 0xff81),
                4: (0xffc00000, 0x7fc00001, 0x7f800001, 0xff800001)}


def with_bits(t, index, pattern):
    """``t`` with entry ``index`` replaced by the raw bit pattern (set in the target dtype: a
    dtype conversion would canonicalize the NaN)."""
    view = bits(t.clone())
    width = 8 * t.element_size()
    view[index] = pattern - (1 << width) if pattern >= 1 << (width - 1) else pattern
    return view.view(t.dtype)


def adversarial_rows(vocab=VOCAB, dtype=None, seed=0):
    """[rows, vocab] logits: realistic rows plus every non-finite pattern the argument covers."""
    import torch
    dtype = dtype or torch.bfloat16
    g = torch.Generator().manual_seed(seed)
    rows = []

    def add(r):
        rows.append(r.to(dtype))
    for scale in (0.5, 4.0, 30.0):                      # flat .. peaked distributions
        add(torch.randn(vocab, generator=g) * scale)
    peaked = torch.randn(vocab, generator=g)
    peaked[7] = 60.0                                   # top1 ~ 1, entropy ~ 0
    add(peaked)
    tie = torch.zeros(vocab)
    tie[[3, 1000, vocab - 1]] = 9.0                    # tied maxima
    add(tie)
    add(torch.zeros(vocab))                            # uniform: top1 = 1/V
    add(-torch.zeros(vocab))                           # all -0.0
    sub = torch.full((vocab,), 1e-40)                  # subnormal (fp32 only; bf16 keeps its own)
    sub[::2] = -1e-40
    add(sub)
    big = torch.randn(vocab, generator=g) * 1e30
    big[11] = 3.0e38                                   # near the largest finite value
    add(big)
    neg = torch.randn(vocab, generator=g)
    neg[: vocab // 2] = float('-inf')                  # finite max, many -inf
    add(neg)
    one = torch.full((vocab,), float('-inf'))
    one[5] = -3.0                                      # a single finite entry: top1 = 1, entropy 0
    add(one)
    add(torch.full((vocab,), float('-inf')))           # all -inf: top1 NaN, entropy NaN
    nan = torch.randn(vocab, generator=g)
    nan[123] = float('nan')                            # one NaN
    add(nan)
    pinf = torch.randn(vocab, generator=g)
    pinf[77] = float('inf')                            # one +inf
    add(pinf)
    both = torch.randn(vocab, generator=g)
    both[[4, 9]] = torch.tensor([float('inf'), float('-inf')])
    add(both)
    add(torch.full((vocab,), float('nan')))           # all NaN
    for i, pattern in enumerate(NAN_PATTERNS[rows[0].element_size()]):
        rows.append(with_bits(torch.randn(vocab, generator=g).to(dtype), 31 + i, pattern))
    rows.append(with_bits(torch.zeros(vocab, dtype=dtype), slice(None), NAN_PATTERNS[rows[0].element_size()][0]))
    return torch.stack(rows)                           # (the last row: every entry -qNaN)


def stock_and_lean(row_mask, logits_steps, max_reqs=None, max_draft=STEPS):
    """Run the unwrapped stock ``record_top1`` and ``lean.record`` over the same per-step logits.

    ``logits_steps``: list of [n, vocab] tensors (one per Markov step). Returns the two
    [max_reqs, 2, max_draft] float32 statistics buffers.
    """
    import torch
    n = logits_steps[0].shape[0]
    max_reqs = max_reqs or n
    device = logits_steps[0].device
    stock = SimpleNamespace(_dsv41_row_stats=torch.zeros(max_reqs, 2, max_draft, device=device))
    slim = SimpleNamespace(_dsv41_row_stats=torch.zeros(max_reqs, 2, max_draft, device=device))
    record_top1 = getattr(row_mask.record_top1, '__wrapped__', row_mask.record_top1)
    # The stock body starts with features_enabled(n), which reads the bridge's module state:
    # stand in an uncapped state for the call (restored after), as a served c1 step has.
    saved = row_mask._STATE
    if saved is None:
        row_mask._STATE = SimpleNamespace(row_max_batch=0)
    try:
        for step, logits in enumerate(logits_steps):
            record_top1(stock, logits, step)
            lean.record(slim, logits, step)
    finally:
        row_mask._STATE = saved
    return stock._dsv41_row_stats, slim._dsv41_row_stats


def compare_stats(stock, slim):
    """Problems (empty = equivalent): top1 raw bits, entropy column NaN classification, lean entropy +0.0 or NaN."""
    problems = []
    if not bool((bits(stock[:, 0]) == bits(slim[:, 0])).all()):
        problems.append('top1 bits differ')
    if not bool((stock[:, 1].isnan() == slim[:, 1].isnan()).all()):
        problems.append('entropy NaN pattern differs')
    if bool(stock[:, 1].isinf().any()):
        problems.append('stock entropy is infinite (argument assumes finite or NaN)')
    lean_ent = slim[:, 1]
    finite = ~lean_ent.isnan()
    if not bool((bits(lean_ent[finite]) == 0).all()):
        problems.append('lean entropy is not +0.0 where finite')
    return problems


def decision_outputs(policy_mod, decision_mod, policy_path, conf, stats, num_reqs, num_drafts,
                     max_reqs=8, max_draft=STEPS, p_hat_width=0, interpret=False):
    """live_len, row_live and p_hat of one uniform step from the given per-slot features.

    ``conf``: [max_reqs, max_draft], ``stats``: [max_reqs, 2, max_draft] (slot order = batch order).
    ``interpret`` runs ``decision.compute_row_mask`` (the Triton kernel under TRITON_INTERPRET=1);
    otherwise ``decision.reference_row_mask`` (its torch transcription).
    """
    import torch
    max_tokens = max_reqs * (max_draft + 1)
    policy = policy_mod.load_policy(policy_path, max_reqs, max_draft, [0] + [max_draft] * max_reqs)
    state = policy_mod.RowMaskState(max_reqs, max_tokens, max_draft, torch.device('cpu'))
    state.set_policy(policy)
    state.conf_by_slot.copy_(conf)
    state.stats_by_slot.copy_(stats)
    state.row_live.fill_(0)                     # every row must be written by the decision
    idx = torch.arange(max_reqs, dtype=torch.int32)
    p_hat = torch.full((num_reqs, p_hat_width), -7.0) if p_hat_width else None
    run = decision_mod.compute_row_mask if interpret else decision_mod.reference_row_mask
    run(state, idx, num_reqs, num_drafts, p_hat)
    return dict(live_len=state.live_len[:num_reqs].clone(), row_live=state.row_live.clone(),
                p_hat=None if p_hat is None else p_hat.clone(), weights=state.weights.clone())


def compare_decisions(policy_mod, decision_mod, policy_path, conf, stock, slim, interpret=False,
                      shapes=None, p_hat_width=0, info=None):
    """Problems over every (num_reqs, num_drafts) shape: live_len and row_live raw-bit equal;
    with a host width (not served in current-r2) p_hat by ``finite_bits_and_nan_class``.

    ``info`` (a dict) accumulates ``p_hat_nan`` (NaN p_hat entries compared) and
    ``p_hat_nan_bits_differ`` (shapes whose shared NaN entries differ in raw bits)."""
    problems, compared = [], 0
    info = {} if info is None else info
    info.setdefault('p_hat_nan', 0)
    info.setdefault('p_hat_nan_bits_differ', 0)
    max_reqs = stock.shape[0]
    shapes = shapes or [(b, k) for b in range(1, max_reqs + 1) for k in range(1, STEPS + 1)]
    for b, k in shapes:
        a = decision_outputs(policy_mod, decision_mod, policy_path, conf, stock, b, k, max_reqs,
                             p_hat_width=p_hat_width, interpret=interpret)
        z = decision_outputs(policy_mod, decision_mod, policy_path, conf, slim, b, k, max_reqs,
                             p_hat_width=p_hat_width, interpret=interpret)
        for key in ('live_len', 'row_live'):
            if not bool((bits(a[key]) == bits(z[key])).all()):
                problems.append(f'{key} differs at num_reqs={b} num_drafts={k}')
        if p_hat_width:
            if not finite_bits_and_nan_class(a['p_hat'], z['p_hat']):
                problems.append(f'p_hat finite bits or NaN class differ at num_reqs={b} num_drafts={k}')
            info['p_hat_nan'] += int(a['p_hat'].isnan().sum())
            info['p_hat_nan_bits_differ'] += int(not nan_bits_equal(a['p_hat'], z['p_hat']))
        compared += 1
    return problems, compared


def dispatched_ops(fn, *args):
    """aten overloads ``fn`` dispatches (TorchDispatchMode: after composite decomposition, i.e.
    what reaches a backend kernel), in call order. A device-agnostic launch-count proxy."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(str(func.overloadpacket.__name__))
            return func(*args, **(kwargs or {}))
    with Record() as rec:
        fn(*args)
    return rec.ops


def record_ops(row_mask, vocab=VOCAB, dtype=None):
    """Dispatched ops of one stock ``record_top1`` and one ``lean.record`` call, c1 served shape."""
    import torch
    x = torch.randn(1, vocab).to(dtype or torch.bfloat16)
    spec = SimpleNamespace(_dsv41_row_stats=torch.zeros(1, 2, STEPS))
    record_top1 = getattr(row_mask.record_top1, '__wrapped__', row_mask.record_top1)
    saved = row_mask._STATE
    if saved is None:
        row_mask._STATE = SimpleNamespace(row_max_batch=0)
    try:
        stock = dispatched_ops(record_top1, spec, x, 0)
    finally:
        row_mask._STATE = saved
    return dict(stock=stock, lean=dispatched_ops(lean.record, spec, x, 0))
