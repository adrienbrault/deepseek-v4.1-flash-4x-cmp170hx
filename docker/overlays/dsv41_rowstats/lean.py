# SPDX-License-Identifier: MIT
"""The lean statistics write: stage-r18 ``record_top1`` minus its dead entropy term.

stage-r18 (``vllm/utils/dsv41_row_mask.py``, sha256 50abe80d...) records, per DSpark Markov
step and per request row, ``top1 = exp(amax(x) - logsumexp(x))`` and the entropy
``-sum(xlogy(p, p))`` of ``p = softmax(x)``, ``x = logits.float()`` over the full vocabulary.
The only reader of the entropy is the row-mask decision (``dsv41_depth.decision``, kernel and
torch reference), as ``z += w_ent * ent`` with the policy's entropy weight. The served policy
(Morrowmake v1.0.2 calibration) lists features [bias, logit_conf, logit_top1], so ``w_ent`` is
exactly 0 at every position (``load_policy`` leaves unlisted features at 0).

With ``w_ent == 0`` the term is a signed zero when ``ent`` is finite and NaN when ``ent`` is NaN
(entropy is never infinite: it is bounded by log(vocabulary) for any finite row). ``ent`` is NaN
exactly when the row holds a NaN or a +inf or is entirely -inf, which are exactly the rows whose
``top1`` is NaN. So ``top1 * 0.0`` (+0.0 for finite top1, NaN for NaN top1) gives the decision the
same term up to the sign of a zero and, on NaN rows, the NaN's sign and payload, which this does
not reproduce (the stock payload is whatever the device's softmax/xlogy/sum/neg chain produced;
matching it would mean running that chain). Finite rows: ``z + (+0.0)`` and ``z + (-0.0)`` differ
only for ``z == -0.0``, where ``exp(-z) == 1`` either way, so every ``p_i`` is bitwise unchanged.
NaN rows: ``z``, ``p_i`` and the running product are NaN in both, payloads possibly different.

Contract (stage-r18 decision, policy and bridge, current-r2 with ``DSV41_DEPTH_HOST=0``):
bitwise-identical integer decisions (``live_len``, ``row_live``, hence the routing mask and the
accepted tokens) and bitwise-identical finite values for every input, with matching NaN
classification. NaN sign/payload equality is not claimed and is not observable, because:

* the decision's floats (``z``, ``p``, ``prod``, ``expected``, ``score``) reach its outputs only
  through ``take = ok & where(use_tau, prod >= tau, score >= best)``; an ordered comparison with
  a NaN operand is false whatever its payload, so ``best``/``best_k`` and the stored
  ``live_len``/``row_live`` do not depend on it (decision.py:141-160, reference 216-234);
* the only floating-point output, ``p_hat``, is stored with ``mask = k <= p_width`` and
  ``p_width = 0`` unless ``DSV41_DEPTH_HOST=1`` (decision.py:130,143,270-274; policy.py:215;
  begin_step 334-343); the hook refuses a boot with ``p_hat_width != 0``;
* ``row_live`` is consumed as an integer mask (``mask_routes``, ``cap_drafts``: the target MoE
  routes and the rejection sampler), no logger reads statistics, and the slot publication copies
  rows without arithmetic.

Kernels: the top1 expression is textually the stage-r18 one (``amax``, ``logsumexp``, ``sub``,
``exp``, then the strided copy into column 0). Removed per Markov step: ``softmax``, ``xlogy``,
``sum``, ``neg``. Replaced: the strided copy into column 1 stays (copying the new term). Added:
one ``mul`` on the [n] top1 vector. Net: three fewer full-vocabulary-or-reduction ops per step,
of which softmax/xlogy/sum read the full row; dispatched-op counts in RESULTS.md, CUDA kernel
counts and the time saved are unmeasured.
"""


def record(speculator, logits, step):
    """stage-r18 ``record_top1`` body after its ``features_enabled`` check, entropy elided."""
    import torch
    x = logits.float()
    n = x.shape[0]
    out = speculator._dsv41_row_stats
    top1 = torch.exp(x.amax(dim=-1) - torch.logsumexp(x, dim=-1))
    out[:n, 0, step] = top1
    out[:n, 1, step] = top1 * 0.0
