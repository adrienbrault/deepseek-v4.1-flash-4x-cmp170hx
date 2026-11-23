# SPDX-License-Identifier: MIT
"""Strict metadata and row selection; no sampler parameter changes."""
import math
import struct

KEY = '_dsv41_fly_name_v1'


def names(raw, adapted):
    if not raw or not adapted or raw == adapted:
        raise ValueError('two distinct incumbent names required')
    return {raw: raw, adapted: adapted, raw + '-fast': raw, adapted + '-fast': adapted}


def stamp(params, served, mapping):
    """Keep ordinary requests completely unmodified; reserve our internal key."""
    extra = params.extra_args
    if extra is not None and not isinstance(extra, dict):
        raise ValueError('extra_args must be a dictionary or None')
    if extra is not None and KEY in extra:
        raise ValueError('reserved FLy metadata supplied by client')
    if served not in mapping:
        raise ValueError('unknown served name')
    if served != mapping[served]:
        # Copy both containers: request objects and their nested dictionaries can be reused.
        import copy
        params = copy.copy(params)
        params.extra_args = dict(extra or {}, **{KEY: served})
    return params


def request_name(params, mapping, raw, adapted, lora):
    extra = params.extra_args if params is not None else None
    if extra is not None and not isinstance(extra, dict):
        raise ValueError('malformed extra_args')
    name = (extra or {}).get(KEY)
    twin = adapted if lora is not None else raw
    if name is None:
        return twin
    if not isinstance(name, str) or name not in mapping or name == mapping[name] or mapping[name] != twin:
        raise ValueError('FLy metadata does not match the actual base/LoRA')
    return name


def row_flags(slots, metadata, enabled, mapping):
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    slots = list(slots)
    if len(set(slots)) != len(slots) or any(type(i) is not int or i < 0 for i in slots):
        raise ValueError('invalid/repeated persistent slot')
    selected = []
    for slot in slots:
        name = metadata[slot]  # never silently reuse/fallback for a missing slot
        if name not in mapping:
            raise ValueError('unknown row name')
        selected.append(enabled and name != mapping[name])
    return selected


def merge(standard, fly, flags, torch):
    """Select whole request rows, retaining native strides/values and RNG keys.

    Both calls receive the ORIGINAL tensors, including their storage offsets.
    No compaction, logit processing, seed mutation or floating point math here.
    """
    s, sn = standard
    f, fn = fly
    if s.shape != f.shape or sn.shape != fn.shape or len(s.shape) != 2 or len(sn.shape) != 1:
        raise ValueError('incompatible native outputs')
    if s.shape[0] != len(flags) or sn.shape[0] != len(flags):
        raise ValueError('row count mismatch')
    if s.dtype != f.dtype or sn.dtype != fn.dtype or s.device != f.device or sn.device != fn.device:
        raise ValueError('native dtype/device mismatch')
    mask = torch.tensor(flags, dtype=torch.bool, device=s.device)
    return torch.where(mask[:, None], f, s), torch.where(mask, fn, sn)


def identity(choice):
    """Canonical IEEE-754 bytes of every exposed logprob, with exact IDs.

    JSON lexical spelling is not a float representation; pack the decoded API
    doubles, preserving signed zero. No text-only identity substitute.
    """
    ids, lp = choice.get('token_ids'), choice.get('logprobs')
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('token IDs missing')
    if not isinstance(lp, dict):
        raise ValueError('logprobs missing')
    logs, tops = lp.get('token_logprobs'), lp.get('top_logprobs')
    if not isinstance(logs, list) or not isinstance(tops, list) or len(logs) != len(ids) or len(tops) != len(ids):
        raise ValueError('short logprob evidence')
    def bits(v):
        if type(v) not in (float, int) or not math.isfinite(v):
            raise ValueError('nonfinite/malformed logprob')
        return struct.pack('>d', v).hex()
    result = []
    for value, top in zip(logs, tops):
        if not isinstance(top, dict) or not top or any(not isinstance(k, str) for k in top):
            raise ValueError('top logprobs missing')
        result.append((bits(value), [(k, bits(v)) for k, v in sorted(top.items())]))
    return ids, result
