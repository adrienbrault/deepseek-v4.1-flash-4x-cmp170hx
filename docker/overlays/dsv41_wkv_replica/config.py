# SPDX-License-Identifier: MIT
"""Switch parsing. Stdlib only; an invalid value fails the boot, never guesses."""
import re

SWITCH = 'DSV41_EG_WKV_REPLICA'
MIN_ENV = 'DSV41_EG_WKV_REPLICA_MIN_ROWS'
MAX_ENV = 'DSV41_EG_WKV_REPLICA_MAX_ROWS'
SHADOW_ENV = 'DSV41_EG_WKV_REPLICA_SHADOW'
DRAIN_ENV = 'DSV41_EG_WKV_REPLICA_DRAIN_FILE'
KEYS = (SWITCH, MIN_ENV, MAX_ENV, SHADOW_ENV, DRAIN_ENV)
# Rows = tokens of one target forward. 8 keeps c1 (K5 -> 6 rows, AV K <= 5) on the stock
# gather, where the gather (~0.10 ms) and three extra shard GEMMs (~0.10 ms) cost the same,
# and moves c2 and up. 48 is the stack's largest FULL decode capture size, so every
# captured decode graph from 8 rows up takes the replica and eager prefills keep the stock
# path (a prefill arm sets a larger maximum explicitly).
DEFAULT_MIN_ROWS = 8
DEFAULT_MAX_ROWS = 48
LIMIT_ROWS = 2048           # --max-num-batched-tokens of the stack
# A lossy INT8 prefill all-gather would make the stock path inexact; the replica is exact.
INT8_AG_ENV = 'VLLM_DSV41_INT8_PREFILL_AG'
DRAIN_FILE_RE = re.compile(r'/tmp/[A-Za-z0-9._-]{1,64}')


class ConfigError(ValueError):
    pass


def _rows(env, key, default):
    raw = env.get(key, str(default))
    if not (raw.isascii() and raw.isdigit() and str(int(raw)) == raw and 1 <= int(raw) <= LIMIT_ROWS):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {key}={raw!r} must be an integer 1..{LIMIT_ROWS}')
    return int(raw)


def options(env):
    """Parsed switches; {'enabled': False} when off. Raises ConfigError on any bad value."""
    value = env.get(SWITCH, '0')
    if value not in ('0', '1'):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {SWITCH}={value!r} must be 0 or 1')
    if value == '0':
        stray = [k for k in KEYS[1:] if env.get(k) not in (None, '')]
        if stray:
            raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {stray} set but {SWITCH} is not 1')
        return dict(enabled=False)
    min_rows = _rows(env, MIN_ENV, DEFAULT_MIN_ROWS)
    max_rows = _rows(env, MAX_ENV, DEFAULT_MAX_ROWS)
    if min_rows > max_rows:
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {MIN_ENV}={min_rows} > {MAX_ENV}={max_rows}')
    shadow = env.get(SHADOW_ENV, '0')
    if shadow not in ('0', '1'):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {SHADOW_ENV}={shadow!r} must be 0 or 1')
    drain = env.get(DRAIN_ENV, '') or None
    if drain is not None and (shadow != '1' or not DRAIN_FILE_RE.fullmatch(drain) or '..' in drain):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {DRAIN_ENV} needs {SHADOW_ENV}=1 and a /tmp/<name> path')
    if env.get(INT8_AG_ENV, '0') not in ('', '0'):
        raise ConfigError(f'DSV41_EG_WKV_REPLICA_REFUSED {INT8_AG_ENV}={env[INT8_AG_ENV]!r}: the stock gather '
                          'would be lossy, so the replica would change the output')
    return dict(enabled=True, min_rows=min_rows, max_rows=max_rows, shadow=shadow == '1', drain_file=drain)


def qualification_rows(min_rows, max_rows):
    """Row counts adoption qualifies natively, per Engram layer: every count the replica serves."""
    return range(min_rows, max_rows + 1)
