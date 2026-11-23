# SPDX-License-Identifier: Apache-2.0
"""Strict, default-off options; safe to import without torch."""
FLAGS = ("DSV41_ATTN_SWA_FUSED", "DSV41_ATTN_TOPK_GLUE")
# 2026-11-09: the fused SWA metadata build's row cap. Not a switch: it only
# widens DSV41_ATTN_SWA_FUSED=1, and its default keeps the reviewed 32 rows.
SWA_MAX_ROWS = "DSV41_ATTN_SWA_FUSED_MAX_ROWS"
SWA_MAX_ROWS_DEFAULT = 32
SWA_MAX_ROWS_LIMIT = 64


def options(environ):
    result = {}
    for flag in FLAGS:
        value = environ.get(flag, "0")
        if value not in ("0", "1"):
            raise ValueError(f"{flag} must be 0 or 1")
        result[flag] = value == "1"
    return result


def swa_max_rows(environ):
    """Rows of the largest decode batch the fused SWA build takes (1-64)."""
    value = environ.get(SWA_MAX_ROWS)
    if value is None:
        return SWA_MAX_ROWS_DEFAULT
    if not (value.isascii() and value.isdigit() and value[0] != "0"
            and int(value) <= SWA_MAX_ROWS_LIMIT):
        raise ValueError(f"{SWA_MAX_ROWS} must be an integer 1-{SWA_MAX_ROWS_LIMIT}")
    if environ.get(FLAGS[0], "0") != "1":
        raise ValueError(f"{SWA_MAX_ROWS} requires {FLAGS[0]}=1")
    return int(value)
