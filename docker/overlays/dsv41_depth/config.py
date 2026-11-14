# SPDX-License-Identifier: Apache-2.0
"""CPU-only settings shared by the planner, scheduler and worker."""
import os
import re

ROW_MAX_BATCH = "DSV41_DEPTH_ROW_MAX_BATCH"


def row_max_batch(env=None):
    """Zero/unset disables the cap; positive N admits live batches <= N."""
    env = os.environ if env is None else env
    value = env.get(ROW_MAX_BATCH, "0")
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(ROW_MAX_BATCH + " must be a nonnegative integer (0: uncapped)")
    return int(value)
