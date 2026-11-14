# SPDX-License-Identifier: Apache-2.0
"""Default-off general plugin. Disabled registration imports only stdlib."""
import os
from .config import options, swa_max_rows


def register():
    selected = options(os.environ)
    swa_max_rows(os.environ)  # a bad or orphaned cap fails the boot, never falls back
    if not any(selected.values()):
        return False
    from .integration import install
    return install(selected)
