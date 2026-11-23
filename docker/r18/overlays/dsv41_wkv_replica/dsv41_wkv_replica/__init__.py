# SPDX-License-Identifier: MIT
"""Default-off vLLM general plugin: Engram ``wkv`` replicated for decode, no output all-gather.

Disabled registration imports only the standard library (config.py).
"""
import os

from .config import options


def register():
    cfg = options(os.environ)
    if not cfg['enabled']:
        return False
    from .hooks import install
    install(cfg)
    return True
