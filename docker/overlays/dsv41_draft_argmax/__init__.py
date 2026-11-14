# SPDX-License-Identifier: MIT
"""Default-off general plugin: sharded greedy DSpark drafting (no logits gather).

Disabled discovery imports nothing beyond the standard library.
"""
import os


def register():
    value = os.environ.get('DSV41_DRAFT_SHARD_ARGMAX', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_DRAFT_SHARD_ARGMAX must be 0 or 1, not %r' % value)
    if value == '0':
        return False
    from .hooks import install
    install()
    return True
