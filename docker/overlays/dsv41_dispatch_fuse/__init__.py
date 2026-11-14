# SPDX-License-Identifier: MIT
"""Default-off general plugin: one-launch MoE dispatch prep for the EXL3 coop path.

Disabled discovery imports nothing beyond the standard library.
"""
import os


def register():
    value = os.environ.get('DSV41_DISPATCH_FUSE', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_DISPATCH_FUSE must be 0 or 1, not %r' % value)
    if value == '0':
        return False
    from .hooks import install
    install()
    return True
