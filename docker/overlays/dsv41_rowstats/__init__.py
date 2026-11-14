# SPDX-License-Identifier: MIT
"""Default-off general plugin: row-mask draft statistics without the dead entropy term.

Disabled discovery imports nothing beyond the standard library.
"""
import os


def register():
    value = os.environ.get('DSV41_ROWSTATS_LEAN', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_ROWSTATS_LEAN must be 0 or 1, not %r' % value)
    if value == '0':
        return False
    from .hooks import install
    install()
    return True
