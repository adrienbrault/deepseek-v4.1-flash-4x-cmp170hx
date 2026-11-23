# SPDX-License-Identifier: MIT
"""Default-off general plugin. Disabled discovery imports no torch or vLLM."""
import os


def register():
    if not (os.environ.get('DSV41_MERGED_WOB') or
            os.environ.get('DSV41_TARGET_ARGMAX', '0') == '1'):
        return False
    from .hooks import install
    install()
    return True
