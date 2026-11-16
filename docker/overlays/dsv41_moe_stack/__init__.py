# SPDX-License-Identifier: MIT
def register():
    import os
    if os.environ.get('DSV41_MOE_STACK','0') == '0': return
    import sys
    # The adapter lives inside this package so the shared dry boot (lib/dry_boot.py
    # hook_evidence) attributes its wrappers to this plugin (2026-11-15 host dry).
    # Alias it so qv_candidate's `import stack_plugin` sees the same module/binding.
    from . import stack_plugin
    sys.modules.setdefault('stack_plugin', stack_plugin)
    stack_plugin.register()
