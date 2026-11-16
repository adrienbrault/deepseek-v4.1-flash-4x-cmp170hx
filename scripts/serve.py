#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the documented serving settings; refuse an incomplete runtime export."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_plan(plan):
    """Keep the exported daily's two additive policies joined to its launcher."""
    if plan['definition'] not in {'current-r7f', 'current-r7g'}:
        raise ValueError('unsupported serving export definition')
    if plan.get('docker_options') != ['--memory', '55834574848', '--memory-swap', '55834574848']:
        raise ValueError('selected definition requires the exact 52 GiB memory pair')
    env = plan['environment']
    if env.get('DSV41_PREFILL_7B_R7D') != '1' or env['VLLM_PLUGINS'].split(',').count('dsv41_prefill_7b') != 1:
        raise ValueError('selected definition requires the bitwise eager prefill serving bootstrap')


def local_inputs():
    values = {}
    file = ROOT / 'scripts/host.env'
    if file.exists():
        for line in file.read_text().splitlines():
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if len(words) != 1 or '=' not in words[0]:
                raise ValueError('host.env accepts only NAME=value assignments')
            key, value = words[0].split('=', 1)
            if key not in {'MODEL_ROOT', 'OFFICIAL_ROOT', 'ADAPTER_ROOT', 'CACHE_ROOT', 'RUNTIME_ASSETS_ROOT'}:
                raise ValueError('unknown host.env input: ' + key)
            values[key] = value
    return {key: os.environ.get(key, values.get(key, '')) for key in
            ('MODEL_ROOT', 'OFFICIAL_ROOT', 'ADAPTER_ROOT', 'CACHE_ROOT', 'RUNTIME_ASSETS_ROOT')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--print', action='store_true', help='print a parameterized plan; default')
    mode.add_argument('--check', action='store_true', help='check local prerequisites without Docker')
    mode.add_argument('--run', action='store_true', help='refused while the export is incomplete')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'config/served.json').read_text())
    validate_plan(plan)
    if not args.check and not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    problems = []
    for key, value in local_inputs().items():
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            problems.append(key + ': supply an existing absolute directory')
    if plan['reproduction_complete'] is not True:
        problems += ['unresolved export input: ' + name for name in plan['required_assets']]
    # No executor exists until the authenticated mount map and lifecycle are exported.
    # Merely changing one JSON flag cannot start an unverified partial recipe.
    if args.run:
        problems.append('serving executor unavailable: authenticated lifecycle export required')
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 2
    print('Local input directories exist; no image, GPU or service check performed.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        sys.exit('serve: ' + str(exc))
