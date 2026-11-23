# SPDX-License-Identifier: MIT
"""Read the real prepared r7a with its own planner; QV owns instrumentation."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

from .common import HERE, ROUNDS, ablit, digest, load, need, pins, read, sha, sibling, sources, write


def qv():
    # QV exports its pure functions with qv_common globals bound at import.
    with sibling('quick-verify-r1', 'qv_plan.py', ('qv_common',)) as module:
        return module


def layout():
    p = ablit()
    paths = read(ROUNDS / 'ablit-r7-r1/f760-source-pins.json')
    def module(suffix):
        found = [k for k in paths if k.endswith(suffix)]
        need(len(found) == 1, 'ambiguous pinned runtime layout')
        return 'vllm.' + found[0][:-3].replace('/', '.')
    return dict(raw=p.RAW, adapted=p.ADAPT, runner=module('/gpu/model_runner.py'),
                rejection=module('/spec_decode/rejection_sampler.py'), lora=module('/gpu/lora_utils.py'),
                sampler=module('/gpu/sample/sampler.py'))


def aliases(argv):
    p, q = ablit(), qv()
    opts, image, cmd = p.parser().split(argv)
    _, _, env, mounts = q.inventory(argv)
    need(image == p.IMAGE and env.get('DSV41_ABLIT_R7') == '1', 'real r7a required')
    need('DSV41_FLY_NAMES' not in env, 'already instrumented')
    need(cmd.count('--served-model-name') == 1, 'explicit served base required')
    cmd = copy.copy(cmd)
    pos = cmd.index('--served-model-name') + 1
    need(cmd[pos] == p.RAW and (pos + 1 == len(cmd) or cmd[pos+1].startswith('-')),
         'unexpected base alias list')
    cmd.insert(pos + 1, p.RAW + '-fast')
    target = p.SITE + '/fly_names'
    need(not any(Path(target) == Path(d) or Path(target) in Path(d).parents or
                 Path(d) in Path(target).parents for d in mounts), 'packet mount overlaps incumbent')
    options = [a for _, _, raw in opts for a in raw]
    import json
    options += ['-e', 'DSV41_FLY_NAMES=' + json.dumps(dict(layout(), control_each_step=True), sort_keys=True),
                '--mount', f'type=bind,source={HERE / "fly_names"},target={target},readonly']
    return ['run', *options, image, *cmd]


def receipt(argv):
    q = qv()
    _, _, _, mounts = q.inventory(argv)
    files = {}
    for dst, row in mounts.items():
        # Same code roots as QV; model/Engram files are authenticated by argv.
        if not dst.startswith((q.SITE + '/', '/opt/dsv41-', '/dsv41-sparse')):
            continue
        need(row['readonly'], 'writable code mount')
        src = Path(row['source']).resolve()
        for path in sorted(src.rglob('*')) if src.is_dir() else [src]:
            if path.is_file() and q.code_data(path):
                files[str(path)] = sha(path)
    need(files, 'no mounted sources')
    return dict(planned_argv_sha256=digest(argv), files=[dict(source=k, sha256=v) for k, v in sorted(files.items())])


def prepare(original, target, hardware_receipt):
    pins()
    target = Path(target).resolve()
    need(target.is_relative_to(HERE / 'out') and not target.exists(), 'new packet/out target required')
    p = ablit()
    original = Path(original).resolve()
    argv = p.verify(original, measurement_allowed=False)
    need(read(original / 'plan.json')['base'] == 'current-r7a', 'prepared daily must be r7a')
    # Rank order and serials come from the capture/serving receipt, never constants.
    hardware_receipt = Path(hardware_receipt).resolve()
    rank_paths = [hardware_receipt / f'rank{r}.json' for r in range(4)]
    serials = hardware_serials([read(p) for p in rank_paths])
    prepared_argv = aliases(argv)
    write(target / 'argv.json', prepared_argv)
    write(target / 'receipt.json', receipt(prepared_argv))
    cfg = dict(schema='quick-verify-candidate/2', argv=str(target/'argv.json'), receipt=str(target/'receipt.json'),
               receipt_sha256=sha(target/'receipt.json'), argv_sha256=digest(prepared_argv),
               adapter=str(HERE/'qv_candidate.py'), adapter_sha256=sha(HERE/'qv_candidate.py'),
               candidate='fly-names-r1', **{'class': 'numeric'}, metric='tok_s', win_cell='ordered:c4',
               path_ids={a:'fly:transport:'+a for a in ('off','on')},
               scheduler_path_ids={a:'fly:admission' for a in ('off','on')},
               temperature=1., profile_share=1., expected_effect='acceptance amortizes the whole target step; prior c4 +25-28%, r7a unmeasured',
               activation_patterns=['FLY_NAMES_INSTALLED'], null_adapter=False,
               grid_cells=[[4,1],[4,2],[4,3],[4,5]], touched_grid_cells=[])
    qv().validate(cfg)
    write(target/'candidate.json', cfg)
    write(target/'prepared.json', dict(original=str(original), baseline_argv_sha256=digest(argv),
          hardware_receipts={str(p): sha(p) for p in rank_paths},
          serials=serials, sources=sources(), schema='fly-names-prepared/1'))
    # Exercise the complete QV composer now, before access to any GPU slot.
    final, paths = compose(target, target/'cpu-plan', 'offline', 'fly-offline', 'offline')
    write(target/'measurement-argv.json', final)
    write(target/'measurement-paths.json', paths)
    return target


def hardware_serials(rows):
    """Schema: bitwise-capture-r1/store.py receipt + hooks.py flush.

    Same policy as profile_r2.rank_receipts, without importing that changing
    packet or writing its pins. Worker-pinning.json has UUIDs, not serials.
    """
    need(len(rows) == 4 and all(type(r['rank']) is int for r in rows)
         and [r['rank'] for r in rows] == list(range(4)), 'rank receipt order')
    need(all(r.get('phase') == 'finalize' and r.get('arm') == 'off' for r in rows), 'finalized OFF receipts required')
    serials = [r['serial'] for r in rows]
    need(all(isinstance(s, str) and s.isdigit() and len(s) > 4 for s in serials)
         and len(set(serials)) == 4, 'invalid card serial receipt')
    return serials


def check(prepared, qualified=True):
    pins()
    prepared = Path(prepared).resolve()
    need(prepared.is_relative_to(HERE/'out') and prepared != HERE/'out', 'prepared path outside packet/out')
    row = read(prepared/'prepared.json')
    need(row['sources'] == sources(), 'packet changed; prepare again')
    need(all(sha(p) == h for p, h in row['hardware_receipts'].items()), 'hardware receipt changed')
    original = ablit().verify(Path(row['original']), measurement_allowed=False)
    need(digest(original) == row['baseline_argv_sha256'], 'baseline changed')
    cfg = read(prepared/'candidate.json')
    qv().validate(cfg)
    if qualified:
        proof = read(prepared/'image-preflight.json')
        need(proof['status'] == 'PASS' and proof['sources'] == sources() and not proof['cuda_initialized'],
             'offline f760 proof missing/stale')
        need(read(HERE/'out/linux-qualified.json') == dict(platform='linux', sources=sources()),
             'Linux tests missing/stale')
    return cfg


def compose(prepared, run, token, name, session):
    cfg = check(prepared, qualified=False)
    q = qv()
    return q.compose(cfg, Path(run), token, name, session, 1120)


def preflight(prepared):
    cfg = check(prepared, qualified=False)
    argv, paths = compose(prepared, Path(prepared)/'cpu-plan', 'offline', 'fly-offline', 'offline')
    q = qv()
    # Preserve exact planner env; compose has deliberately introduced entrypoint/lock.
    env = ablit().parser().environment(ablit().parser().split(argv)[0])
    dry = load(ROUNDS/'lib/dry_boot.py', '_fly_dry_boot')
    paths = {dst: row for dst, row in paths.items() if not dst.startswith('/slot/')}
    command = dry.command(env, paths)
    command[-2:] = ['/fly_image_check.py']
    # No file mounted beneath a read-only directory. QV returns flattened files.
    i = command.index('-i')
    command[i:i] = ['--mount', f'type=bind,source={HERE / "image_check.py"},target=/fly_image_check.py,readonly']
    import json
    payload = dict(plugins=env['VLLM_PLUGINS'].split(','), adapter_sha256=cfg['adapter_sha256'])
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=240)
    (Path(prepared)/'image-preflight.log').write_text(result.stdout + result.stderr)
    need(result.returncode == 0, 'offline f760 dry run failed; see image-preflight.log')
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    need(proof['status'] == 'PASS' and not proof['cuda_initialized'] and not proof['gpu_used'], 'incomplete CPU proof')
    proof['sources'] = sources()
    write(Path(prepared)/'image-preflight.json', proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('original', type=Path); a.add_argument('target', type=Path)
    a.add_argument('hardware_receipt', type=Path)
    for action in ('check', 'preflight', 'serial'):
        a = sub.add_parser(action); a.add_argument('prepared', type=Path)
    a = parser.parse_args()
    if a.action == 'prepare': print(prepare(a.original, a.target, a.hardware_receipt))
    elif a.action == 'preflight': preflight(a.prepared); print('FLY_NAMES_CPU_IMAGE PASS')
    elif a.action == 'serial': print(read(a.prepared/'prepared.json')['serials'][0])
    else: check(a.prepared); print('FLY_NAMES_PLAN PASS')


if __name__ == '__main__': main()
