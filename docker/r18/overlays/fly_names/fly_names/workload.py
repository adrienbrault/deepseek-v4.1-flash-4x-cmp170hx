# SPDX-License-Identifier: MIT
"""Packet workload extension over QV's HTTP, control, FIFO and timing tools."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import math
import http.client
import json
from pathlib import Path
import re
import statistics
import time
from urllib.parse import urlsplit

from .common import digest, need, read, write
from .policy import identity


def rows(run):
    paths = sorted((run/'live').glob('fly-rank*.jsonl'))
    # All-rank transport is covered by QV acknowledgements. Sampling may run
    # only on the designated TP sampler rank; never assume rank index/order.
    result = []
    for path in paths:
        with path.open() as stream:
            for line in stream:
                result.append(dict(file=path.name, **__import__('json').loads(line)))
    return result


def control(run, count, label, diagnostic=False):
    path = run/'live/fly-control.json'
    generation = read(path)['generation'] + 1 if path.exists() else 0
    write(path, dict(generation=generation, cell=label, request_group=digest(label)[:16], admission=count, diagnostic=diagnostic, record=True))


def batch(q, models, c, tokens, salt, temperature, quality=False, *, run=None,
          admission='barrier', prompt_indices=None, diagnostic=False):
    need(type(c) is int and 1 <= c <= 8, 'invalid batch size')
    need(len(models) == c, 'model/batch count mismatch')
    indices = list(range(c)) if prompt_indices is None else list(prompt_indices)
    need(len(indices) == c and all(type(i) is int and 0 <= i < len(q.PROMPTS) for i in indices), 'prompt index mapping')
    need(admission in ('burst', 'ordered', 'barrier'), 'unknown admission policy')
    if run is not None:
        control(run, c if admission == 'barrier' else 1, salt, diagnostic)
    before = time.monotonic()
    seed_for_prompt = getattr(q, 'seed_for_prompt', lambda i: 15+i)
    def payload(i):
        body = q.body(q.PROMPTS[indices[i]], tokens, salt + '-' + str(i), temperature, quality)
        body['model'] = models[i]
        body['seed'] = seed_for_prompt(indices[i])
        body['request_id'] = 'flycell-' + digest(salt)[:16] + '-row' + str(i)
        return body
    def receive(response, i):
        need(len(response['choices']) == 1, 'one response per request required')
        q.completion_result(response, 1, tokens, before)
        need(isinstance(response.get('id'), str), 'request ID missing')
        return dict(id=response['id'], choice=response['choices'][0], model=models[i],
                    prompt_index=indices[i], seed=seed_for_prompt(indices[i]), request=payload(i))
    def request(i):
        return receive(q.post('/v1/completions', payload(i), timeout=35), i)
    def response(connection, i):
        try:
            with connection.getresponse() as reply:
                need(reply.status == 200, 'batch HTTP failure')
                return receive(json.load(reply), i)
        finally:
            connection.close()
    with ThreadPoolExecutor(max_workers=c) as pool:
        if admission == 'burst':
            result = list(pool.map(request, range(c)))
        else:
            # QV quality_capture semantics: synchronous writes, parallel reads.
            # A pool of q.post calls does not preserve wire/admission order.
            url = urlsplit(q.BASE)
            need(url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost'), 'loopback HTTP required')
            futures, sent = [], None
            for i in range(c):
                if sent is not None:
                    time.sleep(max(0, sent + .008 - time.monotonic()))
                connection = http.client.HTTPConnection(url.hostname, url.port, timeout=35)
                try:
                    connection.request('POST', '/v1/completions', json.dumps(payload(i)), {'Content-Type': 'application/json'})
                    sent = time.monotonic()
                except BaseException:
                    connection.close()
                    raise
                futures.append(pool.submit(response, connection, i))
            result = [f.result() for f in futures]
    ended=time.monotonic()
    return dict(rows=result, cell=salt, admission=admission, t0=before,t1=ended,wall_s=ended-before, tokens=c*tokens)


def diagnostic(run, q, switch, raw):
    """First scored cell, entirely OFF: historical burst vs QV vs barrier."""
    from .diagnostic import classify
    switch('off')  # QV's discarded warmup bootstraps the scheduler AddRequest hook.
    captures = {}
    started = time.monotonic()
    for mode in ('burst', 'burst-k3', 'ordered', 'barrier'):
        q.force(run, None if mode == 'burst' else 3)
        for lp in (False, True):
            label = mode + ('-lp' if lp else '-tokens')
            captures[label] = []
            for repeat in range(3):
                need(time.monotonic()-started < 180, 'diagnostic 180-second allowance exhausted')
                g = batch(q, [raw]*4, 4, 16, f'diagnostic-{label}-{repeat}', 0., lp,
                          run=run, admission='burst' if mode == 'burst-k3' else mode, diagnostic=True)
                captures[label].append(g)
                write(run/'diagnostic-captures.json', captures)
    q.force(run, 3)
    for i in range(4):
        for lp in (False, True):
            label = f'c1-{i}-' + ('lp' if lp else 'tokens')
            captures[label] = []
            for repeat in range(2):
                need(time.monotonic()-started < 180, 'diagnostic 180-second allowance exhausted')
                g = batch(q, [raw], 1, 16, f'diagnostic-{label}-{repeat}', 0., lp,
                          run=run, prompt_indices=[i], diagnostic=True)
                captures[label].append(g)
                write(run/'diagnostic-captures.json', captures)
    q.flush(run)
    need(time.monotonic()-started <= 180, 'diagnostic 180-second allowance exhausted')
    observations = []
    for path in sorted((run/'live').glob('fly-diagnostic-rank*.jsonl')):
        observations.extend(json.loads(line) for line in path.read_text().splitlines())
    for groups in captures.values():
        for group in groups:
            for row in group['rows']:
                matched = {r['request_id'] for r in observations if r['cell'] == group['cell'] and same_id(r['request_id'], row['id'])}
                need(len(matched) == 1, 'diagnostic HTTP/internal ID mapping missing/ambiguous')
                row['internal_id'] = matched.pop()
    write(run/'diagnostic-captures.json', captures)
    report = classify(captures, observations, raw)
    report['elapsed_s'] = time.monotonic()-started
    # Solo-vs-batch equality is evidence of shape dependence, never an identity reference.
    report['solo_matches_barrier_tokens'] = [captures[f'c1-{i}-tokens'][0]['rows'][0]['choice']['token_ids'] ==
        captures['barrier-tokens'][0]['rows'][i]['choice']['token_ids'] for i in range(4)]
    write(run/'diagnostic.json', report)
    print('FLY_NAMES_DIAGNOSTIC', report['status'], report['findings'], report['faults'], flush=True)
    if report['serving_bug_suspected']:
        print('FLY_NAMES_SERVING_BUG_SUSPECT instrumented incumbent OFF; see diagnostic.json; serving unchanged', flush=True)
    need(report['status'] == 'PASS', 'OFF diagnostic blocked identity; see diagnostic.json (serving unchanged)')
    return report


def same_id(internal, api):
    # f760 InputProcessor suffix documented in ablit-r7-r1/hooks.py; permit
    # completion's prompt index as well. Anchor it, never substring-match IDs.
    return re.fullmatch(re.escape(api) + r'(?:-\d+)?(?:-[0-9a-f]{8})?', internal) is not None


def witnesses(run, group, require_mixed=False):
    all_rows = rows(run)
    expected = {r['id']: r['model'] for r in group['rows']}
    covered, mixed = set(), False
    for call in all_rows:
        matched = [r for r in call['rows'] if any(same_id(r['request_id'], api) for api in expected)]
        for row in matched:
            api = next(api for api in expected if same_id(row['request_id'], api))
            need(row['name'] == expected[api], 'executed served name differs from HTTP name')
            need(row['fly'] == (row['enabled'] and expected[api].endswith('-fast')), 'FLy use witness differs from name')
            need(row['selected_count'] == (row['fly_count'] if row['fly'] else row['standard_count']),
                 'row selected wrong verifier')
            if row['draft_width'] > 0:
                covered.add(api)
        # Require the REGISTERED batch size, with both policies in one native
        # invocation, rather than mistaking overlapped HTTP for a mixed batch.
        if len(matched) == len(expected) and any(r['fly'] for r in matched) and any(not r['fly'] for r in matched):
            mixed = True
    need(covered == set(expected), 'request lacks speculative row acceptance counters')
    need(not require_mixed or mixed, 'mixed verifier batch not observed at required c')
    by_file = {}
    for call in all_rows:
        for row in call['rows']:
            if row['draft_width'] > 0 and any(same_id(row['request_id'], api) for api in expected):
                by_file.setdefault(call['file'], []).append(row)
    need(by_file, 'no acceptance evidence')
    # TP ranks can replicate sampling. Use one observed rank for aggregation,
    # reporting its receipt rather than multiplying counts by TP size.
    chosen = sorted(by_file)[0]
    acceptance = {}
    for row in by_file[chosen]:
        total = acceptance.setdefault(row['name'], dict(steps=0,drafts=0,accepted_drafts=0,rescued_drafts=0,fly_steps=0))
        total['steps'] += 1; total['drafts'] += row['draft_width']
        total['accepted_drafts'] += max(0,row['selected_count']-1)
        total['rescued_drafts'] += row['rescued_drafts']; total['fly_steps'] += int(row['fly'])
    return dict(requests=len(covered), mixed=mixed, sampler_receipt=chosen, acceptance_by_name=acceptance)


def hard_gate(run, q, switch, raw, adapted):
    evidence = []
    for c in (4, 8):
        cases=[('raw',[raw]*c),('adapted',[adapted]*c),('four-names',[raw,adapted]*(c//2))]
        for tag,normal_names in cases:
            for temperature in (0., 1.):
                label = f'gate-{c}-{tag}-{temperature}'
                switch('off')
                q.force(run, 3)
                warm = batch(q, normal_names, c, 16, label+'-warm-off', temperature, True, run=run)
                write(run/(label+'-warm-off.json'), warm)
                a = batch(q, normal_names, c, 16, label+'-a', temperature, True, run=run)
                b = batch(q, normal_names, c, 16, label+'-b', temperature, True, run=run)
                write(run/(label+'-baseline.json'), dict(a=a,b=b))
                deterministic = all(identity(x['choice']) == identity(y['choice']) for x,y in zip(a['rows'], b['rows']))
                if not deterministic and temperature == 1.:
                    switch('on')
                    q.force(run, 3)
                    models=normal_names[:c//2]+[n+'-fast' for n in normal_names[c//2:]]
                    mixed=batch(q,models,c,16,label+'-nondeterministic-mix',temperature,True,run=run)
                    write(run/(label+'-candidate.json'),dict(mixed=mixed))
                    witness=witnesses(run,mixed,require_mixed=True)
                    evidence.append(dict(cell=label, status='NONDETERMINISTIC_T1', identity_claim=False,witness=witness))
                    write(run/'identity-progress.json', evidence)
                    continue
                need(deterministic, 'T0 baseline not bit-identical: '+label)
                switch('on')
                q.force(run, 3)
                warm = batch(q, normal_names, c, 16, label+'-warm-on', temperature, True, run=run)
                write(run/(label+'-warm-on.json'), warm)
                normal = batch(q, normal_names, c, 16, label+'-on', temperature, True, run=run)
                write(run/(label+'-candidate.json'), dict(normal=normal))
                need(all(identity(x['choice']) == identity(y['choice']) for x,y in zip(a['rows'], normal['rows'])),
                     'existing-name ON vs current-r7a OFF identity mismatch: '+label)
                model_mix = normal_names[:c//2]+[n+'-fast' for n in normal_names[c//2:]]
                mixed = batch(q, model_mix, c, 16, label+'-mix', temperature, True, run=run)
                after = batch(q, normal_names, c, 16, label+'-after', temperature, True, run=run)
                write(run/(label+'-candidate.json'), dict(normal=normal,mixed=mixed,after=after))
                need(all(identity(a['rows'][i]['choice']) == identity(mixed['rows'][i]['choice']) for i in range(c//2)),
                     'existing-name identity changed beside fast rows: '+label)
                need(all(identity(x['choice']) == identity(y['choice']) for x,y in zip(a['rows'], after['rows'])),
                     'existing-name identity changed after fast rows: '+label)
                witness = witnesses(run, mixed, require_mixed=True)
                evidence.append(dict(cell=label,status='PASS',witness=witness))
                write(run/'identity-progress.json', evidence)
    need(any(r['status'] == 'PASS' for r in evidence), 'empty identity gate')
    result = dict(status='PASS', cells=evidence, bytes='API decoded IEEE754 binary64, signed zero retained',
                  baseline='authenticated current-r7a native standard OFF in same boot; barrier, forced K3, discarded per-switch warmup')
    write(run/'identity.json', result)
    return result


def speed(run, q, switch, raw, adapted):
    # AB / BA: alternate which arm runs first. No observed-result retries.
    report = []
    for twin in (raw, adapted):
        for c in (4,8,1):
            blocks = []
            for i, arm in enumerate(('normal','fast','fast','normal')):
                switch('on')
                model = twin + ('-fast' if arm == 'fast' else '')
                group = batch(q, [model]*c, c, 64, f'speed-{twin}-{c}-{i}', 1., run=run)
                witness = witnesses(run, group)
                q.flush(run)
                fifo=[__import__('json').loads(line) for line in (run/'live/steps.jsonl').read_text().splitlines()]
                cadence=q.step_window(fifo,group['t0'],group['t1'],c,None,q.protocol())
                blocks.append(dict(arm=arm, tok_s=group['tokens']/group['wall_s'], witness=witness,cadence=cadence, **group))
                write(run/f'speed-{twin}-{c}.json', blocks)
            gains = [blocks[1]['tok_s']/blocks[0]['tok_s']-1, blocks[2]['tok_s']/blocks[3]['tok_s']-1]
            report.append(dict(name=twin,c=c,pair_gains_pct=[100*x for x in gains],
                               verdict='WIN' if all(x >= .01 for x in gains) else 'NO_WIN', blocks=blocks))
    write(run/'speed.json', report)
    return report


def positional_kl(a, b):
    need(len(a) == len(b) and a, 'quality population missing')
    values = []
    for prompt in range(4):
        for pos in range(16):
            left = Counter(r['choice']['token_ids'][pos] for group in a for r in [group['rows'][prompt]])
            right = Counter(r['choice']['token_ids'][pos] for group in b for r in [group['rows'][prompt]])
            keys = left.keys() | right.keys(); z = len(a) + .5*(len(keys)+1)
            values.append(sum((right[t]+.5)/z*math.log((right[t]+.5)/(left[t]+.5)) for t in keys))
    return statistics.fmean(values)


def quality(run, q, switch, raw, adapted):
    result = []
    for twin in (raw, adapted):
        captures = {}
        for label, suffix in (('standard-a',''),('fly','-fast'),('standard-b','')):
            switch('on')
            captures[label] = []
            for seed in (15,16,17,18):
                # Own seed offset also supplies an independent standard null.
                offset = seed + (1000 if label == 'standard-b' else 0)
                # batch uses frozen per-row seeds; use seeded() to vary them
                # without changing temperature/top_p/default sampling settings.
                wrapper = Seeded(q, offset)
                captures[label].append(batch(wrapper,[twin+suffix]*4,4,16,f'quality-{twin}-{label}-{seed}',1.,True,run=run))
            write(run/f'quality-{twin}-{label}.json', captures[label])
        tasks = []
        control(run, 1, 'arithmetic-'+twin)
        for suffix in ('','-fast'):
            for i,(x,y) in enumerate(((17,25),(83,19),(43,36),(125,87),(91,73),(318,429))):
                body = dict(model=twin+suffix,messages=[dict(role='user',content=f'What is {x} + {y}? Return only the integer.')],
                            temperature=1.,seed=15,max_tokens=96,cache_salt=f'task-{twin}-{suffix}-{i}')
                response = q.post('/v1/chat/completions',body)
                choice = response['choices'][0]; msg = choice.get('message') or {}
                answer = msg.get('content') or ''
                tasks.append(dict(name=twin+suffix,index=i,correct=answer.strip()==str(x+y),
                                  content=answer,finish_reason=choice.get('finish_reason')))
                write(run/f'quality-{twin}-tasks.json',tasks)
        result.append(dict(name=twin,positional_kl=positional_kl(captures['standard-a'],captures['fly']),
                           independent_null_kl=positional_kl(captures['standard-a'],captures['standard-b']),tasks=tasks))
    write(run/'quality.json',dict(status='REPORT',results=result,
          limits='Four prompts x four seeds, 16 positions; smoothed positional marginal KL, not output-law KL. '
                 'Six arithmetic items per served name; no broad quality or promotion claim.'))


class Seeded:
    def __init__(self,q,seed): self.q,self.seed=q,seed
    def __getattr__(self,key): return getattr(self.q,key)
    def seed_for_prompt(self,index): return self.seed
    def post(self,path,body,**kw):
        return self.q.post(path,dict(body,seed=self.seed),**kw)
