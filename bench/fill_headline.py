#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
# SPDX-License-Identifier: MIT
"""Export r7f/r7g headline-report/2 into sanitized results, figures and card data."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RAW = 'deepseek-v4.1-flash-1017'
CS = (1, 4, 8)
CONTEXTS = ((1024, 2048), (32768, 2048), (65536, 8192))
CATEGORIES = ('multi_turn', 'translation', 'summarization', 'qa', 'math_reasoning', 'rag')
METRICS = ('output_tok_s', 'per_stream_tok_s', 'ttft_p50_ms', 'ttft_p99_ms', 'acceptance_length')
TASKS = '19546008de5db5c50ebf28542ac5da4e240f7d9da7d01af7d08508711dc1698a'
CORPUS = '354df35349a56e3ea59d2d7744250c23e9e5926a6c34e4401fef9031fc0dfdc8'
BEGIN, END = '[headline-begin]: #', '[headline-end]: #'
COLD_DIR = 'bench/results/2026-11-16-current-r7e-suite'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, label, nullable=False):
    if nullable and value is None:
        return None
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, 'missing/invalid metric: ' + label)
    return value


def digest(value):
    need(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value), 'invalid SHA256')
    return value


def spec_key(size, c):
    return f'specbench-{size}-c{c}'


def code_key(p, a, c, mode='shared'):
    return f'code-{mode}-t{p}-a{a}-c{c}'


def clean_cell(row, key):
    status = row.get('status')
    need(status == 'COMPLETE', 'partial cell refused: ' + key)
    fields = METRICS + ('e2e_p50_s',) if key.startswith('specbench') else (
        'aggregate_tok_s', 'per_stream_tok_s', 'ttft_p50_ms', 'ttft_p99_ms', 'step_ms', 'accepted_per_step')
    result = dict(status='COMPLETE')
    for name in fields:
        result[name] = number(row.get(name), key + '/' + name, nullable=name in ('step_ms', 'accepted_per_step'))
    need(result['ttft_p99_ms'] >= result['ttft_p50_ms'], 'inverted TTFT percentiles: ' + key)
    if key.startswith('code'):
        c = int(key.rsplit('-c', 1)[1])
        need(row.get('cache_confirmed') is True and row.get('requests') == c and row.get('batches') == 1,
             'code request count/batch/cache proof differs: ' + key)
        result.update(cache_confirmed=True, requests=c, batches=1)
    else:
        c = int(key.rsplit('-c', 1)[1]); size = key.split('-')[1]
        n = 120 if size == 'subset' else 480
        need(row.get('requests') == n and row.get('output_tokens') == n * 256 and row.get('concurrency') == c
             and row.get('size') == size, 'Spec-Bench geometry differs: ' + key)
        need(row.get('acceptance_source') in ('vLLM client acceptance_length', '1 + accepted draft tokens / drafts'),
             'unknown acceptance source')
        result.update(requests=n, output_tokens=n * 256, acceptance_source=row['acceptance_source'])
    return result


def normalize(report, report_sha, calibration=None):
    need(report.get('schema') == 'headline-report/2', 'expected headline-report/2 from headline-pack-r1/report.py')
    need(report.get('daily') in ('current-r7f', 'current-r7g'), 'unsupported daily')
    need(report.get('measured_names') == [RAW], 'base-name measurements required')
    need(report.get('serving_after') is False and report.get('promotion_eligible') is False, 'report scope differs')
    need(report.get('status') == 'MEASURED', 'INCOMPLETE report refused')
    need(report.get('mode') in ('first', 'routine'), 'subset report required; use first calibration or routine plus --calibration-report')
    need(type(report.get('separate_contexts')) is bool, 'context mode missing')
    expected = {spec_key(s, c) for s in (('subset', 'full') if report['mode'] == 'first' else ('subset',)) for c in CS}
    expected |= {code_key(p, a, c, mode) for mode in (('shared', 'separate') if report['separate_contexts'] else ('shared',))
                 for p, a in CONTEXTS for c in CS}
    need(set(report['cells']) == expected, 'cell geometry differs from registered report')
    cells = {k: clean_cell(report['cells'][k], k) for k in sorted(expected)}
    subset = report['subset_manifest']
    need(subset['schema'] == 'headline-subset/2' and subset['count'] == 120 and subset['seed'] == 7310
         and set(subset['categories']) == set(CATEGORIES), 'subset manifest differs')
    subset_clean = dict(count=120, seed=7310, per_category=20, categories=list(CATEGORIES),
                        id_list_sha256=digest(subset['id_list_sha256']), full_rows_sha256=digest(subset['full_rows_sha256']))
    need(report['tasks_sha256'] == TASKS and report['corpus_sha256'] == CORPUS, 'frozen code input changed')
    need(report.get('starts'), 'boot starts missing')
    starts = [datetime.strptime(v, '%Y-%m-%d %H:%M:%S Paris').replace(tzinfo=ZoneInfo('Europe/Paris')).astimezone(timezone.utc)
              for v in report['starts']]
    need(len(starts) == report['boots'] and 0 < report['boots'] == report['planned_boots'], 'boot accounting differs')
    need(report['planned_boots'] == (3 if report['mode'] == 'first' else 1), 'boot allocation differs')
    ratios = {}
    if report['mode'] == 'first':
        need(set(report['ratios']) == {'1', '4', '8'}, 'calibration ratios missing')
        for c in CS:
            a, b = (cells[spec_key(size, c)] for size in ('subset', 'full'))
            raw_ratio = report['ratios'][str(c)]
            need(raw_ratio['status'] == 'COMPLETE', 'calibration incomplete')
            metrics = {}
            for k in METRICS:
                value = a[k] / b[k]; observed = raw_ratio['metrics'][k]
                need(math.isclose(number(observed['subset_over_full'], 'ratio'), value, rel_tol=1e-9)
                     and observed['flag'] == (abs(value - 1) > .02), 'calibration ratio inconsistent')
                metrics[k] = dict(subset_over_full=value, flag=abs(value - 1) > .02)
            ratios[str(c)] = dict(status='COMPLETE', metrics=metrics)
    else:
        need(calibration and calibration['configuration'] == report['daily'] and calibration['subset'] == subset_clean
             and calibration['publication_ready'] and calibration['mode'] == 'first', 'matching first-run calibration required')
        ratios = calibration['ratios']
    distributions = {}
    need(set(report['prompt_distributions']['categories']) == set(CATEGORIES), 'prompt strata differ')
    for cat in CATEGORIES:
        distributions[cat] = {}
        for size, count in (('full', 80), ('subset', 20)):
            raw = report['prompt_distributions']['categories'][cat][size]
            need(raw['count'] == count, 'stratum size differs')
            distributions[cat][size] = dict(count=count, **{k: number(raw[k], 'distribution/' + k)
                for k in ('min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max', 'mean', 'total')})
    ready = (report['status'] == 'MEASURED' and report['boots'] == report['planned_boots']
             and all(r['status'] == 'COMPLETE' for r in cells.values())
             and all(r['status'] == 'COMPLETE' for r in ratios.values()))
    return dict(schema='headline-summary/1', publication_ready=ready, date_utc=min(starts).date().isoformat(),
        configuration=report['daily'], source='headline-pack-r1', report_sha256=report_sha, source_key=digest(report['source_key']),
        starts_utc=[v.strftime('%Y-%m-%dT%H:%M:%SZ') for v in starts], boots=report['boots'], mode=report['mode'],
        measured_name=RAW, excluded_names=[RAW+'-abliterated', RAW+'-fast', RAW+'-abliterated-fast'],
        subset=subset_clean, prompt_distributions=distributions, ratios=ratios, cells=cells,
        tasks_sha256=TASKS, corpus_sha256=CORPUS,
        calibration_report_sha256=calibration['report_sha256'] if calibration else report_sha,
        calibration_date_utc=calibration['date_utc'] if calibration else min(starts).date().isoformat())


def fmt(row, name, digits=1):
    v = row.get(name)
    return f'{v:,.{digits}f}' if v is not None else ('n/a' if row.get('status') == 'COMPLETE' else 'TODO · '+row.get('status', 'PENDING'))


def cold_prefill(root=ROOT):
    """The dated r7e reference is independent of the headline report's TTFT."""
    r = read(root/COLD_DIR/'summary.json')
    need(r['date_utc'] == '2026-11-16' and r['configuration'] == 'current-r7e', 'cold prefill reference differs')
    pre = r['prefill']
    need(pre['output_tokens_per_request'] == 1 and pre['temperature'] == 0 and pre['thinking'] is False,
         'cold prefill protocol differs')
    need(pre['cache'] == 'zero prompt-cache hits; host cache warm-steady', 'cold prefill cache protocol differs')
    rows = pre['cells']
    need(len(rows) == 6 and {(v['prompt_tokens'], v['concurrency']) for v in rows} ==
         {(p, c) for p in (2048, 8192) for c in CS}, 'cold prefill geometry differs')
    cells = [dict(prompt_tokens=v['prompt_tokens'], concurrency=v['concurrency'],
                  input_tok_s=number(v['input_tok_s'], 'cold prefill')) for v in rows]
    return dict(date_utc=r['date_utc'], configuration=r['configuration'], results=COLD_DIR,
        summary_sha256=sha(root/COLD_DIR/'summary.json'), output_tokens_per_request=1, temperature=0,
        thinking=False, cache=pre['cache'], cells=cells, report_status=r['report_status'])


def peak_cell(cells, c):
    candidates = [(code_key(p, a, c), cells.get(code_key(p, a, c), {})) for p, a in CONTEXTS]
    return max(candidates, key=lambda kv: kv[1].get('aggregate_tok_s', 0) or 0)


def tables(s):
    cells = s['cells']; date = s['date_utc']; daily = s['configuration']
    path = f'bench/results/{date}-{daily}-headline'
    real = [cells[spec_key('subset', c)] for c in CS]
    peak1_key, peak1 = peak_cell(cells, 1)
    peak8_key, peak8 = peak_cell(cells, 8)
    cold = s['cold_prefill']; cold_peak = max(cold['cells'], key=lambda r: r['input_tok_s'])
    long8 = cells[code_key(65536, 8192, 8)]
    lines = [BEGIN, '',
        f'Measured {date} UTC on **{daily}**, TP4, four CMP 170HX cards at 175 W per card, EAQ 2.9 bpw routed experts and four PCIe Gen2 x16 links. [Results and protocol]({path}/NOTES.md), [numeric records]({path}/summary.json). Only `{RAW}` is timed; the abliterated and opt-in fast names are excluded.', '',
        '![Spec-Bench real-work throughput and first-token latency](docs/img/headline-agent.svg)', '',
        f'- **Real work:** Spec-Bench frozen 120-prompt subset, 256 forced output tokens, T=0 and all arrivals at once subject to c1/c4/c8. At eight agents: output {fmt(real[2],"output_tok_s")} tok/s over benchmark wall, {fmt(real[2],"per_stream_tok_s")} tok/s per stream and TTFT p50/p99 {fmt(real[2],"ttft_p50_ms")} / {fmt(real[2],"ttft_p99_ms")} ms.',
        f'- **Code peaks:** highest measured c1 stream {fmt(peak1,"per_stream_tok_s")} tok/s (`{peak1_key}`); highest c8 sum {fmt(peak8,"aggregate_tok_s")} tok/s (`{peak8_key}`). Cached code, 400 forced tokens, T=0 and seed 15; measured common decode window with no interleaved prefill.',
        f'- **Parallel agents at depth:** 64K cached code plus an 8K append, eight agents, 400 forced tokens, T=0 and seed 15: {fmt(long8,"per_stream_tok_s")} tok/s each, {fmt(long8,"aggregate_tok_s")} tok/s sum and {long8["ttft_p50_ms"]/1000:.3f} s first token p50. The four-agent row and the 32K+2K companions are below.',
        f'- **Dated cold prefill:** {cold["date_utc"]} UTC, r7e suite v2, nominal 2K/8K prompts at c1/c4/c8, one output token, T=0, thinking off, zero prompt-cache hits and warm-steady host page cache. Peak input rate {cold_peak["input_tok_s"]:,.0f} tok/s at {cold_peak["prompt_tokens"]:,} tokens and c{cold_peak["concurrency"]}; r7f uses the same kernels. [Fixed-work reference](#loaded-suite-v2-2026-11-16-utc).', '',
        '### Real agent work', '',
        'Output rate divides completion tokens by client wall time; per-stream rate is 1000 / TPOT p50. TTFT includes HTTP, scheduling and first decode. Spec-Bench TTFT is not a cold-prefill speed. DSpark acceptance length follows the source recorded per cell.', '',
        '| streams | output tok/s, wall | per-stream tok/s | TTFT p50 / p99, ms | DSpark acceptance length |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for c, r in zip(CS, real):
        lines.append(f'| {c} | {fmt(r,"output_tok_s")} | {fmt(r,"per_stream_tok_s")} | {fmt(r,"ttft_p50_ms")} / {fmt(r,"ttft_p99_ms")} | {fmt(r,"acceptance_length",2)} |')
    if s['mode'] == 'first':
        lines += ['', 'The same calibration also measures all 480 prompts with the same sampler and output length:', '',
            '| streams | full-480 output tok/s | full-480 per-stream tok/s | TTFT p50 / p99, ms | acceptance length |',
            '| ---: | ---: | ---: | ---: | ---: |']
        for c in CS:
            r = cells[spec_key('full', c)]
            lines.append(f'| {c} | {fmt(r,"output_tok_s")} | {fmt(r,"per_stream_tok_s")} | {fmt(r,"ttft_p50_ms")} / {fmt(r,"ttft_p99_ms")} | {fmt(r,"acceptance_length",2)} |')
    lines += ['', '### Cached code matrix', '',
        'Frozen code tasks reuse a shared cached 1K+2K, 32K+2K or 64K+8K context and append, with 400 forced output tokens, T=0, seed 15 and EOS suppression. Every request confirms prefix residency. The aggregate sums measured stream rates inside the common decode interval; per-stream is their arithmetic mean. Append TTFT describes a separate phase of the same request.', '',
        '![Cached-code peaks, parallel agents and append TTFT at depth](docs/img/headline-code.svg)', '',
        '| cached context + append | streams | mean per-stream tok/s | decode sum tok/s | append TTFT p50 / p99, ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for p, a in CONTEXTS:
        for c in CS:
            r = cells[code_key(p, a, c)]
            lines.append(f'| {p//1024}K + {a//1024}K | {c} | {fmt(r,"per_stream_tok_s")} | {fmt(r,"aggregate_tok_s")} | {fmt(r,"ttft_p50_ms")} / {fmt(r,"ttft_p99_ms")} |')
    lines += ['', 'Code cells contain one batch per point; p99 over 1, 4 or 8 requests is descriptive. Forced outputs can continue beyond a finished function; no code correctness score is implied.', '',
        '### Subset calibration', '',
        f'The frozen subset contains 20 prompts in each of six strata. Calibration date: {s["calibration_date_utc"]} UTC, {daily}. Each 120/480 pair shares a boot; the c1 pair and c4/c8 pairs use separate boots. Routine measurements use one boot and retain the matching calibration. Ratios are subset / full; flags outside 0.98–1.02 are descriptive.', '',
        '| streams | output ratio | per-stream ratio | TTFT p50 ratio | TTFT p99 ratio | acceptance ratio |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in CS:
        rr = s['ratios'][str(c)]
        vals = [f'{rr["metrics"][k]["subset_over_full"]:.4f}'+(' FLAG' if rr['metrics'][k]['flag'] else '') for k in METRICS]
        lines.append(f'| {c} | '+' | '.join(vals)+' |')
    return '\n'.join(lines + ['', END])


def move_fixed_reference(markdown):
    """Preserve the pushed suite tables once, outside the replaceable markers."""
    start, end = markdown.index(BEGIN), markdown.index(END)
    block = markdown[start:end]
    title = '### Loaded suite v2, 2026-11-16 UTC'
    if title not in block:
        return markdown
    reference = block[block.index(title):].strip().replace(title, '## Loaded suite v2, 2026-11-16 UTC', 1)
    markdown = markdown[:start] + BEGIN + '\n\n' + markdown[end:]
    need('## Loaded suite v2, 2026-11-16 UTC' not in markdown, 'duplicate fixed-work reference')
    anchor = '\n## Reproducing\n'
    need(anchor in markdown, 'README fixed-reference destination missing')
    return markdown.replace(anchor, '\n' + reference + '\n' + anchor, 1)


def card_data(base, s, result_dir):
    card = deepcopy(base); ready = s['publication_ready']; cells = s['cells']
    agent = [cells.get(spec_key('subset', c), {}) for c in CS]
    peak1_key, peak1 = peak_cell(cells, 1); peak8_key, peak8 = peak_cell(cells, 8)
    short_code = [cells.get(code_key(1024, 2048, c), {}) for c in CS]
    depth = [cells.get(code_key(65536, 8192, c), {}) for c in (4, 8)]
    cold = s['cold_prefill']; cold_peak = max(cold['cells'], key=lambda r: r['input_tok_s'])
    value = lambda r, k: round(r[k], 1) if ready else None
    display = lambda r, k, digits=0: fmt(r, k, digits) if ready else 'Pending'
    latency = lambda r: f'{r["ttft_p50_ms"]/1000:.1f} s' if ready else 'Pending'
    peak_text = f'{cold_peak["input_tok_s"]:,.0f}'
    card.update(source=s['date_utc'], method='T=0 · Spec-Bench 120×256 · cached code 400 tokens · dated cold prefill',
        publication_status='READY' if ready else 'PENDING_HEADLINE_CALIBRATION',
        headline=dict(results=result_dir, configuration=s['configuration'], cells=cells,
                      ratios=s['ratios'], subset=s['subset'], cold_prefill=cold,
                      report_sha256=s.get('report_sha256', ''), source_key=s.get('source_key', '')))
    card['one'] = dict(agents=1, precision=1, matchChart=True, metric='throughput · tok/s',
        kinds=[['Spec-Bench', value(agent[0], 'output_tok_s'), 'wall output'],
               ['Code 1K cached', value(short_code[0], 'aggregate_tok_s'), 'decode sum · 2K append']],
        rate=value(agent[0], 'output_tok_s'), ttft=display(agent[0], 'ttft_p50_ms')+' ms',
        kindsNote='Spec-Bench 256 tokens · cached code 400 tokens')
    card['many'] = dict(agents=8, precision=1, sum=value(agent[2], 'output_tok_s'),
        each=value(agent[2], 'per_stream_tok_s'), ttft=display(agent[2], 'ttft_p50_ms')+' ms',
        metric='Spec-Bench · wall output', eachLabel='tok/s per stream',
        style='lines', seriesLabel='Spec-Bench · wall output')
    card['conc'] = dict(x=list(CS), sum=[value(r, 'output_tok_s') for r in agent],
        each=[value(r, 'per_stream_tok_s') for r in agent],
        peak=dict(label='Code 1K cached · decode sum',
                  sum=[value(r, 'aggregate_tok_s') for r in short_code],
                  each=[value(r, 'per_stream_tok_s') for r in short_code]))
    card['depth'] = dict(src='2026-11-16 · suite v2 · cold c1', prefillLabel='cold prefill, tok/s',
        prefill=[[r['prompt_tokens'], r['input_tok_s']] for r in cold['cells'] if r['concurrency'] == 1], decode=[],
        decodeNote=[display(depth[1], 'per_stream_tok_s')+' tok/s each', '64K+8K · 8 agents · 400 tokens'])
    card['setup'] = [[f'64K+8K · {c} agents',
        f'{display(r,"per_stream_tok_s")} each · {display(r,"aggregate_tok_s")} sum · {latency(r)} first token']
        for c, r in zip((4, 8), depth)]
    def peak_label(key, agents):
        prefix, append = next((p, a) for p, a in CONTEXTS if code_key(p, a, agents) == key)
        count = '1 agent' if agents == 1 else f'{agents} agents'
        return f'{count} · {prefix//1024}K + {append//1024}K'
    card['evals'] = dict(kind='peaks', note='tok/s · decode sum', items=[
        [display(peak8, 'aggregate_tok_s', 1), peak_label(peak8_key, 8)],
        [display(peak1, 'aggregate_tok_s', 1), peak_label(peak1_key, 1)],
        [peak_text, 'cold prefill tok/s · c8', cold['date_utc']]])
    card['facts'] = [[display(agent[2], 'output_tok_s', 1), 'Spec-Bench wall tok/s · 8 agents'],
        [display(peak1, 'aggregate_tok_s', 1), 'peak code tok/s · '+peak_label(peak1_key, 1)],
        [display(peak8, 'aggregate_tok_s', 1), 'peak code tok/s · '+peak_label(peak8_key, 8)],
        [peak_text, 'peak cold prefill tok/s · c8']]
    card['kpis'] = [card['facts'][0], [display(agent[0], 'output_tok_s', 1), 'Spec-Bench wall tok/s · 1 agent'],
        [display(agent[2], 'ttft_p50_ms')+' ms', 'first token p50 · 8 agents'], *card['facts'][1:]]
    card['provenance'] = dict(sourceRun='headline-pack-r1; '+s['configuration'],
        recipeResults='https://'+card['repo']+'/tree/main/'+result_dir, inputsSha256=s['subset'].get('full_rows_sha256', ''),
        cachedCode=result_dir+'/summary.json',
        qualityBasis='Performance only; no accuracy score. Base name only.',
        protocols='Spec-Bench 120 stratified prompts, 256 forced tokens, T=0; per stream = 1000 / TPOT p50; wall = completion tokens / benchmark duration. Code: shared cached 1K+2K / 32K+2K / 64K+8K, 400 forced tokens, T=0, seed15; common-window per-stream mean and measured sum.',
        prefill=f'{COLD_DIR}/summary.json; 2026-11-16 UTC; current-r7e suite v2, r7f same kernels; nominal 2K/8K at c1/c4/c8; one output token, T=0, thinking off, zero prompt-cache hits, host cache warm-steady. Chart c1; peak c8 at 8K. Chain same-hour-window check failed; retained per-boot summaries.',
        hardware='config/served.json; four CMP 170HX, TP4, EAQ 2.9 bpw; 175 W/card configured caps, draw unmeasured',
        logo=base.get('provenance', {}).get('logo', 'Lobehub DeepSeek logo, MIT'))
    update = dict(date=s['date_utc'], what='Parallel agents at depth, code peaks and dated cold prefill',
        headline=f'64K+8K c8: {display(depth[1],"per_stream_tok_s")} each, {display(depth[1],"aggregate_tok_s")} sum, {latency(depth[1])} first token',
        deltas=[['Cached code · 64K+8K · c4 / c8', '—',
                 ' / '.join(display(r,'aggregate_tok_s',1) for r in depth), 'tok/s sum'],
                ['Code peaks · c1 / c8', '—', display(peak1,'per_stream_tok_s',1)+' / '+display(peak8,'aggregate_tok_s',1), 'per stream / sum'],
                ['Cold prefill · 8K · c8 · 2026-11-16', '—', peak_text+' tok/s', 'suite v2'],
                ['Spec-Bench 120 · c8', '—', display(agent[2],'output_tok_s',1)+' / '+display(agent[2],'per_stream_tok_s',1), 'wall / stream tok/s']],
        history=None)
    return card, update


def notes(s):
    return (f'# Headline measurements, {s["date_utc"]} UTC\n\n'
        f'Configuration: {s["configuration"]}. Source: headline-pack-r1, headline-report/2, SHA256 `{s["report_sha256"]}`. '
        f'Publication ready: {str(s["publication_ready"]).lower()}. Boot starts (UTC): '+', '.join(s['starts_utc'])+'.\n\n'
        'Spec-Bench uses 120 stratified prompts (20 in each of six categories), 256 forced tokens, temperature zero, request-rate infinity, and reset cache per cell. Wall output tok/s includes prefill and request turnover. Per-stream tok/s is 1000 / TPOT p50. Acceptance source is retained per cell. The 120/480 calibration pairs share a boot at each concurrency; first calibration splits c1 and c4/c8 into two boots, followed by one code boot. Ratios outside 0.98–1.02 are descriptive. Routine results require a matching first-run calibration.\n\n'
        'Code uses shared cached contexts of 1024, 32768 and 65536 tokens with 2048, 2048 and 8192-token appends respectively, 400 forced output tokens, temperature zero, seed 15 and ignore_eos. Decode uses the common interval max(first token) to min(last generated token/finish evidence), continuous token accounting, and no interleaved prefill. Each cell has one scored batch at c1/c4/c8. Per-stream rate is the mean of measured stream rates in that window. Prefix residency must be confirmed on every request. One-batch TTFT p99 is descriptive. Optional step/acceptance counters may be n/a; they use snapshots wholly inside the window and can be affected by metric publication lag.\n\n'
        'TP4, four CMP 170HX cards, four Gen2 x16 links, 175 W configured per card, 52 GiB container cap, DSpark K5, adaptive verification off, FULL_DECODE_ONLY, capture sizes 1–48. Only the base served name is timed. Both-name identity is a correctness gate; abliterated and fast names have no headline performance claim. Serving remains absent after measurement.\n\n'
        'Cold prefill carries the dated 2026-11-16 current-r7e suite-v2 reference at nominal 2K/8K, c1/c4/c8, one output token, T=0 and thinking off, with zero prompt-cache hits and warm-steady host page cache. R7f has the same kernels. The summary retains all six cells, the reference file hash and its chain-window limitation. The card plots c1 and labels its maximum c8 input rate as a dated peak; Spec-Bench TTFT and cached-append TTFT never become cold-prefill rates.\n\n'
        'The summary is a numeric allowlist export. It excludes prompts, token IDs, raw client records, lifetime identifiers, arbitrary report prose and private paths. Hashes retain input/source provenance. This local exporter does not authenticate the original sealed boot evidence; report.py performs that operator-side check before export. No GPU run is performed here.\n')


def fill(report_path, cards_root, root=ROOT, calibration_path=None):
    report = read(report_path)
    calibration = normalize(read(calibration_path), sha(calibration_path)) if calibration_path else None
    s = normalize(report, sha(report_path), calibration=calibration)
    s['cold_prefill'] = cold_prefill(root)
    served = read(root/'config/served.json')
    need(served['definition'] == s['configuration'], 'report daily differs from mirrored serving definition; apply the selected definition first')
    need(served.get('promoted_utc'), 'record the actual selection UTC timestamp before filling')
    datetime.strptime(served['promoted_utc'], '%Y-%m-%dT%H:%M:%SZ')
    rel = f'bench/results/{s["date_utc"]}-{s["configuration"]}-headline'
    markdown = (root/'README.md').read_text()
    need(markdown.count(BEGIN) == markdown.count(END) == 1, 'README headline markers missing/ambiguous')
    markdown = move_fixed_reference(markdown)
    start, end = markdown.index(BEGIN), markdown.index(END) + len(END)
    markdown = markdown[:start] + tables(s) + markdown[end:]
    if s['publication_ready']:
        need(not re.search(r'\bTODO(?:_|\b)', markdown), 'resolve remaining README selection/acceptance placeholders before filling')
    card_path = cards_root/'recipes/deepseek-v4.1-flash.json'
    card, update = card_data(read(root/'bench/headline-card.template.json'), s, rel)
    outputs = {root/rel/'summary.json': encoded(s), root/rel/'NOTES.md': notes(s), root/'README.md': markdown,
        card_path: encoded(card), cards_root/'recipes/deepseek-v4.1-flash.update.json': encoded(update)}
    # Validate all inputs/rendered text before the first write. Never copy source files.
    for path, text in outputs.items():
        need('\x00' not in text, 'invalid output')
    # Complete chart generation before modifying publication files.
    with tempfile.TemporaryDirectory(prefix='headline-figures-') as tmp:
        stage = Path(tmp)
        (stage/'summary.json').write_text(encoded(s))
        subprocess.run([sys.executable, str(root/'bench/plot.py'), '--headline', str(stage/'summary.json'),
                        '--output-dir', str(stage)], check=True)
        for name in ('headline-agent.svg', 'headline-code.svg'):
            outputs[root/'docs/img'/name] = (stage/name).read_text()
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    print(f'Wrote {rel}, README tables/charts and recipe card JSON; publication_ready={s["publication_ready"]}')
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path, help='headline-report.json from report.py; not a boot report or Markdown table')
    p.add_argument('--cards-root', type=Path, required=True, help='local recipe-cards checkout')
    p.add_argument('--calibration-report', type=Path, help='matching first-run headline-report.json for a later routine report')
    a = p.parse_args()
    fill(a.report, a.cards_root, calibration_path=a.calibration_report)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as e:
        sys.exit('fill_headline: ' + str(e))
