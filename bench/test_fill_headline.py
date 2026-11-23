#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic synthetic report/2 export tests; no model, Docker or server."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import fill_headline as f

OPERATOR_REPORT = None
KEEP_OUTPUT = None


def fixture():
    """Shape follows report.evaluate(), measure.real_cell() and probes.summarize()."""
    cells = {}
    life = dict(id='synthetic-lifetime', started='2026-11-16T21:00:00Z')
    for size, count in (('subset', 120), ('full', 480)):
        for c in f.CS:
            rate = {1:130., 4:240., 8:280.}[c] * (1.01 if size == 'subset' else 1)
            r = dict(concurrency=c, output_tok_s=rate, req_s=rate/256,
                ttft_p50_ms=200., ttft_p99_ms=500., tpot_p50_ms=8., tpot_p99_ms=10., per_stream_tok_s=125.,
                e2e_p50_s=2.24, tau=3., cached_prompt_fraction=0., flags=[], sample_sha256='a'*64,
                duration_s=count*256/rate, output_tokens=count*256, requests=count,
                acceptance_length=3., acceptance_source='vLLM client acceptance_length', status='COMPLETE', size=size, lifetime=life)
            if c == 1:
                r['prefill'] = dict(definition='synthetic proxy', median_input_tok_s=500., buckets=[],
                    prompt_distribution=dict(count=count, min=100, p25=100, p50=100, p75=100, p90=100, p99=100, max=100, total=100*count),
                    sampler_prompt_lengths=[100]*count, server_input_lengths=[100]*count,
                    sampler_distribution=dict(min=100, p50=100, max=100, total=100*count))
            cells[f.spec_key(size,c)] = r
    for p, a in f.CONTEXTS:
        for c in f.CS:
            rate = {1:200., 4:420., 8:500.}[c] * (1 if p == 1024 else .9 if p == 32768 else .8)
            cells[f.code_key(p,a,c)] = dict(status='COMPLETE', batches=1, requests=c, aggregate_tok_s=rate,
                per_stream_tok_s=rate/c, step_ms=None, accepted_per_step=None, ttft_p50_ms=a/2., ttft_p99_ms=a/2.+50.,
                cache_confirmed=True, cached_tokens=[p]*c,
                cache_proofs=[dict(status='PASS',cached_tokens=[p]*c)],
                counter_window='suite-v2 drafts/accepted semantics; metric snapshots wholly inside common decode-only interval; whole-batch counters diagnostic only')
    distributions = {}
    for cat in f.CATEGORIES:
        d = lambda n: dict(count=n,min=100,p25=100,p50=100,p75=100,p90=100,p99=100,max=100,mean=100,total=100*n)
        distributions[cat] = dict(full=d(80),subset=d(20),ratios={k:1. for k in ('p25','p50','p75','p90','p99','mean')},flags=[])
    ratios = {}
    for c in f.CS:
        a,b = (cells[f.spec_key(size,c)] for size in ('subset','full'))
        ratios[str(c)] = dict(status='COMPLETE', metrics={k:dict(subset_over_full=a[k]/b[k],flag=abs(a[k]/b[k]-1)>.02) for k in f.METRICS})
    return dict(schema='headline-report/2', status='MEASURED', cells=cells, daily='current-r7f',
        measured_names=[f.RAW], excluded_names={name:'outside headline workload' for name in
            (f.RAW+'-abliterated',f.RAW+'-fast',f.RAW+'-abliterated-fast')}, source_key='b'*64,
        starts=['2026-11-16 22:00:00 Paris','2026-11-16 22:40:00 Paris','2026-11-16 23:20:00 Paris'], boots=3, planned_boots=3,
        mode='first', section_wall_s={'real-c1':dict(total_s=2400,measurements={'specbench_s':1300})}, serving_after=False,
        promotion_eligible=False, ratios=ratios,
        subset_manifest=dict(schema='headline-subset/2',seed=7310,algorithm='character-length quartets within six strata; minimum seeded ID hash in each quartet',
            categories=list(f.CATEGORIES),ids=list(range(120)),id_list_sha256='c'*64,full_rows_sha256='d'*64,count=120),
        prompt_distributions=dict(unit='raw sampler tokens, exact served tokenizer; before chat template',categories=distributions),
        tasks_sha256=f.TASKS,corpus_sha256=f.CORPUS,separate_contexts=False)


class ExportTests(unittest.TestCase):
    def normalize(self, report=None, **kw):
        result = f.normalize(report or fixture(), 'e'*64, **kw)
        result['cold_prefill'] = f.cold_prefill()
        return result

    def test_complete_and_exact_metrics(self):
        s = self.normalize()
        self.assertTrue(s['publication_ready'])
        self.assertEqual(s['date_utc'],'2026-11-16')
        self.assertEqual(s['starts_utc'][0],'2026-11-16T21:00:00Z')
        self.assertEqual(s['cells']['specbench-subset-c4']['output_tok_s'],242.4)
        self.assertEqual(s['cells']['code-shared-t65536-a8192-c8']['aggregate_tok_s'],400.)
        self.assertIsNone(s['cells']['code-shared-t1024-a2048-c1']['step_ms'])

    def test_partial_and_missing_boot_refuse(self):
        for mutate in (lambda r:r.update(status='INCOMPLETE'),
                       lambda r:r['cells']['specbench-subset-c4'].update(status='INCOMPLETE'),
                       lambda r:r.update(boots=2, starts=r['starts'][:2])):
            r=fixture();mutate(r)
            with self.assertRaises(ValueError): self.normalize(r)

    def test_missing_cell_and_wrong_schema_refuse(self):
        for mutate in (lambda r:r['cells'].pop('code-shared-t32768-a2048-c8'),lambda r:r.update(schema='headline-report/1')):
            r=fixture();mutate(r)
            with self.assertRaises(ValueError): self.normalize(r)

    def test_nan_bool_zero_and_negative_refuse(self):
        for v in (float('nan'),float('inf'),True,0,-1,None):
            r=fixture();r['cells']['specbench-subset-c1']['output_tok_s']=v
            with self.assertRaises(ValueError): self.normalize(r)

    def test_missing_acceptance_and_code_rates_refuse(self):
        for key, name in (('specbench-subset-c1','acceptance_length'),
                          ('code-shared-t65536-a8192-c8','aggregate_tok_s'),
                          ('code-shared-t65536-a8192-c4','per_stream_tok_s')):
            r=fixture();r['cells'][key][name]=None
            with self.assertRaises(ValueError): self.normalize(r)

    def test_wrong_ratio_and_cache_proof_refuse(self):
        for mutate in (lambda r:r['ratios']['4']['metrics']['output_tok_s'].update(subset_over_full=3),
                       lambda r:r['cells']['code-shared-t1024-a2048-c4'].update(cache_confirmed=False),
                       lambda r:r['cells']['code-shared-t1024-a2048-c4'].update(requests=8),
                       lambda r:r['cells']['specbench-subset-c4'].update(requests=480)):
            r=fixture();mutate(r)
            with self.assertRaises(ValueError): self.normalize(r)

    def test_ratio_flags_are_descriptive(self):
        r=fixture();r['cells']['specbench-subset-c4']['output_tok_s']=300.
        r['ratios']['4']['metrics']['output_tok_s']=dict(subset_over_full=1.25,flag=True)
        s=self.normalize(r);self.assertTrue(s['publication_ready']);self.assertIn('1.2500 FLAG',f.tables(s))

    def test_allowlist_strips_private_strings_and_token_ids(self):
        r=fixture();marker='DO_NOT_EXPORT_PRIVATE_SENTINEL'
        r['cells']['specbench-subset-c1']['reason']=marker
        r['cells']['specbench-subset-c1']['token_ids']=[999999999]
        r['excluded_names'][f.RAW+'-fast']=marker
        r['subset_manifest']['ids']=[marker]*120
        s=self.normalize(r);encoded=f.encoded(s)
        self.assertNotIn(marker,encoded);self.assertNotIn('token_ids',encoded)
        self.assertNotIn('lifetime',encoded);self.assertNotIn('sampler_prompt_lengths',encoded)

    def test_routine_needs_matching_calibration(self):
        r=fixture();r.update(mode='routine',boots=1,planned_boots=1,starts=r['starts'][:1],ratios={})
        r['cells']={k:v for k,v in r['cells'].items() if not k.startswith('specbench-full')}
        with self.assertRaises(ValueError):self.normalize(r)
        cal=self.normalize();self.assertTrue(self.normalize(r,calibration=cal)['publication_ready'])
        cal['subset']['id_list_sha256']='f'*64
        with self.assertRaises(ValueError):self.normalize(r,calibration=cal)

    def test_draft_card_cannot_keep_a_previous_rate(self):
        s=self.normalize();s['publication_ready']=False
        card,update=f.card_data(f.read(f.ROOT/'bench/headline-card.template.json'),s,'synthetic-results')
        self.assertIsNone(card['one']['rate']);self.assertIsNone(card['many']['sum'])
        self.assertEqual(card['conc']['sum'],[None,None,None])
        self.assertIsNone(update['history'])
        self.assertEqual(card['conc']['each'],[None,None,None])
        self.assertEqual(card['evals']['items'][2],['3,961','cold prefill tok/s · c8','2026-11-16'])

    def test_card_c1_matches_chart_and_prefill_is_dated(self):
        s=self.normalize()
        card, update=f.card_data(f.read(f.ROOT/'bench/headline-card.template.json'),s,'numeric-results')
        self.assertEqual(card['one']['rate'], 131.3)
        self.assertEqual(card['one']['kinds'][0], ['Spec-Bench', 131.3, 'wall output'])
        self.assertEqual(card['one']['rate'], card['conc']['sum'][0])
        self.assertEqual(card['one']['kinds'][1][1], card['conc']['peak']['sum'][0])
        self.assertEqual(card['one']['precision'], card['many']['precision'])
        self.assertEqual(card['many']['sum'],282.8)
        self.assertEqual(card['many']['each'],125.)
        self.assertEqual(card['conc']['each'],[125.,125.,125.])
        self.assertEqual(card['setup'],[
            ['64K+8K · 4 agents','84 each · 336 sum · 4.1 s first token'],
            ['64K+8K · 8 agents','50 each · 400 sum · 4.1 s first token']])
        self.assertEqual(card['evals']['kind'],'peaks')
        self.assertEqual(card['depth']['prefill'],[[2048,2946],[8192,3680]])
        self.assertEqual(len(card['headline']['cold_prefill']['cells']),6)
        self.assertEqual(card['headline']['cold_prefill']['summary_sha256'],f.sha(f.ROOT/f.COLD_DIR/'summary.json'))
        self.assertEqual(card['power']['perGpu'],[175]*4)
        self.assertIsNone(update['history'])
        self.assertTrue(all(d[1]=='—' for d in update['deltas']))
        self.assertNotIn('ttftLabel',card['one'])
        self.assertNotIn('prefillNote',card['depth'])

    def test_peaks_select_highest_measured_context_per_concurrency(self):
        s=self.normalize()
        # Maximum c8 throughput can occur at depth; do not silently pick 1K.
        s['cells']['code-shared-t32768-a2048-c8']['aggregate_tok_s']=900.
        s['cells']['code-shared-t65536-a8192-c1']['aggregate_tok_s']=250.
        s['cells']['code-shared-t65536-a8192-c1']['per_stream_tok_s']=250.
        card, _=f.card_data(f.read(f.ROOT/'bench/headline-card.template.json'),s,'numeric-results')
        self.assertEqual(card['evals']['items'][:2], [['900.0', '8 agents · 32K + 2K'], ['250.0', '1 agent · 64K + 8K']])
        self.assertEqual(card['one']['kinds'][1][1], 200.)
        self.assertIn('`code-shared-t32768-a2048-c8`',f.tables(s))
        self.assertIn('`code-shared-t65536-a8192-c1`',f.tables(s))

    def test_cold_reference_refuses_missing_geometry_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/f.COLD_DIR/'summary.json'
            target.parent.mkdir(parents=True)
            r=f.read(f.ROOT/f.COLD_DIR/'summary.json');r['prefill']['cells'].pop()
            target.write_text(f.encoded(r))
            with self.assertRaises(ValueError):f.cold_prefill(root)

    def test_marker_migration_keeps_fixed_reference_once(self):
        original = ('# Recipe\n\n## Numbers\n\n'+f.BEGIN+'\n\nInterim note\n\n'
            '### Loaded suite v2, 2026-11-16 UTC\n\nDated reference.\n\n'
            '| c8 | 456.7 |\n\n'+f.END+'\n\n## Served configuration\n\nSelected.\n\n## Reproducing\n')
        moved=f.move_fixed_reference(original)
        self.assertEqual(moved.count('## Loaded suite v2, 2026-11-16 UTC'),1)
        self.assertGreater(moved.index('## Loaded suite v2'),moved.index(f.END))
        self.assertIn('| c8 | 456.7 |',moved)
        self.assertNotIn('Interim note',moved)
        self.assertEqual(moved,f.move_fixed_reference(moved))

    def test_launcher_joins_prefill_and_verification_policies_with_memory_pair(self):
        spec=importlib.util.spec_from_file_location('serve',f.ROOT/'scripts/serve.py')
        serve=importlib.util.module_from_spec(spec);spec.loader.exec_module(serve)
        plan=f.read(f.ROOT/'config/served.json');serve.validate_plan(plan)
        for mutate in (
            lambda r: r['docker_options'].__setitem__(r['docker_options'].index('--memory') + 1, '38654705664'),
            lambda r: r['environment'].update(VLLM_DSV41_PREFILL_SHARE='off'),
            lambda r: r['environment'].update(VLLM_DSV41_ADAPTIVE_VERIFY_POLICY=''),
            lambda r: r['policy'].update(adaptive_verify_sha256='0' * 64),
        ):
            bad=deepcopy(plan);mutate(bad)
            with self.assertRaises(ValueError):serve.validate_plan(bad)

    def test_operator_report_schema_and_markdown(self):
        if not OPERATOR_REPORT:
            self.skipTest('supply --operator-report for original report.py schema/renderer check')
        tree=ast.parse(OPERATOR_REPORT.read_text())
        evaluate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        result=next(n.value for n in evaluate.body if isinstance(n,ast.Return))
        self.assertEqual(set(fixture()),{k.arg for k in result.keywords})
        markdown=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='markdown')
        class Probes:
            geometry=staticmethod(lambda separate:[('shared',p,a,c) for p,a in f.CONTEXTS for c in f.CS])
            key=staticmethod(lambda mode,p,a,c:f.code_key(p,a,c,mode))
        scope={'MODEL':f.RAW,'json':json,'own':lambda name:Probes}
        exec(compile(ast.Module(body=[markdown],type_ignores=[]),'operator-markdown','exec'),scope)
        rendered=scope['markdown'](fixture())
        self.assertIn('subset c8',rendered);self.assertIn('64K + 8K',rendered)

    def test_end_to_end_deterministic_and_failure_without_writes(self):
        with tempfile.TemporaryDirectory(prefix='headline-fill-') as tmp:
            root=Path(tmp)/'recipe';cards=Path(tmp)/'cards'
            for rel in ('bench/plot.py','bench/headline-card.template.json',f.COLD_DIR+'/summary.json'):
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f.ROOT/rel,p)
            (root/'README.md').write_text(
                f'# Recipe\n\n## Numbers\n\n{f.BEGIN}\n{f.END}\n\n'
                '## Loaded suite v2, 2026-11-16 UTC\n\n| 8 burst | 456.7 |\n'
            )
            (root/'config').mkdir()
            (root/'config/served.json').write_text(f.encoded({'definition': 'current-r7f', 'promoted_utc': '2026-11-16T18:25:00Z'}))
            (cards/'recipes').mkdir(parents=True)
            # A local reference file has only shape/hardware; no measurements are imported.
            source=Path(__file__).resolve().parent/'headline-card.template.json'
            shutil.copyfile(source,cards/'recipes/deepseek-v4.1-flash.json')
            report=Path(tmp)/'headline-report.json';report.write_text(f.encoded(fixture()))
            f.fill(report,cards,root)
            first={str(p.relative_to(tmp)):p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            f.fill(report,cards,root)
            second={str(p.relative_to(tmp)):p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            self.assertEqual(set(first),set(second))
            for name in first:
                self.assertEqual(first[name],second[name],name)
            text=(root/'README.md').read_text()
            self.assertNotIn('TODO',text)
            self.assertEqual(text.count('## Loaded suite v2, 2026-11-16 UTC'),1)
            self.assertGreater(text.index('## Loaded suite v2'),text.index(f.END))
            self.assertIn('| 8 burst | 456.7 |',text)
            self.assertIn('### Cached code matrix',text)
            self.assertIn('**Parallel agents at depth:**',text)
            self.assertNotIn('scheduled for the night',text)
            self.assertIn('full-480 output tok/s',text)
            self.assertNotIn('PUBLICATION BLOCKED',(root/'README.md').read_text())
            self.assertEqual(f.read(cards/'recipes/deepseek-v4.1-flash.json')['many']['sum'],282.8)
            if KEEP_OUTPUT:
                shutil.copytree(tmp,KEEP_OUTPUT,dirs_exist_ok=True)
            # Changing a report's daily refuses before updating any target file.
            r=fixture();r['daily']='current-r7g';report.write_text(f.encoded(r))
            with self.assertRaises(ValueError):f.fill(report,cards,root)
            self.assertEqual((root/'README.md').read_bytes(),second['recipe/README.md'])
            # An incomplete cell must not mutate any output, including cards/figures.
            r=fixture();r['cells']['code-shared-t65536-a8192-c8']={'status':'SKIPPED','reason':'synthetic timeout'}
            report.write_text(f.encoded(r))
            with self.assertRaises(ValueError):f.fill(report,cards,root)
            for name, data in second.items():
                if name != 'headline-report.json':
                    self.assertEqual((Path(tmp)/name).read_bytes(),data,name)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--operator-report',type=Path);p.add_argument('--keep-output',type=Path)
    a,other=p.parse_known_args();OPERATOR_REPORT=a.operator_report;KEEP_OUTPUT=a.keep_output
    unittest.main(argv=[__file__]+other)
