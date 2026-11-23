# SPDX-License-Identifier: MIT
"""Exact existing selectors, scoped to target graph capture; no eager change."""
import functools
import importlib.util
import json
import re
from pathlib import Path

ROWS = frozenset((2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,21,24,25,28,30,32,36,42))

def need(ok, why):
    if not ok: raise ValueError(why)

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def raw_control(): return json.loads(Path('/qv/r7b-cell.json').read_text())
def identity_control():
    c = raw_control()
    return dict(c, enabled=False) if 'k' in c else c
def timing_control():
    c = raw_control()
    return c if 'k' in c else dict(c, enabled=False)

def kind(d):
    u = d.uniform_token_count or (d.num_tokens//d.num_reqs if d.num_reqs and d.num_tokens % d.num_reqs == 0 else 0)
    if (getattr(d.cg_mode, 'name', str(d.cg_mode)) != 'FULL' or d.num_ubatches != 1 or
        not 1 <= d.num_reqs <= 8 or
        u not in (2,3,4,6) or d.num_tokens != d.num_reqs*u or
        d.uniform_token_count is None and d.max_query_len != u): return None
    if d.num_tokens == 16 and d.num_reqs == 4 and u == 4: return 'k2'
    return 'rows' if d.num_tokens in ROWS else None

def wanted(arm, d): return arm == 'stack' and kind(d) is not None or arm == kind(d)

class Binding:
    bytes = 0  # no new native workspace; existing per-call ABI scratch is reused
    def __init__(self, coop, rows, k2, rows_selector, k2_selector):
        self.coop, self.rows, self.k2 = coop, rows, k2
        self.rows_selector, self.k2_selector = rows_selector, k2_selector
        self.on = False; self.descriptor = None; self.layer = None
        self.calls = self.eligible_calls = 0
        self.by_kind = dict(rows=0, k2=0)

    def choose(self, args):
        d = self.descriptor
        if not self.on or d is None or not self.coop.eligible(args): return None
        if args[0].shape[0] != d.num_tokens: return None
        family = kind(d)
        if family is None: return None
        selector = self.rows_selector if family == 'rows' else self.k2_selector
        lib = self.rows if family == 'rows' else self.k2
        if family == 'k2':
            match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', str(self.layer))
            if match is None or int(match[1]) not in range(18,23): return None
        if selector.select(None, lib, args, {selector.SWITCH:'1'},
                           role='verify', requests=d.num_reqs) is not lib: return None
        return family, lib

    def install(self):
        old = self.coop.execute
        def execute(*args):
            selected = self.choose(args)
            if selected is None: return old(*args)
            family, lib = selected
            self.calls += 1; self.eligible_calls += 1; self.by_kind[family] += 1
            return lib.exl3_moe_coop(*args)
        self.coop.execute = execute
        import vllm_exl3.exl3 as exl3
        original = exl3.apply_exl3_fused_moe
        @functools.wraps(original)
        def moe(*a, **kw):
            saved = self.layer
            owner = a[3] if len(a) > 3 else kw['layer']
            self.layer = (getattr(owner, '_exl3_prefix', None) or getattr(owner, 'prefix', None)
                          or getattr(owner, 'layer_name', None))
            try: return original(*a, **kw)
            finally: self.layer = saved
        exl3.apply_exl3_fused_moe = moe

def install():
    import hashlib
    bundle = Path('/opt/stack-native')
    need(hashlib.sha256((bundle/'seal.json').read_bytes()).hexdigest() == (bundle/'seal.sha256').read_text().strip(),
         'native seal changed')
    seal = json.loads((bundle/'seal.json').read_text())
    need(seal['schema'] == 'r18-stack-native/1', 'unknown stack seal schema')
    need(all(hashlib.sha256((bundle/n).read_bytes()).hexdigest() == h for n,h in seal['binaries'].items()),
         'exact measured binary changed')
    from native_build import check_r18_environment
    import torch
    check_r18_environment(seal, torch)
    import vllm_exl3.coop_kernel as coop
    incumbent = coop.load()
    rows = load('/opt/stack-native/opera_rows_family.so', 'opera_rows_family')
    k2 = load('/opt/stack-native/opera_k2_fixed16.so', 'opera_k2_fixed16')
    need(incumbent.opera_coop_variant_v1() == rows.opera_coop_variant_v1() == k2.opera_coop_variant_v1() == 1,
         'native ABI differs')
    selectors = [load('/stack-rounds/'+p+'/common.py', '_stack_'+p) for p in
                 ('moe-rows-family-r1', 'moe-k2-fixed16-r1')]
    b = Binding(coop, rows, k2, *selectors); b.install(); return b
