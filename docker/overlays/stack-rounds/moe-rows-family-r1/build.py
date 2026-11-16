# SPDX-License-Identifier: MIT
"""Same fixed-I arithmetic, one family DSO; no CUDA initialisation during build."""
from common import *
with module('build.py') as sibling: incumbent = sibling.incumbent
GUARD = '('+' || '.join('p.bsz == '+str(r) for r in ROWS if r>1)+')'
def materialize(out):
    with trim() as (t,b,_):
        t.pins(); root,native = b.materialize(out,'fixed640')
    before = {str(f.relative_to(native)):sha(f) for f in native.rglob('*') if f.is_file()}
    file=native/'quant/opera_coop.cu'; old=file.read_text(); new=replace(old,'p.bsz == 16',GUARD)
    file.write_text(new)
    after = {f:sha(native/f) for f in before}
    need([f for f in before if before[f]!=after[f]] == ['quant/opera_coop.cu'],'change exceeds row guard')
    need(new.replace(GUARD,'p.bsz == 16')==old,'arithmetic changed')
    write(root/'proof.json',dict(before=before,after=after,rows=ROWS,change='row membership only; row1 native fallback'))
    return root,native
def adapted():
    source=(R/'c8-moe24-r1/build.py').read_text()
    source=source[:source.index("if __name__ == '__main__'")]
    scope=dict(__name__='_family_build'); exec(compile(source.replace('opera_rows24','opera_rows_family'),'<shared-build>','exec'),scope)
    scope['materialize']=materialize
    return scope
if __name__ == '__main__': adapted()['main']()
