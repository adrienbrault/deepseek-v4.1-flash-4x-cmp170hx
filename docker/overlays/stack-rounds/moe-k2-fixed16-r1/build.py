# SPDX-License-Identifier: MIT
"""Add a separate K2 instantiation, preserving the qualified K3 source."""
from common import *
with module('build.py') as b: incumbent=b.incumbent
def materialize(out):
    with trim() as (t,b,_):
        t.pins(); root,native=b.materialize(out,'fixed640')
    before={str(f.relative_to(native)):sha(f) for f in native.rglob('*') if f.is_file()}
    file=native/'quant/opera_coop.cu'; old=file.read_text()
    start=old.index('static bool trim_fixed_ok'); end=old.index('\nvoid opera_coop_launch',start)
    guard=replace(old[start:end],'trim_fixed_ok','k2_fixed_ok')
    guard=replace(guard,'Kg == 3 && Kd == 3','Kg == 2 && Kd == 2')
    start=old.index('    if (p.I == 512 && trim_fixed_ok'); end=old.index('    // TRIM_FIXED_DISPATCH',start)
    dispatch=old[start:end].replace('trim_fixed_ok','k2_fixed_ok').replace('<3,','<2,')
    need(dispatch.count('k2_fixed_ok')==2 and dispatch.count('<2,')==8,'K2 instantiations changed')
    new=old.replace('\nvoid opera_coop_launch','\n'+guard+'\nvoid opera_coop_launch',1)
    new=replace(new,'    // TRIM_FIXED_DISPATCH',dispatch+'    // TRIM_FIXED_DISPATCH')
    need(new.replace(guard+'\n','',1).replace(dispatch,'',1)==old,'change exceeds K2 additions')
    file.write_text(new)
    after={f:sha(native/f) for f in before}
    need([f for f in before if before[f]!=after[f]]==['quant/opera_coop.cu'],'kernel body changed')
    write(root/'proof.json',dict(before=before,after=after,change='separate exact-K2 guard/instantiations; K3 and arithmetic unchanged'))
    return root,native
def adapted_build():
    scope=adapted('c8-moe24-r1','build.py',[('opera_rows24','opera_k2_fixed16',4)])
    scope['materialize']=materialize
    return scope
if __name__=='__main__': adapted_build()['main']()
