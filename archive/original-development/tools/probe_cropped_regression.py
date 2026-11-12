"""Distinguish an existing sediment input guard failure from hydraulic regression."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.sediment.river_morphodynamics import combined_midpoint as case
from validation.sediment.river_morphodynamics import shoreline_hydraulic_probe as probe
from braids_core.sediment.river_suspension import SplitCapacity
out=ROOT/'validation/_test1_gpu_fix/cropped_regression';out.mkdir(exist_ok=False)
case.prepared=lambda name:ROOT/'validation/_test1_gpu_fix/unit_v1/current_midpoint_inputs0'/name
original=SplitCapacity.evaluate
def evaluate(self,q,hi,**kw):
    try:return original(self,q,hi,**kw)
    except ValueError:
        cq=kw.get('capacity_q',q);c=kw.get('chezy',np.full(q.shape[1],self.c))
        bad=(c<20)|(q[0]<=0)|(cq[0]<=0)|~np.isfinite(c)
        np.savez(out/(label+'_failure.npz'),q=q,cq=cq,c=c,hi=hi,bad=bad)
        print(label,'bad',np.flatnonzero(bad),'chezy',c[bad], 'depth',q[0,bad],flush=True)
        raise
SplitCapacity.evaluate=evaluate
corrected=probe.native_code
def old(code,mode):
    code=corrected(code,mode)
    if mode=='positive_bounded':
        needle='     if(eta-zb<dry_tolerance) phi=min(phi,(h-dry_tolerance)/max(h-(eta-zb),tiny(1.d0)))'
        assert code.count(needle)==1
        code=code.replace(needle,needle+'\n     if(any(rectangle_width(:,c)<=0.d0).and.abs(zb-v(bed_primitive_index(),c))>=h) phi=0.d0')
    return code
results={}
for label in ('corrected','old'):
    probe.native_code=corrected if label=='corrected' else old
    a=case.make('cropped','cpu',out/label,'midpoint')
    try:
        for _ in range(10):a.step(.002)
        results[label]='complete'
    except Exception as exc:results[label]=str(exc)
(out/'RESULT.json').write_text(json.dumps(results,indent=2));print(results)
