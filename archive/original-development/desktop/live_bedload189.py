"""Observe actual interior/exterior formula inputs during public groin execution."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
import numpy as np
from braids_core.sediment.river_capacity import ResolvedRiverCapacity
from braids_core.sediment.bed_columns_v101 import host
from bedload_oracle189 import field
from stage_secondary185 import save,sha
import check_smart183 as run
run.OUT=ROOT/'validation/_bend189/live'
original=ResolvedRiverCapacity.evaluate
rows=[]


def checked(self,q,hi,*,capacity_q=None,chezy=None):
    result=original(self,q,hi,capacity_q=capacity_q,chezy=chezy)
    self._audit189_count=getattr(self,'_audit189_count',0)+1
    if self._audit189_count<=4 or self._audit189_count%128==0:
        cq=q if capacity_q is None else capacity_q
        c=self.xp.full(q.shape[1],self.c) if chezy is None else chezy
        expected=field(self,q,hi,cq,c);actual=host(result)
        np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=1e-12)
        h=host(q[0]);z0=host(self.z0);cs=host(c)
        ref_c=np.maximum(1e-6,np.sqrt(9.81)/.41*np.log1p(np.maximum(.01,h)/np.maximum(1e-8,np.e*z0)))
        np.testing.assert_allclose(cs,ref_c,rtol=1e-12,atol=1e-13)
        factor=.41*cs/(np.sqrt(9.81)*np.log1p(h*float(np.float32(.368))/z0))
        np.testing.assert_allclose(host(cq[1:]),host(q[1:])*factor,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(host(self.hiding),1.,rtol=0,atol=1e-14)
        if self.identity['method'] in ('capacity.mpm','capacity.wang_fredsoe'):
            np.testing.assert_allclose(host(self.d90),1.5*self.d,rtol=1e-12)
        if self.is_smart_jaggi:np.testing.assert_allclose(host(self.hydraulic_chezy),20.,rtol=1e-12)
        assert self.d==.005 and self.solid_density==2650 and self.bed_factor==1.
        rows.append(dict(method=self.identity['method'],cells=q.shape[1],call=self._audit189_count,
            max_field_error=float(abs(actual-expected).max()),depth_min_m=float(h.min()),depth_max_m=float(h.max()),
            sediment_chezy_min=float(cs.min()),sediment_chezy_max=float(cs.max()),
            d90_min_m=float(host(self.d90).min()),d90_max_m=float(host(self.d90).max())))
    return result


if __name__=='__main__':
    ResolvedRiverCapacity.evaluate=checked
    sys.argv+=['--runtime',str(RUNTIME),'--tag','groin']
    run.main()
    law=sys.argv[sys.argv.index('--law')+1];backend=sys.argv[sys.argv.index('--backend')+1]
    save(run.OUT/f'groin/{law}_{backend}/INPUT_AUDIT.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),
        sampled_calls=len(rows),sampled_cells=sum(r['cells'] for r in rows),rows=rows,complete=True))
