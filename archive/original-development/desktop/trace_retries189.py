"""Identify the failed guard on short dry-groin checkpoint continuations."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')];os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.river_mixtures import RiverMixtureSolver
from braids_core import river_worker
verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
original=RiverMixtureSolver.finish;rows=[]


def observed(self,qold,qnew,water,*,morphology=True):
    try:return original(self,qold,qnew,water,morphology=morphology)
    except Exception as exc:
        if str(exc)!='Class transport budget or bed increment failed':raise
        p=self.physics;g=p.g
        # These test projects are transport-only, one finite fully mixed bed.
        relative=np.abs(p.ztrial-p.z0)/np.maximum(qnew[0],p.sediment_depth);i=int(np.argmax(relative))
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=self.material.boundary_mass(p.class_face_integral,g['length'],outside)
        closure=float(np.max(np.abs(np.sum(p.class_delta*g['area'],axis=1)-boundary)))
        rows.append(dict(dt_s=p._dt,relative_bed_increment=float(relative[i]),limiting_cell=i,
            depth_m=float(qnew[0,i]),bed_increment_m=float(p.ztrial[i]-p.z0[i]),
            sediment_depth_m=p.sediment_depth,closure_kg=closure,finite=bool(np.isfinite(p.ztrial).all())))
        raise


if __name__=='__main__':
    RiverMixtureSolver.finish=observed;results={}
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        source=ROOT/'validation/_bend189/dry/groin'/(law+'_cpu')
        output=ROOT/'validation/_bend189/retry_trace'/law
        checkpoint=next((source/'checkpoints').glob('t_8_*.npz'));rows=[]
        status=river_worker.run(source/'prepared',output,restart=checkpoint,until=8.05)
        assert status['status']=='segment_complete' and status['time_s']==8.05
        assert all(r['closure_kg']<=1e-10 and r['finite'] and r['relative_bed_increment']>.05 for r in rows)
        results[law]=dict(status=status,rejected_inputs=rows)
        print('TRACE',law,'rejected',len(rows),flush=True)
    save(ROOT/'validation/_bend189/retry_trace/CHECK.json',results)
