"""Read-only operator tracing on independent 0.2 s warm-checkpoint continuations."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_wall188/desktop_candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop')]
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save,sha
from braids_desktop.results import ResultsStore
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.river_capacity import ResolvedRiverCapacity
from braids_core.sediment.bed_columns_v101 import host
from braids_core import river_worker
from bedload_oracle189 import field
verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
original=ResolvedRiverCapacity.evaluate
best=None;distance=np.inf;reference=None


def capture(self,q,hi,*,capacity_q=None,chezy=None):
    global best,distance
    value=original(self,q,hi,capacity_q=capacity_q,chezy=chezy)
    if q.shape!=reference.shape:return value
    delta=float(abs(host(q)-reference).max())
    if delta>=distance:return value
    cq=q if capacity_q is None else capacity_q
    c=self.xp.full(q.shape[1],self.c) if chezy is None else chezy
    expected=field(self,q,hi,cq,c)
    np.testing.assert_allclose(host(value),expected,rtol=2e-11,atol=1e-11)
    best=dict(q=host(q).copy(),hi=host(hi).copy(),capacity_q=host(cq).copy(),
        chezy=host(c).copy(),hydraulic_chezy=host(self.hydraulic_chezy).copy(),
        z0=host(self.z0).copy(),value=host(value).copy(),
        no_steering=field(self,q,self.xp.zeros_like(hi),cq,c))
    distance=delta
    return value


if __name__=='__main__':
    ResolvedRiverCapacity.evaluate=capture
    base=ROOT/'validation/_bend189'
    sources={'on':ROOT/'validation/_wall188/release/wet_prescribed_cpu',
             **{k:base/'runs'/f'{k}_cpu' for k in ('off','secondary','viscosity')}}
    for mode,source in sources.items():
        store=ResultsStore(source/'run');reference=store.read(int(np.flatnonzero(store.times==200.)[0])).q
        best=None;distance=np.inf
        out=base/'operators_v2'/mode;out.mkdir(parents=True,exist_ok=False)
        checkpoint=source/'checkpoints/t_200_step_00001000.npz'
        status=river_worker.run(source/'prepared',out/'run',restart=checkpoint,until=200.2)
        assert status['status']=='segment_complete' and status['time_s']==200.2
        # The split operator evaluates after its hydraulic predictor, not at
        # the checkpoint. Record the actual input and this small displacement.
        assert best is not None and distance<1e-4, distance
        np.savez(out/'INPUTS.npz',**best)
        save(out/'CHECK.json',dict(status=status,warm_q_max_difference=distance,
             phase='capacity evaluation nearest the 200 s checkpoint after hydraulic predictor',
             checkpoint=str(checkpoint),runtime=sha(RUNTIME/'RUNTIME.json')))
        print('CAPTURE',mode,'warm predictor difference',distance,flush=True)
