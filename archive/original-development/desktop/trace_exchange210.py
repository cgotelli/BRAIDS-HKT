"""Observe real native exchange inputs/outputs; no source or state substitution."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_exchange210/candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','gpu'],required=True);a=p.parse_args()
OUT=ROOT/('validation/_exchange210/trace_'+a.backend);OUT.mkdir(exist_ok=False)
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
from braids_core.sediment.jiang_cao_exchange import JiangCaoExchange
from braids_core.river_worker import run
import numpy as np
original=JiangCaoExchange.exchange;rows=[]
def trace(self,state,ce,dt):
    result=original(self,state,ce,dt);xp=self.xp;B=1-self.material.porosity
    s=xp.asarray(state);v=result.state
    rows.append(dict(dt=dt,target_max=float(xp.max(ce)),above_packing_cells=int(xp.count_nonzero(xp.asarray(ce)>=B)),
        input_concentration_max=float(xp.max(s[3]/xp.where(s[0]>0,s[0],1))),
        output_concentration_max=float(xp.max(v[3]/xp.where(v[0]>0,v[0],1))),
        stock_min=float(xp.min(v[4])),stock_exhausted_cells=int(xp.count_nonzero(result.status==1)),
        solid_error_max=float(xp.max(abs(v[3]+v[4]-s[3]-s[4]))),
        water_error_max=float(xp.max(abs(v[0]-v[3]+self.material.porosity*v[4]/B-(s[0]-s[3]+self.material.porosity*s[4]/B))))))
    return result
JiangCaoExchange.exchange=trace
result=run(ROOT/f'validation/_exchange210/strong_{a.backend}/prepared',OUT/'run',until=.02)
assert result['status']=='complete' and result['time_s']==.02
assert sum(r['above_packing_cells'] for r in rows)>0
assert max(r['output_concentration_max'] for r in rows)<.6
(OUT/'TRACE.json').write_text(json.dumps(rows,indent=2))
summary=dict(calls=len(rows),calls_above_packing=sum(r['above_packing_cells']>0 for r in rows),
    target_max=max(r['target_max'] for r in rows),actual_max=max(r['output_concentration_max'] for r in rows),
    solid_error_max=max(r['solid_error_max'] for r in rows),water_error_max=max(r['water_error_max'] for r in rows))
# Old checkpoints cannot be reinterpreted under the changed source identity.
old=next((ROOT/f'validation/_shallow209/v3_flume_{a.backend}/checkpoints').glob('*.npz'))
try:run(ROOT/f'validation/_exchange210/flume_{a.backend}/prepared',OUT/'old_restart_rejected',restart=old)
except (ValueError,RuntimeError) as exc:summary['old_checkpoint_rejected']=str(exc)
else:raise AssertionError('Old source checkpoint was accepted')
(OUT/'CHECK.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
