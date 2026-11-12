"""Frozen alpha111 failure replay; no production source edits."""
import json,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
runtime=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0,str(runtime));sys.dont_write_bytecode=True
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_core import river_worker
from braids_core.sediment.river_midpoint import RiverMidpoint
from braids_core.sediment.bed_columns_v101 import host

backend='gpu' if '--gpu' in sys.argv else 'cpu'
out=ROOT/('validation/_test1_amr_optimization/failure_probe'+('_gpu' if backend=='gpu' else ''));out.mkdir(parents=True,exist_ok=False)
shutil.copytree(ROOT/f'validation/_test1_amr_profile/{backend}_clean/prepared',out/'prepared')
original=RiverMidpoint._single;linear=RiverMidpoint.linear;seen=[False]
def single(self,base,dt,at):
    try:return original(self,base,dt,at)
    except Exception as exc:
        if seen[0]:raise
        seen[0]=True;tb=exc.__traceback__
        while tb and tb.tb_frame.f_code is not original.__code__:tb=tb.tb_next
        if tb is None:raise
        loc=tb.tb_frame.f_locals;mid=loc['mid'].copy();defect=loc['defect'].copy();xp=self.xp
        result=dict(time=at,dt=dt,error=str(exc),iterations=loc['iteration'],
                    norm=float(xp.linalg.norm(defect)),tolerance=loc['tolerance'],probes=[],variants=[])
        index=int(xp.argmax(abs(defect)))%base.shape[1];a=self.solver;d=a.data
        result['cell']=dict(index=index,base_depth=float(base[0,index]),mid_depth=float(mid[0,index]),
            area=float(d['area'][index]),external_faces=int(np.count_nonzero((d['owner']==index)&(d['neighbour']<0))),
            open_faces=int(np.count_nonzero((d['owner']==index)&(d['neighbour']<0)&(d['bid']>=0))))
        if backend=='gpu':
            records=[]
            for eps in (0.,2.**-12):
                q=mid+eps*loc['delta'];q[0]=xp.maximum(q[0],0.)
                r,w=self.rate(q,dt,at+.5*dt);op=a.hydraulics.flux
                record={k:host(v).copy() for k,v in op.buffers.items() if isinstance(v,xp.ndarray)}
                record.update(trace=host(op.inputs['trace']).copy(),q=host(q),rate=host(r),water=host(w),
                              exterior_flux=host(a.exterior_bed.flux).copy())
                records.append(record);np.savez(out/f'TRACE_{len(records)}.npz',**record)
            result['trace_differences']={}
            for key in records[0]:
                diff=abs(records[1][key]-records[0][key]);j=np.unravel_index(np.argmax(diff),diff.shape)
                result['trace_differences'][key]=dict(max=float(diff[j]),index=[int(i) for i in j],
                    before=float(records[0][key][j]),after=float(records[1][key][j]))
        for name,direction in [('acoustic',loc['delta']),('picard',-defect)]:
            for exponent in (0,4,8,12,16,20,24,28):
                eps=2.**(-exponent);q=mid+eps*direction;q[0]=xp.maximum(q[0],0.)
                rate,water=self.rate(q,dt,at+.5*dt);f=q-base-.5*dt*rate
                result['probes'].append(dict(direction=name,eps=eps,norm=float(xp.linalg.norm(f)),
                    residual_change=float(xp.linalg.norm(f-defect)),state_change=float(xp.linalg.norm(q-mid))))
        def live_linear(rhs,old,tau,target):
            live=sys._getframe(1).f_locals['mid']
            return linear(self,rhs,live,tau,target)
        self.linear=live_linear;began=time.perf_counter();ev=self.evaluations
        try:
            q,w,diag=original(self,base,dt,at)
            result['variants'].append(dict(name='current_iterate_acoustic',status='converged',diag=diag))
        except Exception as error:
            result['variants'].append(dict(name='current_iterate_acoustic',status='failed',error=str(error)))
        finally:
            self.linear=linear.__get__(self);result['variants'][-1].update(wall_s=time.perf_counter()-began,evaluations=self.evaluations-ev)
        np.savez(out/'STATE.npz',base=host(base),mid=host(mid),defect=host(defect),z=host(a.z),**{k:d[k] for k in ('area','owner','neighbour','bid')})
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        raise InterruptedError('Frozen failure captured and replayed')
RiverMidpoint._single=single
print(river_worker.run(out/'prepared',out/'run',until=.4),flush=True)
