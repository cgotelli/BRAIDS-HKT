"""Freeze and probe the installed Test1 inlet failure without changing input."""
import json,os,sys,time,importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
corrected='--corrected' in sys.argv
out=ROOT/('validation/_test1_gpu_fix/residual_corrected' if corrected else 'validation/_test1_gpu_fix/residual_probe');out.mkdir(parents=True,exist_ok=False)
runtime=Path(os.environ['APPDATA'])/'QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime'
sys.path.insert(0,str(runtime));sys.dont_write_bytecode=True
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_core import river_worker
from braids_core.sediment.river_midpoint import RiverMidpoint
from braids_core.sediment.bed_columns_v101 import host

original=RiverMidpoint._single
captured=False
def probe(self,base,dt,at):
    global captured
    try:return original(self,base,dt,at)
    except Exception as exc:
        if captured or 'cell=3949' not in str(exc):raise
        captured=True;tb=exc.__traceback__
        while tb and tb.tb_frame.f_code.co_name!='_single':tb=tb.tb_next
        state=tb.tb_frame.f_locals;mid=state['mid'].copy();a=self.solver;xp=a.xp
        if corrected:
            path=ROOT/'validation/sediment/river_morphodynamics/shoreline_hydraulic_probe.py'
            spec=importlib.util.spec_from_file_location('corrected_probe',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.install(a,'positive_bounded')
        arrays={k:host(state[k]).copy() for k in ('base','mid','defect','rate','water')}
        arrays.update(z=host(a.z),q=host(a.q),dt=np.array(dt),time=np.array(at))
        arrays.update({f'g_{k}':v for k,v in a.data.items() if isinstance(v,np.ndarray)})
        np.savez(out/'FAILED_STATE.npz',**arrays)
        rows=[];saved=[]
        for delta in (-1e-6,-1e-8,-1e-10,-1e-12,0.,1e-12,1e-10,1e-8,1e-6):
            q=mid.copy();q[0,3949]+=delta;r,w=self.rate(q,dt,at+.5*dt)
            op=a.hydraulics.flux
            record={k:host(v).copy() for k,v in op.buffers.items() if isinstance(v,xp.ndarray)}
            record['trace']=host(op.inputs['trace']).copy();record['rate']=host(r);record['water']=host(w)
            saved.append(record)
            np.savez(out/f'RATE_{len(rows):02d}.npz',**record)
            rows.append(dict(delta_h=delta,h=float(q[0,3949]),rate=host(r[:,3949]).tolist(),
                defect=host((q-base-.5*dt*r)[:,3949]).tolist()))
        reference=saved[4]
        for row,r in zip(rows,saved):
            for key in ('closed','fl','fr','trace','local','riemann'):
                if key in r:
                    diff=r[key]!=reference[key]
                    row[key+'_changed']=np.flatnonzero(np.any(diff,axis=0) if diff.ndim==2 else diff).tolist()
        solve=None
        if corrected:
            started=time.perf_counter();before=self.evaluations
            end,water,diagnostics=original(self,base,dt,at)
            solve=dict(wall_s=time.perf_counter()-started,evaluations=self.evaluations-before,diagnostics=diagnostics)
            np.savez(out/'SOLVED.npz',q=host(end),water=host(water))
        (out/'PROBE.json').write_text(json.dumps(dict(error=str(exc),dt=dt,time=at,rows=rows,corrected_solve=solve),indent=2))
        print(json.dumps(rows),flush=True)
        raise InterruptedError('Diagnostic frozen state captured')
RiverMidpoint._single=probe
parent=Path('C:/Users/STREEM/Desktop/Test1/runs')
print(river_worker.run(parent/'prepare_20261021_162018_0bf405',out/'run',
    restart=parent/'run_20261021_162057_b146ad/checkpoint.npz',until=.95))
