"""Read-only instrumentation of preview35; isolated public-worker replay."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--project',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    import numpy as np
    from braids_core.sediment.jiang_cao_wetdry import JiangCaoShoreline
    from braids_core.sediment.jiang_cao_adjustment import AdjustmentOperator
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    rows=[];saved=False
    def record(s,st,phase,dt=0.,coef=None):
        nonlocal saved
        xp=s.xp;g=s.mesh.g;host=s.a.host;h=st[0];wet=h>s.dry
        v=st[1:3]/xp.where(wet,h,1.);U=xp.hypot(*v);I=st[6]/xp.where(wet,h,1.)
        w,_=s.shallow.weight(h,s.roughness.parameters.d90_m,xp)
        raw=s.direction.evaluate(v,w>0);n=s.shallow.resistance(s.roughness,st[:3]);hr=.01
        aa=xp.sqrt(9.81)*n/(.4*xp.maximum(h,hr)**(1/6));L=xp.maximum(h,hr)*(1-2*aa)/(.32*aa)
        i=int(xp.argmin((g['xy'][0]-.1)**2+(g['xy'][1]-.5)**2))
        off=host(s.direction.offsets);nb=host(s.direction.neighbours)[off[i]:off[i+1]]
        row=dict(t=s.time,step=s.steps,phase=phase,dt=dt,cell=i,xy=host(g['xy'][:,i]).tolist(),
                 h=float(h[i]),u=float(v[0,i]),v=float(v[1,i]),I=float(I[i]),hk=float(I[i]/U[i]) if U[i]>0 else 0.,
                 k=float(raw[6,i]),target=float(U[i]*h[i]*raw[6,i]),L=float(L[i]),weight=float(w[i]),
                 n=float(n[i]),profile_a=float(aa[i]),gradient=host(raw[:4,i]).tolist(),
                 neighbours=[dict(cell=int(j),xy=host(g['xy'][:,j]).tolist(),h=float(h[j]),u=float(v[0,j]),v=float(v[1,j]),active=bool(w[j]>0)) for j in nb])
        rows.append(row)
        if phase=='failed' and not saved:
            np.savez(out/'FAILED_STATE.npz',state=host(st),accepted=host(s.state),
                     **{k:host(v) for k,v in g.items() if hasattr(v,'shape')})
            saved=True
    relax=AdjustmentOperator.relax
    def traced_relax(self,state,dt,coefficient_state=None):
        record(self.owner,state,'source_before',dt,coefficient_state)
        result=relax(self,state,dt,coefficient_state)
        record(self.owner,result,'source_after',dt,coefficient_state)
        return result
    relative=JiangCaoShoreline.relative_discharge
    def traced_relative(self,st,cap):
        record(self,st,'residual')
        try:return relative(self,st,cap)
        except Exception:
            record(self,st,'failed');raise
    AdjustmentOperator.relax=traced_relax;JiangCaoShoreline.relative_discharge=traced_relative
    doc=json.loads(Path(a.project).read_text());(out/'project.json').write_text(json.dumps(doc,indent=2))
    result={}
    try:
        prep=prepare(out/'project.json',out/'prepared',asset_cache=out/'assets')
        result=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    except Exception as e:result=dict(error=repr(e))
    finally:
        (out/'TRACE.json').write_text(json.dumps(rows,indent=2));(out/'CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
