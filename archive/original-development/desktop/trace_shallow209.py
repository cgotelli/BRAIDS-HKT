"""Read-only candidate instrumentation of packing rejection; no capped target."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_shallow209/candidate_v2'
sys.path[:0]=[str(RUNTIME),str(ROOT)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--project',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    import numpy as np
    from braids_core.sediment.jiang_cao_wetdry import JiangCaoShoreline
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    from braids_core.river_model import resolve_model
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    original=JiangCaoShoreline.target;rows=[];last=None;first=None
    def trace(s,st):
        nonlocal first,last
        try:return original(s,st)
        except BendTrialRejected as exc:
            if 'packing' not in str(exc):raise
            xp=s.xp;host=s.a.host;h=st[0];n=s.shallow.resistance(s.roughness,st[:3])
            w,_=s.shallow.bedload_weight(h,.001,s.dry,xp)
            raw=s.exchange.capacity.evaluate(st[:3],s.mesh.gradient(st[5]),n,
                concentration=st[3]/xp.where(h>0,h,1)).values
            ce=raw[0]*w/xp.where(xp.hypot(st[1],st[2])>0,xp.hypot(st[1],st[2]),1)
            i=int(xp.argmax(ce));g=s.mesh.g
            row=dict(time=s.time,steps=s.steps,cell=i,xy=host(g['xy'][:,i]).tolist(),
                h=float(h[i]),velocity=host(st[1:3,i]/h[i]).tolist(),
                ce=float(ce[i]),packing=s.B,weight=float(w[i]),raw_capacity=float(raw[0,i]),
                theta=float(raw[3,i]),critical=float(raw[4,i]),manning=float(n[i]),
                actual_c=float(st[3,i]/h[i]))
            rows.append(row);last=dict(state=host(st).copy(),accepted=host(s.state).copy(),
                **{k:host(v) for k,v in g.items() if hasattr(v,'shape')})
            if first is None:first=last
            raise
    JiangCaoShoreline.target=trace
    doc=json.loads(Path(a.project).read_text());request=doc['model']['request'];request['execution']['backend']='cpu'
    doc['model']=resolve_model(request).document()
    file=out/'project.json';file.write_text(json.dumps(doc,indent=2))
    result={}
    try:
        prep=prepare(file,out/'prepared',asset_cache=out/'assets')
        result=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    except Exception as exc:result=dict(error=str(exc))
    if first is not None:np.savez(out/'FIRST_REJECTION.npz',**first);np.savez(out/'LAST_REJECTION.npz',**last)
    (out/'TRACE.json').write_text(json.dumps(rows,indent=2));(out/'CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(result=result,rejections=len(rows),first=rows[:1],last=rows[-1:])),flush=True)
if __name__=='__main__':main()
