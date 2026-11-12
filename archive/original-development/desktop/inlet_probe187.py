"""Isolated boundary diagnostic; does not mutate the held runtime."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'validation/_secondary185/candidate'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save,sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('trace','transported'),required=True)
    ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    ap.add_argument('--until',type=float,default=20.)
    args=ap.parse_args();out=ROOT/'validation/_inlet187'/(args.mode+'_'+args.backend)
    out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop.results import ResultsStore
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    old=river_worker.core.factory;owners=[];trace=[];seen=set()
    def instrument(*aa,**kk):
        build=old(*aa,**kk)
        def construct(*a,**k):
            owner=build(*a,**k);owners.append(owner);p=owner.physics;g=p.g;xp=p.xp
            ext=(g['neighbour']<0)&(g['bid']>=0);scalar=p.scalar_faces
            host=lambda a:np.asarray(a) if args.backend=='cpu' else a.get()
            def incoming(q,hi,water):
                scalar(q,hi,water)
                if args.mode=='transported':
                    ids=xp.flatnonzero(ext&(water<0.));cells=g['owner'][ids]
                    p.scalarflux[ids]=water[ids]*xp.where(q[0,cells]>p.parameters.dry,
                        hi[cells]/xp.maximum(q[0,cells],p.parameters.dry),0.)
                bucket=int(owner.time*10+1e-6)
                if bucket not in seen and (bucket<20 or bucket%10==0):
                    seen.add(bucket);ids=xp.flatnonzero(ext&(water<0.));cells=g['owner'][ids]
                    trace.append(dict(time_s=owner.time,faces=host(ids).tolist(),
                        cells=host(cells).tolist(),xy=host(g['xy'][:,cells]).tolist(),
                        curvature=host(p.inputs[4,cells]).tolist(),source=host(p.outputs[3,cells]).tolist(),
                        sink=host(p.outputs[4,cells]).tolist(),equilibrium=host(p.outputs[5,cells]).tolist(),
                        transported=host(hi[cells]/xp.maximum(q[0,cells],p.parameters.dry)).tolist(),
                        water=host(water[ids]).tolist(),flux=host(p.scalarflux[ids]).tolist()))
            p.scalar_faces=incoming
            return owner
        construct.__dict__.update(build.__dict__);return construct
    river_worker.core.factory=instrument
    save(out/'INPUT.json',dict(mode=args.mode,backend=args.backend,runtime=sha(RUNTIME/'RUNTIME.json'),
        diagnostic=True,until=args.until,description='Incoming owner-transported I; no equilibrium imported' if args.mode=='transported' else 'Unchanged v6 boundary traced'))
    start=time.perf_counter()
    result=river_worker.run(ROOT/f'validation/_mesh186/refined_{args.backend}/prepared',out/'run',until=args.until,
        on_frame=lambda path,row:print(args.mode,args.backend,row['time_s'],flush=True) if row['time_s']%10==0 else None)
    save(out/'TRACE.json',trace)
    s=ResultsStore(out/'run');f0=s.read(0);f=s.read(-1)
    xy=np.array([p.centroid.coords[0] for p in f.polygons]).T;ids=xy[0]<5
    d=json.loads((out/'run/DIAGNOSTICS.json').read_text())
    assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in d)
    save(out/'CHECK.json',dict(status=result,elapsed=time.perf_counter()-start,
        inlet_v_max=float(abs(f.q[2,ids]/f.q[0,ids]).max()),
        inlet_I_max=float(abs(f.arrays['W'][ids]/f.q[0,ids]).max()),
        inlet_dz_min_mm=float(1000*(f.z-f0.z)[ids].min()),inlet_dz_max_mm=float(1000*(f.z-f0.z)[ids].max())))
    print((out/'CHECK.json').read_text(),flush=True)

if __name__=='__main__':main()
