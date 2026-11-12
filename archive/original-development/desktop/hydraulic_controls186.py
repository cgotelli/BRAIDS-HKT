"""Twenty-second cold hydraulic contrasts; diagnostic, never release settings."""
import argparse,json,os,sys,time
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'validation/_secondary185/candidate'
OUT=ROOT/'validation/_mesh186'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from stage_secondary185 import save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=('betac_zero','zero_incoming_secondary','matched_energy_slope'))
    ap.add_argument('--backend',default='cpu',choices=('cpu','gpu'))
    args=ap.parse_args();out=OUT/(args.mode+'_'+args.backend)
    out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop.results import ResultsStore
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    old=river_worker.core.factory
    def instrument(*aa,**kk):
        if args.mode=='matched_energy_slope':
            kk=deepcopy(kk)
            coarse=json.loads((OUT/'coarse_cpu/prepared/NORMAL_SLOPES.json').read_text())
            slope=coarse['boundaries']['Upstream']['energy_slope']
            b=kk['process_parameters']['boundaries']
            b['hydraulics']['Upstream']['energy_slope']=slope
            b['normal_slopes']['boundaries']['Upstream']['energy_slope']=slope
        build=old(*aa,**kk)
        def construct(*a,**k):
            owner=build(*a,**k);p=owner.physics
            if args.mode=='betac_zero':
                # Remove only secondary momentum, retaining the original eddy
                # viscosity, scalar transport and its diffusion.
                p.betac=0.;p.controls[6]=0.
            elif args.mode=='zero_incoming_secondary':
                scalar=p.scalar_faces;g=p.g
                exterior=(g['neighbour']<0)&(g['bid']>=0)
                def incoming(q,hi,water):
                    scalar(q,hi,water)
                    p.scalarflux[exterior&(water<0.)]=0.
                p.scalar_faces=incoming
            return owner
        construct.__dict__.update(build.__dict__);return construct
    river_worker.core.factory=instrument
    save(out/'INPUT.json',dict(mode=args.mode,backend=args.backend,runtime=sha(RUNTIME/'RUNTIME.json'),
        diagnostic=True,installed=False,prepared=str(OUT/f'refined_{args.backend}/prepared'),
        scope='Hydraulic-only 0-20 s wet start; sediment starts 200 s; no checkpoint qualification',
        modified={'betac_zero':'Betac=0, viscosity retained',
                  'zero_incoming_secondary':'Incoming open-boundary secondary intensity zero; interior source and momentum unchanged',
                  'matched_energy_slope':'In-memory boundary forcing uses coarse mesh energy slope 0.0049765625; full secondary boundary/physics retained'}[args.mode]))
    begin=time.perf_counter()
    status=river_worker.run(OUT/f'refined_{args.backend}/prepared',out/'run',until=20.,
        on_frame=lambda path,row:print(args.mode,args.backend,row['time_s'],flush=True))
    assert status['status']=='segment_complete' and status['time_s']==20.
    s=ResultsStore(out/'run');f0=s.read(0);xy=np.array([p.centroid.coords[0] for p in f0.polygons]).T
    ids=np.flatnonzero(xy[0]<5);rows=[]
    for i,t in enumerate(s.times):
        f=s.read(i);v=f.q[2]/f.q[0];I=f.arrays['W']/f.q[0]
        assert np.array_equal(f.z,f0.z)
        rows.append(dict(time_s=float(t),v_max=float(abs(v[ids]).max()),I_max=float(abs(I[ids]).max())))
    d=json.loads((out/'run/DIAGNOSTICS.json').read_text())
    assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in d)
    save(out/'CHECK.json',dict(status=status,elapsed_s=time.perf_counter()-begin,inlet=rows,
         water_residual_m3=max(abs(r['water_residual_m3']) for r in d)))
    print('DONE',args.mode,args.backend,json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
