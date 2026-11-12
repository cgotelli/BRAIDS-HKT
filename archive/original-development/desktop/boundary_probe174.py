"""Same-state open-boundary replay; no production or FLOW source changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_corner174 import setup,project,BASE
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore


def native_state(folder,xy,t):
    ix,iy=np.rint(xy[:,0]/.5+.5).astype(int),np.rint(xy[:,1]/.5+.5).astype(int)
    with Dataset(folder/'trim-sim_P057.nc') as d:
        j=int(np.flatnonzero(np.asarray(d['time'][:])==t)[0])
        z=-np.asarray(d['DPS'][j])[ix,iy];eta=np.asarray(d['S1'][j]);u=np.asarray(d['U1'][j,0]);v=np.asarray(d['V1'][j,0])
        uv=np.stack([.5*(u+np.roll(u,1,0)),.5*(v+np.roll(v,1,1))])[:,ix,iy]
        h=eta[ix,iy]-z
        # Native staggered boundary flux reconstructed with the actual Flood
        # upwind-depth rule, not cell-centred velocity. Initial fixed DPU is
        # valid only before morph activation; assert that below.
        assert abs(np.asarray(d['DPS'][j])-np.asarray(d['DPS'][0])).max()==0
        dpu=np.asarray(d['DPU0'][:]);depth=np.empty((2,16))
        for k,i in enumerate((0,80)):
            vel=u[i,1:17];a=eta[i,1:17];b=eta[i+1,1:17]
            depth[k]=dpu[i,1:17]+np.where(vel>=.001,a,np.where(vel<=-.001,b,np.maximum(a,b)))
        flux=u[[0,80],1:17]*depth*.5
        metadata=dict(face_flux_m3_s=flux,face_depth_m=depth,face_velocity=u[[0,80],1:17],
                      boundary_stage=eta[[0,81],1:17],tangent_velocity=v[[0,81],:])
    with Dataset(folder/'trih-sim_P057.nc') as d:
        metadata['cross_section_discharge']=np.asarray(d['CTR'][j])
    return np.ascontiguousarray(np.vstack([h,h*uv])),np.ascontiguousarray(z),metadata


def main(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup();out=BASE/'boundary_probes_v2'/backend;out.mkdir(parents=True,exist_ok=False)
    cases={}
    for label,folder,t in [('base40',BASE,40),('base120',BASE,120),('base300',BASE,300),
                            ('base600',BASE/'long',600),('mix300',BASE/'mixing',300)]:
        s=ResultsStore(folder/'cpu/warmup');f=s.read(int(np.flatnonzero(s.times==t)[0]))
        xy=np.array([p.centroid.coords[0] for p in f.polygons])
        q,z,native=native_state(folder/'delft',xy,t)
        cases[label+'_BRAIDS']=(f.q,f.z,native)
        cases[label+'_FLOW']=(q,z,native)
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;e=owner.exterior_bed
            save(out/'OWNER.json',e.identity if hasattr(e,'identity') else owner.identity)
            oldz=owner.z.copy()
            for name,(q,z,native) in cases.items():
                owner.z[:]=xp.asarray(z);e.evaluate(xp.asarray(q),owner.z,time=40.)
                arrays=dict(q=q,z=z,xy=xy,trace=host(e.trace).copy(),flux=host(e.flux).copy(),
                            exterior_bed=host(e.p.exterior_bed).copy())
                arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','bid')})
                arrays.update({'native_'+k:v for k,v in native.items()})
                ic=e.inlet_compatibility
                arrays.update({'inlet_'+k:host(getattr(ic,k)).copy() for k in
                    ('stage','target_stage','exterior_velocity','mass_balance','branch')})
                np.savez(out/(name+'.npz'),**arrays)
            owner.z[:]=oldz
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    save(out/'project.json',project(backend))
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    result=river_worker.run(prepared,out/'construction',until=.001)
    save(out/'CHECK.json',dict(result=result,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
                             script_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.glob('*.npz')}))
    print('Boundary probes complete',backend,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('backend',choices=('cpu','gpu'));main(ap.parse_args().backend)
