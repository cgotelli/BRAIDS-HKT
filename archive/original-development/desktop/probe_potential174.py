"""Compare the true corner-volume residual with an analytic potential field."""
import json
from pathlib import Path
import sys
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_corner174 import setup,project,BASE
from stage_wall172 import save,sha
from corner_exact174 import mean_velocity,mean_rate,projected_point_acceleration


def main(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    from braids_desktop.results import ResultsStore
    setup();out=BASE/'potential'/backend;out.mkdir(parents=True,exist_ok=False)
    f=ResultsStore(ROOT/'validation/_jet173/fixed/cpu/run').read(0)
    xy=np.array([p.centroid.coords[0] for p in f.polygons]);local=xy-[8,3]
    patch=(abs(local[:,0])<2)&(abs(local[:,1])<2)
    q=f.q.copy();z=f.z.copy();z[patch]=0;q[0,patch]=.5
    for c in np.flatnonzero(patch):q[1:,c]=.5*mean_velocity(*local[c])
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint;oldz=owner.z.copy()
            owner.z[:]=xp.asarray(z);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
            rate,water=c.rate(c.interval_base,1e-6,40.)
            arrays=dict(q=q,z=z,xy=xy,rate=host(rate).copy(),water=host(water).copy())
            arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
            if backend=='gpu':
                arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
            np.savez(out/'RESIDUAL.npz',**arrays);owner.z[:]=oldz
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',project(backend))
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'construction_run',until=.001)
    a=np.load(out/'RESIDUAL.npz');rows=[]
    for point in ([7.75,2.75],[7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75]):
        c=int(np.flatnonzero(np.all(xy==point,axis=1))[0]);uv=q[1:,c]/q[0,c];exact=mean_rate(*local[c])
        acc=(a['rate'][1:,c]-uv*a['rate'][0,c])/q[0,c]
        rows.append(dict(xy=point,exact_rate=exact.tolist(),braids_rate=a['rate'][:,c].tolist(),
            exact_acceleration=(exact[1:]/.5).tolist(),braids_acceleration=acc.tolist(),
            projected_exact_point_acceleration=projected_point_acceleration(*local[c]).tolist()))
    save(out/'CHECK.json',dict(status=status,rows=rows,
        scope='Instantaneous constant-depth potential corner control, not steady SWE or a FLOW trajectory',
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        sources={p.name:sha(p) for p in (Path(__file__),ROOT/'desktop/corner_exact174.py')}))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu');a=ap.parse_args();main(a.backend)
