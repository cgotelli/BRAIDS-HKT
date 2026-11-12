"""Frozen-state flux and energy accounting for zero-viscosity groin flow."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import document,setup
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore
BASE=ROOT/'validation/_zero175'


def main(backend,spatial):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup()
    if spatial=='minmod':
        from reconstruction175 import install
        install(backend)
    out=BASE/'probes_v2'/spatial/backend;out.mkdir(parents=True,exist_ok=False)
    s=ResultsStore(ROOT/'validation/_corner174/cpu/warmup')
    cases={f'base{t}':s.read(int(np.flatnonzero(s.times==t)[0])) for t in (40,80,120,200,300)}
    xy=np.array([p.centroid.coords[0] for p in cases['base40'].polygons])
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint
            save(out/'OWNER.json',owner.identity)
            oldz=owner.z.copy()
            for name,f in cases.items():
                q=np.ascontiguousarray(f.q);owner.z[:]=xp.asarray(f.z)
                c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,1e-6,float(name[4:]))
                arrays=dict(q=q,z=f.z,xy=xy,rate=host(rate).copy(),water=host(water).copy(),
                    boundary_trace=host(owner.exterior_bed.trace).copy(),
                    exterior_bed=host(owner.exterior_bed.p.exterior_bed).copy())
                arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                if backend=='gpu':
                    arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(name+'.npz'),**arrays)
            owner.z[:]=oldz
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    save(out/'project.json',document(backend))
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'construction',until=.001)
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        script_sha256=sha(__file__),spatial=spatial,files={p.name:sha(p) for p in out.glob('*.npz')}))


def energy(a):
    """Exact discrete energy work, not a proof of high-order entropy stability."""
    q=a['q'];z=a['z'];uv=q[1:]/q[0];g=9.81
    w=np.vstack([g*(q[0]+z)-.5*np.sum(uv*uv,axis=0),uv])
    owner=a['owner'];neighbour=a['neighbour'];n=a['normal'];length=a['length']
    interior=neighbour>=0;wall=(neighbour<0)&(a['bid']<0);opened=(neighbour<0)&~wall
    # psi = w.F-G = g*h^2*u_n/2. Per-cell sum of its face contributions is zero.
    psi=.5*g*q[0,owner]*np.sum(q[1:,owner]*n,axis=0)
    terms=-np.sum(w[:,owner]*a['fl'],axis=0)+psi
    r=neighbour[interior]
    terms[interior]+=np.sum(w[:,r]*a['fr'][:,interior],axis=0)-.5*g*q[0,r]*np.sum(q[1:,r]*n[:,interior],axis=0)
    # Exterior entropy flux uses the actual boundary state and exterior bed.
    # Interior bed-source work remains in the distinct left/right fluxes.
    tr=a['boundary_trace'][:,opened];u=tr[1:]/tr[0]
    G=(g*(tr[0]+a['exterior_bed'][opened])+.5*np.sum(u*u,axis=0))*a['fl'][0,opened]
    boundary_G=np.sum(G*length[opened])
    terms[opened]+=G
    terms*=length
    derivative=float(np.sum(w*a['rate']*a['area']))
    closure=float(terms.sum()-derivative-boundary_G)
    assert abs(closure)<1e-10,closure
    return dict(energy_rate=derivative,boundary_energy_out=float(boundary_G),
        net_spatial_production=float(terms.sum()),closure=closure,
        interior=float(terms[interior].sum()),walls=float(terms[wall].sum()),open_boundaries=float(terms[opened].sum()),
        positive_interior=float(np.maximum(terms[interior],0).sum()),
        negative_interior=float(np.minimum(terms[interior],0).sum()),
        maximum_face_production=float(terms.max()),maximum_face_xy=a['mid'][:,terms.argmax()].tolist()),terms


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    ap.add_argument('--spatial',choices=('baseline','minmod'),default='baseline');a=ap.parse_args()
    main(a.backend,a.spatial)
