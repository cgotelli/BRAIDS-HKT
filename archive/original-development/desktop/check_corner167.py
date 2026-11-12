"""Hydraulic corner residual/trace audit on the current portable solver."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from braids_core import river_worker
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.bed_columns_v101 import host
from check_bedload161 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--variant',default='baseline');ap.add_argument('--tag');a=ap.parse_args()
    out=ROOT/f'validation/_corner167/{a.tag or a.variant}/{a.backend}'
    out.mkdir(parents=True,exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    if a.variant!='baseline':
        from corner_candidate167 import install as candidate
        candidate(a.backend,a.variant)
    base=np.load(ROOT/'validation/_scour166/v3/mpm_cpu/BRAIDS.npz')
    native=np.load(ROOT/'validation/_scour166/v3/mpm_cpu/NATIVE.npz')
    xy=base['xy'];h=native['h']
    cases={'BRAIDS':base['q'],'FLOW_PRIMARY':np.vstack([h,h*native['primary']])}
    # Smooth local test above y=3 with exact slip wall. Evaluate only faces
    # inside x=8..10; the global groin has other walls not obeyed by this field.
    cases['AFFINE']=np.array([np.full(len(xy),.5),.5*(1.+.2*(xy[:,0]-9.)),.5*.3*(xy[:,1]-3.)])
    # u has nonzero tangential shear normal to the wall, which slip alone must
    # not erase. v vanishes at the wall; no no-slip condition is prescribed.
    cases['SHEAR']=np.array([np.full(len(xy),.5),.5*(1.+.2*(xy[:,1]-3.)),.5*.3*(xy[:,1]-3.)])
    cases['REST']=np.array([.6-base['z'],np.zeros(len(xy)),np.zeros(len(xy))])
    cases['QUADRATIC']=np.array([np.full(len(xy),.5),.5*(1.+.2*(xy[:,0]-9.)),.5*.3*(xy[:,1]-3.)**2])
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);p=owner.physics;xp=owner.xp;c=owner.midpoint
            if a.variant!='baseline':
                from corner_candidate167 import configure
                configure(owner,a.variant)
            for label,q in cases.items():
                z=base['z'].copy() if label in ('BRAIDS','FLOW_PRIMARY','REST') else np.zeros(p.n)
                owner.z[:]=xp.asarray(z)
                c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,.0001,40.)
                data=dict(q=q,z=z,xy=xy,rate=host(rate).copy(),water=host(water).copy())
                data.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                if a.backend=='gpu':
                    data.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    data.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                    data.update({'geo_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.geometry.items()})
                else:
                    trace=np.zeros((2,4,p.nf));grad=np.zeros((4,2,p.n));flags=np.zeros((3,2,p.nf),np.int32)
                    fun=owner.hydraulics.lib.bend_wall_traces
                    fun.argtypes=[ctypes.c_void_p]*5;fun.restype=None
                    qq=np.ascontiguousarray(q)
                    fun(*(v.ctypes.data for v in (qq,z,grad,trace,flags)))
                    data.update(trace=trace,grad=grad,flags=flags)
                np.savez(out/(label+'.npz'),**data)
            owner.z[:]=xp.asarray(base['z'])
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    try:
        source=ROOT/f'validation/_shoal165/groin/mpm_{a.backend}/project.json'
        prepared=river_worker.prepare(source,out/'prepared')
        status=river_worker.run(prepared,out/'construction_run',until=.001)
        save(out/'AUDIT.json',dict(variant=a.variant,backend=a.backend,status=status,
             runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),script_sha256=sha(__file__),
             outputs={p.name:sha(p) for p in out.glob('*.npz')}))
    finally:river_worker.core.factory=original


if __name__=='__main__':main()
