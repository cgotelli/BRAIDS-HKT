"""Bounded, divergence-free slip-corner control, not a steady SWE solution.

The 174 potential field has unbounded velocity at the mathematical vertex.
This independent control replaces r^(2/3) in its streamfunction by
r^2/(r^2+epsilon^2)^(2/3), epsilon=1 m. The velocity is continuous and zero
at the vertex. This is an explicitly manufactured test, not a physical
regularization of the production groin or a fit to FLOW.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from check_jet173 import setup,document

EPSILON=1.


def psi(x,y):
    r=np.hypot(x,y);t=np.mod(np.arctan2(y,x),2*np.pi)
    return .75*r*r/(r*r+EPSILON**2)**(2/3)*np.sin(2*t/3)


def velocity(x,y):
    r=np.hypot(x,y);t=np.mod(np.arctan2(y,x),2*np.pi)
    if r==0:return np.zeros(2)
    f=r*r/(r*r+EPSILON**2)**(2/3)
    df=2*r/(r*r+EPSILON**2)**(2/3)-(4/3)*r**3/(r*r+EPSILON**2)**(5/3)
    return .75*np.array([df*np.sin(t)*np.sin(2*t/3)+(2/3)*f/r*np.cos(t)*np.cos(2*t/3),
                        -df*np.cos(t)*np.sin(2*t/3)+(2/3)*f/r*np.sin(t)*np.cos(2*t/3)])


def integral(fn,lo,hi):return quad(fn,lo,hi,epsabs=1e-12,epsrel=1e-12)[0]


def mean_velocity(x,y,dx=.5):
    w,e=x-dx/2,x+dx/2;s,n=y-dx/2,y+dx/2
    return np.array([integral(lambda t:psi(t,n)-psi(t,s),w,e),
                    -integral(lambda t:psi(e,t)-psi(w,t),s,n)])/dx**2


def flux(x,y,axis,dx=.5):
    lo=(y if axis==0 else x)-dx/2;hi=lo+dx
    def f(t):
        u=velocity(x,t) if axis==0 else velocity(t,y)
        z=.5*u[axis]*np.r_[1.,u];z[axis+1]+=.5*9.81*.5**2
        return z
    return np.array([integral(lambda t:f(t)[k],lo,hi)/dx for k in range(3)])


def rate(x,y,dx=.5):
    return -(flux(x+dx/2,y,0,dx)-flux(x-dx/2,y,0,dx)
             +flux(x,y+dx/2,1,dx)-flux(x,y-dx/2,1,dx))/dx


def main(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup();out=ROOT/'validation/_quadrature176/regular_v2'/backend;out.mkdir(parents=True,exist_ok=False)
    a=dict(np.load(ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz'))
    xy=a['xy'];local=xy-[8,3];mask=(abs(local[:,0])<2)&(abs(local[:,1])<2)
    q=a['q'].copy();z=a['z'].copy()
    for c in np.flatnonzero(mask):q[:,c]=np.r_[.5,.5*mean_velocity(*local[c])]
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint;old=owner.z.copy()
            owner.z[:]=xp.asarray(z);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
            residual,water=c.rate(c.interval_base,1e-6,40.)
            arrays=dict(q=q,z=z,xy=xy,rate=host(residual).copy(),water=host(water).copy())
            arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
            if backend=='gpu':
                arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                arrays.update({'geo_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.geometry.items()})
            np.savez(out/'RESIDUAL.npz',**arrays)
            owner.z[:]=old;return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    save(out/'project.json',document(backend));p=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(p,out/'construction',until=.001)
    v=np.load(out/'RESIDUAL.npz');records=[]
    for point in ([7.75,2.75],[7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75]):
        c=int(np.flatnonzero(np.all(xy==point,axis=1))[0]);truth=rate(*(np.array(point)-[8,3]))
        records.append(dict(xy=point,expected_rate=truth.tolist(),actual_rate=v['rate'][:,c].tolist()))
    save(out/'CHECK.json',dict(status=status,records=records,epsilon_m=EPSILON,scope='bounded instantaneous manufactured field, not steady flow',
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),source_sha256=sha(__file__),production_changed=False,Delft3D_changed=False))
    print(records)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu');main(ap.parse_args().backend)
