"""Attribute the corner error before selecting a bounded research correction."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from tip178 import velocity,exact_rate
from quadrature176 import Flux,values_to_trace,gather
from check_jet173 import setup,document
from stage_wall172 import save,sha
BASE=ROOT/'validation/_trace179'
POINTS=([7.75,2.75],[7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75])


def linear(a):
    """Independent recomputation of the existing length/distance² LS fit."""
    q=a['q'];v=np.vstack([q[0]+a['z'],q[1:]/q[0],a['z']]);nc=q.shape[1]
    grad=np.zeros((4,2,nc));neighbours=[[] for _ in range(nc)];walls=[[] for _ in range(nc)]
    for f,(o,n) in enumerate(zip(a['owner'],a['neighbour'])):
        if n>=0:neighbours[o].append((int(n),f));neighbours[n].append((int(o),f))
        elif a['bid'][f]<0:walls[o].append(f)
    for c,ns in enumerate(neighbours):
        if len(ns)<2:continue
        d=np.array([a['xy'][n]-a['xy'][c] for n,f in ns]);w=np.array([a['length'][f] for n,f in ns])/np.sum(d*d,axis=1)
        delta=np.array([v[:,n]-v[:,c] for n,f in ns]);mat=d.T@(d*w[:,None]);rhs=d.T@(delta*w[:,None])
        if np.linalg.det(mat)>1e-12*np.trace(mat)**2:grad[:,:,c]=np.linalg.solve(mat,rhs).T
    return v,grad,walls


def faceclip(a,data,extend=False):
    v,grad,walls=linear(a);values=a['rec_face_value'].copy();lo=a['rec_vlo'].copy();hi=a['rec_vhi'].copy()
    for c,fs in enumerate(walls):
        if not fs:continue
        if extend:
            for f in fs:
                n=a['normal'][:,f];r=2*np.dot(a['mid'][:,f]-a['xy'][c],n)*n
                dv=grad[1:3,:,c]@r;dv-=np.dot(dv,n)*n
                ghost=v[1:3,c]+dv-2*np.dot(v[1:3,c],n)*n
                lo[1:3,c]=np.minimum(lo[1:3,c],ghost);hi[1:3,c]=np.maximum(hi[1:3,c],ghost)
        for side,key in enumerate(('owner','neighbour')):
            for f in np.flatnonzero(a[key]==c):
                # Change only the fallback velocity, not an available WENO trace.
                if data['weno_cells'][0,side,f]>=0:continue
                u=v[1:3,c]+grad[1:3,:,c]@(a['mid'][:,f]-a['xy'][c])
                values[1:3,side,f]=np.maximum(lo[1:3,c],np.minimum(hi[1:3,c],u))
    return values_to_trace(a,values)


def main():
    from braids_core import river_worker
    setup();BASE.mkdir(parents=True,exist_ok=False)
    a=dict(np.load(ROOT/'validation/_tip178/probes/gpu/bounded.npz'))
    ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
    truth=np.array([exact_rate(*(np.array(p)-[8,3])) for p in POINTS]).T
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);flux=Flux(o);rows={};data={}
            traces={'current':a['rec_trace'],'faceclip':faceclip(a,o.data),'faceclip_tangent':faceclip(a,o.data,True)}
            for name in ('oracle_normal','oracle_tangent','oracle_both','oracle_wall'):
                values=a['rec_face_value'].copy()
                for f in range(len(a['owner'])):
                    if np.max(abs(a['mid'][:,f]-[8,3]))>1.5:continue
                    if name=='oracle_wall' and a['neighbour'][f]>=0:continue
                    exact=velocity(*(a['mid'][:,f]-[8,3]));n=a['normal'][:,f]
                    for side in (0,1):
                        u=values[1:3,side,f];delta=exact-u
                        if name=='oracle_normal':delta=np.dot(delta,n)*n
                        elif name=='oracle_tangent':delta-=np.dot(delta,n)*n
                        values[1:3,side,f]+=delta
                traces[name]=values_to_trace(a,values)
            for name,t in traces.items():
                l,r=flux(a,t);rate=gather(a,l,r);err=rate[:,ids]-truth
                rows[name]=dict(mass_rms=float(np.sqrt(np.mean(err[0]**2))),momentum_rms=float(np.sqrt(np.mean(err[1:]**2))),
                    first_rate=rate[:,ids[2]].tolist(),truth=truth[:,2].tolist())
                data[name+'_trace']=t;data[name+'_rate']=rate
            np.savez(BASE/'ATTRIBUTION.npz',**data);save(BASE/'ATTRIBUTION.json',rows)
            print(json.dumps(rows,indent=2),flush=True);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(BASE/'project.json',document('cpu'))
    p=river_worker.prepare(BASE/'project.json',BASE/'prepared');river_worker.run(p,BASE/'construction',until=.001)


if __name__=='__main__':main()
