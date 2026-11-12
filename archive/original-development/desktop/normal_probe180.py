"""Isolate missing reflected normal-velocity stencils without changing tangents."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from pair180 import BASE
from check_jet173 import setup,document
from stage_wall172 import save
from quadrature176 import Flux,gather,values_to_trace
from tip178 import exact_rate
from regular_corner176 import rate as regular_rate
from trace179 import POINTS
from axis_stencil176 import geometry


def weno(v):
    p=np.array([(2*v[0]-7*v[1]+11*v[2])/6,(-v[1]+5*v[2]+2*v[3])/6,(2*v[2]+5*v[3]-v[4])/6])
    a=np.array([v[0]-2*v[1]+v[2],v[1]-2*v[2]+v[3],v[2]-2*v[3]+v[4]])
    c=np.array([v[0]-4*v[1]+3*v[2],v[1]-v[3],3*v[2]-4*v[3]+v[4]])
    b=13/12*a*a+.25*c*c;eps=1e-40+1e-14*np.max(v*v)
    weights=np.array([.1,.6,.3])*(1+(abs(b[0]-b[2])/(b+eps))**2)
    return np.dot(weights,p)/weights.sum()


def normal_trace(a,data):
    fields,changed=geometry(data);v=a['rec_face_value'].copy();count=0
    for c,axis in changed:
        ids=fields['wall_stencil_ids'][:,axis,c];kind=fields['wall_stencil_kind'][:,axis,c]
        if np.any(a['q'][0,ids]<=.001):continue
        stencil=a['q'][axis+1,ids]/a['q'][0,ids]*(1-kind)
        for side,key in enumerate(('owner','neighbour')):
            for f in np.flatnonzero(a[key]==c):
                if abs(a['normal'][axis,f])<1-1e-12:continue
                if a['neighbour'][f]<0 and a['bid'][f]>=0:continue
                positive=a['mid'][axis,f]>a['xy'][c,axis]
                val=weno(stencil if positive else stencil[::-1])
                v[axis+1,side,f]=np.clip(val,a['rec_vlo'][axis+1,c],a['rec_vhi'][axis+1,c]);count+=1
    return values_to_trace(a,v),count


def main():
    from braids_core import river_worker
    setup();out=BASE/'normal_only';out.mkdir(exist_ok=False)
    data={'known':dict(np.load(ROOT/'validation/_tip178/probes/gpu/bounded.npz')),
          'regular':dict(np.load(ROOT/'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz')),
          'developed':dict(np.load(ROOT/'validation/_tip178/probes/gpu/BRAIDS200.npz'))}
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);flux=Flux(o);summary={}
            fields,changed=geometry(o.data);np.savez(out/'GEOMETRY.npz',**fields,changed=np.array(changed))
            for name,a in data.items():
                t,count=normal_trace(a,o.data);l,r=flux(a,t);rate=gather(a,l,r)
                arrays=dict(a);arrays.update(rec_trace=t,fl=l,fr=r,rate=rate);np.savez(out/(name+'.npz'),**arrays)
                ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
                row=dict(rates=rate[:,ids].T.tolist(),trace_values_changed=count)
                if name!='developed':
                    fun=exact_rate if name=='known' else regular_rate
                    true=np.array([fun(*(np.array(p)-[8,3])) for p in POINTS]).T;e=rate[:,ids]-true
                    row.update(mass_rms=float(np.sqrt(np.mean(e[0]**2))),momentum_rms=float(np.sqrt(np.mean(e[1:]**2))))
                summary[name]=row
            save(out/'CHECK.json',summary);print(json.dumps(summary,indent=2),flush=True);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',document('cpu'))
    p=river_worker.prepare(out/'project.json',out/'prepared');river_worker.run(p,out/'construction',until=.001)


if __name__=='__main__':main()
