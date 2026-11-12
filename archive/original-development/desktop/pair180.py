"""Paired normal-state diagnostic through the unchanged augmented f-wave solver."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import setup,document
from stage_wall172 import save
from quadrature176 import Flux,gather,values_to_trace
from tip178 import velocity,exact_rate
from trace179 import POINTS
BASE=ROOT/'validation/_pair180'


def paired(a,mode='froude'):
    t=a['rec_trace'].copy();inside=a['neighbour']>=0
    inside&=(t[0]>.001)&(t[4]>.001)
    # No walls/open boundaries or dry-side reconstruction is changed.
    ids=np.flatnonzero(inside);n=a['normal'][:,ids]
    vl=t[1:3,ids]/t[0,ids];vr=t[5:7,ids]/t[4,ids]
    ul=np.sum(vl*n,axis=0);ur=np.sum(vr*n,axis=0)
    if mode=='froude':
        theta=np.minimum(1.,np.maximum(np.linalg.norm(vl,axis=0)/np.sqrt(9.81*t[0,ids]),np.linalg.norm(vr,axis=0)/np.sqrt(9.81*t[4,ids])))
    elif mode=='central':theta=np.zeros(len(ids))
    else:raise ValueError(mode)
    delta=.5*(1-theta)*(ur-ul)
    t[1:3,ids]+=t[0,ids]*delta*n;t[5:7,ids]-=t[4,ids]*delta*n
    return t


def main():
    from braids_core import river_worker
    setup();BASE.mkdir(parents=True,exist_ok=False)
    data={k:dict(np.load(ROOT/f'validation/_tip178/probes/gpu/{f}.npz')) for k,f in [('known','bounded'),('developed','BRAIDS200')]}
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);flux=Flux(o);summary={}
            for label,a in data.items():
                ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
                traces={'baseline':a['rec_trace'],'froude':paired(a),'central':paired(a,'central')}
                if label=='known':
                    # Diagnostic only: exact normal velocities on interior faces,
                    # independently of the wall-state error.
                    v=a['rec_face_value'].copy()
                    for f in np.flatnonzero(a['neighbour']>=0):
                        if np.max(abs(a['mid'][:,f]-[8,3]))>1.5:continue
                        ex=velocity(*(a['mid'][:,f]-[8,3]));n=a['normal'][:,f]
                        for side in (0,1):v[1:3,side,f]+=np.dot(ex-v[1:3,side,f],n)*n
                    traces['oracle_interior_normal']=values_to_trace(a,v)
                rows={}
                for mode,t in traces.items():
                    l,r=flux(a,t);rate=gather(a,l,r);arr=dict(a);arr.update(fl=l,fr=r,rate=rate,rec_trace=t)
                    np.savez(BASE/f'{label}_{mode}.npz',**arr)
                    result=dict(rates=rate[:,ids].T.tolist())
                    if label=='known':
                        true=np.array([exact_rate(*(np.array(p)-[8,3])) for p in POINTS]).T;e=rate[:,ids]-true
                        result.update(mass_rms=float(np.sqrt(np.mean(e[0]**2))),momentum_rms=float(np.sqrt(np.mean(e[1:]**2))))
                    rows[mode]=result
                summary[label]=rows
            save(BASE/'ATTRIBUTION.json',summary);print(json.dumps(summary['known'],indent=2),flush=True);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(BASE/'project.json',document('cpu'))
    p=river_worker.prepare(BASE/'project.json',BASE/'prepared');river_worker.run(p,BASE/'construction',until=.001)


if __name__=='__main__':main()
