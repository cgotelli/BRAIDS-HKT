"""Isolated normal-only reflected constraints in the hydraulic LS fit."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
BASE=ROOT/'validation/_incident181'


def reflected_fit(a):
    """Velocity tensor LS augmented by only the normal ghost condition.

    The ghost location is the reflection of the physical cell centroid.
    Tangential ghost velocity is NOT prescribed. Each independent wall adds
    one rank-one normal constraint; Sherman--Morrison solves the joint fit.
    """
    from trace179 import linear
    v,grad,walls=linear(a);grad0=grad.copy();rows=[]
    for c,fs in enumerate(walls):
        if not fs or a['q'][0,c]<=1e-6:continue
        neighbours=[]
        for f,(o,n) in enumerate(zip(a['owner'],a['neighbour'])):
            if n<0:continue
            if o==c:neighbours.append((n,f))
            if n==c:neighbours.append((o,f))
        if len(neighbours)<2:continue
        r=np.array([a['xy'][j]-a['xy'][c] for j,f in neighbours])
        w=np.array([a['length'][f] for j,f in neighbours])/np.sum(r*r,axis=1)
        mat=r.T@(r*w[:,None])
        if np.linalg.det(mat)<=1e-12*np.trace(mat)**2:continue
        inv=np.kron(np.eye(2),np.linalg.inv(mat));g=grad[1:3,:,c].reshape(4).copy()
        for f in fs:
            n=a['normal'][:,f];d=2*np.dot(a['mid'][:,f]-a['xy'][c],n);r=d*n
            if d<=1e-12:continue
            row=np.outer(n,r).reshape(4);target=-2*np.dot(v[1:3,c],n);weight=a['length'][f]/(d*d)
            p=inv@row;den=1/weight+row@p;g+=(target-row@g)/den*p;inv-=np.outer(p,p)/den
        grad[1:3,:,c]=g.reshape(2,2)
        rows.append(dict(cell=int(c),xy=a['xy'][c].tolist(),before=grad0[1:3,:,c].tolist(),after=grad[1:3,:,c].tolist()))
    return v,grad,walls,rows


def traces(a,data):
    from quadrature176 import values_to_trace
    v,grad,walls,rows=reflected_fit(a);values=a['rec_face_value'].copy()
    for c,fs in enumerate(walls):
        if not fs:continue
        faces=np.flatnonzero((a['owner']==c)|(a['neighbour']==c));theta=np.ones(2)
        for f in faces:
            delta=grad[1:3,:,c]@(a['mid'][:,f]-a['xy'][c])
            for k in range(2):
                bound=(a['rec_vhi'][k+1,c]-v[k+1,c]) if delta[k]>0 else (a['rec_vlo'][k+1,c]-v[k+1,c])
                if abs(delta[k])>1e-30:theta[k]=min(theta[k],bound/delta[k])
        grad[1:3,:,c]*=np.maximum(0,theta)[:,None]
        for f in faces:
            side=0 if a['owner'][f]==c else 1
            slot=int(data['wall_slot'][side,f])
            if slot>=0 and a['rec_wall_ready'][slot,c]:continue
            if data['weno_cells'][0,side,f]>=0:continue
            values[1:3,side,f]=v[1:3,c]+grad[1:3,:,c]@(a['mid'][:,f]-a['xy'][c])
    return values_to_trace(a,values),rows


def main():
    import json
    from check_jet173 import setup,document
    from stage_wall172 import save
    from tip178 import exact_rate
    from regular_corner176 import rate as regular_rate
    from trace179 import POINTS
    from quadrature176 import Flux,gather
    from braids_core import river_worker
    setup();BASE.mkdir(parents=True,exist_ok=False)
    sources={'bounded':'validation/_tip178/probes/gpu/bounded.npz',
        'regular':'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz',
        'developed':'validation/_tip178/probes/gpu/BRAIDS200.npz'}
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);flux=Flux(o);summary={}
            for name,path in sources.items():
                a=dict(np.load(ROOT/path));t,rows=traces(a,o.data);l,r=flux(a,t);rate=gather(a,l,r)
                arrays=dict(a);arrays.update(rec_trace=t,fl=l,fr=r,rate=rate);np.savez(BASE/(name+'.npz'),**arrays)
                ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
                result=dict(rates=rate[:,ids].T.tolist(),gradients=[r for r in rows if r['cell'] in ids])
                if name!='developed':
                    fun=exact_rate if name=='bounded' else regular_rate
                    true=np.array([fun(*(np.array(p)-[8,3])) for p in POINTS]).T;e=rate[:,ids]-true
                    result.update(mass_rms=float(np.sqrt(np.mean(e[0]**2))),momentum_rms=float(np.sqrt(np.mean(e[1:]**2))))
                summary[name]=result
            save(BASE/'PROTOTYPE.json',summary);print(json.dumps(summary,indent=2),flush=True);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(BASE/'project.json',document('cpu'))
    p=river_worker.prepare(BASE/'project.json',BASE/'prepared');river_worker.run(p,BASE/'construction',until=.001)


if __name__=='__main__':main()
