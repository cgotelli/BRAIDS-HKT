"""Research wall-normal quadratic with original bounds and conservative mean."""
import numpy as np


def geometry(a):
    """One full straight wall and opposite full fluid face; no cropped admission."""
    nc=len(a['area']);ids=np.full(nc,-1,np.int32);axis=np.full(nc,-1,np.int32);sign=np.zeros(nc)
    for f in np.flatnonzero((a['neighbour']<0)&(a['bid']<0)):
        c=int(a['owner'][f]);n=a['normal'][:,f];j=int(np.argmax(abs(n)));dx=a['length'][f]
        if abs(n[j])<1-1e-12 or abs(a['area'][c]-dx*dx)>1e-10:continue
        if np.max(abs(a['mid'][:,f]-a['xy'][c]-.5*dx*n))>1e-10:continue
        fs=np.flatnonzero((a['owner']==c)|(a['neighbour']==c))
        if len(fs)!=4 or np.sum((a['neighbour'][fs]<0)&(a['bid'][fs]<0))!=1:continue
        otherface=[ff for ff in fs if np.max(abs(a['mid'][:,ff]-a['xy'][c]+.5*dx*n))<1e-10]
        if len(otherface)!=1:continue
        ff=otherface[0];r=int(a['neighbour'][ff] if a['owner'][ff]==c else a['owner'][ff])
        if r<0 or abs(a['area'][r]-dx*dx)>1e-10 or np.max(abs(a['xy'][r]-a['xy'][c]+dx*n))>1e-10:continue
        ids[c]=r;axis[c]=j;sign[c]=n[j]
    return ids,axis,sign


def trace(a,wall_only=False):
    from quadrature176 import values_to_trace
    ids,axis,sign=geometry(a);values=a['rec_face_value'].copy();rows=[]
    for c in np.flatnonzero(ids>=0):
        r=ids[c];j=axis[c];other=1-j;sigma=sign[c];dx=np.sqrt(a['area'][c]);h=a['q'][0,c]
        if min(h,a['q'][0,r])<=1e-6:continue
        p=a['q'][1+j,c];p1=a['q'][1+j,r]
        aa=3.5*p-.5*p1;bb=.75*p1-2.25*p
        # s is inward distance / dx. Tangential term has zero cell mean and
        # vanishes on the wall; it reproduces the maintained limited slope.
        cross=2*dx*(h*a['rec_grad'][1+j,other,c]+p/h*(a['rec_grad'][0,other,c]-a['rec_grad'][3,other,c]))
        fs=np.flatnonzero((a['owner']==c)|(a['neighbour']==c));theta=1.;target=[]
        for f in fs:
            side=0 if a['owner'][f]==c else 1
            if a['neighbour'][f]<0 and a['bid'][f]>=0:theta=0.;break
            disp=(a['mid'][:,f]-a['xy'][c])/dx;s=.5-sigma*disp[j]
            hf=a['rec_trace'][4*side,f]
            if hf<=1e-6:theta=0.;break
            poly=aa*s+bb*s*s+cross*s*disp[other]
            delta=(poly-p)/hf;centre=p/hf
            lo=min(a['rec_vlo'][j+1,c],centre);hi=max(a['rec_vhi'][j+1,c],centre)
            if delta>0:theta=min(theta,(hi-centre)/delta)
            elif delta<0:theta=min(theta,(lo-centre)/delta)
            target.append((f,side,hf,poly))
        theta=max(0,theta)
        for f,side,hf,poly in target:
            if wall_only and a['neighbour'][f]>=0:continue
            values[1+j,side,f]=(p+theta*(poly-p))/hf
        rows.append(dict(cell=int(c),xy=a['xy'][c].tolist(),theta=theta,normal_momentum=p,coefficients=[aa,bb,cross]))
    return values_to_trace(a,values),rows


def main():
    import json
    from incident181 import ROOT,BASE
    from check_jet173 import setup,document
    from stage_wall172 import save
    from tip178 import exact_rate
    from regular_corner176 import rate as regular_rate
    from trace179 import POINTS
    from quadrature176 import Flux,gather
    from braids_core import river_worker
    setup();out=BASE/'polynomial';out.mkdir(exist_ok=False)
    sources={'bounded':'validation/_tip178/probes/gpu/bounded.npz','regular':'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz','developed':'validation/_tip178/probes/gpu/BRAIDS200.npz'}
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);flux=Flux(o);summary={}
            for name,path in sources.items():
                a=dict(np.load(ROOT/path));t,rows=trace(a);l,r=flux(a,t);rate=gather(a,l,r)
                arrays=dict(a);arrays.update(rec_trace=t,fl=l,fr=r,rate=rate);np.savez(out/(name+'.npz'),**arrays)
                ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
                result=dict(rates=rate[:,ids].T.tolist(),polynomial=[r for r in rows if r['cell'] in ids])
                if name!='developed':
                    fun=exact_rate if name=='bounded' else regular_rate
                    true=np.array([fun(*(np.array(p)-[8,3])) for p in POINTS]).T;e=rate[:,ids]-true
                    result.update(mass_rms=float(np.sqrt(np.mean(e[0]**2))),momentum_rms=float(np.sqrt(np.mean(e[1:]**2))))
                summary[name]=result
            save(out/'CHECK.json',summary);print(json.dumps(summary,indent=2),flush=True);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',document('cpu'))
    p=river_worker.prepare(out/'project.json',out/'prepared');river_worker.run(p,out/'construction',until=.001)


if __name__=='__main__':main()
