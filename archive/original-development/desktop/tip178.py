"""Groin-tip same-state audit and bounded known-answer input, unchanged engines."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop'),
             str(ROOT/'validation/sediment/delft3d_current')]
from stage_wall172 import save,sha
from check_jet173 import setup,document
from boundary_probe174 import native_state
from regular_corner176 import psi as base_psi, velocity as base_velocity, integral
BASE=ROOT/'validation/_tip178'
TIMES=tuple(range(200,221))


def window(r):
    """C2 compact envelope; original bounded-corner field inside r=1.1 m."""
    if r<=1.1:return 1.,0.
    if r>=1.9:return 0.,0.
    s=(r-1.1)/.8
    return 1-10*s**3+15*s**4-6*s**5,(-30*s**2+60*s**3-30*s**4)/.8


def psi(x,y):return window(np.hypot(x,y))[0]*base_psi(x,y)


def velocity(x,y):
    r=np.hypot(x,y)
    if r==0:return np.zeros(2)
    w,dw=window(r)
    return w*base_velocity(x,y)+dw*base_psi(x,y)*np.array([y,-x])/r


def average(x,y,dx=.5):
    w,e=x-dx/2,x+dx/2;s,n=y-dx/2,y+dx/2
    return np.array([integral(lambda t:psi(t,n)-psi(t,s),w,e),
                    -integral(lambda t:psi(e,t)-psi(w,t),s,n)])/dx**2


def point_advection(x,y):
    """Independent continuum acceleration at the actual staggered location."""
    d=1e-5;u=velocity(x,y)
    dx=(velocity(x+d,y)-velocity(x-d,y))/(2*d)
    dy=(velocity(x,y+d)-velocity(x,y-d))/(2*d)
    return np.array([-u[0]*dx,-u[1]*dy])


def exact_rate(x,y,dx=.5):
    """Conservative volume rate from independent compact-field face integrals."""
    def flux(x,y,axis):
        lo=(y if axis==0 else x)-dx/2
        def val(t):
            uv=velocity(x,t) if axis==0 else velocity(t,y)
            f=.5*uv[axis]*np.r_[1.,uv];f[axis+1]+=.5*9.81*.5**2
            return f
        return np.array([integral(lambda t:val(t)[k],lo,lo+dx)/dx for k in range(3)])
    return -(flux(x+dx/2,y,0)-flux(x-dx/2,y,0)+flux(x,y+dx/2,1)-flux(x,y-dx/2,1))/dx


def probe(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    from braids_desktop.results import ResultsStore
    setup();out=BASE/'probes'/backend;out.mkdir(parents=True,exist_ok=False)
    s=ResultsStore(ROOT/'validation/_corner174/cpu/warmup');cases={}
    for t in TIMES:
        f=s.read(int(np.flatnonzero(s.times==t)[0]));xy=np.array([p.centroid.coords[0] for p in f.polygons])
        cases[f'BRAIDS{t}']=(f.q,f.z,t)
        q,z,_=native_state(ROOT/'validation/_corner174/delft',xy,t)
        cases[f'FLOW{t}']=(q,z,t)
    q=np.vstack([np.full(len(xy),.5),np.zeros((2,len(xy)))])
    for i,(x,y) in enumerate(xy-[8,3]):
        if np.hypot(x,y)<2.3:q[1:,i]=.5*average(x,y)
    cases['bounded']=(q,np.zeros(len(xy)),0.)
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint;oldz=owner.z.copy()
            save(out/'OWNER.json',owner.identity);checks={}
            for name,(q,z,t) in cases.items():
                owner.z[:]=xp.asarray(z);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,1e-6,t)
                arrays=dict(q=q,z=z,xy=xy,rate=host(rate).copy(),water=host(water).copy(),
                    boundary_trace=host(owner.exterior_bed.trace).copy(),exterior_bed=host(owner.exterior_bed.p.exterior_bed).copy())
                arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                if backend=='gpu':
                    arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(name+'.npz'),**arrays)
                checks[name]=dict(theta=c.minimum_theta,minimum_depth=float(q[0].min()))
            owner.z[:]=oldz;save(out/'REPLAYS.json',checks);return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',document(backend,300.))
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'construction',until=.001)
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        script_sha256=sha(__file__),times=TIMES,production_modified=False,Delft3D_modified=False,
        scope='Independent operator replays, not continuation or a model-to-model matched state. '
              'FLOW face velocities averaged to BRAIDS cells. Bounded control is local to the tip.'))
    print('Completed',backend,len(cases),'replays',flush=True)


def native(dt):
    """Two milliseconds of native time; manufactured initial data only."""
    import runtime
    from check_bedload161 import replace,table
    parent=ROOT/'validation/_corner174/delft';out=BASE/f'known_delft_{dt:g}'
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest,name
        shutil.copy2(parent/name,out/name)
    changes=dict(Tstop='.002',Dt=str(dt),Flmap=f'0 {dt} .002',Flhis=f'0 {dt} .002',Ccofu='1000',Ccofv='1000')
    code=(out/'sim_P057.mdf').read_text()
    for k,v in changes.items():code=replace(code,k,v)
    (out/'sim_P057.mdf').write_text(code)
    (out/'mor.mor').write_text(replace((out/'mor.mor').read_text(),'MorStt','1'))
    x=(np.arange(82)-.5)*.5;y=(np.arange(18)-.5)*.5
    u=np.array([[velocity(xx+.25-8,yy-3)[0] for xx in x] for yy in y])
    v=np.array([[velocity(xx-8,yy+.25-3)[1] for xx in x] for yy in y])
    bed=np.zeros((18,82));eta=np.full_like(bed,.5)
    np.savetxt(out/'bed.dep',bed,fmt='%.17e')
    np.savetxt(out/'initial.ini',np.concatenate([eta,u,v]),fmt='%.17e')
    (out/'flume.bct').write_text(table('upstream','total discharge (t)','m3/s',0.,True)+
                               table('downstream','water elevation (z)','m',.5,True))
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(mdf=changes,bed='flat zero',initial='compact bounded corner streamfunction; exact face velocities',
                     upstream_Q=0.,downstream_stage=.5,MorStt=1),
        script_sha256=sha(__file__),Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=120)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu','delft'));p.add_argument('--dt',type=float,default=.0001)
    a=p.parse_args();native(a.dt) if a.backend=='delft' else probe(a.backend)
