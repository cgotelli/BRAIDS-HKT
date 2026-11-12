"""Section symmetry, matched-state pressure/advection and transient-wave evidence."""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import sha,save
from review_hydraulics162 import load,verify_native
from braids_desktop.results import ResultsStore
from wall_pressure170 import analytic_pressure

BASE=ROOT/'validation/_jet173'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_jet173'
G=9.81


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def shape(v,y):
    basis=np.sin(np.pi*np.asarray(y)/8)
    amplitude=float(np.dot(v,basis)/np.dot(basis,basis))
    return dict(peak=float(np.max(v)),minimum=float(np.min(v)),
                mirror_relative_rms=rms(v-v[::-1])/max(rms(v),1e-15),
                first_sine_amplitude=amplitude,nonfirst_sine_relative_rms=rms(v-amplitude*basis)/max(rms(v),1e-15))


def ledger(a,c):
    parts=np.zeros((5,3));by_axis=np.zeros((2,3));maximum=0.
    faces=np.flatnonzero((a['owner']==c)|(a['neighbour']==c))
    for f in faces:
        assert a['neighbour'][f]>=0 or a['bid'][f]<0
        side=int(a['owner'][f]!=c);sign=1-2*side;n=a['normal'][:,f]
        local=a['local'][:,f];central=[];pressure=[]
        for k in (0,1):
            h,mn,mt,_=local[4*k:4*k+4]
            uv=np.array([mn*n[0]-mt*n[1],mn*n[1]+mt*n[0]])/h
            central.append(np.r_[mn,mn*uv]);pressure.append(np.r_[0.,.5*G*h*h*n])
        adv=np.mean(central,axis=0);press=np.mean(pressure,axis=0)
        h,mn,mt,_=local[4*side:4*side+4]
        raw=np.array([mn,mn*mn/h+.5*G*h*h,mn*mt/h])+sign*a['riemann'][3*side:3*side+3,f]
        rebuilt=np.r_[raw[0],raw[1]*n[0]-raw[2]*n[1],raw[1]*n[1]+raw[2]*n[0]]
        wall=a['neighbour'][f]<0
        if wall:
            for v in (adv,press,rebuilt):v[0]=0.;v[1:]=np.dot(v[1:],n)*n
        wave=rebuilt-adv-press
        t=a['rec_trace'];corr=.5*G*(a['q'][0,c]+t[8+side,f])*(t[3+4*side,f]-a['z'][c])
        bed=np.r_[0.,corr*n]
        wall_delta=np.zeros(3)
        if wall:
            exact=np.r_[0.,analytic_pressure(local[0],local[1])*n]
            wall_delta=exact-rebuilt;rebuilt=exact
        actual=a['fl' if side==0 else 'fr'][:,f]
        maximum=max(maximum,float(abs(actual-rebuilt-bed).max()))
        pieces=-sign*a['length'][f]/a['area'][c]*np.array([adv,press,wave,bed,wall_delta])
        parts+=pieces
        by_axis[int(abs(n[1])>.5)]+=pieces.sum(0)
    assert maximum<1e-12,maximum
    assert abs(parts.sum(0)-a['rate'][:,c]).max()<1e-11
    h=a['q'][0,c];uv=a['q'][1:,c]/h
    acceleration=(parts[:,1:]-parts[:,:1]*uv)/h
    drag=-G/45**2*np.linalg.norm(uv)*uv/h
    return dict(parts=acceleration.tolist(),drag=drag.tolist(),total=(acceleration.sum(0)+drag).tolist(),
                axis_total=((by_axis[:,1:]-by_axis[:,:1]*uv)/h).tolist(),closure=maximum)


def native_terms(xy):
    path=ROOT/'validation/_corner167/native_momentum_fine'
    verify_native(path)
    ix=np.rint(xy[:,0]/.5+.5).astype(int);iy=np.rint(xy[:,1]/.5+.5).astype(int)
    with Dataset(path/'trim-sim_P057.nc') as f:
        times=np.asarray(f['time'][:]);j=int(np.argmin(abs(times-40)))
        terms={}
        for name,k in [('pressure','MOM_VPRESSURE'),('x_advection','MOM_UDVDX'),('y_advection','MOM_VDVDY'),('drag','MOM_VBEDSHEAR'),('total','MOM_DVDT')]:
            v=np.asarray(f[k][j]).reshape(82,18)
            terms[name]=.5*(v[ix,iy]+v[ix,iy-1])
        v=np.asarray(f['V1'][:,0]);v=.5*(v[:,ix,iy]+v[:,ix,iy-1])
        derivative=(3*v[j]-4*v[j-1]+v[j-2])/.02
        raw=np.asarray(f['V1'][j,0,9,0:17]);z=np.asarray(f['S1'][j,9,1:17])
    return terms,derivative,raw,z


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,meta=load(ROOT/'validation/_delivery172/package/groin_mpm_cpu/full',ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    xy=data['xy'];m=xy[:,0]==4.25;y=xy[m,1];cells=np.flatnonzero(m)
    terms,derivative,raw,eta=native_terms(xy)
    samples=[]
    for j,t in enumerate(data['store'].times):
        if 0<t<=40:samples.append(dict(time_s=float(t),BRAIDS=shape(data['bu'][j,1,m],y),FLOW=shape(data['nu'][j,1,m],y)))
    probes={};parity={}
    points=np.flatnonzero(m|((xy[:,0]>=7.75)&(xy[:,0]<=8.75)&(xy[:,1]>=2.75)&(xy[:,1]<=4.25)))
    for p in sorted((BASE/'probes/gpu').glob('*.npz')):
        a=np.load(p);c=np.load(BASE/'probes/cpu'/p.name)
        parity[p.stem]=dict(rate=float(abs(c['rate']-a['rate']).max()),water=float(abs(c['water']-a['water']).max()))
        assert parity[p.stem]['rate']<1e-11 and parity[p.stem]['water']<1e-11
        rows=[dict(cell=int(i),xy=xy[i].tolist(),**ledger(a,i)) for i in points]
        probes[p.stem]=rows
    lookup=lambda label: [r for r in probes[label] if r['xy'][0]==4.25]
    b=np.array([r['total'][1] for r in lookup('BRAIDS40')]);n=np.array([r['total'][1] for r in lookup('FLOW40')])
    crossed=np.array([r['total'][1] for r in lookup('BRAIDS40_FLOWdepth')])
    velocity_swap=np.array([r['total'][1] for r in lookup('FLOW40_BRAIDSdepth')])
    attribution=dict(own_state_rate_difference_rms=rms(b-terms['total'][m]),
                     same_FLOW_state_rate_difference_rms=rms(n-terms['total'][m]),
                     depth_swap_change_rms=rms(crossed-b),
                     FLOWdepth_BRAIDSvelocity_rate_difference_rms=rms(crossed-terms['total'][m]),
                     BRAIDSdepth_FLOWvelocity_rate_difference_rms=rms(velocity_swap-terms['total'][m]),
                     native_diagnostic_closure=rms(sum(terms[k][m] for k in ('pressure','x_advection','y_advection','drag'))-terms['total'][m]),
                     native_time_derivative_check=rms(terms['total'][m]-derivative[m]))
    save(OUT/'PROBES.json',dict(samples=samples,paired_operator=parity,probes=probes,attribution=attribution,
        native_terms={k:v[m].tolist() for k,v in terms.items()},native_derivative=derivative[m].tolist(),
        raw_native_face_v=raw.tolist(),native_eta=eta.tolist(),verified_native=meta,
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        inputs={str(p.relative_to(ROOT)):sha(p) for backend in ('cpu','gpu') for p in (BASE/'probes'/backend).glob('*.npz')}))
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for field,title,row in [('bu','BRAIDS',0),('nu','FLOW 2DH',1)]:
        for t in (20,25,30,35,40):
            j=int(np.flatnonzero(data['store'].times==t)[0]);axs[row,0].plot(y,data[field][j,1,m],label=f'{t} s')
        axs[row,0].set(title=title+' transverse velocity',xlabel='y (m)',ylabel='v (m/s)');axs[row,0].legend(fontsize=8)
    axs[0,1].plot(y,data['bh'][8,m]+data['bz'][8,m],'o-',label='BRAIDS')
    axs[0,1].plot(y,eta,'s--',label='FLOW');axs[0,1].set(title='Water surface at 40 s',xlabel='y (m)',ylabel='Elevation (m)');axs[0,1].legend()
    axs[1,1].plot(y,b,'o-',label='BRAIDS on own state')
    axs[1,1].plot(y,n,'s-',label='BRAIDS on FLOW state')
    axs[1,1].plot(y,terms['total'][m],'k--',label='FLOW native diagnostic')
    axs[1,1].set(title='Instantaneous transverse acceleration at 40 s',xlabel='y (m)',ylabel='dv/dt (m/s²)');axs[1,1].legend(fontsize=8)
    for ax in (axs[0,0],axs[1,0],axs[1,1]):ax.axhline(0,color='gray',lw=.7)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle('x = 4.25 m: transient shape and same-state operator comparison')
    fig.savefig(OUT/'UPSTREAM_ATTRIBUTION.png',dpi=155);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    r=lookup('BRAIDS40');a=np.array([s['parts'] for s in r]);drag=np.array([s['drag'] for s in r])
    for k,label in enumerate(('Central advection','Central pressure','Interface wave/source','Bed quadrature','Exact wall correction')):
        axs[0].plot(y,a[:,k,1],label=label)
    axs[0].plot(y,drag[:,1],':',label='Resistance');axs[0].plot(y,b,'k',lw=2,label='Total')
    for name,v in terms.items():axs[1].plot(y,v[m],label=name)
    for ax,title in zip(axs,('BRAIDS on its own state','FLOW on its own state')):
        ax.set(title=title,xlabel='y (m)',ylabel='Transverse acceleration (m/s²)');ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Not identical discrete terms: conservative face ledger versus staggered velocity diagnostics')
    fig.savefig(OUT/'TRANSVERSE_TERMS.png',dpi=155);plt.close(fig)
    print(json.dumps(dict(samples=samples,attribution=attribution),indent=2))


if __name__=='__main__':main()
