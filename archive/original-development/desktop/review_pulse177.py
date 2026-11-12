"""Characteristic separation, known-answer reflections and matched controls."""
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
from pulse177 import BASE,H,U,C,G,DX,AMP,SIGMA,END,initial,definition
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_pulse177'


def load_braids(case,backend):
    folder=BASE/case/backend;store=ResultsStore(folder/'run')
    frames=[store.read(i) for i in range(len(store.times))]
    xy=np.array([p.centroid.coords[0] for p in frames[0].polygons])
    x=np.unique(xy[:,0]);h=np.array([f.q[0] for f in frames]);uv=np.array([f.q[1:]/f.q[0] for f in frames])
    z=np.array([f.z for f in frames]);assert abs(z).max()<1e-12
    rows=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
    return dict(t=store.times,x=x,h=np.stack([h[:,xy[:,0]==a].mean(1) for a in x],1),
        u=np.stack([uv[:,0,xy[:,0]==a].mean(1) for a in x],1),
        transverse_spread_h_m=max(float(np.ptp(h[:,xy[:,0]==a],axis=1).max()) for a in x),
        transverse_velocity_max_m_s=float(abs(uv[:,1]).max()),
        water_budget_m3=max(abs(r['water_residual_m3']) for r in rows),
        initial_q=frames[0].q,xy=xy)


def load_native(case):
    folder=BASE/case/'delft'
    identity=json.loads((folder/'execution.json').read_text());assert identity['success']
    for row in identity['outputs']:assert sha(folder/row['name'])==row['sha256']
    with Dataset(folder/'trih-sim_P057.nc') as history:
        integral=np.asarray(history['FLTR'][:]);net=integral[:,0]-integral[:,1]
    with Dataset(folder/'trim-sim_P057.nc') as d:
        h=np.asarray(d['S1'][:,1:81,1:17])+np.asarray(d['DPS'][:,1:81,1:17])
        ufaces=np.asarray(d['U1'][:,0,0:81,1:17]);u=.5*(ufaces[:,:-1]+ufaces[:,1:])
        v=np.asarray(d['V1'][:,0,1:81,1:17])
        x=np.asarray(d['XZ'][1:81,4]);t=np.asarray(d['time'][:])
        assert np.array_equal(x,(np.arange(80)+.5)*DX)
        assert np.all(np.asarray(d['KCS'][1:81,1:17])==1)
        return dict(t=t,x=x,h=h.mean(2),u=u.mean(2),
            transverse_spread_h_m=float(np.ptp(h,axis=2).max()),
            transverse_velocity_max_m_s=float(abs(v).max()),
            water_budget_m3=float(abs((h-h[0]).sum((1,2))*DX*DX-net).max()),
            ufaces=ufaces.mean(2),identity=identity)


def analytic(x,t,case,boundary=40.):
    """Linearized SWE, first reflection; cell-averaged perturbations."""
    nodes,weights=np.polynomial.legendre.leggauss(16)
    xx=np.asarray(x)[None,:,None]+DX/2*nodes;tt=np.asarray(t)[:,None,None]
    plus,minus=U+C,U-C
    f=lambda s:AMP*np.exp(-.5*(s/SIGMA)**2)
    if case=='right':
        aplus=f(xx-plus*tt-25)
        aminus=-f(boundary-plus*(tt-(xx-boundary)/minus)-25)
    elif case=='left':
        aminus=f(xx-minus*tt-15)
        aplus=(C-U)/(C+U)*f(minus*(xx/plus-tt)-15)
    else:
        aplus=aminus=np.zeros((len(t),len(x),len(nodes)))
    return aplus@weights/2,aminus@weights/2


def characteristics(data,control):
    dh=data['h']-control['h'];du=data['u']-control['u']
    return .5*(dh+H/C*du),.5*(dh-H/C*du)


def peaks(t,incident,reflected,case,x):
    if case=='right':
        ti=(x-25)/(U+C);tr=(40.-25)/(U+C)+(x-40.)/(U-C)
        sigma=SIGMA/(U+C);sign=-1
    else:
        ti=(x-15)/(U-C);tr=-15/(U-C)+x/(U+C)
        sigma=SIGMA/(C-U);sign=1
    im=np.abs(t-ti)<3*sigma;rm=np.abs(t-tr)<3*sigma
    i=np.flatnonzero(im)[np.argmax(incident[im])]
    r=np.flatnonzero(rm)[np.argmax(sign*reflected[rm])]
    energy_factor=(C+U)/(C-U) if case=='left' else (C-U)/(C+U)
    energy=energy_factor*np.trapezoid(reflected[rm]**2,t[rm])/np.trapezoid(incident[im]**2,t[im])
    return dict(incident_m=float(incident[i]),reflected_m=float(reflected[r]),
        signed_amplitude_ratio=float(reflected[r]/incident[i]),
        return_to_incident_characteristic_energy=float(energy),
        incident_peak_s=float(t[i]),reflected_peak_s=float(t[r]),
        expected_incident_s=float(ti),expected_reflected_s=float(tr))


def operators():
    results={}
    for backend in ('cpu','gpu'):
        for label,sign,x in (('inlet',-1,0.),('outlet',1,40.)):
            plus=np.load(BASE/'operators'/backend/f'{1e-5:+.0e}_{sign:+d}.npz')
            minus=np.load(BASE/'operators'/backend/f'{-1e-5:+.0e}_{sign:+d}.npz')
            select=(plus['mid'][0]==x)&(plus['bid']>=0)&(plus['neighbour']<0)
            a,b=[z['trace'][:,select].mean(1) for z in (plus,minus)]
            dh=(a[0]-b[0])/2e-5;du=(a[1]/a[0]-b[1]/b[0])/2e-5
            ap,am=.5*(dh+H/C*du),.5*(dh-H/C*du)
            incoming,outgoing=(am,ap) if label=='inlet' else (ap,am)
            result=dict(outgoing_characteristic_derivative=float(incoming),
                returned_characteristic_derivative=float(outgoing),
                reflected_to_incident=float(outgoing/incoming),
                total_discharge_derivative=float(((plus['flux'][0,select]-minus['flux'][0,select])*plus['length'][select]).sum()/2e-5))
            results[backend+'_'+label]=result
    for name in (BASE/'operators/cpu').glob('*.npz'):
        c=np.load(name);g=np.load(BASE/'operators/gpu'/name.name)
        assert np.array_equal(c['trace'],g['trace']) and np.array_equal(c['flux'],g['flux'])
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={backend:{case:(load_native(case) if backend=='FLOW' else load_braids(case,backend))
                   for case in ('base','left','right')} for backend in ('cpu','gpu','FLOW')}
    result=dict(definition={c:definition(c) for c in ('base','left','right')},
        production_changed=False,Delft3D_modified=False,
        baselines={},pulses={},backend={},initial={},operators=operators(),source_sha256=sha(__file__))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for backend,sets in data.items():
        b=sets['base'];result['baselines'][backend]=dict(
            max_depth_drift_m=float(abs(b['h']-H).max()),max_velocity_drift_m_s=float(abs(b['u']-U).max()),
            transverse_depth_spread_m=b['transverse_spread_h_m'],transverse_velocity_m_s=b['transverse_velocity_max_m_s'])
        result['baselines'][backend]['water_budget_m3']=b['water_budget_m3']
        assert abs(b['h']-H).max()<.0001,backend+' uniform control failed'
        for row,case in enumerate(('left','right')):
            d=sets[case];p,m=characteristics(d,b)
            j=int(np.argmin(abs(d['x']-(8.25 if case=='left' else 32.25))))
            incident,reflected=(m[:,j],p[:,j]) if case=='left' else (p[:,j],m[:,j])
            result['pulses'][backend+'_'+case]=peaks(d['t'],incident,reflected,case,d['x'][j])
            ep,em=analytic(d['x'],d['t'],case)
            result['pulses'][backend+'_'+case].update(
                depth_analytic_rms_m=float(np.sqrt(np.mean((p+m-ep-em)**2))),
                depth_analytic_max_m=float(abs(p+m-ep-em).max()),
                water_budget_m3=d.get('water_budget_m3'))
            result['initial'][backend+'_'+case]=dict(
                max_depth_vs_definition_m=float(abs(d['h'][0]-initial(d['x'],case)[0]).max()))
            axes[row,0].plot(d['t'],incident*1000,label=backend,ls='--' if backend=='gpu' else '-')
            axes[row,1].plot(d['t'],reflected*1000,label=backend,ls='--' if backend=='gpu' else '-')
    for row,case in enumerate(('left','right')):
        d=data['cpu'][case];j=int(np.argmin(abs(d['x']-(8.25 if case=='left' else 32.25))))
        ep,em=analytic(d['x'],d['t'],case)
        for ax,y in zip(axes[row],(em[:,j],ep[:,j]) if case=='left' else (ep[:,j],em[:,j])):
            ax.plot(d['t'],1000*y,'k:',label='Linear SWE');ax.axhline(0,color='gray',lw=.6)
            ax.set(xlabel='Time (s)',ylabel='Characteristic surface amplitude (mm)');ax.legend();ax.grid(alpha=.2)
        axes[row,0].set_title(('Inlet' if case=='left' else 'Outlet')+': incident wave')
        axes[row,1].set_title(('Inlet' if case=='left' else 'Outlet')+': reflected wave')
        result['backend'][case]={k:float(abs(data['cpu'][case][k]-data['gpu'][case][k]).max()) for k in ('h','u')}
    fig.suptitle('Zero-viscosity boundary pulse: matching no-pulse controls subtracted')
    fig.savefig(OUT/'BOUNDARY_PULSES.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for case,ax in zip(('left','right'),axes):
        for name,sets in data.items():
            d=sets[case];p,m=characteristics(d,sets['base']);outgoing=m if case=='left' else p
            center=15 if case=='left' else 25
            selected=(d['x']<center-2)&(d['x']>1) if case=='left' else (d['x']>center+2)&(d['x']<39)
            ids=np.flatnonzero(selected);distance=abs(d['x'][ids]-center);order=np.argsort(distance)
            ax.plot(distance[order],outgoing[:,ids].max(0)[order]/AMP,label=name,ls='--' if name=='gpu' else '-')
        ax.axhline(1,color='k',ls=':',label='Linear nondissipative amplitude')
        ax.set(title=('Upstream' if case=='left' else 'Downstream')+' propagation before reflection',
            xlabel='Distance from initial pulse centre (m)',ylabel='Peak / initial analytic amplitude',ylim=(0,1.05))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('FLOW damps the wave more strongly before it reaches the boundary')
    fig.savefig(OUT/'PROPAGATION_ATTENUATION.png',dpi=155);plt.close(fig)
    np.savez(OUT/'SERIES.npz',**{f'{b}_{c}_{k}':v[k] for b,s in data.items() for c,v in s.items() for k in ('t','x','h','u')})
    save(OUT/'DIAGNOSTICS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
