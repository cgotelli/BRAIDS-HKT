"""Exact discrete budgets, native momentum diagnostics and raw-cell figures."""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from tip178 import BASE,TIMES,point_advection,velocity,exact_rate as known_rate
from probe_zero175 import energy
from quadrature176 import gather
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore
from review_hydraulics162 import verify_native
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_tip178'
G=9.81
POINTS=([7.75,2.75],[7.75,3.25],[8.25,3.25],[8.75,3.25],[8.25,3.75])


def split(a):
    """Exact algebraic split, not identification of pure numerical viscosity."""
    n=a['normal'];adv=[];pressure=[]
    for side in (0,1):
        h,mn,mt,_=a['local'][4*side:4*side+4]
        u=np.divide(np.array([mn*n[0]-mt*n[1],mn*n[1]+mt*n[0]]),h,
                    out=np.zeros((2,len(h))),where=h>0)
        adv.append(np.vstack([mn,mn*u]));pressure.append(np.vstack([np.zeros(len(h)),.5*G*h*h*n]))
    adv=np.mean(adv,axis=0);pressure=np.mean(pressure,axis=0)
    wall=(a['neighbour']<0)&(a['bid']<0);adv[:,wall]=0.
    left=[];right=[]
    for side in (0,1):
        cell=a['owner'] if side==0 else np.maximum(0,a['neighbour']);t=a['rec_trace']
        corr=.5*G*(a['q'][0,cell]+t[8+side])*(t[3+4*side]-a['z'][cell])
        bed=np.vstack([np.zeros(len(cell)),corr*n]);actual=a['fl' if side==0 else 'fr']
        parts=np.array([adv*(abs(n[0])>0)[None,:],adv*(abs(n[1])>0)[None,:],pressure,actual-adv-pressure-bed,bed])
        (left if side==0 else right).extend(parts)
    rates=np.array([gather(a,l,r) for l,r in zip(left,right)])
    assert np.max(abs(rates.sum(0)-a['rate']))<2e-12
    uv=a['q'][1:]/a['q'][0]
    acceleration=(rates[:,1:]-uv*rates[:,0,None,:])/a['q'][0]
    return rates,acceleration


def wall_rows(a,terms):
    wall=(a['neighbour']<0)&(a['bid']<0);m=a['mid'];n=a['normal'];rows=[]
    for f in np.flatnonzero(wall&(m[0]>=8)&(m[0]<=10)&(m[1]<=3)):
        c=a['owner'][f];h=a['rec_trace'][0,f];un=np.dot(a['rec_trace'][1:3,f],n[:,f])/h
        bed=.5*G*(a['q'][0,c]+a['rec_trace'][8,f])*(a['rec_trace'][3,f]-a['z'][c])
        pstar=np.dot(a['fl'][1:,f],n[:,f])-bed;hydro=.5*G*h*h
        rows.append(dict(face=int(f),mid=m[:,f].tolist(),normal=n[:,f].tolist(),
            depth=h,normal_trace_velocity=un,pressure=pstar,hydrostatic_pressure=hydro,
            pressure_ratio=pstar/hydro,mass_flux=float(a['fl'][0,f]),
            tangential_traction=float(np.dot(a['fl'][1:,f],[-n[1,f],n[0,f]])),
            entropy_work=float(terms[f])))
    return rows


def patches(a):
    """Local balance = patch exterior flux plus internal bed-source difference."""
    xy=a['xy'];o=a['owner'];r=a['neighbour'];inside=r>=0;ll=a['length'];rows={}
    masks={'leading_tip':(abs(xy[:,0]-8)<=1)&(abs(xy[:,1]-3)<=1),
           'trailing_tip':(abs(xy[:,0]-10)<=1)&(abs(xy[:,1]-3)<=1),
           'wake':(xy[:,0]>=11)&(xy[:,0]<20)}
    for name,selected in masks.items():
        so=selected[o];sr=inside&selected[np.maximum(0,r)];cross=so^sr;internal=so&sr
        outer=(-a['fl'][:,cross]*so[cross]+a['fr'][:,cross]*sr[cross])*ll[cross]
        source=(a['fr'][:,internal]-a['fl'][:,internal])*ll[internal]
        actual=(a['rate'][:,selected]*a['area'][selected]).sum(1)
        error=np.max(abs(actual-outer.sum(1)-source.sum(1)))
        rows[name]=dict(rate=actual.tolist(),outer_transfer=outer.sum(1).tolist(),
            interior_source=source.sum(1).tolist(),closure=float(error),
            depth_rate_rms=float(np.sqrt(np.mean(a['rate'][0,selected]**2))))
    return rows


def native_terms(folder,times,xy):
    verify_native(folder)
    ix,iy=np.rint(xy[:,0]/.5+.5).astype(int),np.rint(xy[:,1]/.5+.5).astype(int)
    keys=[['MOM_UDUDX','MOM_VDUDY','MOM_UPRESSURE','MOM_UBEDSHEAR','MOM_DUDT'],
          ['MOM_UDVDX','MOM_VDVDY','MOM_VPRESSURE','MOM_VBEDSHEAR','MOM_DVDT']]
    raw=[];projected=[]
    with Dataset(folder/'trim-sim_P057.nc') as d:
        ts=np.asarray(d['time'][:]);idx=np.array([int(np.flatnonzero(np.isclose(ts,t,atol=1e-10,rtol=0))[0]) for t in times])
        for axis,names in enumerate(keys):
            aa=[];bb=[]
            for k in names:
                arr=np.asarray(d[k][idx]).reshape(len(idx),82,18)
                aa.append(arr);bb.append((.5*(arr+np.roll(arr,1,axis+1)))[:,ix,iy])
            raw.append(aa);projected.append(bb)
        info=dict(max_viscosity=float(abs(np.asarray(d['VICUV'][idx])).max()),
                  bed_change=float(abs(np.asarray(d['DPS'][idx])-np.asarray(d['DPS'][0])).max()))
    return np.array(raw),np.array(projected),info


def main():
    OUT.mkdir(parents=True,exist_ok=True);metrics={};fields={};parity={}
    for label in ('BRAIDS','FLOW'):
        rows=[];samples=[]
        for t in TIMES:
            a=dict(np.load(BASE/f'probes/gpu/{label}{t}.npz'));c=np.load(BASE/f'probes/cpu/{label}{t}.npz')
            parity[f'{label}{t}']={k:float(abs(c[k]-a[k]).max()) for k in ('rate','water')}
            en,term=energy(a);rate,acc=split(a);wall=wall_rows(a,term)
            ids=[int(np.flatnonzero(np.all(a['xy']==p,axis=1))[0]) for p in POINTS]
            rows.append(dict(time=t,energy=en,wall=wall,patches=patches(a),
                momentum_acceleration_parts=acc[:,:,ids].tolist(),
                maximum_mass_shared=float(abs(a['fl'][0]-a['fr'][0]).max()),
                points=[dict(xy=p,depth=float(a['q'][0,j]),velocity=(a['q'][1:,j]/a['q'][0,j]).tolist(),
                    depth_rate=float(a['rate'][0,j]),spatial_acceleration=acc[:,:,j].sum(0).tolist()) for p,j in zip(POINTS,ids)]))
            samples.append(dict(a=a,terms=term,acc=acc))
        metrics[label]=rows;fields[label]=samples
    xy=a['xy'];raw,nacc,ni=native_terms(ROOT/'validation/_corner174/delft',TIMES,xy)
    native=dict(info=ni,first_row_cross_max=float(abs(raw[0,1,:,16:21,7]).max()),
        first_row_x=[8,8.5,9,9.5,10],first_row_mean_terms=raw[0,:,:,16:21,7].mean(1).tolist(),
        momentum_closure_max=float(abs(nacc[:,:4].sum(1)-nacc[:,4]).max()),
        momentum_closure_tip_region_max=float(abs((nacc[:,:4].sum(1)-nacc[:,4])[:,:,(xy[:,0]>6)&(xy[:,0]<13)]).max()))
    known={}
    for dt in (.0001,.00005):
        folder=BASE/f'known_delft_{dt:g}';r,pr,inf=native_terms(folder,[dt],xy)
        x=np.array([8.,8.5,9.]);expected=np.array([point_advection(v-8,.25)[:,0] for v in x]).T
        # pressure begins to respond to the discrete velocity divergence. At
        # the initial instant the exact constant-depth field has no pressure gradient.
        known[str(dt)]=dict(x=x.tolist(),expected_x_advection=expected[0].tolist(),
            expected_cross_advection=expected[1].tolist(),native_x_advection=r[0,0,0,16:19,7].tolist(),
            native_cross_advection=r[0,1,0,16:19,7].tolist(),native_pressure=r[0,2,0,16:19,7].tolist(),
            native_total=r[0,4,0,16:19,7].tolist(),info=inf)
        with Dataset(folder/'trim-sim_P057.nc') as d:
            u=np.asarray(d['U1'][0,0]);v=np.asarray(d['V1'][0,0]);z=np.asarray(d['DPS'][0]);h=np.asarray(d['S1'][0])+z
            wallmax=max(float(abs(u[np.asarray(d['KCU'])==0]).max()),float(abs(v[np.asarray(d['KCV'])==0]).max()))
            known[str(dt)].update(initial_wet_depth_error=float(abs(h[np.asarray(d['KCS'])==1]-.5).max()),
                initial_wall_velocity_max=wallmax,
                initial_U_projection_error=max(abs(u[i,7]-velocity(i*.5-8,.25)[0]) for i in range(16,19)))
    aa=np.load(BASE/'probes/gpu/bounded.npz');cc=np.load(BASE/'probes/cpu/bounded.npz');rec=[]
    for p in POINTS:
        j=int(np.flatnonzero(np.all(xy==p,axis=1))[0]);truth=known_rate(*(np.array(p)-[8,3]))
        uv=aa['q'][1:,j]/aa['q'][0,j];acc=(aa['rate'][1:,j]-uv*aa['rate'][0,j])/.5
        rec.append(dict(xy=p,exact_rate=truth.tolist(),BRAIDS_rate=aa['rate'][:,j].tolist(),
            exact_velocity_acceleration=(truth[1:]/.5).tolist(),BRAIDS_velocity_acceleration=acc.tolist()))
    known['BRAIDS']=dict(cells=rec,backend_rate_max=float(abs(aa['rate']-cc['rate']).max()),
        scope='Volume-averaged instantaneous momentum derivative; FLOW values are pointwise at staggered U locations. Do not rank these unlike error norms.')
    summary={}
    for label,rows in metrics.items():
        lead=[next(w for w in v['wall'] if w['mid']==[8.25,3.]) for v in rows]
        summary[label]=dict(energy_production_min=min(v['energy']['net_spatial_production'] for v in rows),
            energy_production_max=max(v['energy']['net_spatial_production'] for v in rows),
            groin_wall_work_mean=float(np.mean([sum(w['entropy_work'] for w in v['wall']) for v in rows])),
            leading_wall_work_mean=float(np.mean([w['entropy_work'] for w in lead])),
            leading_pressure_ratio_range=[min(w['pressure_ratio'] for w in lead),max(w['pressure_ratio'] for w in lead)],
            leading_normal_velocity_range=[min(w['normal_trace_velocity'] for w in lead),max(w['normal_trace_velocity'] for w in lead)],
            local_budget_max=max(p['closure'] for v in rows for p in v['patches'].values()))
    save(OUT/'DIAGNOSTICS.json',dict(summary=summary,replays=metrics,backend=parity,native=native,known=known,
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),production_changed=False))
    make_figures(fields,raw,known,summary)
    print(json.dumps(dict(summary=summary,native=native,known=known),indent=2))


def make_figures(fields,raw,known,summary):
    s=ResultsStore(ROOT/'validation/_corner174/cpu/warmup');f=s.read(0)
    verts=[np.asarray(p.exterior.coords) for p in f.polygons]
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,label in enumerate(('BRAIDS','FLOW')):
        a=fields[label][10]['a'];xy=a['xy'];uv=a['q'][1:]/a['q'][0];mask=(xy[:,0]>=6)&(xy[:,0]<=13)&(xy[:,1]<=6)
        pc=PolyCollection(verts,array=np.hypot(*uv),cmap='viridis',clim=(0,2.5),edgecolors=(0,0,0,.12),linewidths=.3)
        axs[0,col].add_collection(pc);axs[0,col].quiver(xy[mask,0],xy[mask,1],uv[0,mask],uv[1,mask],scale=22,width=.003)
        axs[0,col].set_title(('BRAIDS' if col==0 else 'FLOW velocities averaged to cells')+' at 210 s')
        mean=np.mean([v['terms'] for v in fields[label]],axis=0)
        interior=a['neighbour']>=0;m=a['mid'];norm=a['normal'];tangent=np.array([-norm[1],norm[0]])
        lines=np.stack([m-.5*a['length']*tangent,m+.5*a['length']*tangent],axis=1).transpose(2,1,0)
        lc=LineCollection(lines, array=mean,cmap='RdBu_r',linewidths=3,clim=(-.04,.04));axs[1,col].add_collection(lc)
        axs[1,col].set_title('BRAIDS operator on '+('its own states' if col==0 else 'mapped FLOW states')+'\nmean numerical energy work, 200–220 s')
        for ax in axs[:,col]:
            ax.add_patch(plt.Rectangle((8,0),2,3,color='.7'));ax.set(xlim=(6,13),ylim=(1,6),aspect='equal',xlabel='x (m)',ylabel='y (m)')
            ax.plot(8,3,'ko',ms=4)
    fig.colorbar(pc,ax=axs[0,:],label='Speed (m/s)');fig.colorbar(lc,ax=axs[1,:],label='Energy work / density (m⁵/s³)\nnegative = removal')
    fig.savefig(OUT/'TIP_TRANSFERS.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for label,color in [('BRAIDS','#1764ab'),('FLOW','#b34b23')]:
        rows=[];eg=[]
        for v in fields[label]:
            en,terms=energy(v['a']);ws=wall_rows(v['a'],terms);rows.append(next(w for w in ws if w['mid']==[8.25,3.]));eg.append(en['net_spatial_production'])
        text='BRAIDS on '+('own states' if label=='BRAIDS' else 'mapped FLOW states')
        axs[0,0].plot(TIMES,[r['pressure_ratio'] for r in rows],label=text,color=color)
        axs[0,1].plot(TIMES,[r['normal_trace_velocity'] for r in rows],label=text,color=color)
        axs[1,0].plot(TIMES,eg,label=text,color=color)
    axs[0,0].set(title='First top-wall face at (8.25, 3) m',ylabel='Riemann pressure / trace hydrostatic pressure');axs[0,0].axhline(1,color='.4',ls=':')
    axs[0,1].set(title='Incoming reconstructed state, not wall flow',ylabel='Outward-normal trace velocity (m/s)')
    axs[1,0].set(title='Full spatial operator energy balance',ylabel='Production / density (m⁵/s³)')
    names=['Pressure','Streamwise advection','Cross-advection','Bed drag','Acceleration']
    order=[2,0,1,3,4];vals=raw[0,:,:,16:21,7].mean(1)
    for k,name in zip(order,names):axs[1,1].plot([8,8.5,9,9.5,10],vals[k],'o-',label=name)
    axs[1,1].set(title='Actual FLOW terms, first row, mean 200–220 s',xlabel='Staggered U coordinate x (m)',ylabel='Acceleration (m/s²)')
    for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7);ax.axhline(0,color='.6',lw=.5)
    for ax in (axs[0,0],axs[0,1],axs[1,0]):ax.set_xlabel('Time (s)')
    fig.savefig(OUT/'WALL_AND_NATIVE_BALANCE.png',dpi=150);plt.close(fig)
    k=known['0.0001'];fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axs[:2],('x_advection','cross_advection'),('Streamwise advection','Cross-advection')):
        ax.plot(k['x'],k['expected_'+key],'ko-',label='Analytic initial field')
        ax.plot(k['x'],k['native_'+key],'s--',label='FLOW dt = 0.1 ms')
        ax.plot(k['x'],known['5e-05']['native_'+key],'+:',label='FLOW dt = 0.05 ms')
        ax.set(title=title,xlabel='Staggered U coordinate x (m)',ylabel='Acceleration (m/s²)');ax.legend(fontsize=8)
    b=known['BRAIDS']['cells'];xx=np.arange(len(b));errors=np.array([np.array(v['BRAIDS_velocity_acceleration'])-v['exact_velocity_acceleration'] for v in b])
    axs[2].bar(xx-.18,errors[:,0],.36,label='x acceleration error');axs[2].bar(xx+.18,errors[:,1],.36,label='y acceleration error')
    axs[2].set_xticks(xx,[f"{v['xy'][0]}\n{v['xy'][1]}" for v in b]);axs[2].set(title='BRAIDS also has local corner error',xlabel='Cell centre x / y (m)',ylabel='Cell-average acceleration error (m/s²)');axs[2].legend(fontsize=8)
    for ax in axs:ax.axhline(0,color='.5',lw=.5);ax.grid(alpha=.2)
    fig.savefig(OUT/'KNOWN_CORNER.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
