"""Matched settling, boundary, and post-spinup bedload evidence for milestone174."""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from review_settling174 import gate,rms,LIMITS
from review_hydraulics162 import verify_native,norm
from braids_desktop.results import ResultsStore
BASE=ROOT/'validation/_corner174'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner174'


def pair(folder,kind='warmup',native_folder=None):
    native_folder=folder/'delft' if native_folder is None else native_folder
    verify_native(native_folder)
    s=ResultsStore(folder/'cpu'/kind);f=[s.read(i) for i in range(len(s.times))]
    xy=np.array([p.centroid.coords[0] for p in f[0].polygons]);area=f[0].arrays['area']
    ix,iy=np.rint(xy[:,0]/.5+.5).astype(int),np.rint(xy[:,1]/.5+.5).astype(int)
    b=dict(t=s.times,h=np.array([v.q[0] for v in f]),z=np.array([v.z for v in f]),
           uv=np.array([v.q[1:]/v.q[0] for v in f]))
    with Dataset(native_folder/'trim-sim_P057.nc') as d:
        times=np.asarray(d['time'][:]);j=np.searchsorted(times,s.times);assert np.array_equal(times[j],s.times)
        z=-np.asarray(d['DPS'][j])[:,ix,iy];eta=np.asarray(d['S1'][j])[:,ix,iy]
        u,v=[np.asarray(d[k][j,0]) for k in ('U1','V1')]
        uv=np.stack([.5*(u+np.roll(u,1,1)),.5*(v+np.roll(v,1,2))],axis=1)[:,:,ix,iy]
        n=dict(t=s.times,h=eta-z,z=z,uv=uv)
        nf=np.asarray(d['SBUUA'][j])[:,0];mt=np.asarray(d['MFTAVG'][j])*86400
        mass=np.asarray(d['MSED'][j])[:,0,0,ix,iy]
        native=dict(feed=nf[:,0,1:17].sum(1)*.5*mt,export=nf[:,80,1:17].sum(1)*.5*mt,
            net_solid=(mass-mass[0]).sum(1)*.25/2650,
            max_viscosity=float(abs(np.asarray(d['VICUV'][j])).max()))
    with Dataset(native_folder/'trih-sim_P057.nc') as d:
        native.update(instant_Q=np.asarray(d['CTR'][j]),cumulative_Q=np.asarray(d['FLTR'][j]))
    return b,n,xy,area,s,f,native


def backend(folder,kind,b,area):
    s=ResultsStore(folder/'gpu'/kind);fs=[s.read(i) for i in range(len(s.times))]
    assert np.array_equal(s.times,b['t'])
    g=dict(t=s.times,h=np.array([f.q[0] for f in fs]),z=np.array([f.z for f in fs]),
           uv=np.array([f.q[1:]/f.q[0] for f in fs]))
    metrics={k:norm(b[k]-g[k]) for k in ('h','z','uv')}
    delta=abs(b['h']-g['h']);j,c=np.unravel_index(np.argmax(delta),delta.shape)
    metrics['depth_max_location']=dict(time_s=float(s.times[j]),cell=int(c),
        xy=list(fs[0].polygons[c].centroid.coords[0]))
    exceed=np.flatnonzero(delta.max(1)>.001)
    metrics['first_depth_over_1mm_s']=float(s.times[exceed[0]]) if len(exceed) else None
    if kind=='warmup':metrics['settling']=[gate(**g,end=t,area=area) for t in (s.times[-1]-20,s.times[-1])]
    return metrics


def budget(folder,kind):
    rows=json.loads((folder/kind/'DIAGNOSTICS.json').read_text())
    return dict(water_residual_m3=max(abs(v['water_residual_m3']) for v in rows),
        sediment_residual_kg=max(abs(w) for v in rows for w in v['class_residual_kg']),
        wall_integrals=max(abs(w) for v in rows for w in v['wall_integrals']),
        minimum_stock_kg_m2=min(v['minimum_stock'] for v in rows))


def boundaries():
    rows={}
    for path in sorted((BASE/'boundary_probes_v2/cpu').glob('*.npz')):
        c=np.load(path);g=np.load(BASE/'boundary_probes_v2/gpu'/path.name)
        assert np.array_equal(c['flux'],g['flux']) and np.array_equal(c['trace'],g['trace'])
        record={}
        for k,(label,x,sign) in enumerate([('inlet',0.,-1.),('outlet',40.,1.)]):
            selected=np.flatnonzero((c['neighbour']<0)&(c['bid']>=0)&(c['mid'][0]==x))
            order=selected[np.argsort(c['mid'][1,selected])]
            q=sign*c['flux'][0,order]*c['length'][order];nq=c['native_face_flux_m3_s'][k]
            eta=c['trace'][0,order]+c['exterior_bed'][order]
            record[label]=dict(total_m3_s=float(q.sum()),native_total_m3_s=float(nq.sum()),
                segment_flux_min_m3_s=float(q.min()),segment_flux_max_m3_s=float(q.max()),
                native_segment_min_m3_s=float(nq.min()),native_segment_max_m3_s=float(nq.max()),
                reverse_faces=int((q<0).sum()),native_reverse_faces=int((nq<0).sum()),
                stage_min_m=float(eta.min()),stage_max_m=float(eta.max()),
                segment_flux_difference_rms_m3_s=rms(q-nq),
                total_relative_difference=float((q.sum()-nq.sum())/nq.sum()),
                inlet_transverse_flux_max=float(abs(c['flux'][2,order]).max()) if k==0 else None)
        rows[path.stem]=record
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);sets={};evidence={}
    for label,folder in [('zero300',BASE),('zero600',BASE/'long'),('mix300',BASE/'mixing'),('inlet_total',BASE/'inlet_total')]:
        b,n,xy,area,s,fs,native=pair(folder,native_folder=BASE/'mixing/delft' if label=='inlet_total' else None)
        sets[label]=(b,n,s,fs,native)
        assert abs(b['z']-b['z'][0]).max()==abs(n['z']-n['z'][0]).max()==0
        assert abs(b['z'][0]-n['z'][0]).max()<1e-12 and abs(b['h'][0]-n['h'][0]).max()<3e-8
        end=s.times[-1]
        metrics=dict(settling={k:[gate(**d,end=t,area=area) for t in (end-20,end)]
            for k,d in [('BRAIDS',b),('FLOW',n)]},backend=backend(folder,'warmup',b,area),
            budgets={k:budget(folder/k,'warmup') for k in ('cpu','gpu')},native_viscosity=native['max_viscosity'])
        metrics['difference_at_activation']={k:norm(b[k][-1]-n[k][-1]) for k in ('h','uv')}
        metrics['sections']={}
        for x in (4.25,8.25,10.25,15.25):
            sel=xy[:,0]==x
            metrics['sections'][str(x)]={k:dict(v_absmax=float(abs(d['uv'][-1,1,sel]).max()),
                u_max=float(d['uv'][-1,0,sel].max()),eta_mean=float((d['h'][-1,sel]+d['z'][-1,sel]).mean()))
                for k,d in [('BRAIDS',b),('FLOW',n)]}
        evidence[label]=metrics
    old=sets['mix300'][0];new=sets['inlet_total'][0]
    evidence['inlet_total']['effect_vs_reference_inlet']={k:norm(new[k][-1]-old[k][-1]) for k in ('h','uv')}
    evidence['inlet_total']['effect_vs_reference_inlet']['upstream_mean_eta_change_m']=float(
        (new['h'][-1]-old['h'][-1])[xy[:,0]==4.25].mean())
    assert all(r['passed'] for rows in evidence['mix300']['settling'].values() for r in rows)
    assert all(r['passed'] for r in evidence['mix300']['backend']['settling'])
    b,n,xy,area,s,fs,native=pair(BASE/'mixing','mobile');sets['mobile']=(b,n,s,fs,native)
    bd=b['z']-b['z'][0];nd=n['z']-n['z'][0]
    mobile=dict(backend=backend(BASE/'mixing','mobile',b,area),series=[],
        budgets={k:budget(BASE/'mixing'/k,'mobile') for k in ('cpu','gpu')},
        native_sediment_residual_m3=float(abs(native['net_solid']-(native['feed']-native['export'])).max()))
    for time in (300,310,320,330,340):
        j=int(np.flatnonzero(s.times==time)[0]);row=dict(time_s=time,bed_difference=norm(bd[j]-nd[j]))
        for label,dz in [('BRAIDS',bd[j]),('FLOW',nd[j])]:
            row[label]=dict(scour_m=float(-dz.min()),deposition_m=float(dz.max()),
                scour_xy=xy[dz.argmin()].tolist(),deposition_xy=xy[dz.argmax()].tolist(),
                eroded_bulk_m3=float(-np.dot(np.minimum(dz,0),area)),
                deposited_bulk_m3=float(np.dot(np.maximum(dz,0),area)))
        mobile['series'].append(row)
    evidence.update(mobile=mobile,boundary_replays=boundaries(),limits=LIMITS,
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),production_changed=False,
        viscosity_diagnostic='0.05 m2/s in both; same coefficient, distinct depth/staggering operators; not calibrated')
    save(OUT/'MIXING_AND_BOUNDARIES.json',evidence)
    verts=[np.asarray(p.exterior.coords) for p in fs[0].polygons]
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    colors=['#2375bb','#e57b22']
    for row,label in enumerate(('zero600','mix300')):
        b,n,store,frames,native=sets[label]
        for name,d,col in zip(('BRAIDS','FLOW'),(b,n),colors):
            std=[rms((d['h'][max(0,j-20):j+1]+d['z'][max(0,j-20):j+1])-
                     (d['h'][max(0,j-20):j+1]+d['z'][max(0,j-20):j+1]).mean(0))*1000 for j in range(len(d['t']))]
            axes[row,0].plot(d['t'],std,label=name,color=col)
            k=np.flatnonzero(np.all(xy==[8.25,3.25],axis=1))[0]
            axes[row,1].plot(d['t'],d['h'][:,k]+d['z'][:,k],color=col,label=name)
        axes[row,0].axhline(2,color='grey',ls=':',label='2 mm RMS gate')
        axes[row,0].set(title=('Zero mixing' if row==0 else 'Matched viscosity 0.05 m²/s'),ylabel='20 s water-surface variability RMS (mm)',xlabel='Time (s)')
        axes[row,1].set(title='First corner cell (8.25, 3.25 m)',ylabel='Water elevation (m)',xlabel='Time (s)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'MIXING_SETTLING.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    b,n,*_=sets['mix300']
    for col,x in enumerate((4.25,8.25,10.25,15.25)):
        select=np.flatnonzero(xy[:,0]==x);select=select[np.argsort(xy[select,1])]
        for label,d,color in zip(('BRAIDS','FLOW'),(b,n),colors):
            vals=[(d['h'][-21:]+d['z'][-21:]).mean(0),d['uv'][-21:,0].mean(0),d['uv'][-21:,1].mean(0)]
            for row in range(3):axes[row,col].plot(xy[select,1],vals[row][select],'.-',label=label,color=color)
        axes[0,col].set_title(f'x = {x:g} m, mean 280–300 s')
        for row in range(3):
            axes[row,col].set(xlabel='y (m)',ylabel=['Water elevation (m)','u (m/s)','v (m/s)'][row]);axes[row,col].grid(alpha=.2)
            if row:axes[row,col].axhline(0,color='grey',lw=.6)
    axes[0,0].legend();fig.suptitle('Settled hydraulic comparison — identical viscosity coefficient, no sediment yet')
    fig.savefig(OUT/'SETTLED_SECTIONS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for row,(label,arr) in enumerate([('BRAIDS',bd),('FLOW',nd)]):
        for col,time in enumerate((310,320,340)):
            j=int(np.flatnonzero(s.times==time)[0]);ax=axes[row,col]
            p=PolyCollection(verts,array=arr[j]*1000,cmap='RdBu_r',clim=(-30,30),edgecolor='none')
            ax.add_collection(p);ax.set(xlim=(5,18),ylim=(0,8),title=f'{label}: {time-300} s bed evolution',xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
            ax.fill([8,10,10,8],[0,0,3,3],color='.6')
    fig.colorbar(p,ax=axes.ravel().tolist(),label='Bed change (mm)',shrink=.8,extend='both')
    fig.suptitle('MPM activated at 300 s after BOTH models passed settling checks; MF 1')
    fig.savefig(OUT/'SETTLED_BEDLOAD.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for label,arr,col in [('BRAIDS',bd,colors[0]),('FLOW',nd,colors[1])]:
        axes[0].plot(s.times-300,-arr.min(1)*1000,label=label+' scour',color=col)
        axes[0].plot(s.times-300,arr.max(1)*1000,label=label+' deposition',color=col,ls='--')
        axes[1].plot(s.times-300,-np.minimum(arr,0)@area,label=label+' eroded',color=col)
        axes[1].plot(s.times-300,np.maximum(arr,0)@area,label=label+' deposited',color=col,ls='--')
    axes[0].set(ylabel='Extreme bed change (mm)',xlabel='Time since sediment activation (s)')
    axes[1].set(ylabel='Bulk bed volume (m³)',xlabel='Time since sediment activation (s)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    fig.savefig(OUT/'BEDLOAD_TRENDS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for col,label in enumerate(('zero600','mix300')):
        b,n,store,frames,native=sets[label]
        t=store.times;ins=store._boundary_mask(frames[0],'Upstream');outs=store._boundary_mask(frames[0],'Downstream')
        cum=np.array([[f.arrays['water'][ins].sum(),-f.arrays['water'][outs].sum()] for f in frames])
        q=np.diff(cum,axis=0)/np.diff(t)[:,None]
        axes[0,col].plot(t[1:],q[:,1],label='BRAIDS output-window mean',color=colors[0])
        axes[0,col].plot(t,native['instant_Q'][:,1],label='FLOW instantaneous section',color=colors[1])
        axes[0,col].axhline(4,color='.4',ls=':',label='Prescribed inflow')
        axes[0,col].set(title=('Zero mixing' if col==0 else 'Matched viscosity 0.05 m²/s'),xlabel='Time (s)',ylabel='Outlet discharge (m³/s)')
    a=np.load(BASE/'boundary_probes_v2/cpu/mix300_FLOW.npz')
    own=np.load(BASE/'boundary_probes_v2/cpu/mix300_BRAIDS.npz')
    for col,(x,sgn,title) in enumerate(((0,-1,'Inlet distribution at 300 s'),(40,1,'Outlet distribution at 300 s'))):
        i=np.flatnonzero((a['neighbour']<0)&(a['bid']>=0)&(a['mid'][0]==x));i=i[np.argsort(a['mid'][1,i])]
        y=a['mid'][1,i]
        for lab,data,style,color in [('BRAIDS, own state',own,'-',colors[0]),('BRAIDS, projected FLOW state',a,'--','#7659a3')]:
            axes[1,col].plot(y,sgn*data['flux'][0,i]*data['length'][i],style,label=lab,color=color)
        axes[1,col].plot(y,a['native_face_flux_m3_s'][col],'.-',label='FLOW, native staggered flux',color=colors[1])
        axes[1,col].set(title=title,xlabel='y (m)',ylabel='Flow per 0.5 m segment (m³/s)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'BOUNDARY_AUDIT.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
    for col,x in enumerate((4.25,8.25)):
        sel=np.flatnonzero(xy[:,0]==x);sel=sel[np.argsort(xy[sel,1])]
        for label,d,color,style in [('BRAIDS reference-stage inlet',old,colors[0],'-'),
            ('BRAIDS total-discharge inlet',new,'#7659a3','--'),('FLOW total-discharge inlet',sets['mix300'][1],colors[1],'-')]:
            vals=[(d['h'][-21:]+d['z'][-21:]).mean(0),d['uv'][-21:,0].mean(0),d['uv'][-21:,1].mean(0)]
            for row in range(3):axes[row,col].plot(xy[sel,1],vals[row][sel],style,label=label,color=color)
        axes[0,col].set_title(f'x = {x:g} m; mean 280–300 s')
        for row in range(3):
            axes[row,col].set(xlabel='y (m)',ylabel=['Water elevation (m)','u (m/s)','v (m/s)'][row]);axes[row,col].grid(alpha=.2)
            if row:axes[row,col].axhline(0,color='.5',lw=.6)
    axes[0,0].legend(fontsize=7);fig.suptitle('Existing inlet-formulation contrast — same Q, viscosity and outlet')
    fig.savefig(OUT/'INLET_CONTRAST.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(mobile=mobile,settled_hydro=evidence['mix300'],boundaries=evidence['boundary_replays']['mix300_FLOW']),indent=2))


if __name__=='__main__':main()
