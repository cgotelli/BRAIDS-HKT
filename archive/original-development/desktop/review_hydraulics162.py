"""Raw-field hydraulic attribution and matched short mobile-bed qualification."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save,SPECS
from braids_desktop.results import ResultsStore

BASE=ROOT/'validation/_hydraulics162'
OLD=ROOT/'validation/_bedload161_matched'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_hydraulics162'


def norm(a):
    return dict(rms=float(np.sqrt(np.mean(a*a))),maximum=float(abs(a).max()))


def verify_native(folder):
    inp=json.loads((folder/'INPUTS.json').read_text())
    for name,checksum in inp['inputs'].items():
        assert (folder/name).resolve().is_relative_to(folder.resolve())
        assert sha(folder/name)==checksum,name
    run=json.loads((folder/'execution.json').read_text())
    assert run['success'] and run['executable_sha256']=='09f451b673fcf855ae8f07d819686e7a429ab9382ff6da372702aa3a46583623'
    for row in run['outputs']:assert sha(folder/row['name'])==row['sha256']
    return run


def load(run,native):
    receipt=verify_native(native)
    store=ResultsStore(run);frames=[store.read(j) for j in range(len(store.times))]
    f=frames[0];xy=np.array([p.centroid.coords[0] for p in f.polygons])
    ix=np.rint(xy[:,0]/.5+.5).astype(int);iy=np.rint(xy[:,1]/.5+.5).astype(int)
    with Dataset(native/'trim-sim_P057.nc') as d:
        active=np.asarray(d['KCS'][:])==1
        assert active.sum()==len(xy) and active[ix,iy].all()
        assert np.array_equal(np.asarray(d['XZ'][:])[ix,iy],xy[:,0])
        assert np.array_equal(np.asarray(d['YZ'][:])[ix,iy],xy[:,1])
        assert np.array_equal(f.arrays['area'],np.full(len(xy),.25))
        assert np.array_equal(np.asarray(d['time'][:]),store.times)
        nz=-np.asarray(d['DPS'][:])[:,ix,iy]
        nh=np.asarray(d['S1'][:])[:,ix,iy]-nz
        # UUU/VVV are sediment characteristic velocities (dwnvel), NOT the
        # primary hydraulic velocity unknowns. Keep both observables explicit.
        characteristic=np.stack([np.asarray(d[k][:])[:,ix,iy] for k in ('UUU','VVV')],axis=1)
        fu=np.asarray(d['U1'][:,0]);fv=np.asarray(d['V1'][:,0])
        nu=np.stack([.5*(fu+np.roll(fu,1,axis=1)),.5*(fv+np.roll(fv,1,axis=2))],axis=1)[:,:,ix,iy]
        assert abs(f.z-nz[0]).max()<1e-12
        assert abs(f.q[0]-nh[0]).max()<3e-8
        assert abs(nu[0]).max()==0 and abs(f.q[1:]).max()==0
        nx=int(xy[:,0].max()/.5+.5)
        nflux=np.asarray(d['SBUUA'][:])[:,0]
        mt=np.asarray(d['MFTAVG'][:])*86400
        feed=nflux[:,0,1:17].sum(axis=1)*.5*mt
        export=nflux[:,nx,1:17].sum(axis=1)*.5*mt
        mass=np.asarray(d['MSED'][:])[:,0,0,ix,iy]
        vbed=np.sum((mass-mass[0])*.25,axis=1)/2650
        budget=vbed-(feed-export)
        inlet_water=np.asarray(d['U1'][:])[:,0,0,1:17]
        outlet_water=np.asarray(d['U1'][:])[:,0,nx,1:17]
        max_nu=float(abs(np.asarray(d['VICUV'][:])).max())
    with Dataset(native/'trih-sim_P057.nc') as d:
        ctr=np.asarray(d['CTR'][:]);fltr=np.asarray(d['FLTR'][:])
    bh=np.array([f.q[0] for f in frames]);bz=np.array([f.z for f in frames])
    bu=np.array([f.q[1:]/f.q[0] for f in frames])
    common=xy[:,0]<20
    series=[dict(time_s=float(t),depth=norm(bh[j]-nh[j]),
        common_first20m_depth=norm(bh[j,common]-nh[j,common]),
        velocity_components=norm(bu[j]-nu[j]),
        native_outlet_velocity_min=float(outlet_water[j].min()),
        native_Q_in_m3_s=float(ctr[j,0]),native_Q_out_m3_s=float(ctr[j,1]),
        braids_cumulative_in_m3=float(frames[j].arrays['water'][store._boundary_mask(frames[j],'Upstream')].sum()),
        braids_cumulative_out_m3=float(-frames[j].arrays['water'][store._boundary_mask(frames[j],'Downstream')].sum())) for j,t in enumerate(store.times)]
    data=dict(store=store,frames=frames,xy=xy,bh=bh,nh=nh,bu=bu,nu=nu,nchar=characteristic,bz=bz,nz=nz,ctr=ctr,fltr=fltr,common=common)
    result=dict(native_execution=receipt,cells=len(xy),series=series,
        initial_bed_max_m=float(abs(frames[0].z-nz[0]).max()),initial_depth_max_m=float(abs(bh[0]-nh[0]).max()),
        native_horizontal_viscosity_max_m2_s=max_nu,native_sediment_residual_m3=float(abs(budget).max()),
        native_feed_m3=float(feed[-1]),native_export_m3=float(export[-1]),native_minimum_stock_kg_m2=float(mass.min()),
        native_initial_stock_kg_m2=[float(mass[0].min()),float(mass[0].max())])
    return data,result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases={
        'short':(OLD/'braids_mpm_cpu/run',OLD/'delft_mpm_v8'),
        'straight':(BASE/'domain_v2/straight/braids_cpu/run',BASE/'domain_v2/straight/delft'),
        'long':(BASE/'domain_v2/long/braids_cpu/run',BASE/'domain_v2/long/delft'),
        'cyclic':(BASE/'baseline/braids_mpm_cpu/run',BASE/'domain_v2/cyclic/delft')}
    data={};summary={}
    for key,(b,n) in cases.items():data[key],summary[key]=load(b,n)
    # Isolated counterfactuals; neither is promoted to the solver.
    ref=data['short'];contrasts={}
    for name in ('baseline','tangent','half_cfl'):
        s=ResultsStore(BASE/name/'braids_mpm_cpu/run');f=s.read(-1)
        contrasts[name]=dict(depth_to_FLOW=norm(f.q[0]-ref['nh'][8]),depth_to_original_BRAIDS=norm(f.q[0]-ref['bh'][8]),
            maximum_momentum_to_original=float(abs(f.q[1:]-ref['frames'][8].q[1:]).max()))
    assert contrasts['baseline']['depth_to_original_BRAIDS']['maximum']==0
    gpu=ResultsStore(BASE/'domain_v2/long/braids_gpu/run').read(-1)
    long_parity=norm(gpu.q-data['long']['frames'][-1].q)
    assert long_parity['maximum']<1e-9
    # Explicit same-state inputs remove trajectory/wake differences.
    replays=[]
    for j in (1,2,4,6,8):
        for source in ('FLOW','BRAIDS'):
            name=f'{source}_{5*j:04.0f}.npz'
            with np.load(BASE/'operator_v3/cpu'/name) as c,np.load(BASE/'operator_v3/gpu'/name) as g:
                parity=norm(c['rate']-g['rate']);assert parity['maximum']<1e-12
                assert np.array_equal(c['water'],g['water'])
                mask=(c['neighbour']<0)&(c['mid'][0]==20)
                ins=(c['neighbour']<0)&(c['mid'][0]==0)
                outside=c['neighbour']<0;wall=outside&(c['bid']<0)
                assert np.max(abs(c['water'][wall]))==0
                balance=float(np.dot(c['rate'][0],c['area'])+np.dot(c['water'][outside],c['length'][outside]))
                assert abs(balance)<1e-12
                qout=float(np.dot(c['water'][mask],c['length'][mask]))
                qin=float(-np.dot(c['water'][ins],c['length'][ins]));assert abs(qin-4)<1e-10
                replays.append(dict(source=source,source_observable='FLOW characteristic UUU/VVV' if source=='FLOW' else 'BRAIDS cell momentum/depth',
                    time_s=5*j,rate_parity=parity,water_balance_m3_s=balance,
                    Q_in_m3_s=qin,Q_out_m3_s=qout,native_Q_out_m3_s=float(ref['ctr'][j,1]),
                    out_Q_relative_to_native=float((qout-ref['ctr'][j,1])/ref['ctr'][j,1])))
    # Separate the native momentum-scheme configuration from a source edit.
    native_scheme=norm(data['cyclic']['nh'][8]-ref['nh'][8])
    short_updates=[]
    for seconds in (10,30):
        native=BASE/'native_replay'/str(seconds);verify_native(native)
        with np.load(BASE/'operator_v3/cpu'/f'STEP_FLOW_{seconds:04d}.npz') as c,np.load(BASE/'operator_v3/gpu'/f'STEP_FLOW_{seconds:04d}.npz') as g:
            a=c['braids_final'];b=c['native_final'];u=a[1:]/a[0]-b[1:]/b[0];speed=np.linalg.norm(u,axis=0)
            assert np.array_equal(a,g['braids_final'])
            short_updates.append(dict(source_time_s=seconds,window_s=.01,dt_s=.001,depth_difference=norm(a[0]-b[0]),
                velocity_components_difference=norm(u),velocity_vector_max_m_s=float(speed.max()),
                velocity_max_xy=c['xy'][speed.argmax()].tolist(),depth_max_xy=c['xy'][abs(a[0]-b[0]).argmax()].tolist(),
                backend_final_exact=True,
                caveat='FLOW primary staggered face velocities interpolated to cell centres; not identical discrete velocity DOFs or a binary native restart.'))
    mobile={};mobile_data={}
    for formula in SPECS:
        d,r=load(BASE/'mobile'/('braids_'+formula+'_cpu')/'run',BASE/'mobile'/('delft_'+formula))
        final=d['frames'][-1];first=d['frames'][0];s=d['store']
        assert r['native_initial_stock_kg_m2']==[159.,159.]
        assert abs(first.arrays['mass'].sum(axis=(0,1))-159.).max()<1e-10
        gf=ResultsStore(BASE/'mobile'/('braids_'+formula+'_gpu')/'run').read(-1)
        bd=d['bz'][-1]-d['bz'][0];nd=d['nz'][-1]-d['nz'][0];common=d['common']
        assert abs(d['bz'][:9]-d['bz'][0]).max()==0 and abs(d['nz'][:9]-d['nz'][0]).max()==0
        r.update(bed_difference=norm(bd-nd),first20m_bed_difference=norm((bd-nd)[common]),
            bed_correlation=float(np.corrcoef(bd,nd)[0,1]),
            parity_bed=norm(final.z-gf.z),parity_depth=norm(final.q[0]-gf.q[0]),
            braids_feed_m3=float(final.arrays['bed_boundary'][:,s._boundary_mask(final,'Upstream')].sum()),
            braids_export_m3=float(-final.arrays['bed_boundary'][:,s._boundary_mask(final,'Downstream')].sum()))
        for name,z in (('BRAIDS',bd),('FLOW',nd)):
            r[name]=dict(scour_m=float(-z.min()),deposition_m=float(z.max()),scour_xy=d['xy'][z.argmin()].tolist(),
                deposition_xy=d['xy'][z.argmax()].tolist(),net_solid_m3=float(np.dot(z,first.arrays['area'])*.6))
        for backend in ('cpu','gpu'):
            diagnostics=json.loads((BASE/'mobile'/('braids_'+formula+'_'+backend)/'run/DIAGNOSTICS.json').read_text())
            r[backend+'_budgets']=dict(water_residual_m3=max(abs(v['water_residual_m3']) for v in diagnostics),
                sediment_residual_kg=max(abs(w) for v in diagnostics for w in v['class_residual_kg']),
                wall_integrals=max(abs(w) for v in diagnostics for w in v['wall_integrals']),
                min_stock_kg_m2=min(v['minimum_stock'] for v in diagnostics))
            assert r[backend+'_budgets']['water_residual_m3']<1e-8
            assert r[backend+'_budgets']['sediment_residual_kg']<1e-8
            assert r[backend+'_budgets']['wall_integrals']==0
            assert r[backend+'_budgets']['min_stock_kg_m2']>=0
        assert r['native_sediment_residual_m3']<1e-8
        assert abs(r['native_feed_m3']-.002)<1e-10 and abs(r['braids_feed_m3']-.002)<1e-10
        assert r['parity_bed']['maximum']<.001 and r['parity_depth']['maximum']<.001
        mobile[formula]=r;mobile_data[formula]=d
    sources={}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in ('mom_fls.f90','mom_cyclic.f90','upwhu.f90','cucbp.f90','rdbcg.f90','dwnvel.f90'):
            members=[p for p in archive.namelist() if p.endswith('/'+name)];assert len(members)==1
            path=ROOT/'resources'/members[0]
            assert archive.read(members[0])==path.read_bytes()
            sources[str(path.relative_to(ROOT))]=sha(path)
    save(OUT/'AUDIT.json',dict(cases=summary,contrasts=contrasts,long_hydraulic_backend_parity=long_parity,
        same_state=replays,ten_step_primary_velocity_replays=short_updates,native_Cyclic_minus_Flood_at40=native_scheme,mobile=mobile,
        read_only_Delft3D_source_archive_matches=sources,
        caveat='Hydraulic plots interpolate native U1/V1 to centres. Boundary replays use UUU/VVV characteristic-state projection, separately labelled. Neither is an identical staggered degree-of-freedom mapping.',
        installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        scripts={p.name:sha(p) for p in (ROOT/'desktop').glob('*162.py')}))
    save(OUT/'QUALIFICATION.json',dict(hydraulic_diagnostic_complete=True,common_inputs_verified=True,
        all_short_runs_complete=True,cpu_gpu_same_state_water_flux_exact=True,
        cpu_gpu_rate_roundoff=True,wall_and_volume_balance_pass=True,three_law_mobile_controls_pass=True,
        full_coupled_Delft3D_equivalence=False,sharp_groin_tip_bed_peak_agreement=False,
        production_numerical_fix_proven=False,installed_runtime_changed=False,
        Delft3D_source_or_runtime_modified=False,no_physical_coefficient_tuning=True,
        rejected_changes=['incoming tangent extrapolation','lower CFL as the supposed discrepancy fix']))
    fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    for key,label in (('short','20 m groin'),('long','40 m groin'),('straight','20 m straight')):
        rows=summary[key]['series'][:9];t=[r['time_s'] for r in rows]
        axes[0].plot(t,[r['common_first20m_depth']['rms']*1000 for r in rows],'o-',label=label)
    axes[0].set_ylabel('Depth RMS difference [mm], common x < 20 m');axes[0].legend(fontsize=8)
    for key,style in (('short','-'),('long','--')):
        d=data[key];frames=d['frames'][:9];t=d['store'].times[:9]
        bvol=[-f.arrays['water'][d['store']._boundary_mask(f,'Downstream')].sum() for f in frames]
        axes[1].plot(t,bvol,style,label='BRAIDS '+key)
        axes[1].plot(t,d['fltr'][:9,1],style,label='FLOW '+key)
    axes[1].set_ylabel('Cumulative net water export [m³]');axes[1].legend(fontsize=8)
    for source,style in (('FLOW','o-'),('BRAIDS','s--')):
        rows=[r for r in replays if r['source']==source]
        axes[2].plot([r['time_s'] for r in rows],[r['Q_out_m3_s'] for r in rows],style,label='BRAIDS on '+('FLOW characteristic projection' if source=='FLOW' else 'own state'))
    axes[2].plot(ref['store'].times[:9],ref['ctr'][:9,1],'k:',label='Native FLOW discharge')
    axes[2].set_ylabel('Instantaneous net outlet discharge [m³/s]');axes[2].legend(fontsize=7)
    for ax in axes:ax.set_xlabel('Physical time [s]');ax.grid(alpha=.2)
    fig.suptitle('Boundary operator versus evolving wake: same initial bed, depth, mesh spacing and Q')
    fig.savefig(OUT/'HYDRAULIC_DIAGNOSIS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for row,key in enumerate(('short','long')):
        d=data[key];mask=(d['xy'][:,0]>5)&(d['xy'][:,0]<20);polys=[np.array(p.exterior.coords) for p in d['frames'][0].polygons]
        for col,(u,h,title) in enumerate(((d['nu'][8],d['nh'][8],'FLOW'),(d['bu'][8],d['bh'][8],'BRAIDS'))):
            ax=axes[row,col];pc=PolyCollection([p for p,m in zip(polys,mask) if m],array=np.linalg.norm(u[:,mask],axis=0),cmap='Reds',clim=(0,3.5))
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');sel=np.flatnonzero(mask)[::2]
            ax.quiver(d['xy'][sel,0],d['xy'][sel,1],u[0,sel],u[1,sel],scale=35,width=.002)
            ax.set_title(f'{title}: {key} domain, t=40 s');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
            fig.colorbar(pc,ax=ax,label='Speed [m/s]',shrink=.75)
    fig.suptitle('Raw cells: the sharp-corner jet/wake differs before sediment moves')
    fig.savefig(OUT/'WAKE_COMPARISON.png',dpi=170);plt.close(fig)
    labels=['MPM','Ashida–Michiue','Wang–Fredsøe (bedload)']
    fig,axes=plt.subplots(3,3,figsize=(15,9),layout='constrained')
    for row,(formula,d) in enumerate(mobile_data.items()):
        bd=(d['bz'][-1]-d['bz'][0])*1000;nd=(d['nz'][-1]-d['nz'][0])*1000
        mask=d['common'];polys=[np.array(p.exterior.coords) for p,m in zip(d['frames'][0].polygons,mask) if m]
        limit=max(abs(bd[mask]).max(),abs(nd[mask]).max())
        for col,(values,title) in enumerate(((nd,'FLOW'),(bd,'BRAIDS'),(bd-nd,'BRAIDS − FLOW'))):
            ax=axes[row,col];pc=PolyCollection(polys,array=values[mask],cmap='RdBu_r',clim=(-limit,limit))
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(labels[row]+' | '+title,fontsize=10)
            ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');fig.colorbar(pc,ax=ax,label='Bed change [mm]',shrink=.7)
    fig.suptitle('40 m domain, first 20 m shown; t=60 s / 20 s mobile; raw 0.5 m cells')
    fig.savefig(OUT/'BEDLOAD_COMPARISON.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    for col,(formula,d) in enumerate(mobile_data.items()):
        ids=np.flatnonzero((d['xy'][:,1]==3.25)&(d['xy'][:,0]>5)&(d['xy'][:,0]<13));ids=ids[np.argsort(d['xy'][ids,0])]
        for val,label,style in ((d['bz'][-1]-d['bz'][0],'BRAIDS','s-'),(d['nz'][-1]-d['nz'][0],'FLOW','o--')):
            axes[0,col].plot(d['xy'][ids,0],val[ids]*1000,style,label=label)
        for val,label,style in ((d['bu'][8,0],'BRAIDS','s-'),(d['nu'][8,0],'FLOW','o--')):
            axes[1,col].plot(d['xy'][ids,0],val[ids],style,label=label)
        axes[1,col].plot(d['xy'][ids,0],d['nchar'][8,0,ids],':',color='grey',label='FLOW sediment characteristic')
        axes[0,col].set_title(labels[col]);axes[0,col].set_ylabel('Bed change at60 s [mm]')
        axes[1,col].set_ylabel('Streamwise velocity at40 s [m/s]')
        for ax in axes[:,col]:ax.axvspan(8,10,color='grey',alpha=.1);ax.axhline(0,color='grey',lw=.7);ax.set_xlabel('x [m]');ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,0].legend(fontsize=8);fig.suptitle('First fluid row above the groin: local peak/sign disagreement remains visible')
    fig.savefig(OUT/'GROIN_DETAIL.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,seconds in enumerate((10,30)):
        with np.load(BASE/'operator_v3/cpu'/f'STEP_FLOW_{seconds:04d}.npz') as d:
            a=d['braids_final'];b=d['native_final'];delta_u=np.linalg.norm(a[1:]/a[0]-b[1:]/b[0],axis=0)
            for col,(values,title,cmap) in enumerate(((1000*(a[0]-b[0]),'Depth difference [mm]','RdBu_r'),(delta_u,'Velocity-vector difference [m/s]','magma'))):
                ax=axes[row,col];limit=abs(values).max();pc=PolyCollection([np.array(p.exterior.coords) for p in ref['frames'][0].polygons],array=values,cmap=cmap,clim=(-limit,limit) if col==0 else (0,limit))
                ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(f'{seconds:g} s field + 10 steps: {title}');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');fig.colorbar(pc,ax=ax,shrink=.75)
    fig.suptitle('Only 0.01 s of hydraulic evolution: discrepancy starts at the groin, not the open boundaries')
    fig.savefig(OUT/'TEN_STEP_LOCALIZATION.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(contrasts=contrasts,long_hydraulic_backend_parity=long_parity,
        mobile={k:{name:v[name] for name in ('bed_difference','first20m_bed_difference','bed_correlation','parity_bed','parity_depth','BRAIDS','FLOW')} for k,v in mobile.items()}),indent=2))


if __name__=='__main__':main()
