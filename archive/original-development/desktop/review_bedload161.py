"""Authenticate and compare raw matched groin results; never smooth solver fields."""
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
from check_bedload161 import BASE,SPECS,sha,save,FEED,STOCK,SPIN
from braids_desktop.results import ResultsStore
from braids_core.sediment.river_capacity import ResolvedRiverCapacity
from braids_core.sediment.bed_columns_v101 import host
def verify(folder):
    inputs=json.loads((folder/'INPUTS.json').read_text())
    for name,checksum in inputs['inputs'].items():
        assert (folder/name).resolve().is_relative_to(folder.resolve()) and sha(folder/name)==checksum,name
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['success'] and execution['executable_sha256']=='09f451b673fcf855ae8f07d819686e7a429ab9382ff6da372702aa3a46583623'
    for row in execution['outputs']:assert sha(folder/row['name'])==row['sha256']
    return execution
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bedload161'
OUT.mkdir(parents=True,exist_ok=True)
LABELS={'mpm':'MPM','ashida_michiue_generalized':'Ashida–Michiue','wang_fredsoe':'Wang–Fredsøe (bedload)'}
COLORS={'mpm':'#2368a2','ashida_michiue_generalized':'#c45228','wang_fredsoe':'#6b47a1'}

def stats(a,b):
    delta=a-b
    return dict(rms=float(np.sqrt(np.mean(delta**2))),maximum=float(abs(delta).max()),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))

def replay(ds,formula,backend,ix,iy):
    op=ResolvedRiverCapacity(backend,diameter=.001,chezy=45.,capacity=dict(method='capacity.'+formula,parameters=SPECS[formula][1]),folder=BASE/'replay'/formula/backend)
    xp=op.xp;rows=[]
    for it in (8,9,24):
        get=lambda k:np.asarray(ds[k][it]).squeeze()[...,ix,iy]
        h=get('S1')+get('DPS');umod=get('UMOD');ustar=get('USTAR')
        z0=get('ZUMOD')/np.expm1(.41*umod/np.maximum(ustar,1e-30))
        c=np.maximum(20.,np.sqrt(9.81)/.41*np.log1p(np.maximum(.01,h)/(np.e*z0)))
        q=np.ascontiguousarray([h,h*get('UUU'),h*get('VVV')])
        factor=.41*c/(np.sqrt(9.81)*np.log1p(h*float(np.float32(.368))/z0))
        cq=q.copy();cq[1:]*=factor
        op.z0=xp.asarray(z0);op.d90=xp.full(len(h),.001000000008);op.hiding=xp.ones(len(h))
        target=np.array([get('SBCU'),get('SBCV')])
        got=host(op.evaluate(xp.asarray(q),xp.zeros(len(h)),capacity_q=xp.asarray(cq),chezy=xp.asarray(c)))[3:5]/2650
        rows.append(dict(time_s=float(ds['time'][it]),maximum_m2_s=float(abs(got-target).max()),
            relative_l2=float(np.linalg.norm(got-target)/max(np.linalg.norm(target),1e-30)),
            native_peak_m2_s=float(np.linalg.norm(target,axis=0).max())))
    return rows

def main(do_replay=False):
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    summary={};all_data={}
    for formula in SPECS:
        native=BASE/('delft_'+formula+'_v8');receipt=verify(native)
        stores=[ResultsStore(BASE/('braids_'+formula+'_'+b)/'run') for b in ('cpu','gpu')]
        c,g=stores; f0=c.read(0);xy=np.array([p.centroid.coords[0] for p in f0.polygons]);area=f0.arrays['area']
        assert np.array_equal(area,np.full(616,.25))
        ix=np.rint(xy[:,0]/.5+.5).astype(int);iy=np.rint(xy[:,1]/.5+.5).astype(int)
        frames=[c.read(i) for i in range(len(c.times))]
        with Dataset(native/'trim-sim_P057.nc') as ds:
            mask=np.asarray(ds['KCS'][:])==1
            assert mask.sum()==616==len(xy) and mask[ix,iy].all()
            assert np.array_equal(np.asarray(ds['XZ'][:])[ix,iy],xy[:,0])
            assert np.array_equal(np.asarray(ds['YZ'][:])[ix,iy],xy[:,1])
            assert np.array_equal(ds['time'][:],c.times)
            nz=-np.asarray(ds['DPS'][:])[:,ix,iy]
            nh=np.asarray(ds['S1'][:])[:,ix,iy]-nz
            nu=np.stack([np.asarray(ds[k][:])[:,ix,iy] for k in ('UUU','VVV')],axis=1)
            mass=np.asarray(ds['MSED'][:])[:,0,0,ix,iy]
            assert np.max(abs(nz[0]-f0.z))<1e-12
            assert np.max(abs(nh[0]-f0.q[0]))<3e-8
            assert np.max(abs(mass[0]-f0.arrays['mass'].sum(axis=(0,1))))<1e-9
            assert np.max(abs(nz[:9]-nz[0]))==0
            assert np.max(abs(np.array([f.z for f in frames[:9]])-f0.z))==0
            # FLOW SBUUA/SBVVA are cumulative morphological-time means, not
            # last-output-interval means. Their exterior integral closes Exner.
            morph_time=np.asarray(ds['MFTAVG'][:])*86400
            flux_u=np.asarray(ds['SBUUA'][:])[:,0]
            outlet_bedload=np.asarray(ds['SBUU'][:])[:,0,40,1:17]
            outlet_water=np.asarray(ds['U1'][:])[:,0,40,1:17]
            vin=flux_u[:,0,1:17].sum(axis=1)*.5*morph_time
            vout=flux_u[:,40,1:17].sum(axis=1)*.5*morph_time
            vbed=np.sum((mass-mass[0])*area,axis=1)/2650
            budget=vbed-(vin-vout)
            bz=np.array([f.z for f in frames]);bh=np.array([f.q[0] for f in frames])
            bu=np.array([f.q[1:]/f.q[0] for f in frames])
            nd=nz-nz[0];bd=bz-bz[0]
            final=frames[-1];gf=g.read(-1)
            diagnostics=[r for store in stores for r in json.loads((store.output/'DIAGNOSTICS.json').read_text())]
            parity={k:float(np.max(abs(final.arrays[k]-gf.arrays[k]))) for k in ('q','z','mass','water','bed_boundary')}
            parity['depth_m']=float(np.max(abs(final.q[0]-gf.q[0])))
            parity['bed_rms_m']=float(np.sqrt(np.mean((final.z-gf.z)**2)))
            parity['strict_0_1_mm_bed_pass']=parity['z']<1e-4
            # User accepts millimetre-scale coupled differences. Preserve the
            # tighter failed screen explicitly rather than relabelling it.
            assert parity['z']<.001 and parity['depth_m']<.001
            r=dict(native_execution=receipt,cells=len(xy),initial=dict(bed_max_m=float(abs(nz[0]-f0.z).max()),
                depth_max_m=float(abs(nh[0]-f0.q[0]).max()),stock_max_kg_m2=float(abs(mass[0]-159).max())),
                parity=parity,hydraulic_at40=dict(depth=stats(bh[8],nh[8]),velocity=stats(bu[8],nu[8])),
                hydraulic_change_30_to40=dict(BRAIDS_depth=stats(bh[8],bh[6]),Delft3D_depth=stats(nh[8],nh[6])),
                bed_at120=stats(bd[-1],nd[-1]),native_budget_residual_m3=float(abs(budget).max()),
                native_solid_feed_m3=float(vin[-1]),native_solid_export_m3=float(vout[-1]),
                braids_water_residual_m3=max(abs(x['water_residual_m3']) for x in diagnostics),
                braids_sediment_residual_kg=max(abs(v) for x in diagnostics for v in x['class_residual_kg']),
                wall_water_sediment_integral=max(abs(v) for x in diagnostics for v in x['wall_integrals']),
                minimum_braids_stock_kg_m2=min(x['minimum_stock'] for x in diagnostics),
                braids_solid_feed_m3=float(final.arrays['bed_boundary'][:,c._boundary_mask(final,'Upstream')].sum()),
                braids_solid_export_m3=float(-final.arrays['bed_boundary'][:,c._boundary_mask(final,'Downstream')].sum()),
                minimum_delft_stock_kg_m2=float(mass.min()))
            r['outlet_audit']=dict(hydraulic_policy=c.project['boundary_states']['Downstream']['stage_policy'],
                native_minimum_water_velocity_m_s=float(outlet_water.min()),
                native_minimum_signed_bedload_m2_s=float(outlet_bedload[9:].min()),
                native_inward_water_face_samples=int(np.sum(outlet_water[9:]<0)),
                native_inward_sediment_face_samples=int(np.sum(outlet_bedload[9:] < -1e-16)))
            assert r['outlet_audit']['hydraulic_policy']=='stage.subcritical_backflow'
            assert r['outlet_audit']['native_inward_sediment_face_samples']==0
            assert r['braids_water_residual_m3']<1e-8 and r['braids_sediment_residual_kg']<1e-8
            assert r['wall_water_sediment_integral']==0 and r['minimum_braids_stock_kg_m2']>=0
            assert r['minimum_delft_stock_kg_m2']>=0 and r['native_budget_residual_m3']<1e-8
            assert abs(r['braids_solid_feed_m3']-.008)<1e-10 and abs(vin[-1]-.008)<1e-10
            for key,d in [('BRAIDS',bd),('Delft3D',nd)]:
                last=d[-1];r[key]=dict(scour_mm=float(-last.min()*1000),deposition_mm=float(last.max()*1000),
                    scour_xy_m=xy[last.argmin()].tolist(),deposition_xy_m=xy[last.argmax()].tolist(),
                    eroded_solid_m3=float(np.sum(np.maximum(-last,0)*area)*.6),
                    deposited_solid_m3=float(np.sum(np.maximum(last,0)*area)*.6),
                    net_solid_m3=float(np.sum(last*area)*.6))
            if do_replay:
                r['same_input_operator']={b:replay(ds,formula,b,ix,iy) for b in ('cpu','gpu')}
                assert max(v['relative_l2'] for rows in r['same_input_operator'].values() for v in rows)<1e-5
            summary[formula]=r;all_data[formula]=(frames,xy,nd,bd,nh,bh,nu,bu,vin,vout,vbed)
    formula='ashida_michiue_generalized'
    half_b=BASE/('braids_'+formula+'_cpu_dt0.05')
    half_d=BASE/('delft_'+formula+'_v8_dt0.005')
    if (half_b/'STATUS.json').exists() and (half_d/'execution.json').exists():
        verify(half_d)
        regular=ResultsStore(BASE/('braids_'+formula+'_cpu')/'run').read(-1)
        half=ResultsStore(half_b/'run').read(-1)
        with Dataset(BASE/('delft_'+formula+'_v8')/'trim-sim_P057.nc') as a, Dataset(half_d/'trim-sim_P057.nc') as b:
            summary['timestep_control']=dict(BRAIDS_bed_m=stats(regular.z,half.z),BRAIDS_depth_m=stats(regular.q[0],half.q[0]),
                Delft3D_bed_m=stats(np.asarray(a['DPS'][-1])[ix,iy],np.asarray(b['DPS'][-1])[ix,iy]),
                Delft3D_depth_m=stats(np.asarray(a['S1'][-1]+a['DPS'][-1])[ix,iy],np.asarray(b['S1'][-1]+b['DPS'][-1])[ix,iy]))
    save(OUT/'COMPARISON.json',summary)
    save(OUT/'QUALIFICATION.json',dict(
        runtime_manifest_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        scope='Single-grain bedload; static aligned 0.5 m erodible groin; prescribed feed; 120 physical seconds',
        source_law_replay_pass=do_replay,
        initial_geometry_bed_stock_pass=True,
        common_feed_conservation_nonnegative_stock_wall_flux_pass=True,
        cpu_gpu_bed_within_0_1_mm=all(summary[k]['parity']['strict_0_1_mm_bed_pass'] for k in SPECS),
        cpu_gpu_bed_and_depth_within_1_mm=True,
        full_coupled_Delft3D_equivalence=False,
        reason='Transport-law replay passes; transient hydraulic forcing and local groin peak differences remain.',
        no_new_physics=True,no_solver_modification=True,no_Delft3D_source_or_runtime_modification=True,
        scripts={p.name:sha(p) for p in (Path(__file__),ROOT/'desktop/check_bedload161.py',ROOT/'desktop/run_bedload161.py')},
        native_output_histories='History SBTR/SBTRC are zero in this bedload-only configuration; use raw morphology-averaged face fluxes and check global bed inventory instead.'))
    fig,axes=plt.subplots(3,3,figsize=(16,9),layout='constrained')
    for row,(formula,values) in enumerate(all_data.items()):
        frames,xy,nd,bd,*_=values;lim=max(abs(nd[-1]).max(),abs(bd[-1]).max())*1000
        for col,(data,title) in enumerate(((nd[-1],'Delft3D 2DH'),(bd[-1],'BRAIDS CPU'),(bd[-1]-nd[-1],'BRAIDS − Delft3D'))):
            ax=axes[row,col];pc=PolyCollection([np.asarray(p.exterior.coords) for p in frames[0].polygons],array=data*1000,cmap='RdBu_r',clim=(-lim,lim),edgecolors='none')
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(LABELS[formula]+' | '+title,fontsize=10)
            ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');fig.colorbar(pc,ax=ax,label='Bed change [mm]',shrink=.7)
    fig.suptitle('Same initial bed, grain and total feed — 120 s (80 s mobile bed); raw 0.5 m cells')
    fig.savefig(OUT/'BED_COMPARISON.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for formula,values in all_data.items():
        frames,xy,nd,bd,*_=values;t=np.array([f.time_s for f in frames]);area=frames[0].arrays['area']
        for d,label,style in ((bd,'BRAIDS','-'),(nd,'Delft3D','--')):
            axes[0].plot(t,-d.min(axis=1)*1000,style,color=COLORS[formula],label=LABELS[formula]+' '+label)
            axes[1].plot(t,d.max(axis=1)*1000,style,color=COLORS[formula])
            axes[2].plot(t,np.sum(d*area,axis=1)*.6,style,color=COLORS[formula])
    for ax,label in zip(axes,['Maximum scour [mm]','Maximum deposition [mm]','Net solid bed change [m³]']):
        ax.set_xlabel('Physical time [s]');ax.set_ylabel(label);ax.axvline(SPIN,color='grey',lw=.8);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(OUT/'EVOLUTION.png',dpi=160);plt.close(fig)
    frames,xy,nd,bd,nh,bh,nu,bu,*_=all_data['mpm']
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,(hh,uu,title) in enumerate(((nh[8],nu[8],'Delft3D 2DH'),(bh[8],bu[8],'BRAIDS'))):
        for row,(values,label,cmap) in enumerate(((hh,'Depth [m]','Blues'),(np.linalg.norm(uu,axis=0),'Speed [m/s]','Reds'))):
            ax=axes[row,col];lim=(0,max(nh[8].max(),bh[8].max())) if row==0 else (0,max(np.linalg.norm(nu[8],axis=0).max(),np.linalg.norm(bu[8],axis=0).max()))
            pc=PolyCollection([np.asarray(p.exterior.coords) for p in frames[0].polygons],array=values,cmap=cmap,clim=lim,edgecolors='none');ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(title+' | '+label);fig.colorbar(pc,ax=ax,shrink=.8)
    fig.suptitle('Hydraulic state at 40 s, before any bed update; no turbulence or secondary-flow closure')
    fig.savefig(OUT/'HYDRAULICS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,6),layout='constrained')
    for col,(formula,values) in enumerate(all_data.items()):
        frames,xy,nd,bd,nh,bh,nu,bu,*_=values
        for row,y in enumerate((3.25,3.75)):
            select=(xy[:,1]==y)&(xy[:,0]>5)&(xy[:,0]<13)
            ids=np.flatnonzero(select);ids=ids[np.argsort(xy[ids,0])]
            ax=axes[row,col];ax.plot(xy[ids,0],nd[-1,ids]*1000,'o--',label='Delft3D')
            ax.plot(xy[ids,0],bd[-1,ids]*1000,'s-',label='BRAIDS')
            ax.axhline(0,color='grey',lw=.7);ax.axvspan(8,10,alpha=.1,color='grey')
            ax.set_title(LABELS[formula]+f' | y={y:g} m');ax.set_xlabel('x [m]');ax.set_ylabel('Bed change [mm]');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Groin-tip detail at 120 s — first and second fluid rows above the solid')
    fig.savefig(OUT/'GROIN_DETAIL.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for formula,values in all_data.items():
        frames,xy,nd,bd,nh,bh,nu,bu,vin,vout,vbed=values;t=np.array([f.time_s for f in frames]);area=frames[0].arrays['area']
        axes[0].plot(t,vbed,'--',color=COLORS[formula],label=LABELS[formula]+' Delft3D')
        axes[0].plot(t,np.sum(bd*area,axis=1)*.6,'-',color=COLORS[formula],label=LABELS[formula]+' BRAIDS')
        r=summary[formula]
        axes[1].plot(t,vin,color=COLORS[formula],label=LABELS[formula]+' Delft3D')
        store=ResultsStore(BASE/('braids_'+formula+'_cpu')/'run')
        bfeed=[f.arrays['bed_boundary'][:,store._boundary_mask(f,'Upstream')].sum() for f in frames]
        axes[1].plot(t,bfeed,'--',color=COLORS[formula],label=LABELS[formula]+' BRAIDS')
    for ax,title in zip(axes,['Net solid bed change [m³]','Cumulative supplied solid volume [m³]']):
        ax.set_ylabel(title);ax.set_xlabel('Physical time [s]');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'BUDGETS.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main('--replay' in sys.argv)
