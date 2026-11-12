"""Raw-cell figures and bounded qualification; no experimental-fit claim."""
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
from review_hydraulics162 import load,norm,verify_native
from braids_desktop.results import ResultsStore
BASE=ROOT/'validation/_groin163'
OLD=ROOT/'validation/_hydraulics162/mobile'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_groin163'


def budgets(run):
    rows=json.loads((run/'DIAGNOSTICS.json').read_text())
    result=dict(water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
                solid_residual_kg=max(abs(v) for r in rows for v in r['class_residual_kg']),
                wall_integral=max(abs(v) for r in rows for v in r['wall_integrals']),
                minimum_stock=min(r['minimum_stock'] for r in rows))
    assert result['water_residual_m3']<1e-8 and result['solid_residual_kg']<1e-8
    assert result['wall_integral']==0 and result['minimum_stock']>=0
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={};data={};olddata={}
    for law in SPECS:
        run=BASE/f'coupled/braids_{law}_cpu/run'
        a,r=load(run,OLD/f'delft_{law}')
        old,_=load(OLD/f'braids_{law}_cpu/run',OLD/f'delft_{law}')
        gpu=ResultsStore(BASE/f'coupled/braids_{law}_gpu/run').read(-1)
        store=a['store'];final=a['frames'][-1]
        feed=float(final.arrays['bed_boundary'][:,store._boundary_mask(final,'Upstream')].sum())
        assert abs(feed-.002)<1e-12
        assert r['native_initial_stock_kg_m2']==[159.,159.]
        assert abs(a['frames'][0].arrays['mass'].sum(axis=(0,1))-159.).max()<1e-10
        assert abs(a['bz'][:9]-a['bz'][0]).max()==0
        z=a['bz'][-1]-a['bz'][0];nz=a['nz'][-1]-a['nz'][0]
        oz=old['bz'][-1]-old['bz'][0]
        r.update(braids_feed_m3=feed,bed=norm(z-nz),bed_common=norm((z-nz)[a['common']]),old_bed=norm(oz-nz),
                 braids_scour_m=float(-z.min()),braids_deposition_m=float(z.max()),
                 native_scour_m=float(-nz.min()),native_deposition_m=float(nz.max()),
                 cpu_gpu_bed=norm(a['frames'][-1].z-gpu.z),
                 cpu_gpu_q=norm(a['frames'][-1].q-gpu.q),
                 correction_bed=norm(z-oz),
                 cpu=budgets(run),gpu=budgets(BASE/f'coupled/braids_{law}_gpu/run'))
        assert r['cpu_gpu_bed']['maximum']<1e-7 and r['cpu_gpu_q']['maximum']<1e-7
        results[law]=r;data[law]=a;olddata[law]=old
    restart={}
    for backend in ('cpu','gpu'):
        reference=ResultsStore(BASE/f'coupled/braids_mpm_{backend}/run').read(-1)
        resumed=ResultsStore(BASE/f'restart_mobile_final/braids_mpm_{backend}/run').read(-1)
        fields={k:bool(np.array_equal(reference.arrays[k],resumed.arrays[k]))
                for k in reference.arrays if k in resumed.arrays and reference.arrays[k].dtype.kind not in 'OUS'}
        # Snapshot clock counters may describe this segment, not the full run.
        for k in ('q','z','mass','water','bed_boundary'):
            assert fields[k],k
        restart[backend]=dict(start_s=50,end_s=60,exact_fields=fields,
                               budgets=budgets(BASE/f'restart_mobile_final/braids_mpm_{backend}/run'))
    affine={}
    for name,slope in [('AFFINE',.2),('INWARD',-.2),('FAST',2.),('TANGENT',0.)]:
        a=np.load(BASE/f'affine_all/cpu/{name}_0010.npz')
        b=np.load(BASE/f'reflected_affine_all/cpu/{name}_0010.npz')
        g=np.load(BASE/f'reflected_affine_all/gpu/{name}_0010.npz')
        wall=(a['neighbour']<0)&(a['bid']<0)&np.isclose(a['mid'][1],3)&(a['mid'][0]>8)&(a['mid'][0]<10)
        before=a['trace'][0,2,wall];after=b['trace'][0,2,wall]
        assert abs(after).max()<1e-13
        assert abs(b['trace'][0,1,wall]-1.).max()<1e-13
        np.testing.assert_allclose(b['rate'],g['rate'],atol=2e-12,rtol=2e-12)
        assert np.max(abs(b['water'][wall]))==0
        affine[name]=dict(slope_s_inv=slope,before_normal_trace_max_m_s=float(abs(before).max()),
                          after_normal_trace_max_m_s=float(abs(after).max()),
                          cpu_gpu_rate=norm(b['rate']-g['rate']),wall_water_flux_max=0.)
    operator={}
    for sec in (10,30,40):
        for label in ('BRAIDS','FLOW_PRIMARY','FLOW_CHARACTERISTIC'):
            a=np.load(BASE/f'cpu/{label}_{sec:04d}.npz');b=np.load(BASE/f'gpu/{label}_{sec:04d}.npz')
            delta={k:norm(a[k]-b[k]) for k in ('rate','water','bedcell','effective_q','class_flux')}
            assert max(v['maximum'] for v in delta.values())<1e-11
            operator[f'{label}_{sec}']=delta
    a=np.load(BASE/'gpu/FLOW_PRIMARY_0040.npz')
    i=int(np.argmin(np.sum((a['xy']-[7.75,3.25])**2,axis=1)))
    fs=np.where((a['owner']==i)|(a['neighbour']==i))[0]
    sgn=np.where(a['owner'][fs]==i,1.,-1.)
    flux=np.where((a['owner'][fs]==i)[None,:],a['fl'][:,fs],a['fr'][:,fs])
    pieces=-sgn*flux[1]*a['length'][fs]/a['area'][i]
    transverse=np.abs(a['normal'][1,fs])>.5
    corner=dict(xy=a['xy'][i].tolist(),face_ids=fs.tolist(),face_midpoints=a['mid'][:,fs].T.tolist(),
                streamwise_momentum_rate_per_face=pieces.tolist(),
                transverse_advection_rate=float(pieces[transverse].sum()),
                full_spatial_momentum_rate=float(a['rate'][1,i]))
    with Dataset(OLD/'delft_mpm/trim-sim_P057.nc') as ds:
        kfu=np.asarray(ds['KFU'][8]);kfv=np.asarray(ds['KFV'][8])
        active=kfu.astype(bool)
        north=kfv*np.roll(kfv,-1,axis=0)
        south=np.roll(north,1,axis=1)
        gate=active & ((north==0)|(south==0))
        corner['native_east_u_point']=[16,7]
        corner['native_south_ub_is_zero_by_wall_mask']=bool(south[16,7]==0)
        corner['native_north_stencil_open']=bool(north[16,7]!=0)
        corner['native_cross_advection_suppressed']=bool(gate[16,7])
        assert corner['native_cross_advection_suppressed']
    cyclic,cycr=load(BASE/'coupled/braids_mpm_cpu/run',BASE/'delft_mpm_cyclic')
    ref=data['mpm']
    cycr.update(native_momentum_setting_depth_change_40s=norm(cyclic['nh'][8]-ref['nh'][8]),
                native_momentum_setting_bed_change_60s=norm(cyclic['nz'][-1]-ref['nz'][-1]),
                caveat='Cyclic changes the whole native momentum discretization, not just one corner guard; not adopted as acceptance reference.')
    sources={}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in ('mom_fls.f90','upwhu.f90','dwnvel.f90','upwbed.f90','adjust_bedload.f90'):
            members=[n for n in archive.namelist() if n.endswith('/'+name)];assert len(members)==1
            path=ROOT/'resources'/members[0]
            assert path.read_bytes()==archive.read(members[0])
            sources[str(path.relative_to(ROOT))]=sha(path)
    maintained=[ROOT/'braids_core/sediment'/name for name in ('wall_bed_hydraulics.py','wall_velocity_bounds.py')]
    tests=[ROOT/'tests'/name for name in ('test_wall_velocity_bounds163.py','test_wall_bed_hydraulics.py')]
    save(OUT/'AUDIT.json',dict(mobile=results,restart=restart,affine=affine,same_state_backend=operator,
         corner=corner,native_cyclic_control=cycr,read_only_Delft3D_sources=sources,
         installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
         maintained_checkout={str(p.relative_to(ROOT)):sha(p) for p in maintained+tests},
         face_chain_sha256=sha(OUT/'FACE_CHAIN.json'),
         scripts={p.name:sha(p) for p in (ROOT/'desktop').glob('*163.py')}))
    save(OUT/'QUALIFICATION.json',dict(wall_bound_defect_reproduced=True,
         reflected_bounds_pass_affine_and_three_law_coupled_checks=True,cpu_gpu_pass=True,
         sediment_restart_exact=True,conservation_pass=True,
         corrected_wall_transform_in_maintained_checkout=True,
         coupled_tests_use_identified_isolated_transform=True,
         public_package_qualification=False,
         main_groin_discrepancy_closed=False,experimental_validation=False,
         no_transport_coefficient_tuning=True,no_Delft3D_source_changes=True,installed_desktop_changed=False))

    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    xs=np.arange(4)
    for name,color in [('AFFINE','#2563eb'),('INWARD','#b91c1c')]:
        before=np.load(BASE/f'affine_all/cpu/{name}_0010.npz')
        after=np.load(BASE/f'reflected_affine_all/cpu/{name}_0010.npz')
        wall=(before['neighbour']<0)&np.isclose(before['mid'][1],3)&(before['mid'][0]>8)&(before['mid'][0]<10)
        axes[0].plot(before['mid'][0,wall],before['trace'][0,2,wall],'o--',color=color,label=f'{name.lower()}: before')
        axes[0].plot(after['mid'][0,wall],after['trace'][0,2,wall],'x-',color=color,label=f'{name.lower()}: corrected')
    axes[0].axhline(0,color='k',lw=.7);axes[0].set(xlabel='Wall position x (m)',ylabel='Normal trace velocity (m/s)',title='Manufactured linear field: expected 0')
    axes[0].legend(fontsize=8)
    axes[1].bar(xs,pieces,color=['#2563eb' if v else '#b45309' for v in transverse])
    axes[1].set_xticks(xs,[f'({x:g}, {y:g})' for x,y in a['mid'][:,fs].T],rotation=25)
    axes[1].axhline(0,color='k',lw=.7);axes[1].set(ylabel='Contribution to d(hu)/dt (m²/s²)',title='BRAIDS on native primary projection\nCell (7.75, 3.25), t = 40 s',xlabel='Incident face midpoint (m)')
    xx,yy=np.meshgrid(np.arange(82)*.5,(np.arange(18)-.5)*.5,indexing='ij')
    view=gate&(xx>6)&(xx<12)&(yy>2)&(yy<5)
    axes[2].scatter(xx[view],yy[view],marker='x',color='#b91c1c',label='FLOW Flood: transverse term gated')
    axes[2].fill([8,10,10,8],[0,0,3,3],color='.4');axes[2].plot(7.75,3.25,'o',color='#2563eb',label='Audited BRAIDS cell')
    axes[2].set(xlim=(6.5,11.5),ylim=(2,5),aspect='equal',title='Different momentum locations / masks',xlabel='x (m)',ylabel='y (m)');axes[2].legend(fontsize=8)
    fig.savefig(OUT/'WALL_AND_CORNER.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for row,law in enumerate(SPECS):
        d=data[law];od=olddata[law]
        values=[od['bz'][-1]-od['bz'][0],d['bz'][-1]-d['bz'][0],d['nz'][-1]-d['nz'][0]]
        limit=max(abs(v).max() for v in values)*1000
        polygons=[np.asarray(p.exterior.coords) for p in d['frames'][0].polygons]
        for col,(title,v) in enumerate(zip(('BRAIDS before','BRAIDS corrected wall bounds','Unmodified FLOW Flood'),values)):
            ax=axes[row,col]
            mesh=PolyCollection(polygons,array=v*1000,cmap='RdBu_r',clim=(-limit,limit),edgecolors=(0,0,0,.15),linewidths=.3)
            ax.add_collection(mesh);ax.fill([8,10,10,8],[0,0,3,3],color='.35')
            ax.set(xlim=(6,13),ylim=(1.5,6),aspect='equal',title=f'{title}\n{law.replace("_generalized","").replace("_"," ")} / t = 60 s',xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(mesh,ax=axes[row,:],label='Bed change (mm)',shrink=.7)
    fig.savefig(OUT/'BEDLOAD_RAW_CELLS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    xy=ref['xy'];sel=np.isclose(xy[:,1],3.25)&(xy[:,0]>=6)&(xy[:,0]<=13)
    order=np.where(sel)[0][np.argsort(xy[sel,0])];x=xy[order,0]
    for title,d,color in [('BRAIDS before',olddata['mpm'],'#64748b'),('BRAIDS corrected',ref,'#2563eb')]:
        axes[0,0].plot(x,d['bh'][8,order],'o-',label=title,color=color)
        axes[0,1].plot(x,d['bu'][8,0,order],'o-',label=title,color=color)
        axes[1,0].plot(x,1000*(d['bz'][-1,order]-d['bz'][0,order]),'o-',label=title,color=color)
    for title,d,color in [('FLOW Flood',ref,'#b45309'),('FLOW Cyclic control',cyclic,'#9333ea')]:
        axes[0,0].plot(x,d['nh'][8,order],'s--',label=title,color=color)
        axes[0,1].plot(x,d['nu'][8,0,order],'s--',label=title+' primary average',color=color)
        axes[1,0].plot(x,1000*(d['nz'][-1,order]-d['nz'][0,order]),'s--',label=title,color=color)
    axes[0,1].plot(x,ref['nchar'][8,0,order],':',color='#b45309',label='FLOW Flood sediment characteristic')
    for label,color in [('BRAIDS','#2563eb'),('FLOW_PRIMARY','#16a34a'),('FLOW_CHARACTERISTIC','#b45309')]:
        arr=np.load(BASE/f'cpu/{label}_0040.npz')
        mag=np.linalg.norm(arr['bedcell'][3:5],axis=0)/2650
        axes[1,1].plot(x,mag[order]*1e3,'o-',label=label.replace('_',' '),color=color)
    axes[0,0].set(ylabel='Water depth (m)',title='At 40 s, before bed moves')
    axes[0,1].set(ylabel='Streamwise velocity (m/s)',title='Primary and sediment observables kept distinct')
    axes[1,0].set(ylabel='MPM bed change (mm)',title='At 60 s: 20 s mobile, not equilibrium')
    axes[1,1].set(ylabel='MPM capacity (L/s/m)',title='Public law replay at 40 s, different state inputs')
    for ax in axes.flat:
        ax.set_xlabel('x (m), first row above groin: y = 3.25 m');ax.axhline(0,color='k',lw=.5);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'TIP_SECTIONS.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(mobile={k:{n:r[n] for n in ('bed_common','cpu_gpu_bed','correction_bed')} for k,r in results.items()},corner=corner),indent=2))


if __name__=='__main__':main()
