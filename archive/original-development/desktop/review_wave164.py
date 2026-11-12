"""Short-window flux qualification; preserve source/native provenance and raw cells."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import SPECS,save,sha
from review_hydraulics162 import load,norm
from review_groin163 import budgets
from braids_desktop.results import ResultsStore

BASE=ROOT/'validation/_wave164'
NATIVE=ROOT/'validation/_hydraulics162/mobile'
PREVIOUS=ROOT/'validation/_groin163/coupled'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wave164'
LABELS={'mpm':'MPM','ashida_michiue_generalized':'Ashida–Michiue','wang_fredsoe':'Wang–Fredsøe'}


def probe_rate(a,i,flux):
    fs=np.where((a['owner']==i)|(a['neighbour']==i))[0]
    sign=np.where(a['owner'][fs]==i,1.,-1.)
    return float(-np.sum(sign*flux[fs]*a['length'][fs])/(.6*a['area'][i]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit={};data={};before={};tip={}
    for law in SPECS:
        run=BASE/f'coupled/capacity_diagonal_{law}_cpu/run'
        a,r=load(run,NATIVE/f'delft_{law}')
        old,_=load(PREVIOUS/f'braids_{law}_cpu/run',NATIVE/f'delft_{law}')
        gpu=ResultsStore(BASE/f'coupled/capacity_diagonal_{law}_gpu/run').read(-1)
        i=int(np.argmin(np.sum((a['xy']-[8.25,3.25])**2,axis=1)))
        dz=a['bz'][-1]-a['bz'][0];native=a['nz'][-1]-a['nz'][0];oldz=old['bz'][-1]-old['bz'][0]
        maintained={}
        for backend in ('cpu','gpu'):
            p=BASE/f'maintained/{law}_{backend}/run'
            f=ResultsStore(p).read(-1)
            reference=ResultsStore(BASE/f'coupled/capacity_diagonal_{law}_{backend}/run').read(-1)
            fields={k:norm(f.arrays[k]-reference.arrays[k]) for k in ('q','z','mass','water','bed_boundary')}
            assert all(v['maximum']==0 for v in fields.values()),fields
            maintained[backend]=dict(isolated_comparison=fields,budgets=budgets(p))
        r.update(bed=norm(dz-native),bed_common=norm((dz-native)[a['common']]),
                 previous_bed_common=norm((oldz-native)[a['common']]),
                 maximum_scour_m=float(-dz.min()),maximum_deposition_m=float(dz.max()),
                 scour_xy=a['xy'][dz.argmin()].tolist(),deposition_xy=a['xy'][dz.argmax()].tolist(),
                 native_scour_m=float(-native.min()),native_deposition_m=float(native.max()),
                 native_scour_xy=a['xy'][native.argmin()].tolist(),native_deposition_xy=a['xy'][native.argmax()].tolist(),
                 cpu_gpu_bed=norm(a['frames'][-1].z-gpu.z),cpu_gpu_q=norm(a['frames'][-1].q-gpu.q),
                 cpu=budgets(run),gpu=budgets(BASE/f'coupled/capacity_diagonal_{law}_gpu/run'),maintained=maintained)
        assert r['cpu_gpu_bed']['maximum']<1e-7
        feed=float(a['frames'][-1].arrays['bed_boundary'][:,a['store']._boundary_mask(a['frames'][-1],'Upstream')].sum())
        assert abs(feed-.002)<1e-12
        assert abs(a['bz'][:9]-a['bz'][0]).max()==0
        r['feed_m3']=feed
        tip[law]=dict(xy=a['xy'][i].tolist(),time_s=a['store'].times.tolist(),
                      before_mm=(1000*(old['bz'][:,i]-old['bz'][0,i])).tolist(),
                      after_mm=(1000*(a['bz'][:,i]-a['bz'][0,i])).tolist(),
                      native_mm=(1000*(a['nz'][:,i]-a['nz'][0,i])).tolist())
        audit[law]=r;data[law]=a;before[law]=old
    candidate_metrics={}
    for mode in ('hll','bed_diagonal','morph_llf','capacity_diagonal'):
        a,_=load(BASE/f'coupled/{mode}_mpm_cpu/run',NATIVE/'delft_mpm')
        dz=a['bz'][-1]-a['bz'][0];nz=a['nz'][-1]-a['nz'][0]
        i=np.argmin(np.sum((a['xy']-[8.25,3.25])**2,axis=1))
        candidate_metrics[mode]=dict(bed_common=norm((dz-nz)[a['common']]),
            scour_mm=float(-1000*dz.min()),deposition_mm=float(1000*dz.max()),tip_mm=float(1000*dz[i]))
    restart={}
    for backend in ('cpu','gpu'):
        a=ResultsStore(BASE/f'maintained/mpm_{backend}/run').read(-1)
        b=ResultsStore(BASE/f'maintained_final/mpm_{backend}/run').read(-1)
        fields={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
        assert all(fields.values())
        restart[backend]=dict(start_s=50.,end_s=60.,exact=fields,budgets=budgets(BASE/f'maintained_final/mpm_{backend}/run'))
    half=ResultsStore(BASE/'half_step/capacity_diagonal_mpm_cpu/run').read(-1)
    dt=dict(dt_max_before_s=.1,dt_max_after_s=.05,cfl_before=1.,cfl_after=.5,
            bed=norm(half.z-data['mpm']['frames'][-1].z),depth=norm(half.q[0]-data['mpm']['frames'][-1].q[0]))
    assert dt['bed']['maximum']<1e-4
    decomposition={}
    for label in ('BRAIDS','FLOW_PRIMARY','FLOW_CHARACTERISTIC'):
        a=np.load(BASE/f'probe/matrix_mpm_cpu/{label}.npz')
        b=np.load(BASE/f'probe/capacity_diagonal_mpm_cpu/{label}.npz')
        g=np.load(BASE/f'probe/capacity_diagonal_mpm_gpu/{label}.npz')
        i=int(np.argmin(np.sum((a['xy']-[8.25,3.25])**2,axis=1)))
        # The captured pre-wave vector still has a wall-normal component;
        # the later physical wall guard makes its transported flux exactly 0.
        # Apply that same guard before forming a cell's flux divergence.
        physical_flux=a['wave_physical_normal'].copy()/2650
        physical_flux[(a['neighbour']<0)&(a['bid']<0)]=0.
        physical=probe_rate(a,i,physical_flux)*1000
        old=probe_rate(a,i,a['class_flux'][0])*1000;new=probe_rate(b,i,b['class_flux'][0])*1000
        parts=[probe_rate(a,i,a['wave_diagnostic'][k]/2650)*1000 for k in range(11,15)]
        central=probe_rate(a,i,a['wave_diagnostic'][19]/2650)*1000
        # Wave diagnostics are zero on the solid wall; the final wall flux
        # is also zero. Thus this sum includes all transported contributions.
        assert abs(old-central-sum(parts))<1e-12
        parity={k:norm(b[k]-g[k]) for k in ('class_flux','wave_diagnostic','bedcell','capacity_gradient')}
        assert parity['class_flux']['maximum']<1e-11
        decomposition[label]=dict(physical_mm_s=physical,old_final_mm_s=old,new_final_mm_s=new,
            central_capacity_mm_s=central,jump_contributions_mm_s=dict(zip(('water_mass','normal_momentum_pressure','transverse_momentum','sediment_flux'),parts)),cpu_gpu=parity)
    stability=json.loads((BASE/'stability/capacity_diagonal_v2/WAVE_PROBE.json').read_text())
    assert all(r['inferred_growth_rate_s']<0 for r in stability['records'])
    regressions={}
    for name in ('bend','shoal'):
        folder=BASE/'regression';filename='wave.npz' if name=='bend' else 'frames.npz'
        c=np.load(folder/f'{name}_capacity_diagonal_cpu/run'/filename)
        g=np.load(folder/f'{name}_capacity_diagonal_gpu/run'/filename)
        old=np.load(folder/f'{name}_matrix_cpu/run'/filename)
        r=dict(cpu_gpu_bed=norm(c['z']-g['z']),new_vs_matrix_bed=norm(c['z']-old['z']),
               minimum_depth_m=float(c['q'][:,0].min()),finite=bool(np.isfinite(c['q']).all() and np.isfinite(c['z']).all()))
        assert r['finite'] and r['minimum_depth_m']>=0 and r['cpu_gpu_bed']['maximum']<1e-6
        if name=='shoal':
            h=np.load(folder/'shoal_capacity_diagonal_cpu_half/run/frames.npz')
            quarter=np.load(folder/'shoal_capacity_diagonal_cpu_quarter/run/frames.npz')
            old_half=np.load(folder/'shoal_matrix_cpu_half/run/frames.npz')
            r['half_step_bed']=norm(c['z']-h['z'])
            r['half_to_quarter_bed']=norm(h['z']-quarter['z'])
            r['full_to_quarter_bed']=norm(c['z']-quarter['z'])
            r['previous_matrix_half_step_bed']=norm(old['z']-old_half['z'])
            r['exploratory_0p1mm_screen_pass']=r['half_step_bed']['maximum']<1e-4
            r['asymptotic_temporal_convergence_established']=False
            # Keep the failed strict exploratory screen visible, not silently
            # relabelled as an accuracy pass or hidden in a global RMS.
            r['temporal_scope']='Historical split EH/MF10 replay; local millimetre sensitivity remains. Not a universal accuracy qualification.'
            geometry=np.load(folder/'shoal_capacity_diagonal_cpu/case/coarse/input/analysis.npz')
            for key,diff in [('half_step_bed',c['z']-h['z']),('half_to_quarter_bed',h['z']-quarter['z'])]:
                frame,cell=np.unravel_index(np.argmax(abs(diff)),diff.shape)
                r[key]['worst_xy']=geometry['xy'][cell].tolist()
                r[key]['physical_depth_m']=float(c['q'][frame,0,cell])
            r['cpu_status']=json.loads((folder/'shoal_capacity_diagonal_cpu/run/STATUS.json').read_text())
            r['gpu_status']=json.loads((folder/'shoal_capacity_diagonal_gpu/run/STATUS.json').read_text())
            for key in ('cpu_status','gpu_status'):
                assert r[key]['water_budget_max_m3']<1e-10 and r[key]['solid_budget_max_m3']<1e-10
        else:
            for backend in ('cpu','gpu'):
                r[backend]=json.loads((folder/f'bend_capacity_diagonal_{backend}/run/METRICS.json').read_text())
            assert r['new_vs_matrix_bed']['maximum']<.001
        regressions[name]=r
    sources={}
    with zipfile.ZipFile(ROOT/'resources/Delft3D-main.zip') as archive:
        for name in ('upwbed.f90','dwnvel.f90','adjust_bedload.f90','mom_fls.f90'):
            members=[n for n in archive.namelist() if n.endswith('/'+name)];assert len(members)==1
            path=ROOT/'resources'/members[0];assert path.read_bytes()==archive.read(members[0])
            sources[str(path.relative_to(ROOT))]=sha(path)
    maintained=[ROOT/'braids_core/sediment'/n for n in ('bed_wave_flux.py','river_capacity.py','river_suspension.py','wall_velocity_bounds.py','wall_bed_hydraulics.py')]
    result=dict(cases=audit,tip=tip,candidates=candidate_metrics,restart=restart,timestep=dt,
                decomposition=decomposition,stability=stability,regressions=regressions,
                native_source_integrity=sources,maintained={str(p.relative_to(ROOT)):sha(p) for p in maintained},
                installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'))
    assert result['installed_runtime_sha256']=='8e72c97e1782ba3cca50c32189c9da47cbb278f175d754a6dc831e620d6f0e5e'
    save(OUT/'AUDIT.json',result)
    save(OUT/'QUALIFICATION.json',dict(bounded_flux_milestone_complete=True,current_worker_cpu_gpu=True,
        three_single_grain_bedload_laws=True,conservation=True,exact_restart=True,local_alternating_mode_decays=True,
        short_transcritical_stability_and_accepted_bend=True,transcritical_0p1mm_temporal_screen=False,
        universal_transcritical_accuracy_qualified=False,second_order_physical_reconstruction_unchanged=True,
        physical_transport_laws_unchanged=True,hydraulic_fwave_unchanged=True,
        installed_desktop_changed=False,packaged_release=False,experimental_validation=False,
        full_groin_Delft3D_agreement=False,long_term_stability_claim=False))

    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for row,law in enumerate(SPECS):
        d=data[law];o=before[law];values=[o['bz'][-1]-o['bz'][0],d['bz'][-1]-d['bz'][0],d['nz'][-1]-d['nz'][0]]
        limit=max(abs(v).max() for v in values)*1000
        polygons=[np.asarray(p.exterior.coords) for p in d['frames'][0].polygons]
        for col,(title,v) in enumerate(zip(('Previous BRAIDS','Corrected BRAIDS','Unmodified Delft3D FLOW'),values)):
            ax=axes[row,col];mesh=PolyCollection(polygons,array=v*1000,cmap='RdBu_r',clim=(-limit,limit),edgecolors=(0,0,0,.15),linewidths=.3)
            ax.add_collection(mesh);ax.fill([8,10,10,8],[0,0,3,3],color='.35');ax.plot(8.25,3.25,'o',ms=5,mfc='none',mec='k')
            ax.set(xlim=(6,14),ylim=(1.5,6),aspect='equal',title=f'{title}\n{LABELS[law]}: 20 s morphology',xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(mesh,ax=axes[row,:],label='Bed change (mm)',shrink=.7)
    fig.savefig(OUT/'BEDLOAD_RAW_CELLS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for col,law in enumerate(SPECS):
        d=data[law];o=before[law];xy=d['xy'];idx=np.where(np.isclose(xy[:,1],3.25)&(xy[:,0]>=6)&(xy[:,0]<=14))[0]
        idx=idx[np.argsort(xy[idx,0])]
        for label,z,color,style in [('Previous BRAIDS',o['bz'],'#64748b','o-'),('Corrected BRAIDS',d['bz'],'#2563eb','o-'),('Delft3D FLOW',d['nz'],'#b45309','s--')]:
            axes[0,col].plot(xy[idx,0],1000*(z[-1,idx]-z[0,idx]),style,label=label,color=color,ms=3)
        for label,key,color,style in [('Previous BRAIDS','before_mm','#64748b','o-'),('Corrected BRAIDS','after_mm','#2563eb','o-'),('Delft3D FLOW','native_mm','#b45309','s--')]:
            axes[1,col].plot(tip[law]['time_s'],tip[law][key],style,color=color,label=label,ms=3)
        axes[0,col].set(title=LABELS[law],xlabel='x (m), first row above groin y=3.25 m',ylabel='Bed change at 60 s (mm)')
        axes[1,col].set(xlim=(40,60),xlabel='Physical time (s); sediment starts at 40 s',ylabel='Bed change at (8.25, 3.25) (mm)')
    for ax in axes.flat:ax.axhline(0,color='k',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'TIP_SECTIONS_AND_TIME.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    r=decomposition['BRAIDS'];pieces=r['jump_contributions_mm_s']
    axes[0].bar(range(4),list(pieces.values()),color=['#2563eb','#b91c1c','#64748b','#b45309'])
    axes[0].set_xticks(range(4),['Water mass','Normal momentum\n+ pressure','Transverse\nmomentum','Sediment flux'])
    axes[0].set(ylabel='Contribution to bed rate (mm/s)',title='Old coupled bed-row dissipation\nMPM groin tip at 40 s');axes[0].axhline(0,color='k',lw=.7)
    rates=[r['physical_mm_s'],r['old_final_mm_s'],r['new_final_mm_s']]
    axes[1].bar(['Physical capacity\n(before wave term)','Previous final\nflux','Corrected final\nflux'],rates,color=['#64748b','#b91c1c','#2563eb'])
    axes[1].set(ylabel='Net bed rate (mm/s)',title='Same state; no transport-law tuning');axes[1].axhline(0,color='k',lw=.7)
    fig.savefig(OUT/'FLUX_ATTRIBUTION.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    selected=[r for r in stability['records'] if r['backend']=='cpu' and r['dt_s']==.0001 and r['perturbation_m']==1e-8]
    axes[0].bar(range(len(selected)),[r['inferred_growth_rate_s'] for r in selected],color='#2563eb')
    axes[0].set_xticks(range(len(selected)),[f"Fr {r['Froude']:.2f}\nMF {r['morfac']:g}" for r in selected]);axes[0].axhline(0,color='k',lw=.6)
    axes[0].set(ylabel='Alternating-mode growth rate (1/s)',title='Negative means damping (CPU/GPU tested)')
    for mode,color in [('matrix','#64748b'),('capacity_diagonal','#2563eb')]:
        a=np.load(BASE/f'regression/shoal_{mode}_cpu/run/frames.npz')
        # Original Cartesian shoal ordering is supplied by the retained analysis.
        geometry=np.load(BASE/f'regression/shoal_{mode}_cpu/case/coarse/input/analysis.npz')
        key='centroids' if 'centroids' in geometry else 'xy'
        if key in geometry:
            xy=geometry[key];axis=xy[:,0];sel=np.isclose(xy[:,1],np.unique(xy[:,1])[len(np.unique(xy[:,1]))//2])
        else:
            axis=np.arange(a['z'].shape[1]);sel=np.ones_like(axis,dtype=bool)
        ix=np.where(sel)[0][np.argsort(axis[sel])]
        axes[1].plot(axis[ix],1000*(a['z'][-1,ix]-a['z'][0,ix]),'o-',ms=2,label=mode.replace('_',' '),color=color)
    axes[1].set(xlabel='x (m)' if key in geometry else 'Cell index',ylabel='Bed change (mm)',title='Short transcritical shoal: 4 s / MF10');axes[1].legend(fontsize=8)
    bend=np.load(BASE/'regression/bend_capacity_diagonal_cpu/run/wave.npz');old=np.load(BASE/'regression/bend_matrix_cpu/run/wave.npz')
    axes[2].plot(bend['time']-624,1000*abs(bend['z']-old['z']).max(axis=1),label='maximum')
    axes[2].plot(bend['time']-624,1000*np.sqrt(np.mean((bend['z']-old['z'])**2,axis=1)),label='RMS')
    axes[2].set(xlabel='Physical replay time (s), MF25',ylabel='New minus former matrix bed (mm)',title='Accepted-bend regression (not a new validation)');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'STABILITY_REGRESSIONS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,6),layout='constrained')
    geometry=np.load(BASE/'regression/shoal_capacity_diagonal_cpu/case/coarse/input/analysis.npz')
    xy=geometry['xy'];quarter=np.load(BASE/'regression/shoal_capacity_diagonal_cpu_quarter/run/frames.npz')
    for col,(suffix,step) in enumerate([('',.0025),('_half',.00125),('_quarter',.000625)]):
        a=np.load(BASE/f'regression/shoal_capacity_diagonal_cpu{suffix}/run/frames.npz')
        for row,v in enumerate([a['z'][-1]-a['z'][0],a['z'][-1]-quarter['z'][-1]]):
            limit=13 if row==0 else 1.7
            sc=axes[row,col].scatter(xy[:,0],xy[:,1],c=v*1000,cmap='RdBu_r',vmin=-limit,vmax=limit,marker='s',s=44)
            axes[row,col].set(aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f'dt={step:g} s'+('\nBed change after 4 s / MF10' if row==0 else '\nDifference from shortest step'))
            fig.colorbar(sc,ax=axes[row,col],label='mm',shrink=.7)
    fig.savefig(OUT/'SHOAL_TIME_STEP_LIMITATION.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(cases={law:{k:audit[law][k] for k in ('bed_common','previous_bed_common','maximum_scour_m','maximum_deposition_m','cpu_gpu_bed')} for law in SPECS},tip=tip,candidates=candidate_metrics,timestep=dt,regressions={k:{x:v for x,v in r.items() if x not in ('cpu','gpu')} for k,r in regressions.items()}),indent=2))


if __name__=='__main__':main()
