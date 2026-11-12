"""Terrain protection, measured timestep limiter and unchanged-geometry contrasts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import analyze_merged_fwave115 as shared
from braids_core.sediment.adaptive_river_worker import hierarchy
from braids_core.sediment.river_terrain_refinement import terrain_features
from braids_core.sediment.river_field_policy import FieldSettings
BASE=ROOT/'validation/_terrain_amr116';shared.BASE=BASE
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/terrain_amr116'
read=lambda p:json.loads(p.read_text())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    names=['baseline_cpu_warm_profile','cpu_reuse_offterrain','cpu_terrain_clean','gpu_terrain_clean',
        'cpu_terrain_cfl05','cpu_terrain_dt04','gpu_terrain_dt04','cpu_terrain_implicit','gpu_terrain_implicit',
        'cpu_clock','gpu_clock','cpu_clock_cfl05','cpu_clock_dt04','cpu_terrain_bedload','gpu_terrain_bedload',
        'cpu_final','gpu_final']
    frames={n:shared.frames(n) for n in names if (BASE/n/'run/frames/INDEX.json').exists()}
    pairs=[('cpu_reuse_offterrain','baseline_cpu_warm_profile'),('cpu_terrain_clean','cpu_reuse_offterrain'),
        ('gpu_terrain_clean','cpu_terrain_clean'),('cpu_terrain_cfl05','cpu_terrain_clean'),
        ('cpu_terrain_dt04','cpu_terrain_clean'),('gpu_terrain_dt04','cpu_terrain_dt04'),
        ('cpu_terrain_clean','cpu_terrain_implicit'),('gpu_terrain_implicit','cpu_terrain_implicit'),
        ('cpu_clock','cpu_terrain_clean'),('gpu_clock','cpu_clock'),('cpu_clock_cfl05','cpu_clock'),
        ('cpu_clock_dt04','cpu_terrain_dt04'),('gpu_terrain_bedload','cpu_terrain_bedload'),
        ('gpu_final','cpu_final'),('cpu_final','cpu_terrain_implicit'),('cpu_final','cpu_clock')]
    comparisons={a+' vs '+b:[shared.compare(x,y) for x,y in zip(frames[a],frames[b])]
        for a,b in pairs if a in frames and b in frames}
    runs={n:read(BASE/n/'PROFILE_SUMMARY.json') for n in names if (BASE/n/'PROFILE_SUMMARY.json').exists()}
    summaries={}
    for n,d in runs.items():
        e=read(BASE/n/'EXPERIMENT.json');trials=e.get('explicit_trials',[])
        dt=[r['dt'] for r in trials if r['error'] is None]
        summaries[n]=dict(phases=d['phases'],worker_s=d['worker_call_s'],preparation_s=d['preparation_s'],
            hydraulic=d['result']['hydraulic'],method=d['result']['integrator'],
            dt_quantiles_s=np.quantile(dt,[0,.25,.5,.75,1]).tolist() if dt else [],
            cells=[len(v['area']) for v in frames[n]],outer_rejections=len(d['result']['rejected_trials']))
    src=BASE/'cpu_terrain_audit';trace=read(src/'TERRAIN_EXPERIMENT.json')['cfl_traces']
    p=read(src/'prepared/core/prepared/profile.json');tree=hierarchy(p)
    with np.load(src/'TERRAIN.npz') as f:terrain={k:f[k] for k in f.files}
    pins=set(terrain['nodes'].tolist());pin_checks={}
    for n in names:
        if n not in frames or n in ('baseline_cpu_warm_profile','cpu_reuse_offterrain'):continue
        root=BASE/n/'run/frames';index=read(root/'INDEX.json');checks=[]
        for ep in index['epochs']:
            with np.load(root/ep) as f:checks.append(pins.issubset(f['leaves'].tolist()))
        pin_checks[n]=all(checks)
        assert pin_checks[n],n+' lost terrain protection'
    budgets={}
    for name in runs:
        rows=read(BASE/name/'run/DIAGNOSTICS.json')
        budgets[name]=dict(water_residual_max_m3=max(abs(r['water_residual_m3']) for r in rows),
            class_residual_max_kg=max(max(abs(v) for v in r['class_residual_kg']) for r in rows),
            wall_flux_max=max(max(abs(v) for v in r['wall_integrals']) for r in rows),
            minimum_stock=min(r['minimum_stock'] for r in rows),
            minimum_suspension=min(r['minimum_suspension'] for r in rows))
    with np.load(BASE/'cpu_terrain_clean/CFL_FINAL.npz') as f:frequency=f['frequency']
    area=frames['cpu_terrain_clean'][-1]['area']
    cutoff=[dict(area_cutoff_m2=v,cells_below=int(np.count_nonzero(area<v)),
        bound_of_remaining_cells_s=float(.45/np.max(frequency[area>=v]))) for v in (0.,2.5,6.25,12.5,24.99)]
    metrics=dict(runs=summaries,comparisons=comparisons,budgets=budgets,terrain=read(src/'TERRAIN.json'),
        terrain_pins_retained=pin_checks,limiter=dict(final=trace[-1],
            same_location_fraction=sum(r['xy']==trace[-1]['xy'] for r in trace)/len(trace),
            diagnostic_area_cutoffs=cutoff,note='Diagnostic only; no small-cell CFL condition was bypassed'))
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2))

    origin=frames['cpu_terrain_clean'][0]['origin'];xy=terrain['xy']-origin
    polys=[np.array(c.vertices_xy)-origin for c in tree.finest.cells]
    fig,axes=plt.subplots(1,4,figsize=(15,6.5),layout='constrained')
    fine0=tree.offsets[-2];wanted=np.zeros(tree.count,int);wanted[terrain['nodes']]=tree.finest_level
    levels=[]
    for name in ('baseline_cpu_warm_profile','cpu_terrain_clean'):
        d=frames[name][0]
        # Exact active resolution from the initial geometry epoch.
        idx=read(BASE/name/'run/frames/INDEX.json')['frames'][0]
        with np.load(BASE/name/'run/frames'/idx['epoch']) as f:lv=tree.node_level[f['leaves']]
        levels.append((d,lv))
    for ax,arr,title,cmap in [(axes[0],terrain['bed'],'Initial terrain at 5 m','terrain'),
                              (axes[1],terrain['tag'].astype(int),'Detected slope breaks','Oranges')]:
        pc=PolyCollection(polys,array=arr,cmap=cmap,edgecolors='none');ax.add_collection(pc);ax.autoscale_view()
        fig.colorbar(pc,ax=ax,orientation='horizontal',label='Elevation (m)' if ax is axes[0] else '0 = not tagged, 1 = tagged')
        ax.set_title(title,fontsize=10)
    for ax,(d,lv),title in zip(axes[2:],levels,('Previous dry-start mesh','Terrain-protected dry-start mesh')):
        pc=PolyCollection([np.array(poly.exterior.coords)+d['origin']-origin for poly in d['polys']],
            array=lv.astype(float),cmap='viridis',norm=BoundaryNorm([-.5,.5,1.5,2.5],256),edgecolors='#444444',linewidth=.13)
        ax.add_collection(pc);ax.autoscale_view();ax.set_title(title+'\n'+str(len(lv))+' active cells',fontsize=10)
        cb=fig.colorbar(pc,ax=ax,orientation='horizontal',ticks=[0,1,2]);cb.ax.set_xticklabels(['20 m','10 m','5 m'])
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlabel('Local east (m)');ax.set_ylabel('Local north (m)')
    fig.suptitle('Test1: initial slope breaks stay finest even when dry\n5-degree slope change + 0.10 m detail threshold; one finest-cell halo')
    fig.savefig(OUT/'TERRAIN_REFINEMENT.png',dpi=170);plt.close(fig)

    a=frames['cpu_terrain_clean'][-1];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    wet=np.flatnonzero(a['q'][0]>.001);pts=np.concatenate([np.array(a['polys'][i].exterior.coords) for i in wet])
    pc=PolyCollection([np.array(poly.exterior.coords) for poly in a['polys']],
        array=np.ma.masked_where(a['q'][0]<.001,a['q'][0]),cmap='Blues',edgecolors='#666666',linewidth=.5)
    axes[0].add_collection(pc);axes[0].set_xlim(pts[:,0].min()-3,pts[:,0].max()+3);axes[0].set_ylim(pts[:,1].min()-3,pts[:,1].max()+3)
    loc=np.array(trace[-1]['xy'])-a['origin'];axes[0].scatter(*loc,c='red',s=60,marker='x',label='Controlling inlet cell: 2.48 m²')
    axes[0].set_aspect('equal');axes[0].legend(fontsize=8);axes[0].set_title('Inlet after 8 seconds');fig.colorbar(pc,ax=axes[0],label='Water depth (m)')
    axes[0].set_xlabel('Local east (m)');axes[0].set_ylabel('Local north (m)')
    axes[1].plot([r['time'] for r in trace],[r['bound_s'] for r in trace],label='Local stage CFL bound (0.45)',lw=1.2)
    trials=read(BASE/'cpu_terrain_clean/EXPERIMENT.json')['explicit_trials']
    axes[1].scatter([r['time'] for r in trials if r['error'] is None],[r['dt'] for r in trials if r['error'] is None],s=4,label='Accepted hydraulic substep')
    axes[1].axhline(.1,color='#888888',linestyle='--',label='Outer ceiling: 0.1 s')
    axes[1].set_ylim(0,.105);axes[1].set_xlabel('Simulation time (s)');axes[1].set_ylabel('Seconds');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    axes[1].set_title('The cropped inlet sets dt, not the dry levees')
    fig.savefig(OUT/'TIMESTEP_LIMITER.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for key,label in [('cpu_terrain_clean vs cpu_reuse_offterrain','Terrain on vs off'),
                      ('cpu_clock_cfl05 vs cpu_clock','CFL 0.50 vs 0.45, corrected clock'),
                      ('cpu_terrain_dt04 vs cpu_terrain_clean','Outer ceiling 0.4 vs 0.1 s'),
                      ('cpu_final vs cpu_terrain_implicit','Final candidate vs implicit, terrain on')]:
        if key not in comparisons:continue
        rows=comparisons[key]
        for ax,k in zip(axes,('h_max_mm','h_wet_rms_mm')):ax.plot([r['time_s'] for r in rows],[r[k] for r in rows],label=label)
    for ax,title in zip(axes,('Maximum depth difference','Wet-area RMS depth difference')):
        ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.set_ylabel('mm');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'DEPTH_COMPARISONS.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,backend in zip(axes,('cpu','gpu')):
        ordered=[(f'{backend}_terrain_implicit','Implicit'),(f'{backend}_final','Candidate 0.1 s'),(f'{backend}_terrain_dt04','Trial 0.4 s')]
        ordered=[(n,l) for n,l in ordered if n in summaries];bottom=np.zeros(len(ordered))
        for phase,color in [('stepping','#d95f02'),('amr','#7570b3'),('initialization','#66a61e'),('output','#e6ab02'),('checkpoint','#a6761d'),('load','#bbbbbb'),('other','#999999')]:
            v=np.array([summaries[n]['phases'].get(phase,0) for n,_ in ordered]);ax.bar([l for _,l in ordered],v,bottom=bottom,label=phase,color=color);bottom+=v
        for i,y in enumerate(bottom):ax.text(i,y+.1,f'{y:.2f}',ha='center')
        ax.set_ylim(0,max(bottom)*1.15);ax.set_title(backend.upper()+' with terrain refinement');ax.set_ylabel('Worker wall time (s)')
    axes[1].legend(fontsize=8);fig.suptitle('First 8 simulated seconds; preparation excluded; same domain and physical input')
    fig.savefig(OUT/'PERFORMANCE.png',dpi=170);plt.close(fig)
    sources=[ROOT/p for p in ['braids_core/sediment/river_terrain_refinement.py','braids_core/sediment/river_terrain_refinement.h',
        'braids_core/sediment/river_field_policy.py','braids_core/sediment/adaptive_river_worker.py',
        'braids_core/sediment/river_kernel_cache.py','braids_core/river_project.py','braids_qgis/gui/river_dock.py',
        'validation/sediment/river_morphodynamics/sediment_wave.py','tools/profile_terrain_amr116.py',
        'tools/analyze_terrain_amr116.py','tests/test_river_terrain_refinement.py','tests/test_river_kernel_cache.py',
        'validation/hydraulics/terrain_amr116/explicit_clock.py','tests/test_river_explicit_remainder_clock.py',
        'tests/support/qgis_terrain_refinement.py']]
    with zipfile.ZipFile(OUT/'SOURCES.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.write(p,p.relative_to(ROOT).as_posix())
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in [*BASE.glob('*.xml'),*BASE.glob('*/PROFILE_SUMMARY.json'),*BASE.glob('*/EXPERIMENT.json'),
              *BASE.glob('*/TERRAIN_EXPERIMENT.json'),*BASE.glob('*/PROJECT.json'),*BASE.glob('*/run/frames/INDEX.json')]:
        manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    import shutil
    shutil.copyfile(BASE/'qgis_ui/MESH_CONTROLS.png',OUT/'MESH_CONTROLS.png')
    for key,rows in comparisons.items():
        print(key,'maximum depth mm',max(r['h_max_mm'] for r in rows),'final',rows[-1])
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
