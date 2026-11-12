"""Figures and measured checks for the Test1 hydraulic repair."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def load(run):
    row=json.loads((run/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(run/'frames'/row['epoch']) as f:g={k:f[k] for k in f.files}
    with np.load(run/'frames'/row['file']) as f:q={k:f[k] for k in f.files}
    return g,q,row

def main():
    base=ROOT/'validation/_test1_gpu_fix'
    out=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/test1_hydro111'
    out.mkdir(parents=True,exist_ok=True)
    cpu=base/'cold_cpu_v2/run';gpu=base/'cold_gpu_v2/run'
    g,a,row=load(cpu);h,b,_=load(gpu)
    np.testing.assert_array_equal(g['leaves'],h['leaves'])
    delta=a['q']-b['q'];stats={}
    for name,run in [('cpu',cpu),('gpu',gpu)]:
        geo,state,frame=load(run);status=json.loads((run/'STATUS.json').read_text())
        diag=json.loads((run/'DIAGNOSTICS.json').read_text());events=json.loads((run/'ADAPTATION.json').read_text())
        stats[name]=dict(status=status,water_volume_m3=float(geo['area']@state['q'][0]),
            expected_inlet_volume_m3=11.*frame['time_s'],minimum_depth_m=float(state['q'][0].min()),
            max_water_residual_m3=max(abs(r['water_residual_m3']) for r in diag),
            wall_flux_max=max(abs(v) for r in diag for v in r['wall_integrals']),
            amr_events=len(events),remeshes=sum(e['changed'] for e in events),
            amr_wall_s=sum(e['adaptation_total_s'] for e in events))
    stats['paired']=dict(max_depth_m=float(abs(delta[0]).max()),
        rms_depth_m=float(np.sqrt(np.average(delta[0]**2,weights=g['area']))),
        max_discharge_m2_s=float(abs(delta[1:]).max()),same_leaf_mesh=True)
    if (base/'half_dt_gpu/run/STATUS.json').exists():
        from braids_core.sediment.adaptive_river_worker import load_prepared
        _,_,tree=load_prepared(base/'half_dt_gpu/prepared/core/prepared')
        j,c,cr=load(base/'half_dt_gpu/run')
        if abs(cr['time_s']-row['time_s'])<1e-12:
            def fine(geo,state):
                leaves=tuple(geo['leaves']);cover=tree.leaf_cover(leaves)[tree.offsets[-2]:]
                return state['q'][:,np.searchsorted(leaves,cover)]
            first,second=fine(h,b),fine(j,c);difference=first-second;areas=np.array([v.area_m2 for v in tree.finest.cells])
            wet=np.maximum(first[0],second[0])>.001
            stats['timestep']=dict(time_s=row['time_s'],normal_outer_cap_s=.1,smaller_outer_cap_s=.025,
                amr_interval_s=.05,max_depth_m=float(abs(difference[0]).max()),
                rms_depth_m=float(np.sqrt(np.average(difference[0]**2,weights=areas))),
                wet_rms_depth_m=float(np.sqrt(np.average(difference[0,wet]**2,weights=areas[wet]))),
                comparison='piecewise-constant states on shared finest hierarchy; different AMR meshes allowed')
    polygons=[g['vertices'][a:b] for a,b in zip(g['offsets'][:-1],g['offsets'][1:])]
    origin=np.array([2571200.,1106900.]);wet=np.flatnonzero(np.maximum(a['q'][0],b['q'][0])>.001)
    points=np.concatenate([polygons[i]-origin for i in wet]);lo=points.min(0)-4;hi=points.max(0)+4
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    maximum=max(a['q'][0].max(),b['q'][0].max());bound=max(float(abs(delta[0]).max()*1000),1e-9)
    for ax,values,title,cmap,limits in zip(axes,[a['q'][0],b['q'][0],delta[0]*1000],
        ['CPU depth at 2 s','GPU depth at 2 s','CPU minus GPU depth'],['Blues','Blues','RdBu_r'],[(0,maximum),(0,maximum),(-bound,bound)]):
        pc=PolyCollection([p-origin for p in polygons],array=values,cmap=cmap,clim=limits,linewidths=.12,edgecolors='.8')
        ax.add_collection(pc);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=title,xlabel='Local easting (m)',ylabel='Local northing (m)');ax.set_aspect('equal')
        fig.colorbar(pc,ax=ax,shrink=.7,label='mm' if title.startswith('CPU minus') else 'm')
    fig.savefig(out/'CPU_GPU_INLET.png',dpi=160);plt.close(fig)
    old=base/'residual_probe';new=base/'residual_corrected'
    d=json.loads((new/'PROBE.json').read_text());x=np.array([r['delta_h'] for r in d['rows']])*1e6
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for folder,label,color in ((old,'Alpha110','#b43b3b'),(new,'Corrected','#1769aa')):
        arrays=[np.load(folder/f'RATE_{i:02d}.npz') for i in range(len(x))]
        axes[0].plot(x,[v['trace'][0,7690] for v in arrays],'.-',label=label,color=color)
        axes[1].plot(x,[v['rate'][1,3949] for v in arrays],'.-',label=label,color=color)
        for v in arrays:v.close()
    axes[0].set(ylabel='Reconstructed face depth (m)',title='Cropped inlet cell reconstruction')
    axes[1].set(ylabel='Momentum rate (m²/s²)',title='Continuous residual after correction')
    for ax in axes[:2]:ax.set_xlabel('Cell depth perturbation (µm)');ax.legend();ax.grid(alpha=.2)
    maps=json.loads((base/'MAP_BENCHMARK.json').read_text());xpos=np.arange(3)
    axes[2].bar(xpos-.18,[v['polygon_s'] for v in maps],.36,label='Polygon intersections',color='#b43b3b')
    axes[2].bar(xpos+.18,[v['hierarchy_s'] for v in maps],.36,label='Exact hierarchy map',color='#1769aa')
    axes[2].set(xticks=xpos,xticklabels=['Coarsen','Refine','Unchanged'],ylabel='Map construction (s)',title='Identical map arrays, less work');axes[2].legend(fontsize=8)
    fig.savefig(out/'CAUSE_AND_OPTIMIZATION.png',dpi=160);plt.close(fig)
    stats['frozen_step']=d['corrected_solve'];stats['map_benchmark']=maps
    (out/'METRICS.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
