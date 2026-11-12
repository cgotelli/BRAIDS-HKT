"""Exact-overlap field comparisons and retained rejected small-cell evidence."""
import json,sys,hashlib,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import analyze_merged_fwave115 as shared
BASE=ROOT/'validation/_mesh_modes117'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/mesh_modes117'
read=lambda p:json.loads(p.read_text())

def main():
    OUT.mkdir(parents=True,exist_ok=True);shared.BASE=BASE
    frames={n:shared.frames(n) for n in ('cpu_dynamic','gpu_dynamic','cpu_static','gpu_static','smallcell_cpu','smallcell_gpu')}
    shared.BASE=ROOT/'validation/_terrain_amr116';frames['reference_explicit']=shared.frames('cpu_final')
    shared.BASE=ROOT/'validation/_test1_amr_interval';frames['old_dynamic']=shared.frames('cpu_installed_1s')
    pairs=[('gpu_dynamic','cpu_dynamic'),('gpu_static','cpu_static'),('cpu_static','cpu_dynamic'),
        ('cpu_dynamic','old_dynamic'),('smallcell_cpu','reference_explicit'),('smallcell_gpu','smallcell_cpu')]
    comparisons={a+' vs '+b:[shared.compare(x,y) for x,y in zip(frames[a],frames[b])] for a,b in pairs}
    runs={}
    for n in ('cpu_dynamic','gpu_dynamic','cpu_static','gpu_static','smallcell_cpu','smallcell_gpu'):
        s=read(BASE/n/'PROFILE_SUMMARY.json');events=read(BASE/n/'run/ADAPTATION.json');diags=read(BASE/n/'run/DIAGNOSTICS.json')
        runs[n]=dict(worker_s=s['worker_call_s'],preparation_s=s['preparation_s'],phases=s['phases'],
            hydraulic=s['result']['hydraulic'],cells=[len(a['area']) for a in frames[n]],
            assessments=len(events),regrids=sum(bool(e.get('changed')) for e in events),
            max_water_residual_m3=max(abs(r['water_residual_m3']) for r in diags),
            max_wall_flux=max(max(r['wall_integrals']) for r in diags))
    metrics=dict(runs=runs,comparisons=comparisons,smallcell_candidate_adopted=False,
        production_integrator='implicit_fwave_midpoint_river_v1')
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,7),layout='constrained')
    for ax,n,title in zip(axes,('cpu_dynamic','cpu_static'),('Dynamic depth only','Static terrain slope breaks')):
        d=frames[n][0];width=np.sqrt(d['area']);pc=PolyCollection([np.array(p.exterior.coords) for p in d['polys']],
            array=width,cmap='viridis_r',clim=(0,20),edgecolors='#333333',linewidth=.22)
        ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal')
        ax.set_title(f'{title}\n{len(width)} initial cells');ax.set_xlabel('Local east (m)');ax.set_ylabel('Local north (m)')
        fig.colorbar(pc,ax=ax,orientation='horizontal',label='Square root of actual cell area (m)')
    fig.suptitle('Test1: same domain, finest spacing 5 m, three levels\nCropped boundary cells remain finest in both modes')
    fig.savefig(OUT/'MESH_COMPARISON.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=np.arange(4);bottom=np.zeros(4);names=['cpu_dynamic','cpu_static','gpu_dynamic','gpu_static']
    for key,label in [('stepping','Hydraulic and physical stepping'),('amr','AMR'),('initialization','Owner initialization'),('output','Output'),('checkpoint','Checkpoint'),('other','Other'),('load','Loading')]:
        values=[runs[n]['phases'].get(key,0) for n in names]
        axes[0].bar(x,values,bottom=bottom,label=label);bottom+=values
    axes[0].set_xticks(x,['CPU dynamic','CPU static','GPU dynamic','GPU static'],rotation=15)
    axes[0].set_ylabel('Worker wall time (s), excluding preparation');axes[0].legend(fontsize=7)
    rows=comparisons['cpu_static vs cpu_dynamic']
    axes[1].plot([r['time_s'] for r in rows],[r['h_max_mm'] for r in rows],label='Maximum depth difference')
    axes[1].plot([r['time_s'] for r in rows],[r['h_wet_rms_mm'] for r in rows],label='Wet-area RMS')
    axes[1].set_xlabel('Simulated time (s)');axes[1].set_ylabel('Static minus dynamic depth difference (mm)')
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.suptitle('Unchanged semi-implicit solver: Test1, first 8 seconds')
    fig.savefig(OUT/'PERFORMANCE_AND_DEPTH.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    rows=comparisons['smallcell_cpu vs reference_explicit']
    axes[0].plot([r['time_s'] for r in rows],[r['h_max_mm'] for r in rows],label='Maximum')
    axes[0].plot([r['time_s'] for r in rows],[r['h_wet_rms_mm'] for r in rows],label='Wet-area RMS')
    axes[0].set_xlabel('Simulated time (s)');axes[0].set_ylabel('Depth difference (mm)');axes[0].legend();axes[0].grid(alpha=.2)
    a=frames['smallcell_cpu'][-1];b=frames['reference_explicit'][-1]
    assert len(a['area'])==len(b['area']);diff=1000*(a['q'][0]-b['q'][0]);extent=np.max(abs(diff))
    pc=PolyCollection([np.array(p.exterior.coords) for p in a['polys']],array=diff,cmap='RdBu_r',clim=(-extent,extent),edgecolors='#888888',linewidth=.25)
    axes[1].add_collection(pc);wet=(a['q'][0]>.001)|(b['q'][0]>.001)
    pts=np.concatenate([np.array(a['polys'][i].exterior.coords) for i in np.flatnonzero(wet)])
    axes[1].set_xlim(pts[:,0].min()-3,pts[:,0].max()+3);axes[1].set_ylim(pts[:,1].min()-3,pts[:,1].max()+3)
    axes[1].set_aspect('equal');axes[1].set_xlabel('Local east (m)');axes[1].set_ylabel('Local north (m)')
    fig.colorbar(pc,ax=axes[1],label='Redistribution minus original depth (mm)')
    fig.suptitle('Rejected small-cell rate redistribution: conservation alone is insufficient')
    fig.savefig(OUT/'REJECTED_SMALLCELL.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(runs=runs,comparison_maxima={k:max(r['h_max_mm'] for r in v) for k,v in comparisons.items()}),indent=2))

if __name__=='__main__':main()
