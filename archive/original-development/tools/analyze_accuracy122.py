"""Raw-cell figures and provenance for automatic hydraulic accuracy control."""
from pathlib import Path
import json,sys,hashlib,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_accuracy122'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/accuracy122'


def main():
    from braids_core.sediment.adaptive_river_worker import read_live_frame
    OUT.mkdir(exist_ok=True,parents=True)
    folders={p.name:p for p in BASE.iterdir() if (p/'PROFILE_SUMMARY.json').is_file()}
    for name,rel in [('old_cpu','_axis_boundary120/aligned_cpu_main'),('old_gpu','_axis_boundary120/aligned_gpu_warm'),
                     ('fine_reference','_wetfront121/jfnk_gpu_tol7_eighth')]:folders[name]=ROOT/'validation'/rel
    runs={};states={}
    for name,path in folders.items():
        profile=json.loads((path/'PROFILE_SUMMARY.json').read_text())
        if not profile['result'] or profile['result']['time_s']!=3.75:continue
        _,g,s=read_live_frame(path/'run');q=s['q'];wet=q[0]>.001
        runs[name]=dict(path=str(path),profile=profile,volume_m3=float(q[0]@g['area']),
            minimum_depth_m=float(q[0].min()),maximum_depth_m=float(q[0].max()),
            wet_area_m2=float(g['area'][wet].sum()),
            maximum_wet_speed_m_s=float((np.linalg.norm(q[1:,wet],axis=0)/q[0,wet]).max()))
        states[name]=q;geom=g
    comparisons={}
    for a in states:
        comparisons[a]={}
        for b in states:
            q,r=states[a],states[b];delta=q[0]-r[0];wet=(q[0]>.001)|(r[0]>.001);both=(q[0]>.001)&(r[0]>.001)
            comparisons[a][b]=dict(max_depth_mm=float(abs(delta).max()*1000),
                wet_rms_depth_mm=float(np.mean(delta[wet]**2)**.5*1000),max_cell=int(abs(delta).argmax()),
                maximum_conserved_difference=float(abs(q-r).max()),
                wet_velocity_rms_m_s=float(np.mean(np.sum((q[1:,both]/q[0,both]-r[1:,both]/r[0,both])**2,axis=0))**.5))
    (OUT/'SUMMARY.json').write_text(json.dumps(dict(runs=runs,comparisons=comparisons),indent=2))
    selected='final_gpu' if 'final_gpu' in states else 'maintained_gpu'
    selected_cpu='final_cpu' if 'final_cpu' in states else 'maintained_cpu'
    names=['old_cpu',selected_cpu,'old_gpu',selected]
    labels=['Existing CPU','Automatic CPU','Existing GPU','Automatic GPU']
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,kind in zip(axs,['stepping','worker_call_s']):
        values=[runs[n]['profile']['phases']['stepping'] if kind=='stepping' else runs[n]['profile']['worker_call_s'] for n in names]
        bars=ax.bar(labels,values,color=['gray','#2563a8','gray','#d55e00']);ax.bar_label(bars,fmt='%.2f s')
        ax.set_ylim(0,max(values)*1.17);ax.tick_params(axis='x',rotation=15);ax.set_ylabel('Wall time (s)')
        ax.set_title('Hydraulic stepping' if kind=='stepping' else 'Worker including startup and output (warm cache)')
    fig.suptitle('Same 39,399 whole cells, dry start to 3.75 s; preparation excluded')
    fig.savefig(OUT/'PERFORMANCE.png',dpi=160);plt.close(fig)
    polys=[geom['vertices'][geom['offsets'][i]:geom['offsets'][i+1]]-np.array([2571550,1106690]) for i in range(len(geom['area']))]
    centres=np.array([p.mean(0) for p in polys]);keep=(centres[:,1]>0)&(centres[:,1]<30)&(centres[:,0]>0)&(centres[:,0]<40)
    fig,axs=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    for ax,name,title in zip(axs,[selected_cpu,selected,'fine_reference'],['Automatic CPU','Automatic GPU','Fine temporal reference']):
        pc=PolyCollection([p for p,k in zip(polys,keep) if k],array=states[name][0,keep],cmap='Blues',clim=(0,.45),edgecolor='gray',linewidth=.15)
        ax.add_collection(pc);ax.set(xlim=(0,40),ylim=(0,30),aspect='equal',title=title,xlabel='Local easting (m)')
    axs[0].set_ylabel('Local northing (m)');fig.colorbar(pc,ax=axs,label='Depth (m)',shrink=.7)
    fig.savefig(OUT/'RAW_DEPTHS.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,ref,title in zip(axs,['fine_reference','auto_gpu_half_tolerance'],['Versus fine-step reference','Versus half local tolerances']):
        delta=1000*(states[selected][0]-states[ref][0])
        pc=PolyCollection([p for p,k in zip(polys,keep) if k],array=delta[keep],cmap='RdBu_r',clim=(-1,1),edgecolor='gray',linewidth=.15)
        ax.add_collection(pc);ax.set(xlim=(0,40),ylim=(0,30),aspect='equal',title=title,xlabel='Local easting (m)')
    axs[0].set_ylabel('Local northing (m)');fig.colorbar(pc,ax=axs,label='Automatic depth minus control (mm)',shrink=.7)
    fig.savefig(OUT/'TEMPORAL_DIFFERENCE.png',dpi=170);plt.close(fig)
    print(json.dumps({n:dict(stepping_s=v['profile']['phases']['stepping'],hydraulic=v['profile']['result']['hydraulic']) for n,v in runs.items()},indent=2))


if __name__=='__main__':main()
