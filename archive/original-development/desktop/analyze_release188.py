"""Cold-run evidence and raw-cell figures for Desktop188, not display smoothing."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/_wall188'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall188'
sys.path[:0]=[str(BASE/'desktop_candidate'),str(ROOT)]
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha


def stats(delta,area):
    return dict(max_mm=float(1000*abs(delta).max()),rms_mm=float(1000*np.sqrt(np.dot(area,delta*delta)/area.sum())))


def bed(frame,initial):
    dz=frame.z-initial.z;area=frame.arrays['area']
    return dict(erosion_m3=float(np.dot(area,np.maximum(-dz,0))),deposition_m3=float(np.dot(area,np.maximum(dz,0))),
        minimum_mm=float(1000*dz.min()),maximum_mm=float(1000*dz.max()),
        scour_xy=list(frame.polygons[int(dz.argmin())].centroid.coords[0]),
        deposition_xy=list(frame.polygons[int(dz.argmax())].centroid.coords[0]))


def plot(ax,frame,values,bound,title):
    obj=PolyCollection([np.array(p.exterior.coords) for p in frame.polygons],array=values,
        cmap='RdBu_r',clim=(-bound,bound),edgecolors='none',rasterized=True)
    ax.add_collection(obj);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title,fontsize=10)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');return obj


def main():
    result=dict(runtime=sha(BASE/'desktop_candidate/RUNTIME.json'),cold={})
    for start in ('wet','dry'):
        paths={b:BASE/f'release/{start}_prescribed_{b}' for b in ('cpu','gpu')}
        if '--partial' in sys.argv and not all((p/'restart/CHECK.json').exists() for p in paths.values()):continue
        stores={b:ResultsStore(p/'run') for b,p in paths.items()}
        first=stores['cpu'].read(0);area=first.arrays['area'];x=np.array([p.centroid.x for p in first.polygons]);inlet=x<5
        assert np.array_equal(stores['cpu'].times,stores['gpu'].times)
        for k in ('q','z','W','mass'):assert np.array_equal(first.arrays[k],stores['gpu'].read(0).arrays[k]),k
        for k in ('vertices','offsets','levels'):assert np.array_equal(first.mesh[k],stores['gpu'].read(0).mesh[k]),k
        history=[]
        for i,t in enumerate(stores['cpu'].times):
            c=stores['cpu'].read(i);g=stores['gpu'].read(i)
            history.append(dict(time=float(t),bed=stats(c.z-g.z,area),depth=stats(c.q[0]-g.q[0],area),
                inlet_bed_mm=float(1000*abs(c.z-g.z)[inlet].max()),inlet_depth_mm=float(1000*abs(c.q[0]-g.q[0])[inlet].max())))
        c,g=(stores[b].read(-1) for b in ('cpu','gpu'))
        old=ResultsStore(ROOT/f'validation/_inlet187/{start}_prescribed_cpu/run').read(-1)
        row=dict(initial_and_mesh_exact=True,cells=len(area),final=history[-1],history=history,
            maximum_over_frames={k:max(h[k]['max_mm'] for h in history) for k in ('bed','depth')},
            current_bed=bed(c,first),previous_bed=bed(old,first),method_bed_change=stats(c.z-old.z,area),
            checks={b:json.loads((p/'CHECK.json').read_text()) for b,p in paths.items()},
            restarts={b:json.loads((p/'restart/CHECK.json').read_text()) for b,p in paths.items()})
        for b in ('cpu','gpu'):
            assert all(row['restarts'][b]['frame_exact'].values()) and all(row['restarts'][b]['checkpoint_arrays_exact'].values())
        assert row['final']['bed']['max_mm']<1. and row['final']['depth']['max_mm']<1.,row['final']
        result['cold'][start]=row
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        values=[1000*(c.z-first.z),1000*(g.z-first.z),1000*(g.z-c.z),1000*(c.z-old.z)]
        titles=['CPU bed change','GPU bed change','GPU minus CPU bed','New minus previous limiter bed']
        bounds=[max(abs(values[0]).max(),abs(values[1]).max())]*2+[max(abs(values[2]).max(),1e-6),max(abs(values[3]).max(),1e-6)]
        for ax,v,b,title in zip(axes.ravel(),values,bounds,titles):
            obj=plot(ax,c,v,b,title+' (mm)');fig.colorbar(obj,ax=ax,shrink=.8)
        fig.suptitle(start.capitalize()+' cold start, 220 s: 200 s hydraulics + 20 s mobile bed, MF5')
        fig.savefig(REPORT/f'RELEASE_{start.upper()}_BED.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,k in zip(axes,('bed','depth')):
            ax.semilogy([h['time'] for h in history],[max(h[k]['max_mm'],1e-10) for h in history],label='Maximum')
            ax.semilogy([h['time'] for h in history],[max(h[k]['rms_mm'],1e-10) for h in history],label='Area RMS')
            ax.axhline(1,color='black',ls=':');ax.axvline(200,color='gray',ls='--');ax.grid(alpha=.2)
            ax.set(xlabel='Hydraulic time (s)',ylabel=k.capitalize()+' CPU/GPU difference (mm)');ax.legend()
        fig.savefig(REPORT/f'RELEASE_{start.upper()}_HISTORY.png',dpi=170);plt.close(fig)
    if '--partial' in sys.argv:
        save(REPORT/'RELEASE_PARTIAL.json',result)
        print(json.dumps({k:v['final'] for k,v in result['cold'].items()},indent=2));return
    smoke={}
    for tag,law in [('amr','smart_jaggi'),*[('fixed',k) for k in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi')]]:
        paths={b:BASE/'smoke'/tag/(law+'_'+b) for b in ('cpu','gpu')}
        c,g=(ResultsStore(paths[b]/'full').read(-1) for b in ('cpu','gpu'))
        for k in ('vertices','offsets','levels'):assert np.array_equal(c.mesh[k],g.mesh[k]),(tag,law,k)
        row=dict(bed=stats(c.z-g.z,c.arrays['area']),depth=stats(c.q[0]-g.q[0],c.arrays['area']),
            checks={b:json.loads((p/'CHECK.json').read_text()) for b,p in paths.items()})
        assert row['bed']['max_mm']<1. and row['depth']['max_mm']<1.,(tag,law,row)
        smoke[tag+'_'+law]=row
    result['smoke']=smoke
    save(REPORT/'RELEASE.json',result)
    print(json.dumps({k:v['final'] for k,v in result['cold'].items()},indent=2))


if __name__=='__main__':main()
