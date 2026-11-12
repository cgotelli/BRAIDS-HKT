"""Raw groin fields, conservation, backend agreement and clock diagnosis."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)]
from braids_desktop.results import ResultsStore
from review_boundaries159 import audit
BASE=ROOT/'validation/_boundaries160'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_boundaries160'
OUT.mkdir(parents=True,exist_ok=True)
def selected(store,t):return store.read(int(np.argmin(abs(store.times-t))))
def centres(f):
    from shapely.geometry import Polygon
    return np.array([p.centroid.coords[0] for p in f.polygons])
def picture(ax,f,values,title,lo=None,hi=None,cmap='viridis'):
    pc=PolyCollection([np.asarray(p.exterior.coords) for p in f.polygons],array=values,cmap=cmap,edgecolors='none',clim=(lo,hi))
    ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    return pc
def main():
    summary={};stores={}
    for case in ('out_weir','weir_recovered','bedload_stage_weir','closed_weir'):
        for backend in ('cpu','gpu'):
            name=case+'_'+backend;r,data=audit(BASE/'weir_runs'/name);summary[name]=r;stores[name]=data[0]
            f=data[1][-1];xy=centres(f);h=f.q[0];speed=np.linalg.norm(f.q[1:],axis=0)/np.maximum(h,1e-6)
            mask=xy[:,0]>xy[:,0].max()-2
            r['final_last_2m_max_Fr']=float(np.max(speed[mask]/np.sqrt(9.81*h[mask])))
            if case!='closed_weir':assert r['final_last_2m_max_Fr']<1.,r
        c=stores[case+'_cpu'].read(-1);g=stores[case+'_gpu'].read(-1)
        summary[case+'_parity']={k:float(np.max(abs(c.arrays[k]-g.arrays[k]))) for k in ('q','z','bed_boundary','water')}
        assert np.max(abs(c.q[0]-g.q[0]))<.001
        summary[case+'_parity']['depth_m']=float(np.max(abs(c.q[0]-g.q[0])))
    for case in ('jump','dry'):
        for backend in ('cpu','gpu'):
            r,data=audit(BASE/'general'/(case+'_'+backend));f=data[1][-1];xy=centres(f)
            h=f.q[0];fr=np.linalg.norm(f.q[1:],axis=0)/np.maximum(h,1e-6)/np.sqrt(9.81*np.maximum(h,1e-6))
            r['final_last_2m_max_Fr']=float(fr[xy[:,0]>xy[:,0].max()-2].max())
            assert r['final_last_2m_max_Fr']<1
            summary['general_'+case+'_'+backend]=r
    for formula in ('ashida_michiue_generalized','wang_fredsoe'):
        pair=[]
        for backend in ('cpu','gpu'):
            folder=BASE/'finite_runs'/(formula+'_'+backend);s=ResultsStore(folder/'run');f=s.read(-1)
            d=json.loads((folder/'run/DIAGNOSTICS.json').read_text());dz=f.z-f.arrays['initial_bed_m'];pair.append(f)
            r=dict(minimum_change_mm=float(dz.min()*1000),maximum_change_mm=float(dz.max()*1000),
                water_residual_m3=max(abs(row['water_residual_m3']) for row in d),
                sediment_residual_kg=max(abs(v) for row in d for v in row['class_residual_kg']),
                minimum_stock=min(row['minimum_stock'] for row in d))
            assert r['minimum_change_mm']<0<r['maximum_change_mm'] and r['minimum_stock']>=0 and r['water_residual_m3']<1e-7 and r['sediment_residual_kg']<1e-7
            summary['finite_'+formula+'_'+backend]=r
        summary['finite_'+formula+'_parity']={k:float(np.max(abs(pair[0].arrays[k]-pair[1].arrays[k]))) for k in ('q','z','mass')}
        assert np.max(abs(pair[0].z-pair[1].z))<.001
    formulas=('mpm','ashida_michiue_generalized','wang_fredsoe')
    for formula in formulas:
        for backend in ('cpu','gpu'):
            name=formula+'_'+backend;folder=BASE/'bedload_runs'/name
            store=stores[name]=ResultsStore(folder/'run');fs=[store.read(i) for i in range(len(store.times))]
            d=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
            summary[name]=dict(time_s=float(store.times[-1]),
                water_residual_m3=max(abs(r['water_residual_m3']) for r in d),
                sediment_residual_kg=max(abs(v) for r in d for v in r['class_residual_kg']),
                walls=max(abs(v) for r in d for v in r['wall_integrals']),
                minimum_stock=min(r['minimum_stock'] for r in d),
                deposition_max_mm=float((fs[-1].z-fs[-1].arrays['initial_bed_m']).max()*1000),
                minimum_sample_dt_s=min(float(f.arrays['bedload_sample_dt_s']) for f in fs[1:]),
                output_max_m2_s=[float(np.linalg.norm(f.arrays['bedload_vector_m2_s'],axis=0).max()) for f in fs])
            r=summary[name];assert r['water_residual_m3']<1e-7 and r['sediment_residual_kg']<1e-7 and r['walls']==0 and r['minimum_stock']>=0
            assert r['deposition_max_mm']>0 and r['minimum_sample_dt_s']>.001,r
        c=stores[formula+'_cpu'].read(-1);g=stores[formula+'_gpu'].read(-1)
        summary[formula+'_parity']={k:float(np.max(abs(c.arrays[k]-g.arrays[k]))) for k in ('q','z','bed_boundary','water','bedload_vector_m2_s')}
        assert np.max(abs(c.q[0]-g.q[0]))<.001 and np.max(abs(c.z-g.z))<.001
    old=ResultsStore(BASE/'clock_before3/run');new=stores['mpm_cpu']
    diffs={k:[] for k in ('q','z','mass','bed_boundary')}
    for t in old.times:
        a,b=selected(old,t),selected(new,t)
        for k in diffs:diffs[k].append(float(np.max(abs(a.arrays[k]-b.arrays[k]))))
    summary['clock_only_trajectory']={k:max(v) for k,v in diffs.items()}
    assert summary['clock_only_trajectory']['z']<.001 and summary['clock_only_trajectory']['q']<.001
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for store,label in ((old,'Before: last-step sample includes numerical clock tail'),(new,'Corrected output clock')):
        maxima=[np.linalg.norm(store.read(i).arrays['bedload_vector_m2_s'],axis=0).max() for i in range(len(store.times))]
        axes[0].plot(store.times,maxima,'o-',ms=3,label=label)
        axes[1].plot(store.times,[1000*(store.read(i).z-store.read(i).arrays['initial_bed_m']).max() for i in range(len(store.times))],label=label)
    axes[0].set_yscale('log');axes[0].set_ylabel('Maximum last-step bedload [m²/s]');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Maximum deposition [mm]');axes[1].set_xlabel('Physical time [s]');axes[1].set_xlim(40,220)
    fig.suptitle('Same 0.5 m groin: output-clock correction, unchanged MPM physics');fig.tight_layout();fig.savefig(OUT/'SNAPSHOT_CLOCK.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8));f=stores['out_weir_cpu'].read(-1);xy=centres(f);h=f.q[0];fr=np.linalg.norm(f.q[1:],axis=0)/np.maximum(h,1e-6)/np.sqrt(9.81*np.maximum(h,1e-6))
    fig.colorbar(picture(axes[0,0],f,h+f.z,'Weir: water elevation at 120 s'),ax=axes[0,0],label='m')
    fig.colorbar(picture(axes[0,1],f,fr,'Weir: Froude number at 120 s',0,1),ax=axes[0,1])
    for backend in ('cpu','gpu'):
        s=stores['out_weir_'+backend];water=[]
        for i in range(len(s.times)):
            frame=s.read(i);water.append(frame.arrays['water'][s._boundary_mask(frame,'Downstream')].sum())
        axes[1,0].plot(s.times[1:],-np.diff(water)/np.diff(s.times),label=backend)
        frame=s.read(-1);pos=centres(frame);unique=np.unique(pos[:,0]);eta=frame.q[0]+frame.z
        axes[1,1].plot(unique,[np.mean(eta[pos[:,0]==x]) for x in unique],label=backend)
    axes[1,0].axhline(4,color='k',ls=':',label='Inlet 4 m³/s');axes[1,0].set(xlabel='Time [s]',ylabel='Accepted outlet discharge [m³/s]');axes[1,0].legend()
    axes[1,1].plot(unique,[np.mean(f.z[xy[:,0]==x]) for x in unique],color='sienna',label='Bed')
    axes[1,1].plot([19.9,19.9],[.08,.15],color='k',lw=4,label='Crest');axes[1,1].set(xlabel='x [m]',ylabel='Elevation [m]');axes[1,1].legend()
    fig.suptitle('General weir control: backwater, subcritical approach, no imported water');fig.tight_layout();fig.savefig(OUT/'WEIR.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9));frames=[selected(stores[f+'_'+b],120) for f in formulas for b in ('cpu','gpu')]
    hi=max(float((f.z-f.arrays['initial_bed_m']).max()*1000) for f in frames)
    for ax,f,name in zip(axes.flat,frames,[f+' '+b for f in formulas for b in ('CPU','GPU')]):
        pc=picture(ax,f,1000*(f.z-f.arrays['initial_bed_m']),name+' — 120 s',0,hi,'YlOrBr')
        fig.colorbar(pc,ax=ax,label='Deposition [mm]')
    fig.suptitle('One grain, initially non-erodible base, capacity feed; no suspension or banks');fig.tight_layout();fig.savefig(OUT/'BEDLOAD_FORMULAS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,formula in enumerate(('ashida_michiue_generalized','wang_fredsoe')):
        for col,backend in enumerate(('cpu','gpu')):
            f=ResultsStore(BASE/'finite_runs'/(formula+'_'+backend)/'run').read(-1)
            pc=picture(axes[row,col],f,1000*(f.z-f.arrays['initial_bed_m']),formula+' '+backend+' — 60 s',-50,50,'RdBu')
            fig.colorbar(pc,ax=axes[row,col],label='Bed change [mm]')
    fig.suptitle('50 mm erodible stock: raw erosion/deposition, 20 s of active bedload, MF1');fig.tight_layout();fig.savefig(OUT/'FINITE_BED.png',dpi=160);plt.close(fig)
    (OUT/'COMPARISON.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if 'parity' in k or k=='clock_only_trajectory'},indent=2))
if __name__=='__main__':main()
