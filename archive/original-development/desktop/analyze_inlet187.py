"""Raw-cell inlet comparison and provenance, no display smoothing."""
import argparse,json,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_inlet187'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_inlet187'
sys.path[:0]=[str(OUT/'prescribed_candidate'),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha

def at(path,t):
    s=ResultsStore(path);return s.read(next(i for i,v in enumerate(s.times) if abs(v-t)<1e-7))

def xy(f):return np.array([p.centroid.coords[0] for p in f.polygons]).T

def draw(ax,f,field,limit,title,region=True):
    m=xy(f)[0]<5 if region else np.ones(len(f.z),bool)
    coll=PolyCollection([np.asarray(p.exterior.coords) for p,keep in zip(f.polygons,m) if keep],
        array=np.asarray(field)[m],cmap='RdBu_r',clim=(-limit,limit),edgecolors='none',rasterized=True)
    ax.add_collection(coll);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title,fontsize=10)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    return coll

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',choices=('wet','dry','both'),default='both');args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    summary=json.loads((REPORT/'COMPARISON.json').read_text()) if args.start!='both' and (REPORT/'COMPARISON.json').exists() else {}
    for start in (('wet','dry') if args.start=='both' else (args.start,)):
        paths={b:OUT/f'{start}_prescribed_{b}/run' for b in ('cpu','gpu')}
        old=ROOT/('validation/_mesh186/refined_cpu/run' if start=='wet' else 'validation/_secondary185/qualified_cpu_v6/to_220')
        c=at(paths['cpu'],220.);g=at(paths['gpu'],220.);initial=at(paths['cpu'],0.)
        m=xy(c)[0]<5;area=c.arrays['area'];dz=c.z-initial.z;diff=c.z-g.z
        for k in ('q','z','W','mass'):
            assert np.array_equal(initial.arrays[k],at(paths['gpu'],0.).arrays[k]),k
        for k in ('vertices','offsets','levels'):
            assert np.array_equal(initial.mesh[k],at(paths['gpu'],0.).mesh[k]),k
        warm=at(paths['cpu'],200.);oldwarm=at(old,200.);oldlast=at(old,220.)
        metrics=dict(cells=len(c.z),max_bed_difference_mm=float(abs(diff).max()*1000),
            bed_area_rms_mm=float(np.sqrt(np.sum(area*diff**2)/area.sum())*1000),
            bed_cell_rms_mm=float(np.sqrt(np.mean(diff**2))*1000),
            max_depth_difference_mm=float(abs(c.q[0]-g.q[0]).max()*1000),
            inlet_max_bed_difference_mm=float(abs(diff[m]).max()*1000),
            inlet_max_depth_difference_mm=float(abs(c.q[0,m]-g.q[0,m]).max()*1000),
            max_bed_difference_xy=xy(c)[:,int(abs(diff).argmax())].tolist(),
            inlet_bed_min_mm=float(dz[m].min()*1000),inlet_bed_max_mm=float(dz[m].max()*1000),
            inlet_v_at_200=float(abs(warm.q[2,m]/warm.q[0,m]).max()),
            inlet_I_at_200=float(abs(warm.arrays['W'][m]/warm.q[0,m]).max()),
            bend_I_at_200=float(abs(warm.arrays['W'][~m]/warm.q[0,~m]).max()),
            old_inlet_v_at_200=float(abs(oldwarm.q[2,m]/oldwarm.q[0,m]).max()),
            old_inlet_I_at_200=float(abs(oldwarm.arrays['W'][m]/oldwarm.q[0,m]).max()),
            old_inlet_bed_min_mm=float((oldlast.z-oldwarm.z)[m].min()*1000),
            old_inlet_bed_max_mm=float((oldlast.z-oldwarm.z)[m].max()*1000))
        summary[start]=metrics
        fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,f,v,label in [(axs[0],c,1000*diff,'CPU minus GPU bed (mm)'),
                            (axs[1],c,1000*(c.q[0]-g.q[0]),'CPU minus GPU depth (mm)')]:
            obj=draw(ax,f,v,max(abs(v).max(),1e-12),label,False)
            peak=int(abs(v).argmax());px,py=xy(f)[:,peak]
            ax.plot(px,py,'ko',ms=5);ax.annotate(f'Peak ({px:.2f}, {py:.2f})', (px,py),xytext=(8,8),textcoords='offset points',fontsize=8)
            fig.colorbar(obj,ax=ax,shrink=.7)
        fig.suptitle(start.capitalize()+' start at 220 s: remaining errors are not the inlet bands')
        fig.savefig(REPORT/f'{start.upper()}_BACKEND.png',dpi=170);plt.close(fig)
        rows=[(oldwarm.q[2]/oldwarm.q[0],warm.q[2]/warm.q[0],'Transverse velocity at 200 s (m/s)'),
              (oldwarm.arrays['W']/oldwarm.q[0],warm.arrays['W']/warm.q[0],'Secondary intensity at 200 s (m/s)'),
              (1000*(oldlast.z-oldwarm.z),1000*dz,'Bed change at 220 s (mm)')]
        fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        for i,(v0,v1,label) in enumerate(rows):
            lim=max(float(abs(v0[m]).max()),float(abs(v1[m]).max()),1e-12)
            for j,(f,v) in enumerate(((oldwarm if i<2 else oldlast,v0),(warm if i<2 else c,v1))):
                obj=draw(axs[i,j],f,v,lim,('Before' if j==0 else 'Prescribed upstream I = 0')+'\n'+label)
                axs[i,j].text(.02,.02,f'Raw range {v[m].min():.4g} to {v[m].max():.4g}',transform=axs[i,j].transAxes,
                    fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            fig.colorbar(obj,ax=axs[i,:],shrink=.75)
        fig.suptitle(start.capitalize()+' start: unchanged refined mesh, full interior secondary physics, bed slope OFF')
        fig.savefig(REPORT/f'{start.upper()}_INLET.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(2,1,figsize=(12,9),layout='constrained')
        for ax,f,v,label in [(axs[0],warm,warm.arrays['W']/warm.q[0],'Secondary intensity at 200 s (m/s)'),
                            (axs[1],c,1000*dz,'Bed change at 220 s (mm)')]:
            lim=max(float(abs(v).max()),1e-12);obj=draw(ax,f,v,lim,label,False);fig.colorbar(obj,ax=ax,shrink=.8)
        fig.suptitle(start.capitalize()+' start: interior bend response remains active')
        fig.savefig(REPORT/f'{start.upper()}_BENDS.png',dpi=170);plt.close(fig)
        # Time histories distinguish a bounded early transient from regrowth.
        fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for path,label in [(old,'Before'),(paths['cpu'],'Corrected CPU'),(paths['gpu'],'Corrected GPU')]:
            s=ResultsStore(path);t=[];v=[];I=[];d=[]
            for i,time in enumerate(s.times):
                if time>220:continue
                f=s.read(i);t.append(time);wet=f.q[0]>1e-6
                vel=np.divide(f.q[2],f.q[0],out=np.zeros(len(f.z)),where=wet)
                intensity=np.divide(f.arrays['W'],f.q[0],out=np.zeros(len(f.z)),where=wet)
                v.append(abs(vel[m]).max());I.append(abs(intensity[m]).max());d.append(abs((f.z-initial.z)[m]).max()*1000)
            for ax,y in zip(axs,(v,I,d)):ax.plot(t,y,label=label)
        for ax,label in zip(axs,('max inlet |v| (m/s)','max inlet |I| (m/s)','max inlet |bed change| (mm)')):
            ax.set(xlabel='Physical time (s)',ylabel=label);ax.grid(alpha=.25);ax.axvline(200,color='gray',ls=':')
        axs[0].legend();fig.savefig(REPORT/f'{start.upper()}_HISTORY.png',dpi=170);plt.close(fig)
    summary['runtime']=sha(OUT/'prescribed_candidate/RUNTIME.json')
    save(REPORT/'COMPARISON.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
