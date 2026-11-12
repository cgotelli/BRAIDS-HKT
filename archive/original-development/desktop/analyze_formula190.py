"""Published curve and raw-cell paired backend figures for the new laws."""
import sys,json
from stage_formula190 import ROOT,TARGET,OUT
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from test_ancey_recking190 import phi
from test_recking190 import oracle
from stage_secondary185 import save,sha
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'
LABELS={'ancey_recking_2023':'Ancey & Recking (2023)','recking_2010':'Recking (2010)'}

def main():
    REPORT.mkdir(parents=True,exist_ok=True);rows={}
    for start in ('wet','dry'):
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for r,(law,label) in enumerate(LABELS.items()):
            paths=[OUT/'runs'/start/(law+'_'+b) for b in ('cpu','gpu')]
            checks=[json.loads((p/'CHECK.json').read_text()) for p in paths]
            stores=[ResultsStore(p/'full') for p in paths]
            assert np.array_equal(stores[0].times,stores[1].times)
            maxbed=maxdepth=0.;hotspot=None
            for i in range(len(stores[0].times)):
                a,b=[s.read(i) for s in stores]
                for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(a.mesh[k],b.mesh[k])
                if i==0:
                    for k in ('q','z','mass'):np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
                delta=abs(b.z-a.z)
                if delta.max()>maxbed:
                    j=int(delta.argmax());hotspot=dict(time_s=float(stores[0].times[i]),
                        xy=list(a.polygons[j].centroid.coords[0]),cpu_depth_m=float(a.q[0,j]),gpu_depth_m=float(b.q[0,j]))
                maxbed=max(maxbed,float(abs(b.z-a.z).max()));maxdepth=max(maxdepth,float(abs(b.q[0]-a.q[0]).max()))
            assert maxbed<.001 and maxdepth<.001,(law,start,maxbed,maxdepth)
            xy=np.array([p.centroid.coords[0] for p in a.polygons]);z0=.1-.01*xy[:,0]
            da,db=(a.z-z0)*1000,(b.z-z0)*1000;limit=max(abs(da).max(),abs(db).max())
            for ax,values,title in zip(axes[r],[da,db,db-da],['CPU','GPU','GPU minus CPU']):
                scale=max(float(abs(values).max()),1e-9) if 'minus' in title else limit
                p=PolyCollection([np.asarray(p.exterior.coords) for p in a.polygons],array=values,
                    cmap='RdBu_r',clim=(-scale,scale),edgecolors='none')
                ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(label+'\n'+title)
                ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.colorbar(p,ax=ax,shrink=.8,label='Bed change (mm)')
            rows[start+'_'+law]=dict(maximum_backend_bed_mm=maxbed*1000,maximum_backend_depth_mm=maxdepth*1000,
                final_bed_rms_mm=float(np.sqrt(np.average((db-da)**2,weights=a.arrays['area']))),
                maximum_difference_location=hotspot,checks=checks)
        fig.suptitle(start.capitalize()+' start: 16 s groin, bedload active from 4 s, MF1; native cells')
        fig.savefig(REPORT/('GROIN_'+start.upper()+'.png'),dpi=160);plt.close(fig)
    theta=np.geomspace(.005,10,801);fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].loglog(theta,phi(theta),label='Ancey & Recking, Eq. 42')
    axes[0].loglog(theta,8*np.maximum(theta-.047,0)**1.5,'--',label='MPM textbook curve (no drag reduction)')
    axes[0].set_title('Smooth low-transport tail and high-stress scaling')
    for s in (.001,.01,.05):axes[1].loglog(theta,oracle(theta,s),label=f'Recking 2010, S={s:g}')
    axes[1].set_title('Explicit reach slope changes partial-mobility transport')
    for ax in axes:
        ax.set_xlabel('Shields stress');ax.set_ylabel('Dimensionless solid-volume bedload Φ')
        ax.set_ylim(1e-12,1e4);ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    fig.savefig(REPORT/'PUBLISHED_CURVES.png',dpi=160);plt.close(fig)
    save(REPORT/'COMPARISON.json',dict(runtime_sha256=sha(TARGET/'RUNTIME.json'),cases=rows))
    print({k:{x:y for x,y in v.items() if x!='checks'} for k,v in rows.items()})
if __name__=='__main__':main()
