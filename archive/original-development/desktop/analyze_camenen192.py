"""Equation reproduction and raw-cell coupled CPU/CUDA figures."""
import json,sys,os
from stage_camenen192 import ROOT,OUT,TARGET,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.river_capacity import ResolvedRiverCapacity
from test_smart_jaggi183 import probe
from test_camenen192 import SPEC,phi,critical
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_camenen192'
def main():
    REPORT.mkdir(parents=True,exist_ok=True);verify()
    install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    theta=np.geomspace(.003,3.,601);d=.005;scale=np.sqrt(9.81*1.65*d**3)
    expected=phi(theta);ax[0].loglog(theta,expected,'k-',label='Independent published Eq. 81')
    errors={}
    for backend,marker in [('cpu','o'),('gpu','x')]:
        op=ResolvedRiverCapacity(backend,diameter=d,chezy=35.,capacity=SPEC,folder=OUT/'curve'/backend)
        got=probe(op,35*np.sqrt(theta*1.65*d),.2,35.)[0]/scale
        np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-100)
        errors[backend]=float(np.max(abs(got/expected-1)))
        ax[0].loglog(theta[::30],got[::30],marker,ms=5,label='BRAIDS '+backend.upper())
    ax[0].axvline(critical(),color='gray',ls=':',label='Computed critical Shields')
    ax[0].set(xlabel='Shields stress θ',ylabel='Dimensionless bedload Φ',ylim=(1e-10,100),title='D50 = 5 mm; total hydraulic shear')
    diam=np.geomspace(.08,200,501)
    ax[1].semilogx(diam,critical(diam*.001),label='ν = 1.0 × 10⁻⁶ m²/s')
    ax[1].semilogx(diam,critical(diam*.001,nu=1.5e-6),'--',label='ν = 1.5 × 10⁻⁶ m²/s')
    ax[1].set(xlabel='Median grain size D50 (mm)',ylabel='Critical Shields θcr',ylim=(0,None),title='Soulsby–Whitehouse; authors’ Eq. 49')
    for a in ax:a.grid(alpha=.2,which='both');a.legend(fontsize=8)
    fig.suptitle('Camenen & Larson — steady-current bedload only')
    fig.savefig(REPORT/'EQUATIONS.png',dpi=170);plt.close(fig)
    cases={}
    for start in ('wet','dry'):
        paths=[OUT/os.environ.get('BRAIDS_RUN_TAG','qualification')/'runs'/start/('camenen_larson_2005_'+b) for b in ('cpu','gpu')]
        checks=[json.loads((p/'CHECK.json').read_text()) for p in paths]
        stores=[ResultsStore(p/'full') for p in paths]
        np.testing.assert_array_equal(stores[0].times,stores[1].times)
        maxbed=maxdepth=0.;series=[];hotspot=None
        for i,t in enumerate(stores[0].times):
            a,b=[s.read(i) for s in stores]
            for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(a.mesh[k],b.mesh[k])
            if i==0:
                for k in ('q','z','mass'):np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
            if float(abs(b.z-a.z).max())>maxbed:
                j=int(abs(b.z-a.z).argmax());hotspot=dict(time_s=float(t),xy=list(a.polygons[j].centroid.coords[0]),
                    cpu_depth_m=float(a.q[0,j]),gpu_depth_m=float(b.q[0,j]))
            maxbed=max(maxbed,float(abs(b.z-a.z).max()));maxdepth=max(maxdepth,float(abs(b.q[0]-a.q[0]).max()))
            series.append(dict(time_s=float(t),bed_max_difference_mm=float(abs(b.z-a.z).max()*1000)))
        assert maxbed<.001 and maxdepth<.001,(start,maxbed,maxdepth)
        xy=np.array([p.centroid.coords[0] for p in a.polygons]);z0=.1-.01*xy[:,0]
        da,db=(a.z-z0)*1000,(b.z-z0)*1000;limit=max(abs(da).max(),abs(db).max())
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
        for axis,values,title in zip(axes,[da,db,db-da],['CPU','GPU','GPU minus CPU']):
            scale=max(float(abs(values).max()),1e-9) if 'minus' in title else limit
            poly=PolyCollection([np.asarray(p.exterior.coords) for p in a.polygons],array=values,
                cmap='RdBu_r',clim=(-scale,scale),edgecolors=(0,0,0,.14),linewidths=.15)
            axis.add_collection(poly);axis.autoscale_view();axis.set_aspect('equal')
            axis.set(xlabel='x (m)',ylabel='y (m)',title=title)
            fig.colorbar(poly,ax=axis,shrink=.7,label='Bed change (mm)')
        fig.suptitle(start.capitalize()+' start groin, 16 s, bedload from 4 s, MF1; native cells (no smoothing)')
        fig.savefig(REPORT/('GROIN_'+start.upper()+'.png'),dpi=170);plt.close(fig)
        cases[start]=dict(maximum_backend_bed_mm=maxbed*1000,maximum_backend_depth_mm=maxdepth*1000,
            final_bed_rms_mm=float(np.sqrt(np.average((db-da)**2,weights=a.arrays['area']))),
            maximum_difference_location=hotspot,checks=checks,series=series)
    save(REPORT/'COMPARISON.json',dict(runtime=sha(TARGET/'RUNTIME.json'),equation_relative_error=errors,cases=cases))
    print(json.dumps(dict(equation_relative_error=errors,cases={k:{a:b for a,b in v.items() if a not in ('checks','series')} for k,v in cases.items()}),indent=2))
if __name__=='__main__':main()
