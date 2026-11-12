"""Quantify the old boundary feedback and the rejected extrapolation control."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_inlet187'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_inlet187'
sys.path[:0]=[str(OUT/'prescribed_candidate'),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from braids_desktop.results import ResultsStore
from stage_secondary185 import save

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    traces={label:json.loads((OUT/path/'TRACE.json').read_text()) for label,path in
        [('Old equilibrium inflow','trace_gpu'),('Transported extrapolation (rejected)','transported_cpu')]}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label,rows in traces.items():
        t=[r['time_s'] for r in rows]
        for ax,key,title in zip(axes.flat,('curvature','equilibrium','transported','flux'),
            ('Maximum incoming-cell |curvature| (1/m)','Maximum adjacent equilibrium |Ieq| (m/s)',
             'Maximum transported inlet |I| (m/s)','Maximum imported |Q I| per face width (m³/s²)')):
            ax.plot(t,[max(abs(np.array(r[key])),default=0.) for r in rows],label=label)
            ax.set(xlabel='Physical time (s)',ylabel=title);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.savefig(REPORT/'BOUNDARY_TRACE.png',dpi=170);plt.close(fig)
    r=min(traces['Old equilibrium inflow'],key=lambda r:abs(r['time_s']-.2))
    i=int(np.argmax(abs(np.asarray(r['equilibrium']))));f=ResultsStore(ROOT/'validation/_mesh186/refined_gpu/run').read(1)
    c=r['cells'][i];length=.1;area=f.arrays['area'][c]
    # Use the saved 1 s depth only for an illustrative ratio; no claim that
    # this is the exact depth of the 0.2 s trace.
    crossing=abs(r['water'][i])*length/(f.q[0,c]*area)
    sample={key:r[key][i] for key in ('curvature','equilibrium','transported','source','sink','water','flux')}
    sample.update(time_s=r['time_s'],cell=c,xy=np.asarray(r['xy'])[:,i].tolist(),
        relaxation_time_s=1/r['sink'][i],estimated_crossing_frequency_s_1=float(crossing),
        estimate_note='Cell area/face length exact; illustrative crossing frequency uses saved 1 s depth, not unsaved 0.2 s depth')
    rejected={}
    for b in ('cpu','gpu'):
        p=OUT/f'wet_{b}';s=ResultsStore(p/'run');f=s.read(-1);m=np.array([p.centroid.x for p in f.polygons])<5
        rejected[b]=dict(last_complete_frame_s=s.times[-1],inlet_v=float(abs(f.q[2,m]/f.q[0,m]).max()),
            inlet_I=float(abs(f.arrays['W'][m]/f.q[0,m]).max()),
            disposition='Stopped deliberately after inlet regrowth; interrupted run is not a qualified completion')
    save(REPORT/'TRACE.json',dict(first_row_sample=sample,rejected_transport_extrapolation=rejected))
    print(json.dumps(sample,indent=2))

if __name__=='__main__':main()
