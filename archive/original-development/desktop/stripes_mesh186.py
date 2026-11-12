"""Raw-cell inlet evidence; no smoothing or numerical/physical edits."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_desktop.results import ResultsStore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_mesh186'


def frame(path,t):
    store=ResultsStore(path)
    i=np.flatnonzero(np.isclose(store.times,t,rtol=0,atol=1e-8))
    if len(i)!=1:raise ValueError((str(path),t,store.times.tolist()))
    return store.read(int(i[0]))


def map_field(ax,f,a,lo,hi,title,cmap='RdBu_r'):
    poly=[np.asarray(p.exterior.coords) for p in f.polygons]
    pc=PolyCollection(poly,array=a,cmap=cmap,edgecolors='#333333',linewidths=.18)
    pc.set_clim(lo,hi);ax.add_collection(pc)
    ax.set(xlim=(0,10),ylim=(-2.05,2.05),xlabel='x (m)',ylabel='y (m)',title=title)
    ax.set_aspect('equal')
    return pc


def initial():
    REPORT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'desktop/examples/two_meanders184/advanced_on/runs/to_400',
           ROOT/'validation/_secondary185/qualified_cpu_v6/to_220']
    fs=[frame(p,220) for p in paths]
    fig,axs=plt.subplots(2,2,figsize=(14,7),layout='constrained')
    for row,(f,title) in enumerate(zip(fs,('Original coarse CPU, wet start, preview29','Refined CPU, dry start, corrected v6'))):
        p=map_field(axs[row,0],f,1000*(f.z-f.arrays['initial_bed_m']),-120,120,title+' | bed change (mm)')
        fig.colorbar(p,ax=axs[row,0],shrink=.7)
        p=map_field(axs[row,1],f,f.mesh['levels'],0,2,'Refinement levels',cmap='viridis')
        fig.colorbar(p,ax=axs[row,1],shrink=.7,ticks=[0,1,2])
    fig.suptitle('Historical inlet snapshots at 220 s: mesh, initial wetness and runtime are NOT matched')
    fig.savefig(REPORT/'HISTORICAL_STRIPES.png',dpi=170);plt.close(fig)


if __name__=='__main__':initial()
