"""Numerical comparisons and raw dynamic-mesh figures for the delivered repair."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_amr150'

def load(path,index=-1):
    path=Path(path)/'frames';doc=json.loads((path/'INDEX.json').read_text());row=doc['frames'][index]
    with np.load(path/row['epoch']) as f:mesh={k:f[k].copy() for k in f.files}
    with np.load(path/row['file']) as f:state={k:f[k].copy() for k in f.files}
    return mesh,state

def compare(a,b):
    am,av=load(a);bm,bv=load(b)
    np.testing.assert_array_equal(am['leaves'],bm['leaves'])
    return {k:float(np.max(abs(av[k]-bv[k]))) for k in ('q','z','mass','water','bed_boundary','sus_boundary')}

report=dict(hydrodynamic_backends=compare(OUT/'final_cpu/run',OUT/'final_gpu/run'),
    cpu_restart=compare(OUT/'final_cpu/run',OUT/'final_split_cpu/continued'),
    breakline_backends=compare(OUT/'final_breakline_cpu/run',OUT/'final_breakline_gpu/run'),
    bedload_control_backends=compare(OUT/'control_cpu/run',OUT/'control_gpu/run'),
    bedload_unchanged_cpu=compare(ROOT/'validation/_breaklines147/final_bedload_cpu/run',OUT/'control_cpu/run'),
    bedload_unchanged_gpu=compare(ROOT/'validation/_breaklines147/final_bedload_gpu/run',OUT/'control_gpu/run'))
assert max(report['cpu_restart'].values())==0
assert max(report['bedload_unchanged_cpu'].values())==0
assert max(report['bedload_unchanged_gpu'].values())==0
assert report['hydrodynamic_backends']['q']<1e-5
assert report['breakline_backends']['q']<1e-5
assert report['bedload_control_backends']['z']<1e-8
(OUT/'COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
origin=np.array([2571480.,1106944.]);states=[load(OUT/f'final_{b}/run') for b in ('cpu','gpu')]
common=max(float(s['q'][0].max()) for _,s in states)
for ax,(mesh,state),label in zip(axes[:2],states,['CPU','GPU']):
    polys=[mesh['vertices'][a:b]-origin for a,b in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
    h=state['q'][0];wet=h>=.001
    ax.add_collection(PolyCollection(polys,facecolors='#f5f6f8',edgecolors='#b6bcc3',linewidths=.25))
    layer=PolyCollection([p for p,ok in zip(polys,wet) if ok],array=h[wet],cmap='Blues',edgecolors='#687d90',linewidths=.3)
    layer.set_clim(0,common);ax.add_collection(layer)
    ax.set_xlim(-60,55);ax.set_ylim(-4,130);ax.set_aspect('equal');ax.set_title(label+' at 20 s')
    ax.set_xlabel('Easting relative to inlet (m)');ax.set_ylabel('Northing relative to inlet (m)')
    ax.plot(-6.4586538756,13.0717263864,'rx',ms=7)
fig.colorbar(layer,ax=axes[:2],label='Water depth (m)',shrink=.8)
for b,style in [('cpu','-'),('gpu','--')]:
    events=json.loads((OUT/f'final_{b}/run/ADAPTATION.json').read_text())
    axes[2].plot([e['time_s'] for e in events],[e['to_cells'] for e in events],style,label=b.upper())
axes[2].axvline(3,color='#b93838',ls=':',label='Previous failure')
axes[2].set_xlabel('Simulation time (s)');axes[2].set_ylabel('Active mesh cells')
axes[2].set_title('Identical refinement/coarsening history');axes[2].legend();axes[2].grid(alpha=.2)
fig.suptitle('Dynamic AMR repair: raw cells and water depth; red cross marks former wall-check failure')
fig.savefig(OUT/'COMPARISON.png',dpi=160);plt.close(fig)
