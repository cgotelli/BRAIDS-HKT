"""Aggregate measured qualification evidence and draw unsmoothed comparisons."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from stage_collapse222 import ROOT,OUT,TARGET,REPORT,save,sha

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    root=OUT/'qualification2';runs={};paired=[]
    for p in root.glob('*/IDENTITY.json'):
        d=json.loads(p.read_text());runs[d['bed'],d['sus'],d['backend']]=p.parent
    assert len(runs)==142
    for bed,sus,backend in runs:
        if backend!='cpu':continue
        a,b=(np.load(runs[bed,sus,k]/'FIELDS.npz') for k in ('cpu','gpu'))
        dz=(a['z']-b['z'])*1000
        paired.append(dict(bed=bed,suspension=sus,max_bed_mm=float(abs(dz).max()),rms_bed_mm=float(np.sqrt(np.mean(dz**2)))))
    temporal=[json.loads(p.read_text()) for p in (OUT/'qualification3').glob('*/TEMPORAL.json')]
    checks=[json.loads(p.read_text()) for r in (root,OUT/'qualification3') for p in r.rglob('METRICS.json')]
    source=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/avalan.f90'
    save(OUT/'METRICS.json',dict(runtime=sha(TARGET/'RUNTIME.json'),paired=paired,temporal=temporal,
        max_water_residual_m3=max(r['water'] for r in checks),max_sediment_residual_kg=max(r['sediment'] for r in checks),
        exact_restart_cases=sum(r.get('restart_exact',False) for r in checks),
        Delft3D_sources_read_only={str(source.relative_to(ROOT)):sha(source)}))
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].plot(np.arange(1,72),[max(1e-12,p['max_bed_mm']) for p in paired],'.')
    ax[0].set(yscale='log',xlabel='Compatible transport configuration',ylabel='Maximum bed difference (mm)',title='CPU versus GPU: 71 configurations')
    ax[0].text(.02,.92,'Values below 1e-12 mm plotted at that floor',transform=ax[0].transAxes,fontsize=8)
    ax[1].bar(np.arange(len(temporal)),[p['max_bed_mm'] for p in temporal]);ax[1].set(xlabel='MPM / MPM+GP / VR1993 (CPU and GPU)',ylabel='Maximum bed difference (mm)',title='Halving timestep: 1 ms to 0.5 ms')
    fig.savefig(REPORT/'QUALIFICATION.png',dpi=170);plt.close(fig)
    # Actual polygon cells, no interpolation that could hide near-wall defects.
    cases=ROOT/'desktop/examples/collapse222';data={}
    for case in sorted(cases.iterdir()):
        index=json.loads((case/'runs/example/frames/INDEX.json').read_text());frames=[]
        for row in index['frames']:
            with np.load(case/'runs/example/frames'/row['file']) as f:frames.append({k:f[k].copy() for k in f.files})
        mesh=np.load(case/'runs/example/frames'/index['frames'][-1]['epoch'])
        vertices=mesh['vertices'];off=mesh['offsets'];polys=[vertices[off[i]:off[i+1]] for i in range(len(off)-1)]
        xy=np.array([p[:-1].mean(0) for p in polys]);data[case.name]=(frames,polys,xy)
    fig,axs=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
    limit=max(float(abs(f[-1]['z']-f[0]['z']).max()) for f,p,x in data.values())*1000
    names=['cpu_MF25_no_collapse','cpu_MF25_collapse']
    for ax,name in zip(axs[0],names):
        f,p,xy=data[name];v=(f[-1]['z']-f[0]['z'])*1000
        pc=PolyCollection(p,array=v,cmap='RdBu_r',clim=(-limit,limit),edgecolors='0.35',linewidths=.25)
        ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set(title=name.replace('_',' '),xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(pc,ax=ax,label='Bed change at 0.4 s (mm)')
    for name,(f,p,xy) in data.items():
        # Same physical cells for each fixed-mesh example. Narrow section at x~1.
        take=np.abs(xy[:,0]-1.)<.13;order=np.where(take)[0][np.argsort(xy[take,1])]
        axs[1,0].plot(xy[order,1],f[-1]['z'][order],'.-',label=name.replace('_',' '),lw=1)
        axs[1,1].plot([float(v['time']) for v in f],[float(v['collapse_gross'].sum()) for v in f],'.-',label=name.replace('_',' '),lw=1)
    first,_,xy=data['cpu_MF25_collapse'];axs[1,0].plot(xy[order,1],first[0]['z'][order],'k--',label='Initial bed')
    axs[1,0].set(xlabel='y (m)',ylabel='Bed elevation (m)',title='Raw bank section near x = 1 m');axs[1,0].legend(fontsize=7)
    axs[1,1].set(xlabel='Hydraulic time (s)',ylabel='Gross bed redistribution (kg)',title='Collapse is not multiplied by MF');axs[1,1].legend(fontsize=7)
    fig.savefig(REPORT/'BANK_COMPARISON.png',dpi=180);plt.close(fig)
    # Independent algebraic contrast, not a new Delft3D simulation.
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    area=np.array([1.,.3]);z=np.array([2.,0.]);distance=.65;angle=np.deg2rad(32)
    flat=np.repeat(z@area/area.sum(),2)
    request=(z[0]-z[1]-np.tan(angle)*distance)/(1/area[0]+1/area[1])
    repose=z+np.array([-request/area[0],request/area[1]])
    for y,name in ((z,'Initial bed'),(flat,'Ordinary threshold / equalization'),(repose,'Jiang-Cao repose target (32 degrees)')):ax.plot([0,distance],y,'o-',label=name)
    ax.set(xlabel='Centroid distance (m)',ylabel='Bed elevation (m)',title='The two collapse models intentionally have different targets');ax.legend(fontsize=8)
    fig.savefig(REPORT/'COLLAPSE_METHODS.png',dpi=170);plt.close(fig)
    # Explicit scientific scope for the synthetic example.
    a=data['cpu_MF25_collapse'][0][-1];b=data['gpu_MF25_collapse'][0][-1]
    save(OUT/'EXAMPLE_COMPARISON.json',dict(cpu_gpu_max_bed_mm=float(abs(a['z']-b['z']).max()*1000),
        scope='Closed synthetic partially dry finite bank, cropped walls; no experimental validation or new FLOW trajectory.'))
    print('Figures and measured metrics written',flush=True)
if __name__=='__main__':main()
