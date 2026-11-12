"""Compare retained short CPU/CUDA runs and make inspectable mesh/field plots."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1]/'validation/_breaklines147'
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)

def read(folder,last=True):
    path=folder/'frames';index=json.loads((path/'INDEX.json').read_text());row=index['frames'][-1 if last else 0]
    with np.load(path/row['file']) as a:values={k:a[k].copy() for k in a.files}
    with np.load(path/row['epoch']) as a:mesh={k:a[k].copy() for k in a.files}
    polys=[mesh['vertices'][lo:hi] for lo,hi in zip(mesh['offsets'],mesh['offsets'][1:])]
    return values,mesh,polys

results={}
for case in ('lake','wetfront','bedload'):
    roots=[ROOT/f'final_{case}_{b}' for b in ('cpu','gpu')]
    states=[read(r/'run') for r in roots];a,b=[s[0] for s in states]
    np.testing.assert_array_equal(states[0][1]['leaves'],states[1][1]['leaves'])
    result=dict(depth_cpu_gpu_max_m=float(abs(a['q'][0]-b['q'][0]).max()),bed_cpu_gpu_max_m=float(abs(a['z']-b['z']).max()),
        momentum_cpu_gpu_max_m2_s=float(abs(a['q'][1:]-b['q'][1:]).max()),minimum_depth_m=float(min(a['q'][0].min(),b['q'][0].min())))
    assert result['depth_cpu_gpu_max_m']<1e-7 and result['bed_cpu_gpu_max_m']<1e-7
    for root in roots:
        name=root.name.rsplit('_',1)[-1];rows=json.loads((root/'run/DIAGNOSTICS.json').read_text())
        result[name]=dict(water_budget_max_m3=max(abs(r['water_residual_m3']) for r in rows),
            sediment_budget_max_kg=max(max(abs(x) for x in r['class_residual_kg']) for r in rows),
            wall_integral_max=max(max(abs(x) for x in r['wall_integrals']) for r in rows),
            actual_regrids=sum(r['changed'] for r in json.loads((root/'run/ADAPTATION.json').read_text())))
        assert result[name]['water_budget_max_m3']<1e-9 and result[name]['sediment_budget_max_kg']<1e-8
        assert result[name]['wall_integral_max']==0
        first=read(root/'run',False)[0];last=read(root/'run')[0]
        if case=='lake':
            result[name]['rest_depth_change_m']=float(abs(last['q'][0]-first['q'][0]).max())
            result[name]['rest_momentum_m2_s']=float(abs(last['q'][1:]).max())
            assert result[name]['rest_depth_change_m']<1e-12 and result[name]['rest_momentum_m2_s']<1e-12
        if case=='wetfront':assert result[name]['actual_regrids']>=1
        if case=='bedload':
            resumed=read(root/'restarted')[0]
            result[name]['restart_q_max']=float(abs(resumed['q']-last['q']).max())
            result[name]['restart_bed_max_m']=float(abs(resumed['z']-last['z']).max())
            result[name]['bed_change_max_m']=float(abs(last['z']-first['z']).max())
            assert result[name]['restart_q_max']<1e-12 and result[name]['restart_bed_max_m']<1e-12
            assert result[name]['bed_change_max_m']>0
    results[case]=result
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    initial=read(roots[0]/'run',False)[0]
    fields=[s[0]['q'][0] if case!='bedload' else (s[0]['z']-initial['z'])*1000 for s in states]
    lo=min(x.min() for x in fields);hi=max(x.max() for x in fields)
    for i,ax in enumerate(axes):
        v=fields[i] if i<2 else fields[1]-fields[0]
        collection=PolyCollection(states[0][2],array=v,cmap='Blues' if case!='bedload' and i<2 else 'RdBu_r',edgecolors='#777777',linewidths=.25)
        if i<2:collection.set_clim((lo if case!='bedload' else -max(abs(lo),abs(hi))), (hi if case!='bedload' else max(abs(lo),abs(hi))))
        else:
            bound=max(float(abs(v).max()),1e-12);collection.set_clim(-bound,bound)
        ax.add_collection(collection);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(('CPU','GPU','GPU − CPU')[i])
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.colorbar(collection,ax=ax,shrink=.65,label='Depth (m)' if case!='bedload' else 'Bed change (mm)')
        if i==2:ax.text(.5,-.38,f'Maximum absolute difference: {abs(v).max():.3g}',ha='center',transform=ax.transAxes)
        doc=json.loads((roots[0]/'braids_project.json').read_text())
        for row in doc['settings']['breaklines']:
            p=np.array(row['points_xy']+[row['points_xy'][0]]);ax.plot(p[:,0],p[:,1],color='#c43fad',lw=1.7)
    fig.suptitle(case+' — 2 simulated seconds; magenta outlines are permeable mesh constraints')
    fig.savefig(OUT/(case.upper()+'.png'),dpi=160);plt.close(fig)
(OUT/'COMPARISON.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
