"""S3 figures and audited source/replay/coupled receipts from retained outputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
from stage_suspended216 import OUT,REPORT,FILES,save,sha
from validation.sediment.river_morphodynamics.unified_vr1993_reference import HERE,replay
REPORT.mkdir(parents=True,exist_ok=True)
def fields(folder,entry):
    with np.load(folder/'run/frames'/entry['file']) as a:return {k:a[k].copy() for k in a.files}
cases={}
for p in (OUT/'packaged_v3').glob('test_worker*/METRICS.json'):
    r=json.loads(p.read_text());ix=json.loads((p.parent/'run/frames/INDEX.json').read_text())
    r['epochs']=len(ix['epochs']);r['path']=str(p.parent.relative_to(ROOT))
    cases[r['case'],r['bedload'],r['backend']]=(r,p.parent,ix)
assert len(cases)==16
parity=[]
for name in ('open','cropped','amr','entrainment'):
    for bed in (False,True):
        a,b=[fields(cases[name,bed,k][1],cases[name,bed,k][2]['frames'][-1]) for k in ('cpu','gpu')]
        parity.append(dict(case=name,bedload=bed,errors={k:dict(max_abs=float(np.max(abs(a[k]-b[k]))),
            rms=float(np.sqrt(np.mean((a[k]-b[k])**2)))) for k in ('q','z','mass','suspended')}))
save(REPORT/'COUPLED.json',dict(cases=[v[0] for v in cases.values()],backend=parity))
save(REPORT/'TEMPORAL.json',[json.loads(p.read_text()) for p in (OUT/'packaged_v3').glob('test_timestep*/TEMPORAL.json')])
# Same-input reproduction of the source model; never modify its runtime or input.
replays=[]
for name in ('default','specified_roughness','evolving_dss'):
    for backend in ('cpu','gpu'):
        replays.append(replay(HERE/'reports/unified_qgis_v1/vr1993_flow_v1'/name,name,backend,REPORT/('replay_'+name)))
save(REPORT/'FLOW_REPLAY.json',replays)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for backend,marker in (('cpu','o'),('gpu','x')):
    with np.load(REPORT/'replay_default'/f'{backend}.npz') as a:
        for ax,our,native,title in ((axes[0],11,None,'Bedload magnitude (kg/m/s)'),(axes[1],2,2,'Equilibrium concentration (kg/m³)')):
            x=[];y=[]
            for it in (1,10,20):
                for k in (0,1):
                    v=a[f'native_{it}_{k}'];x.extend(np.hypot(v[0],v[1]) if native is None else v[native]);y.extend(a[f'braids_{it}_{k}'][our])
            ax.scatter(x,y,s=10,marker=marker,alpha=.6,label=backend.upper())
            lo,hi=min(x),max(x);ax.plot([lo,hi],[lo,hi],color='.5',lw=.7)
            ax.set(xlabel='Unmodified FLOW 2DH',ylabel='BRAIDS same-input producer',title=title);ax.legend()
fig.savefig(REPORT/'FLOW_REPLAY.png',dpi=170);plt.close(fig)
# Raw AMR cells. No interpolated shoreline or smoothed colour comparison.
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
arrays=[];polygons=[]
for backend in ('cpu','gpu'):
    r,folder,ix=cases['amr',True,backend];entry=ix['frames'][-1];a=fields(folder,entry);arrays.append(a)
    with np.load(folder/'run/frames'/entry['epoch']) as m:polygons.append([m['vertices'][lo:hi] for lo,hi in zip(m['offsets'][:-1],m['offsets'][1:])])
for col,(label,func,cmap) in enumerate((('Water depth (m)',lambda a:a['q'][0],'Blues'),
    ('Suspended inventory (kg/m²)',lambda a:a['suspended'][0],'Oranges'),
    ('Bed change (mm)',lambda a:(a['z']-a['initial_bed_m'])*1000,'RdBu_r'))):
    vals=[func(a) for a in arrays];vmin=min(v.min() for v in vals);vmax=max(v.max() for v in vals)
    if col==2:vmax=max(abs(vmin),abs(vmax));vmin=-vmax
    for row,backend in enumerate(('cpu','gpu')):
        ax=axes[row,col];p=PolyCollection(polygons[row],array=vals[row],cmap=cmap,clim=(vmin,vmax),edgecolor='.5',linewidth=.3)
        ax.add_collection(p);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')
        ax.set_title(f'{backend.upper()}\n{label}',fontsize=11)
        fig.colorbar(p,ax=ax,shrink=.8)
fig.savefig(REPORT/'AMR_CPU_GPU.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for bed,color in ((False,'tab:blue'),(True,'tab:orange')):
    for backend,style in (('cpu','-'),('gpu','--')):
        r,folder,ix=cases['entrainment',bed,backend];times=[];susp=[];total=[]
        for entry in ix['frames']:
            a=fields(folder,entry);times.append(entry['time_s']);susp.append(float((a['suspended']*a['area']).sum()))
            total.append(float(((a['mass'].sum(0)+a['suspended'])*a['area']).sum()))
        axes[0].plot(times,susp,style,color=color,label=f'{"Combined" if bed else "Suspended only"}, {backend.upper()}')
        axes[1].plot(times,np.array(total)-total[0],style,color=color)
axes[0].set(xlabel='Time (s)',ylabel='Suspended mass (kg)',title='Entrainment from a 10 µm finite bed');axes[0].legend(fontsize=8)
axes[1].set(xlabel='Time (s)',ylabel='Bed + suspension mass change (kg)',title='Closed-domain conservation')
fig.savefig(REPORT/'FINITE_BED.png',dpi=170);plt.close(fig)
source=ROOT/'resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src'
paths=[ROOT/n for n in FILES]+[source/n for n in ('bedbc1993.f90','bedtr1993.f90','calseddf1993.f90','soursin_2d.f90')]
paths += [ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute_sediment/erosed.f90',
          ROOT/'tests/test_suspended_vr216.py',ROOT/'tests/test_river_vr1993_binding.py']
save(REPORT/'SOURCES.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
print(json.dumps(dict(cases=len(cases),max_bed_backend_m=max(r['errors']['z']['max_abs'] for r in parity),
    max_suspended_backend=max(r['errors']['suspended']['max_abs'] for r in parity),
    max_FLOW_relative_l2=max(v for r in replays for row in r['rows'] for k,v in row.items() if k.endswith('_relative_l2'))),indent=2))
