"""Rebuild S2 numerical receipts and figures from the preserved short tests."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
from test_suspended_core215 import capacity,skew_grid,transport,settling_speed
from braids_core.sediment.river_capacity import ResolvedRiverCapacity,WIRE_SIZE
from braids_core.sediment.bed_columns_v101 import host
DATA=ROOT/'validation/_suspended215/qualified'
RUNS=ROOT/'validation/_suspended215/final_s2'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_suspension215'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def fields(folder,row):
    with np.load(folder/'whole/frames'/row['file']) as z:return {k:z[k].copy() for k in z.files}
cases={}
for receipt in RUNS.glob('test_current_worker*/QUALIFICATION.json'):
    r=json.loads(receipt.read_text());folder=receipt.parent
    ix=json.loads((folder/'whole/frames/INDEX.json').read_text())
    r['epochs']=len(ix['epochs']);r['run']=str(folder.relative_to(ROOT))
    cases[r['case'],r['backend']]=(r,folder,ix)
assert len(cases)==12
parity=[]
for name in ('closed','open','cropped','amr','mixed','entrainment'):
    a,b=[fields(cases[name,k][1],cases[name,k][2]['frames'][-1]) for k in ('cpu','gpu')]
    errors={k:dict(max_abs=float(np.max(abs(a[k]-b[k]))),rms=float(np.sqrt(np.mean((a[k]-b[k])**2))))
            for k in ('q','z','mass','suspended')}
    parity.append(dict(case=name,errors=errors))
save('COUPLED.json',dict(cases=[v[0] for v in cases.values()],backend=parity))
temporal=[json.loads(p.read_text()) for p in RUNS.glob('test_short_timestep_convergenc*/TEMPORAL.json')]
save('TEMPORAL.json',temporal)

# The common flux: old normal-distance assumption versus affine exact solution.
op=ResolvedRiverCapacity('cpu',diameter=.0001,chezy=40.,capacity=capacity(),folder=DATA/'figures_operator')
g=skew_grid();n=len(g['area']);nf=len(g['owner']);grad=np.array([.2,-.1]);h=np.full(n,.3)
m=((2+grad@g['xy'])*h)[None,:];expected=-.02*.3*(grad@g['normal']);inside=g['neighbour']>=0
_,old,_=transport(op,g,h,m,np.zeros(nf),np.zeros((1,nf)),.01,.02,correct=False)
_,new,_=transport(op,g,h,m,np.zeros(nf),np.zeros((1,nf)),.01,.02)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ax=axes[0]
for f in np.flatnonzero(inside):
    ids=[g['owner'][f],g['neighbour'][f]];ax.plot(*g['xy'][:,ids],color='.8',lw=.8)
ax.scatter(*g['xy'],c=m[0]/h,cmap='viridis',s=40);ax.set_aspect('equal')
ax.set(title='Skew cell centres: manufactured linear concentration',xlabel='x (m)',ylabel='y (m)')
ax=axes[1];ids=np.arange(inside.sum())
ax.plot(ids,expected[inside],color='black',label='Exact normal flux')
ax.scatter(ids,old[0,inside],marker='x',color='tab:red',label='Previous two-point flux')
ax.scatter(ids,new[0,inside],facecolor='none',edgecolor='tab:blue',label='Corrected CPU/GPU kernel')
ax.set(xlabel='Internal face',ylabel='Diffusive flux (kg/m/s)',title='Tangential offset no longer creates a false flux');ax.legend(fontsize=8)
fig.savefig(OUT/'OBLIQUE_DIFFUSION.png',dpi=170);plt.close(fig)
save('DIFFUSION.json',dict(old_max_abs_error=float(np.max(abs(old[0,inside]-expected[inside]))),
    corrected_max_abs_error=float(np.max(abs(new[0,inside]-expected[inside]))),units='kg/m/s'))

# Exact exponential frozen-depth/frozen-hindering source, CPU and GPU.
fig,ax=plt.subplots(figsize=(7,4),layout='constrained');curves=[]
for d in (80e-6,110e-6,500e-6):
    hh=.2;mass=.04;ws=settling_speed(d,1.65,20)*(1-mass/(hh*1600))**5
    times=np.linspace(0,3*hh/ws,31);expected=mass*np.exp(-ws*times/hh)
    result={}
    for backend in ('cpu','gpu'):
        p=ResolvedRiverCapacity(backend,diameter=d,chezy=40.,capacity=capacity(d),folder=DATA/f'figure_settling_{backend}_{d}')
        xp=p.xp;out=xp.empty((1,1));delta=out.copy();diag=xp.empty((3,1,1));values=[]
        for dt in times:
            p.engine.launch('settling_only_exchange',1,1,1,WIRE_SIZE,float(dt),1.,1e-10,1600.,20.,
                xp.asarray([d]),xp.asarray([[hh],[0.],[0.]]),xp.asarray([[mass]]),p.wire[None,:].copy(),out,delta,diag)
            values.append(float(host(out)[0,0]))
        result[backend]=np.array(values)
    line=ax.plot(times,expected/mass,label=f'Exact, Ds={d*1e6:g} µm')[0]
    ax.scatter(times[::3],result['cpu'][::3]/mass,marker='o',facecolor='none',edgecolor=line.get_color(),s=30)
    ax.scatter(times[1::3],result['gpu'][1::3]/mass,marker='x',color=line.get_color(),s=20)
    curves.append(dict(diameter_m=d,CPU_max_abs_kg_m2=float(max(abs(result['cpu']-expected))),
                       GPU_max_abs_kg_m2=float(max(abs(result['gpu']-expected)))))
ax.set(xlabel='Time (s)',ylabel='Suspended inventory / initial inventory',ylim=(0,1.04),
       title='Frozen-coefficient settling source: circles CPU, crosses GPU');ax.legend(fontsize=8)
fig.savefig(OUT/'SETTLING.png',dpi=170);plt.close(fig);save('SETTLING.json',curves)

# Raw cells, wet front and deposition: no display interpolation or smoothing.
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
_,reference_folder,reference_ix=cases['amr','cpu']
reference=fields(reference_folder,reference_ix['frames'][-1])
limits=[float(reference['q'][0].max()),float(reference['suspended'][0].max()),
        float(((reference['z']-reference['initial_bed_m'])*1e6).max())]
for row,backend in enumerate(('cpu','gpu')):
    receipt,folder,ix=cases['amr',backend];entry=ix['frames'][-1];a=fields(folder,entry)
    with np.load(folder/'whole/frames'/entry['epoch']) as mesh:
        polygons=[mesh['vertices'][lo:hi] for lo,hi in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
    for col,(key,unit,cmap) in enumerate((('depth','Depth (m)','Blues'),('suspended','Suspended mass (kg/m²)','Oranges'),('deposition','Bed change (µm)','YlGn'))):
        value=a['q'][0] if key=='depth' else a['suspended'][0] if key=='suspended' else (a['z']-a['initial_bed_m'])*1e6
        ax=axes[row,col];collection=PolyCollection(polygons,array=value,cmap=cmap,edgecolor='.45',linewidth=.3,
            clim=(0,limits[col]))
        ax.add_collection(collection);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')
        ax.set_title(f'{backend.upper()} at {receipt["time_s"]:g} s\n{unit}',fontsize=10)
        fig.colorbar(collection,ax=ax,shrink=.8)
fig.savefig(OUT/'AMR_CPU_GPU.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for backend,style in (('cpu','-'),('gpu','--')):
    for name,color in (('closed','tab:blue'),('entrainment','tab:orange')):
        r,folder,ix=cases[name,backend];time=[];susp=[];bed=[]
        for entry in ix['frames']:
            a=fields(folder,entry);time.append(entry['time_s'])
            susp.append(float((a['suspended']*a['area']).sum()));bed.append(float((a['mass']*a['area']).sum()))
        axes[0].plot(time,susp,style,color=color,label=f'{name}, {backend.upper()}')
        axes[1].plot(time,np.array(bed)+susp-(bed[0]+susp[0]),style,color=color,label=f'{name}, {backend.upper()}')
axes[0].set(xlabel='Time (s)',ylabel='Suspended mass (kg)',title='Settling and finite-bed entrainment')
axes[1].set(xlabel='Time (s)',ylabel='Closed total-mass residual (kg)',title='Bed plus suspended inventory')
axes[0].legend(fontsize=8);fig.savefig(OUT/'EXCHANGE_BUDGETS.png',dpi=170);plt.close(fig)
sources=['braids_core/sediment/river_suspension.py','braids_core/sediment/river_suspension.h',
         'braids_core/sediment/river_capacity.py','tests/test_suspended_core215.py',
         'tests/test_suspended_exchange215.py','desktop/check_suspended215.py','desktop/report_suspended215.py',
         'resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/soursin_2d.f90']
save('SOURCES.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
save('DELIVERY.json',dict(scope='checkout S2 common suspended machinery; no Desktop installation',
    installed_runtime_sha256=hashlib.sha256((ROOT/'desktop/runtime/RUNTIME.json').read_bytes()).hexdigest(),
    expected_preview39='0eea57c99b1f6e6818c96de222826d492900356df5141e66709b63f92ad9c114',
    installed_suspension_drafts_remain_blocked=True))
print(json.dumps(dict(cases=len(cases),parity=parity),indent=2))
