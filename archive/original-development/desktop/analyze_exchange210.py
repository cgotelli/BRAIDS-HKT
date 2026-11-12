"""Raw-cell and independent ODE figures for the bounded exchange qualification."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_exchange210';RUNTIME=WORK/'candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_exchange210'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from braids_core.sediment.portable_river_runtime import install,_load_lib
install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
_dependency=_load_lib(RUNTIME/'precompiled/0201330b952952aedb9a/libdl.dll')
REPORT.mkdir(parents=True,exist_ok=True)

def compare(a,b):
    np.testing.assert_allclose(a.times,b.times,atol=1e-12,rtol=0)
    record={};exact=True
    for i in range(len(a.times)):
        f,g=a.read(i),b.read(i);np.testing.assert_array_equal(f.mesh['leaves'],g.mesh['leaves'])
        for name,x,y,scale in [('depth_mm',f.q[0],g.q[0],1000),('bed_mm',f.z,g.z,1000),
            ('mobile_mm',f.arrays['mobile_bedload_solid_m'],g.arrays['mobile_bedload_solid_m'],1000),
            ('intensity_m_s',f.arrays['secondary_intensity_m_s'],g.arrays['secondary_intensity_m_s'],1)]:
            e=scale*(y-x);v=record.setdefault(name,dict(maximum=0.,rms=0.))
            v['maximum']=max(v['maximum'],float(abs(e).max()))
            v['rms']=max(v['rms'],float(np.sqrt(np.average(e*e,weights=f.mesh['area']))))
        exact=exact and all(np.array_equal(f.arrays[k],g.arrays[k]) for k in f.arrays)
    record['all_saved_arrays_exact']=exact
    return record

def checks(folder):
    status=json.loads((folder/'run/STATUS.json').read_text());diag=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
    r={k:status[k] for k in ('status','time_s','steps')}
    r.update(rejected_trials=len(status['rejections']),regrids=sum(e['changed'] for e in status['adaptation_events']),
        budgets={k:max(abs(d[k]) for d in diag) for k in ('solid_residual_m3','water_plus_pore_residual_m3')})
    if (folder/'RESTART.json').exists():r['exact_restart']=all(json.loads((folder/'RESTART.json').read_text())['exact'].values())
    return r

def summary(store):
    f=store.read(-1);dz=f.z-f.arrays['initial_bed_m'];area=f.mesh['area']
    record=dict(depth_max_m=float(f.q[0].max()),bed_min_m=float(dz.min()),bed_max_m=float(dz.max()),
        erosion_bulk_m3=float(np.sum(np.maximum(-dz,0)*area)),deposition_bulk_m3=float(np.sum(np.maximum(dz,0)*area)))
    record['concentration_max_saved']=max(float(store.read(i).arrays['mobile_bedload_concentration'].max()) for i in range(len(store.times)))
    record['target_max_saved']=max(float(np.max(f.arrays['bedload_capacity_m2_s']/np.maximum(np.hypot(f.q[1],f.q[2]),1e-300))) for f in [store.read(i) for i in range(len(store.times))])
    return record

records={};stores={}
for case in ('strong','flume','front'):
    for backend in ('cpu','gpu'):
        name=case+'_'+backend;folder=WORK/name;stores[name]=ResultsStore(folder/'run')
        records[name]=dict(checks=checks(folder),summary=summary(stores[name]))
    records[case+'_backend']=compare(stores[case+'_cpu'],stores[case+'_gpu'])
records['strong_half_dt']=compare(stores['strong_cpu'],ResultsStore(WORK/'strong_half_cpu/run'))
for case,old in [('flume','v3_flume'),('front','front')]:
    for backend in ('cpu','gpu'):
        records[case+'_previous_'+backend]=compare(ResultsStore(ROOT/f'validation/_shallow209/{old}_{backend}/run'),stores[case+'_'+backend])

# Frozen rejected cell, unchanged source equation, independently time integrated.
from tests.test_jiang_cao196 import independent_exchange,host
from braids_core.river_jiang_cao import material
from braids_core.sediment.jiang_cao_exchange import JiangCaoExchange
mat=material(json.loads((WORK/'strong_cpu/project.json').read_text()));B=1-mat.porosity
s=np.load(ROOT/'validation/_shallow209/packing_v2_cpu/FIRST_REJECTION.npz')['state'][:,12:13]
times=np.linspace(0,.01,61);ce=.6062802676380055
cpu=JiangCaoExchange('cpu',mat);gpu=JiangCaoExchange('gpu',mat)
v=np.array([host(cpu.exchange(s,ce,t).state)[:,0] for t in times])
g=np.array([host(gpu.exchange(s,ce,t).state)[:,0] for t in times])
ref=np.array([independent_exchange(s[:,0],ce,t,mat) for t in times])
records['frozen_source']=dict(max_ode_state_error=float(abs(v-ref).max()),max_backend_state_error=float(abs(v-g).max()),
    initial_c=float(s[3,0]/s[0,0]),final_c=float(v[-1,3]/v[-1,0]),target=ce,packing=B)
np.savez_compressed(REPORT/'SOURCE_COMPARISON.npz',time=times,cpu=v,gpu=g,independent=ref,initial=s,target=ce)
fig,ax=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for arr,label,style in [(ref,'Independent ODE','-'),(v,'CPU','--'),(g,'GPU',':')]:
    ax[0,0].plot(times*1000,arr[:,3]/arr[:,0],style,label=label)
    ax[0,1].plot(times*1000,arr[:,0]*1000,style,label=label)
    ax[1,0].plot(times*1000,(arr[:,5]-s[5,0])*1000,style,label=label)
ax[0,0].axhline(B,color='black',ls='--',label='Actual packing bound')
ax[0,0].axhline(ce,color='tab:red',ls=':',label='Uncapped forcing target')
for c0,label in [(np.nextafter(B,0),'ce just below B'),(B,'ce = B'),(np.nextafter(B,1),'ce just above B')]:
    vs=np.array([host(cpu.exchange(s,c0,t).state)[:,0] for t in times])
    ax[1,1].plot(times*1000,vs[:,3]/vs[:,0],label=label)
for a,title,y in zip(ax.flat,['Actual concentration remains admissible','Mixture depth','Conservative bed exchange','Continuous crossing of the former singularity'],['Solid volume fraction','Depth (mm)','Bed change (mm)','Solid volume fraction']):
    a.set(title=title,xlabel='Elapsed source time (ms)',ylabel=y);a.legend(fontsize=8);a.grid(alpha=.2)
fig.suptitle('Frozen rejected inlet cell: native source versus independent E-D integration')
fig.savefig(REPORT/'SOURCE_COMPARISON.png',dpi=165);plt.close(fig)

fig,ax=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
f=stores['strong_cpu'].read(-1);g=stores['strong_gpu'].read(-1)
for i,frame in enumerate([f,g]):
    for j,(vals,label,cm) in enumerate([(frame.q[0]*1000,'Depth (mm)','Blues'),((frame.z-frame.arrays['initial_bed_m'])*1000,'Bed change (mm)','RdBu_r')]):
        if j==0:lo,hi=0,max(f.q[0].max(),g.q[0].max())*1000
        else:hi=max(abs(f.z-f.arrays['initial_bed_m']).max(),abs(g.z-g.arrays['initial_bed_m']).max())*1000;lo=-hi
        p=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],array=vals,cmap=cm,edgecolors='#777777',linewidths=.25);p.set_clim(lo,hi)
        ax[j,i].add_collection(p);ax[j,i].autoscale();ax[j,i].set_aspect('equal');fig.colorbar(p,ax=ax[j,i],label=label,shrink=.7)
        ax[j,i].set_title(('CPU' if i==0 else 'GPU')+' at 0.8 s')
for j,err in enumerate([(g.q[0]-f.q[0])*1000,(g.z-f.z)*1000]):
    hi=max(abs(err).max(),1e-13);p=PolyCollection([np.asarray(p.exterior.coords) for p in f.polygons],array=err,cmap='RdBu_r',edgecolors='#777777',linewidths=.25);p.set_clim(-hi,hi)
    ax[j,2].add_collection(p);ax[j,2].autoscale();ax[j,2].set_aspect('equal');fig.colorbar(p,ax=ax[j,2],label='GPU - CPU (mm)',shrink=.7)
    ax[j,2].set_title('Backend difference, separate scale')
for a in ax.flat:a.set(xlabel='x (m)',ylabel='y (m)')
fig.suptitle('Previously failing dry inlet: raw cells, cropped oblique walls and three regrids')
fig.savefig(REPORT/'STRONG_INLET.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for name,label,style in [('strong_cpu','CPU','-'),('strong_gpu','GPU','--'),('strong_half_cpu','Half dt CPU',':')]:
    store=ResultsStore(WORK/name/'run');t=store.times;ff=[store.read(i) for i in range(len(t))]
    ax[0,0].plot(t,[f.arrays['mobile_bedload_concentration'].max() for f in ff],style,label=label)
    ax[0,1].plot(t,[1000*np.min(f.z-f.arrays['initial_bed_m']) for f in ff],style,label=label)
    ds=json.loads((WORK/name/'run/DIAGNOSTICS.json').read_text())
    for key,color in [('solid_residual_m3','tab:brown'),('water_plus_pore_residual_m3','tab:blue')]:
        ax[1,0].plot([d['time_s'] for d in ds],[d[key] for d in ds],style,color=color,label=label+' '+key.split('_')[0])
ax[0,0].axhline(B,color='black',ls='--',label='Packing bound')
for a in ax[0]:a.axvline(.008,color='grey',ls=':',label='Previous failure time')
for backend,style in [('cpu','-'),('gpu','--')]:
    a=stores['flume_'+backend];old=ResultsStore(ROOT/f'validation/_shallow209/v3_flume_{backend}/run')
    ax[1,1].plot(a.times,[np.max(abs(a.read(i).z-old.read(i).z))*1000 for i in range(len(a.times))],style,label=backend)
for a,title,y in zip(ax.flat,['Bounded actual sediment concentration','Scour development and timestep control','Conservative water and solid budgets','Flume regression versus milestone209'],['Max solid fraction','Minimum bed change (mm)','Phase residual (m³)','Max bed difference (mm)']):
    a.set(title=title,xlabel='Simulation time (s)',ylabel=y);a.legend(fontsize=7);a.grid(alpha=.2)
fig.savefig(REPORT/'TRAJECTORY_AND_REGRESSION.png',dpi=170);plt.close(fig)
(REPORT/'COMPARISON.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2),flush=True)
