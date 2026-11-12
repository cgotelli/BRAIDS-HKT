"""Raw-cell comparisons and independent operator audit of shallow treatment209."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_shallow209'
RUNTIME=WORK/'candidate_v2';REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shallow209'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from braids_desktop.results import ResultsStore
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from braids_core.sediment.portable_river_runtime import install,_load_lib
install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
_dependency=_load_lib(RUNTIME/'precompiled/0201330b952952aedb9a/libdl.dll')
REPORT.mkdir(parents=True,exist_ok=True)

def compare(a,b):
    assert np.allclose(a.times,b.times,atol=1e-12,rtol=0)
    out={}
    for i in range(len(a.times)):
        f,g=a.read(i),b.read(i);np.testing.assert_array_equal(f.mesh['leaves'],g.mesh['leaves'])
        for name,x,y,scale in [('depth_mm',f.q[0],g.q[0],1000),('bed_mm',f.z,g.z,1000),
            ('mobile_mm',f.arrays['mobile_bedload_solid_m'],g.arrays['mobile_bedload_solid_m'],1000),
            ('intensity_m_s',f.arrays['secondary_intensity_m_s'],g.arrays['secondary_intensity_m_s'],1)]:
            e=scale*(y-x);v=out.setdefault(name,dict(maximum=0.,rms=0.))
            v['maximum']=max(v['maximum'],float(abs(e).max()))
            v['rms']=max(v['rms'],float(np.sqrt(np.average(e*e,weights=f.mesh['area']))))
    return out

def coverage(store):
    f=store.read(-1);dz=f.z-f.arrays['initial_bed_m'];area=f.mesh['area']
    return dict(time_s=f.time_s,depth_max_m=float(f.q[0].max()),bed_min_m=float(dz.min()),bed_max_m=float(dz.max()),
        erosion_bulk_m3=float(np.sum(np.maximum(-dz,0)*area)),deposition_bulk_m3=float(np.sum(np.maximum(dz,0)*area)),
        capacity_max_m2_s=float(f.arrays['bedload_capacity_m2_s'].max()),
        capacity_positive_cells=int(np.count_nonzero(f.arrays['bedload_capacity_m2_s'])),
        concentration_max=float(f.arrays['mobile_bedload_concentration'].max()),
        secondary_intensity_max_m_s=float(abs(f.arrays['secondary_intensity_m_s']).max()))

def run_checks(folder):
    status=json.loads((folder/'run/STATUS.json').read_text());diag=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
    r=dict(status=status['status'],time_s=status['time_s'],steps=status['steps'],rejected_trials=len(status['rejections']),
        regrids=sum(e['changed'] for e in status['adaptation_events']),
        budgets={k:max(abs(d[k]) for d in diag) for k in ('solid_residual_m3','water_plus_pore_residual_m3')})
    if (folder/'RESTART.json').exists():r['exact_restart']=all(json.loads((folder/'RESTART.json').read_text())['exact'].values())
    return r

records={};stores={}
for backend in ('cpu','gpu'):
    folder=WORK/('v2_flume_'+backend);stores[backend]=ResultsStore(folder/'run')
    records[backend]=dict(coverage=coverage(stores[backend]),checks=run_checks(folder))
old=ResultsStore(ROOT/'validation/_startup208/flume_cpu/run')
records['previous']=coverage(old);records['backend']=compare(stores['cpu'],stores['gpu'])
for backend in ('cpu','gpu'):
    folder=WORK/('v2_half_flume_'+backend)
    if folder.exists():records['half_dt_'+backend]=compare(stores[backend],ResultsStore(folder/'run'))
records['old_vs_new']=compare(old,stores['cpu'])
for backend,folder in [('cpu','packing_v2_cpu'),('gpu','v2_oblique_gpu_run')]:
    records['oblique_'+backend]=run_checks(WORK/folder)
if records['oblique_cpu']['status']==records['oblique_gpu']['status']=='complete':
    records['oblique_backend']=compare(ResultsStore(WORK/'packing_v2_cpu/run'),ResultsStore(WORK/'v2_oblique_gpu_run/run'))

f=stores['cpu'].read(-1);g=stores['gpu'].read(-1);b=old.read(-1)
x=.025+.05*np.arange(500);sel=x<1.;row=15
fig,ax=plt.subplots(2,2,figsize=(11.5,7.4),layout='constrained')
for frame,name,color in [(b,'Previous shared 10D gate','grey'),(f,'Separate entrainment CPU','tab:blue'),(g,'Separate entrainment GPU','tab:orange')]:
    style='--' if name.endswith('GPU') else '-'
    ax[0,0].plot(x[sel],frame.q[0].reshape(30,500)[row,sel]*1000,style,color=color,label=name)
    ax[0,1].plot(x[sel],(frame.z-frame.arrays['initial_bed_m']).reshape(30,500)[row,sel]*1000,style,color=color,label=name)
    ax[1,0].plot(x[sel],frame.arrays['bedload_capacity_m2_s'].reshape(30,500)[row,sel]*1e6,style,color=color,label=name)
for j in range(1,len(stores['cpu'].times)):
    frame=stores['cpu'].read(j)
    ax[1,1].plot(x[sel],(frame.z-frame.arrays['initial_bed_m']).reshape(30,500)[row,sel]*1000,label=f'{frame.time_s:g} s')
for a,title,ylabel in zip(ax.flat,['Water depth at 2 s','Bed change at 2 s','Entrainment capacity at 2 s','CPU bed development'],
    ['Depth (mm)','Bed change (mm)','Capacity (10^-6 m²/s)','Bed change (mm)']):
    a.set(title=title,xlabel='x (m)',ylabel=ylabel);a.axhline(0,color='black',lw=.5);a.legend(fontsize=8)
fig.suptitle('Reconstructed flume startup: erosion is no longer disabled by secondary-profile validity\nCentreline, first metre of 25 m; transient response, not braiding validation')
fig.savefig(REPORT/'FLUME_COMPARISON.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(2,3,figsize=(12,6.2),layout='constrained')
for i,frame in enumerate((b,f,g)):
    dz=(frame.z-frame.arrays['initial_bed_m']).reshape(30,500)*1000
    im=ax[0,i].pcolormesh(np.arange(21)*.05,np.arange(31)*.05,dz[:,:20],cmap='RdBu_r',vmin=-.25,vmax=.25,shading='flat')
    ax[0,i].set_title(['Previous shared gate','New CPU','New GPU'][i]);ax[0,i].set_aspect('equal')
    fig.colorbar(im,ax=ax[0,i],label='Bed change (mm)',shrink=.7)
err=(g.z-f.z).reshape(30,500)*1000;scale=max(abs(err).max(),1e-13)
im=ax[1,0].pcolormesh(np.arange(21)*.05,np.arange(31)*.05,err[:,:20],cmap='RdBu_r',vmin=-scale,vmax=scale,shading='flat')
ax[1,0].set_title('GPU - CPU (separate error scale)');ax[1,0].set_aspect('equal');fig.colorbar(im,ax=ax[1,0],label='Difference (mm)',shrink=.7)
for backend,style in [('cpu','-'),('gpu','--')]:
    d=json.loads((WORK/('v2_flume_'+backend)/'run/DIAGNOSTICS.json').read_text())
    for k,color in [('solid_residual_m3','tab:brown'),('water_plus_pore_residual_m3','tab:blue')]:
        ax[1,1].plot([r['time_s'] for r in d],[r[k] for r in d],style,color=color,label=backend+' '+k.split('_')[0])
    flow=stores[backend].boundary_flow('inlet',magnitude=False)
    solid=stores[backend].boundary_flow('inlet',quantity='bedload',magnitude=False)
    records[backend]['water_feed_error_m3_s']=float(abs(flow.values-.0025).max())
    records[backend]['solid_feed_error_kg_s']=float(abs(solid.values-.00286).max())
ax[1,1].set(title='Conservative exchange and transport',xlabel='Time (s)',ylabel='Phase residual (m³)');ax[1,1].legend(fontsize=7)
ax[1,2].axis('off');ax[1,2].text(.02,.95,'Same DEM, 5 cm cells, initial state and feed\n\nWater: 0.0025 m³/s\nSolid: 0.00286 kg/s\nD50 = D90 = 1 mm; MF = 1\n\nSecondary profile remains inactive here.\nNo claim of bars, braids or equilibrium.',va='top',fontsize=10)
for a in list(ax[0])+[ax[1,0]]:a.set(xlabel='x (m)',ylabel='y (m)')
fig.suptitle('Raw cell bed change: no smoothing, no added initial perturbation')
fig.savefig(REPORT/'RAW_CELLS_AND_BUDGETS.png',dpi=170);plt.close(fig)

# Independent uniform-slope constitutive audit; actual depth is never floored.
from braids_core.river_jiang_cao import material
from braids_core.sediment.jiang_cao_capacity import JiangCaoCapacity
from braids_core.sediment.jiang_cao_roughness import RoughnessParameters,RoughnessOperator
from braids_core.sediment.jiang_cao_wetdry import ShallowTreatment,GrainScaleBedloadTreatment
doc=json.loads((WORK/'v2_flume_cpu/flume.braids.json').read_text());mat=material(doc)
h=np.linspace(.00005,.022,440);speed=.3;q=np.vstack((h,speed*h,0*h));bedgrad=np.vstack((0*h-.01,0*h));curves={}
for backend in ('cpu','gpu'):
    op=RoughnessOperator(backend,RoughnessParameters(mat,.001));xp=op.xp;host=lambda a:np.asarray(a) if backend=='cpu' else xp.asnumpy(a)
    qx=xp.asarray(q);n=GrainScaleBedloadTreatment().resistance(op,qx)
    cap=JiangCaoCapacity(backend,mat.capacity_material).evaluate(qx,xp.asarray(bedgrad),n).values
    oldw,_=ShallowTreatment().weight(xp.asarray(h),.001,xp);w,_=GrainScaleBedloadTreatment().bedload_weight(xp.asarray(h),.001,1e-6,xp)
    c=host(cap);nn=host(n);theta=nn**2*speed**2/(h**(1/3)*1.65*.001)
    critical=.047*(1-.01/np.tan(np.pi/6))/np.sqrt(1+.01**2)
    expected=8*np.sqrt(1.65*9.81*.001**3)*np.maximum(theta-critical,0)**1.5
    np.testing.assert_allclose(c[0],expected,atol=1e-17,rtol=3e-13)
    curves[backend]=dict(n=nn,theta=c[3],critical=c[4],raw=c[0],new=c[0]*host(w),old=c[0]*host(oldw),secondary_weight=host(oldw))
for k in curves['cpu']:np.testing.assert_allclose(curves['cpu'][k],curves['gpu'][k],rtol=3e-13,atol=1e-17)
r=curves['cpu'];newce=r['new']/(h*speed);oldce=r['old']/(h*speed)
fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
ax[0,0].plot(h*1000,r['n']);ax[0,0].set(title='Unchanged resistance continuation',ylabel='Manning n')
ax[0,1].plot(h*1000,r['theta'],label='Actual-depth Shields');ax[0,1].plot(h*1000,r['critical'],label='Slope-modified threshold');ax[0,1].legend(fontsize=8);ax[0,1].set(title='Capacity shear and threshold',ylabel='Shields number')
for key,name in [('raw','Unweighted law above D'),('old','Previous 10D gate'),('new','Submerged-grain treatment')]:
    value=np.where(h>=.001,r[key],np.nan) if key=='raw' else r[key]
    ax[0,2].plot(h*1000,value*1e6,label=name)
ax[0,2].set(title='Capacity magnitude',ylabel='Capacity (10^-6 m²/s)');ax[0,2].legend(fontsize=8)
ax[1,0].plot(h*1000,newce,label='New');ax[1,0].plot(h*1000,oldce,label='Previous');ax[1,0].set(title='Equilibrium mobile concentration',ylabel='Solid volume fraction');ax[1,0].legend(fontsize=8)
for c0,label in [(0.,'Initially clear water'),(.005,'Feed concentration = 0.005')]:ax[1,1].plot(h*1000,2*mat.settling_velocity*(newce-c0)*1000,label=label)
ax[1,1].axhline(0,color='grey',lw=.5);ax[1,1].set(title='Frozen-state E minus D',ylabel='Net entrainment (mm solid/s)');ax[1,1].legend(fontsize=8)
ax[1,2].plot(h*1000,GrainScaleBedloadTreatment().bedload_weight(h,.001,1e-6,np)[0],label='Bedload entrainment')
ax[1,2].plot(h*1000,r['secondary_weight'],label='Secondary covariance/drift');ax[1,2].set(title='Independent process activation',ylabel='Weight');ax[1,2].legend(fontsize=8)
for a in ax.flat:a.set_xlabel('Physical depth (mm)');a.set_xlim(0,22)
fig.suptitle('Uniform-slope operator audit | slope 1%, U = 0.3 m/s, D = 1 mm\nNo fitted coefficient; depth matching, capacity and secondary validity traced separately')
fig.savefig(REPORT/'OPERATOR_AUDIT.png',dpi=170);plt.close(fig)
np.savez(REPORT/'OPERATOR_CURVES.npz',depth=h,**r)
records['operator']=dict(speed_m_s=speed,slope=.01,maximum_backend_difference={k:float(abs(curves['cpu'][k]-curves['gpu'][k]).max()) for k in r},max_ce=float(newce.max()))
trace=json.loads((WORK/'packing_cpu/TRACE.json').read_text());records['rejected_subgrain_trial']=dict(first=trace[0],last=trace[-1],rejections=len(trace))
trace=json.loads((WORK/'packing_v2_cpu/TRACE.json').read_text());records['submerged_grain_stress_limit']=dict(first=trace[0],last=trace[-1],rejections=len(trace))
# Independent ODE only: no production target guard or native source is bypassed.
from tests.test_jiang_cao196 import independent_exchange
with np.load(WORK/'packing_v2_cpu/LAST_REJECTION.npz') as archive:
    initial=archive['state'][:6,trace[-1]['cell']].copy()
ce=trace[-1]['ce'];times=np.linspace(0,.01,51)
trajectory=np.stack([independent_exchange(initial,ce,float(t),material=mat) for t in times])
actual=trajectory[:,3]/trajectory[:,0];B=1-mat.porosity
solid=trajectory[:,3]+trajectory[:,4]
water=trajectory[:,0]-trajectory[:,3]+mat.porosity*trajectory[:,4]/B
assert (actual<B).all() and (trajectory[:,4]>=-1e-12).all()
np.savez(REPORT/'SOURCE_DIAGNOSTIC.npz',time_s=times,state=trajectory,capacity_target=ce)
records['source_diagnostic']=dict(interpretation='Independent integration of unchanged E-D equations only; not adopted by CPU/GPU kernel',
    target=ce,actual_initial=float(actual[0]),actual_final=float(actual[-1]),duration_s=float(times[-1]),
    solid_error=float(abs(solid-solid[0]).max()),water_error=float(abs(water-water[0]).max()),
    invariant_W_m=float(initial[0]-initial[3]/B),
    reason='Existing exact source assumes finite H=B*W/(B-ce); ce>=B violates that analytic branch, not necessarily actual c<B')
fig,ax=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
status=json.loads((WORK/'packing_v2_cpu/run/STATUS.json').read_text())
ax[0].semilogx([r['dt_s'] for r in status['rejections']],[r['ce'] for r in trace],'o-',label='Rejected capacity target')
ax[0].axhline(.6,color='black',ls=':',label='Existing target limit');ax[0].set(xlabel='Trial timestep (s)',ylabel='Solid volume ratio',title='Identical input rejected before exchange');ax[0].legend(fontsize=8)
ax[1].plot(times*1000,actual,label='Actual c from independent E-D integration')
ax[1].axhline(ce,color='tab:orange',ls='--',label='Frozen capacity target');ax[1].axhline(.6,color='black',ls=':',label='Packing bound on actual c')
ax[1].set(xlabel='Source integration time (ms)',ylabel='Solid volume ratio',title='Valid transient without a finite equilibrium');ax[1].legend(fontsize=8)
fig.suptitle('Remaining dry-inlet failure: target restriction in the exact exchange branch\nIndependent ODE diagnostic, not an admitted coupled correction')
fig.savefig(REPORT/'TARGET_LIMIT.png',dpi=170);plt.close(fig)
for backend in ('cpu','gpu'):
    records['wet_oblique_'+backend]=run_checks(WORK/('v3_wet_oblique_'+backend))
if records['wet_oblique_cpu']['status']==records['wet_oblique_gpu']['status']=='complete':
    records['wet_oblique_backend']=compare(ResultsStore(WORK/'v3_wet_oblique_cpu/run'),ResultsStore(WORK/'v3_wet_oblique_gpu/run'))
records['final_candidate_flume']={}
for backend in ('cpu','gpu'):
    current=ResultsStore(WORK/('v3_flume_'+backend)/'run');previous=stores[backend]
    exact=True
    for i in range(len(current.times)):
        x,y=current.read(i),previous.read(i)
        exact &= all(np.array_equal(x.arrays[k],y.arrays[k]) for k in x.arrays)
    assert exact
    records['final_candidate_flume'][backend]=dict(all_saved_arrays_identical_to_v2=exact,
        checks=run_checks(WORK/('v3_flume_'+backend)))
front={b:ResultsStore(WORK/('front_'+b)/'run') for b in ('cpu','gpu')}
records['front_backend']=compare(front['cpu'],front['gpu'])
records['front']={b:run_checks(WORK/('front_'+b)) for b in front}
assert all(r['regrids']>0 and r['status']=='complete' for r in records['front'].values())
f,g=front['cpu'].read(-1),front['gpu'].read(-1)
fig,ax=plt.subplots(2,3,figsize=(11.5,6.6),layout='constrained')
for row,(fc,gc,title) in enumerate([(f.q[0]*1000,g.q[0]*1000,'depth'),
    ((f.z-f.arrays['initial_bed_m'])*1000,(g.z-g.arrays['initial_bed_m'])*1000,'bed change')]):
    limit=max(abs(fc).max(),abs(gc).max(),1e-13)
    for j,(frame,values,name) in enumerate([(f,fc,'CPU'),(g,gc,'GPU'),(g,gc-fc,'GPU minus CPU')]):
        lim=max(abs(values).max(),1e-13) if j==2 else limit
        bounds=(-lim,lim) if row or j==2 else (0,lim)
        art=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],array=values,
            norm=Normalize(*bounds),cmap='RdBu_r' if row or j==2 else 'Blues',edgecolor='grey',linewidth=.25)
        ax[row,j].add_collection(art);ax[row,j].autoscale_view();ax[row,j].set_aspect('equal');ax[row,j].set_xlim(0,1)
        ax[row,j].set(title=name+' '+title,xlabel='x (m)',ylabel='y (m)');fig.colorbar(art,ax=ax[row,j],label='mm',shrink=.7)
fig.suptitle('Low-discharge dry front with cropped oblique walls | 0.8 s, five real regrids\nSame specific discharge and slope as flume; distinct strong-inlet failure retained')
fig.savefig(REPORT/'CROPPED_FRONT.png',dpi=170);plt.close(fig)
(REPORT/'COMPARISON.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
