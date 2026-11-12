"""Matched raw-cell diagnostics; no smoothing or changed simulated physics."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'validation/_startup208'
RUNTIME=WORK/'candidate';sys.path[:0]=[str(RUNTIME),str(ROOT)]
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
import numpy as np
from osgeo import gdal
# Standalone operator diagnostic does not construct the hydraulic owner first.
# Preload its already authenticated libdl dependency from the same runtime,
# rather than relying on an installed compiler's DLL search path.
from braids_core.sediment.portable_river_runtime import _load_lib
_dependency=_load_lib(RUNTIME/'precompiled/0201330b952952aedb9a/libdl.dll')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from braids_desktop.results import ResultsStore
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_startup208'
REPORT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(name,value):(REPORT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def xy(frame):return np.array([[p.centroid.x,p.centroid.y] for p in frame.polygons]).T
def compare(a,b):
    np.testing.assert_allclose(a.times,b.times,atol=2e-15,rtol=0)
    maximum={k:0. for k in ('depth_mm','bed_mm','mobile_mm','intensity_m_s')};rms=dict(maximum)
    for i in range(len(a.times)):
        f,g=a.read(i),b.read(i);np.testing.assert_array_equal(f.mesh['leaves'],g.mesh['leaves'])
        for k,x,y,scale in [('depth_mm',f.q[0],g.q[0],1000),('bed_mm',f.z,g.z,1000),
            ('mobile_mm',f.arrays['mobile_bedload_solid_m'],g.arrays['mobile_bedload_solid_m'],1000),
            ('intensity_m_s',f.arrays['secondary_intensity_m_s'],g.arrays['secondary_intensity_m_s'],1)]:
            e=(y-x)*scale;maximum[k]=max(maximum[k],float(abs(e).max()))
            rms[k]=max(rms[k],float(np.sqrt(np.average(e*e,weights=f.mesh['area']))))
    return dict(maximum=maximum,maximum_frame_area_weighted_rms=rms)
def paint(ax,f,v,title,cmap,limits,zoom=None):
    artist=PolyCollection([np.array(p.exterior.coords) for p in f.polygons],array=v,cmap=cmap,
        norm=Normalize(*limits),edgecolors='#6d7279',linewidths=.15)
    ax.add_collection(artist);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    if zoom:ax.set_xlim(*zoom)
    return artist
records={};stores={}
for key,prefix in [('oblique','packaged_oblique_'),('flume','flume_')]:
    stores[key]={b:ResultsStore(WORK/(prefix+b)/'run') for b in ('cpu','gpu')}
    cpu,gpu=stores[key].values();record=compare(cpu,gpu);record['backends']={}
    assert max(record['maximum'][k] for k in ('depth_mm','bed_mm'))<.1
    for backend in ('cpu','gpu'):
        folder=WORK/(prefix+backend);diag=json.loads((folder/'run/DIAGNOSTICS.json').read_text())
        restart=json.loads((folder/'RESTART.json').read_text());assert all(restart['exact'].values())
        record['backends'][backend]=dict(status=json.loads((folder/'run/STATUS.json').read_text()),
            budget={k:max(abs(r[k]) for r in diag) for k in ('solid_residual_m3','water_plus_pore_residual_m3')},restart_exact=True)
        assert max(record['backends'][backend]['budget'].values())<1e-10
    records[key]=record
    f,g=(s.read(-1) for s in stores[key].values());fig,axes=plt.subplots(2,3,figsize=(12.5,7.3),layout='constrained')
    for row,(v,u,label) in enumerate([(f.q[0]*1000,g.q[0]*1000,'depth (mm)'),
                  ((f.z-f.arrays['initial_bed_m'])*1000,(g.z-g.arrays['initial_bed_m'])*1000,'bed change (mm)')]):
        lim=(0,max(float(v.max()),float(u.max()),1e-12)) if row==0 else (-max(abs(v).max(),abs(u).max(),1e-12),max(abs(v).max(),abs(u).max(),1e-12))
        zoom=(0,1) if key=='flume' else None
        for j,(ff,vv,label2) in enumerate([(f,v,'CPU'),(g,u,'GPU')]):
            art=paint(axes[row,j],ff,vv,label2+' '+label,'Blues' if row==0 else 'RdBu_r',lim,zoom)
            fig.colorbar(art,ax=axes[row,j],shrink=.7)
        diff=u-v;lim=max(abs(diff).max(),1e-13)
        art=paint(axes[row,2],f,diff,'GPU - CPU (mm)','RdBu_r',(-lim,lim),zoom)
        fig.colorbar(art,ax=axes[row,2],shrink=.7)
    fig.suptitle(f'{key.capitalize()} startup, t = {f.time_s:g} s | raw cells, separate error scales'+
                 ('\nFirst metre of the 25 m flume; transient deposition, not a braided pattern' if key=='flume' else '\nDry start with real AMR; guard unchanged'))
    fig.savefig(REPORT/(key.upper()+'_COMPARISON.png'),dpi=160);plt.close(fig)

half=ResultsStore(WORK/'half_oblique_cpu/run');records['oblique']['half_dt']=compare(stores['oblique']['cpu'],half)
assert records['oblique']['half_dt']['maximum']['depth_mm']<1.
assert records['oblique']['half_dt']['maximum']['bed_mm']<1.
records['oblique']['half_dt_gpu']=compare(stores['oblique']['gpu'],ResultsStore(WORK/'half_oblique_gpu/run'))
records['finest_oblique']=compare(ResultsStore(WORK/'finest_oblique_cpu/run'),ResultsStore(WORK/'finest_oblique_gpu/run'))

# Independent NumPy normal equations on the identical first rejected state.
with np.load(WORK/'trace_cpu/FAILED_STATE.npz') as f:
    st=f['state'];coords=f['xy'];i=16;o=f['owner'];j=f['neighbour']
    nb=np.unique(np.r_[j[o==i],o[j==i]]);nb=nb[nb>=0];vel=st[1:3]/np.where(st[0]>1e-6,st[0],1)
    stencil={}
    for name,mask in [('profile_mask',st[0]>.01),('hydraulic_wet_mask',st[0]>1e-6)]:
        nodes=nb[mask[nb]];D=(coords[:,nodes]-coords[:,i,None]).T;w=1/np.sum(D*D,axis=1)
        A=D.T@(w[:,None]*D);B=D.T@(w[:,None]*(vel[:,nodes]-vel[:,i,None]).T);G=np.linalg.solve(A,B).T
        t=vel[:,i]/np.linalg.norm(vel[:,i]);k=(t[0]**2*G[1,0]+t[0]*t[1]*(G[1,1]-G[0,0])-t[1]**2*G[0,1])/np.linalg.norm(vel[:,i])
        stencil[name]=dict(neighbours=nodes.tolist(),condition_number=float(np.linalg.cond(A)),gradient=G.tolist(),curvature_m_inverse=float(k))
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for node in [i,*nb]:
    ax[0].scatter(*coords[:,node],s=80,c='tab:orange' if st[0,node]<.01 else 'tab:blue')
    ax[0].annotate(f'{node}: h={st[0,node]*1000:.2f} mm',coords[:,node],xytext=(5,6),textcoords='offset points',fontsize=8)
    if node!=i:ax[0].plot(coords[0,[i,node]],coords[1,[i,node]],color='grey',lw=1)
ax[0].set(xlabel='x (m)',ylabel='y (m)',title='Shallow wet neighbour 17 was excluded',xlim=(.04,.53),ylim=(.14,.79))
ax[1].bar(['Old depth mask','Physical wet mask'],[stencil[k]['gradient'][1][0] for k in stencil],color=['tab:red','tab:blue'])
ax[1].set(ylabel='dv/dx (1/s)',title='Same-state fit: false gradient removed');ax[1].axhline(0,color='grey',lw=.6)
trace=json.loads((WORK/'trace_cpu/TRACE.json').read_text());old=[r for r in trace if r['phase']=='source_after']
ax[2].plot([r['t'] for r in old],[abs(r['hk']) for r in old],label='Old near-inlet I/U',color='tab:red')
store=stores['oblique']['cpu'];values=[]
for t in range(len(store.times)):
    f=store.read(t);pos=xy(f);cell=np.argmin((pos[0]-.1)**2+(pos[1]-.5)**2)
    U=np.linalg.norm(f.q[1:,cell])/f.q[0,cell] if f.q[0,cell]>1e-6 else 0
    values.append(abs(f.arrays['secondary_intensity_m_s'][cell]/U) if U>0 else 0)
ax[2].plot(store.times,values,'o-',label='Corrected near-inlet I/U',color='tab:blue')
ax[2].axhline(.1,color='black',ls=':',label='Unchanged envelope');ax[2].set(xlabel='Time (s)',ylabel='Effective |I/U|',title='No clipping or guard relaxation');ax[2].legend(fontsize=8)
fig.suptitle('Inlet diagnosis: a hydraulic stencil was incorrectly filtered by constitutive depth')
fig.savefig(REPORT/'INLET_DIAGNOSIS.png',dpi=170);plt.close(fig)
records['inlet_stencil']=stencil

# Same-state capacity diagnostic only: do NOT run unqualified shallow physics.
from braids_core.river_jiang_cao import material
from braids_core.sediment.jiang_cao_capacity import JiangCaoCapacity
from braids_core.sediment.jiang_cao_roughness import RoughnessParameters,RoughnessOperator
from braids_core.sediment.jiang_cao_wetdry import ShallowTreatment
doc=json.loads((WORK/'flume_cpu/flume.braids.json').read_text());mat=material(doc)
f=stores['flume']['cpu'].read(-1);pos=xy(f);h=f.q[0];wet=h>1e-6
assert len(h)==15000 and np.allclose(pos[0].reshape(30,500)[0],.025+.05*np.arange(500))
gy,gx=np.gradient(f.z.reshape(30,500),.05,.05);grad=np.vstack((gx.ravel(),gy.ravel()))
rough=RoughnessOperator('cpu',RoughnessParameters(mat,.001));n=ShallowTreatment().resistance(rough,f.q)
raw=JiangCaoCapacity('cpu',mat.capacity_material).evaluate(f.q,grad,n,concentration=f.arrays['mobile_bedload_concentration'])
unweighted=raw.values[0]
sub=h>.002
actual=rough.evaluate(f.q[:,sub]);a=np.sqrt(9.81)*actual[9]/(.4*h[sub]**(1/6))
rec=dict(cells=len(h),wet_cells=int(wet.sum()),depth_max_m=float(h.max()),active_entrainment_cells=int(np.sum(h>.01)),
    full_activation_cells=int(np.sum(h>.02)),capacity_max_m2_s=float(f.arrays['bedload_capacity_m2_s'].max()),
    same_state_unweighted_capacity_max_m2_s=float(unweighted.max()),diagnostic_only_not_advanced=True,
    bed_change_min_m=float((f.z-f.arrays['initial_bed_m']).min()),bed_change_max_m=float((f.z-f.arrays['initial_bed_m']).max()),
    raw_log_profile_diagnostic_depth_min_m=.002,raw_log_profile_a_min=float(a.min()),raw_log_profile_a_max=float(a.max()),
    profile_a_below_half_count=int(np.sum(a<.5)),raw_log_profile_sample_count=len(a))
records['flume']['physical_coverage']=rec
store=stores['flume']['cpu'];w=store.boundary_flow('inlet',magnitude=False);s=store.boundary_flow('inlet',quantity='bedload',magnitude=False)
np.testing.assert_allclose(w.values,.0025,rtol=1e-9,atol=1e-13)
np.testing.assert_allclose(s.values,.00286,rtol=1e-9,atol=1e-13)
records['flume']['inlet_water_max_error_m3_s']=float(np.max(abs(w.values-.0025)))
records['flume']['inlet_solid_max_error_kg_s']=float(np.max(abs(s.values-.00286)))
row=np.arange(15*500,16*500);sel=row[pos[0,row]<1.]
fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for index in range(1,len(store.times)):
    frame=store.read(index);ax[0,0].plot(pos[0,sel],frame.q[0,sel]*1000,label=f'{frame.time_s:g} s')
ax[0,0].axhline(10,color='black',ls=':',label='Current entrainment activation');ax[0,0].set(xlabel='x (m)',ylabel='Depth (mm)',title='Early wetting front stays below activation');ax[0,0].legend(fontsize=8)
ax[0,1].plot(pos[0,sel],unweighted[sel]*1e6,label='Unweighted same-state law (diagnostic only)')
ax[0,1].plot(pos[0,sel],f.arrays['bedload_capacity_m2_s'][sel]*1e6,label='Actual simulation: shallow weight applied')
ax[0,1].set(xlabel='x (m)',ylabel='Capacity (10^-6 m²/s)',title='Zero erosion is imposed by the extension');ax[0,1].legend(fontsize=8)
ax[1,0].plot(pos[0,sel],(f.z-f.arrays['initial_bed_m'])[sel]*1000);ax[1,0].set(xlabel='x (m)',ylabel='Bed change (mm)',title='Deposited external feed; no bar claim');ax[1,0].axhline(0,color='grey',lw=.5)
for backend,style in [('cpu','-'),('gpu','--')]:
    diag=json.loads((WORK/('flume_'+backend)/'run/DIAGNOSTICS.json').read_text())
    for key,color in [('solid_residual_m3','tab:brown'),('water_plus_pore_residual_m3','tab:blue')]:
        ax[1,1].plot([r['time_s'] for r in diag],[r[key] for r in diag],style,color=color,label=backend+' '+key.split('_')[0])
ax[1,1].set(xlabel='Time (s)',ylabel='Budget residual (m³)',title='Conservation, not physical validation');ax[1,1].legend(fontsize=8)
fig.suptitle('Reconstructed paper flume: why a long braiding run is premature')
fig.savefig(REPORT/'FLUME_ACTIVATION.png',dpi=160);plt.close(fig)
save('COMPARISON.json',dict(runtime=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest(),cases=records))
print(json.dumps({k:v for k,v in rec.items()},indent=2));print(json.dumps({k:v['maximum'] for k,v in records.items() if 'maximum' in v},indent=2))
