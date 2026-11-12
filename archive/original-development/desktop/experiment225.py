"""Current-format, documented Jiang-Cao Case 1 reconstruction and CPU trials."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,hashlib,json,os,shutil,sys,time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent/'BRAIDS_JiangCao_Experiment225'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_experiment225'
PYTHON=ROOT/'desktop/.venv/Scripts/python.exe'
IDENTITY='9caa8c7cbea26bb8b98efe4efd37923985845d731e75d213e31a3a6ec498b054'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')

def imports(runtime):
    sys.path[:0]=[str(runtime),str(ROOT)]
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)

def create():
    assert not PROJECT.exists(), 'Preserve existing projects'
    receipt=ROOT/'desktop/cleanup225_receipt.json'
    if not receipt.exists():receipt=Path('D:/BRAIDS_archive/20261030_cleanup225/CLEANUP.json')
    cleanup=json.loads(receipt.read_text(encoding='utf-8-sig'))
    assert len(cleanup['records'])>=4 and all(r['all_file_sha256_equal'] for r in cleanup['records'])
    assert shutil.disk_usage(ROOT).free>=60*2**30, 'Reclaim space before creating the case'
    runtime=ROOT/'desktop/runtime';assert sha(runtime/'RUNTIME.json')==IDENTITY
    imports(runtime)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from osgeo import gdal,osr
    from PySide6.QtWidgets import QApplication
    from braids_desktop import services
    from braids_desktop.forms import SettingsPanel
    from braids_core.river_model import new_model,new_bedload_selection,resolve_model
    from braids_core.river_jiang_cao import METHOD,DEFAULTS,COLLAPSE,ROUGHNESS
    from braids_core.river_project import validate_project
    from braids_core.river_open_boundary import new_state
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
    from braids_core.sediment.river_column_methods import method_ids
    app=QApplication([])
    source=REPORT/'inputs';source.mkdir(parents=True,exist_ok=True)
    dx=.025;nx=1040;ny=100;x=-.5+(np.arange(nx)+.5)*dx;y=2.-(np.arange(ny)+.5)*dx
    noise=gaussian_filter(np.random.default_rng(225).normal(size=(ny,nx)),sigma=4.,mode='reflect')
    inside=(x[None,:]>=0)&(x[None,:]<=25)&(y[:,None]>=0)&(y[:,None]<=1.5)
    taper=np.clip(np.minimum(x,25-x)/.5,0.,1.)[None,:]
    noise=(noise-noise[inside].mean())*taper
    noise-=taper*(noise[inside].sum()/np.broadcast_to(taper,noise.shape)[inside].sum())
    noise*=.0001/np.max(np.abs(noise[inside]))
    z=np.broadcast_to(.01*(25-x),noise.shape).copy()+noise
    dem=source/'flume_perturbed.tif'
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),nx,ny,1,gdal.GDT_Float64,options=['COMPRESS=LZW'])
    crs=osr.SpatialReference();crs.ImportFromEPSG(2056)
    ds.SetGeoTransform((-.5,dx,0,2.,0,-dx));ds.SetProjection(crs.ExportToWkt());ds.GetRasterBand(1).WriteArray(z);ds=None
    d=services.default_document(str(dem),[(0,0),(25,0),(25,1.5),(0,1.5)],'EPSG:2056',backend='cpu',
        minimum_size_m=.05,levels=1,mesh_mode='static_terrain',end_time_s=18000.)
    qw=.0025;feed=.00286;qs=feed/2650.;ratio=qs/qw
    dom=PolygonDomain(**d['settings']['domain'])
    for xx,name,kind,control in ((0,'inlet','inlet',dict(method='inlet.normal_discharge',discharge_m3_s=qw+qs,energy_slope='mesh')),
            (25,'outlet','outlet',dict(method='outlet.critical'))):
        edge=next(e.edge_id for e in dom.edges if e.start_xy[0]==e.end_xy[0]==xx)
        d=services.assign_boundary(d,[edge],name,kind,control)
    m=new_model(backend='cpu');m['hydrodynamics'].update(secondary_flow='secondary.prognostic',roughness=ROUGHNESS)
    s=new_bedload_selection();s['capacity']=dict(method=METHOD,parameters=dict(DEFAULTS),grain_sizes_m=dict(d50=.001,d90=.001))
    s['grains'][0]['diameter_m']=.001;s['avalanche']=COLLAPSE;m['sediment']=dict(enabled=True,selection=s)
    d['model']=resolve_model(m).document()
    columns=replace(ColumnMethodsV101(**d['settings']['columns']),fraction_ids=('sand',),diameter_m=(.001,),bed='mixed',base_update=1)
    d['settings']['columns']=asdict(columns);d['column_methods']=method_ids(columns)
    d['settings']['controls'].update(diameter=.001,threads=4,morfac=1.,betac=0.,diffusivity=0.,viscosity=0.)
    d['settings']['adaptation'].update(domain_border_enabled=False,interval=1.)
    d['settings'].update(dt_max_s=.005,output_interval_s=60.,checkpoint_interval_s=300.)
    d['initial'].update(stage_m=0.,velocity_m_s=[0.,0.],layer_thickness_m=[.2],layer_fractions=[[1.]])
    for name in ('inlet','outlet'):
        d['boundary_states'][name]=new_state(d['settings']['hydraulics'][name]['method'],sediment=True)
    d['boundary_states']['inlet']['composition']=dict(method='composition.reference_volume_fraction',fractions=[1.])
    d['settings']['sediment_boundaries']['inlet']=dict(kind='inlet_solid_water_ratio',solid_water_ratio=[ratio],concentration_kg_m3=[0.])
    d=services._resolve_boundary_state(d)
    panel=SettingsPanel();panel.load_document(d);d=panel.document(d);panel.close()
    d['settings']['checkpoint_interval_s']=300.
    d['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=300.)
    d['name']='Jiang-Cao Case 1 Run 1 — MF1, 0.1 mm seeded perturbation'
    d=validate_project(d);services.save_project_folder(PROJECT,d,source_base=source)
    for n in ('logs','benchmarks','references'): (PROJECT/n).mkdir(exist_ok=True)
    for n in ('chino.pdf','sup.pdf','jian.pdf'):shutil.copy2(ROOT/n,PROJECT/'references'/n)
    manifest=json.loads((runtime/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():
        assert sha(runtime/n)==h,n
        dst=PROJECT/'runtime'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(runtime/n,dst)
    shutil.copy2(runtime/'RUNTIME.json',PROJECT/'runtime/RUNTIME.json')
    for n in ('launch.py','worker.py'):shutil.copy2(ROOT/'desktop'/n,PROJECT/n)
    audit=dict(target='Jiang-Cao Case 1 Run 1 / Garcia Lugo Run 27; reconstructed settings, not an author deck',
        runtime=IDENTITY,paper_sha256={n:sha(ROOT/n) for n in ('chino.pdf','sup.pdf','jian.pdf')},
        published=dict(length_m=25.,width_m=1.5,slope=.01,water_discharge_m3_s=qw,sediment_feed_kg_s=feed,
            uniform_grain_m=.001,nominal_mesh_m=.05,duration_s=18000.),
        assumptions=dict(initial_water='dry',initial_mobile_bedload=0.,initial_secondary=0.,
            grain_descriptors='D50=D90=1 mm explicitly selected for the paper uniform idealization; not a measured grading curve or inferred percentile',
            mesh='fixed 5 cm Cartesian squares; paper uses triangles of 5 cm characteristic size',
            inlet='full-width prescribed mixture Q with normal mesh-slope reference',outlet='critical export, free exterior bed',
            finite_erodible_depth_m=.2,porosity=.4,solid_density_kg_m3=2650.,water_density_kg_m3=1000.,
            molecular_viscosity_m2_s=1e-6,repose_angle_deg=30.,gravity_m_s2=9.81,paper_gravity_m_s2=9.8,
            perturbation=dict(approved_by_user=True,seed=225,maximum_abs_m=.0001,rms_m=float(np.sqrt(np.mean(noise[inside]**2))),
                filter_sigma_m=.1,end_taper_m=.5,mean_m=float(noise[inside].mean())),
            coordinates='Synthetic metre coordinates; EPSG2056 is metadata only, not a surveyed site',
            model_extensions=['augmented f-wave, not paper HLLC','named shallow-grain treatment','finite-adjustment secondary flow']),
        conversion=dict(solid_volume_m3_s=qs,solid_water_ratio=ratio,mixture_discharge_m3_s=qw+qs),
        clocks=dict(MF=1.,end_s=18000.,output_s=60.,checkpoint_s=300.,maximum_dt_s=.005),
        future='Qualify MF>1 phase/bed/water acceleration separately after this baseline; do not change the running model.')
    save(PROJECT/'INPUT_AUDIT.json',audit);save(REPORT/'INPUT_AUDIT.json',audit)
    (PROJECT/'Open experiment.cmd').write_text('@echo off\nstart "" "'+str(PYTHON.with_name('pythonw.exe'))+'" -B "%~dp0launch.py" "%~dp0."\n',encoding='utf8')
    print('Created current project and frozen runtime:',PROJECT,flush=True)

def trial(threads,label,end,backend='cpu',cfl=None,dt_max=None):
    runtime=PROJECT/'runtime';imports(runtime)
    from braids_desktop.services import load_project_folder
    from braids_desktop.preparation import prepare
    from braids_core.river_project import validate_project
    from braids_core import river_worker
    from braids_core.sediment.jiang_cao_adaptive import JiangCaoAdaptive
    d=load_project_folder(PROJECT);d['settings']['controls']['threads']=threads
    if cfl is not None:d['settings']['controls']['cfl']=float(cfl)
    if dt_max is not None:d['settings']['dt_max_s']=float(dt_max)
    from copy import deepcopy
    from braids_core.river_model import resolve_model
    request=deepcopy(d['model']['request']);request['execution']['backend']=backend
    d['model']=resolve_model(request).document()
    d['settings'].update(end_time_s=end,output_interval_s=end,checkpoint_interval_s=end)
    d['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=end)
    folder=PROJECT/'benchmarks'/label;folder.mkdir(exist_ok=False)
    save(folder/'project.json',validate_project(d))
    t=time.perf_counter();prep=prepare(folder/'project.json',folder/'prepared',asset_cache=PROJECT/'cache')
    prepare_s=time.perf_counter()-t;durations=[];original=JiangCaoAdaptive.step
    def timed(self,dt):
        start=time.perf_counter()
        try:return original(self,dt)
        finally:durations.append(time.perf_counter()-start)
    JiangCaoAdaptive.step=timed;t=time.perf_counter()
    result=river_worker.run(prep,folder/'run',checkpoint_dir=folder/'checkpoints')
    elapsed=time.perf_counter()-t;JiangCaoAdaptive.step=original
    assert result['status']=='complete',result
    import numpy as np
    row=dict(label=label,backend=backend,threads=threads,cfl=d['settings']['controls']['cfl'],dt_max_s=d['settings']['dt_max_s'],
        simulated_s=result['time_s'],steps=result['steps'],prepare_s=prepare_s,
        execution_s=elapsed,step_s=sum(durations),warm_step_median_s=float(np.median(durations[5:])),
        accepted_step_calls=len(durations),rejections=len(result['rejections']),runtime=sha(runtime/'RUNTIME.json'),
        output=str(folder/'run'),status='complete')
    save(folder/'TIMING.json',row);print(json.dumps(row),flush=True)

def select(threads,backend='cpu',cfl=None,dt_max=None):
    imports(PROJECT/'runtime')
    from braids_desktop import services
    d=services.load_project_folder(PROJECT);d['settings']['controls']['threads']=threads
    if cfl is not None:d['settings']['controls']['cfl']=float(cfl)
    if dt_max is not None:d['settings']['dt_max_s']=float(dt_max)
    from copy import deepcopy
    from braids_core.river_model import resolve_model
    request=deepcopy(d['model']['request']);request['execution']['backend']=backend
    d['model']=resolve_model(request).document()
    services.save_project_folder(PROJECT,d)
    services.update_project_session(PROJECT,prepared='prepared/experiment225',output='runs/experiment225',log='logs/experiment225.log')
    print('Selected',backend,'with',threads,'CPU threads',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','trial','select']);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--label');p.add_argument('--end',type=float,default=.25)
    p.add_argument('--cfl',type=float);p.add_argument('--dt-max',type=float);args=p.parse_args()
    if args.action=='create':create()
    elif args.action=='trial':trial(args.threads,args.label,args.end,args.backend,args.cfl,args.dt_max)
    else:select(args.threads,args.backend,args.cfl,args.dt_max)

if __name__=='__main__':main()
