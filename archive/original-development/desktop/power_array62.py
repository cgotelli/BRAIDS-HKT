"""Bounded power/wake audit. Frozen public-runtime base, no fitted coefficients."""
import argparse,csv,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_array62'
BASE=ROOT/'desktop/runtime';RUNTIME=ROOT/'desktop/qualification/power_array62/candidate'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Power62'
FILES=['braids_core/river_turbines.py','braids_core/sediment/river_turbine_flow.py']
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT));sys.dont_write_bytecode=True
os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stage():
    REPORT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(parents=True,exist_ok=False)
    old=read(BASE/'RUNTIME.json');files=dict(old['files'])
    for n,h in files.items():
        assert sha(BASE/n)==h,n
        target=RUNTIME/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,target)
    for n in FILES:shutil.copy2(ROOT/n,RUNTIME/n);files[n]=sha(RUNTIME/n)
    save(RUNTIME/'RUNTIME.json',dict(old,files=files))
    save(REPORT/'STAGE.json',dict(parent_runtime_sha256=sha(BASE/'RUNTIME.json'),runtime_sha256=sha(RUNTIME/'RUNTIME.json'),changed=FILES,native_unchanged=True))

def controls():
    # Same measured single-device curve used in the earlier Mycek comparison.
    with (ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figures6a_7a.csv').open() as f:raw=list(csv.DictReader(f))
    curves={key:sorted([(float(r['TSR']),float(r['coefficient'])) for r in raw if r['field']==key and float(r['inflow_m_s'])==.8 and float(r['TI_percent'])==3.]) for key in ('Ct','Cp')}
    return dict(mode='prescribed_tsr_table',tip_speed_ratio=3.67,performance=[dict(tip_speed_ratio=x,thrust_coefficient=y,power_coefficient=float(np.interp(x,*np.array(curves['Cp']).T))) for x,y in curves['Ct'] if x>0])

def create(case):
    from osgeo import gdal,osr
    from braids_desktop import services,project_file
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_model import resolve_model
    from braids_core.river_turbines import definition,default
    # Mycek single-device physical dimensions, also used as a transparent
    # tandem diagnostic. Prescribed local TSR is not the experiment's fixed RPM.
    length,width,depth,D=18.,4.,2.,.7;speed=1. if case=='single_fast' else .8
    backend='cpu' if case.endswith('_cpu') else 'gpu'
    folder=HOME/case;folder.mkdir(parents=True,exist_ok=False);dem=folder/'DEM.tif'
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),182,42,1,gdal.GDT_Float64)
    ds.SetGeoTransform((-.1,.1,0.,4.1,0.,-.1));s=osr.SpatialReference();s.ImportFromEPSG(2056)
    ds.SetProjection(s.ExportToWkt());ds.GetRasterBand(1).Fill(0.);ds=None
    doc=services.default_document(dem,[[0.,0.],[length,0.],[length,width],[0.,width]],'EPSG:2056',name=case,backend=backend,minimum_size_m=.1,levels=1,mesh_mode='static_terrain',end_time_s=60.)
    r=doc['model']['request'];r['hydrodynamics']['roughness']='roughness.chezy';doc['model']=resolve_model(r).document()
    doc['roughness_parameters']['chezy_m_half_s']=1000.;doc['settings']['controls'].update(chezy=1000.,cfl=.5,threads=8)
    domain=PolygonDomain(**doc['settings']['domain'])
    for x,name,kind,bc in [(0.,'Upstream','inlet',dict(method='inlet.total_discharge',discharge_m3_s=speed*depth*width)),(length,'Downstream','outlet',dict(method='outlet.stage',stage_m=depth))]:
        doc=services.assign_boundary(doc,[e.edge_id for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x],name,kind,bc)
    doc['initial'].update(stage_m=depth,velocity_m_s=[speed,0.])
    doc['settings'].update(dt_max_s=.2,output_interval_s=2.,checkpoint_interval_s=60.)
    doc['settings']['adaptation'].update(domain_border_enabled=False,terrain_enabled=True)
    t=definition([5.,2.],[1.,0.],name='Upstream rotor',diameter=D,hub_height=.9,minimum_bed_clearance=.1,minimum_surface_clearance=.1);t['control']=controls()
    if 'rpm4' in case:
        table=t['control']['performance']
        t['control']=dict(mode='fixed_rpm_table',model_name='Mycek 2014 Part I measured U0.8 TI3 curve',
            rotational_speed_rpm=4*.8/(D/2)*30/math.pi,performance=table,outside_curve='off')
    rows=[t]
    if case.startswith('tandem'):
        distance=5. if '5D' in case else (6. if '6D' in case else 10.)
        # This constructed deck places rotors exactly on its declared lattice.
        # Equivalent decimal arithmetic at 6D differed by one ULP from the
        # lattice edge and exposed a retained mesh-builder failure in v1.
        x=round((5.+distance*D)/.1)*.1
        other=json.loads(json.dumps(t));other.update(id='Downstream rotor',center_xy=[x,2.]);rows.append(other)
    doc['turbines']=default();doc['turbines'].update(turbines=rows,ambient_viscosity_m2_s=.585/(2.9*18.75)*.8*.7)
    doc=services.validate_document(doc)
    saved=services.save_project_folder(folder/'Project_data',doc)
    project_file.write(folder/'Project.braids.json',folder/'Project_data',saved)
    save(folder/'SPEC.json',dict(case=case,backend=backend,D=D,H=depth,U=speed,domain=[length,width],smallest_cell_m=.1,
         runtime_sha256=sha(RUNTIME/'RUNTIME.json'),performance_curve_source='Mycek 2014 Part I, Figures 6a/7a, U=.8, TI=3%',
         purpose='Part II nominal TSR4 point comparison; mixing and 2D depth interpretation remain assumptions' if 'rpm4' in case else 'Controlled numerical tandem/flow-speed diagnostic; not independent tandem experimental validation',
         fixed_nu_m2_s=doc['turbines']['ambient_viscosity_m2_s'],mixing='Frozen previous Mycek nondimensional viscosity; not a turbulence-intensity model',
         controller='Fixed RPM from nominal U=.8 and TSR=4 for both rotors' if 'rpm4' in case else 'Prescribed local TSR=3.67; not fixed RPM or author tandem operating points',
         averaging_s=[40.,60.],end_s=60.))
    return folder

def run(case):
    from braids_core.sediment import portable_river_runtime
    from braids_core import river_worker
    from braids_desktop import preparation
    if not portable_river_runtime.ACTIVE:portable_river_runtime.install(read(RUNTIME/'PRECOMPILED.json'),RUNTIME)
    folder=create(case);project=folder/'Project_data/braids_project.json'
    print('Preparing',case,project,flush=True)
    p=preparation.prepare(project,folder/'prepared',asset_cache=HOME/'assets')
    begin=time.monotonic();result=river_worker.run(p,folder/'run')
    save(folder/'DONE.json',dict(result=result,wall_seconds=time.monotonic()-begin));print(case,result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','run']);p.add_argument('case',nargs='?');a=p.parse_args()
    stage() if a.action=='stage' else run(a.case)
