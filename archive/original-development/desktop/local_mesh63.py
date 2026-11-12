"""Stage and exercise sparse turbine refinement through the public worker."""
import argparse,json,hashlib,shutil,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime'
OUT=ROOT/'desktop/qualification/local_mesh63';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/local_mesh63'
FILES=['braids_core/boundaries/local_refinement.py','braids_core/boundaries/breaklines.py','braids_core/boundaries/amr.py',
 'braids_core/river_turbines.py','braids_core/river_project.py','braids_core/sediment/adaptive_river_worker.py',
 'braids_core/sediment/river_turbine_flow.py','braids_desktop/preparation.py','braids_desktop/turbines.py']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stage(refresh=False):
 REPORT.mkdir(parents=True,exist_ok=True);old=read(BASE/'RUNTIME.json')
 if not refresh:
  TARGET.mkdir(parents=True,exist_ok=False)
  for n,h in old['files'].items():
   assert sha(BASE/n)==h,n
   p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
 for n in FILES:
  p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
 (TARGET/'braids_desktop/__init__.py').write_text('__version__ = "0.1.0-preview61"\n')
 files=dict(old['files']);files.update({n:sha(TARGET/n) for n in FILES+['braids_desktop/__init__.py']})
 save(TARGET/'RUNTIME.json',dict(old,files=files))
 save(REPORT/'STAGE.json',dict(parent=sha(BASE/'RUNTIME.json'),runtime=sha(TARGET/'RUNTIME.json'),changed=[n for n,h in files.items() if h!=old['files'].get(n)]))
def qualify():
 import numpy as np
 sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT))
 if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
 from osgeo import gdal,osr
 from braids_desktop import services,preparation,project_file
 from braids_desktop.results import ResultsStore
 from braids_core import river_worker,river_turbines as turbines
 from braids_core.river_model import resolve_model
 from braids_core.boundaries.domain import PolygonDomain
 from braids_core.sediment.portable_river_runtime import install
 install(read(TARGET/'PRECOMPILED.json'),TARGET)
 home=Path.home()/'Desktop/BRAIDS_Turbine_LocalMesh63';home.mkdir(exist_ok=True)
 dem=home/'DEM.tif'
 if not dem.exists():
  ds=gdal.GetDriverByName('GTiff').Create(str(dem),162,82,1,gdal.GDT_Float64);ds.SetGeoTransform((-.25,.25,0,20.25,0,-.25))
  s=osr.SpatialReference();s.ImportFromEPSG(2056);ds.SetProjection(s.ExportToWkt());ds.GetRasterBand(1).Fill(0.);ds=None
 states={};metrics={}
 for mode in ['static_terrain','dynamic_depth']:
  for backend in ['cpu','gpu']:
   name=mode+'_'+backend;folder=home/(name+'_final');folder.mkdir(exist_ok=False)
   doc=services.default_document(dem,[[0,0],[40,0],[40,20],[0,20]],'EPSG:2056',name=name,backend=backend,minimum_size_m=2.,levels=3,mesh_mode=mode,end_time_s=.25)
   req=doc['model']['request'];req['hydrodynamics']['roughness']='roughness.chezy';doc['model']=resolve_model(req).document()
   doc['roughness_parameters']['chezy_m_half_s']=100.;doc['settings']['controls'].update(chezy=100.,threads=4,cfl=.5)
   d=PolygonDomain(**doc['settings']['domain'])
   for x,n,k,b in [(0,'In','inlet',dict(method='inlet.total_discharge',discharge_m3_s=32.)),(40,'Out','outlet',dict(method='outlet.stage',stage_m=2.))]:
    doc=services.assign_boundary(doc,[e.edge_id for e in d.edges if e.start_xy[0]==e.end_xy[0]==x],n,k,b)
   doc['initial'].update(stage_m=2.,velocity_m_s=[.8,0.]);doc['settings'].update(dt_max_s=.03125,output_interval_s=.03125,checkpoint_interval_s=.125)
   doc['settings']['adaptation'].update(interval=.0625,terrain_enabled=(mode=='static_terrain'))
   doc['turbines']=turbines.default();doc['turbines']['turbines']=[turbines.definition([8.,8.],[1.,.15]),turbines.definition([24.,14.],[1.,0.],name='Large',diameter=1.4)]
   doc=services.save_project_folder(folder/'Project_data',doc);project_file.write(folder/'Project.braids.json',folder/'Project_data',doc)
   start=time.perf_counter();p=preparation.prepare(folder/'Project_data/braids_project.json',folder/'prepared',asset_cache=home/'assets');seconds=time.perf_counter()-start
   full=river_worker.run(p,folder/'run');assert full['status']=='complete',full
   river_worker.run(p,folder/'part',until=.125);river_worker.run(p,folder/'restart',restart=folder/'part/checkpoint.npz')
   f=ResultsStore(folder/'run').read(-1);r=ResultsStore(folder/'restart').read(-1)
   np.testing.assert_array_equal(f.q,r.q);np.testing.assert_array_equal(f.z,r.z)
   states[name]=f.q.copy();diag=read(folder/'run/DIAGNOSTICS.json')
   assert all(abs(v['water_residual_m3'])<=v['water_roundoff_limit_m3'] for v in diag)
   metrics[name]=dict(prepare_seconds=seconds,cells=len(f.polygons),time=f.time_s,exact_restart=True,water_residual_max=max(abs(v['water_residual_m3']) for v in diag),turbines=read(p/'TURBINES.json')['rotors'],result=full)
   save(REPORT/'COUPLED.json',metrics);print(name,len(f.polygons),flush=True)
  error=float(np.max(abs(states[mode+'_cpu']-states[mode+'_gpu'])));assert error<1e-9,error
  metrics[mode+'_backend_error']=error
 save(REPORT/'COUPLED.json',metrics)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','refresh','qualify']);a=p.parse_args()
 stage(a.action=='refresh') if a.action in ('stage','refresh') else qualify()
