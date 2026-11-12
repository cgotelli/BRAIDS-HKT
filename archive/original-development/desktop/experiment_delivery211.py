"""Self-contained MF1 flume reconstruction through the installed Desktop worker."""
import json,sys,subprocess,time
from pathlib import Path
from stage_delivery211 import ROOT,OUT,TARGET,sha,save,verify

FOLDER=ROOT.parent/'BRAIDS_JiangCao_MF1'
RUNTIME=ROOT/'desktop/runtime'

def execute(commands):
    from braids_desktop import services
    prep=FOLDER/'prepared/flume211';output=FOLDER/'runs/flume211'
    logdir=FOLDER/'logs';logdir.mkdir(exist_ok=True)
    timings={}
    for command in commands:
        action=command[0];start=time.perf_counter()
        print('Installed worker',action,str(FOLDER),flush=True)
        with (logdir/(action+'.log')).open('x',encoding='utf-8') as log:
            code=subprocess.call([sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        timings[action]=time.perf_counter()-start
        if code:raise RuntimeError(f'{action} failed; inspect {logdir/(action+".log")}')
        services.update_project_session(FOLDER,prepared=prep.relative_to(FOLDER).as_posix(),**({'output':output.relative_to(FOLDER).as_posix(),'log':'logs/run.log'} if action=='run' else {}))
    save(OUT/'EXPERIMENT.json',dict(project=str(FOLDER),runtime=sha(RUNTIME/'RUNTIME.json'),timings_s=timings))
    print('Completed 20 s MF1',timings,flush=True)

def main():
    assert sha(RUNTIME/'RUNTIME.json')==sha(TARGET/'RUNTIME.json');verify(RUNTIME)
    sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
    if '--run-prepared' in sys.argv:
        from braids_desktop import services
        audit=json.loads((FOLDER/'INPUT_AUDIT.json').read_text())
        assert audit['runtime']==sha(RUNTIME/'RUNTIME.json')
        services.load_project_folder(FOLDER)
        services.update_project_session(FOLDER,prepared='prepared/flume211')
        return execute([['run','--input',str(FOLDER/'prepared/flume211'),'--output',str(FOLDER/'runs/flume211'),'--checkpoint-dir',str(FOLDER/'checkpoints/flume211')]])
    from test_jiang_cao207 import project
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_open_boundary import new_state,resolve
    from braids_core.river_project import validate_project
    from braids_desktop import services
    from osgeo import gdal,osr
    import numpy as np
    assert not FOLDER.exists(),FOLDER
    source=OUT/'experiment_input';source.mkdir(exist_ok=False)
    d=project(source/'fixture','gpu',collapse=True,opened=True)
    dem=source/'plane.tif';dx=.025;nx=1040;ny=100
    x=-.5+(np.arange(nx)+.5)*dx
    z=np.broadcast_to(.01*(25-x),(ny,nx)).copy()
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),nx,ny,1,gdal.GDT_Float64)
    crs=osr.SpatialReference();crs.ImportFromEPSG(2056)
    ds.SetGeoTransform((-.5,dx,0,2.,0,-dx));ds.SetProjection(crs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(z);ds=None
    d['name']='Jiang-Cao Case 1 Run 1 - MF1 reconstruction'
    d['terrain']['dem_path']=str(dem)
    dom=d['settings']['domain'];dom['vertices_xy']=[[0,0],[25,0],[25,1.5],[0,1.5]]
    domain=PolygonDomain(**dom);d['settings']['grid']=services.geometry_grid(domain,.05,1)
    d['settings']['assignments']=[]
    for x,name,kind in ((0,'inlet','inlet'),(25,'outlet','outlet')):
        e=next(e for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x)
        d['settings']['assignments'].append(dict(edge_id=e.edge_id,bc_id=name,kind=kind))
    qw=.0025;ms=.00286;rho=2650.;qs=ms/rho;ratio=qs/qw
    d['settings']['hydraulics']=dict(inlet=dict(method='inlet.normal_discharge',discharge_m3_s=qw+qs,energy_slope='mesh'),outlet=dict(method='outlet.critical'))
    d['boundary_states']={n:new_state(b['method'],sediment=True) for n,b in d['settings']['hydraulics'].items()}
    # Match the single-grain editor's explicit inlet composition from creation.
    d['boundary_states']['inlet']['composition']=dict(method='composition.reference_volume_fraction',fractions=[1.])
    d['settings']['sediment_boundaries']['inlet']['solid_water_ratio']=[ratio]
    d['initial'].update(stage_m=0.,velocity_m_s=[0.,0.],layer_thickness_m=[.2])
    d['settings'].update(end_time_s=20.,dt_max_s=.005,output_interval_s=1.,checkpoint_interval_s=10.)
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(inlet='inlet',outlet='outlet'),sediment=d['model']['physics']['sediment'],ng=1,end_time=20.)
    d=validate_project(d)
    services.save_project_folder(FOLDER,d,source_base=source)
    save(FOLDER/'INPUT_AUDIT.json',dict(
        target='Jiang-Cao Case 1 Run 1 / Garcia Lugo Run 27; reconstruction, not author input deck',
        reference='validation/sediment/river_morphodynamics/reports/desktop_braiding194/SPECIFICATION.json',
        published=dict(length_m=25.,width_m=1.5,slope=.01,water_discharge_m3_s=qw,sediment_feed_kg_s=ms,D50_m=.001,nominal_mesh_m=.05),
        assumed=dict(initial_water='dry',initial_velocity=[0,0],initial_mobile_solid=0.,initial_secondary=0.,
          inlet='full-width prescribed mixture Q with mesh-slope normal reference',outlet='critical export',
          solid_density_kg_m3=rho,water_density_kg_m3=1000.,porosity=.4,molecular_viscosity_m2_s=1e-6,
          finite_erodible_depth_m=.2,repose_angle_deg=30.,perturbation='none imposed',datum='outlet bed zero',
          gravity_m_s2=9.81,paper_gravity_m_s2=9.8,MF=1.,mesh='fixed Cartesian squares; not author triangles',
          coordinates='synthetic local metres, EPSG2056 metadata only; not a geographic site'),
        conversion=dict(solid_volume_discharge_m3_s=qs,solid_water_ratio=ratio,mixture_discharge_m3_s=qw+qs),
        model=d['model'],runtime=sha(RUNTIME/'RUNTIME.json'),scope='20 physical and morphological seconds; startup, not equilibrium or braid validation'))
    prep=FOLDER/'prepared/flume211';output=FOLDER/'runs/flume211'
    commands=[['prepare','--project',str(FOLDER/'braids_project.json'),'--output',str(prep),'--asset-cache',str(FOLDER/'cache')],
      ['run','--input',str(prep),'--output',str(output),'--checkpoint-dir',str(FOLDER/'checkpoints/flume211')]]
    execute(commands)

if __name__=='__main__':main()
