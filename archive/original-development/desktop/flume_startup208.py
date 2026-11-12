"""Reconstructed paper dimensions, explicitly assumed startup/BCs, public path."""
import argparse,json,sys,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',choices=['cpu','gpu'],required=True)
    p.add_argument('--runtime');p.add_argument('--end',type=float,default=2.);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if a.runtime:sys.path.insert(0,str(Path(a.runtime).resolve()))
    from braids_core.sediment.portable_river_runtime import install
    if a.runtime:
        from braids_core.river_launcher import verify
        verify();install(json.loads((Path(a.runtime)/'PRECOMPILED.json').read_text()),Path(a.runtime).resolve())
    else:install()
    from test_jiang_cao207 import project
    from braids_core.boundaries.domain import PolygonDomain
    from braids_desktop.services import geometry_grid
    from braids_core.river_open_boundary import new_state,resolve
    from braids_core.river_project import validate_project
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    from osgeo import gdal,osr
    import numpy as np
    d=project(out/'input',a.backend,collapse=True,opened=True)
    # Replacement of THIS newly generated fixture DEM only, never a user asset.
    dem=out/'plane.tif';dx=.025;nx=1040;ny=100
    x=-.5+(np.arange(nx)+.5)*dx
    z=np.broadcast_to(.01*(25-x),(ny,nx)).copy()
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),nx,ny,1,gdal.GDT_Float64)
    crs=osr.SpatialReference();crs.ImportFromEPSG(2056)
    ds.SetGeoTransform((-.5,dx,0,2.,0,-dx));ds.SetProjection(crs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(z);ds=None
    d['terrain']['dem_path']=str(dem)
    dom=d['settings']['domain'];dom['vertices_xy']=[[0,0],[25,0],[25,1.5],[0,1.5]]
    domain=PolygonDomain(**dom);d['settings']['grid']=geometry_grid(domain,.05,1)
    d['settings']['assignments']=[]
    for x,name,kind in ((0,'inlet','inlet'),(25,'outlet','outlet')):
        e=next(e for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x)
        d['settings']['assignments'].append(dict(edge_id=e.edge_id,bc_id=name,kind=kind))
    qw=.0025;ms=.00286;rho=2650.;qs=ms/rho;ratio=qs/qw
    d['settings']['hydraulics']=dict(inlet=dict(method='inlet.normal_discharge',discharge_m3_s=qw+qs,energy_slope='mesh'),
                                     outlet=dict(method='outlet.critical'))
    d['boundary_states']={n:new_state(b['method'],sediment=True) for n,b in d['settings']['hydraulics'].items()}
    d['settings']['sediment_boundaries']['inlet']['solid_water_ratio']=[ratio]
    d['initial'].update(stage_m=0.,velocity_m_s=[0.,0.],layer_thickness_m=[.2])
    d['settings'].update(end_time_s=a.end,dt_max_s=.005,output_interval_s=.5,checkpoint_interval_s=a.end/2)
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(inlet='inlet',outlet='outlet'),sediment=d['model']['physics']['sediment'],ng=1,end_time=a.end)
    d=validate_project(d)
    specification=dict(target='Jiang-Cao Case 1 Run 1; bounded reconstruction, not original author input',
        published=dict(length_m=25.,width_m=1.5,slope=.01,water_discharge_m3_s=qw,sediment_feed_kg_s=ms,D50_m=.001,nominal_mesh_m=.05),
        assumed=dict(initial_water='dry',initial_velocity=[0,0],initial_mobile_solid=0.,initial_secondary=0.,
                     inlet='full-width uniform prescribed mixture Q with mesh-slope normal reference',outlet='free critical/supercritical export',
                     solid_density_kg_m3=rho,water_density_kg_m3=1000.,porosity=.4,molecular_viscosity_m2_s=1e-6,
                     finite_erodible_depth_m=.2,repose_angle_deg=30.,perturbation='none imposed',datum='outlet bed 0 m',
                     gravity_m_s2=9.81,paper_gravity_m_s2=9.8,MF=1.,mesh='fixed Cartesian squares, not author triangles'),
        conversion=dict(solid_volume_discharge_m3_s=qs,solid_water_ratio=ratio,mixture_discharge_m3_s=qw+qs),
        model=d['model'],scope='2 s startup; not hydraulic equilibrium, braid emergence or experimental validation')
    (out/'INPUT_AUDIT.json').write_text(json.dumps(specification,indent=2))
    file=out/'flume.braids.json';file.write_text(json.dumps(d,indent=2))
    prep=prepare(file,out/'prepared',asset_cache=out/'assets')
    result=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    (out/'CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
