"""Low-discharge dry front on cropped oblique walls with actual AMR."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'validation/_shallow209/candidate_v3'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',required=True,choices=['cpu','gpu']);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    from test_jiang_cao207 import project
    from braids_core.boundaries.domain import PolygonDomain
    from braids_desktop.services import geometry_grid
    from braids_core.river_open_boundary import new_state,resolve
    from braids_core.river_project import validate_project
    from braids_desktop.preparation import prepare
    from braids_core.river_worker import run
    from osgeo import gdal
    import numpy as np
    d=project(out/'input',a.backend,adaptive=True,opened=True)
    # This is the newly generated test DEM, not an existing user's terrain.
    ds=gdal.Open(d['terrain']['dem_path'],gdal.GA_Update)
    gt=ds.GetGeoTransform();x=gt[0]+(np.arange(ds.RasterXSize)+.5)*gt[1]
    ds.GetRasterBand(1).WriteArray(np.broadcast_to(.01*(3.2-x),(ds.RasterYSize,ds.RasterXSize)).copy());ds=None
    dom=d['settings']['domain'];dom['vertices_xy']=[[0,0],[3.2,.2],[3.2,1],[0,.8]]
    domain=PolygonDomain(**dom);d['settings']['grid']=geometry_grid(domain,.05,2)
    d['settings']['assignments']=[]
    for x,name,kind in ((0,'inlet','inlet'),(3.2,'outlet','outlet')):
        edge=next(e for e in domain.edges if e.start_xy[0]==e.end_xy[0]==x)
        d['settings']['assignments'].append(dict(edge_id=edge.edge_id,bc_id=name,kind=kind))
    ratio=(.00286/2650)/.0025;qw=.0025*.8/1.5
    d['settings']['hydraulics']=dict(inlet=dict(method='inlet.normal_discharge',discharge_m3_s=qw*(1+ratio),energy_slope='mesh'),outlet=dict(method='outlet.critical'))
    d['boundary_states']={n:new_state(b['method'],sediment=True) for n,b in d['settings']['hydraulics'].items()}
    d['settings']['sediment_boundaries']['inlet']['solid_water_ratio']=[ratio]
    d['initial'].update(stage_m=0.,velocity_m_s=[0.,0.],layer_thickness_m=[.2])
    d['settings'].update(end_time_s=.8,dt_max_s=.002,output_interval_s=.1,checkpoint_interval_s=.4)
    d['settings']['adaptation'].update(interval=.05,buffer_cells=1)
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(inlet='inlet',outlet='outlet'),sediment=d['model']['physics']['sediment'],ng=1,end_time=.8)
    d=validate_project(d);file=out/'project.json';file.write_text(json.dumps(d,indent=2))
    prep=prepare(file,out/'prepared',asset_cache=out/'assets');result=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    (out/'CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],steps=result['steps'],regrids=sum(e['changed'] for e in result['adaptation_events']))))
if __name__=='__main__':main()
