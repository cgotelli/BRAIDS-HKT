"""Generate, but do not run, the four-control dry-channel validation matrix.

All decks have identical topography, initial conditions, Q, resistance and
physical clocks. Only the declared inlet control, backend and mesh mode vary.
No historical user project or reference runtime is changed.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def create(output):
    from osgeo import gdal,osr
    from scipy.optimize import brentq
    from test_river_project import project_document
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core.river_open_boundary import new_state,resolve
    from braids_gpu.atomic_output import write_json_atomic
    out=output.resolve();out.mkdir(parents=True,exist_ok=False);dem=out/'plane_dem.tif'
    width=20.;length=100.;fine=5.;slope=.001;Q=11.;manning=.03;z0=100.
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),120,40,1,gdal.GDT_Float64)
    ds.SetGeoTransform((-10.,1.,0.,30.,0.,-1.));srs=osr.SpatialReference();srs.ImportFromEPSG(2056)
    ds.SetProjection(srs.ExportToWkt());x=-10.+np.arange(120)+.5
    ds.GetRasterBand(1).WriteArray(np.broadcast_to(z0-slope*x,(40,120)).copy());ds.FlushCache();ds=None
    domain=PolygonDomain('normal129_channel',((0.,0.),(length,0.),(length,width),(0.,width)),'EPSG:2056')
    assignments=[]
    for edge in domain.edges:
        if edge.start_xy[0]==edge.end_xy[0]:
            kind='inlet' if edge.start_xy[0]==0 else 'outlet'
            assignments.append(asdict(BoundaryAssignment(edge.edge_id,kind,kind)))
    hn=brentq(lambda h:(width*h)/manning*((width*h)/(width+2*h))**(2/3)*slope**.5-Q,1e-6,10.,xtol=1e-14)
    hc=(Q*Q/(9.81*width*width))**(1/3);datum=z0-slope*fine/2
    methods=dict(normal=dict(method='inlet.normal_discharge',discharge_m3_s=Q,energy_slope='mesh'),
        critical=dict(method='inlet.critical_discharge',discharge_m3_s=Q),
        stage=dict(method='inlet.stage_discharge',discharge_m3_s=Q,stage_m=datum+hn),
        depth=dict(method='inlet.depth_discharge',discharge_m3_s=Q,depth_m=hn,datum_m=datum))
    decks=[]
    for adaptive in (False,True):
        for backend in ('cpu','gpu'):
            for name,control in methods.items():
                d=project_document(dem,backend=backend);s=d['settings'];r=d['model']['request']
                r['hydrodynamics']['roughness']='roughness.manning';d['model']=resolve_model(r).document()
                d['roughness_parameters']['manning_s_m_third']=manning
                s.update(domain=dict(domain_id=domain.domain_id,vertices_xy=domain.vertices_xy,spatial_reference=domain.crs_wkt),
                    assignments=deepcopy(assignments),grid=asdict(NativeGrid(5,1,0.,0.,20.,20.)),levels=3,
                    end_time_s=100.,dt_max_s=1.,output_interval_s=10.,adaptive=adaptive)
                s['controls'].update(cfl=.45,threads=4)
                s['adaptation'].update(interval=1.,coarsen_delay=2,terrain_enabled=not adaptive)
                d['initial'].update(stage_m=90.,velocity_m_s=[0.,0.])
                s['hydraulics']=dict(inlet=deepcopy(control),outlet=dict(method='outlet.critical'))
                d['boundary_states']={n:new_state(v['method']) for n,v in s['hydraulics'].items()}
                d['boundary_resolution']=resolve(s['hydraulics'],d['boundary_states'],dict(inlet='inlet',outlet='outlet'),
                    sediment=None,ng=1,end_time=100.)
                label=('dynamic' if adaptive else 'static')+'_'+backend+'_'+name;d['name']=label
                validate_project(d);write_json_atomic(out/(label+'.json'),d);decks.append(label+'.json')
    record=dict(geometry='100 m x 20 m rectangular channel, 1 per mille slope, finest5m, 3 nested levels',
        initial='dry bed, zero momentum, sediment disabled',discharge_m3_s=Q,manning_s_m_third=manning,
        slope_m_m=slope,normal_depth_m=hn,normal_velocity_m_s=Q/(width*hn),critical_depth_m=hc,
        critical_velocity_m_s=Q/(width*hc),critical_froude=1.,fixed_depth_datum_m=datum,
        reference_stage_m=datum+hn,normal_criterion='Q=A R^(2/3) sqrt(S)/n with sidewalls P=B+2h',
        stage_depth_relation='stage/depth controls intentionally identical to analytic normal incoming reference',
        interpretation='Reference normal/critical states are analytical checks; wet subcritical backwater-compatible actual traces need not remain uniform/critical',
        cfl=.45,dt_max_s=1.,end_time_s=100.,decks=decks,source_file=str(Path(__file__).resolve()))
    write_json_atomic(out/'INPUTS.json',record);print(json.dumps(record,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);create(p.parse_args().output)
