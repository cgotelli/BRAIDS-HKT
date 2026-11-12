"""Closed cropped-wall 2D hydraulic CFL experiments on the public worker.

The diagonal dam initial water field is explicitly injected BEFORE preparation
hashes are made, not by editing an authenticated checkpoint. Its actual fields
and the hook identity are recorded separately from the template project recipe.
No maintained numerical source, sediment formula or installed plugin changes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def make_project(output,case,backend):
    from osgeo import gdal,osr
    from test_river_project import project_document
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.river_open_boundary import resolve
    from braids_core.river_project import validate_project
    from braids_core.river_model import resolve_model
    from braids_gpu.atomic_output import write_json_atomic
    output.mkdir(parents=True,exist_ok=False)
    dem=output/'dem.tif'
    ds=gdal.GetDriverByName('GTiff').Create(str(dem),48,48,1,gdal.GDT_Float64)
    ds.SetGeoTransform((-2.,.5,0.,22.,0.,-.5));srs=osr.SpatialReference();srs.ImportFromEPSG(2056)
    ds.SetProjection(srs.ExportToWkt())
    x=-2.+(np.arange(48)+.5)*.5;y=22.-(np.arange(48)+.5)*.5
    z=.04*x[None,:]+.03*y[:,None] if case=='lake' else np.zeros((48,48))
    ds.GetRasterBand(1).WriteArray(z);ds.FlushCache();ds=None
    d=project_document(dem,backend=backend);s=d['settings']
    d['name']='Isolated130 closed cropped '+case
    s['domain']=dict(domain_id='closed130',vertices_xy=((.2,.2),(19.8,.4),(19.6,19.8),(.4,19.6)),
                     spatial_reference='EPSG:2056')
    s.update(assignments=[],grid=asdict(NativeGrid(20,20,0.,0.,1.,1.)),levels=1,
        hydraulics={},sediment_boundaries={},end_time_s=2.,dt_max_s=.2,output_interval_s=.2,adaptive=False)
    s['controls'].update(cfl=.5,threads=4)
    s['adaptation'].update(interval=1.,coarsen_delay=2,terrain_enabled=True)
    d['initial'].update(stage_m=.6 if case=='lake' else -1.,velocity_m_s=[0.,0.])
    request=d['model']['request'];request['hydrodynamics']['roughness']='roughness.manning'
    d['model']=resolve_model(request).document();d['roughness_parameters']['manning_s_m_third']=.03
    d['boundary_states']={};d['sediment_schedules']={}
    d['boundary_resolution']=resolve({}, {}, {},sediment=None,ng=1,end_time=2.)
    validate_project(d);write_json_atomic(output/'PROJECT.json',d)
    return output/'PROJECT.json'


def run(output,case,backend,cap):
    from braids_core import river_worker as w
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.shoreline_solver import ShorelineSolver
    from braids_core.sediment.bed_columns_v101 import host
    from braids_gpu.atomic_output import write_json_atomic,save_arrays_atomic
    from tools.research_wet_cfl130 import candidate
    project=make_project(output,case,backend)
    original_prepare=w.prepare_recipe;original_factory=core.factory;original_step=ShorelineSolver.step
    owners=[];history=[];steps=[];initial_data={}

    def prepare(*args,**kwargs):
        callback=kwargs.get('on_fields')
        def initialize(tree,fields):
            xy=np.asarray([c.centroid_xy for c in tree.finest.cells])
            if case=='dam':
                h=np.where(xy[:,0]+xy[:,1]<15.,.3,0.)
                fields['q']=np.stack((h,0.*h,0.*h))
            initial_data.update(q=fields['q'].copy(),xy=xy,
                area=np.asarray([c.area_m2 for c in tree.finest.cells]))
            save_arrays_atomic(output/'INITIAL_OVERRIDE.npz',**initial_data)
            if callback:callback(tree,fields)
        kwargs['on_fields']=initialize
        return original_prepare(*args,**kwargs)

    def factory(*args,**kwargs):
        make=original_factory(*args,**kwargs)
        def create(*a,**k):
            owner=make(*a,**k);owners.append(owner);return owner
        create.required_leaves=make.required_leaves
        return create

    def step(self,dt,*args,**kwargs):
        result=original_step(self,dt,*args,**kwargs)
        steps.append(dict(time_s=self.time,diagnostics=result['diagnostics']))
        return result

    def sample(folder,row):
        a=owners[-1];q=host(a.q);d=a.data;p=a.physics
        v=np.divide(q[1:],q[0],out=np.zeros_like(q[1:]),where=q[0]>p.parameters.dry)
        walls=d['neighbour']<0;wall_cells=np.unique(d['owner'][walls])
        record=dict(time_s=row['time_s'],cells=p.n,minimum_depth_m=float(q[0].min()),
            maximum_depth_m=float(q[0].max()),maximum_speed_m_s=float(np.linalg.norm(v,axis=0).max()),
            water_m3=float(q[0]@d['area']),water_residual_m3=float(row['water_residual_m3']),
            minimum_cell_area_m2=float(d['area'].min()),cropped_cells=int(np.count_nonzero(d['area']<.999)),
            wall_integral=float(np.max(abs(host(a.water_ledger)[walls]),initial=0.)),
            wall_depth_max_m=float(q[0,wall_cells].max()),minimum_theta=float(getattr(a.solver.midpoint,'minimum_theta',1.)))
        for threshold in (.001,.01):
            wet=q[0]>threshold;xy=d['xy'][:,wet]
            record[f'wet_area_{threshold:g}_m2']=float(d['area'][wet].sum())
            record[f'front_diagonal_{threshold:g}_m']=float((xy[0]+xy[1]).max(initial=0.)/np.sqrt(2.))
        history.append(record)

    w.prepare_recipe=prepare;core.factory=factory;ShorelineSolver.step=step
    source_before=w.source_identity()
    try:
        with candidate(cap) as context:
            prepared=w.prepare(project,output/'prepared')
            status=w.run(prepared,output/'run',on_frame=sample)
            actual_identity=context['identity']
    finally:
        w.prepare_recipe=original_prepare;core.factory=original_factory;ShorelineSolver.step=original_step
    assert source_before==w.source_identity()
    rejections=[r for step in steps for r in step['diagnostics']['hydraulic_rejections']]
    final=core.read_live_frame(output/'run')[2]
    record=dict(case=case,backend=backend,source_runtime=source_before,research=actual_identity,
        initialization=dict(hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definition='h=.3m where centroid x+y<15m, otherwise0; zero momentum' if case=='dam'
                       else 'eta=.6m; bed=.04x+.03y; zero momentum',
            actual_field_sha256=hashlib.sha256((output/'INITIAL_OVERRIDE.npz').read_bytes()).hexdigest(),
            note='Private initialized fields frozen by prepared receipts; template recipe alone does not reproduce diagonal dam'),
        status=status,history=history,steps=steps,
        substeps=sum(s['diagnostics']['hydraulic_substeps'] for s in steps),
        rejections=rejections,
        finite=all(np.isfinite(v).all() for v in final.values()),
        source_unchanged=True,timing_scope='Concurrent qualification, not a performance benchmark')
    if case=='lake':
        record['stationary_q_component_max_m']=np.max(abs(final['q']-initial_data['q']),axis=1).tolist()
    write_json_atomic(output/'RESEARCH.json',record)
    checks=dict(completed=status['status']=='complete',nonnegative=all(r['minimum_depth_m']>=0 for r in history),
        conservation=max(abs(r['water_residual_m3']) for r in history)<1e-9,
        closed_walls=all(r['wall_integral']==0 for r in history),finite=record['finite'])
    print(json.dumps(dict(case=case,backend=backend,cfl=cap,checks=checks,substeps=record['substeps'],
        rejections=len(rejections),lake_drift=record.get('stationary_q_component_max_m')),indent=2),flush=True)
    if not all(checks.values()):raise RuntimeError('2D qualification checks failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['lake','dam'],required=True)
    p.add_argument('--backend',choices=['cpu','gpu'],required=True);p.add_argument('--cfl',type=float,required=True)
    a=p.parse_args();run(a.output.resolve(),a.case,a.backend,a.cfl)


if __name__=='__main__':main()
