"""New validation-only projects derived from aaa; originals remain read-only."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--method',choices=['critical','normal','stage','depth'],default='critical')
    ap.add_argument('--until',type=float,default=40.)
    ap.add_argument('--cfl',type=float,default=.5)
    ap.add_argument('--dt-max',type=float,default=.5)
    ap.add_argument('--stage',type=float)
    ap.add_argument('--mesh-mode',choices=['original','dynamic_water','static_terrain'],default='original')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from braids_core import river_worker as w
    from braids_core.river_project import SCHEMA,validate_project
    from braids_core.river_model import resolve_model
    from braids_core.river_open_boundary import new_state,resolve
    from braids_core.boundaries.axis_aligned import derive_axis_aligned
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_gpu.atomic_output import write_json_atomic
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.bed_columns_v101 import host
    from braids_core.sediment.shoreline_solver import ShorelineSolver
    raw=args.source.read_bytes();d=deepcopy(json.loads(raw));s=d['settings']
    d['schema']=SCHEMA;d['name']='Inlet129 '+args.method+' '+args.backend
    r=d['model']['request'];r['execution']['backend']=args.backend;d['model']=resolve_model(r).document()
    s.update(end_time_s=args.until,dt_max_s=args.dt_max,output_interval_s=min(10.,args.until))
    s['controls']['cfl']=args.cfl
    if args.mesh_mode!='original':
        s['adaptive']=args.mesh_mode=='dynamic_water'
        s['adaptation']['terrain_enabled']=args.mesh_mode=='static_terrain'
    g=NativeGrid(**s['grid']);ratio=2**(s['levels']-1)
    fine=NativeGrid(g.nx*ratio,g.ny*ratio,g.xlower_m,g.ylower_m,g.dx_m/ratio,g.dy_m/ratio)
    domain,assignments,audit=derive_axis_aligned(PolygonDomain(**s['domain']),[BoundaryAssignment(**v) for v in s['assignments']],fine)
    s['domain']=dict(domain_id=domain.domain_id,vertices_xy=domain.vertices_xy,spatial_reference=domain.crs_wkt)
    s['assignments']=[asdict(a) for a in assignments]
    name=next(n for n,v in s['hydraulics'].items() if v['method'].startswith('inlet.'))
    if args.method=='critical':control=dict(method='inlet.critical_discharge',discharge_m3_s=11.)
    elif args.method=='normal':control=dict(method='inlet.normal_discharge',discharge_m3_s=11.,energy_slope='mesh')
    else:
        if args.stage is None:raise ValueError('Explicit recorded stage is required for stage/depth validation')
        control=(dict(method='inlet.stage_discharge',discharge_m3_s=11.,stage_m=args.stage) if args.method=='stage'
            else dict(method='inlet.depth_discharge',discharge_m3_s=11.,datum_m=464.,depth_m=args.stage-464.))
    s['hydraulics'][name]=control;d['boundary_states'][name]=new_state(control['method'])
    kinds={a.bc_id:a.kind for a in assignments}
    d['boundary_resolution']=resolve(s['hydraulics'],d['boundary_states'],kinds,sediment=None,ng=1,end_time=args.until)
    validate_project(d)
    write_json_atomic(out/'PROJECT.json',d)
    write_json_atomic(out/'INPUT_CHANGES.json',dict(source=str(args.source),sha256=hashlib.sha256(raw).hexdigest(),
        new_schema=SCHEMA,backend=args.backend,method=control,geometry=audit,cfl=args.cfl,dt_max=args.dt_max,end=args.until))
    started_prepare=time.perf_counter();prepared=w.prepare(out/'PROJECT.json',out/'prepared')
    preparation_s=time.perf_counter()-started_prepare
    make_orig=core.factory;owners=[];history=[];steps=[]
    def factory(*a,**kw):
        make=make_orig(*a,**kw)
        def build(leaves,mesh):
            owner=make(leaves,mesh);owners.append(owner);return owner
        build.required_leaves=make.required_leaves
        return build
    core.factory=factory
    step_orig=ShorelineSolver.step
    def step(self,dt,*a,**kw):
        began=time.perf_counter();result=step_orig(self,dt,*a,**kw)
        steps.append(dict(time_s=self.time,dt=dt,wall_s=time.perf_counter()-began,diagnostics=result['diagnostics']))
        return result
    ShorelineSolver.step=step
    def sample(folder,row):
        a=owners[-1];p=a.physics;data=a.data;q=host(a.q);z=host(a.z);ext=a.exterior_bed
        ext.evaluate(a.q,a.z,time=a.time)
        f=np.flatnonzero((data['bid']==0)&(data['neighbour']<0));cells=np.unique(data['owner'][f])
        h=q[0];v=np.divide(q[1:],h,out=np.zeros_like(q[1:]),where=h>p.parameters.dry)
        trace=host(ext.trace);flux=host(ext.flux)
        normal_speed=(v[:,data['owner'][f]]*data['normal'][:,f]).sum(0)
        wet_trace=trace[0,f]>p.parameters.dry
        compatible=getattr(ext,'inlet_compatibility',None)
        if compatible is not None:wet_trace &= host(compatible.branch)[f]>0
        exterior_bed=host(p.exterior_bed)
        levels=trace[0,f][wet_trace]+exterior_bed[f][wet_trace]
        weighted_area=trace[0,f][wet_trace]*data['length'][f][wet_trace]
        stage_mean=float(np.average(levels,weights=weighted_area)) if len(levels) else None
        signed_rates=-flux[0,f]*data['length'][f]
        frequency=host(a.midpoint.frequency);limiting=int(np.argmax(frequency))
        record=dict(time_s=a.time,cells=p.n,discharge_m3_s=-float((flux[0,f]*data['length'][f]).sum()),
            gross_inflow_m3_s=float(np.maximum(signed_rates,0.).sum()),
            gross_backflow_m3_s=float(np.maximum(-signed_rates,0.).sum()),
            requested_stage_m=float(ext.forcing[1,0]) if ext.forcing.ndim==2 else float(ext.forcing[ext.nb]),
            mean_trace_stage_m=stage_mean,
            actual_trace_stage_min_m=float(levels.min()) if len(levels) else None,
            actual_trace_stage_max_m=float(levels.max()) if len(levels) else None,
            stage_spread_m=float(np.ptp(levels)) if len(levels) else 0.,
            maximum_frequency_s_inverse=float(frequency[limiting]),limiting_cell=limiting,
            limiting_area_m2=float(data['area'][limiting]),limiting_depth_m=float(h[limiting]),
            maximum_velocity_m_s=float(np.linalg.norm(v,axis=0).max()),maximum_inlet_outward_velocity_m_s=float(normal_speed.max()),
            maximum_depth_m=float(h.max()),wet_area_m2=float(data['area'][h>1e-6].sum()),
            water_m3=float(h@data['area']),inlet_depth_range_m=[float(h[cells].min()),float(h[cells].max())])
        if compatible is not None and hasattr(compatible,'exterior_velocity'):
            record.update(virtual_exterior_velocity_m_s=float(host(compatible.exterior_velocity)[0]),
                boundary_root_iterations=int(host(compatible.iterations)[0,0]),
                maximum_face_riemann_iterations=int(host(compatible.iterations)[1,0]),
                branch_counts={str(i):int(np.count_nonzero(host(compatible.branch)[f]==i)) for i in range(5)})
            if hasattr(compatible,'mass_balance'):
                balance=host(compatible.mass_balance)[:,0]
                record.update(boundary_Q_error_m3_s=float(balance[0]),
                    boundary_gross_flux_m3_s=float(balance[1]),boundary_Q_roundoff_bound_m3_s=float(balance[2]))
        history.append(record);print(json.dumps(record),flush=True)
    began=time.perf_counter()
    status=w.run(prepared,out/'run',on_frame=sample)
    write_json_atomic(out/'QUALIFICATION.json',dict(status=status,wall_s=time.perf_counter()-began,preparation_s=preparation_s,
        runtime=w.source_identity(),history=history,steps=steps))
    assert args.source.read_bytes()==raw
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
