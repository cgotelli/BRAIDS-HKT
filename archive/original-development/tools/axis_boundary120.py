"""Bounded whole-cell boundary qualification on explicit private project copies."""
from pathlib import Path
import argparse
import hashlib
import json
from dataclasses import asdict
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_axis_boundary120'


def project():
    from braids_core.boundaries.axis_aligned import derive_axis_aligned
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.river_project import load_project,validate_project
    from braids_gpu.atomic_output import write_json_atomic
    source=Path('C:/Users/STREEM/Documents/BRAIDS/1m.json')
    raw=source.read_bytes();doc=load_project(source);s=doc['settings']
    ratio=2**(s['levels']-1);g=NativeGrid(**s['grid'])
    fine=NativeGrid(g.nx*ratio,g.ny*ratio,g.xlower_m,g.ylower_m,g.dx_m/ratio,g.dy_m/ratio)
    d,a,audit=derive_axis_aligned(PolygonDomain(**s['domain']),
        [BoundaryAssignment(**r) for r in s['assignments']],fine)
    BASE.mkdir(parents=True,exist_ok=False)
    write_json_atomic(BASE/'ORIGINAL.json',doc)
    s['domain']=dict(domain_id=d.domain_id,vertices_xy=d.vertices_xy,spatial_reference=d.crs_wkt)
    s['assignments']=[asdict(r) for r in a]
    doc['name']+=' whole-cell aligned boundary qualification'
    validate_project(doc)
    write_json_atomic(BASE/'ALIGNED.json',doc)
    audit['source_project']=str(source);audit['source_sha256']=hashlib.sha256(raw).hexdigest()
    write_json_atomic(BASE/'GEOMETRY.json',audit)
    print(json.dumps(audit,indent=2))
    assert source.read_bytes()==raw


def prepare(which):
    from braids_core import river_worker
    out=BASE/(which+'_prepared')
    river_worker.prepare(BASE/(which.upper()+'.json'),out)
    print(out)


def deliver():
    """New, labelled project copies only; no installed runtime modification."""
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_gpu.atomic_output import write_json_atomic
    target=Path('C:/Users/STREEM/Documents/BRAIDS/axis_boundary120')
    target.mkdir(parents=True,exist_ok=False)
    paths={}
    for backend in ('cpu','gpu'):
        doc=json.loads((BASE/'ALIGNED.json').read_text())
        doc['name']=f'1 m axis-aligned {backend.upper()} - experimental geometry'
        doc['model']['request']['execution']['backend']=backend
        doc['model']=resolve_model(doc['model']['request']).document()
        validate_project(doc)
        path=target/f'1m_axis_aligned_{backend}.json';write_json_atomic(path,doc)
        paths[backend]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    audit=json.loads((BASE/'GEOMETRY.json').read_text())
    write_json_atomic(target/'GEOMETRY_CHANGES.json',audit)
    record=dict(projects=paths,installed_plugin_changed=False,status='experimental_geometry_not_production_qualification',
        original_unchanged=True,end_time_s=60.,
        limitation='Matched cold-start checks reach 3.75 s. Smaller hydraulic timesteps change local depth by 18.6 mm maximum; remaining nonlinear wet-front restriction is unresolved.',
        report=str(ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/axis_boundary120/REPORT.md'))
    write_json_atomic(target/'READ_ME.json',record);write_json_atomic(BASE/'DELIVERY.json',record)
    print(json.dumps(record,indent=2))


def run(which,backend,until,label,dt,hydraulic_cap=None):
    if hydraulic_cap is not None:
        # Isolated temporal accuracy control; unchanged spatial/midpoint law.
        from braids_core.sediment.river_midpoint import RiverMidpoint
        from braids_core.sediment.bend_methods_v1 import BendTrialRejected
        original=RiverMidpoint._single
        def capped(self,base,dt,at):
            if dt>hydraulic_cap*(1+1e-12):
                raise BendTrialRejected('Qualification hydraulic substep cap')
            return original(self,base,dt,at)
        RiverMidpoint._single=capped
    from tools import profile_test1_amr
    sys.argv=[profile_test1_amr.__file__,'--source',str(BASE/(which+'_prepared')),
        '--output',str(BASE/(f'{which}_{backend}_{label}')),'--backend',backend,
        '--until',str(until),'--checkout']
    if dt is not None:sys.argv+=['--dt-max',str(dt)]
    profile_test1_amr.main()
    if hydraulic_cap is not None:
        from braids_gpu.atomic_output import write_json_atomic
        write_json_atomic(BASE/(f'{which}_{backend}_{label}')/'TEMPORAL_CONTROL.json',
            dict(maximum_hydraulic_substep_s=hydraulic_cap,
                 note='Process-local stricter rejection only, no altered operator or tolerance'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('project','prepare','run','deliver'))
    ap.add_argument('--which',choices=('original','aligned'),default='aligned')
    ap.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    ap.add_argument('--until',type=float,default=3.75);ap.add_argument('--label',default='main')
    ap.add_argument('--dt-max',type=float)
    ap.add_argument('--hydraulic-cap',type=float)
    args=ap.parse_args()
    if args.action=='project':project()
    elif args.action=='prepare':prepare(args.which)
    elif args.action=='deliver':deliver()
    else:run(args.which,args.backend,args.until,args.label,args.dt_max,args.hydraulic_cap)
