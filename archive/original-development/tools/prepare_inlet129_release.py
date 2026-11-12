"""Derive explicit new release qualification decks; never migrate user inputs.

The frozen alpha127 validation decks remain unchanged. Every changed JSON value
and any deliberately realigned open-section geometry is recorded. No solver or
binding-recording run is started by this tool.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core.river_model import INTEGRATOR,resolve_model
from braids_core.river_project import SCHEMA,validate_project
from braids_core.river_open_boundary import resolve as resolve_boundaries
from braids_core.boundaries.axis_aligned import derive_axis_aligned,finest_grid,validate_open_grid
from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from braids_gpu.atomic_output import write_json_atomic


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(old,new,path=''):
    if isinstance(old,dict) and isinstance(new,dict):
        result=[]
        for key in sorted(set(old)|set(new)):
            name=path+'/'+key
            if key not in old:result.append(dict(path=name,before=None,after=new[key]))
            elif key not in new:result.append(dict(path=name,before=old[key],after=None))
            else:result.extend(differences(old[key],new[key],name))
        return result
    return [] if old==new else [dict(path=path,before=old,after=new)]


def derive(original):
    d=deepcopy(original);d['schema']=SCHEMA
    request=d['model']['request'];request['execution']['integrator']=INTEGRATOR
    d['model']=resolve_model(request).document()
    s=d['settings']
    for row in s['hydraulics'].values():
        if row['method']=='inlet.normal_discharge':row['energy_slope']='mesh'
    domain=PolygonDomain(**s['domain']);assignments=tuple(BoundaryAssignment(**v) for v in s['assignments'])
    grid=finest_grid(NativeGrid(**s['grid']),s['levels']);alignment=None
    try:validate_open_grid(domain,assignments,grid)
    except ValueError as exc:
        new,mapped,alignment=derive_axis_aligned(domain,assignments,grid)
        alignment['qualification_reason']=str(exc)
        s['domain']=dict(domain_id=new.domain_id,vertices_xy=new.vertices_xy,spatial_reference=new.crs_wkt)
        s['assignments']=[asdict(v) for v in mapped]
    kinds={a['bc_id']:a['kind'] for a in s['assignments'] if a['kind']!='wall'}
    d['boundary_resolution']=resolve_boundaries(s['hydraulics'],d['boundary_states'],kinds,
        sediment=d['model']['physics']['sediment'],ng=len(s['columns']['fraction_ids']),end_time=s['end_time_s'])
    d=json.loads(json.dumps(d))
    validate_project(d)
    return d,alignment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'validation/_delivery127/projects')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_inlet129/release_projects')
    args=parser.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    if not source.is_relative_to(ROOT/'validation') or not out.is_relative_to(ROOT/'validation') or source==out:
        raise ValueError('This tool only derives separate repository validation decks')
    files=sorted(source.glob('*.json'))
    if not files:raise ValueError('No frozen qualification inputs')
    out.mkdir(parents=True,exist_ok=True);records=[]
    for path in files:
        original_bytes=path.read_bytes();original=json.loads(original_bytes)
        doc,alignment=derive(original)
        target=out/path.name
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8'))!=doc:
                raise ValueError('Refusing to replace differing qualification deck: '+str(target))
        else:write_json_atomic(target,doc)
        dem=Path(doc['terrain']['dem_path'])
        if not dem.is_absolute():dem=path.parent/dem
        records.append(dict(source=str(path),source_sha256=hashlib.sha256(original_bytes).hexdigest(),
            target=str(target),target_sha256=sha(target),changes=differences(original,doc),alignment=alignment,
            dem_exists=dem.is_file(),backend=doc['model']['request']['execution']['backend']))
        if path.read_bytes()!=original_bytes:raise ValueError('Frozen validation source changed')
    result=dict(purpose='Explicit new v19 release qualification inputs; not user-project migration',
        schema=SCHEMA,integrator=INTEGRATOR,source=str(source),output=str(out),
        inputs=records,counts=dict(total=len(records),cpu=sum(r['backend']=='cpu' for r in records),
            gpu=sum(r['backend']=='gpu' for r in records),geometry_derived=sum(r['alignment'] is not None for r in records),
            missing_dem=sum(not r['dem_exists'] for r in records)),
        recording_started=False,clock_overrides=False,physics_changes='Normal-depth slopes now prepared from the actual finest mesh; method and geometry changes explicit above')
    write_json_atomic(out.parent/'RELEASE_INPUT_PROVENANCE.json',result)
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
