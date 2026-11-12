"""Create two explicitly changed, frozen research input decks; no user edits."""
from pathlib import Path
from dataclasses import asdict
import copy
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from braids_core.river_project import validate_project
    from braids_core.boundaries.axis_aligned import derive_axis_aligned
    from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.river_worker import source_identity
    from braids_gpu.atomic_output import write_json_atomic
    base=ROOT/'validation/_explicit_fwave124';base.mkdir(exist_ok=False)
    source=ROOT/'validation/_open_boundary123/projects/River1m_cpu.json'
    original=ROOT/'validation/_axis_boundary120/ORIGINAL.json'
    doc=json.loads(source.read_text());old=json.loads(original.read_text())
    records=[]
    for size in (1,5):
        d=copy.deepcopy(doc);s=d['settings']
        if size==5:
            s['domain']=copy.deepcopy(old['settings']['domain'])
            s['assignments']=copy.deepcopy(old['settings']['assignments'])
            xy=s['domain']['vertices_xy'];dx=20.
            x0=math.floor(min(v[0] for v in xy)/dx)*dx;y0=math.floor(min(v[1] for v in xy)/dx)*dx
            s['grid']=dict(nx=math.ceil((max(v[0] for v in xy)-x0)/dx),ny=math.ceil((max(v[1] for v in xy)-y0)/dx),
                           xlower_m=x0,ylower_m=y0,dx_m=dx,dy_m=dx)
            g=NativeGrid(**s['grid']);fine=NativeGrid(g.nx*4,g.ny*4,x0,y0,5.,5.)
            domain,assignments,audit=derive_axis_aligned(PolygonDomain(**s['domain']),
                [BoundaryAssignment(**v) for v in s['assignments']],fine)
            s['domain']=dict(domain_id=domain.domain_id,vertices_xy=domain.vertices_xy,spatial_reference=domain.crs_wkt)
            s['assignments']=[asdict(v) for v in assignments]
            write_json_atomic(base/'5m_geometry_change.json',audit)
        s.update(end_time_s=4.,output_interval_s=.5,dt_max_s=.5,adaptive=False)
        d['name']=f'RESEARCH explicit124 matched {size} m';validate_project(d)
        folder=base/f'input_{size}m';(folder/'core').mkdir(parents=True)
        write_json_atomic(folder/'PROJECT.json',d)
        write_json_atomic(folder/'core/RECIPE.json',dict(dem_path=d['terrain']['dem_path']))
        records.append(dict(size_m=size,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            project=str(folder/'PROJECT.json'),project_sha256=hashlib.sha256((folder/'PROJECT.json').read_bytes()).hexdigest(),
            changed_clocks=dict(end=4.,output=.5,outer_cap=.5),static_terrain=True,
            geometry_change='none from alpha123 River1m' if size==1 else 'same original polygon; open sections explicitly aligned on 5 m lattice'))
    release=ROOT/'dist/unified-alpha123-final/BRAIDS-0.1.0-alpha.123-QGIS-windows-x86_64.zip'
    write_json_atomic(base/'INPUTS.json',dict(runtime_id=source_identity(),inputs=records,
        frozen_release=str(release),release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),
        acceptance='Few mm depth differences acceptable; report velocity, wet-front, discharge and roundoff conservation separately. No remeshing, sediment or production adoption in this milestone.'))
    print(base)


if __name__=='__main__':main()
