"""Frozen/current unchanged-mesh geometry authentication, no solver stepping."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core.sediment.river_amr_transfer import CellMap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,default=ROOT/'validation/_speed130/production_gpu2m/prepared/core/prepared/cache/19353700b6e88fa1daff/geometry.npz')
    parser.add_argument('--out',type=Path,default=ROOT/'validation/_amr131/GEOMETRY_BENCHMARK.json')
    args=parser.parse_args()
    before=ROOT/'validation/_amr131/source_before/braids_core/sediment/river_amr_transfer.py'
    node=next(n for n in ast.parse(before.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CellMap')
    scope=dict(np=np,dataclass=dataclass,hashlib=hashlib,Polygon=Polygon,STRtree=STRtree)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(before),'exec'),scope)
    original=scope['CellMap']
    with np.load(args.geometry,allow_pickle=False) as data:
        xy=data['vertices_xy'];offsets=data['vertex_offsets'];areas=data['cell_scalar'][:,0]
        polygons=[xy[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    rows=[];maps={}
    for label in ('original','current','current','original'):
        method=original if label=='original' else CellMap
        started=time.perf_counter()
        mapping=method.polygons(polygons,[p.copy() for p in polygons],areas,areas.copy())
        elapsed=time.perf_counter()-started
        rows.append(dict(label=label,wall_s=elapsed));maps[label]=mapping
    equal={name:bool(np.array_equal(getattr(maps['original'],name),getattr(maps['current'],name)))
        for name in ('offsets','indices','weights','source_area','target_area')}
    assert all(equal.values()) and maps['original'].identity==maps['current'].identity
    averages={name:float(np.mean([r['wall_s'] for r in rows if r['label']==name])) for name in maps}
    result=dict(scope='isolated immutable geometry authentication; not worker throughput',
        geometry=str(args.geometry),geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        cells=len(polygons),shapely=shapely.__version__,schedule='ABBA',samples=rows,mean_s=averages,
        speedup=averages['original']/averages['current'],arrays_exact=equal,map_identity=maps['current'].identity,
        source_before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
        source_after_sha256=hashlib.sha256((ROOT/'braids_core/sediment/river_amr_transfer.py').read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
