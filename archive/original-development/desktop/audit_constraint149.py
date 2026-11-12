"""Inspect protected coverage on the retained real-project mesh."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_merge149'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'candidate'))
from braids_desktop.preparation import decode_tree
from shapely.geometry import LineString,Polygon,mapping
from shapely.ops import unary_union
tree=decode_tree(json.loads(next((OUT/'user_cpu/model_layers').rglob('TREE.json')).read_text()))
p=json.loads((OUT/'SOURCE.json').read_text())['settings'];rows=[]
for r in p['breaklines']:
    wanted=LineString(r['points_xy']).intersection(Polygon(p['domain']['vertices_xy']))
    got=unary_union([LineString([f.start_xy,f.end_xy]) for f in tree.finest.faces if r['id'] in f.breakline_ids])
    missing=wanted.difference(got.buffer(1e-8))
    print('COVERAGE',wanted.length,got.length,missing.length,flush=True)
    print('BOUNDS',wanted.bounds,got.bounds,'distance',wanted.hausdorff_distance(got),'bufferarea',got.buffer(1e-8).area,flush=True)
    from shapely.affinity import translate
    x,y=wanted.coords[0];print('LOCAL MISSING',translate(wanted,-x,-y).difference(translate(got,-x,-y).buffer(1e-8)).length,flush=True)
    print(missing.wkt,flush=True)
    for f in tree.finest.faces:
        edge=LineString([f.start_xy,f.end_xy])
        if not f.breakline_ids and edge.distance(missing)<1e-7:
            rows.append(dict(id=f.face_id,owner=f.owner,neighbour=f.neighbour,start=f.start_xy,end=f.end_xy,length=f.length_m))
(OUT/'MISSING_CONSTRAINT.json').write_text(json.dumps(dict(missing=mapping(missing),faces=rows),indent=2)+'\n')
