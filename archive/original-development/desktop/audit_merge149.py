"""Read-only user-project geometry reproduction; retain local merge diagnostics."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/runtime'))
from braids_core.boundaries import breaklines as b
from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
from braids_core.boundaries.inlet_io import NativeGrid
from shapely.geometry import mapping,Point

OUT=ROOT/'validation/_merge149';OUT.mkdir(exist_ok=True)
source=Path('C:/Users/STREEM/Desktop/0101ttt/River_data/braids_project.json')
doc=json.loads(source.read_text(encoding='utf-8'));p=doc['settings']
(OUT/'SOURCE.json').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
original=b._merge
def traced(polys,faces,barriers,area_floor,tol,stop,keys=None):
    stamp=time.perf_counter();result=original(polys,faces,barriers,area_floor,tol,stop,keys)
    if keys is None:
        shapes,index,groups=result
        small=[i for i,s in enumerate(shapes) if s.area<area_floor*(1-1e-11)]
        print('MERGED',len(polys),len(shapes),'small',len(small),'seconds',time.perf_counter()-stamp,flush=True)
        records=[]
        for i in small:
            neighbors=set()
            for f in faces:
                a=int(index[f.owner]);c=-1 if f.neighbour<0 else int(index[f.neighbour])
                if a!=c and i in (a,c):neighbors.update([a,c])
            rows=[]
            for j in sorted(neighbors-{-1,i}):
                shared=[f for f in faces if f.neighbour>=0 and set((int(index[f.owner]),int(index[f.neighbour])))=={i,j}]
                combined=b._clean(shapes[i].union(shapes[j]))
                rows.append(dict(id=j,area=shapes[j].area,usable=b._usable(combined,tol),
                    protected=any(f.breakline_ids for f in shared),shared=sum(f.length_m for f in shared),geometry=mapping(shapes[j])))
            records.append(dict(id=i,area=shapes[i].area,geometry=mapping(shapes[i]),neighbors=rows))
        (OUT/'FAILURES.json').write_text(json.dumps(dict(origin=[p['grid']['xlower_m'],p['grid']['ylower_m']],small=records),indent=2)+'\n')
        import pickle
        with (OUT/'merge_input.pkl').open('wb') as stream:pickle.dump((polys,faces,barriers,area_floor,tol),stream)
    return result
b._merge=traced
try:b.build_breakline_hierarchy(PolygonDomain(**p['domain']),[BoundaryAssignment(**a) for a in p['assignments']],NativeGrid(**p['grid']),p['breaklines'],levels=p['levels'])
except ValueError as exc:print(type(exc).__name__,str(exc),flush=True)
