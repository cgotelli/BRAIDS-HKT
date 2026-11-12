"""Compare hierarchy overlap arithmetic against exact polygon intersections."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.sediment.adaptive_river_worker import load_prepared
from braids_core.sediment.river_leaf_controller import RiverLeafController
from braids_core.sediment.river_amr_transfer import CellMap

def nested_map(tree,source,target):
    source=tuple(source);target=tuple(target)
    sc=tree.leaf_cover(source);tc=tree.leaf_cover(target)
    sl={n:i for i,n in enumerate(source)};tl={n:i for i,n in enumerate(target)}
    cells=tree.cells
    sa=np.array([cells[n].area_m2 for n in source]);ta=np.array([cells[n].area_m2 for n in target])
    groups=[[] for _ in target]
    for i,n in enumerate(source):
        if tc[n]>=0:groups[tl[int(tc[n])]].append(i)
    offsets=[0];indices=[];weights=[]
    for j,n in enumerate(target):
        ids=[sl[int(sc[n])]] if sc[n]>=0 else groups[j]
        indices.extend(ids);weights.extend(min(sa[i],ta[j])/ta[j] for i in ids);offsets.append(len(indices))
    return CellMap.create(offsets,indices,weights,sa,ta)

def main():
    _,_,t=load_prepared(Path('validation/_test1_gpu_fix/cold_gpu_v1/prepared/core/prepared'))
    fine=tuple(range(t.offsets[-2],t.count));coarse=RiverLeafController.pin_boundaries(t,t.roots)
    rows=[]
    for source,target in ((fine,coarse),(coarse,fine),(coarse,coarse)):
        s=t.select(source);d=t.select(target)
        before=time.perf_counter();ref=CellMap.polygons([c.vertices_xy for c in s.cells],[c.vertices_xy for c in d.cells],
            [c.area_m2 for c in s.cells],[c.area_m2 for c in d.cells]);old=time.perf_counter()-before
        before=time.perf_counter();new=nested_map(t,source,target);elapsed=time.perf_counter()-before
        for k in ('offsets','indices','weights','source_area','target_area'):np.testing.assert_array_equal(getattr(ref,k),getattr(new,k))
        rows.append(dict(source=len(source),target=len(target),polygon_s=old,hierarchy_s=elapsed,speedup=old/elapsed,identical=True))
    out=ROOT/'validation/_test1_gpu_fix/MAP_BENCHMARK.json';out.write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
