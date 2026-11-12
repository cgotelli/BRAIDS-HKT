"""Bound the choice of serial shared-index construction on this workstation."""
from concurrent.futures import ThreadPoolExecutor
import gc,json,statistics,sys,time
from pathlib import Path
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55';sys.path.insert(0,str(OUT/'candidate'))
from braids_desktop.preparation import decode_tree


def batch(cells):return [Polygon(c.vertices_xy,getattr(c,'holes_xy',())) for c in cells]


def main():
    for path in (ROOT/'desktop/qualification/decimal_mesh54/assets/geometry').glob('*/TREE.json'):
        raw=json.loads(path.read_text())
        if len(raw['meshes'][-1]['cells'])==36000:break
    else:raise RuntimeError('Retained decimal-grid fixture missing')
    cells=decode_tree(raw).finest.cells;chunks=[cells[i:i+1024] for i in range(0,len(cells),1024)];result={}
    for workers in [1,2,4,8]:
        values=[]
        for trial in range(3):
            gc.collect();start=time.perf_counter()
            if workers==1:polys=batch(cells)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:polys=[p for part in pool.map(batch,chunks) for p in part]
            values.append(time.perf_counter()-start);assert len(polys)==len(cells);del polys
        result[str(workers)]=dict(seconds=values,median_s=statistics.median(values));print(workers,result[str(workers)],flush=True)
    (OUT/'GEOMETRY_THREAD_BENCHMARK.json').write_text(json.dumps(dict(cells=len(cells),batch_size=1024,results=result,scope='polygon construction only; identical ordered batching; other shared-index work unchanged'),indent=2)+'\n')


if __name__=='__main__':main()
