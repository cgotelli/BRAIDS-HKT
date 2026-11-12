"""Isolate geometry handoff on the actual prepared mesh, without disk or stepping."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_mesh138/HANDOFF.json')
    a=parser.parse_args()
    runtime=ROOT/'validation/_mesh138/candidate2'
    sys.path.insert(0,str(runtime))
    from braids_core.river_launcher import verify
    manifest=verify()
    import numpy as np
    from braids_core.sediment.adaptive_river_worker import hierarchy
    from braids_core.sediment.river_leaf_controller import RiverLeafController
    spec=importlib.util.spec_from_file_location('braids_core.sediment._mesh138_old',
        ROOT/'validation/_geometry137/candidate3/braids_core/sediment/river_leaf_controller.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    project=json.loads((ROOT/'validation/_mesh138/after_cpu/prepared/core/prepared/profile.json').read_text())
    tree=hierarchy(project);leaves=tuple(range(tree.offsets[-2],tree.count))
    def make(ls,mesh):
        return SimpleNamespace(polygon=SimpleNamespace(polygons=tuple(np.array(c.vertices_xy) for c in mesh.cells)),
            columns=SimpleNamespace(area=np.array([c.area_m2 for c in mesh.cells])))
    rows=[]
    for repeat in range(6):
        for variant in ('before','after','after','before'):
            gc.collect();start=time.perf_counter()
            if variant=='before':controller=old.RiverLeafController(tree,leaves,make(leaves,tree.select(leaves)))
            else:controller=RiverLeafController.from_factory(tree,leaves,make)
            elapsed=time.perf_counter()-start
            assert controller.leaves==leaves
            rows.append(dict(repeat=repeat,variant=variant,elapsed_s=elapsed))
            del controller
    # First ABBA is a declared warm-up; retain its raw timings as evidence.
    summary={v:statistics.median(r['elapsed_s'] for r in rows if r['variant']==v and r['repeat']>0)
             for v in ('before','after')}
    result=dict(cells=len(leaves),rows=rows,median_s=summary,
        reduction_percent=100*(1-summary['after']/summary['before']),
        scope='Backend-independent initial geometry and cell-order admission only; no solver construction, I/O or timestepping; first ABBA warm-up excluded')
    assert verify()==manifest
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
