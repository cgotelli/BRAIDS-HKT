"""Bounded geometry profiling and clean repeated breakline timings."""
import argparse,cProfile,io,json,pstats,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/breaklines57'


def main(variant,large=False,clean=False):
    runtime=ROOT/'desktop/qualification/turbine_live56/candidate' if variant=='baseline' else ROOT
    sys.path.insert(0,str(runtime))
    from braids_core.boundaries.breaklines import build_breakline_hierarchy
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    OUT.mkdir(parents=True,exist_ok=True)
    scale=2 if large else 1;length=96.*scale;width=32.*scale;shift=np.array([2600000.,1200000.])
    domain=PolygonDomain('profile57',(np.array([[0.,0.],[length,0.],[length,width],[0.,width]])+shift).tolist(),'EPSG:2056')
    grid=NativeGrid(int(length/8),int(width/8),*shift,8.,8.)
    rows=[dict(id='Levee '+str(k+1),closed=False,points_xy=(np.array([[0.,y],[length*.25,y+2.25],[length*.625,y-1.75],[length,y+1.125]])+shift).tolist()) for k,y in enumerate([width*.25,width*.75])]
    results=[]
    for repeat in range(3 if clean else 1):
        profiler=cProfile.Profile();profiler.enable() if not clean else None
        start=time.perf_counter();tree=build_breakline_hierarchy(domain,[],grid,rows,levels=4);elapsed=time.perf_counter()-start
        if not clean:
            profiler.disable();profiler.dump_stats(OUT/(variant+('_large' if large else '')+'.prof'))
            stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(32)
            (OUT/(variant+('_large' if large else '')+'_PROFILE.txt')).write_text(stream.getvalue())
        results.append(dict(seconds=elapsed,counts=[len(m.cells) for m in tree.meshes],faces=[len(m.faces) for m in tree.meshes]))
        print(variant,repeat,results[-1],flush=True)
    # Exact serialized geometry for equivalent implementation comparisons.
    from braids_desktop.preparation import encode_tree
    (OUT/(variant+('_large' if large else '')+'_TREE.json')).write_text(json.dumps(encode_tree(tree),separators=(',',':')))
    (OUT/(variant+('_large' if large else '')+('_CLEAN' if clean else '')+'.json')).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['baseline','candidate']);p.add_argument('--large',action='store_true');p.add_argument('--clean',action='store_true');a=p.parse_args();main(a.variant,a.large,a.clean)
