"""Clean geometry/mapping profiles; fixed inputs, isolated runtime variants."""
import argparse,cProfile,io,json,math,pstats,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55'


def run(variant,level_study=False,clean=False):
    runtime=ROOT/('desktop/qualification/decimal_mesh54/candidate' if variant=='baseline' else 'desktop/qualification/turbine_array55/candidate')
    sys.path.insert(0,str(runtime))
    from braids_core import river_turbines
    from braids_core.river_project import worker_profile
    from braids_core.sediment.adaptive_river_worker import hierarchy
    OUT.mkdir(parents=True,exist_ok=True)
    doc=json.loads((ROOT/'desktop/qualification/decimal_mesh54/qualified_cpu/on/input.json').read_text());result={}
    # Warm imports, not geometry. Timings remain sequential and use the same
    # source grid, terrain-independent mapping bed, and process conditions.
    cases=[(level,name) for level in ([1,4] if level_study else [2]) for name in (['off','one'] if level_study else ['off','one','three'])]
    for level,mode in cases:
        name=('level'+str(level)+'_' if level_study else '')+mode
        p=deepcopy(doc)
        coarse=.1*2**(level-1);p['settings']['levels']=level;p['settings']['grid'].update(dx_m=coarse,dy_m=coarse,nx=math.ceil(30/coarse),ny=math.ceil(12/coarse))
        if mode=='off':p['turbines']=None
        elif mode=='three':
            first=p['turbines']['turbines'][0];p['turbines']['turbines']=[dict(deepcopy(first),id='Rotor '+str(k+1),center_xy=xy) for k,xy in enumerate([[2600010.,1200005.],[2600010.,1200007.],[2600018.,1200006.]])]
        profile=worker_profile(p);profiler=cProfile.Profile()
        if not clean:profiler.enable()
        start=time.perf_counter()
        tree=hierarchy(profile);geometry=time.perf_counter()-start;stamp=time.perf_counter()
        mapped=river_turbines.prepare(p['turbines'],tree,np.zeros(len(tree.finest.cells)))
        mapping=time.perf_counter()-stamp
        if not clean:
            profiler.disable();profiler.dump_stats(OUT/(variant+'_'+name+'.prof'));stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(28)
            (OUT/(variant+'_'+name+'_PROFILE.txt')).write_text(stream.getvalue())
        result[name]=dict(geometry_s=geometry,turbine_mapping_s=mapping,total_s=geometry+mapping,finest_cells=len(tree.finest.cells),pinned=0 if mapped is None else len(mapped['required_leaves']))
        print(variant,name,result[name],flush=True)
    (OUT/(variant+('_LEVELS' if level_study else '')+('_CLEAN' if clean else '')+'_PROFILE.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','candidate'],required=True);p.add_argument('--level-study',action='store_true');p.add_argument('--clean',action='store_true');args=p.parse_args();run(args.variant,args.level_study,args.clean)
