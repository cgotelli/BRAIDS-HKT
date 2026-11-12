"""Explicitly labelled research hooks around the current public Test1 workflow.

Original projects, production sources, installed package and reference runs stay
unchanged. Experimental identities are mandatory and saved beside every run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--integration',choices=('implicit','explicit'),required=True)
    ap.add_argument('--merged',action='store_true');ap.add_argument('--fixed',action='store_true')
    ap.add_argument('--threads',type=int,default=8);ap.add_argument('--cfl',type=float,default=.45)
    ap.add_argument('--bedload',action='store_true',help='Explicit private MPM 0.5 mm homogeneous bedload-only contrast')
    extra,remaining=ap.parse_known_args()
    if '--checkout' not in remaining:remaining.append('--checkout')
    out=Path(remaining[remaining.index('--output')+1]).resolve()
    if out.exists():raise FileExistsError(out)
    from braids_core import river_worker as worker
    from braids_core.sediment import adaptive_river_worker as core
    from braids_gpu.atomic_output import write_json_atomic
    from validation.hydraulics.merged_fwave115 import geometry,native,explicit
    native.install()
    if extra.merged:
        trees={}
        def hierarchy(p,stop=None):
            core.check_stop(stop);key=json.dumps([p['domain'],p['assignments'],p['grid'],p['levels']],sort_keys=True)
            if key not in trees:trees[key]=geometry.build(p,stop)
            return trees[key]
        core.hierarchy=hierarchy
    prepare=worker.prepare
    def prepare_research(project,*args,**kwargs):
        path=Path(project);d=json.loads(path.read_text())
        if extra.fixed:d['settings']['adaptive']=False
        d['settings']['controls']['threads']=extra.threads
        from braids_core.river_model import resolve_model,new_bedload_selection
        if extra.bedload:
            s=new_bedload_selection();s['grains'][0]['id']=d['settings']['columns']['fraction_ids'][0]
            s['bed']=dict(composition='bed.layered',transport_layer='transport_layer.constant',coarse_layer=False,sorting='sorting.off')
            d['model']['request']['sediment']=dict(enabled=True,selection=s)
            d['settings']['suspension']['bed_factor']=1.
        d['model']=resolve_model(d['model']['request']).document()
        if extra.bedload:
            from braids_core.river_open_boundary import new_state,resolve
            d['boundary_states']={n:new_state(r['method'],sediment=True) for n,r in d['settings']['hydraulics'].items()}
            d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],
                {r['bc_id']:r['kind'] for r in d['settings']['assignments']},
                sediment=d['model']['physics']['sediment'],ng=1,end_time=d['settings']['end_time_s'])
        write_json_atomic(path,d)
        return prepare(project,*args,**kwargs)
    worker.prepare=prepare_research
    factory=core.factory;identities=[];trials=[];owners=[]
    def create(*args,**kwargs):
        make=factory(*args,**kwargs)
        def construct(*a,**kw):
            result=make(*a,**kw)
            if extra.integration=='explicit':explicit.install(result.solver,extra.cfl)
            result.solver.identity['research_geometry']=dict(merged=extra.merged,
                source_sha256=hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest())
            identities.append(result.solver.identity.copy())
            owners.append(result)
            return result
        return construct
    core.factory=create
    run=core.run
    def research_run(*args,**kwargs):
        result=run(*args,**kwargs)
        result=dict(result,integrator=explicit.METHOD if extra.integration=='explicit' else result['integrator'],
            scope='isolated_merged_fwave115_research',QGIS_admitted=False)
        status=Path(args[1])/'STATUS.json'
        write_json_atomic(status,result)
        return result
    core.run=research_run
    one=explicit.Explicit._single
    def trace(self,q,dt,t):
        import time
        start=time.perf_counter();ev=self.evaluations;error=None
        try:return one(self,q,dt,t)
        except Exception as exc:error=str(exc);raise
        finally:trials.append(dict(time=t,dt=dt,evaluations=self.evaluations-ev,error=error,wall_s=time.perf_counter()-start))
    explicit.Explicit._single=trace
    from tools import profile_test1_amr as profiler
    sys.argv=[str(Path(profiler.__file__)),*remaining]
    try:profiler.main()
    finally:
        if out.exists():
            if owners and extra.integration=='explicit':
                from braids_core.sediment.bed_columns_v101 import host
                from braids_gpu.atomic_output import save_arrays_atomic
                a=owners[-1];m=a.solver.midpoint
                save_arrays_atomic(out/'CFL_FINAL.npz',speed=host(m.speed),frequency=host(m.frequency),
                    owner=a.data['owner'],neighbour=a.data['neighbour'],normal=a.data['normal'],xy=a.data['xy'])
            write_json_atomic(out/'EXPERIMENT.json',dict(research_only=True,settings=vars(extra),
                actual_integrator=explicit.METHOD if extra.integration=='explicit' else 'implicit_fwave_midpoint_river_v1',
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path(geometry.__file__),Path(native.__file__),Path(explicit.__file__),Path(explicit.__file__).with_suffix('.h')]},
                solver_identities=identities,explicit_trials=trials,production_and_plugin_changed=False))

if __name__=='__main__':main()
