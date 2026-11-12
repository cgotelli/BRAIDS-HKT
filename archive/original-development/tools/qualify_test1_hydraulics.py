"""Bounded cold Test1 run, preserving user inputs and recording costs."""
import argparse,cProfile,json,pstats,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_gpu.atomic_output import write_json_atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--until',type=float,default=2.)
    ap.add_argument('--seconds',type=float,default=300.)
    ap.add_argument('--dt-max',type=float)
    ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path('C:/Users/STREEM/Desktop/Test1/test1.json')
    doc=json.loads(source.read_text(encoding='utf-8-sig'))
    dem=Path(doc['terrain']['dem_path'])
    if not dem.is_absolute():doc['terrain']['dem_path']=str((source.parent/dem).resolve())
    doc['model']['request']['execution']['backend']=args.backend
    doc['model']=resolve_model(doc['model']['request']).document()
    if args.dt_max is not None:doc['settings']['dt_max_s']=args.dt_max
    write_json_atomic(out/'PROJECT.json',doc)
    before=time.perf_counter();prep=river_worker.prepare(out/'PROJECT.json',out/'prepared')
    preparation=time.perf_counter()-before
    from braids_core.sediment.river_midpoint import RiverMidpoint
    original=RiverMidpoint._single;start=[None];stop=out/'STOP';rows=[];profile=cProfile.Profile()
    def single(self,base,dt,at):
        if start[0] is None:
            start[0]=time.perf_counter()
            if args.profile:profile.enable()
        began=time.perf_counter();ev=self.evaluations;error=None
        try:return original(self,base,dt,at)
        except Exception as exc:error=str(exc);raise
        finally:
            rows.append(dict(time=at,dt=dt,wall_s=time.perf_counter()-began,error=error,evaluations=self.evaluations-ev))
            if time.perf_counter()-start[0]>args.seconds:stop.touch()
    RiverMidpoint._single=single
    result=None;began=time.perf_counter()
    try:
        result=river_worker.run(prep,out/'run',until=args.until,stop_file=stop,
            on_frame=lambda folder,row:print(json.dumps(row),flush=True))
    finally:
        profile.disable()
        write_json_atomic(out/'SUMMARY.json',dict(backend=args.backend,preparation_wall_s=preparation,
            run_wall_s=time.perf_counter()-began,result=result,singles=rows))
        if args.profile:
            profile.dump_stats(str(out/'PROFILE.pstats'))
            with (out/'PROFILE.txt').open('w') as f:pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumulative').print_stats(65)
        print(json.dumps(result),flush=True)
if __name__=='__main__':main()
