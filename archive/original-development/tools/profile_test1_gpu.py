"""Bounded diagnostic replay of a user checkpoint using the installed runtime.

No numerical changes: timing wrappers only. Source inputs are read-only and
the replay writes exclusively into a new validation folder.
"""
import argparse,cProfile,functools,json,os,pstats,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=35.)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(os.environ['APPDATA'])/'QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime'
    sys.path.insert(0,str(root));sys.dont_write_bytecode=True
    from braids_core.river_launcher import verify
    verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((root/'PRECOMPILED.json').read_text()),root)
    from braids_core import river_worker
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.river_exterior_normal import NormalSection
    from braids_core.sediment.river_exterior_bed import ExteriorBed
    from braids_core.sediment.river_exterior_outlet import CriticalOutlet
    import cupy as cp
    prof=cProfile.Profile();clock=[None];rows=[];timings={};stop=out/'STOP'
    def wrap(cls,name,label):
        original=getattr(cls,name)
        @functools.wraps(original)
        def call(self,*a,**kw):
            if label=='single' and clock[0] is None:
                clock[0]=time.perf_counter();prof.enable()
            cp.cuda.get_current_stream().synchronize();began=time.perf_counter();error=None
            before=(self.evaluations,self.matvecs) if label=='single' else None
            try:return original(self,*a,**kw)
            except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
            finally:
                cp.cuda.get_current_stream().synchronize();elapsed=time.perf_counter()-began
                row=timings.setdefault(label,dict(calls=0,wall_s=0.));row['calls']+=1;row['wall_s']+=elapsed
                if label=='single':
                    item=dict(time_s=a[2],dt_s=a[1],wall_s=elapsed,error=error,
                        evaluations=self.evaluations-before[0],matvecs=self.matvecs-before[1])
                    rows.append(item);print(json.dumps(item),flush=True)
                if clock[0] is not None and time.perf_counter()-clock[0]>=args.seconds:
                    stop.touch(exist_ok=True)
        setattr(cls,name,call)
    for cls,name,label in ((RiverMidpoint,'_single','single'),(RiverMidpoint,'rate','rate'),
            (RiverMidpoint,'linear','linear'),(NormalSection,'solve','normal_inlet'),
            (ExteriorBed,'evaluate','exterior'),(CriticalOutlet,'evaluate','critical_outlet')):
        if hasattr(cls,name):wrap(cls,name,label)
    parent=Path('C:/Users/STREEM/Desktop/Test1/runs')
    start=time.perf_counter();result=None
    try:
        result=river_worker.run(parent/'prepare_20261021_162018_0bf405',out/'replay',
            restart=parent/'run_20261021_162057_b146ad/checkpoint.npz',until=.95,stop_file=stop)
    finally:
        prof.disable();prof.dump_stats(str(out/'PROFILE.pstats'))
        with (out/'PROFILE.txt').open('w') as stream:
            pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
        report=dict(runtime=str(root),timings=timings,singles=rows,status=result,
            total_wall_s=time.perf_counter()-start,profile_window_s=None if clock[0] is None else time.perf_counter()-clock[0],
            caution='Nested inclusive synchronized timings; do not sum rows. Diagnostic wrappers add overhead.')
        (out/'PROFILE.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
