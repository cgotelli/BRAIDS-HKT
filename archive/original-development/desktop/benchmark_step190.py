"""One cold run per process, sequential; execute with no other solver jobs."""
import argparse,os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_step190/clean_timing'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime');ap.add_argument('--backend');ap.add_argument('--label')
    args=ap.parse_args()
    if args.runtime:
        runtime=(ROOT/args.runtime).resolve();sys.path[:0]=[str(runtime),str(ROOT)]
        from braids_core.river_launcher import verify
        from braids_core.sediment.portable_river_runtime import install
        from braids_core import river_worker
        verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
        prepared=ROOT/'validation/_step190/runs'/args.label/('wang_fredsoe_'+args.backend)/'prepared'
        output=OUT/(args.label+'_'+args.backend)
        status=river_worker.run(prepared,output)
        assert status['status']=='complete',status
        print(args.label,args.backend,status['wall_s'],len(status['rejected_trials']),flush=True);return
    os.environ['OMP_NUM_THREADS']='4';OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for backend in ('cpu','gpu'):
        for label,runtime in [('baseline','validation/_wall188/desktop_candidate'),('candidate','validation/_step190/candidate')]:
            path=OUT/(label+'_'+backend+'.log')
            with path.open('w') as stream:
                result=subprocess.run([sys.executable,__file__,'--runtime',runtime,'--backend',backend,'--label',label],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(str(path))
            status=json.loads((OUT/(label+'_'+backend)/'STATUS.json').read_text())
            rows.append(dict(label=label,backend=backend,status=status))
            print(label,backend,status['wall_s'],flush=True)
    (OUT/'TIMINGS.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
