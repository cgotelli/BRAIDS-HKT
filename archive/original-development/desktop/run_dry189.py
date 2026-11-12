"""Bounded public dry-start checks, sequential within each backend."""
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from stage_secondary185 import ROOT,save
OUT=ROOT/'validation/_bend189/dry';OUT.mkdir(parents=True,exist_ok=True)
def sequence(backend):
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        with (OUT/(law+'_'+backend+'.log')).open('x',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(ROOT/'desktop/dry_bedload189.py'),'--backend',backend,'--law',law],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=900)
        if r.returncode:raise RuntimeError(law+' '+backend+' failed')
        print('PASS DRY AMR',law,backend,flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in [pool.submit(sequence,b) for b in ('cpu','gpu')]:job.result()
    save(OUT/'JOBS.json',dict(complete=True,public_runs=8))
