"""Bounded public CPU/GPU qualification; separate backend processes, no fallback."""
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from stage_secondary185 import ROOT,save
OUT=ROOT/'validation/_wall188'


def execute(label,script,args):
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(ROOT/'desktop'/script),*args],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=7200)
    if result.returncode:raise RuntimeError(label+' failed; see its retained log')
    print('PASS',label,flush=True)


def sequence(backend):
    for dry in (False,True):
        label=('dry_' if dry else 'wet_')+backend
        args=['--backend',backend]+(['--dry'] if dry else [])
        execute(label,'qualify_wall188.py',args)
        execute(label+'_restart','qualify_wall188.py',args+['--restart'])
    execute('amr_'+backend,'smoke_wall188.py',['--backend',backend,'--dry-amr','--secondary','--tag','amr'])
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        execute(law+'_'+backend,'smoke_wall188.py',['--backend',backend,'--law',law,'--tag','fixed'])


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in [pool.submit(sequence,b) for b in ('cpu','gpu')]:job.result()
    save(OUT/'JOBS.json',dict(complete=True,performance_comparison=False,concurrent_backends=True))
