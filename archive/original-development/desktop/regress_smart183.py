"""Retained preview28 versus candidate on the three existing bedload formulas."""
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from stage_smart183 import ROOT,OUT,BASE,TARGET,sha,BASE_HASH

def sequence(backend):
    logs=OUT/'regression_logs';logs.mkdir(exist_ok=True)
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe'):
        for name,runtime in (('baseline',BASE),('candidate',TARGET)):
            label=f'{name}_{law}_{backend}'
            with (logs/(label+'.log')).open('x',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,str(ROOT/'desktop/check_smart183.py'),'--backend',backend,
                    '--law',law,'--runtime',str(runtime),'--tag','regression_'+name],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if result.returncode:raise RuntimeError(label+' failed; see log')
            print(label,'complete',flush=True)

if __name__=='__main__':
    assert sha(BASE/'RUNTIME.json')==BASE_HASH,'Restore the verified preview28 reference from its retained files before rerunning this pre-install comparison'
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(sequence,b) for b in ('cpu','gpu')]
        for j in jobs:j.result()
