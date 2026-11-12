"""Bounded matched groin campaign, selected-backend processes and captured logs."""
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'validation/_boundaries161/matched_logs'
LOG.mkdir(parents=True,exist_ok=True)
FORMULAS=('mpm','ashida_michiue_generalized','wang_fredsoe')

def sequence(engine,backend='cpu'):
    for formula in FORMULAS:
        label=engine+'_'+formula+'_'+backend
        with (LOG/(label+'.log')).open('x',encoding='utf-8') as log:
            command=[sys.executable,'-W','ignore','-B',str(ROOT/'desktop/check_bedload161.py'),engine,formula,'--backend',backend]
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:raise RuntimeError(label+' failed; inspect log')
        print(label+' complete',flush=True)

if __name__=='__main__':
    # Independent validation processes; not solver hybridism or timing evidence.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(sequence,*case) for case in (('delft','cpu'),('braids','cpu'),('braids','gpu'))]
        for job in jobs:job.result()
