"""Sequential, isolated clean timings; no concurrent CPU/GPU benchmark."""
import subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='4'
for tag,runtime,laws in (
    ('baseline','validation/_wall188/desktop_candidate',('wang_fredsoe',)),
    ('candidate','validation/_step190/candidate',('mpm','ashida_michiue_generalized','smart_jaggi'))):
    for law in laws:
        for backend in ('cpu','gpu'):
            command=[sys.executable,'desktop/run_step190.py','--runtime',runtime,'--tag',tag,
                     '--law',law,'--backend',backend,'--dry-amr']
            p=ROOT/'validation/_step190/logs'/f'{tag}_{law}_{backend}.log'
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(tag,law,backend,result.returncode,flush=True)
            if result.returncode:raise RuntimeError(str(p))
