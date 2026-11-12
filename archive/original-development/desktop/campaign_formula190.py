import os,sys,subprocess
from stage_formula190 import ROOT,OUT
os.environ['OMP_NUM_THREADS']='4'
for dry in (False,True):
    for law in ('ancey_recking_2023','recking_2010'):
        for backend in ('cpu','gpu'):
            p=OUT/'logs'/f'{law}_{backend}_{dry}.log';p.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'desktop/run_formula190.py','--law',law,'--backend',backend]
            if dry:command+=['--dry-amr']
            with p.open('w') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print(law,backend,dry,result.returncode,flush=True)
            if result.returncode:raise RuntimeError(str(p))
