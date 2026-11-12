"""Six bounded packaged paths and separate-process continuation, no long run."""
import subprocess,sys,json
from stage_bundle207 import ROOT,OUT,TARGET,save,sha
logs=OUT/'logs';logs.mkdir(exist_ok=True)
rows=[]
for scenario in ('front','bank','open'):
    for backend in ('cpu','gpu'):
        name=scenario+'_'+backend;case=OUT/'release_v3'/name
        commands=[['desktop/qualify_bundle207.py','--output',str(case),'--runtime',str(TARGET),'--backend',backend,'--scenario',scenario],
                  ['desktop/check_bundle207.py','--runtime',str(TARGET),'--case',str(case)]]
        for i,args in enumerate(commands):
            log=logs/(name+f'_v3_{i}.log')
            with log.open('w',encoding='utf-8') as f:
                result=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            rows.append(dict(command=args,returncode=result.returncode,log=str(log)))
            save(OUT/'CAMPAIGN_v3.json',dict(runtime=sha(TARGET/'RUNTIME.json'),rows=rows))
            if result.returncode:raise RuntimeError(str(log))
        print(name,'run and fresh restart passed',flush=True)
