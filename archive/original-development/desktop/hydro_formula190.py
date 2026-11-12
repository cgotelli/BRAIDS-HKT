"""Hydraulics-only field regression, sediment still disabled by default."""
import subprocess,sys,os,json
from stage_formula190 import ROOT,TARGET,OUT
env=dict(os.environ,OMP_NUM_THREADS='4')
for label,runtime in [('baseline',ROOT/'validation/_wall188/desktop_candidate'),('candidate',TARGET)]:
    for backend in ('cpu','gpu'):
        env['BRAIDS_AUDIT_RUNTIME']=str(runtime)
        path=OUT/'hydro'/label/backend;log=OUT/'logs'/f'hydro_{label}_{backend}.log'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'desktop/check_boundaries159.py','--case','zero',
                '--backend',backend,'--output',str(path)],env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        print(label,backend,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(str(log))
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_desktop.results import ResultsStore
import numpy as np
from stage_secondary185 import save
rows={}
for backend in ('cpu','gpu'):
    stores=[ResultsStore(OUT/'hydro'/label/backend/'project/runs/audit') for label in ('baseline','candidate')]
    assert np.array_equal(stores[0].times,stores[1].times)
    errors={k:0. for k in ('q','z','mass','water','bed_boundary')}
    for i in range(len(stores[0].times)):
        a,b=[s.read(i) for s in stores]
        for k in errors:errors[k]=max(errors[k],float(abs(a.arrays[k]-b.arrays[k]).max(initial=0)))
    assert max(errors.values())==0.,errors;rows[backend]=errors
save(OUT/'HYDRO.json',rows)
