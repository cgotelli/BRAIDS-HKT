"""Actual Desktop child process run and restart on the selected package."""
import os,sys,subprocess
from stage_suspended218 import ROOT,OUT,TARGET,BASE,REPORT,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from test_suspended_dl218 import document
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np
app=QApplication([])
label='INSTALLED_LAUNCH' if '--installed' in sys.argv else 'CANDIDATE_LAUNCH'
folder=OUT/label;assert not folder.exists();folder.mkdir();rows=[]
for backend in ('cpu','gpu'):
    case=folder/backend;case.mkdir()
    doc=document(case/'input',backend)
    project=case/'project.json';save(project,doc)
    prefix=[sys.executable,str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH=str(RUNTIME)+os.pathsep+str(ROOT))
    def invoke(label,args):
        with (case/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(prefix+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode==0,(label,case/(label+'.log'))
    invoke('prepare',['prepare','--project',str(project),'--output',str(case/'prepared'),'--asset-cache',str(case/'assets')])
    invoke('run',['run','--input',str(case/'prepared'),'--output',str(case/'run'),'--checkpoint-dir',str(case/'checkpoints')])
    checkpoint=next((case/'checkpoints').glob('t_0.02_*.npz'))
    invoke('restart',['run','--input',str(case/'prepared'),'--output',str(case/'resumed'),'--restart',str(checkpoint)])
    a,b=read_live_frame(case/'run')[2],read_live_frame(case/'resumed')[2]
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    rows.append(dict(backend=backend,prepare=True,run=True,exact_restart=True,folder=str(case.relative_to(ROOT))))
save(OUT/(label+'.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),cases=rows))
print(label,'Desktop child prepare/run/restart pass on CPU/GPU',flush=True)
