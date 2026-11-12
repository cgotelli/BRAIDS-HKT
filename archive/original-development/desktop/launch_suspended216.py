"""Exercise installed Desktop's real child preparation/run/checkpoint launch."""
import os,sys,json,subprocess
from pathlib import Path
from stage_suspended216 import ROOT,OUT,BASE,REPORT,sha,save
sys.path[:0]=[str(BASE),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from test_suspended_vr216 import document
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np
app=QApplication([])
folder=OUT/'installed_launch';assert not folder.exists()
folder.mkdir();rows=[]
for backend in ('cpu','gpu'):
    case=folder/backend;case.mkdir()
    doc=document(case/'input',backend,'open',True)
    project=case/'project.json';save(project,doc)
    prefix=[sys.executable,str(ROOT/'desktop/worker.py'),str(BASE/'braids_core/river_launcher.py')]
    env=dict(os.environ,OMP_NUM_THREADS='4')
    env['PYTHONPATH']=str(BASE)+os.pathsep+str(ROOT)
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
save(REPORT/'INSTALLED_LAUNCH.json',dict(runtime=sha(BASE/'RUNTIME.json'),cases=rows))
print('Installed Desktop child preparation, run and restart pass on CPU/GPU',flush=True)
