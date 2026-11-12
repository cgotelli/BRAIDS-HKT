"""Real installed child-worker replay on both explicitly chosen backends."""
import sys,json,subprocess
from stage_delivery211 import ROOT,OUT,TARGET,sha,save,verify
runtime=ROOT/'desktop/runtime';verify(runtime);assert sha(runtime/'RUNTIME.json')==sha(TARGET/'RUNTIME.json')
sys.path[:0]=[str(runtime),str(ROOT)]
from braids_desktop.results import ResultsStore
import numpy as np
checks=[]
for backend in ('cpu','gpu'):
    source=OUT/('oblique_'+backend);output=OUT/('installed_'+backend)
    assert not output.exists()
    with (OUT/('installed_'+backend+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(runtime/'braids_core/river_launcher.py'),'run','--input',str(source/'prepared'),'--output',str(output),'--checkpoint-dir',str(OUT/('installed_checkpoints_'+backend))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,backend
    a=ResultsStore(source/'run');b=ResultsStore(output)
    np.testing.assert_array_equal(a.times,b.times)
    for i in range(len(a.times)):
        f,g=a.read(i),b.read(i)
        for k in f.arrays:np.testing.assert_array_equal(f.arrays[k],g.arrays[k],err_msg=backend+' '+k)
    checks.append(dict(backend=backend,all_frames_exact=True,frames=len(a.times)))
    print(backend,'installed child all frames exact',flush=True)
save(OUT/'INSTALLED_CHECK.json',dict(runtime=sha(runtime/'RUNTIME.json'),checks=checks))
