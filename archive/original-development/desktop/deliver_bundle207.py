"""Two self-contained examples through the actual installed Desktop child CLI."""
import sys,subprocess,json
from stage_bundle207 import ROOT,OUT,TARGET,sha,save,verify
runtime=ROOT/'desktop/runtime';assert sha(runtime/'RUNTIME.json')==sha(TARGET/'RUNTIME.json');verify(runtime)
sys.path[:0]=[str(runtime),str(ROOT)]
from braids_desktop import services
from braids_desktop.results import ResultsStore
import numpy as np
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207'
rows=[]
for scenario,backend,name in [('front','gpu','moving_front'),('bank','cpu','protected_bank')]:
    source=OUT/'release_v3'/(scenario+'_'+backend)
    doc=json.loads((source/'project.json').read_text());doc['name']='Jiang–Cao research — '+name.replace('_',' ')
    folder=ROOT/'desktop/examples/jiang_cao207'/name;assert not folder.exists()
    services.save_project_folder(folder,doc,source_base=source)
    prep=folder/'prepared/qualification207';run=folder/'runs/qualification207';checkpoints=folder/'checkpoints'
    commands=[['prepare','--project',str(folder/'braids_project.json'),'--output',str(prep),'--asset-cache',str(folder/'cache')],
        ['run','--input',str(prep),'--output',str(run),'--checkpoint-dir',str(checkpoints)]]
    for i,command in enumerate(commands):
        with (OUT/'logs'/('installed_'+name+f'_{i}.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(runtime/'braids_core/river_launcher.py'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Installed worker failed: '+name)
    a=ResultsStore(run).read(-1);b=ResultsStore(source/'run').read(-1)
    same={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in a.arrays}
    assert all(same.values()),same
    rows.append(dict(project=str(folder/'braids_project.json'),run=str(run),backend=backend,fields_exact=same))
    print(name,'installed child prepare/run exact',flush=True)
save(REPORT/'DELIVERY.json',dict(runtime=sha(runtime/'RUNTIME.json'),examples=rows))
