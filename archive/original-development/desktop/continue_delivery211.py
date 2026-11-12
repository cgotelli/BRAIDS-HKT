"""Check genuine clock extension from the delivered 20 s checkpoint to 21 s."""
import json,sys,subprocess
from stage_delivery211 import ROOT,OUT,sha,save
from experiment_delivery211 import FOLDER,RUNTIME
sys.path[:0]=[str(RUNTIME),str(ROOT)]
from braids_core.river_worker import load_prepared,validate_continuation
from braids_core.river_project import validate_project
from braids_core import river_folder
from braids_desktop import services
from braids_desktop.results import ResultsStore
import numpy as np
ready='--ready' in sys.argv
suffix='_ready' if ready else ''
prep=FOLDER/('prepared/flume211'+suffix);base=FOLDER/('runs/flume211'+suffix)
assert json.loads((base/'STATUS.json').read_text())['status']=='complete'
doc,receipt=load_prepared(prep);doc['settings']['end_time_s']=21.
continuation=FOLDER/('continuation_check'+suffix+'.json')
assert not continuation.exists();save(continuation,validate_project(doc))
validate_continuation(load_prepared(prep)[0],doc)
checkpoint=FOLDER/('checkpoints/flume211'+suffix+'/t_20_step_00004000.npz');assert checkpoint.is_file()
output=FOLDER/('runs/continuation21'+suffix);assert not output.exists()
command=[sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py'),'run',
 '--input',str(prep),'--output',str(output),'--restart',str(checkpoint),'--continuation-project',str(continuation),
 '--checkpoint-dir',str(FOLDER/('checkpoints/continuation21'+suffix))]
with (FOLDER/('logs/continuation21'+suffix+'.log')).open('x',encoding='utf-8') as log:
    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,result.returncode
a,b=ResultsStore(base),ResultsStore(output)
f,g=a.read(-1),b.read(0)
exact={k:bool(np.array_equal(f.arrays[k],g.arrays[k])) for k in f.arrays}
assert all(exact.values()),exact
s=json.loads((output/'STATUS.json').read_text());assert s['status']=='complete' and s['time_s']==21.
for action,run,log in ([('run','runs/continuation21_ready','logs/continuation21_ready.log')] if ready else [('prepare','prepared/flume211','logs/prepare.log'),('run','runs/flume211','logs/run.log'),('run','runs/continuation21','logs/continuation21.log')]):
    river_folder.record_event(FOLDER,'desktop_worker',action=action,status='complete',output=run,prepared='prepared/flume211'+suffix,log=log,backend='gpu')
# Show the whole 0-20 s evolution when opening, not just the one-second proof.
services.update_project_session(FOLDER,prepared='prepared/flume211'+suffix,output='runs/flume211'+suffix,log='logs/run_ready.log' if ready else 'logs/run.log')
save(OUT/('CONTINUATION_READY.json' if ready else 'CONTINUATION.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),checkpoint=str(checkpoint),start_fields_exact=exact,status=s,
    original_end_s=20.,extended_end_s=21.,active_project_end_s=20.))
print('Installed checkpoint 20 -> 21 s passed; restart initial fields exact',flush=True)
