"""Canonicalize this generated example through the actual editor before running."""
import os,sys,json,subprocess,time
from stage_delivery211 import ROOT,OUT,sha,save
from experiment_delivery211 import FOLDER,RUNTIME
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(RUNTIME),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'editor_preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS211Qualification');app.setApplicationName('ProjectRoundtrip')
from braids_desktop import services
from braids_desktop.forms import SettingsPanel
from braids_core import river_folder
from braids_core.river_worker import validate_continuation
from braids_desktop.results import ResultsStore
import numpy as np

doc=services.load_project_folder(FOLDER);p=SettingsPanel();p.load_document(doc)
ready=p.document(doc);p.load_document(ready);twice=p.document(ready)
assert validate_continuation(ready,twice)==20.
save(OUT/'GENERATED_INPUT_BEFORE_EDITOR.json',doc)
save(OUT/'GENERATED_INPUT_FROM_EDITOR.json',ready)
services.save_project_folder(FOLDER,ready,source_base=FOLDER,session={'desktop':{'advanced_hydrodynamics':True}})
p.close()
prep=FOLDER/'prepared/flume211_ready';output=FOLDER/'runs/flume211_ready'
assert not prep.exists() and not output.exists()
timings={}
for command in [
 ['prepare','--project',str(FOLDER/'braids_project.json'),'--output',str(prep),'--asset-cache',str(FOLDER/'cache')],
 ['run','--input',str(prep),'--output',str(output),'--checkpoint-dir',str(FOLDER/'checkpoints/flume211_ready')]]:
    action=command[0];logname='logs/'+action+'_ready.log';start=time.perf_counter()
    print('Editor-generated installed',action,flush=True)
    with (FOLDER/logname).open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'desktop/worker.py'),str(RUNTIME/'braids_core/river_launcher.py'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    timings[action]=time.perf_counter()-start
    assert result.returncode==0,logname
    river_folder.record_event(FOLDER,'desktop_worker',action=action,status='complete',prepared='prepared/flume211_ready',
      output=(prep if action=='prepare' else output).relative_to(FOLDER).as_posix(),log=logname,backend='gpu')
    services.update_project_session(FOLDER,prepared='prepared/flume211_ready',**({'output':'runs/flume211_ready','log':logname} if action=='run' else {}))
a=ResultsStore(FOLDER/'runs/flume211');b=ResultsStore(output)
np.testing.assert_array_equal(a.times,b.times)
for i in range(len(a.times)):
    f,g=a.read(i),b.read(i)
    for k in f.arrays:np.testing.assert_array_equal(f.arrays[k],g.arrays[k],err_msg=k)
save(OUT/'READY_EXPERIMENT.json',dict(runtime=sha(RUNTIME/'RUNTIME.json'),all_frames_exact_to_first_run=True,
    timings_s=timings,prepared=str(prep),run=str(output),editor_stable=True))
print('Delivered UI-generated run complete and all fields identical',flush=True)
