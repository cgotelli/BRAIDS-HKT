"""Project-specific editor roundtrip, independent of Windows visual automation."""
import sys,os,json
from stage_delivery211 import ROOT,OUT,save
from experiment_delivery211 import FOLDER,RUNTIME
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'editor_preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS211Qualification');app.setApplicationName('ProjectRoundtrip')
from braids_desktop.forms import SettingsPanel
from braids_desktop import services
from braids_core.river_worker import load_prepared,validate_continuation
p=SettingsPanel();assert not p.sediment_enabled.isChecked()
doc=services.load_project_folder(FOLDER);p.load_document(doc);got=p.document(doc)
changes=[]
def walk(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in a.keys()|b.keys():walk(a.get(k),b.get(k),path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
    elif a!=b:changes.append(dict(path=path,before=a,after=b))
walk(doc,got)
save(OUT/'EDITOR_ROUNDTRIP_READY.json',dict(changes=changes))
print(json.dumps([x for x in changes if x['path']!='/editor_state'],indent=2))
original=load_prepared(FOLDER/'prepared/flume211_ready')[0]
# The folder saver canonicalizes its absolute editor asset back to a contained
# relative path before DesktopRunner reads the saved document.
from copy import deepcopy
def persisted(value):
    value=deepcopy(value)
    from pathlib import Path
    asset=Path(value['terrain']['dem_path'])
    if asset.is_absolute():value['terrain']['dem_path']=asset.relative_to(FOLDER).as_posix()
    return value
got=persisted(got)
validate_continuation(original,got)
p.end_time.setValue(40.);extended=p.document(doc)
assert validate_continuation(original,persisted(extended))==40.
save(OUT/'EDITOR_CHECK.json',dict(current_continuation=True,extended_end_s=40.,physical_changes=False,default_sediment=False))
p.close()
print('Delivered editor preserves physics and allows 40 s continuation',flush=True)
