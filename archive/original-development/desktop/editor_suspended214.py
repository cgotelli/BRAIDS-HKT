"""Real Qt S1 tabs, narrow layout, saved pending draft and bedload projection."""
import os,sys,json
from stage_suspended214 import ROOT,OUT,TARGET,BASE,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests'),str(ROOT/'desktop')]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS214Qualification');app.setApplicationName('SuspendedSetup')
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_desktop._avac.theme import apply_theme
from braids_desktop import editor_state,workspace
from braids_core.river_project import validate_project
from test_jiang_cao207 import project
from test_suspended_setup214 import fill,DL
apply_theme(app,'light')
f=SettingsPanel();f.resize(380,900);f.show()
doc=project(OUT/'editor_case',opened=True)
f.load_document(doc);runnable=f.document(doc)
assert runnable['model']==doc['model'] and runnable['settings']==doc['settings']
g=SettingsPanel();g.load_document(validate_project(json.loads(json.dumps(runnable))))
assert g.document(runnable)['model']==runnable['model'];g.close()
set_choice(f.physics.capacity,'capacity.recking_2010');fill(f.physics)
before=f.physics.model('cpu','roughness.manning',False,False)
f.physics.suspension_on.setChecked(True);set_choice(f.physics.suspended_formula,DL);fill(f.physics)
root=OUT/'saved_draft'
ui=editor_state.capture(f,root)
workspace.save_editor(root,doc,ui,{},has_domain=True,has_terrain=True,issue='S1 suspension qualification pending')
loaded=workspace.read_editor(root)
assert loaded['run_project_id'] is None
try:workspace.check_runnable(root,loaded['document'])
except ValueError:pass
else:raise AssertionError('Pending draft must not run')
g=SettingsPanel();editor_state.restore(g,loaded['ui'],root)
assert g.physics.transport_setup()==f.physics.transport_setup()
g.close()
f.set_step(f.step_names.index('Sediment transport'))
images=OUT/('installed_figures' if '--installed' in sys.argv else 'figures');images.mkdir(exist_ok=True)
pages=[]
for index in range(f.sediment_tabs.count()):
    f.sediment_tabs.setCurrentIndex(index)
    for _ in range(6):app.processEvents()
    scroll=f.stack.currentWidget()
    scroll.verticalScrollBar().setValue(0)
    assert scroll.horizontalScrollBar().maximum()==0,(index,scroll.horizontalScrollBar().maximum())
    name=f.sediment_tabs.tabText(index)
    assert f.grab().save(str(images/(name.lower().replace(' ','_')+'.png')))
    pages.append(name)
f.physics.suspension_on.setChecked(False)
assert f.physics.model('cpu','roughness.manning',False,False)==before
f.close()
save(OUT/('INSTALLED_EDITOR.json' if '--installed' in sys.argv else 'EDITOR.json'),dict(
    runtime=sha(RUNTIME/'RUNTIME.json'),tabs=pages,width_px=380,theme='light',
    pending_draft_saved_and_reopened=True,pending_draft_blocked=True,
    bedload_document_unchanged=True,existing_project_roundtrip=True,
    render_scope='Real Qt offscreen components; no live mouse automation'))
print('Four tabs, 380px layout, draft roundtrip and unchanged bedload projection passed',flush=True)
