"""Exercise the real Desktop shell and diagnostic map, without a new run."""
import os,sys,time,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
from configuration79 import ROOT,TARGET,REPORT,read,save
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else TARGET
sys.path[:0]=[str(runtime),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase,QFont
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'smoke_settings'))
app=QApplication([])
for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10));apply_theme(app)
project=Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern.braids.json'
original=project.read_bytes()
w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w.show();w.open_path(project,confirmed=True)
until=time.monotonic()+60
while w._tasks and time.monotonic()<until:app.processEvents();time.sleep(.02)
assert not w._tasks and not errors,errors
w._select_step('BC & IC');app.processEvents()
p=w.forms;assert p.initial_mode.currentData()=='stage' and p.stage.value()==502.3
until=time.monotonic()+60
while w._tasks and time.monotonic()<until:app.processEvents();time.sleep(.02)
assert not w._tasks and not errors,errors
assert p.stage.isEnabled() and abs(p.u.value()-.288)<1e-12 and abs(p.v.value()-1.8)<1e-12
p.stack.currentWidget().ensureWidgetVisible(p.v);app.processEvents()
w.grab().save(str(REPORT/'INITIAL_CONDITIONS.png'))
d=p.document();assert d['initial']['stage_m']==502.3
p.stage.setValue(502.5);assert p.document()['initial']['stage_m']==502.5
from braids_desktop.river_physics import set_choice
set_choice(p.initial_mode,'dry');assert p.document()['initial']['velocity_m_s']==[0.,0.]
p.load_document(d)
w._select_step('Hydrokinetic turbines');app.processEvents()
assert 'Near-wake exclusion' in ' '.join(q.text() for q in p.turbines.findChildren(__import__('PySide6.QtWidgets',fromlist=['QLabel']).QLabel))
w.grab().save(str(REPORT/'TURBINE_CONFIGURATION.png'))
w.dirty=False;w.close();app.processEvents();assert project.read_bytes()==original
save(REPORT/('INSTALLED_SMOKE.json' if runtime==ROOT/'desktop/runtime' else 'DELIVERY_SMOKE.json'),dict(passed=True,runtime=str(runtime),project=str(project),initial_conditions_visible=True,wet_dry_edit_roundtrip=True,wake_explanation=True,user_project_unchanged=True,errors=errors))
print('Initial conditions and turbine configuration Desktop smoke passed.')
