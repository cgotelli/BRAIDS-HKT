"""Real Qt settings/results, plus moved-folder selected-checkpoint preflight."""
import json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/_sediment158'
runtime=ROOT/'desktop/runtime' if '--installed' in sys.argv else BASE/'candidate'
sys.path[:0]=[str(runtime),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop.accordion import AccordionSection
from braids_desktop._avac.theme import apply_theme
from braids_desktop import services
from braids_desktop.execution import DesktopRunner

app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
tag='installed' if '--installed' in sys.argv else 'candidate'
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(BASE/('review_settings_'+tag)))
import tempfile
project=Path(tempfile.mkdtemp(prefix='ui_project_'+tag+'_',dir=BASE))/'project';shutil.copytree(BASE/'cpu_final/project',project)
w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w._background=lambda work,done,label:done(work())
w.open_path(project,confirmed=True);w.show();app.processEvents()
out=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_sediment158';out.mkdir(parents=True,exist_ok=True)
for step in ('Domain','Sediment transport','Morpho evolution','Sediment transport BC'):
    w.forms.set_step(step)
    for section in w.forms.findChildren(AccordionSection):section.setExpanded(True)
    app.processEvents();w.forms.grab().save(str(out/(step.replace(' ','_')+'_UI.png')))
    assert w.forms.stack.currentWidget().horizontalScrollBar().maximum()==0
w.results.load_output(project/'runs/first');assert w.results.store is not None;w.set_mode('analyze')
w.results.variable.setCurrentIndex(w.results.variable.findData('bed_change_m'));app.processEvents()
w.results.set_transect([[12,0],[12,8]])
w.results.profile_quantity.setCurrentIndex(w.results.profile_quantity.findData('bedload'))
app.processEvents();w.grab().save(str(out/'DESKTOP_RESULTS.png'))
assert abs(w.results.store.transect([[12,0],[12,8]],quantity='bedload').values).max()>0
cols,rows=w.results.store.measurement_table([[12,0],[12,8]],profile=True)
assert 'bedload_solid_m3_s' in cols
w.layers['Profiles'].setChecked(False);assert not any(r['kind']=='profile' for r in w.map._measurement_features)
# Exercise the UI runner's real identity preflight, no solver replacement.
current=services.load_project_folder(project);current.pop('editor_state',None)
current['settings']['end_time_s']=240.;current['settings']['checkpoint_interval_s']=240.
services.save_project_folder(project,current)
checkpoint=max((project/'checkpoints/first').glob('*.json'),key=lambda p:json.loads(p.read_text())['time_s']).with_suffix('.npz')
runner=DesktopRunner(runtime=runtime);calls=[];runner._launch=lambda *a,**k:calls.append((a,k))
runner.run(project,None,restart=checkpoint)
assert '--continuation-project' in calls[0][0][4] and '--checkpoint-dir' in calls[0][0][4]
assert calls[0][1]['prepared'].resolve()==(project/'prepared/base').resolve()
assert not errors,errors
w.dirty=False;w.close();print('UI, sediment profiles, hidden profiles, moved checkpoint and continuation preflight passed')
