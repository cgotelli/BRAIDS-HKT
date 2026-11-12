"""Repeatable assembled-window qualification without changing user projects."""
from pathlib import Path
import os
import sys
import time
import json
sys.dont_write_bytecode=True

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent));sys.path.insert(0,str(BASE/'runtime'))
os.environ['QT_QPA_PLATFORM']='offscreen'
_dll=[]
if os.name=='nt':
    for directory in (Path(sys.base_prefix)/'Library/bin',Path(sys.base_prefix)):
        if directory.is_dir():
            _dll.append(os.add_dll_directory(str(directory)))
            os.environ['PATH']=str(directory)+os.pathsep+os.environ['PATH']
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop.demo import create_demo
from braids_desktop._avac.theme import apply_theme

app=QApplication([]);app.setApplicationName('BRAIDS Desktop tests')
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(BASE/'evidence/ui142/settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app)
root=BASE/'test_projects'/('assembled_'+time.strftime('%Y%m%d_%H%M%S'))
create_demo(root,end_time_s=5.)
window=MainWindow(runtime=BASE/'runtime');errors=[];window.error=lambda value:errors.append(str(value))
window.show();window.resize(1550,960)

def await_idle(timeout=120):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents();time.sleep(.02)
        if window.runner.action=='run' and not window.live.isChecked():
            assert not window.live.isEnabled()
            window.set_mode('analyze');assert not window.results.isEnabled() and window.results.store is None
            window._select_step('Run')
        if errors:raise RuntimeError(errors)
        if not window._tasks and not window.runner.is_running:
            # A finished prepare may schedule the chained run on the next turn.
            app.processEvents();time.sleep(.05);app.processEvents()
            if not window._tasks and not window.runner.is_running:return
    raise TimeoutError('Assembled app operation did not finish')

window.open_path(root,confirmed=True);await_idle()
assert window.has_domain and window.map.terrain is not None
assert not window.forms.sediment_enabled.isChecked()
window.forms.set_step('Domain');window.workflow.setCurrentRow(0)
evidence=BASE/'evidence/ui142';evidence.mkdir(parents=True,exist_ok=True)
app.processEvents();window.grab().save(str(evidence/'DOMAIN.png'))
window._select_step('Mesh');app.processEvents();window.grab().save(str(evidence/'MESH.png'))
window.prepare();await_idle(180)
assert window.map.preview_mesh_item is not None
assert not window.log_toggle.isChecked()
app.processEvents();window.grab().save(str(evidence/'MESH_PREVIEW.png'))
window.hydrodynamics_action.setChecked(True);window._select_step('Advanced hydrodynamics')
app.processEvents();window.grab().save(str(evidence/'ADVANCED_HYDRODYNAMICS.png'))
window.hydrodynamics_action.setChecked(False)
window._select_step('Boundaries');app.processEvents();window.grab().save(str(evidence/'BOUNDARIES.png'))
from PySide6.QtCore import QTimer
def capture_boundary():
    popup=app.activeModalWidget()
    if popup:popup.grab().save(str(evidence/'BOUNDARY_DIALOG.png'));popup.reject()
QTimer.singleShot(150,capture_boundary);window.edit_boundary('Upstream')
window._select_step('Run');window.live.setChecked(False);app.processEvents();window.grab().save(str(evidence/'RUN.png'))
window.run();await_idle(180)
assert window.results.store is None
window.set_mode('analyze')
assert window.results.store is not None
assert abs(window.results.store.times[-1]-5.)<1e-10
window.log_toggle.setChecked(False)
from PySide6.QtTest import QTest
QTest.qWait(150);app.processEvents()
window.grab().save(str(evidence/'RESULTS.png'))
window.flow_lines_action.setChecked(True)
QTest.qWait(500);app.processEvents();window.grab().save(str(evidence/'FLOW_PARTICLES.png'))
assert window.results.particles.timer.isActive()
window.flow_lines_action.setChecked(False)
from braids_desktop.media_export import ExportDialog,export_media
dialog=ExportDialog(window.results.store.times,window.results.slider.value(),(960,540),window)
dialog.show();app.processEvents();dialog.grab().save(str(evidence/'EXPORT_DIALOG.png'));dialog.reject()
def render_export(payload):
    window.results._render(payload);window.results.frame_changed.emit(payload);app.processEvents()
    return window.map.export_image()
selected=window.results.slider.value()
exports=[]
for kind,indices in (('png',[selected]),('mp4',list(range(len(window.results.store.times))))):
    path=evidence/('DEPTH.'+kind)
    if path.exists():path=evidence/('DEPTH_'+time.strftime('%H%M%S')+'.'+kind)
    for name in ('Scale','North arrow','Legends','Time'):window.layers[name].setChecked(False)
    export_media(window.results.store,path,kind=kind,indices=indices,fps=10,size=(960,540),variable='depth_m',mesh_mode='none',render=render_export)
    exports.append(str(path))
for name in ('Scale','North arrow','Legends','Time'):window.layers[name].setChecked(True)
window.results.show_frame()
window.results.set_probe(3,3);window.results.set_probe(8,3)
window.results.set_transect([[6,0],[6,8]]);window.results.set_transect([[12,0],[12,8]])
window.results.select_measurement('gauge','probe_1')
for name in ('Probes','Profiles','Boundary flow'):
    window.workflow.setCurrentRow(list(window.results.pages).index(name));app.processEvents()
    window.grab().save(str(evidence/('RESULTS_'+name.upper().replace(' ','_')+'.png')))
window.workflow.setCurrentRow(0);window.results.edit_probe();app.processEvents();window.grab().save(str(evidence/'EDIT_PROBE.png'));window.map.set_tool('pan')
window.set_mode('model_run');window.sediment_action.setChecked(True);window._select_step('Sediment')
app.processEvents();window.grab().save(str(evidence/'SEDIMENT.png'))
window.sediment_action.setChecked(False);window._select_step('Domain')
from braids_desktop.settings_dialog import SettingsDialog
dialog=SettingsDialog(window);dialog.show();app.processEvents();dialog.grab().save(str(evidence/'SETTINGS.png'));dialog.reject()
assert not any(name=='qgis' or name.startswith('qgis.') for name in sys.modules)
summary=dict(project=str(root),runtime=str(BASE/'runtime'),final_time_s=float(window.results.store.times[-1]),
             frames=len(window.results.store.times),errors=errors,qgis_imported=False,project_file=str(window.project_file),
             probes=len(window.results.probes),profiles=len(window.results.profiles),log_opens_automatically=window.log_toggle.isChecked())
summary['exports']=exports;summary['smooth_default']=window.smooth_action.isChecked();summary['particle_default']=False
(evidence/'ASSEMBLED.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
window.dirty=False;window.close()
