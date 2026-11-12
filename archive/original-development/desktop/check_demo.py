"""Open a private copy of the requested river demo without running it."""
import json
import os
from pathlib import Path
import time
import launch
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from braids_desktop._avac.theme import apply_theme
from braids_desktop.demo import create_river_demo
from braids_desktop.shell import MainWindow
from braids_desktop import services
from braids_desktop.workspace import digest

base=Path(__file__).resolve().parent;evidence=base/'evidence/ui142';evidence.mkdir(parents=True,exist_ok=True)
source=Path('C:/Users/STREEM/Desktop/01010101');before=digest(source/'braids_project.json')
template=services.load_project_folder(base/'examples/river_demo')
root=base/'test_projects'/('river_demo_'+time.strftime('%Y%m%d_%H%M%S'))
doc=create_river_demo(root)
for key in template:
    if key not in ('name','terrain'):assert doc[key]==template[key],key
assert digest(doc['terrain']['dem_path'])==digest(template['terrain']['dem_path'])
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(evidence/'demo_settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if path.is_file():QFontDatabase.addApplicationFont(str(path))
apply_theme(app);window=MainWindow(runtime=base/'runtime');errors=[];window.error=lambda error:errors.append(str(error))
window.resize(1550,960);window.show();window.open_path(root,confirmed=True)
deadline=time.monotonic()+90
while window._tasks and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
assert not window._tasks and not errors,errors
assert window.map.terrain is not None and window.has_domain and not window.runner.is_running
assert before==digest(source/'braids_project.json')
window.geometry_action('zoom_domain')
app.processEvents();window.grab().save(str(evidence/'RIVER_DEMO.png'))
receipt=dict(source=str(source),source_project_sha256=before,source_unchanged=True,
    template_setup_exact=True,dem_sha256=digest(doc['terrain']['dem_path']),copy=str(root),
    opened=True,simulation_started=False,errors=errors)
(evidence/'DEMO.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2));window.dirty=False;window.close()
