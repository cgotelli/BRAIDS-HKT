"""Retained model mesh and temporary ruler: assembled UI, no new simulation."""
import os
import json
import time
from pathlib import Path
import launch
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme

base=Path(__file__).resolve().parent;folder=base/'evidence/tools146';folder.mkdir(parents=True,exist_ok=True)
root=Path(json.loads((base/'evidence/ui142/ASSEMBLED.json').read_text())['project'])
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(folder/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);window=MainWindow(runtime=base/'runtime');errors=[];window.error=lambda error:errors.append(str(error))
window.show();window.open_path(root,confirmed=True);deadline=time.monotonic()+60
while time.monotonic()<deadline:
    app.processEvents();time.sleep(.01)
    if not window._tasks and not window._mesh_load_scheduled:break
assert not window._tasks and not errors,errors
assert window.mode=='model_run' and window.map.preview_mesh_item is not None and window.map.preview_mesh_item.isVisible()
window.geometry_action('zoom_domain');app.processEvents();assert window.grab().save(str(folder/'MODEL_MESH.png'))
items=len(window.map.scene().items());window.map.set_tool('ruler')
for point in ((3.,1.),(7.,1.),(7.,5.)):window.map.ruler.add(point)
window.map.ruler.cursor=(12.,5.);window.map.viewport().update();app.processEvents()
assert window.grab().save(str(folder/'RULER.png'))
assert window.map.ruler.lengths()==[4.,4.,5.] and len(window.map.scene().items())==items
window.map.set_tool('pan');assert not window.map.ruler.points
receipt=dict(errors=errors,prepared=str(window.prepared),model_mesh_visible=True,segment_lengths_m=[4.,4.,5.],
    total_m=13.,temporary_scene_layer=False,clears_on_exit=True,new_simulation_started=False)
(folder/'CHECK.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
window.dirty=False;window.close()
