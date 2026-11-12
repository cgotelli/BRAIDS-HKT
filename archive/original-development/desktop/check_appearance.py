"""Assembled appearance editor check using retained output, no simulation."""
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
from braids_desktop._avac.raster_style import RasterStyleDialog

base=Path(__file__).resolve().parent;folder=base/'evidence/appearance145';folder.mkdir(parents=True,exist_ok=True)
root=Path(json.loads((base/'evidence/ui142/ASSEMBLED.json').read_text())['project'])
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(folder/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);window=MainWindow(runtime=base/'runtime');errors=[];window.error=lambda error:errors.append(str(error))
window.show();window.open_path(root,confirmed=True);deadline=time.monotonic()+60
while window._tasks and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
assert not errors and not window._tasks,errors
window.set_mode('analyze');window.geometry_action('zoom_domain');assert window.results.store is not None
dialog=RasterStyleDialog(window.map);assert dialog.layer.currentData()=='result:depth_m'
dialog.automatic.setChecked(False);dialog.low.setValue(.001);dialog.high.setValue(.08);dialog.opacity.setValue(65);dialog.apply()
dialog.show();app.processEvents();assert dialog.grab().save(str(folder/'APPEARANCE.png'))
dialog.close();assert window.map.result_metadata['limits']==(.001,.08)
assert window.results._smooth_item.opacity()==.65 and not hasattr(window,'terrain_opacity')
window.layers['Mesh'].setChecked(False);window.flow_lines_action.setChecked(True)
tracks=window.results.particles;tracks.timer.stop()
for i in range(26):tracks.advance()
app.processEvents();assert window.grab().save(str(folder/'RESULTS.png'))
receipt=dict(errors=errors,results_style=window.map.styles['result:depth_m'],result_limits=window.map.result_metadata['limits'],
    opacity=window.results._smooth_item.opacity(),terrain_opacity_slider=False,particle_halo=False,new_simulation_started=False)
(folder/'CHECK.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
window.dirty=False;window.close()
