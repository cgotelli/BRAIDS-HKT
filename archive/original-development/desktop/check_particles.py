"""Render the new particle style on a retained test run, without rerunning it."""
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

base=Path(__file__).resolve().parent;evidence=base/'evidence/particles144';evidence.mkdir(parents=True,exist_ok=True)
root=Path(json.loads((base/'evidence/ui142/ASSEMBLED.json').read_text())['project'])
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(evidence/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);window=MainWindow(runtime=base/'runtime');errors=[];window.error=lambda error:errors.append(str(error))
window.show();window.open_path(root,confirmed=True)
deadline=time.monotonic()+60
while window._tasks and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
assert not window._tasks and not errors,errors
window.set_mode('analyze');assert window.results.store is not None
window.geometry_action('zoom_domain')
for name in ('Mesh','Domain','Boundaries'):window.layers[name].setChecked(False)
window.flow_lines_action.setChecked(True);tracks=window.results.particles;tracks.timer.stop()
times=[]
for i in range(28):
    stamp=time.perf_counter();tracks.advance();times.append(time.perf_counter()-stamp)
app.processEvents();assert window.grab().save(str(evidence/'FLOW_TRAILS.png'))
assert window.map.export_image().save(str(evidence/'MAP.png'))
receipt=dict(errors=errors,particles=len(tracks.positions),history_positions=len(tracks.paths),
    tail_segments=len(tracks.item.tail_paths),heads=len(tracks.item.head_points),
    median_advance_ms=1000*sorted(times[20:])[4],new_simulation_started=False)
(evidence/'CHECK.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
window.dirty=False;window.close()
