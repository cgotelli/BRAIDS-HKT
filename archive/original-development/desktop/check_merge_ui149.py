"""Show the privately prepared real project and revised Mesh button order."""
import json,os,sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_merge149';runtime=OUT/'candidate'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtCore import QSettings,Qt,QRectF
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
out=ROOT/'desktop/evidence/merge149';out.mkdir(parents=True,exist_ok=True)
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(out/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);w=MainWindow(runtime=runtime);errors=[];w.error=lambda error:errors.append(str(error))
w._background=lambda work,done,label:done(work())
w.show();w.open_path(OUT/'user_v2_cpu',confirmed=True);app.processEvents();w._select_step('Mesh')
w.prepared=OUT/'user_v2_cpu/prepared/current';w.layers['Mesh'].setChecked(True);w._show_prepared_mesh()
w.geometry_action('zoom_domain');app.processEvents();assert not errors,errors
assert w.grab().save(str(out/'MESH.png'))
rect=QRectF(w.map.to_scene(2571296,1107392),w.map.to_scene(2571311,1107410)).normalized()
w.map.fitInView(rect,Qt.KeepAspectRatio);app.processEvents();assert w.grab().save(str(out/'REPAIRED_BEND.png'))
page=w.forms.pages['Mesh'];buttons={b.text():b for b in page.findChildren(QPushButton)}
def y(item):return item.mapTo(page,item.rect().topLeft()).y()
positions=[y(buttons['Add breakline']),y(w.forms.breakline_list),y(buttons['Remove breakline']),y(buttons['Generate and Visualize mesh'])]
assert positions==sorted(set(positions))
(out/'CHECK.json').write_text(json.dumps(dict(errors=errors,button_y=positions,mesh_visible=w.map.preview_mesh_item.isVisible()),indent=2)+'\n')
w.dirty=False;w.close()
