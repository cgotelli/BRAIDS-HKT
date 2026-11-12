"""Exercise the actual polyline drawing route and prepared mesh in the UI."""
import json
import os
from pathlib import Path
import tempfile
import launch
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
from braids_desktop.demo import create_demo
from braids_desktop import preparation
from braids_core.river_launcher import verify
from braids_core.river_project import worker_profile,load_project
from braids_core.sediment.adaptive_river_worker import hierarchy
from shapely.geometry import LineString
from shapely.ops import unary_union

verify()
base=Path(__file__).resolve().parent;out=base/'evidence/polylines148';out.mkdir(parents=True,exist_ok=True)
case=Path(tempfile.mkdtemp(prefix='polyline148_',dir=base.parent/'validation'))
create_demo(case)
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(out/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);window=MainWindow(runtime=base/'runtime');errors=[]
window.error=lambda error:errors.append(str(error))
window._background=lambda work,done,label:done(work())
window.show();window.open_path(case,confirmed=True);app.processEvents()
window._select_step('Mesh');window.geometry_action('add_breakline')
points=[[2.2,2.7],[5.4,4.2],[10.1,5.8]]
assert window.map.tool=='line'
window.map.points=[p.copy() for p in points];window.map.finish_drawing()
assert not errors and window.save(),errors
doc=load_project(case/'braids_project.json');row=doc['settings']['breaklines'][0]
assert row['closed'] is False and row['points_xy']==points
window.prepared=preparation.prepare(case/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
window.layers['Mesh'].setChecked(True);window._show_prepared_mesh()
window.geometry_action('zoom_domain');app.processEvents()
assert not errors and window.map.preview_mesh_item.isVisible(),errors
assert window.map.breakline_items[0].path().elementCount()==3
tree=hierarchy(worker_profile(doc));faces=unary_union([LineString([f.start_xy,f.end_xy]) for f in tree.finest.faces if f.breakline_ids])
line=LineString(points)
assert line.difference(faces.buffer(1e-9)).length<1e-8
assert faces.difference(line.buffer(1e-9)).length<1e-8
assert min(c.area_m2 for c in tree.finest.cells)>=1-1e-10
assert window.grab().save(str(out/'MESH.png'))
window.geometry_action('edit_breakline');app.processEvents()
assert len(window.map._breakline_handles)==3
assert window.grab().save(str(out/'EDIT.png'))
receipt=dict(errors=errors,project=str(case),points=points,automatically_closed=False,
             minimum_area_m2=min(c.area_m2 for c in tree.finest.cells),
             finest_cells=len(tree.finest.cells),protected_length_m=faces.length,drawn_length_m=line.length,
             mesh_visible=True,edit_handles=len(window.map._breakline_handles))
(out/'CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2));window.dirty=False;window.close()
