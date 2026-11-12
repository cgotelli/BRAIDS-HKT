"""Prepare a fresh standalone UI project and record the assembled mesh tools."""
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'validation/_breaklines147/candidate'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RUNTIME))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
from braids_desktop.demo import create_demo
from braids_desktop import services,preparation
from braids_core.river_launcher import verify

verify()
out=ROOT/'desktop/evidence/breaklines147';out.mkdir(parents=True,exist_ok=True)
case=Path(tempfile.mkdtemp(prefix='ui_case_',dir=ROOT/'validation/_breaklines147'))
doc=create_demo(case)
doc['settings']['breaklines']=json.loads((ROOT/'validation/_breaklines147/final_lake_cpu/braids_project.json').read_text())['settings']['breaklines']
services.save_project_folder(case,doc)
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(out/'settings'))
for filename in ('segoeui.ttf','segoeuib.ttf','seguisym.ttf'):
    font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/filename
    if font.is_file():QFontDatabase.addApplicationFont(str(font))
apply_theme(app);window=MainWindow(runtime=RUNTIME);errors=[]
window.error=lambda error:errors.append(str(error))
window._background=lambda work,done,label:done(work())
window.show();window.open_path(case,confirmed=True);app.processEvents()
assert not errors,errors
assert window.save()
window.prepared=preparation.prepare(case/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
window._select_step('Mesh');window.layers['Mesh'].setChecked(True);window._show_prepared_mesh()
window.geometry_action('zoom_domain');app.processEvents()
assert not errors and window.map.preview_mesh_item.isVisible(),errors
assert window.grab().save(str(out/'MESH.png'))
window.geometry_action('edit_breakline');app.processEvents()
assert len(window.map._breakline_handles)==4
assert window.grab().save(str(out/'EDIT.png'))
receipt=dict(errors=errors,mesh_visible=True,breakline_visible=window.map.breakline_items[0].isVisible(),
             edit_handles=len(window.map._breakline_handles),project=str(case),runtime=str(RUNTIME))
(out/'CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2));window.dirty=False;window.close()
