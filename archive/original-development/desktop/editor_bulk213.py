"""Real Qt grain-size forms, saved projects and explicit roughness selection."""
import os,sys,json
from pathlib import Path
from stage_bulk213 import ROOT,OUT,TARGET,sha,save
RUNTIME=ROOT/'desktop/runtime' if '--installed' in sys.argv else TARGET
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'editor_preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS213Qualification');app.setApplicationName('BulkGrainSizes')
assert QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')>=0
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_desktop._avac.theme import apply_theme
from braids_desktop.bedload_catalog import CATALOG
from test_bulk213 import fill,SIZES,REQUIRED
from test_jiang_cao207 import project
from braids_core.river_jiang_cao import ROUGHNESS,METHOD
from braids_core.river_project import validate_project
apply_theme(app,'light')
f=SettingsPanel();assert not f.sediment_enabled.isChecked()
doc=project(OUT/'editor_project',opened=True)
f.load_document(doc);after=f.document(doc)
assert after['model']==doc['model']
assert after['settings']==doc['settings']
assert f.roughness.currentData()==ROUGHNESS
saved=OUT/'editor_project/project213.json';saved.write_text(json.dumps(after,indent=2))
g=SettingsPanel();g.load_document(validate_project(json.loads(saved.read_text())))
assert g.document(after)['model']==after['model'];g.close()
host=QWidget();layout=QVBoxLayout(host);f.physics.material_box.setParent(host)
layout.addWidget(f.physics.material_box);host.setFixedWidth(380)
images=OUT/'figures';images.mkdir(exist_ok=True);rows={}
for method in CATALOG:
    set_choice(f.physics.capacity,method);fill(f.physics)
    for _ in range(5):app.processEvents();host.adjustSize()
    host.show()
    for _ in range(5):app.processEvents()
    assert host.grab().save(str(images/(method.removeprefix('capacity.')+'.png')))
    rows[method]=dict(rows=[f.physics.grain_sizes.item(r,0).text() for r in range(len(REQUIRED[method]))],
        sizes_m=f.physics.selection()['capacity']['grain_sizes_m'],transported_materials=f.physics.count.value(),
        roughness=f.roughness.currentData())
    assert not f.physics.grains.isColumnHidden(2) and not f.physics.grains.isColumnHidden(4)
save(OUT/('INSTALLED_EDITOR.json' if '--installed' in sys.argv else 'EDITOR.json'),dict(
    version='0.1.0-preview38',runtime=sha(RUNTIME/'RUNTIME.json'),formula_tables=rows,
    project_roundtrip=True,sediment_off_by_default=True,user_projects_modified=False,
    rendering='offscreen real Qt components, 380px light-theme panel'))
host.close();f.close()
print('Nine editable grain tables and current-format project roundtrip passed',flush=True)
