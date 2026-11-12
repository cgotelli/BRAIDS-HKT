"""Offscreen real Qt layout, formula presets and delivered-project roundtrip."""
import sys,os,json
from pathlib import Path
from copy import deepcopy
from stage_grains212 import ROOT,OUT,TARGET,save,sha
RUNTIME=ROOT/'desktop/runtime' if '--installed' in sys.argv else TARGET
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(RUNTIME),str(ROOT)]
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'editor_preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS212Qualification');app.setApplicationName('GrainTable')
assert QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')>=0
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_desktop.bedload_catalog import CATALOG,GRAIN_DERIVATIONS,JC_METHOD
from braids_desktop._avac.theme import apply_theme
from braids_desktop import services
from braids_core.river_worker import load_prepared,validate_continuation,source_identity
apply_theme(app,'light')
p=SettingsPanel();assert not p.sediment_enabled.isChecked()
folder=ROOT.parent/'BRAIDS_JiangCao_MF1'
doc=services.load_project_folder(folder);p.load_document(doc);got=p.document(doc)
original=load_prepared(folder/'prepared/flume211_ready')[0]
def persisted(value):
    value=deepcopy(value);asset=Path(value['terrain']['dem_path'])
    if asset.is_absolute():value['terrain']['dem_path']=asset.relative_to(folder).as_posix()
    return value
def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [x for k in a.keys()|b.keys() for x in differences(a.get(k),b.get(k),path+'/'+k)]
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in differences(u,v,path+'/'+str(i))]
    return [] if a==b else [dict(path=path,before=a,after=b)]
changes=differences(doc,got)
save(OUT/'EDITOR_DIAGNOSTIC.json',dict(editor_changes=changes,
    current_project_vs_prepared=differences(original,persisted(doc))))
print(json.dumps([x for x in changes if not x['path'].startswith('/editor_state')],ensure_ascii=False,indent=2))
validate_continuation(persisted(doc),persisted(got))
# The user's editable project may have changed since delivery. Test continuation
# using the actual frozen prepared document, never overwrite those user edits.
p.load_document(deepcopy(original));got=p.document(original)
validate_continuation(original,persisted(got))
p.end_time.setValue(40.);assert validate_continuation(original,persisted(p.document(original)))==40.
table=p.physics
assert table.grains.item(0,7).text()=='D90 = 1'
host=QWidget();layout=QVBoxLayout(host);table.material_box.setParent(host);layout.addWidget(table.material_box)
host.setFixedWidth(380);rows={};images=OUT/'figures';images.mkdir(exist_ok=True)
for method in CATALOG:
    set_choice(table.capacity,method);table.grains.item(0,1).setText('1')
    for _ in range(10):app.processEvents();host.adjustSize()
    host.show()
    for _ in range(10):app.processEvents()
    name=method.removeprefix('capacity.')
    assert host.grab().save(str(images/(name+'.png')))
    rows[method]=dict(input_header=table.grains.horizontalHeaderItem(1).text(),
        percentile=table.grains.item(0,6).text(),derived=table.grains.item(0,7).text(),
        derived_column_visible=not table.grains.isColumnHidden(7),
        row_count=table.grains.rowCount(),title=table.grain_preset.text())
save(OUT/('INSTALLED_EDITOR_CHECK.json' if '--installed' in sys.argv else 'EDITOR_CHECK.json'),
    dict(runtime=sha(RUNTIME/'RUNTIME.json'),source_identity=source_identity(),formula_tables=rows,
         current_prepared_accepted=True,continuation_end_s=40.,default_sediment=False,
         project_files_modified=False,render='offscreen Qt components, light theme'))
host.close();p.close()
print('All nine grain tables rendered; delivered MF1 project and continuation unchanged',flush=True)
