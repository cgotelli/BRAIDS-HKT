"""Real Qt render and authoritative roundtrip, isolated preferences."""
import os,sys
from stage_supplied220 import ROOT,OUT,TARGET,BASE,REPORT,sha,save
RUNTIME=BASE if '--installed' in sys.argv else TARGET
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'tests')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'ui_preferences'))
app=QApplication([]);app.setOrganizationName('BRAIDS220Qualification');app.setApplicationName('Suspension')
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
from test_supplied220 import document,METHOD
METHODS=(METHOD,)
apply_theme(app,'light')
folder=OUT/('installed_editor' if '--installed' in sys.argv else 'editor')
rows=[]
for method in METHODS:
    short=method.split('.')[-1]
    doc=document(folder/short,'cpu');f=SettingsPanel();f.resize(390,900);f.show();f.load_document(doc)
    again=f.document(doc)
    for k in ('model','settings','initial','supplied_suspension_parameters'):assert again[k]==doc[k],k
    f.set_step(f.step_names.index('Sediment transport'))
    images=REPORT/('installed_ui' if '--installed' in sys.argv else 'ui')/short;images.mkdir(parents=True,exist_ok=True)
    for i in range(4):
        f.sediment_tabs.setCurrentIndex(i)
        for _ in range(6):app.processEvents()
        scroll=f.stack.currentWidget();scroll.verticalScrollBar().setValue(0)
        assert f.grab().save(str(images/f'{i}.png'))
        assert scroll.horizontalScrollBar().maximum()==0,(i,scroll.horizontalScrollBar().maximum())
    assert not f.physics.reference_concentration.isEnabled()
    rows.append(dict(method=method,width_px=390,tabs=4,roundtrip=True,hindering_disabled=True))
    f.close()
save(OUT/('INSTALLED_EDITOR.json' if '--installed' in sys.argv else 'EDITOR.json'),dict(runtime=sha(RUNTIME/'RUNTIME.json'),
    cases=rows,render='real Qt offscreen'))
print('Supplied-suspension real Qt roundtrip and renders pass',flush=True)


