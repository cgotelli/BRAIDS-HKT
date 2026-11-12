"""Round-trip actual staged Qt widgets without altering user preferences."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from check_power_planning75 import document,ROOT,TARGET,REPORT,save
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from braids_desktop.forms import SettingsPanel
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'ui_settings'))
app=QApplication([])
from PySide6.QtGui import QFontDatabase,QFont
for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
from braids_desktop._avac.theme import apply_theme
apply_theme(app)
p=SettingsPanel();rows=[]
for case,count in [('single_high',1),('tandem_high6',2),('staggered_high',3)]:
    d=document(case);p.load_document(d)
    assert p.turbines.power_planning.isChecked() and p.turbines.enabled.isChecked()
    rebuilt=p.document(d)
    assert rebuilt['turbines'] is None and len(rebuilt['power_plan']['layout']['turbines'])==count
    assert all(r['control']['wake']['method']=='gaussian_shape_power_planning' for r in rebuilt['power_plan']['layout']['turbines'])
    p.load_document(rebuilt);again=p.document(rebuilt)
    assert again['power_plan']==rebuilt['power_plan']
    rows.append(dict(case=case,turbines=count,roundtrip=True))
p.set_step('Hydrokinetic turbines');p.resize(600,1000);p.show();app.processEvents();p.grab().save(str(REPORT/'EDITOR.png'));p.close()
save(REPORT/'UI_ROUNDTRIP.json',dict(passed=True,cases=rows));print('UI round trips passed')
