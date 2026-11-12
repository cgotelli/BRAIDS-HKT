"""Actual light-theme Qt editors rendered with isolated test preferences."""
import os,sys,json
from stage_bundle207 import ROOT,OUT,TARGET
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(TARGET),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QSettings
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207'
REPORT.mkdir(parents=True,exist_ok=True)
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'qt_settings'))
app=QApplication([])
for f in ('segoeui.ttf','segoeuib.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
apply_theme(app,'light');p=SettingsPanel();assert not p.sediment_enabled.isChecked()
p.load_document(json.loads((OUT/'release_v3/bank_cpu/project.json').read_text()))
p.resize(520,920);p.show()
for step,filename in [('Sediment transport','UI_TRANSPORT.png'),('Morpho evolution','UI_COLLAPSE.png')]:
    p.set_step(p.step_names.index(step));p.sediment_tabs.setCurrentIndex(1)
    p.physics.transport_box.setExpanded(True);p.collapse_box.setExpanded(True);p.resolved_box.setExpanded(True)
    for _ in range(8):app.processEvents()
    assert p.grab().save(str(REPORT/filename))
p.close()
