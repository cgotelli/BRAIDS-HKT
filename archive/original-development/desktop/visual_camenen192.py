"""Render the actual light-theme Desktop editor without opening a live window."""
import os,sys
from stage_camenen192 import ROOT,OUT,TARGET
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(TARGET),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QSettings
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
from braids_desktop.river_physics import set_choice
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_camenen192'
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'qt_settings'))
app=QApplication([])
for f in ('segoeui.ttf','segoeuib.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
apply_theme(app,'light');REPORT.mkdir(parents=True,exist_ok=True)
p=SettingsPanel();assert not p.sediment_enabled.isChecked()
p.sediment_enabled.setChecked(True);p.physics.bedload_on.setChecked(True)
set_choice(p.physics.capacity,'capacity.camenen_larson_2005')
p.set_step(p.step_names.index('Sediment transport'));p.sediment_tabs.setCurrentIndex(1)
p.physics.transport_box.setExpanded(True);p.resize(470,850);p.show()
for _ in range(8):app.processEvents()
assert p.grab().save(str(REPORT/'UI.png'));p.close()
