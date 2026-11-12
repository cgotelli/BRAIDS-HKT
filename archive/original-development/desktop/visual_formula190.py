"""Render the actual two new Desktop parameter editors, offscreen."""
import os,sys
from stage_formula190 import ROOT,TARGET
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path[:0]=[str(TARGET),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QSettings
from braids_desktop.forms import SettingsPanel
from braids_desktop._avac.theme import apply_theme
from braids_desktop.river_physics import set_choice
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(ROOT/'validation/_formula190/qt_settings'))
app=QApplication([])
for f in ('segoeui.ttf','segoeuib.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
apply_theme(app,'light');REPORT.mkdir(parents=True,exist_ok=True)
for method in ('ancey_recking_2023','recking_2010'):
    p=SettingsPanel();assert not p.sediment_enabled.isChecked()
    p.sediment_enabled.setChecked(True);p.physics.bedload_on.setChecked(True)
    set_choice(p.physics.capacity,'capacity.'+method)
    p.set_step(p.step_names.index('Sediment transport'));p.sediment_tabs.setCurrentIndex(1)
    p.physics.transport_box.setExpanded(True);p.resize(470,850);p.show()
    for _ in range(8):app.processEvents()
    assert p.grab().save(str(REPORT/(method+'_UI.png')));p.close()
