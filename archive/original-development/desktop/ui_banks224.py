import os,sys
from stage_banks224 import ROOT,TARGET,REPORT
sys.path[:0]=[str(TARGET),str(ROOT)]
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from braids_desktop.forms import SettingsPanel
from braids_desktop import services
from braids_desktop._avac.theme import apply_theme
REPORT.mkdir(parents=True,exist_ok=True)
app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
d=services.load_project_folder(ROOT/'desktop/examples/banks224/block_repose')
f=SettingsPanel();f.resize(430,950);f.load_document(d);f.show()
f.set_step(f.step_names.index('Morpho evolution'));f.bank_erosion_box.setExpanded(True)
for _ in range(8):app.processEvents()
assert f.physics.repose_angle_deg.isEnabled() and not f.physics.collapse_slope.isEnabled()
assert f.grab().save(str(REPORT/'UI_BANKS.png'))
assert f.stack.currentWidget().horizontalScrollBar().maximum()==0
f.set_step(f.step_names.index('Sediment transport'));f.sediment_tabs.setCurrentIndex(3)
for _ in range(8):app.processEvents()
assert f.grab().save(str(REPORT/'UI_RESOLVED.png'))
f.close();print('Expanded bank controls and resolved setup captured')
