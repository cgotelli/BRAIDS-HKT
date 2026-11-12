"""Inspect the real grain editor at narrow and expanded panel widths."""
import os,sys
from stage_grains191 import ROOT,OUT,TARGET
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(TARGET),str(ROOT)]
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from braids_desktop.forms import SettingsPanel
from braids_desktop.river_physics import set_choice
from braids_desktop._avac.theme import apply_theme
app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
for method in ('mpm','smart_jaggi','recking_2010'):
    f=SettingsPanel();p=f.physics;p.enabled.setChecked(True);p.bedload_on.setChecked(True)
    set_choice(p.capacity,'capacity.'+method);p.grains.item(0,1).setText('5')
    box=p.material_box;box.setParent(None);box.resize(460,520);box.show()
    for _ in range(6):app.processEvents()
    assert not p.grains.isColumnHidden(6)
    box.grab().save(str(OUT/('GRAIN_'+method.upper()+'.png')))
    box.close();f.close()
