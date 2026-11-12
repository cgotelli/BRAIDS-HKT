"""Render actual MPM UI and result loading against the qualified candidate."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'validation/_bedload157'
installed='--installed' in sys.argv
runtime=ROOT/'desktop/runtime' if installed else base/'candidate'
sys.path[:0]=[str(runtime),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from braids_desktop.shell import MainWindow
from braids_desktop.accordion import AccordionSection
from braids_desktop._avac.theme import apply_theme

app=QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');apply_theme(app,'light')
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(base/'review_settings'))
w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e))
w._background=lambda work,done,label:done(work())
project=Path.home()/'Desktop/BRAIDS MPM groin/Groin.braids.json' if installed else base/'final_cpu/project'
w.open_path(project,confirmed=True);w.show();app.processEvents()
out=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bedload157'
for step in ('Sediment transport','Morpho evolution','Sediment transport BC'):
    w.forms.set_step(step)
    for section in w.forms.findChildren(AccordionSection):section.setExpanded(True)
    app.processEvents();app.processEvents()
    w.forms.grab().save(str(out/(step.replace(' ','_')+'_UI.png')))
    assert w.forms.stack.currentWidget().horizontalScrollBar().maximum()==0
w.results.load_output(project.parent/'Groin_data/runs/cpu' if installed else base/'final_cpu/run');assert w.results.store is not None
w.set_mode('analyze')
w.results.variable.setCurrentIndex(w.results.variable.findData('bed_change_m'))
app.processEvents();app.processEvents();w.grab().save(str(out/'DESKTOP_RESULTS.png'))
assert not errors,errors
if installed:
    from braids_desktop.demo import create_river_demo
    from braids_desktop import services
    fresh=base/'installed_river_demo'
    if not fresh.exists():create_river_demo(fresh)
    assert services.load_project_folder(fresh)['settings']['sediment_start_time_s']==0.
w.dirty=False;w.close();print('MPM UI and results load passed')
