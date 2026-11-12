"""Offscreen Qt rendering of the installed interface on synthetic test data."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'tests')]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings,Qt,QRectF
from PySide6.QtGui import QFontDatabase
from braids_desktop._avac.theme import apply_theme
from braids_desktop.shell import MainWindow
from braids_desktop.demo import create_demo
from braids_desktop.json_editor import JsonEditor
from test_desktop_results import create_run
from braids_core.river_launcher import verify

def main():
    verify()
    folder=ROOT/'validation/_ui182/visual_case'
    from braids_desktop.services import load_project_folder
    doc=load_project_folder(folder) if folder.exists() else create_demo(folder,end_time_s=.02)
    out=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_ui182'
    out.mkdir(parents=True,exist_ok=True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(folder/'qt_settings'))
    app=QApplication([])
    for font in ('segoeui.ttf','segoeuib.ttf','consola.ttf','seguisym.ttf','segmdl2.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
    apply_theme(app,'light')
    w=MainWindow(runtime=ROOT/'desktop/runtime');errors=[];w.error=lambda e:errors.append(str(e))
    w._background=lambda work,done,label:done(work())
    w.open_path(folder,confirmed=True);assert not errors,errors
    w.resize(1400,900);w.show();w._select_step('Mesh')
    for _ in range(12):app.processEvents()
    assert w.grab().save(str(out/'MESH_CONTROLS.png'))
    j=JsonEditor(json.dumps(doc,indent=2));j.resize(1050,680);j.show()
    j.search.setText('domain_border_enabled')
    for _ in range(4):app.processEvents()
    assert j.grab().save(str(out/'JSON_BROWSER.png'));j.close()
    result=folder/'synthetic_results'
    if not result.exists():create_run(result)
    w.results.load_output(result);w.set_mode('analyze')
    w.results.set_sediment_available(True);w.results.variable.setCurrentIndex(w.results.variable.findData('bed_m'))
    w.results.set_transect([[.1,.5],[1.9,.5]]);w.workflow.setCurrentRow(1)
    w.extrema_actions['minimum'].setChecked(True);w.extrema_actions['maximum'].setChecked(True)
    w.map.fitInView(QRectF(w.map.to_scene(-.5,1.6),w.map.to_scene(2.7,-.6)).normalized(),Qt.KeepAspectRatio)
    for _ in range(8):app.processEvents()
    assert w.grab().save(str(out/'RESULT_EXTREMA.png'))
    w.results.plot_transect();d=w.results._dialogs[-1]
    for _ in range(8):app.processEvents()
    assert d.grab().save(str(out/'SPATIAL_PROFILE.png'))
    d.close();w.dirty=False;w.close();app.processEvents()
    assert not errors,errors
    print('Installed UI rendered:',out)
if __name__=='__main__':main()
