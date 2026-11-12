"""Read-only UI review with registered Windows fonts for offscreen evidence."""
from workflow61_ui import *
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtCore import Qt
import numpy as np

QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'inspection_settings'))
app=QApplication([]);app.setOrganizationName('BRAIDSWorkflow61');app.setApplicationName('Inspection')
for f in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
apply_theme(app);app.setFont(QFont('Segoe UI',10))
window=MainWindow(runtime=RUNTIME);errors=[];window.error=lambda e:errors.append(str(e));window.show()
def wait():
    start=time.monotonic()
    while window._tasks or window.runner.is_running:
        app.processEvents();time.sleep(.01)
        if errors:raise RuntimeError(errors)
        if time.monotonic()-start>120:raise TimeoutError(window.status.text())
    app.processEvents()
try:
    ui=read(REPORT/'UI_CHECK.json');window.open_path(Path(ui['projects']['bedload']),confirmed=True);wait()
    window._select_step('Run');wait();window.forms.restart.setText(str(next((window.project_root/'initial_states').glob('source_*.npz'))))
    window.forms.inspect_state.click();wait();window.grab().save(str(REPORT/'STATE_READY_UI.png'))
    window.open_path(Path(ui['projects']['turbine']),confirmed=True);wait();window._select_step('Hydrokinetic turbines');wait();app.processEvents()
    window.grab().save(str(REPORT/'WAKE_CONTROLS_UI.png'))
    window.results.load_output(Path(ui['turbine_run']));window.results.variable.setCurrentIndex(window.results.variable.findData('velocity_m_s'))
    from braids_desktop.result_annotations import SpatialProfileDialog
    dialog=SpatialProfileDialog(window.results,np.array([[9.,1.],[9.,7.]]),'Wake profile',follow=False);dialog.show();app.processEvents();dialog.grab().save(str(REPORT/'PROFILE_DIALOG_UI.png'));dialog.close()
    save(REPORT/'VISUAL_CHECK.json',dict(status='passed',errors=errors,registered_windows_fonts=True))
finally:window.dirty=False;window.close();app.processEvents()
