"""Bounded real Run-button / QProcess live-results smoke, not a physical validation."""
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_live56'
TARGET=OUT/'candidate';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/live_hkt4e'
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.dont_write_bytecode=True;sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT))
if os.name=='nt':
    _dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings,Qt
from PySide6.QtGui import QFont,QFontDatabase
from braids_desktop import services,project_file
from braids_desktop._avac.theme import apply_theme
from braids_desktop.shell import MainWindow
from braids_core.river_launcher import verify


def main():
    assert Path(sys.modules['braids_desktop'].__file__).resolve().is_relative_to(TARGET)
    verify()
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSValidation');app.setApplicationName('TurbineLive56')
    for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
    app.setFont(QFont('Segoe UI',10));apply_theme(app)
    folder=OUT/'ui_project';folder.mkdir(exist_ok=False);data_folder=folder/'Live_data';entry=folder/'Live.braids.json'
    doc=json.loads((ROOT/'desktop/qualification/turbine_array55/candidate_array_cpu/input.json').read_text())
    doc['name']='Live';doc.pop('editor_state',None)
    doc['settings'].update(end_time_s=3.,output_interval_s=.25,checkpoint_interval_s=3.)
    doc=services.save_project_folder(data_folder,doc);project_file.write(entry,data_folder,doc)
    window=MainWindow(runtime=TARGET);errors=[];window.error=lambda exc:errors.append(str(exc))
    worker_log=[];window.runner.log.connect(worker_log.append);window.show();window.open_path(entry,confirmed=True)
    start=time.monotonic()
    def wait_for(predicate,limit=240.):
        while not predicate():
            app.processEvents()
            if errors:raise AssertionError(errors)
            if time.monotonic()-start>limit:raise TimeoutError('Bounded UI smoke timed out')
            time.sleep(.01)
    try:
        wait_for(lambda:window.document_base is not None and not window._tasks)
        assert len(window.forms.turbines.rows)==3;window.live.setChecked(True)
        window.run_button.click()
        wait_for(lambda:window.runner.action=='run')
        window.analyze_button.click();app.processEvents()
        names=[window.workflow.item(i).data(Qt.UserRole) for i in range(window.workflow.count())]
        assert {'Profiles','Boundary flow','Turbines'}<=set(names)
        before=window.results.store is None
        window.workflow.setCurrentRow(names.index('Turbines'));app.processEvents()
        assert window.results.isEnabled() and not any(b.isEnabled() for b in window.results.turbine_buttons)
        wait_for(lambda:window.results.store is not None and window.results.turbine_buttons[0].isEnabled())
        wait_for(lambda:max(r['time_s'] for r in window.results._turbine_rows())>=.25)
        assert window.runner.action=='run','Worker finished before live interaction'
        panel=window.results;panel.turbine_select.setCurrentIndex(1)
        panel.turbine_quantity.setCurrentIndex(panel.turbine_quantity.findData('useful_power_W'))
        panel.turbine_buttons[0].click();app.processEvents();assert panel._dialogs
        panel._dialogs[-1].grab().save(str(REPORT/'LIVE_POWER_PLOT.png'))
        for dialog in list(panel._dialogs):dialog.close()
        target=REPORT/'LIVE_EXPORT.csv';original=QFileDialog.getSaveFileName
        try:
            QFileDialog.getSaveFileName=lambda *a,**k:(str(target),'CSV (*.csv)')
            panel.turbine_buttons[1].click()
        finally:QFileDialog.getSaveFileName=original
        assert window.runner.action=='run' and not errors
        exported=list(csv.DictReader(target.open(encoding='utf-8')))
        assert len({r['turbine_id'] for r in exported})==3
        live_time=max(float(r['time_s']) for r in exported)
        window.grab().save(str(REPORT/'LIVE_TURBINES.png'))
        wait_for(lambda:not window.runner.is_running and not window._tasks)
        rows=panel._turbine_rows();assert max(r['time_s'] for r in rows)==3.
        assert window._result_step=='Turbines' and panel._live_output is None
        result=dict(status='passed',end_time_s=3.,turbines=3,tab_before_first_frame=before,live_plot=True,live_export=True,
            live_export_end_s=live_time,live_export_rows=len(exported),final_rows=len(rows),errors=errors,
            output=str(window.runner.output.relative_to(ROOT)),runtime=str(TARGET.relative_to(ROOT)),
            scope='Actual offscreen Qt project open, Run button, save, preparation, QProcess solver, live navigation, plot and CSV; synthetic curves, no physical-validation claim.')
        (REPORT/'UI_RUN.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        (REPORT/'UI_WORKER.log').write_text('\n'.join(worker_log),encoding='utf-8')
        if window.runner.is_running:
            window.runner.stop();window.runner.process.waitForFinished(20000)
        window.dirty=False;window.close()


if __name__=='__main__':main()
