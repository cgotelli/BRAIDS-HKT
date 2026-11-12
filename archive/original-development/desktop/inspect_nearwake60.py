"""Reopen every delivered project and exercise its saved turbine export."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNTIME=Path(os.environ.get('BRAIDS_VALIDATION_RUNTIME',str(ROOT/'desktop/runtime')))
HOME=Path.home()/'Desktop/BRAIDS_Turbine_NearWake60';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/nearwake60'
os.environ['QT_QPA_PLATFORM']='offscreen';sys.dont_write_bytecode=True;sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtCore import QSettings,Qt
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtWidgets import QApplication,QFileDialog
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme


def inspect(case=None):
    records=json.loads((HOME/'PROJECTS.json').read_text());results=[];errors=[]
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSValidation');app.setApplicationName('NearWake60Inspect')
    for f in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
    app.setFont(QFont('Segoe UI',10));apply_theme(app)
    window=MainWindow(runtime=RUNTIME);window.error=lambda e:errors.append(str(e));window.show()
    def wait(predicate):
        end=time.monotonic()+120
        while not predicate():
            app.processEvents();time.sleep(.02);assert not errors,errors
            if time.monotonic()>end:raise TimeoutError('Project reopen inspection')
    try:
        for name in ([case] if case else ['wake4_gpu','wake8_gpu','wake4_cpu','zero4_gpu','zero8_corrected']):
            r=records[name]
            record=json.loads((REPORT/name/'RECEIPT.json').read_text())
            output=Path(record['output']);status=json.loads((output/'STATUS.json').read_text())
            window.open_path(Path(r['project']),confirmed=True);wait(lambda:window.document_base is not None and not window._tasks)
            assert window.forms.restart.text()=='';assert Path(window._results_output).resolve()==output.resolve()
            window.layers['Mesh'].setChecked(False);window.analyze_button.click()
            wait(lambda:window.results.store is not None and not window._tasks)
            store=window.results.store;assert store.times[-1]>=100.
            assert store.times[-1]==r['configuration']['end'] and status['status']=='complete'
            assert len(window.results.measurement_state()['profiles'])==6
            for i in range(window.workflow.count()):
                if window.workflow.item(i).data(Qt.UserRole)=='Turbines':window.workflow.setCurrentRow(i)
            app.processEvents();rows=window.results._turbine_rows()
            target=REPORT/name/('UI_PRECHECK_PERFORMANCE.csv' if case else 'UI_INSTALLED_PERFORMANCE.csv');original=QFileDialog.getSaveFileName
            try:
                QFileDialog.getSaveFileName=staticmethod(lambda *a,**kw:(str(target),'CSV (*.csv)'));window.results.export_turbine()
            finally:QFileDialog.getSaveFileName=original
            assert target.read_bytes()==(output/'TURBINE_PERFORMANCE.csv').read_bytes()
            with target.open(encoding='utf-8') as f:assert len(list(csv.DictReader(f)))==len(rows)
            if name=='wake8_gpu':
                window.results.variable.setCurrentIndex(window.results.variable.findData('velocity_m_s'))
                window.results.slider.setValue(len(store.times)-1);app.processEvents();window.grab().save(str(REPORT/'DESKTOP_PROJECT.png'))
            results.append(dict(case=name,latest_result_reopened=True,no_automatic_restart=True,comparison_profiles=6,
                turbine_tab=True,csv_exact=True,recorded_end_s=float(store.times[-1]),status=status['status']))
            print('Reopened and exported',name,flush=True)
        assert not errors
        (REPORT/('PROJECT_REOPEN_PRECHECK.json' if case else 'PROJECT_REOPEN_CHECK.json')).write_text(json.dumps(dict(status='passed',errors=errors,projects=results,
            runtime_sha256=hashlib.sha256((RUNTIME/'RUNTIME.json').read_bytes()).hexdigest()),indent=2)+'\n')
    finally:window.dirty=False;window.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');inspect(parser.parse_args().case)
