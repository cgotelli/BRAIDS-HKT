"""Execute an ordinary validation project using the installed Desktop Run button."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation58';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58'
os.environ['QT_QPA_PLATFORM']='offscreen';sys.dont_write_bytecode=True;sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))
if os.name=='nt':_dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtCore import QSettings,Qt
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtWidgets import QApplication,QFileDialog
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
from braids_core.river_launcher import verify


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def run(name,resume=None):
    manifest=json.loads((HOME/'PROJECTS.json').read_text());record=manifest[name];folder=Path(record['folder']);c=record['configuration']
    evidence=REPORT/name;evidence.mkdir(exist_ok=False)
    assert sha(RUNTIME/'RUNTIME.json')==json.loads((REPORT/'LOCKED_SPEC.json').read_text())['runtime_sha256'];verify()
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSValidation');app.setApplicationName('Turbine58')
    for f in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
    app.setFont(QFont('Segoe UI',10));apply_theme(app)
    window=MainWindow(runtime=RUNTIME);errors=[];logs=[];finished=[];window.error=lambda e:errors.append(str(e))
    window.runner.log.connect(logs.append);window.runner.finished.connect(lambda action,success,message:finished.append(dict(action=action,success=success,message=message)))
    window.show();start=time.monotonic();last_notice=0.;captured=False;run_output=None
    def wait_for(predicate,limit=7200.):
        nonlocal last_notice,run_output,captured
        while not predicate():
            app.processEvents();time.sleep(.015)
            if window.runner.action=='run':run_output=window.runner.output
            now=time.monotonic()
            if now-last_notice>=15.:
                print(name,window.status.text(),f'wall={now-start:.1f}s',flush=True);last_notice=now
            if errors:raise RuntimeError(errors)
            if any(not r['success'] for r in finished):raise RuntimeError(finished)
            if now-start>limit:raise TimeoutError('Bounded UI validation timeout')
    try:
        window.open_path(Path(record['project']),confirmed=True);wait_for(lambda:window.document_base is not None and not window._tasks)
        window.layers['Mesh'].setChecked(False);window.live.setChecked(True)
        if resume is not None:window.forms.restart.setText(str(Path(resume).resolve()))
        window.run_button.click();wait_for(lambda:window.runner.action=='run')
        run_output=window.runner.output
        save(evidence/'STARTED.json',dict(case=name,runtime_sha256=sha(RUNTIME/'RUNTIME.json'),output=str(run_output),actual_run_button=True))
        # A small live-result interaction proves the public viewer/worker route.
        # Large OTF fields are analyzed after completion; no live-map rendering
        # is required for their hydraulic calculation.
        if name=='mycek_cpu':
            wait_for(lambda:(run_output/'frames/INDEX.json').is_file())
            window.analyze_button.click();app.processEvents()
            for i in range(window.workflow.count()):
                if window.workflow.item(i).data(Qt.UserRole)=='Turbines':window.workflow.setCurrentRow(i)
            wait_for(lambda:window.results.store is not None and len(window.results.store.times)>1)
            window.results.variable.setCurrentIndex(window.results.variable.findData('velocity_m_s'))
            window.results.turbine_quantity.setCurrentIndex(window.results.turbine_quantity.findData('useful_power_W'))
            window.grab().save(str(evidence/'LIVE_RESULTS.png'));captured=True
            # Keep the remaining loop free of repeated map painting. Live
            # results remain enabled and can be revisited by a human operator.
            window.model_button.click();app.processEvents()
        wait_for(lambda:not window.runner.is_running and not window._tasks)
        assert run_output is not None and (run_output/'frames/INDEX.json').is_file()
        assert finished[-1]['action']=='run' and finished[-1]['success']
        progress=json.loads((run_output/'PROGRESS.json').read_text());assert abs(progress['time_s']-c['end'])<1e-8
        actual=json.loads((run_output/'PROJECT.json').read_text())
        assert actual['model']['request']['execution']['backend']==c['backend'] and actual['model']['physics']['sediment'] is None
        assert actual['settings']['levels']==1 and actual['settings']['controls']['cfl']==.5
        assert actual['settings']['hydraulics']['Upstream']['discharge_m3_s']==c['U']*c['H']*c['W']
        assert sha(RUNTIME/'RUNTIME.json')==json.loads((REPORT/'LOCKED_SPEC.json').read_text())['runtime_sha256']
        diagnostics=json.loads((run_output/'DIAGNOSTICS.json').read_text())
        receipt=dict(status='complete',case=name,output=str(run_output),public_project=str(record['project']),actual_run_button=True,
            live_results_interaction=captured,backend=c['backend'],end_time_s=c['end'],wall_including_open_save_prepare_s=time.monotonic()-start,
            run_project_sha256=sha(run_output/'PROJECT.json'),runtime_sha256=sha(RUNTIME/'RUNTIME.json'),
            resumed_from=str(resume) if resume is not None else None,
            maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),errors=errors,finished=finished)
        save(evidence/'RECEIPT.json',receipt);save(folder/'LAST_VALIDATION_RUN.json',receipt)
        print(json.dumps(receipt),flush=True)
    except Exception as e:
        save(evidence/'FAILURE.json',dict(error=str(e),errors=errors,finished=finished));raise
    finally:
        (evidence/'UI_WORKER.log').write_text('\n'.join(logs),encoding='utf-8')
        if window.runner.is_running:window.runner.stop();window.runner.process.waitForFinished(30000)
        window.dirty=False;window.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--resume');a=p.parse_args();run(a.case,a.resume)
