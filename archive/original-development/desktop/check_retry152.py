"""Real Qt worker failure, geometry correction, asynchronous save and retry."""
from pathlib import Path
from copy import deepcopy
import json,os,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_mesh152'
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,QSettings
from braids_desktop.demo import create_demo
from braids_desktop import services,project_file
from braids_desktop.shell import MainWindow
app=QApplication([]);root=OUT/'installed_retry';doc=create_demo(root,end_time_s=.2)
doc['settings']['solids']=[dict(id='almost_domain',points_xy=[[.1,.1],[19.9,.1],[19.9,7.9],[.1,7.9]])]
doc['settings']['levels']=1;doc['settings']['grid'].update(nx=10,ny=4,dx_m=2.,dy_m=2.)
services.save_project_folder(root,doc);entry=OUT/'Retry.braids.json';project_file.write(entry,root,doc)
good=deepcopy(doc);good['settings']['solids']=[dict(id='island',points_xy=[[6.2,2.3],[9.5,2.8],[9.1,5.6],[6.5,5.4]])]
with tempfile.TemporaryDirectory(prefix='braids-retry-') as temporary:
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,temporary)
    w=MainWindow(runtime=ROOT/'desktop/runtime');errors=[];finished=[];exceptions=[]
    def excepthook(kind,value,tb):exceptions.append(str(value));app.exit(1)
    sys.excepthook=excepthook
    w.error=lambda exc:errors.append(str(exc));w.show();w.open_path(entry,confirmed=True)
    phase=['opening'];view_checks=[0]
    def sample():
        if phase[0]=='opening' and not w._tasks:
            assert w.project_root==root and not errors,errors
            phase[0]='failure';w.runner.prepare(root)
        if w._tasks or w.runner.is_running:
            w.geometry_action('zoom_domain');view_checks[0]+=1
    def done(action,success,message):
        finished.append(dict(action=action,success=success,message=message))
        if phase[0]=='failure':
            assert not success and 'minimum cell area' in message,message
            assert not any('Permission denied' in e for e in errors),errors
            errors.clear();phase[0]='corrected'
            good['terrain']['dem_path']=str((root/good['terrain']['dem_path']).resolve())
            w._load_document(good);w.dirty=True
            QTimer.singleShot(0,lambda:w.save(then=lambda:w.runner.prepare(root)))
        elif action=='prepare':
            assert success,message
            phase[0]='run';QTimer.singleShot(0,lambda:w.runner.run(root,w.prepared))
        else:
            assert success,message
            w.dirty=False;app.exit(0)
    w.runner.finished.connect(done)
    timer=QTimer();timer.setInterval(40);timer.timeout.connect(sample);timer.start()
    QTimer.singleShot(90000,lambda:(exceptions.append('timeout'),app.exit(1)))
    app.exec();timer.stop()
    assert not exceptions and not errors and len(finished)==3,(exceptions,errors,finished)
    assert [x['success'] for x in finished]==[False,True,True]
    w.dirty=False;w.close()
    row=dict(finished=finished,errors=errors,exceptions=exceptions,view_checks=view_checks[0])
    (OUT/'INSTALLED_RETRY.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
