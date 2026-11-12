"""Exercise view-only controls while actual installed prepare/run children work."""
from pathlib import Path
import json,os,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_solids151'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'desktop/runtime'))
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer,QSettings
from braids_desktop.demo import create_demo
from braids_desktop import services
from braids_desktop.shell import MainWindow

app=QApplication([]);root=OUT/'installed_navigation_v2'
doc=create_demo(root,end_time_s=1.)
doc['settings']['solids']=[dict(id='wall',points_xy=[[6.2,2.3],[9.5,2.8],[9.1,5.6],[6.5,5.4]])]
services.save_project_folder(root,doc)
with tempfile.TemporaryDirectory(prefix='braids-navigation-') as temporary:
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,temporary)
    w=MainWindow(runtime=ROOT/'desktop/runtime');errors=[];counts={'prepare':0,'run':0};finished=[]
    w.error=lambda exc:errors.append(str(exc));w._background=lambda work,done,label:done(work())
    w.open_path(root,confirmed=True);w.show();w.live.setChecked(False)
    def sample():
        if not w.runner.is_running:return
        try:
            counts[w.runner.action]+=1;w.geometry_action('zoom_domain');w.map.zoom(1.01)
            w.layers['Solids'].setChecked(not w.layers['Solids'].isChecked())
            w.map.styles['terrain']={'opacity':.7};w._appearance_changed()
            assert w.map.terrain_item.opacity()==.7
            for key in ('fit_domain','pan','select','ruler','zoom_in','zoom_out'):
                assert w.map.tool_buttons[key].isEnabled()
            assert not w.forms.pages['BC & IC'].isEnabled()
        except Exception as exc:errors.append(str(exc))
    def done(action,success,message):
        finished.append(dict(action=action,success=success,message=message))
        if not success:errors.append(message);app.exit(1);return
        if action=='prepare':QTimer.singleShot(0,lambda:w.runner.run(root,w.prepared))
        else:
            # View changes intentionally mark the editor dirty. Do not let the
            # test's QApplication quit trigger an interactive unsaved dialog.
            w.dirty=False;app.exit(0)
    timer=QTimer();timer.setInterval(50);timer.timeout.connect(sample);timer.start()
    w.runner.finished.connect(done)
    QTimer.singleShot(0,lambda:w.runner.prepare(root))
    QTimer.singleShot(60000,lambda:(errors.append('Timed out'),app.exit(1)))
    app.exec();timer.stop()
    assert not w.runner.is_running and not errors,(errors,finished)
    assert counts['prepare']>0 and counts['run']>0 and len(finished)==2
    w.dirty=False;w.close()
    receipt=dict(view_checks=counts,finished=finished,errors=errors)
    (OUT/'INSTALLED_NAVIGATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
