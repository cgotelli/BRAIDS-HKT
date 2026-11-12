"""Open all delivered projects through the actual Desktop shell."""
import os,sys,time,json,csv,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'desktop'))
from power_planning75 import TARGET,REPORT,read,save
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else TARGET
sys.path[:0]=[str(runtime),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication,QFileDialog
from PySide6.QtCore import QSettings,Qt
from PySide6.QtGui import QFontDatabase,QFont
from braids_desktop.shell import MainWindow
from braids_desktop.results import ResultsStore
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'smoke_settings'))
app=QApplication([])
for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10))
from braids_desktop._avac.theme import apply_theme
apply_theme(app)
results=[]
for i,item in enumerate(read(REPORT/'PROJECTS.json')):
    w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w.show();w.open_path(Path(item['project']),confirmed=True)
    until=time.monotonic()+60
    while w._tasks and time.monotonic()<until:app.processEvents();time.sleep(.02)
    app.processEvents();assert not w._tasks and not errors,errors
    assert w.forms.turbines.power_planning.isChecked();assert len(w.forms.turbines.rows)==i+1
    assert w._results_output is not None
    w.set_mode('analyze');w.results.load_output(w._results_output);w._refresh_navigation()
    for j in range(w.workflow.count()):
        if w.workflow.item(j).data(Qt.UserRole)=='Turbines':w.workflow.setCurrentRow(j)
    app.processEvents()
    assert w.results.power_estimate_button.isVisible();records=w.results._turbine_rows();assert len(records)==11*(i+1)
    # Actual UI calculation action and plotting, with failures captured instead
    # of a hidden offscreen message box.
    w.results._fail=lambda e:errors.append(str(e));w.results.power_estimate_button.click();app.processEvents();assert not errors,errors
    idx=w.results.turbine_quantity.findData('useful_power_W');w.results.turbine_quantity.setCurrentIndex(idx);w.results.plot_turbine();app.processEvents();assert not errors,errors
    original=QFileDialog.getSaveFileName
    try:
        with tempfile.TemporaryDirectory(prefix='braids_power75_') as tmp:
            for name,action in [('all',lambda:w.results.export_turbine()),('selected',lambda:w.results.export_turbine(selected=True)),('energy',w.results.export_turbine_energy)]:
                destination=Path(tmp)/(name+'.csv');QFileDialog.getSaveFileName=lambda *a,**kw:(str(destination),'CSV (*.csv)');action()
                assert not errors,errors
                with destination.open(encoding='utf-8',newline='') as f:exported=list(csv.DictReader(f))
                assert len(exported)==(11 if name=='selected' else len(records))
    finally:QFileDialog.getSaveFileName=original
    w.grab().save(str(REPORT/(item['case']+'_DESKTOP.png')))
    for dialog in list(w.results._dialogs):dialog.close()
    results.append(dict(project=item['project'],turbines=i+1,rows=len(records),calculation_button=True,plot=True,exports=['all','selected','energy'],errors=errors))
    w.dirty=False;w.close();app.processEvents()
name='INSTALLED_SMOKE.json' if runtime==ROOT/'desktop/runtime' else 'DELIVERY_SMOKE.json'
save(REPORT/name,dict(passed=True,runtime=str(runtime),projects=results));print('All three projects opened, calculated and plotted.')
