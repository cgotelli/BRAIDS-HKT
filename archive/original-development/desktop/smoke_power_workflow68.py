"""Open the actual current Desktop project and verify power UI/results."""
import os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];runtime=Path(sys.argv[1]);project=Path(sys.argv[2]);report=Path(sys.argv[3])
sys.path[:0]=[str(runtime),str(ROOT)];os.environ['QT_QPA_PLATFORM']='offscreen'
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from braids_desktop.shell import MainWindow
from braids_desktop.results import ResultsStore
app=QApplication([]);QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(report.parent/'smoke_settings'))
w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w.show();w.open_path(project,confirmed=True)
deadline=time.monotonic()+60
while w._tasks and time.monotonic()<deadline:app.processEvents();time.sleep(.02)
app.processEvents();assert not w._tasks and not errors,errors
assert len(w.forms.turbines.rows)==2 and w.forms.turbines.control.currentData()=='fixed_rpm_surface'
assert w.forms.turbines.surface.operation.isChecked()
assert w._results_output is not None
w.results.store=ResultsStore(w._results_output);rows=w.results._turbine_rows();assert len(rows)==82
assert w.results.turbine_quantity.findData('electrical_energy_Wh')>=0
report.write_text(json.dumps(dict(passed=True,project=str(project),runtime=str(runtime),turbines=2,performance_rows=len(rows),errors=errors),indent=2))
w.dirty=False;w.close();app.processEvents();print('Desktop project and turbine results opened')
