"""Exercise the real Desktop shell and diagnostic map, without a new run."""
import os,sys,time,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
from particle_bypass78 import ROOT,TARGET,REPORT,read,save
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else TARGET
sys.path[:0]=[str(runtime),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase,QFont
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'smoke_settings'))
app=QApplication([])
for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10));apply_theme(app)
projects=[str(Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern.braids.json')]
results=[]
for i,project in enumerate(projects):
    w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w.show();w.open_path(Path(project),confirmed=True)
    until=time.monotonic()+60
    while w._tasks and time.monotonic()<until:app.processEvents();time.sleep(.02)
    assert not w._tasks and not errors,errors
    d=w.forms.document();assert len(d['turbines']['turbines'])==3 and 'power_plan' not in d
    w.layers['Mesh'].setChecked(False);w.set_mode('analyze');w.results.load_output(w._results_output);w._refresh_navigation()
    w.results._fail=lambda e:errors.append(str(e));store=w.results.store
    assert len(store.hub_variables)==3
    w.results.slider.setValue(len(store.times)-1)
    for key in store.hub_variables:
        idx=w.results.variable.findData(key);assert idx>=0
        w.results.variable.setCurrentIndex(idx);app.processEvents();assert w.results.show_frame() and not errors,errors
        assert w.map.default_palette('result:'+key)=='Reds'
        assert 'near wake masked' in w.results.caption.text()
        w.results.set_particles(True);app.processEvents()
        if w.results.particles:assert not w.results.particles.timer.isActive()
    frame=store.read(-1);c=frame.polygons[len(frame.polygons)//2].centroid
    profile=store.spatial_profile([[c.x-1,c.y],[c.x+1,c.y]],key,-1,interpolate=True)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'hub_profile.csv';profile.export_csv(p);assert p.stat().st_size>40
    w.results.variable.setCurrentIndex(w.results.variable.findData('velocity_m_s'));app.processEvents()
    w.results.set_particles(True);track=w.results.particles;assert track.timer.isActive()
    import numpy as np
    before=track.frame.q.copy();start=time.monotonic()
    for _ in range(10):track.advance()
    elapsed=time.monotonic()-start;np.testing.assert_array_equal(track.frame.q,before)
    save(REPORT/'PARTICLE_TIMING.json',dict(ticks=10,particles=len(track.positions),elapsed_s=elapsed,seconds_per_tick=elapsed/10))
    w.grab().save(str(REPORT/('BERN_DESKTOP.png' if i==0 else 'LAB_DESKTOP.png')))
    results.append(dict(project=project,hub_planes=3,profile_export=True,particle_overlay_suppressed=True,errors=errors))
    w.dirty=False;w.close();app.processEvents()
save(REPORT/('INSTALLED_SMOKE.json' if runtime==ROOT/'desktop/runtime' else 'DELIVERY_SMOKE.json'),dict(passed=True,runtime=str(runtime),projects=results))
print('Modified Bern Desktop particles, hub planes and profile export passed.')
