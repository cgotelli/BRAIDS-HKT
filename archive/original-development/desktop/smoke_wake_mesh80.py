"""Read-only reproduction using the user's saved 10D / 20D preparations."""
import os,sys,time,hashlib
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
from wake_mesh80 import ROOT,TARGET,REPORT,read,save,sha
runtime=Path(sys.argv[1]) if len(sys.argv)>1 else TARGET
sys.path[:0]=[str(runtime),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase,QFont
from braids_desktop.shell import MainWindow
from braids_desktop import services
from braids_desktop._avac.theme import apply_theme
from braids_core.river_turbines import wake_footprint
QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'smoke_settings'))
app=QApplication([])
for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
app.setFont(QFont('Segoe UI',10));apply_theme(app)
project=Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern.braids.json';original=project.read_bytes()
w=MainWindow(runtime=runtime);errors=[];w.error=lambda e:errors.append(str(e));w.show();w.open_path(project,confirmed=True)
def wait():
    end=time.monotonic()+60
    while w._tasks and time.monotonic()<end:app.processEvents();time.sleep(.02)
    assert not w._tasks and not errors,errors
wait();w.set_mode('model_run');wait()
parent=project.parent/'Bern_data/prepared'
paths=[parent/n for n in ['prepare_20261101_222117_579c4b56','prepare_20261101_222900_21117d8c','prepare_20261101_223159_6440658b']]
records=[];polys=[];path_hashes=[];keys=[]
for prepared,count,length in zip(paths,[9853,10312,9853],[10,20,10]):
    doc=read(prepared/'PROJECT.json');w.forms.load_document(doc);w.prepared=prepared
    w.layers['Mesh'].setChecked(True);w._show_prepared_mesh();wait()
    assert w._preview_current and w.map.preview_mesh_item.isVisible()
    assert f'{count:,} cells' in w.status.text(),w.status.text()
    assert prepared.name in w.log.toPlainText()
    path=w.map.preview_mesh_item.path()
    pts=[(path.elementAt(i).x,path.elementAt(i).y,path.elementAt(i).type.value) for i in range(path.elementCount())]
    path_hashes.append(hashlib.sha256(np.asarray(pts).tobytes()).hexdigest());keys.append(w._mesh_key())
    p,faces=services.prepared_mesh(prepared);assert len(p)==count
    rows=doc['turbines']['turbines'];assert rows[0]['wake_refinement']['length_diameters']==length
    assert all(r['wake_refinement']['length_diameters']==10 for r in rows[1:])
    records.append(dict(prepared=str(prepared),cells=count,faces=faces,mesh_sha256=sha(prepared/'MESH_PREVIEW.npz'),wake_lengths_D=[r['wake_refinement']['length_diameters'] for r in rows]))
    polys.append(p)
assert path_hashes[0]!=path_hashes[1] and path_hashes[0]==path_hashes[2]
assert keys[0]!=keys[1] and keys[0]==keys[2]
e=w.forms.turbines;e.selection.setCurrentIndex(0);e.wake_length.setValue(20)
assert not w._preview_current and not w.map.preview_mesh_item.isVisible()
assert '20D = 20 m' in e.wake_summary.text()
e.selection.setCurrentIndex(1);assert e.wake_length.value()==10
e.selection.setCurrentIndex(0);assert e.wake_length.value()==20
w._select_step('Hydrokinetic turbines');app.processEvents();wait()
w.forms.stack.currentWidget().ensureWidgetVisible(e.wake_summary);app.processEvents()
w.grab().save(str(REPORT/'WAKE_EDITOR.png'))
w.dirty=False;w.close();app.processEvents();assert project.read_bytes()==original
# Compare saved meshes on identical axes, no smoothing or regenerated geometry.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm,ListedColormap
from shapely.geometry import Polygon
rows=read(paths[0]/'PROJECT.json')['turbines']['turbines']
origin=np.asarray(rows[0]['center_xy'])
cmap=ListedColormap(['#377eb8','#9ecae1','#d9e8c1','#f5edd5','#eeeeee']);norm=BoundaryNorm([0,.3,.6,1.1,2.1,20],5)
fig,axes=plt.subplots(1,2,figsize=(10,9),sharex=True,sharey=True,constrained_layout=True)
for j,ax in enumerate(axes):
    local=[np.asarray(p)-origin for p in polys[j]]
    sizes=[np.sqrt(Polygon(p).area) for p in polys[j]]
    coll=PolyCollection(local,array=np.asarray(sizes),cmap=cmap,norm=norm,edgecolors='#555555',linewidths=.18);ax.add_collection(coll)
    doc=read(paths[j]/'PROJECT.json')
    for k,r in enumerate(doc['turbines']['turbines']):
        q=np.asarray(wake_footprint(r).exterior.coords);q=np.vstack([q,q[0]])-origin
        ax.plot(q[:,0],q[:,1],color='#d95f02' if k==0 else '#666666',ls='--',lw=1.5)
        c=np.asarray(r['center_xy'])-origin;ax.plot(*c,'ko',ms=3)
    ax.set(xlim=(-8,8),ylim=(-4,28),aspect='equal',xlabel='East offset (m)',title=f'West upstream: {[10,20][j]}D\n{records[j]["cells"]:,} total cells')
axes[0].set_ylabel('North offset (m)')
bar=fig.colorbar(coll,ax=axes,label='Cell area equivalent square size (m)',ticks=[.15,.45,.85,1.6,3.5],shrink=.55)
bar.ax.set_yticklabels(['0.25','0.5','1','2','4'])
fig.suptitle('Saved Bern mesh: only West upstream changes\nDashed orange: requested wake refinement; other turbines remain 10D')
fig.savefig(REPORT/'MESH_10D_20D.png',dpi=170);plt.close(fig)
result=dict(passed=True,runtime=str(runtime),saved_preparations=records,display_geometry_hashes=path_hashes,old_preview_hidden_on_edit=True,per_turbine_edit_roundtrip=True,user_project_unchanged=True,errors=errors)
save(REPORT/('INSTALLED_SMOKE.json' if runtime==ROOT/'desktop/runtime' else 'DELIVERY_SMOKE.json'),result)
save(REPORT/'METRICS.json',result)
print('Saved 10D / 20D / 10D meshes display correctly; selected-row changes and invalidation pass.')


