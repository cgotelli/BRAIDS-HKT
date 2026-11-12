"""Raw finite-volume comparison, plus standalone Qt layout evidence."""
import os,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_solids151'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'candidate'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
fig,axes=plt.subplots(2,2,figsize=(12,6.7),layout='constrained');comparison={}
for i,case in enumerate(('wetfront','bedload')):
    states=[]
    for j,backend in enumerate(('cpu','gpu')):
        store=ResultsStore(OUT/(case+'_'+backend)/'run');frame=store.read();first=store.read(0);states.append(frame)
        values=frame.q[0] if case=='wetfront' else 1000*(frame.z-first.z)
        ax=axes[i,j];collection=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],
            array=values,cmap='Blues' if case=='wetfront' else 'RdBu_r',edgecolors='#555555',linewidths=.3)
        if j==0:bound=max(float(abs(values).max()),1e-8)
        collection.set_clim(0,.8) if case=='wetfront' else collection.set_clim(-bound,bound)
        ax.add_collection(collection)
        for solid in store.project['settings']['solids']:
            p=np.asarray(solid['points_xy']);ax.fill(p[:,0],p[:,1],color='#8c939a',zorder=3)
        ax.set_xlim(0,20);ax.set_ylim(0,8);ax.set_aspect('equal');ax.set_title(f'{backend.upper()} · {case} · {frame.time_s:g} s')
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.colorbar(collection,ax=ax,label='Depth (m)' if case=='wetfront' else 'Bed change (mm)',shrink=.8)
    a,b=states
    np.testing.assert_array_equal(a.mesh['leaves'],b.mesh['leaves'])
    comparison[case]={k:float(abs(a.arrays[k]-b.arrays[k]).max()) for k in ('q','z','mass','water','bed_boundary')}
    if case=='bedload':comparison[case]['bed_change_maximum_m']=float(abs(a.z-ResultsStore(OUT/'bedload_cpu/run').read(0).z).max())
    assert comparison[case]['q']<1e-9 and comparison[case]['z']<1e-10
fig.suptitle('Impermeable polygon and ellipse; raw cells, identical physical configuration')
fig.savefig(OUT/'COMPARISON.png',dpi=160);plt.close(fig)
(OUT/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n');print(json.dumps(comparison,indent=2))

os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QSettings
from braids_desktop.shell import MainWindow
from braids_desktop._avac.theme import apply_theme
from braids_desktop import services,preparation
app=QApplication([])
for font in ('segoeui.ttf','seguisym.ttf','segmdl2.ttf','seguiemj.ttf'):QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+font)
with tempfile.TemporaryDirectory(prefix='braids-solids-ui-') as temporary:
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,temporary)
    apply_theme(app,'light')
    w=MainWindow(runtime=OUT/'candidate');errors=[];w.error=lambda exc:errors.append(str(exc))
    w._background=lambda work,done,label:done(work())
    w.open_path(OUT/'wetfront_cpu',confirmed=True);w.prepared=OUT/'wetfront_cpu/prepared/current'
    w._show_prepared_mesh();w._select_step('BC & IC');w.resize(1440,960);w.show();app.processEvents()
    w.geometry_action('zoom_domain');app.processEvents();w.grab().save(str(OUT/'BC_IC.png'))
    w._select_step('Mesh');app.processEvents();w.grab().save(str(OUT/'MESH.png'))
    assert not errors,errors
    w.dirty=False;w.close()
