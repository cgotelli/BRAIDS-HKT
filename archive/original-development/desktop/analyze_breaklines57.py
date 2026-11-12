"""Measured mesh setup comparison and refinement-band figures."""
import json,sys,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/breaklines57'
TARGET=OUT/'candidate';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/mesh_grading57'
sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT))


def last(folder):
    row=json.loads((folder/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(folder/'frames'/row['file']) as a:return {k:a[k].copy() for k in a.files}


def figures():
    from braids_core.boundaries.breaklines import build_breakline_hierarchy
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    d=PolygonDomain('figure',[[0.,0.],[128.,0.],[128.,64.],[0.,64.]],'EPSG:2056')
    tree=build_breakline_hierarchy(d,[],NativeGrid(16,8,0.,0.,8.,8.),[],levels=4)
    xy=np.array([c.centroid_xy for c in tree.finest.cells]);wanted=np.zeros(tree.count,int)
    wanted[tree.offsets[-2]+np.flatnonzero((xy[:,0]>63)&(xy[:,0]<64)&(xy[:,1]>28)&(xy[:,1]<36))]=3
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),sharex=True,sharey=True,layout='constrained');counts={}
    palette=['#dbe9f6','#a3cbe7','#549bd0','#175d9c']
    for ax,n in zip(axes,[1,2,3]):
        mesh=tree.select(tree.balanced_leaves(wanted,transition_layers=n));counts[n]=np.bincount([c.level for c in mesh.cells],minlength=4).tolist()
        ax.add_collection(PolyCollection([c.vertices_xy for c in mesh.cells],facecolors=[palette[c.level] for c in mesh.cells],edgecolors='#566573',linewidths=.28))
        ax.plot([63.5,63.5],[28,36],color='#de3b2b',lw=3,label='Finest request')
        ax.set(xlim=(25,102),ylim=(8,56),aspect='equal',title=f'{n} transition row(s) · {len(mesh.cells)} cells',xlabel='x (m)')
    axes[0].set_ylabel('y (m)');axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Same 1 m finest size; 2:1 levels: 1, 2, 4, 8 m\nBands use face neighbours at each level; nearby bands can overlap')
    fig.savefig(REPORT/'REFINEMENT_BANDS.png',dpi=170);plt.close(fig)
    return counts


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    before=json.loads((OUT/'baseline_large_CLEAN.json').read_text());after=json.loads((OUT/'candidate_large_CLEAN.json').read_text())
    a=float(np.median([r['seconds'] for r in before]));b=float(np.median([r['seconds'] for r in after]))
    metrics=dict(geometry_baseline_median_s=a,geometry_candidate_median_s=b,geometry_speedup=a/b,geometry_time_reduction_percent=100*(1-b/a),
        geometry_identical=(OUT/'baseline_large_TREE.json').read_bytes()==(OUT/'candidate_large_TREE.json').read_bytes(),
        benchmark_finest_cells=after[0]['counts'][-1],benchmark_levels=4,benchmark_oblique_breaklines=2,clean_repeats=3)
    backend=[]
    for mode in ['hydro','bedload']:
        cpu=last(OUT/('cpu_'+mode)/'run');gpu=last(OUT/('gpu_'+mode)/'run')
        assert cpu['q'].shape==gpu['q'].shape
        backend.append(dict(mode=mode,maximum_depth_m=float(np.max(abs(cpu['q'][0]-gpu['q'][0]))),maximum_bed_m=float(np.max(abs(cpu['z']-gpu['z'])))))
        for engine in ['cpu','gpu']:
            diagnostics=json.loads((OUT/(engine+'_'+mode)/'run/DIAGNOSTICS.json').read_text())
            assert max(abs(r['class_residual_kg'][0]) for r in diagnostics)<1e-8
        if mode=='bedload':assert np.max(abs(cpu['bedload_vector_m2_s']))>0 and np.max(abs(gpu['bedload_vector_m2_s']))>0
    metrics.update(backend=backend,maximum_backend_depth_m=max(r['maximum_depth_m'] for r in backend),maximum_backend_bed_m=max(r['maximum_bed_m'] for r in backend),band_figure_counts=figures())
    (REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics),flush=True)
    fig,ax=plt.subplots(figsize=(6.5,3.7),layout='constrained');ax.bar(['preview56','preview57'],[a,b],color=['#8c9dab','#3681ac'])
    ax.set(ylabel='Median geometry time (s)',title=f'Two oblique breaklines · four levels · {after[0]["counts"][-1]:,} finest cells')
    for x,y in enumerate([a,b]):ax.text(x,y+.05,f'{y:.2f} s',ha='center')
    ax.set_ylim(0,a*1.2);ax.grid(axis='y',alpha=.2);fig.savefig(REPORT/'GEOMETRY_TIMING.png',dpi=170);plt.close(fig)


def ui():
    import time
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QFont,QFontDatabase
    from braids_desktop import services,project_file
    from braids_desktop.shell import MainWindow
    from braids_desktop._avac.theme import apply_theme
    root=ROOT/'desktop/examples/refinement57';root.mkdir(exist_ok=False)
    doc=json.loads((OUT/'cpu_hydro/input.json').read_text());doc.pop('editor_state',None)
    doc['name']='Refinement';doc['settings'].update(end_time_s=30.,output_interval_s=1.,checkpoint_interval_s=30.,dt_max_s=.5)
    saved=services.save_project_folder(root/'Refinement_data',doc);entry=root/'Refinement.braids.json'
    project_file.write(entry,root/'Refinement_data',saved)
    QSettings.setDefaultFormat(QSettings.IniFormat);QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSValidation');app.setApplicationName('Mesh57')
    for name in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+name)
    app.setFont(QFont('Segoe UI',10));apply_theme(app)
    window=MainWindow(runtime=TARGET);errors=[];window.error=lambda e:errors.append(str(e));window.show()
    start=time.monotonic()
    def wait_for(predicate):
        while not predicate():
            app.processEvents();time.sleep(.02)
            if errors:raise AssertionError(errors)
            if time.monotonic()-start>180:raise TimeoutError('UI preparation timeout')
    try:
        window.open_path(entry,confirmed=True);wait_for(lambda:window.document_base is not None and not window._tasks)
        window._select_step('Mesh');control=window.forms.adaptation['transition_layers'];assert control.value()==3 and control.isEnabled()
        for mode in ['static_terrain','dynamic_depth']:
            window.forms.mesh_mode.setCurrentIndex(window.forms.mesh_mode.findData(mode));assert control.isEnabled()
        window.prepare_button.click();wait_for(lambda:window.runner.action=='prepare');wait_for(lambda:not window.runner.is_running and not window._tasks)
        window.layers['Mesh'].setChecked(True);wait_for(lambda:window.map.preview_mesh_item is not None and not window._tasks)
        window.geometry_action('zoom_domain');app.processEvents();window.grab().save(str(REPORT/'MESH_CONTROLS.png'))
        record=dict(status='passed',project=str(entry.relative_to(ROOT)),transition_layers=3,enabled_in_both_modes=True,
            preparation=json.loads((window.prepared/'DESKTOP_PREPARATION.json').read_text()),errors=errors,
            actual_ui_open_save_prepare_and_preview=True,simulation_launched=False)
        (REPORT/'UI.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    finally:
        if window.runner.is_running:window.runner.stop();window.runner.process.waitForFinished(20000)
        window.dirty=False;window.close()


if __name__=='__main__':ui() if '--ui' in sys.argv else main()
