"""Read-only restart handoff and completed-project Desktop inspection."""
import argparse,collections,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation58'
sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))


def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostics():
    records={}
    for name in ['otf_gpu_segment1',*json.loads((HOME/'PROJECTS.json').read_text())]:
        r=json.loads((REPORT/name/('SEGMENT.json' if name=='otf_gpu_segment1' else 'RECEIPT.json')).read_text())
        s=json.loads((Path(r['output'])/'STATUS.json').read_text())
        records[name]={k:s.get(k) for k in ['status','time_s','wall_s','hydraulic','accepted_steps']}
        records[name]['outer_rejection_reasons']=dict(collections.Counter(v['reason'] for v in s.get('rejected_trials',[])))
        records[name]['maximum_rejected_time_s']=max([v['time_s'] for v in s.get('rejected_trials',[])],default=None)
    save(REPORT/'RUN_DIAGNOSTICS.json',records)
    references=ROOT/'validation/hydrokinetic_turbines'
    files=['digitize_hkt3j.py','digitize_mycek_2014.py','vendor/abolghasemi_2019_depth_average.pdf',
           'vendor/mycek_2014.pdf','vendor/render_abolghasemi/figure3_actual.svg',
           'vendor/render_mycek_a13/performance_page14.svg']
    files.extend(str(p.relative_to(references)) for p in (references/'measurements').glob('*') if p.name in {
        'hkt3j_digitization.json','abolghasemi_2019_figure3_U29.csv','mycek_2014_figures6a_7a.csv',
        'mycek_2014_figure_A12_velocity.csv','mycek_2014_figure_A12_digitization.json'})
    save(REPORT/'REFERENCE_IDENTITIES.json',dict(files={str((references/f).relative_to(ROOT)):sha(references/f) for f in files},
        primary_sources=['https://arxiv.org/abs/1810.09827','https://doi.org/10.1016/j.renene.2013.12.036','https://arxiv.org/abs/1506.03611'],
        measured_profile_projection='Conditional axisymmetric integration through the full water column; not directly measured depth means.',
        digitization='Original PDF vector paths; retained source metadata describes axis scaling and digitization allowances.'))
    print(json.dumps(records,indent=2),flush=True)


def restart():
    from braids_desktop.results import ResultsStore
    segment=json.loads((REPORT/'otf_gpu_segment1/SEGMENT.json').read_text())
    resumed=json.loads((REPORT/'otf_gpu/STARTED.json').read_text())
    old=ResultsStore(segment['output']);new=ResultsStore(resumed['output'])
    a=old.read(-1);b=new.read(0)
    assert a.time_s==b.time_s==105. and a.epoch==b.epoch
    assert set(a.arrays)==set(b.arrays)
    exact={n:bool(np.array_equal(a.arrays[n],b.arrays[n])) for n in a.arrays}
    assert all(exact.values()),exact
    assert np.array_equal(a.water_vector,b.water_vector)
    first=json.loads((Path(segment['output'])/'PROJECT.json').read_text())
    second=json.loads((Path(resumed['output'])/'PROJECT.json').read_text())
    unchanged={n:first[n]==second[n] for n in ['model','settings','turbines','initial','roughness_parameters']}
    assert all(unchanged.values()),unchanged
    record=dict(status='passed',accepted_time_s=a.time_s,all_frame_arrays_byte_values_equal=exact,
        physical_and_numerical_settings_equal=unchanged,
        maximum_depth_m=float(np.max(abs(a.q[0]-b.q[0]))),
        maximum_momentum_m2_s=float(np.max(abs(a.q[1:]-b.q[1:]))),
        maximum_bed_m=float(np.max(abs(a.z-b.z))),
        source_output=segment['output'],continued_output=resumed['output'],
        checkpoint=segment['checkpoint'],checkpoint_sha256=sha(segment['checkpoint']),
        reason='Administrative orchestration time limit only; solver, physical inputs and timestep controls unchanged.')
    save(REPORT/'RESTART_CHECK.json',record);print(json.dumps(record),flush=True)


def inspect():
    os.environ['QT_QPA_PLATFORM']='offscreen'
    if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
    from PySide6.QtCore import QSettings,Qt
    from PySide6.QtGui import QFont,QFontDatabase
    from PySide6.QtWidgets import QApplication,QFileDialog
    from braids_desktop.shell import MainWindow
    from braids_desktop._avac.theme import apply_theme
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(REPORT/'ui_settings'))
    app=QApplication([]);app.setOrganizationName('BRAIDSValidation');app.setApplicationName('Turbine58Inspect')
    for f in ['segoeui.ttf','segoeuib.ttf','seguisym.ttf']:QFontDatabase.addApplicationFont('C:/Windows/Fonts/'+f)
    app.setFont(QFont('Segoe UI',10));apply_theme(app)
    window=MainWindow(runtime=RUNTIME);errors=[];window.error=lambda e:errors.append(str(e))
    window.show();records=[]
    def wait(predicate):
        end=time.monotonic()+120
        while not predicate():
            app.processEvents();time.sleep(.02)
            assert not errors,errors
            if time.monotonic()>end:raise TimeoutError('Completed project reopen inspection')
    try:
        for name,row in json.loads((HOME/'PROJECTS.json').read_text()).items():
            receipt=json.loads((REPORT/name/'RECEIPT.json').read_text())
            window.open_path(Path(row['project']),confirmed=True)
            wait(lambda:window.document_base is not None and not window._tasks)
            assert window.forms.restart.text()==''
            assert Path(window._results_output).resolve()==Path(receipt['output']).resolve()
            window.analyze_button.click();wait(lambda:window.results.store is not None and not window._tasks)
            store=window.results.store;assert store.times[-1]==row['configuration']['end']
            assert window.results.measurement_state()['profiles']
            for i in range(window.workflow.count()):
                if window.workflow.item(i).data(Qt.UserRole)=='Turbines':window.workflow.setCurrentRow(i)
            app.processEvents();rows=window.results._turbine_rows();assert rows[-1]['time_s']==store.times[-1]
            target=REPORT/name/'UI_EXPORTED_PERFORMANCE.csv'
            original=QFileDialog.getSaveFileName
            try:
                QFileDialog.getSaveFileName=staticmethod(lambda *a,**kw:(str(target),'CSV (*.csv)'))
                window.results.export_turbine()
            finally:QFileDialog.getSaveFileName=original
            assert target.read_bytes()==(Path(receipt['output'])/'TURBINE_PERFORMANCE.csv').read_bytes()
            with target.open(encoding='utf-8') as f:exported=list(csv.DictReader(f))
            assert len(exported)==len(rows)
            if name=='mycek_cpu':
                window.results.slider.setValue(len(store.times)-1)
                app.processEvents();window.grab().save(str(REPORT/'COMPLETED_PROJECT.png'))
            records.append(dict(case=name,latest_output_reopened=True,checkpoint_not_automatically_selected=True,
                seeded_profiles_present=True,turbine_tab=True,csv_exact=True,csv_rows=len(exported),end_time_s=store.times[-1]))
            print('Inspected',name,flush=True)
        assert not errors
        save(REPORT/'PROJECT_REOPEN_CHECK.json',dict(status='passed',errors=errors,projects=records))
    finally:
        window.dirty=False;window.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--restart-only',action='store_true');p.add_argument('--diagnostics-only',action='store_true');a=p.parse_args()
    if a.diagnostics_only:diagnostics()
    else:
        restart()
        if not a.restart_only:inspect()
