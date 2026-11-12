"""Breakline batching and explicit mesh transition-band delivery."""
import argparse,hashlib,json,shutil,subprocess,os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/breaklines57'
TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/mesh_grading57'
CHANGED={'braids_core/boundaries/breaklines.py','braids_core/boundaries/amr.py','braids_core/river_project.py',
         'braids_core/sediment/river_field_policy.py','braids_core/sediment/adaptive_river_worker.py',
         'braids_desktop/forms.py','braids_desktop/__init__.py'}
PARENT='05de54d09337a4d7ff44e86c7ea59e1c5e8bdb01c520e605baca74dce1e8bf53'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')


def verify(root):
    d=read(root/'RUNTIME.json')
    for n,h in d['files'].items():assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return d


def stage():
    assert sha(BASE/'RUNTIME.json')==PARENT;old=verify(BASE)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);TARGET.mkdir(exist_ok=False)
    for n in old['files']:
        p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n if n in CHANGED else BASE/n,p)
    new=dict(old,files={n:sha(TARGET/n) for n in old['files']});save(TARGET/'RUNTIME.json',new);verify(TARGET)
    assert {n for n in old['files'] if old['files'][n]!=new['files'][n]}==CHANGED
    save(OUT/'STAGE.json',dict(parent=PARENT,runtime=sha(TARGET/'RUNTIME.json'),files=sorted(CHANGED)))
    print('Staged preview57',sha(TARGET/'RUNTIME.json'))


def gate():
    data=read(OUT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==data['runtime'];new=verify(TARGET)
    previous=ROOT/'desktop/qualification/turbine_live56/candidate';old=verify(previous)
    assert sha(previous/'RUNTIME.json')==PARENT and set(old['files'])==set(new['files'])
    assert {n for n in old['files'] if old['files'][n]!=new['files'][n]}==CHANGED
    for filename,count in [('FINAL_CHECKS.xml',58),('FINAL_UI.xml',20)]:
        suite=next(ET.parse(OUT/filename).getroot().iter('testsuite'))
        assert int(suite.get('tests'))==count and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    metrics=read(REPORT/'METRICS.json');assert metrics['geometry_identical'] and metrics['geometry_speedup']>1.
    assert metrics['maximum_backend_depth_m']<1e-3 and metrics['maximum_backend_bed_m']<1e-3
    for backend in ['cpu','gpu']:
        for mode in ['hydro','bedload']:
            r=read(OUT/(backend+'_'+mode)/'CHECK.json')
            assert r['status']=='passed' and r['restart_exact'] and r['maximum_water_residual_m3']<1e-8
    assert read(REPORT/'UI.json')['status']=='passed'
    prior=ROOT/'validation/hydrokinetic_turbines/reports/live_hkt4e/FINAL_MANIFEST.json';manifest=read(prior)
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
    save(REPORT/'GATES.json',dict(status='passed',runtime=data['runtime'],parent=PARENT,numerical_checks=58,ui_checks=20,
        native_bindings_unchanged=True,prior_manifest_sha256=sha(prior),prior_entries_verified=len(manifest['files'])))
    return data


def install():
    data=gate();verify(BASE);assert sha(BASE/'RUNTIME.json')==PARENT
    command=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\/](launch|worker)\.py|braids_desktop\.app|desktop[\\/]runtime[\\/]braids_core[\\/]river_launcher\.py')} | Select-Object -ExpandProperty ProcessId"
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if processes:raise RuntimeError('Save and close BRAIDS Desktop/workers before installing: '+processes)
    backup=OUT/'preview56_backup';backup.mkdir(exist_ok=False)
    for n in CHANGED|{'RUNTIME.json'}:
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in CHANGED:shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    d=read(ROOT/'desktop/PROVENANCE.json');report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    d.update(version='0.1.0-preview57',ui_qualification=report)
    d['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=data['runtime'],qualification=report,
        modification_scope='Batched breakline face searches, coordinate-roundoff fragment guard, explicit static/dynamic transition rows. Native hydraulic/sediment kernels unchanged. Project format v39.')
    save(ROOT/'desktop/PROVENANCE.json',d)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview57',runtime=data['runtime'],parent=PARENT,files=sorted(CHANGED),native_bindings_unchanged=True))
    print('Installed preview57')


def smoke():
    done=subprocess.run([str(ROOT/'desktop/.venv/Scripts/python.exe'),'-B',str(ROOT/'desktop/launch.py'),'--smoke'],
        cwd=ROOT,env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),capture_output=True,text=True,timeout=60)
    assert done.returncode==0,done.stdout+done.stderr
    save(REPORT/'SMOKE.json',dict(status='passed',runtime=sha(BASE/'RUNTIME.json'),exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr))
    print('Installed launch smoke passed')


def seal():
    assert not (REPORT/'FINAL_MANIFEST.json').exists();data=gate()
    assert sha(BASE/'RUNTIME.json')==data['runtime'];verify(BASE);assert read(REPORT/'SMOKE.json')['status']=='passed'
    files=[p for p in REPORT.iterdir() if p.is_file()]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(TARGET) and not p.is_relative_to(OUT/'preview56_backup') and 'ui_settings' not in p.parts)
    files.append(TARGET/'RUNTIME.json')
    files.extend(ROOT/'desktop'/n for n in ['profile_breaklines57.py','check_breaklines57.py','analyze_breaklines57.py','deliver_breaklines57.py'])
    files.extend(ROOT/'tests'/n for n in ['test_breakline_grading57.py','test_mesh_grading_ui57.py'])
    save(REPORT/'FINAL_MANIFEST.json',dict(status='passed',runtime=data['runtime'],candidate_manifest_authenticates_runtime_payload=True,editable_example_excluded=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print('Sealed preview57 evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','gate','install','smoke','seal']);globals()[parser.parse_args().action]()
