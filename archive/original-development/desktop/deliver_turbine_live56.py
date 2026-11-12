"""UI-only live turbine delivery; the authenticated preview55 solver is unchanged."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'desktop/qualification/turbine_live56';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/live_hkt4e'
CHANGED={'braids_desktop/shell.py','braids_desktop/results.py','braids_desktop/__init__.py'}
PARENT='ba9a8185a231c808f1197d6f7de7a93eeca7443cb40f74229491e3ec753ab937'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')


def verify(root):
    d=read(root/'RUNTIME.json')
    for n,h in d['files'].items():assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return d


def stage():
    assert sha(BASE/'RUNTIME.json')==PARENT
    old=verify(BASE);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);TARGET.mkdir(exist_ok=False)
    for n in old['files']:
        p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n if n in CHANGED else BASE/n,p)
    new=dict(old,files={n:sha(TARGET/n) for n in old['files']})
    save(TARGET/'RUNTIME.json',new);verify(TARGET)
    assert {n for n in old['files'] if old['files'][n]!=new['files'][n]}==CHANGED
    save(OUT/'STAGE.json',dict(parent=PARENT,runtime=sha(TARGET/'RUNTIME.json'),files=sorted(CHANGED)))
    print('Staged preview56',sha(TARGET/'RUNTIME.json'))


def gate():
    data=read(OUT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==data['runtime'];new=verify(TARGET)
    previous=ROOT/'desktop/qualification/turbine_array55/candidate';old=verify(previous)
    assert sha(previous/'RUNTIME.json')==PARENT
    assert {n for n in old['files'] if old['files'][n]!=new['files'][n]}==CHANGED
    assert set(old['files'])==set(new['files'])
    suite=next(ET.parse(OUT/'FINAL_UI.xml').getroot().iter('testsuite'))
    assert int(suite.get('tests'))==17 and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    smoke=read(REPORT/'UI_RUN.json');assert smoke['status']=='passed' and smoke['live_plot'] and smoke['live_export'] and smoke['tab_before_first_frame']
    assert smoke['end_time_s']==3. and smoke['turbines']==3 and smoke['errors']==[]
    previous_manifest=REPORT.parent/'array_hkt4d/FINAL_MANIFEST.json';manifest=read(previous_manifest)
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
    save(REPORT/'GATES.json',dict(status='passed',runtime=data['runtime'],parent=PARENT,ui_tests=17,solver_payload_unchanged=True,
        native_bindings_unchanged=True,prior_manifest_sha256=sha(previous_manifest),prior_entries_verified=len(manifest['files'])))
    return data


def install():
    data=gate();verify(BASE);assert sha(BASE/'RUNTIME.json')==PARENT
    command=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\/](launch|worker)\.py|braids_desktop\.app|desktop[\\/]runtime[\\/]braids_core[\\/]river_launcher\.py')} | Select-Object -ExpandProperty ProcessId"
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if processes:raise RuntimeError('Save and close BRAIDS Desktop/workers before installing: '+processes)
    backup=OUT/'preview55_backup';backup.mkdir(exist_ok=False)
    for n in CHANGED|{'RUNTIME.json'}:
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in CHANGED:shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    d=read(ROOT/'desktop/PROVENANCE.json');report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    d.update(version='0.1.0-preview56',ui_qualification=report)
    d['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=data['runtime'],qualification=report,
        modification_scope='UI-only live turbine navigation and verified plotting/export snapshots. The preview55 hydraulic, turbine, geometry and native solver payload is byte-identical.')
    save(ROOT/'desktop/PROVENANCE.json',d)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview56',runtime=data['runtime'],parent=PARENT,files=sorted(CHANGED),solver_payload_unchanged=True))
    print('Installed preview56')


def smoke():
    import os
    environment=dict(os.environ,QT_QPA_PLATFORM='offscreen')
    done=subprocess.run([str(ROOT/'desktop/.venv/Scripts/python.exe'),'-B',str(ROOT/'desktop/launch.py'),'--smoke'],
        cwd=ROOT,env=environment,capture_output=True,text=True,timeout=60)
    assert done.returncode==0,done.stdout+done.stderr
    save(REPORT/'SMOKE.json',dict(status='passed',version='0.1.0-preview56',runtime=sha(BASE/'RUNTIME.json'),
        exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr))
    print('Installed launch smoke passed')


def seal():
    assert not (REPORT/'FINAL_MANIFEST.json').exists();data=gate()
    assert sha(BASE/'RUNTIME.json')==data['runtime'];verify(BASE)
    assert read(REPORT/'SMOKE.json')['status']=='passed'
    files=[p for p in REPORT.iterdir() if p.is_file()]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(TARGET) and not p.is_relative_to(OUT/'preview55_backup') and 'ui_settings' not in p.parts)
    files.extend([TARGET/'RUNTIME.json',ROOT/'desktop/deliver_turbine_live56.py',ROOT/'desktop/check_turbine_live56.py',ROOT/'tests/test_turbine_live56.py'])
    save(REPORT/'FINAL_MANIFEST.json',dict(status='passed',runtime=data['runtime'],candidate_manifest_authenticates_runtime_payload=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print('Sealed preview56 evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','gate','install','smoke','seal'])
    globals()[parser.parse_args().action]()
