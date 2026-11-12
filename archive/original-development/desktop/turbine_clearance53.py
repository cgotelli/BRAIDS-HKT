"""Stage/install the explicit turbine-clearance update; preserve preview52."""
import hashlib,json,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_clearance53';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/clearance_hkt4b'
PARENT='77f2f5a98f09aa85062d0749e7146f8b6f340a4166aa0d961f7a14532894fe8f'
FILES=['braids_core/river_turbines.py','braids_core/river_project.py','braids_core/sediment/river_turbine_flow.py','braids_desktop/turbines.py','braids_desktop/results.py','braids_desktop/__init__.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def verify(root):
    d=json.loads((root/'RUNTIME.json').read_text())
    for n,h in d['files'].items():assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return d


def stage():
    assert sha(BASE/'RUNTIME.json')==PARENT;old=verify(BASE)
    if not TARGET.exists():
        for n in old['files']:
            p=TARGET/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    m=json.loads(json.dumps(old))
    for n in FILES:shutil.copy2(ROOT/n,TARGET/n);m['files'][n]=sha(TARGET/n)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    assert sha(TARGET/'PRECOMPILED.json')==sha(BASE/'PRECOMPILED.json')
    save(OUT/'STAGE.json',dict(parent=PARENT,runtime=sha(TARGET/'RUNTIME.json'),files={n:sha(ROOT/n) for n in FILES},native_bindings_unchanged=True));print(TARGET)


def install():
    old=verify(BASE);new=verify(TARGET);assert sha(BASE/'RUNTIME.json')==PARENT
    checks=json.loads((REPORT/'GATES.json').read_text());assert checks['status']=='passed' and checks['runtime']==sha(TARGET/'RUNTIME.json')
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|braids_desktop.app|desktop[\\\\/]runtime[\\\\/]braids_core[\\\\/]river_launcher.py')} | Select-Object -ExpandProperty ProcessId"
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if processes:raise RuntimeError('Save and close installed BRAIDS Desktop before replacing it: '+processes)
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};assert changed==set(FILES) and set(old['files'])==set(new['files'])
    backup=OUT/'preview52_backup';assert not backup.exists()
    for n in changed|{'RUNTIME.json'}:
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in changed:shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    p=ROOT/'desktop/PROVENANCE.json';d=json.loads(p.read_text());report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    d.update(version='0.1.0-preview53',ui_qualification=report);d['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),qualification=report,modification_scope='Explicit independent turbine minimum bed/surface clearance limits, inclusive positive thresholds and fixed hub. Current project v38; no old-format migration. Same native hydraulic and actuator arithmetic.');save(p,d)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview53',runtime=sha(BASE/'RUNTIME.json'),parent=PARENT,files=sorted(changed),native_bindings_unchanged=True,independent_runtimes_modified=False));print('Installed preview53')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','install']);getattr(__import__(__name__),p.parse_args().action)()
