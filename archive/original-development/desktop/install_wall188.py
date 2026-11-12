"""Install only the sealed preview30 payload, preserving preview29 and all runs."""
import json,shutil,subprocess
from pathlib import Path
from stage_secondary185 import ROOT,save,sha
OUT=ROOT/'validation/_wall188';TARGET=OUT/'desktop_candidate';BASE=ROOT/'desktop/runtime'
EXPECTED='b171942a93b28a2d5ba13b7e16ba22da09a1424c45f9a632d7707fac455d020f'
PREVIOUS='fddbf4c03a89f4d19aa72e89dc3fb02967470289d0621c1c47d7ab9a73f71ae3'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_wall188'


def verify(root):
    result=json.loads((root/'RUNTIME.json').read_text())
    for n,h in result['files'].items():
        p=(root/n).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,n
    return result


def main():
    proof=json.loads((REPORT/'QUALIFICATION.json').read_text())
    assert proof['qualified'] and proof['runtime']==EXPECTED
    assert sha(TARGET/'RUNTIME.json')==EXPECTED and sha(BASE/'RUNTIME.json')==PREVIOUS
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and its workers before installation: '+running)
    old,new=verify(BASE),verify(TARGET)
    changed={n for n,h in old['files'].items() if new['files'].get(n)!=h}
    added=set(new['files'])-set(old['files'])
    assert not set(old['files'])-set(new['files']),'Preserve historical payload files'
    backup=OUT/'preview29_backup'
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(BASE/n)
        else:shutil.copy2(BASE/n,dst)
    # Preserve the previous live editor files before the separately reviewed
    # source patch. New imports use the fully authenticated packaged editor.
    for n in ('braids_desktop/forms.py','braids_desktop/river_boundaries.py','braids_desktop/__init__.py'):
        dst=backup/'checkout'/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(ROOT/n)
        else:shutil.copy2(ROOT/n,dst)
    for n in sorted(added|changed):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    save(OUT/'INSTALL.json',dict(previous=PREVIOUS,installed=EXPECTED,payloads_verified=len(new['files']),
        backup=str(backup),changed=sorted(changed),added=sorted(added),Delft3D_modified=False,
        QGIS_changed=False,user_projects_changed=False,old_simulation_outputs_deleted=False))
    print('INSTALLED preview30',EXPECTED,len(new['files']),'verified payloads')


if __name__=='__main__':main()
