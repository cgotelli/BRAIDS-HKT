"""Install the qualified preview31, retaining all replaced preview30 payloads."""
import json,shutil,subprocess
from stage_formula190 import ROOT,OUT
from stage_secondary185 import save,sha
from release_formula190 import RELEASE,verify
BASE=ROOT/'desktop/runtime'
PREVIOUS='b171942a93b28a2d5ba13b7e16ba22da09a1424c45f9a632d7707fac455d020f'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_transport190'

def main():
    proof=json.loads((REPORT/'QUALIFICATION.json').read_text())
    assert proof['qualified'] and proof['release_runtime']==sha(RELEASE/'RUNTIME.json')
    assert sha(BASE/'RUNTIME.json')==PREVIOUS
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and its workers before installation: '+running)
    old,new=verify(BASE),verify(RELEASE)
    changed={n for n,h in old['files'].items() if new['files'].get(n)!=h}
    added=set(new['files'])-set(old['files'])
    assert not set(old['files'])-set(new['files']),'No historical payload deletion'
    backup=OUT/'preview30_backup'
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True)
        assert not dst.exists();shutil.copy2(BASE/n,dst)
    for n in sorted(added|changed):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RELEASE/n,dst)
    shutil.copy2(RELEASE/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    save(OUT/'INSTALL.json',dict(previous=PREVIOUS,installed=sha(BASE/'RUNTIME.json'),
        payloads_verified=len(new['files']),backup=str(backup),changed=sorted(changed),added=sorted(added),
        Delft3D_modified=False,QGIS_changed=False,user_projects_changed=False,old_outputs_deleted=False))
    print('INSTALLED preview31',len(new['files']),'verified payloads')
if __name__=='__main__':main()
