"""Promote the verified UI-only release; retain the previous four UI payloads."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_grains212 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,sha,save,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_grains212'

def main():
    suite=ET.parse(OUT/'PACKAGED_REGRESSION.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==619 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped')),suite.attrib
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert set(old['files'])==set(new['files'])
    changed={n for n,h in old['files'].items() if new['files'][n]!=h}
    assert changed==set(FILES)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    editor=json.loads((OUT/'EDITOR_CHECK.json').read_text())
    assert editor['runtime']==sha(TARGET/'RUNTIME.json') and editor['current_prepared_accepted']
    assert len(editor['formula_tables'])==9 and editor['continuation_end_s']==40.
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Close Desktop and its workers before installation: '+running)
    backup=OUT/'preview36_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in changed:shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('PACKAGED_REGRESSION.xml','EDITOR_CHECK.json','STAGE.json'):
        shutil.copy2(OUT/n,REPORT/n)
    for p in (OUT/'figures').glob('*.png'):
        (REPORT/'figures').mkdir(exist_ok=True);shutil.copy2(p,REPORT/'figures'/p.name)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview37',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),payloads=len(new['files']),
        packaged_tests=619,numerical_payloads_unchanged=True,user_projects_modified=False,
        QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview37',sha(BASE/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
