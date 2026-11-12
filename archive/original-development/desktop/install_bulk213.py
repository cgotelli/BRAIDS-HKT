"""Promote authenticated measured-diameter release after portable qualification."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_bulk213 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,sha,save,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bulk213'

def main():
    suite=ET.parse(OUT/'REGRESSION.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==405 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped')),suite.attrib
    assert (OUT/'REGRESSION.xml').stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    changed={n for n,h in old['files'].items() if new['files'][n]!=h}
    assert changed==set(FILES)|{'PRECOMPILED.json'},changed
    added=set(new['files'])-set(old['files'])
    assert all(n.startswith('precompiled/') for n in added)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    editor=json.loads((OUT/'EDITOR.json').read_text())
    assert editor['runtime']==sha(TARGET/'RUNTIME.json') and editor['project_roundtrip']
    assert len(editor['formula_tables'])==9
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop and its workers before installation: '+running)
    backup=OUT/'preview37_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('REGRESSION.xml','EDITOR.json','STAGE.json','BUILD.json','COUPLED_PARITY.json'):
        shutil.copy2(OUT/n,REPORT/n)
    for p in (OUT/'figures').glob('*.png'):
        (REPORT/'figures').mkdir(exist_ok=True);shutil.copy2(p,REPORT/'figures'/p.name)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview38',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),added=len(added),payloads=len(new['files']),
        packaged_tests=405,hydraulic_sources_binaries_unchanged=True,user_projects_modified=False,
        QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview38',sha(BASE/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
