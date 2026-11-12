"""Promote qualified S1 UI only; retain preview38 and every numerical payload."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_suspended214 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify


def main():
    suite=ET.parse(OUT/'REGRESSION.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==607 and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
    assert (OUT/'REGRESSION.xml').stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    changed={n for n,h in old['files'].items() if new['files'][n]!=h}
    added=set(new['files'])-set(old['files'])
    assert changed|added==set(FILES)
    assert added=={'braids_core/sediment/transport_setup.py'}
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    editor=json.loads((OUT/'EDITOR.json').read_text())
    assert editor['runtime']==sha(TARGET/'RUNTIME.json') and editor['pending_draft_blocked']
    assert editor['bedload_document_unchanged'] and editor['existing_project_roundtrip']
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview38_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview39',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source='validation/_suspended214/candidate',runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),
        qualification=provenance['ui_qualification'],modification_scope=
        'S1 independent suspended configuration drafts, combined measured diameter contracts and four-tab sediment UI. '
        'Suspension remains blocked before preparation. All numerical payloads, hydraulic and sediment laws, '
        'CPU/GPU binaries and bedload-only resolved models are unchanged from preview38. '
        '607 packaged checks pass. No new morphodynamic feature or long simulation; no QGIS/Delft3D changes.')
    save(ROOT/'desktop/PROVENANCE.json',provenance)
    REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('REGRESSION.xml','EDITOR.json','STAGE.json','COUPLED_PARITY.json'):
        shutil.copy2(OUT/n,REPORT/n)
    for p in (OUT/'figures').glob('*.png'):
        (REPORT/'figures').mkdir(exist_ok=True);shutil.copy2(p,REPORT/'figures'/p.name)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview39',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),added=sorted(added),
        numerical_payloads_unchanged=True,suspended_execution_admitted=False,
        packaged_checks=607,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview39',sha(BASE/'RUNTIME.json'),flush=True)


if __name__=='__main__':main()
