"""Install only the qualified S5 package, retaining preview41 for recovery."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_suspended218 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify

def main():
    regression=OUT/'REGRESSION.xml'
    suite=ET.parse(regression).getroot().find('testsuite')
    assert int(suite.attrib['tests'])>=890 and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
    assert regression.stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert set(old['files'])<=set(new['files'])
    changed={n for n,h in old['files'].items() if new['files'][n]!=h}
    added=set(new['files'])-set(old['files'])
    assert (changed|added)-{n for n in added if n.startswith('precompiled/')}==set(FILES)|{'PRECOMPILED.json'}
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    before=json.loads((BASE/'PRECOMPILED.json').read_text());after=json.loads((TARGET/'PRECOMPILED.json').read_text())
    for key,value in before.items():assert after[key]==value,key
    assert len(after)==len(before)+1
    for name in ('EDITOR.json','CANDIDATE_LAUNCH.json'):
        receipt=json.loads((OUT/name).read_text());assert receipt['runtime']==sha(TARGET/'RUNTIME.json'),name
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview41_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview42',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source='validation/_suspended218/candidate',runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),
        qualification=provenance['ui_qualification'],modification_scope=
        'S5 de Leeuw2020 Eq26a independent entrainment with explicit D84 Hsk skin partition, Ferguson-Church settling and preferred concentration-profile fit. '
        'Suspension alone or with all nine ordinary bedload formulas; MF1, one bulk material. '
        'Existing native bindings unchanged; one new shared OpenMP/CUDA kernel. Project v30 rejects old formats. '
        'No new bank/collapse physics, no QGIS or Delft3D changes.')
    save(ROOT/'desktop/PROVENANCE.json',provenance);REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('REGRESSION.xml','EDITOR.json','CANDIDATE_LAUNCH.json','STAGE.json','BUILD.json','PARITY.json','OPERATOR.json','FIGURE_METRICS.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview42',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),added=sorted(added),
        existing_native_bindings_unchanged=True,packaged_checks=int(suite.attrib['tests']),
        user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview42',sha(BASE/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
