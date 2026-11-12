"""Install only the qualified measured-Ds supplied-suspension release."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_supplied220 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify

def main():
    regression=OUT/'REGRESSION_COMPLETE.xml';suite=ET.parse(regression).getroot().find('testsuite')
    completion=json.loads((OUT/'COMPLETION.json').read_text())
    assert completion['runtime']==sha(TARGET/'RUNTIME.json')
    assert completion['combined_sha256']==sha(regression)
    for row in completion['inputs']:assert sha(OUT/row['file'])==row['sha256']
    for name,value in completion['source_sha256'].items():assert sha(ROOT/name)==value,name
    assert int(suite.attrib['tests'])==1158 and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
    assert regression.stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert set(old['files'])<=set(new['files'])
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};added=set(new['files'])-set(old['files'])
    assert changed|added==set(FILES)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    assert sha(BASE/'PRECOMPILED.json')==sha(TARGET/'PRECOMPILED.json')
    for name in ('EDITOR.json','CANDIDATE_LAUNCH.json','FIGURE_METRICS.json','QUALIFICATION_METRICS.json','SOURCES.json','EXPORT.json','EXAMPLE.json'):
        receipt=json.loads((OUT/name).read_text());assert receipt['runtime']==sha(TARGET/'RUNTIME.json'),name
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview43_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview44',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source='validation/_supplied220/candidate',runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),
        qualification=provenance['ui_qualification'],modification_scope=
        'Supplied suspension uses explicitly measured Ds, clear-water SettleFrm2 and exponential no-entrainment exchange. '
        'Suspension alone or with all nine ordinary bedload formulas; MF1, one bulk material. '
        'All native bindings and kernels unchanged. Project v32 rejects old formats. '
        'No new morphology, no QGIS or Delft3D changes.')
    save(ROOT/'desktop/PROVENANCE.json',provenance);REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('REGRESSION.xml','REGRESSION_COMPLETE.xml','COMPLETION.json','rebuilt_complete3.xml','recovered_supplied4.xml','rebuilt_vr5.xml',
              'EDITOR.json','CANDIDATE_LAUNCH.json','STAGE.json','PARITY.json','FIGURE_METRICS.json','QUALIFICATION_METRICS.json','SOURCES.json','EXPORT.json','EXAMPLE.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview44',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),added=sorted(added),
        existing_native_bindings_unchanged=True,packaged_checks=int(suite.attrib['tests']),
        qualification='Combined distinct cases; interrupted receipt retained; fresh replacement research fixtures',
        user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview44',sha(BASE/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
