"""Install the tested standalone release, preserving the prior runtime delta."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_collapse222 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify

def main():
    identity=sha(TARGET/'RUNTIME.json');metrics=json.loads((OUT/'METRICS.json').read_text())
    example=json.loads((OUT/'EXAMPLE.json').read_text())
    assert metrics['runtime']==example['runtime']==identity
    receipts={}
    for name in ('qualification','regression3'):
        p=OUT/(name+'.xml');suite=ET.parse(p).getroot().find('testsuite')
        assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
        assert p.stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
        receipts[name]=dict(tests=int(suite.attrib['tests']),sha256=sha(p))
    assert len(metrics['paired'])==71 and len(example['cases'])==4
    for n,h in metrics['Delft3D_sources_read_only'].items():assert sha(ROOT/n)==h
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};added=set(new['files'])-set(old['files'])
    assert changed|added==set(FILES)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    assert sha(BASE/'PRECOMPILED.json')==sha(TARGET/'PRECOMPILED.json')
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview45_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview46',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=identity,
        qualification=provenance['ui_qualification'],modification_scope=
        'Qualify ordinary finite-stock conservative collapse with all 71 ordinary transport configurations and positive MF, '
        'without multiplying collapse by MF. Separate Jiang-Cao repose remains exclusive to that bundle at MF1. '
        'Shared core/UI compatibility and explicit process summary. Native collapse and transport formulas unchanged. '
        'Project v34 rejects old formats. No QGIS or Delft3D changes.')
    save(ROOT/'desktop/PROVENANCE.json',provenance);REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('qualification.xml','qualification2.xml','qualification3.xml','QUALIFICATION_RECEIPT.json','regression3.xml','METRICS.json','EXAMPLE.json','EXAMPLE_COMPARISON.json','STAGE.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview46',runtime=identity,previous=BASE_HASH,
        backup=str(backup),changed=sorted(changed),added=sorted(added),native_bindings_unchanged=True,
        tests=receipts,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview46',identity,flush=True)
if __name__=='__main__':main()
