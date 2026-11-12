"""Install the qualified standalone Exner repose extension; retain prior delta."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_repose223 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify

def main():
    identity=sha(TARGET/'RUNTIME.json');metrics=json.loads((OUT/'METRICS.json').read_text())
    example=json.loads((OUT/'EXAMPLE.json').read_text())
    assert metrics['runtime']==example['runtime']==identity
    receipts={}
    for name in ('qualification_final','regression_final'):
        p=OUT/(name+'.xml');suite=ET.parse(p).getroot().find('testsuite')
        assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
        assert p.stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
        receipts[name]=dict(tests=int(suite.attrib['tests']),sha256=sha(p))
    assert len(metrics['paired'])==71 and len(example['cases'])==5
    for n,h in metrics['Delft3D_sources_read_only'].items():assert sha(ROOT/n)==h
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};added=set(new['files'])-set(old['files'])
    assert changed|added==set(FILES)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    assert sha(BASE/'PRECOMPILED.json')==sha(TARGET/'PRECOMPILED.json')
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview46_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview47',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=identity,
        qualification=provenance['ui_qualification'],modification_scope=
        'Extend Jiang-Cao geometric repose criterion to all 71 ordinary transport configurations with a separately identified finite Exner inventory coupling. '
        'Explicit angle, finite-stock constraints, no extra water/suspension source or MF multiplier. Original coupled Jiang-Cao path remains MF1. '
        'Public project v35 rejects older formats. Native kernels, hydraulic and transport laws, QGIS and Delft3D unchanged.')
    save(ROOT/'desktop/PROVENANCE.json',provenance);REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('qualification_final.xml','regression_final.xml','METRICS.json','EXAMPLE.json','EXAMPLE_COMPARISON.json','STAGE.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview47',runtime=identity,previous=BASE_HASH,
        backup=str(backup),changed=sorted(changed),added=sorted(added),native_bindings_unchanged=True,
        tests=receipts,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview47',identity,flush=True)
if __name__=='__main__':main()
