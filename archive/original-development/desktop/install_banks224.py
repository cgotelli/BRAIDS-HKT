"""Install qualified bank/toe composition; retain the previous release delta."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_banks224 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify

def main():
    identity=sha(TARGET/'RUNTIME.json');metrics=json.loads((OUT/'METRICS.json').read_text())
    example=json.loads((OUT/'EXAMPLE.json').read_text())
    fix=json.loads((OUT/'BINDING_FIX.json').read_text())
    assert metrics['runtime']==fix['runtime']==identity
    assert example['runtime']==fix['parent_runtime']
    receipts={}
    for name in ('qualification_release','regression_release_v2'):
        p=OUT/(name+'.xml');suite=ET.parse(p).getroot().find('testsuite')
        assert all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
        if name=='qualification_release':assert p.stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
        receipts[name]=dict(tests=int(suite.attrib['tests']),sha256=sha(p),
            execution_provenance='QUALIFICATION_PARTS.json' if name=='qualification_release' else fix['parent_runtime'])
    assert len(metrics['paired'])==71 and len(example['cases'])==8
    source_audit=json.loads((OUT/'SOURCE_AUDIT.json').read_text())
    assert sha(ROOT/source_audit['source'])==source_audit['sha256']
    for n,h in metrics['Delft3D_sources_read_only'].items():assert sha(ROOT/n)==h
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};added=set(new['files'])-set(old['files'])
    assert changed|added==set(FILES)|{'PRECOMPILED.json'}|set(fix['added'])
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    old_table=json.loads((BASE/'PRECOMPILED.json').read_text())
    new_table=json.loads((TARGET/'PRECOMPILED.json').read_text())
    assert all(new_table[k]==v for k,v in old_table.items())
    assert set(new_table)-set(old_table)=={fix['new_binding']}
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview47_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview48',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=identity,
        qualification=provenance['ui_qualification'],modification_scope=
        'Qualify accepted-demand dry-bank sharing, surveyed toe supply and retained block release with ordinary bedload/suspension and both collapse choices. '
        'One finite bulk material; MF already in accepted demand, no double multiplier. Export retained block histories/counts. '
        'Public project v36 rejects older formats. Added omitted CPU bank_counts remap binding from unchanged source. '
        'Existing native kernels, hydraulic and transport laws, QGIS and Delft3D unchanged.')
    save(ROOT/'desktop/PROVENANCE.json',provenance);REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('qualification_release.xml','regression_release_v2.xml','METRICS.json','EXAMPLE.json','EXAMPLE_COMPARISON.json','STAGE.json','SOURCE_AUDIT.json','SOURCE_AUDIT.md','QUALIFICATION_PARTS.json','NEGATIVE_SCREENS.json','BINDING_FIX.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview48',runtime=identity,previous=BASE_HASH,
        backup=str(backup),changed=sorted(changed),added=sorted(added),existing_native_bindings_unchanged=True,new_binding=fix['new_binding'],
        tests=receipts,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview48',identity,flush=True)
if __name__=='__main__':main()
