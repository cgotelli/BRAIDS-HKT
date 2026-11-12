"""Release only the fully qualified payload; retain recoverable preview34 backup."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_bundle207 import ROOT,OUT,TARGET,BASE_HASH,FILES,sha,save,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bundle207'
def suite(path,count):
    s=ET.parse(path).getroot().find('testsuite')
    assert int(s.attrib['tests'])==count and all(int(s.attrib[k])==0 for k in ('errors','failures','skipped')),s.attrib
def main():
    suite(OUT/'REGRESSION_FINAL.xml',894);suite(OUT/'PACKAGED_REGRESSION_v3.xml',353)
    m=verify(TARGET);identity=sha(TARGET/'RUNTIME.json')
    c=json.loads((REPORT/'COMPARISON.json').read_text());assert c['runtime']==identity
    for scenario in ('front','bank','open','oblique_wet'):
        for backend in ('cpu','gpu'):
            path=OUT/'release_v3'/(scenario+'_'+backend)
            check=json.loads((path/'CHECK.json').read_text());restart=json.loads((path/'RESTART.json').read_text())
            assert check['runtime']==restart['runtime']==identity and check['compiler_free'] and restart['compiler_free']
            assert check['result']['status']=='complete' and all(restart['exact'].values())
    for backend in ('cpu','gpu'):
        b=json.loads((OUT/'boundaries_v3'/backend/'CHECK.json').read_text())
        assert b['runtime']==identity and len(b['rows'])==12 and all(r['status']['status']=='complete' for r in b['rows'])
        b=json.loads((OUT/'release_controls_v3'/backend/'CHECK.json').read_text())
        assert b['runtime']==identity and len(b['checks'])==8
        assert all(all(r['previous_release_fields_exact'].values()) and r['old_restart_rejected'] for r in b['checks'].values())
    # Source identity must still match the exact code that was packaged/tested.
    for name in FILES:assert sha(ROOT/name)==m['files'][name],name
    oldroot=ROOT/'desktop/runtime';old=verify(oldroot);assert sha(oldroot/'RUNTIME.json')==BASE_HASH
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Close Desktop and its workers before installation: '+running)
    changed={n for n,h in old['files'].items() if m['files'].get(n)!=h};added=set(m['files'])-set(old['files'])
    assert not set(old['files'])-set(m['files'])
    assert changed<=set(FILES)|{'PRECOMPILED.json'}
    assert all(n in FILES or n.startswith('precompiled/') for n in added)
    backup=OUT/'preview34_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(oldroot/n,dst)
    for n in sorted(changed|added):
        dst=oldroot/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',oldroot/'RUNTIME.json');verify(oldroot)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview35',runtime=identity,previous=BASE_HASH,
        backup=str(backup),changed=sorted(changed),added=sorted(added),payloads=len(m['files']),
        tests=894,packaged_tests=353,old_formula_controls=16,boundary_cases=24,
        QGIS_changed=False,Delft3D_modified=False,user_projects_modified=False))
    print('Installed preview35',identity,flush=True)
if __name__=='__main__':main()
