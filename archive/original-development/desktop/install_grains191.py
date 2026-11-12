"""Install only qualified manifest payloads; preserve the previous release."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_grains191 import ROOT,BASE,OUT,TARGET,FILES,sha,verify
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_grains191'

def suite(path,count):
    s=ET.parse(path).getroot().find('testsuite')
    assert int(s.attrib['tests'])==count and all(int(s.attrib[k])==0 for k in ('errors','failures','skipped')),s.attrib

def main():
    suite(OUT/'CONTRACT.xml',82)
    suite(ROOT/'validation/_formula190/GRAINS191_REGRESSION_v2.xml',180)
    manifest=verify(TARGET);checks={}
    for backend in ('cpu','gpu'):
        result=json.loads((TARGET/backend/'CHECK.json').read_text())
        assert result['runtime']==sha(TARGET/'RUNTIME.json') and len(result['checks'])==6
        for law,check in result['checks'].items():
            assert all(check['restart_exact'].values()) and all(check['previous_release_fields_exact'].values())
        checks[backend]=result
    old=verify(BASE)
    changed={n for n,h in old['files'].items() if manifest['files'].get(n)!=h}
    added=set(manifest['files'])-set(old['files'])
    assert not set(old['files'])-set(manifest['files'])
    assert changed|added==set(FILES),(changed,added)
    assert all(not n.endswith(('.h','.dll','.f90','.cu')) for n in changed|added)
    target=ROOT/'desktop/runtime'
    assert sha(target/'RUNTIME.json')==sha(BASE/'RUNTIME.json')
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and its workers first: '+running)
    verify(target)
    backup=OUT/'preview31_backup'
    for n in sorted(changed|{'RUNTIME.json'}):
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();shutil.copy2(target/n,p)
    for n in sorted(changed|added):
        shutil.copy2(TARGET/n,target/n)
    shutil.copy2(TARGET/'RUNTIME.json',target/'RUNTIME.json');verify(target)
    receipt=dict(version='0.1.0-preview32',runtime=sha(target/'RUNTIME.json'),
        previous_runtime=sha(BASE/'RUNTIME.json'),payloads=len(manifest['files']),changed=sorted(changed),added=sorted(added),
        backup=str(backup),contract_tests=82,regression_tests=180,checks=checks,
        native_payload_unchanged=True,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'INSTALL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ('GRAIN_MPM.png','GRAIN_SMART_JAGGI.png','GRAIN_RECKING_2010.png'):
        shutil.copy2(OUT/name,REPORT/name)
    print('Installed preview32:',receipt['runtime'],receipt['payloads'],'payloads verified',flush=True)
if __name__=='__main__':main()
