"""Promote the verified Smart-Jaeggi runtime; preserve the previous source/manifest."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_smart183 import ROOT,BASE,TARGET,OUT,FILES,BASE_HASH,sha,save
EXPECTED='fddbf4c03a89f4d19aa72e89dc3fb02967470289d0621c1c47d7ab9a73f71ae3'

def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for name,h in manifest['files'].items():assert sha(root/name)==h,name
    return manifest

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert sha(TARGET/'RUNTIME.json')==EXPECTED
    old,new=verify(BASE),verify(TARGET)
    suite=ET.parse(OUT/'TESTS_FINAL.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])>=212 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    for backend in ('cpu','gpu'):
        c=json.loads((OUT/'release'/('smart_jaggi_'+backend)/'CHECK.json').read_text())
        assert c['runtime']==EXPECTED and c['status']['status']=='complete' and all(c['restart']['exact'].values())
        assert c['inlet_capacity_oracle']
    compare=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_smart183/COMPARISON.json').read_text())
    assert all(max(v.values())==0 for v in compare['existing_formula_maximum_difference'].values())
    assert all(max(v.values())==0 for v in compare['final_validation_guard_trajectory_difference'].values())
    # Refuse promotion while the application or a worker could hold old modules.
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and its workers before installation: '+running)
    changed={k for k in old['files'] if old['files'][k]!=new['files'][k]}
    assert changed==({k for k in FILES if k in old['files']}|{'PRECOMPILED.json'}),changed
    assert all(sha(ROOT/f)==sha(TARGET/f) for f in FILES)
    backup=OUT/'preview28_files'
    for f in sorted(changed|{'RUNTIME.json'}):
        dst=backup/f;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(BASE/f)
        else:shutil.copy2(BASE/f,dst)
    # Add immutable native packages before replacing sources and the final manifest.
    added=set(new['files'])-set(old['files'])
    for f in sorted(added|changed):
        dst=BASE/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/f,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    save(OUT/'INSTALL.json',dict(previous=BASE_HASH,installed=sha(BASE/'RUNTIME.json'),files_verified=len(new['files']),
        changed=sorted(changed),added=sorted(added),backup=str(backup),
        hydraulic_binaries_unchanged=True,Delft3D_unchanged=True))
    print('Installed',sha(BASE/'RUNTIME.json'),'verified files',len(new['files']))
if __name__=='__main__':main()
