"""Install only the qualified secondary correction, with a retained backup."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_secondary185 import ROOT,BASE,TARGET,OUT,FILES,BASE_HASH,sha,save
EXPECTED='1b078961ad95f0038ba07da0a4aca9c3dd15a7f3cbffff2ec413986d27b4723b'

def verify(root):
    m=json.loads((root/'RUNTIME.json').read_text())
    for name,h in m['files'].items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve())
        assert sha(path)==h,name
    return m

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert sha(TARGET/'RUNTIME.json')==EXPECTED
    old,new=verify(BASE),verify(TARGET)
    suite=ET.parse(OUT/'TESTS.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])>=103 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    r=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_secondary185/CHECKS.json').read_text())
    assert r['runtime']==EXPECTED
    assert r['half_step']['fields']['depth']['max_abs_m']<.001
    assert r['backend']['fields']['depth']['max_abs_m']<1e-7
    assert r['coupled_backend']['fields']['depth']['max_abs_m']<.001
    assert r['coupled_backend']['fields']['z']['max_abs']<.001
    for b in ('cpu','gpu'):
        c=r['continuation'][b]
        assert c['runtime']==EXPECTED and c['status']=='segment_complete' and c['time_s']==220.
        assert all(c['restart_exact'].values()) and c['diagnostics']['secondary_cfl_max']<=.8*(1+1e-12)
        assert r['amr'][b]['status']=='complete' and r['amr'][b]['mesh_epochs']>1
        assert all(r['amr'][b]['exact_restart'].values())
        assert all(v['max_abs']==0 for k,v in r['off_regression'][b]['fields'].items() if k!='depth')
    guard=json.loads((OUT/'OLD_CHECKPOINT.json').read_text())
    assert guard['runtime']==EXPECTED and guard['rejected'] and not guard['output_created']
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and workers before installation: '+running)
    changed={k for k in old['files'] if old['files'][k]!=new['files'][k]}
    assert changed==({k for k in FILES if k in old['files']}|{'PRECOMPILED.json'}),changed
    assert all(sha(ROOT/f)==sha(TARGET/f) for f in FILES)
    backup=OUT/'preview29_files'
    for f in sorted(changed|{'RUNTIME.json'}):
        dst=backup/f;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(BASE/f)
        else:shutil.copy2(BASE/f,dst)
    added=set(new['files'])-set(old['files'])
    for f in sorted(added|changed):
        dst=BASE/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/f,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    save(OUT/'INSTALL.json',dict(previous=BASE_HASH,installed=EXPECTED,files_verified=len(new['files']),
        changed=sorted(changed),added=sorted(added),backup=str(backup),
        hydraulic_binaries_unchanged=True,Delft3D_unchanged=True))
    print('Installed',EXPECTED,'verified files',len(new['files']))
if __name__=='__main__':main()
