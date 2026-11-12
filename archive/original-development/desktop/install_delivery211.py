"""Qualified preview36 promotion with recoverable preview35 backup."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_delivery211 import ROOT,OUT,TARGET,sha,save,verify
OLD='f88145dcf039c589557e9bb1acf7c6dacaf8b1afedec7952e3a533aa9f6ccb6b'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_delivery211'
CHANGED={'PRECOMPILED.json','braids_desktop/__init__.py','braids_core/river_jiang_cao.py',
 'braids_core/sediment/jiang_cao_adjustment.py','braids_core/sediment/jiang_cao_desktop.py',
 'braids_core/sediment/jiang_cao_exchange.h','braids_core/sediment/jiang_cao_exchange.py','braids_core/sediment/jiang_cao_wetdry.py'}

def main():
    suite=ET.parse(OUT/'PACKAGED_REGRESSION.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==519 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped')),suite.attrib
    m=verify(TARGET);identity=sha(TARGET/'RUNTIME.json')
    for scenario in ('front','oblique'):
        for backend in ('cpu','gpu'):
            p=OUT/(scenario+'_'+backend)
            c=json.loads((p/'CHECK.json').read_text());r=json.loads((p/'RESTART.json').read_text())
            assert c['runtime']==identity and c['compiler_free'] and r['compiler_free']
            assert c['result']['status']=='complete' and not c['result']['rejections'] and all(r['exact'].values())
    for backend in ('cpu','gpu'):
        c=json.loads((OUT/'controls'/backend/'CHECK.json').read_text())
        assert c['runtime']==identity and len(c['checks'])==8
        assert all(all(v['previous_release_fields_exact'].values()) and v['old_restart_rejected'] for v in c['checks'].values())
    oldroot=ROOT/'desktop/runtime';old=verify(oldroot);assert sha(oldroot/'RUNTIME.json')==OLD
    changed={n for n,h in old['files'].items() if m['files'].get(n)!=h};added=set(m['files'])-set(old['files'])
    assert changed==CHANGED and not set(old['files'])-set(m['files'])
    assert all(n.startswith('precompiled/321961e4d6ffbde21d15/') for n in added)
    for n in changed-{'PRECOMPILED.json'}:assert sha(ROOT/n)==m['files'][n],n
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Close Desktop and its workers before installation: '+running)
    backup=OUT/'preview35_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(oldroot/n,dst)
    for n in sorted(changed|added):
        dst=oldroot/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',oldroot/'RUNTIME.json');verify(oldroot)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview36',runtime=identity,previous=OLD,backup=str(backup),
      changed=sorted(changed),added=sorted(added),payloads=len(m['files']),packaged_tests=519,
      old_formula_controls=16,lifecycle_cases=4,QGIS_changed=False,Delft3D_modified=False,user_projects_modified=False))
    print('Installed preview36',identity,flush=True)

if __name__=='__main__':main()
