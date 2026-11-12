"""Deliver the qualified single-grain bedload addition, retaining preview32."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_camenen192 import ROOT,OUT,TARGET,BASE_HASH,FILES,sha,verify,save
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_camenen192'
def suite(path):
    s=ET.parse(path).getroot().find('testsuite')
    assert int(s.attrib['tests'])==296 and all(int(s.attrib[k])==0 for k in ('errors','failures','skipped')),s.attrib
def main():
    suite(OUT/'RELEASE_REGRESSION.xml');new=verify(TARGET);identity=sha(TARGET/'RUNTIME.json')
    comparison=json.loads((REPORT/'COMPARISON.json').read_text());assert comparison['runtime']==identity
    for start in ('wet','dry'):
        assert comparison['cases'][start]['maximum_backend_bed_mm']<1.
        for backend in ('cpu','gpu'):
            c=json.loads((OUT/'release/runs'/start/('camenen_larson_2005_'+backend)/'CHECK.json').read_text())
            assert c['runtime']==identity and c['status']['status']=='complete'
            assert all(c['restart_exact'].values()) and c['compiler_free'] and c['inlet_capacity_oracle']
    for backend in ('cpu','gpu'):
        c=json.loads((OUT/'release_controls'/backend/'CHECK.json').read_text())
        assert c['runtime']==identity and len(c['checks'])==6
        assert all(all(v['previous_release_fields_exact'].values()) for v in c['checks'].values())
    target=ROOT/'desktop/runtime';assert sha(target/'RUNTIME.json')==BASE_HASH;old=verify(target)
    script="Get-CimInstance Win32_Process | Where-Object {($_.Name -match 'python|BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',script],text=True).strip()
    if running:raise RuntimeError('Close BRAIDS Desktop and its workers first: '+running)
    changed={n for n,h in old['files'].items() if new['files'].get(n)!=h}
    added=set(new['files'])-set(old['files'])
    assert not set(old['files'])-set(new['files'])
    assert changed<=set(FILES)|{'PRECOMPILED.json'},changed
    assert all(n.startswith('precompiled/') for n in added),added
    backup=OUT/'preview32_backup'
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(target/n,dst)
    for n in sorted(changed|added):
        dst=target/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',target/'RUNTIME.json');verify(target)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview33',runtime=identity,previous_runtime=BASE_HASH,
        payloads=len(new['files']),changed=sorted(changed),added=sorted(added),backup=str(backup),
        tests=296,coupled_new_law_cases=4,old_law_controls=12,hydraulics_changed=False,
        QGIS_changed=False,Delft3D_modified=False,user_projects_changed=False))
    print('Installed preview33',identity,len(new['files']),'payloads verified',flush=True)
if __name__=='__main__':main()
