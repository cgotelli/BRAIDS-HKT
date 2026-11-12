"""Promote qualified S3, retaining the installed release and all user projects."""
import json,shutil,subprocess,xml.etree.ElementTree as ET
from stage_suspended216 import ROOT,OUT,TARGET,BASE,BASE_HASH,FILES,REPORT,sha,save,verify


def main():
    suite=ET.parse(OUT/'REGRESSION.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==773 and all(int(suite.attrib[k])==0 for k in ('failures','errors','skipped')),suite.attrib
    assert (OUT/'REGRESSION.xml').stat().st_mtime>(TARGET/'RUNTIME.json').stat().st_mtime
    new=verify(TARGET);old=verify(BASE);assert sha(BASE/'RUNTIME.json')==BASE_HASH
    assert set(old['files'])<=set(new['files'])
    changed={n for n,h in old['files'].items() if new['files'][n]!=h}
    added=set(new['files'])-set(old['files'])
    assert changed==set(FILES)|{'PRECOMPILED.json'},changed
    assert all(n.startswith('precompiled/') for n in added)
    for n in FILES:assert sha(ROOT/n)==new['files'][n],n
    before=json.loads((BASE/'PRECOMPILED.json').read_text());after=json.loads((TARGET/'PRECOMPILED.json').read_text())
    for key,value in before.items():
        if not key.startswith('validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'):
            assert after[key]==value,key
    editor=json.loads((OUT/'EDITOR.json').read_text())
    assert editor['runtime']==sha(TARGET/'RUNTIME.json') and editor['roundtrip'] and editor['concentration_inputs_visible']
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close Desktop before installation: '+running)
    backup=OUT/'preview39_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        dst=BASE/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,dst)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text())
    provenance.update(version='0.1.0-preview40',ui_qualification=str(REPORT.relative_to(ROOT)/'REPORT.md').replace('\\','/'))
    provenance['solver'].update(source='validation/_suspended216/candidate',runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),
        qualification=provenance['ui_qualification'],modification_scope=
        'S3 Van Rijn 1993 FLOW current-only suspension, matching bedload producer and independent measured D50/D90/Ds; '
        'MF1, one bulk material, no newly admitted bank/collapse combinations. Includes qualified S2 nonorthogonal '
        'scalar diffusion and finite-stock exchange corrections. 773 compiler-free packaged tests pass. '
        'Explicit augmented f-wave hydraulics and its native binaries are unchanged. Project v28 strictly rejects old '
        'projects and previous source identities reject old checkpoints. No QGIS or Delft3D changes.')
    save(ROOT/'desktop/PROVENANCE.json',provenance)
    REPORT.mkdir(parents=True,exist_ok=True)
    for n in ('REGRESSION.xml','EDITOR.json','STAGE.json','BUILD.json','SOURCE.xml','BEDLOAD_PARITY.json'):
        shutil.copy2(OUT/n,REPORT/n)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview40',runtime=sha(BASE/'RUNTIME.json'),
        previous=BASE_HASH,backup=str(backup),changed=sorted(changed),added=sorted(added),
        hydraulics_unchanged=True,suspended_execution_admitted='Van Rijn 1993 FLOW 2DH; IopSus=0; MF1',
        packaged_checks=773,user_projects_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview40',sha(BASE/'RUNTIME.json'),flush=True)


if __name__=='__main__':main()
