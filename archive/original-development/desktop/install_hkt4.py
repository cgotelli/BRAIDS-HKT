"""Publish qualified turbine Desktop delta, keeping independent runtimes intact."""
from stage_hkt4 import ROOT,OUT,TARGET,BASE,FILES,sha,save,verify
import json,shutil,subprocess,xml.etree.ElementTree as ET

REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_hkt4'


def main():
    old=verify(BASE);new=verify(TARGET);previous=sha(BASE/'RUNTIME.json');current=sha(TARGET/'RUNTIME.json')
    assert previous=='a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1'
    assert current=='a76404b9c24fe03927c1530484758cfb9d058603a36fc74eb5ab5ade14b08c86'
    for backend in ['cpu','gpu']:
        check=json.loads((OUT/('public_release_'+backend)/'CHECK.json').read_text());assert check['runtime']==current and check['restart_exact'] and check['resume_cancel_exact'] and check['actual_remesh']
    for name,count in [('REGRESSION.xml',75),('UI_PACKAGED.xml',3)]:
        suite=next(ET.parse(REPORT/name).getroot().iter('testsuite'));assert int(suite.get('tests'))==count and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    metrics=json.loads((REPORT/'METRICS.json').read_text());assert metrics['ordinary_reference']['q_byte_identical']
    assert max(metrics[k]['maximum_backend_depth_m'] for k in ['full','remesh'])<1e-4
    staged=json.loads((OUT/'STAGE.json').read_text())
    for n,h in staged['source_files'].items():assert sha(ROOT/n)==new['files'][n]==h,n
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|braids_desktop.app|desktop[\\\\/]runtime[\\\\/]braids_core[\\\\/]river_launcher.py')} | Select-Object -ExpandProperty ProcessId"
    running=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if running:raise RuntimeError('Save and close the installed Desktop before publication. Independent frozen project runtimes will not be stopped: '+running)
    changed={n for n,h in old['files'].items() if new['files'][n]!=h};added=set(new['files'])-set(old['files']);assert not set(old['files'])-set(new['files'])
    assert changed<=set(FILES)|{'PRECOMPILED.json'}
    backup=OUT/'preview51_backup';assert not backup.exists()
    for n in sorted(changed|{'RUNTIME.json'}):
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'DESKTOP_PROVENANCE.json')
    for n in sorted(changed|added):
        p=BASE/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,p)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    provenance=json.loads((ROOT/'desktop/PROVENANCE.json').read_text());report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    provenance.update(version='0.1.0-preview52',ui_qualification=report)
    provenance['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=current,qualification=report,
        modification_scope='Optional one-rotor zero-yaw hydraulic turbine: rotated polygon bump, common-bed balanced pressure, actual-depth loading and automatic coverage shutdown; explicit CPU/GPU. Placement is perpendicular to a two-click local river direction. Ordinary hydraulics unchanged. Project v37 rejects older formats; independent sediment runtimes remain untouched.')
    save(ROOT/'desktop/PROVENANCE.json',provenance)
    save(REPORT/'INSTALL.json',dict(version='0.1.0-preview52',runtime=current,previous_runtime=previous,changed=sorted(changed),added=sorted(added),backup=str(backup),
        default_shortcut='Launch BRAIDS Desktop.cmd',example=str(ROOT/'desktop/examples/turbine_hkt4/Channel.braids.json'),independent_project_runtimes_modified=False,QGIS_changed=False,Delft3D_modified=False))
    print('Installed preview52',current)


if __name__=='__main__':main()
