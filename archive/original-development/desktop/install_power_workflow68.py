"""Install the qualified Desktop update without modifying frozen hardlinks."""
import os,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
from power_workflow68 import ROOT,TARGET,REPORT,read,save,sha
base=ROOT/'desktop/runtime';old=read(base/'RUNTIME.json');new=read(TARGET/'RUNTIME.json')
assert sha(base/'RUNTIME.json')==read(REPORT/'STAGE_candidate_v3.json')['installed_parent']
assert sha(TARGET/'RUNTIME.json')==read(REPORT/'STAGE_candidate_v3.json')['candidate']
assert read(REPORT/'UI_SMOKE.json')['passed']
for s in ET.parse(REPORT/'TESTS_V3.xml').getroot().iter('testsuite'):assert int(s.get('failures','0'))==int(s.get('errors','0'))==0
for backend in ['cpu','gpu']:
    home=ROOT/'desktop/qualification/power_workflow68'
    assert read(home/('restart_'+backend)/'DONE.json')['exact_restart']
    assert read(home/('variable_'+backend)/'DONE.json')['result']['status']=='complete'
for n,h in new['files'].items():assert sha(TARGET/n)==h,n
query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'BRAIDS[\\/]desktop[\\/](launch.py|runtime)')} | Select-Object -ExpandProperty ProcessId"
active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
if active:raise RuntimeError('Desktop/worker is open; close it before replacement: '+active)
changed={n for n,h in new['files'].items() if h!=old['files'].get(n)}
backup=ROOT/'desktop/qualification/power_workflow68/preview61_backup';backup.mkdir(exist_ok=False)
for n in changed|{'RUNTIME.json'}:
    if (base/n).exists():p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
def replace(source,dest):
    dest=dest.resolve();assert dest.is_relative_to(base.resolve())
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.power68.tmp')
    assert not tmp.exists();shutil.copy2(source,tmp);os.replace(tmp,dest)
try:
    for n in changed:replace(TARGET/n,base/n)
    replace(TARGET/'RUNTIME.json',base/'RUNTIME.json')
    smoke=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/smoke_power_workflow68.py'),str(base),str(Path.home()/'Desktop/BRAIDS_Power68/Power.braids.json'),str(REPORT/'INSTALLED_SMOKE.json')],capture_output=True,text=True,timeout=90)
    assert smoke.returncode==0,smoke.stdout+smoke.stderr
except BaseException:
    for n in changed|{'RUNTIME.json'}:
        if (backup/n).exists():replace(backup/n,base/n)
        elif (base/n).exists():
            p=(base/n).resolve();assert p.is_relative_to(base.resolve());p.unlink()
    raise
provenance=read(ROOT/'desktop/PROVENANCE.json');provenance.update(version='0.1.0-preview62',ui_qualification='validation/hydrokinetic_turbines/reports/power_workflow68/REPORT.md')
provenance['solver'].update(source=str(TARGET.relative_to(ROOT)),runtime_manifest_sha256=sha(base/'RUNTIME.json'),qualification=provenance['ui_qualification'],modification_scope='Explicit device speed/TSR rotor-inflow power, fixed-RPM local-speed shutdown and declared electrical efficiency; fixed mesh and single upstream wake; no overlap qualification. Native binaries unchanged.')
save(ROOT/'desktop/PROVENANCE.json',provenance)
# Earlier retained runtimes must still match every recorded file.
for folder in [ROOT/'desktop/qualification/energy67/candidate_v3',ROOT/'desktop/qualification/local_mesh63/candidate',TARGET]:
    for n,h in read(folder/'RUNTIME.json')['files'].items():assert sha(folder/n)==h,(folder,n)
save(REPORT/'INSTALL.json',dict(passed=True,version='0.1.0-preview62',runtime=sha(base/'RUNTIME.json'),backup=str(backup),changed=sorted(changed),frozen_runtimes_intact=True))
print('Installed preview62; project/result smoke passed; frozen runtimes intact.')
