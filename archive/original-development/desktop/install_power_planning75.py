"""Atomic installation with rollback; preserved runtimes are never patched."""
import os,shutil,subprocess,sys,xml.etree.ElementTree as ET
from power_planning75 import ROOT,BASE,TARGET,REPORT,read,save,sha
assert sha(BASE/'RUNTIME.json')==read(REPORT/'STAGE.json')['installed_parent']
assert read(REPORT/'OPERATORS.json')['passed'] and read(REPORT/'METRICS.json')['passed']
assert read(REPORT/'DELIVERY_SMOKE.json')['passed']
for suite in ET.parse(REPORT/'REGRESSION.xml').getroot().iter('testsuite'):assert int(suite.get('failures','0'))==int(suite.get('errors','0'))==0
old=read(BASE/'RUNTIME.json');new=read(TARGET/'RUNTIME.json')
for n,h in new['files'].items():assert sha(TARGET/n)==h,n
query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'BRAIDS[\\/]desktop[\\/](launch.py|runtime)')} | Select-Object -ExpandProperty ProcessId"
active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
if active:raise RuntimeError('Save and close Desktop before installation: '+active)
changed={n for n,h in new['files'].items() if h!=old['files'].get(n)}
backup=ROOT/'desktop/qualification/power_planning75/preview62_backup';backup.mkdir(exist_ok=False)
for n in changed|{'RUNTIME.json'}:
    if (BASE/n).exists():p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
def replace(source,dest):
    dest=dest.resolve();assert dest.is_relative_to(BASE.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.power75.tmp');assert not tmp.exists();shutil.copy2(source,tmp);os.replace(tmp,dest)
try:
    for n in changed:replace(TARGET/n,BASE/n)
    replace(TARGET/'RUNTIME.json',BASE/'RUNTIME.json')
    run=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/smoke_power_planning75.py'),str(BASE)],capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr
except BaseException:
    for n in changed|{'RUNTIME.json'}:
        if (backup/n).exists():replace(backup/n,BASE/n)
        elif (BASE/n).exists():p=(BASE/n).resolve();assert p.is_relative_to(BASE.resolve());p.unlink()
    raise
v=read(ROOT/'desktop/PROVENANCE.json');v.update(version='0.1.0-preview63',ui_qualification='validation/hydrokinetic_turbines/reports/power_planning75/REPORT.md')
v['solver'].update(source=str(TARGET.relative_to(ROOT)),runtime_manifest_sha256=sha(BASE/'RUNTIME.json'),qualification=v['ui_qualification'],modification_scope='Explicit turbine-free power planning and finite-section Gaussian linear overlap; separate analytical power outputs. Existing coupled native solver unchanged.')
save(ROOT/'desktop/PROVENANCE.json',v)
for frozen in [ROOT/'desktop/qualification/power_workflow68/candidate_v3',ROOT/'desktop/qualification/projection71/candidate']:
    for n,h in read(frozen/'RUNTIME.json')['files'].items():assert sha(frozen/n)==h,(frozen,n)
save(REPORT/'INSTALL.json',dict(passed=True,version=v['version'],runtime_sha256=sha(BASE/'RUNTIME.json'),backup=str(backup),changed=sorted(changed),frozen_runtimes_intact=True))
print('Installed preview63; all three project UI/export checks passed.')
