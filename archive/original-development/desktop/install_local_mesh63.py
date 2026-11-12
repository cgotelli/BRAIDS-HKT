"""Deliver only the qualified local-mesh candidate, with recoverable backup."""
import os,sys,subprocess,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from local_mesh63 import ROOT,BASE,TARGET,OUT,REPORT,read,save,sha

stage=read(REPORT/'STAGE.json');checks=read(REPORT/'COUPLED.json')
assert sha(TARGET/'RUNTIME.json')==stage['runtime'] and sha(BASE/'RUNTIME.json')==stage['parent']
suite=ET.parse(REPORT/'TESTS.xml').getroot()
assert all(int(e.get('failures','0'))==0 and int(e.get('errors','0'))==0 for e in suite.iter('testsuite'))
for mode in ['static_terrain','dynamic_depth']:
 assert checks[mode+'_backend_error']<1e-9
 for b in ['cpu','gpu']:assert checks[mode+'_'+b]['exact_restart'] and checks[mode+'_'+b]['time']==.25
old=read(BASE/'RUNTIME.json');new=read(TARGET/'RUNTIME.json')
for n,h in new['files'].items():assert sha(TARGET/n)==h,n
query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'Desktop[\\/]BRAIDS[\\/]desktop[\\/](launch.py|runtime)')} | Select-Object -ExpandProperty ProcessId"
active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
if active:raise RuntimeError('BRAIDS Desktop/worker must be closed before replacement: '+active)
changed={n for n,h in new['files'].items() if h!=old['files'].get(n)};assert changed==set(stage['changed'])
backup=OUT/'preview60_backup';backup.mkdir(exist_ok=False)
for n in changed|{'RUNTIME.json'}:
 if (BASE/n).exists():p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
for n in changed:
 p=BASE/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,p)
shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json')
shutil.copy2(TARGET/'braids_desktop/__init__.py',ROOT/'braids_desktop/__init__.py')
provenance=read(ROOT/'desktop/PROVENANCE.json');report='validation/hydrokinetic_turbines/reports/local_mesh63/REPORT.md'
provenance.update(version='0.1.0-preview61',ui_qualification=report)
provenance['solver'].update(source='desktop/qualification/local_mesh63/candidate',runtime_manifest_sha256=stage['runtime'],qualification=report,
 modification_scope='Sparse automatic turbine/wake subcells, per-cell area floors and AMR protection; prior audited directed-flow/power-envelope guards. Native libraries and wake/power formulas unchanged.')
save(ROOT/'desktop/PROVENANCE.json',provenance)
project=Path.home()/'Desktop/BRAIDS_Turbine_LocalMesh63/static_terrain_gpu_final/Project.braids.json'
smoke=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/launch.py'),str(project),'--smoke'],cwd=ROOT,env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),capture_output=True,text=True,timeout=180)
save(REPORT/'INSTALLED_SMOKE.json',dict(exit_code=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr,project=str(project)))
assert smoke.returncode==0 and 'Traceback' not in smoke.stderr and 'Traceback' not in smoke.stdout,smoke.stderr+smoke.stdout
save(REPORT/'INSTALL.json',dict(status='passed',version='0.1.0-preview61',runtime=stage['runtime'],parent=stage['parent'],backup=str(backup),changed=sorted(changed)))
print('Installed preview61; current coarse-river project opened successfully.')
