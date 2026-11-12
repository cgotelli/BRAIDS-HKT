"""Deliver the bounded comparison projects and the qualified arithmetic repair."""
import argparse,hashlib,json,os,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation59'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation59'
OUT=ROOT/'desktop/qualification/water_budget59';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
PARENT='1dcf63c3b753ebbbe012313f7ee05932461aeb38d9526f22d41af6c478aa097a'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package():
    records=read(HOME/'PROJECTS.json');python=ROOT/'desktop/.venv/Scripts/python.exe';launch=ROOT/'desktop/launch.py'
    menu=['@echo off','setlocal','cd /d "%~dp0"','echo BRAIDS direct depth-mean wake validation','echo.']
    for i,(name,r) in enumerate(records.items(),1):menu.append(f'echo {i}. {r["configuration"]["folder"]}')
    menu+=['echo 5. Check progress','echo.','choice /c 12345 /n /m "Choose a project: "','if errorlevel 5 goto status']
    for i in range(4,0,-1):menu.append(f'if errorlevel {i} goto case{i}')
    for i,(name,r) in enumerate(records.items(),1):
        relative=Path(r['folder']).name+'\\Validation.braids.json'
        menu += [f':case{i}',f'start "" /b "{python.with_name("pythonw.exe")}" -B "{launch}" "%~dp0{relative}"','exit /b']
    menu += [':status',f'"{python}" -B "%~dp0check_status.py"','pause','exit /b']
    (HOME/'Open validation projects.cmd').write_text('\n'.join(menu)+'\n',encoding='utf-8')
    (HOME/'Check progress.cmd').write_text(f'@echo off\n"{python}" -B "%~dp0check_status.py"\npause\n',encoding='utf-8')
    (Path.home()/'Desktop/Open BRAIDS Direct Wake Validation.cmd').write_text(f'@echo off\ncall "{HOME / "Open validation projects.cmd"}"\n',encoding='utf-8')
    (HOME/'check_status.py').write_text('''import json
from pathlib import Path
root=Path(__file__).resolve().parent
for name,r in json.loads((root/'PROJECTS.json').read_text()).items():
    folder=root/Path(r['folder']).name/'Project_data'
    runs=sorted((folder/'runs').glob('*'))
    try:
        p=json.loads((runs[-1]/'PROGRESS.json').read_text())
        label='bounded CPU verification; accepted-state stopped' if name.endswith('cpu') and p['status']=='cancelled' else p['status']
        print(name+': '+label+'; t='+str(p.get('time_s'))+' s')
    except (IndexError,OSError,ValueError):print(name+': not started / publishing')
''',encoding='utf-8')
    (HOME/'README.md').write_text('''# Direct depth-mean turbine-wake projects

Use **Open validation projects.cmd** or the Desktop shortcut
**Open BRAIDS Direct Wake Validation.cmd**. Keep Validation.braids.json beside
its Project_data directory. All inputs, preparation, outputs, checkpoints and
history stay there; published curves and configuration limits are in references
and VALIDATION_SPEC.json. The local launcher uses the installed BRAIDS environment.

The two **GPU** projects run to 2000 s; comparison averages 1500–2000 s.
The **CPU** companions are deliberately stopped after saving the 100 s comparison
frame. Their later accepted states/checkpoints remain available to continue;
they are not presented as full-duration CPU validation. Select Results and choose
flow velocity. Profiles contains the wake centreline and transverse sections.
Turbines provides histories and CSV. No project restarts automatically.
To see the narrow wake clearly on the map, zoom near x=500 m, y=295 m and
set velocity colour limits to approximately 0.14–0.205 m/s in Map appearance.

The reference is Nash et al. (2015), https://doi.org/10.3390/en81212380,
Figures 6b/7. Figure6b already contains the authors' averages of full-water-column
experimental measurements; no new projection of a hub-plane profile is applied.
The steady-flow setting and constant inlet-equivalent mixing are explicitly
reconstructed, not the authors' four-cycle tidal simulation or exact PML model.
Their mixing coefficient is not reported. See the limitations before interpreting
the comparison as experimental validation.

The displayed Ct=0.90313 is an equivalent combined rotor/support input:
rotor Ct=0.8 plus Cd=0.9 times 9 m² support area divided by rotor area.
Cp=0 and mechanical power=0 are intentional for this passive-disc wake test.
It is not a blade-performance or electrical-power example. Use the preceding
Validation58 Mycek projects for the separate measured loading/power comparison.

The first fixed-budget-floor attempts and one rejected old preparation remain
in history. The corrected runtime retains the same CFL and hydraulic physics;
only the floating-point accounting bound changes. Older preparations/checkpoints
are rejected rather than reinterpreted. Prepare explicitly after a runtime change.

Read **comparisons/REPORT.md** and its figures for the actual outcome, including
failed screens. Check progress prints one snapshot without polling continuously.
''',encoding='utf-8')
    print('Packaged',HOME,flush=True)


def gates():
    stage=read(OUT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==stage['runtime_sha256']
    manifest=read(TARGET/'RUNTIME.json')
    for n,h in manifest['files'].items():assert sha(TARGET/n)==h,n
    for file,count in [('ROUND_OFF_CANDIDATE.xml',2),('FOCUSED_CHECKS.xml',66),('ANALYSIS_CHECKS.xml',4)]:
        suite=next(ET.parse(OUT/file).getroot().iter('testsuite'))
        assert int(suite.get('tests'))==count and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    assert read(OUT/'REGRESSION.json')['status']=='passed'
    m=read(REPORT/'METRICS.json')
    for name in ['nash_gpu','zero_gpu']:
        r=read(REPORT/name/'RECEIPT.json');assert r['status']=='complete' and r['errors']==[] and r['actual_run_button']
        assert r['maximum_water_residual_m3']<1e-6
    for r in m['backend_bounded_100s'].values():assert r['maximum_depth_m']<1e-4 and r['maximum_velocity_m_s']<1e-5
    prior=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58/FINAL_MANIFEST.json';old=read(prior)
    for n,h in old['files'].items():assert sha(ROOT/n)==h,n
    result=dict(status='passed',runtime_sha256=stage['runtime_sha256'],parent=PARENT,arithmetic_tests=2,focused_checks=66,
        analysis_checks=4,exact_hydro_bedload_AMR_restart_controls=4,physical_screens=m['screens'],
        installation_claim='Scale-aware arithmetic repair; physical comparison outcome recorded separately, not inferred from code checks.',
        prior_manifest_sha256=sha(prior),native_bindings_unchanged=True)
    save(REPORT/'GATES.json',result);return result


def install():
    g=gates();assert sha(BASE/'RUNTIME.json')==PARENT
    query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'Desktop[\\/]BRAIDS[\\/]desktop[\\/]launch\.py|Desktop[\\/]BRAIDS[\\/]desktop[\\/]runtime[\\/]')} | Select-Object -ExpandProperty ProcessId"
    active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
    if active:raise RuntimeError('Save and close the installed Desktop/windows before replacement: '+active)
    old=read(BASE/'RUNTIME.json');new=read(TARGET/'RUNTIME.json');changed={n for n in old['files'] if old['files'][n]!=new['files'][n]}
    assert changed=={'braids_core/sediment/shoreline_solver.py','braids_desktop/__init__.py'}
    backup=OUT/'preview57_backup';backup.mkdir(exist_ok=False)
    for name in changed|{'RUNTIME.json'}:
        p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for name in changed:shutil.copy2(TARGET/name,BASE/name)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json')
    d=read(ROOT/'desktop/PROVENANCE.json');report='validation/hydrokinetic_turbines/reports/desktop_validation59/REPORT.md'
    d.update(version='0.1.0-preview58',ui_qualification=report)
    d['solver'].update(source='desktop/qualification/water_budget59/candidate',runtime_manifest_sha256=g['runtime_sha256'],qualification=report,
        modification_scope='Scale-aware FP64 single-increment water-budget bound; original small-domain floor retained. Fluxes, explicit f-wave/CFL, physical laws, native kernels and v39 project format unchanged.')
    save(ROOT/'desktop/PROVENANCE.json',d)
    save(REPORT/'INSTALL.json',dict(status='passed',version='0.1.0-preview58',runtime_sha256=g['runtime_sha256'],changed=sorted(changed),prior_runtime=PARENT))
    smoke=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/launch.py'),'--smoke'],cwd=ROOT,env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),capture_output=True,text=True,timeout=60)
    save(REPORT/'SMOKE.json',dict(exit_code=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr))
    assert smoke.returncode==0,smoke.stdout+smoke.stderr
    print('Installed and smoke-tested preview58',flush=True)


def seal():
    assert not (REPORT/'FINAL_MANIFEST.json').exists();g=gates()
    assert sha(BASE/'RUNTIME.json')==g['runtime_sha256'] and read(REPORT/'SMOKE.json')['exit_code']==0
    assert (REPORT/'REPORT.md').is_file() and read(REPORT/'PROJECT_REOPEN_CHECK.json')['status']=='passed'
    assert read(REPORT/'INSTALLED_IDENTITIES.json')['status']=='passed'
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',REPORT/'DELIVERY_PROVENANCE.json')
    refs=ROOT/'validation/hydrokinetic_turbines';retained=REPORT/'references';retained.mkdir(exist_ok=False)
    names=['measurements/nash_2015_figure6b.csv','measurements/nash_2015_figure6b.json','measurements/nash_2015_figure7.csv','measurements/nash_2015_figure7.json']
    for name in names:shutil.copy2(refs/name,retained/Path(name).name)
    names+=['vendor/nash_2015_2d.pdf','vendor/nash_2015_page10.svg','vendor/nash_2015_page11.svg','digitize_nash_2015.py','digitize_nash_2015_lateral.py']
    save(REPORT/'REFERENCE_IDENTITIES.json',dict(primary_url='https://doi.org/10.3390/en81212380',
        files={str((refs/n).relative_to(ROOT)):sha(refs/n) for n in names},
        interpretation='Author-published full-column experimental average and separate model curves; raw experiment and original input deck unavailable.'))
    for name in ['nash_gpu','zero_gpu','nash_cpu','zero_cpu']:
        r=read(REPORT/name/('RECEIPT.json' if name.endswith('gpu') else 'STARTED.json'))
        source=Path(r['output']);target=REPORT/name/'accepted_metadata';target.mkdir(exist_ok=False)
        for file in ['PROJECT.json','RUN_CONTEXT.json','PROGRESS.json','STATUS.json','DIAGNOSTICS.json','TURBINE_PERFORMANCE.json','TURBINE_PERFORMANCE.csv','TURBINE_PERFORMANCE_INDEX.json']:
            shutil.copy2(source/file,target/file)
        shutil.copy2(source/'frames/INDEX.json',target/'FRAME_INDEX.json')
    files=[p for p in REPORT.rglob('*') if p.is_file() and 'ui_settings' not in p.parts]
    # Only small fixed-floor evidence is included; old/partial run artifacts
    # remain in the editable user folders, not duplicated into the seal.
    files.extend(p for p in OUT.rglob('*') if p.is_file() and ('candidate' in p.parts or p.suffix in {'.json','.xml','.log'}))
    files.extend(ROOT/'desktop'/n for n in ['turbine_validation59.py','run_turbine_validation59_ui.py','analyze_turbine_validation59.py','qualify_water_budget59.py','test_water_budget59.py','check_water_budget59.py','package_turbine_validation59.py','inspect_turbine_validation59.py','verify_water_budget59_installed.py'])
    files.append(ROOT/'tests/test_turbine_validation59.py')
    save(REPORT/'FINAL_MANIFEST.json',dict(status='passed',runtime_sha256=g['runtime_sha256'],
        mutable_user_projects_excluded=True,physical_screens=g['physical_screens'],files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    dest=HOME/'comparisons';dest.mkdir(exist_ok=False)
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix.lower() in {'.md','.json','.png','.csv'}:shutil.copy2(p,dest/p.name)
    print('Sealed comparison evidence and copied figures to',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['package','gates','install','seal']);globals()[p.parse_args().action]()
