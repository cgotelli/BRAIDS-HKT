"""Reviewable local shortcuts and self-contained reference material for the suite."""
import hashlib,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HOME=Path.home()/'Desktop/BRAIDS_Turbine_Validation58'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation58'


STATUS='''import json
from pathlib import Path
root=Path(__file__).resolve().parent
for name,row in json.loads((root/'PROJECTS.json').read_text()).items():
    folder=root/Path(row['folder']).name/'Project_data'
    runs=sorted((folder/'runs').glob('*'))
    if not runs:
        print(name+': not started');continue
    try:
        data=json.loads((runs[-1]/'PROGRESS.json').read_text())
        print(name+': '+str(data.get('status'))+'; t='+str(data.get('time_s'))+' s; '+str(data.get('phase','')))
    except (OSError,ValueError):print(name+': starting or publishing')
'''


def package():
    records=json.loads((HOME/'PROJECTS.json').read_text());python=ROOT/'desktop/.venv/Scripts/python.exe';pythonw=python.with_name('pythonw.exe');launch=ROOT/'desktop/launch.py'
    menu=['@echo off','setlocal','cd /d "%~dp0"','echo BRAIDS turbine validation projects','echo.']
    for i,(key,r) in enumerate(records.items(),1):menu.append(f'echo {i}. {r["configuration"]["folder"]}')
    menu+=['echo 6. Check progress','echo.','choice /c 123456 /n /m "Choose a project: "', 'if errorlevel 6 goto status']
    for i in range(5,0,-1):menu.append(f'if errorlevel {i} goto case{i}')
    for i,(key,r) in enumerate(records.items(),1):
        relative=Path(r['folder']).name+'\\Validation.braids.json'
        menu += [f':case{i}',f'start "" /b "{pythonw}" -B "{launch}" "%~dp0{relative}"','exit /b']
    menu += [':status',f'"{python}" -B "%~dp0check_status.py"','pause','exit /b']
    (HOME/'Open validation projects.cmd').write_text('\n'.join(menu)+'\n',encoding='utf-8')
    (HOME/'Check progress.cmd').write_text(f'@echo off\n"{python}" -B "%~dp0check_status.py"\npause\n',encoding='utf-8')
    (HOME/'check_status.py').write_text(STATUS,encoding='utf-8')
    (Path.home()/'Desktop/Open BRAIDS Turbine Validation.cmd').write_text(f'@echo off\ncall "{HOME / "Open validation projects.cmd"}"\n',encoding='utf-8')
    (HOME/'README.md').write_text('''# BRAIDS turbine validation projects

Use **Open validation projects.cmd** (or the Desktop shortcut) to choose a case.
Each folder contains a normal **Validation.braids.json** project, its **Project_data**
directory and a **references** directory with the published comparison curves.
Keep the project and its data directory together when moving it.

| Project | Purpose | Simulated duration |
|---|---|---:|
| 01 OpenTidalFarm 2D / GPU | Published shallow-water numerical wake, Abolghasemi et al. | 250 s |
| 02 Mycek 0.8 / CPU | Flume wake and conditional performance/load comparison | 40 s |
| 03 Mycek 0.8 / GPU | Same physical project, explicit GPU verification | 40 s |
| 04 Mycek 1.0 / CPU | Different velocity; measured 1.0 m/s Ct/Cp curve held out | 40 s |
| 05 Mycek zero thrust / CPU | Ct=Cp=0, same viscosity/mesh; no-extraction control | 40 s |

Open a project and switch to **Results** to see its latest completed run.
Select flow velocity on the map. **Profiles** contains named transverse sections
and a wake centreline; use spatial-profile mode. **Turbines** plots per-turbine
histories and exports CSV. **Boundary flow** checks the imposed discharge.
Live results remain available when running. Checking progress prints a snapshot;
it does not poll continuously or modify the calculation.

**Run** starts fresh unless a checkpoint is explicitly selected. Results, prepared
meshes, checkpoints, execution logs and project history stay in Project_data.
CPU/GPU selection is explicit. These projects use the installed preview57 runtime;
old-runtime checkpoints are not silently reinterpreted.

The physical comparison uses final windows 200–250 s (OTF) and 30–40 s (Mycek).
The first frames are startup, not equilibrium validation. Each VALIDATION_SPEC.json
records the settings, article, reference hashes and limitations. The paper's
original author code/input deck is not executed. Mycek's wake is measured at hub
height: the analysis shows those raw curves and a separately labelled assumed
depth projection. Input Ct/Cp replay is not blade-geometry prediction. The
1.0 m/s experiment adds a held-out operating-condition comparison.

References: [Abolghasemi, Piggott & Kramer](https://arxiv.org/abs/1810.09827),
[Mycek et al.](https://doi.org/10.1016/j.renene.2013.12.036).

Completed comparisons and figures are in **comparisons/REPORT.md**.
All five runs are complete. The published 2D mean-wake comparison passes;
the experimental wake comparison still depends on the stated depth projection.
The second-speed experimental load check passes without changing the input curve.
''',encoding='utf-8')
    print('Packaged local project shortcuts',HOME)


def seal():
    assert not (REPORT/'FINAL_MANIFEST.json').exists()
    assert (REPORT/'REPORT.md').is_file()
    tests=ET.parse(REPORT/'FINAL_ANALYSIS_TESTS.xml').getroot().find('testsuite')
    assert tests.get('tests')=='4' and tests.get('errors')==tests.get('failures')=='0'
    assert json.loads((REPORT/'PROJECT_REOPEN_CHECK.json').read_text())['status']=='passed'
    assert json.loads((REPORT/'RESTART_CHECK.json').read_text())['status']=='passed'
    metrics=json.loads((REPORT/'METRICS.json').read_text());receipts=[]
    for name,r in json.loads((HOME/'PROJECTS.json').read_text()).items():
        receipt=json.loads((REPORT/name/'RECEIPT.json').read_text());assert receipt['status']=='complete' and receipt['errors']==[]
        assert receipt['actual_run_button'] and receipt['maximum_water_residual_m3']<1e-6
        source=Path(receipt['output']);target=REPORT/name/'accepted_metadata';target.mkdir(exist_ok=False)
        for file in ['PROJECT.json','RUN_CONTEXT.json','PROGRESS.json','STATUS.json','DIAGNOSTICS.json','TURBINE_PERFORMANCE.json','TURBINE_PERFORMANCE.csv','TURBINE_PERFORMANCE_INDEX.json']:
            shutil.copy2(source/file,target/file)
        # Preserve raw artifact identities without duplicating the entire user
        # project. Means, extracted profiles and accepted metadata are sealed.
        index=json.loads((source/'frames/INDEX.json').read_text());shutil.copy2(source/'frames/INDEX.json',target/'FRAME_INDEX.json')
        receipts.append(receipt)
    assert metrics['otf_pooled']['RMSE_pp']<3. and metrics['otf_pooled']['maximum_absolute_pp']<6.
    assert metrics['otf_stationarity_pp']<.5 and metrics['mycek_stationarity_pp']<.5
    assert metrics['mycek_projected_ge_3D']['RMSE_pp']<5.
    assert abs(metrics['mycek_holdout_cpu_loads']['relative_Ct_error'])<.10 and abs(metrics['mycek_holdout_cpu_loads']['relative_Cp_error'])<.10
    assert metrics['backend']['maximum_mean_depth_m']<1e-4
    assert metrics['backend']['maximum_mean_velocity_m_s']<1e-4
    assert metrics['zero_control_maximum_speed_departure_percent']<1.
    segment=json.loads((REPORT/'otf_gpu_segment1/SEGMENT.json').read_text())
    retained=REPORT/'otf_gpu_segment1/accepted_metadata';retained.mkdir(exist_ok=False)
    for name in ['PROJECT.json','RUN_CONTEXT.json','STATUS.json','PROGRESS.json','DIAGNOSTICS.json']:
        shutil.copy2(Path(segment['output'])/name,retained/name)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(ROOT/'desktop/runtime/RUNTIME.json')=='1dcf63c3b753ebbbe012313f7ee05932461aeb38d9526f22d41af6c478aa097a'
    prior=ROOT/'validation/hydrokinetic_turbines/reports/mesh_grading57/FINAL_MANIFEST.json'
    d=json.loads(prior.read_text())
    for n,h in d['files'].items():assert sha(ROOT/n)==h,n
    save=dict(status='passed',runtime_unchanged=True,solver_modified=False,scientific_claim='Scoped mean-wake/conditional experimental comparison; no independent blade-performance or direct measured-depth wake validation',
        mutable_user_projects_excluded=True,prior_manifest_sha256=sha(prior),files={})
    files=[p for p in REPORT.rglob('*') if p.is_file() and 'ui_settings' not in p.parts]
    files.extend(ROOT/'desktop'/n for n in ['turbine_validation58.py','run_turbine_validation58_ui.py','analyze_turbine_validation58.py','package_turbine_validation58.py','inspect_turbine_validation58.py'])
    files.append(ROOT/'tests/test_turbine_validation58.py');save['files']={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    (REPORT/'FINAL_MANIFEST.json').write_text(json.dumps(save,indent=2)+'\n')
    comparisons=HOME/'comparisons';comparisons.mkdir(exist_ok=False)
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix.lower() in ['.md','.json','.png','.csv']:shutil.copy2(p,comparisons/p.name)
    for name in json.loads((HOME/'PROJECTS.json').read_text()):
        (comparisons/name).mkdir()
        for file in ['LOAD_HISTORY.png','UI_EXPORTED_PERFORMANCE.csv','RECEIPT.json']:
            shutil.copy2(REPORT/name/file,comparisons/name/file)
    print('Sealed evidence and copied comparison figures to',comparisons)


if __name__=='__main__':seal() if '--seal' in sys.argv else package()
