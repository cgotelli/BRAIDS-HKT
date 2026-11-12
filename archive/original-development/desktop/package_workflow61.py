"""Deliver only the qualified Desktop workflow changes; keep preview59 frozen."""
import argparse,hashlib,json,os,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/workflow61';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61';HOME=Path.home()/'Desktop/BRAIDS_Workflow61'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def package():
    projects=read(HOME/'PROJECTS.json');python=ROOT/'desktop/.venv/Scripts/pythonw.exe';launch=ROOT/'desktop/launch.py'
    for key,name in projects.items():
        project=Path(name);(project.parent/'Open project.cmd').write_text('@echo off\nstart "" "'+str(python)+'" -B "'+str(launch)+'" "'+str(project)+'"\n',encoding='utf-8')
    menu=['@echo off','setlocal','cd /d "%~dp0"','echo BRAIDS - Wake and saved-state workflows','echo 1. Automatic 10D turbine wake refinement (completed 8 s example)','echo 2. Hydraulic state used for bedload initialization','echo 3. Bedload on a different mesh (completed explicit mapping)','choice /c 123 /n /m "Open project [1-3]: "']
    for i,key in reversed(list(enumerate(['turbine','hydro','bedload'],1))):menu.append(f'if errorlevel {i} (start "" "{python}" -B "{launch}" "{projects[key]}" & exit /b)')
    (HOME/'Open projects.cmd').write_text('\n'.join(menu)+'\n',encoding='utf-8')
    (Path.home()/'Desktop/Open BRAIDS Wake and Restart.cmd').write_text('@echo off\ncall "'+str(HOME/'Open projects.cmd')+'"\n',encoding='utf-8')
    (HOME/'README.txt').write_text('BRAIDS Desktop preview60\n\nOpen projects.cmd offers three editable projects. Results and history are retained in each Project_data folder. No simulation is left running.\n\nTurbines: Refine downstream wake, default 10 rotor diameters, in a 5D-wide corridor. Increase the length or uncheck it, then Prepare. It uses the chosen smallest cell size.\n\nProfiles: Interpolate cell values is checked by default. Uncheck it for the raw steps. Solids remain gaps. CSV records the sampling mode.\n\nSaved states: Run > Browse checkpoint, default Load useful fields. Prepare the target setup, then Check saved state against prepared mesh. Same cells copy directly. If different, click Map saved state to prepared mesh explicitly, then Run. End time must exceed the saved time. The domain, solids and coordinate reference system must be unchanged.\n\nExample 02_Add_bedload_and_map/Hydraulic_checkpoint.npz is the self-contained hydraulic source. The MPM example starts at 0.12 s and ends at 0.24 s. These very short runs test the workflow, not sediment deposition or physical validation. Increase End time to run further. Newly enabled sediment uses the target erodible bed settings.\n\nExact continuation is the other restart mode and preserves the complete same-setup state. Old public v4 checkpoints can be imported as fields from their original project folder, but cannot execute as old solvers. Old project formats are rejected.\n\nTurbine/bedload coupling remains a separate next milestone; do not enable both in one project yet.\n',encoding='utf-8')
    print(HOME/'Open projects.cmd',flush=True)

def gates():
    stage=read(REPORT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==stage['runtime_sha256']
    files=read(TARGET/'RUNTIME.json')['files']
    for n,h in files.items():assert sha(TARGET/n)==h,n
    source_files={n:h for n,h in files.items() if (n.startswith(('braids_core/','braids_gpu/','solver/wave/','solver/cut_cells/')) and Path(n).suffix in ('.py','.h','.cpp','.cu','.f90','.f')) or (str(Path(n).parent).replace('\\','/') in ('validation/sediment/river_morphodynamics','validation/gpu_fwave') and Path(n).suffix in ('.py','.h','.cu')) or n=='braids_qgis/core/raster_surface.py'}
    source_id=hashlib.sha256(json.dumps(source_files,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    suite=next(ET.parse(REPORT/'FOCUSED_CHECKS.xml').getroot().iter('testsuite'))
    assert int(suite.get('tests'))==83 and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    for file in ['UI_CHECK.json','VISUAL_CHECK.json','FIGURE_CHECKS.json']:assert read(REPORT/file)['status']=='passed'
    rows=[]
    for backend in ['cpu','gpu']:
        for case in ['same','mapped','jiang_dynamic','ordinary_dynamic']:
            c=read(REPORT/f'CHECK_{backend}_{case}.json');assert c['status']=='passed' and all(c['exact_restart_fields'].values())
            receipt=read(Path(c['root'])/'target/prepared/PREPARED.json');assert receipt['runtime_id']==source_id,(case,receipt['runtime_id'],source_id)
            rows.append(dict(backend=backend,case=case,root=c['root'],conservation=c['decision']['conservation']))
    previous=ROOT/'validation/hydrokinetic_turbines/reports/nearwake60/FINAL_MANIFEST.json'
    for n,h in read(previous)['files'].items():assert sha(ROOT/n)==h,n
    save(REPORT/'GATES.json',dict(status='passed',runtime_sha256=stage['runtime_sha256'],parent=stage['parent_runtime'],focused_checks=83,coupled_checks=rows,prior_evidence_unchanged=True,native_kernels_unchanged=True))
    print('Release gates passed',flush=True)

def install():
    g=read(REPORT/'GATES.json');assert g['status']=='passed' and sha(TARGET/'RUNTIME.json')==g['runtime_sha256'] and sha(BASE/'RUNTIME.json')==g['parent']
    query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'Desktop[\\/]BRAIDS[\\/]desktop[\\/]launch\.py|Desktop[\\/]BRAIDS[\\/]desktop[\\/]runtime[\\/]')} | Select-Object -ExpandProperty ProcessId"
    active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
    if active:raise RuntimeError('Installed Desktop/worker still active: '+active)
    stage=read(REPORT/'STAGE.json');old=read(BASE/'RUNTIME.json');new=read(TARGET/'RUNTIME.json');changed=set(stage['changed'])
    assert changed=={n for n,h in new['files'].items() if h!=old['files'].get(n)}
    backup=OUT/'preview59_backup';backup.mkdir(exist_ok=False)
    for n in changed|{'RUNTIME.json'}:
        if (BASE/n).exists():p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in changed:
        assert sha(TARGET/n)==new['files'][n];(BASE/n).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');shutil.copy2(TARGET/'braids_desktop/__init__.py',ROOT/'braids_desktop/__init__.py')
    d=read(ROOT/'desktop/PROVENANCE.json');report='validation/hydrokinetic_turbines/reports/desktop_workflow61/REPORT.md'
    d.update(version='0.1.0-preview60',ui_qualification=report)
    d['solver'].update(source='desktop/qualification/workflow61/candidate',runtime_manifest_sha256=g['runtime_sha256'],qualification=report,modification_scope='Optional automatic 10D wake refinement; bounded linear profile display; explicit same-domain accepted-field initialization and conservative changed-mesh mapping at the saved clock; self-contained v5 checkpoints; coordinate-scale geometry closure guard. Native physics and libraries unchanged. Project v40.')
    save(ROOT/'desktop/PROVENANCE.json',d)
    project=read(HOME/'PROJECTS.json')['turbine'];smoke=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/launch.py'),project,'--smoke'],cwd=ROOT,env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),capture_output=True,text=True,timeout=180)
    save(REPORT/'INSTALLED_SMOKE.json',dict(exit_code=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr,project=project));assert smoke.returncode==0,smoke.stdout+smoke.stderr
    save(REPORT/'INSTALL.json',dict(status='passed',version='0.1.0-preview60',runtime_sha256=g['runtime_sha256'],prior_runtime=g['parent'],changed=sorted(changed),backup=str(backup)))
    print('Installed preview60; real project launch passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['package','gates','install']);globals()[p.parse_args().action]()
