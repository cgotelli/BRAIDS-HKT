"""Deliver editable refined-wake projects and the cumulative-budget repair."""
from nearwake60 import *
import shutil,subprocess,xml.etree.ElementTree as ET
TARGET=ROOT/'desktop/qualification/nearwake60/candidate'
OUT=TARGET.parent


def package():
    manifest=read(HOME/'PROJECTS.json');python=ROOT/'desktop/.venv/Scripts/pythonw.exe';launch=ROOT/'desktop/launch.py'
    menu=['@echo off','setlocal','cd /d "%~dp0"','echo BRAIDS near-wake audit','echo.',
          'echo 1. Eight cells across rotor and refined wake (GPU)',
          'echo 2. Four cells across rotor and refined wake (GPU)',
          'echo 3. Matched four-cell CPU check',
          'choice /c 123 /n /m "Open project [1-3]: "']
    for i,name in reversed(list(enumerate(['wake8_gpu','wake4_gpu','wake4_cpu'],1))):
        menu.append(f'if errorlevel {i} (start "" "{python}" -B "{launch}" "{manifest[name]["project"]}" & exit /b)')
    (HOME/'Open projects.cmd').write_text('\n'.join(menu)+'\n',encoding='utf-8')
    (Path.home()/'Desktop/Open BRAIDS Near Wake Validation.cmd').write_text('@echo off\ncall "'+str(HOME/'Open projects.cmd')+'"\n',encoding='utf-8')
    for name,r in manifest.items():
        (Path(r['folder'])/'Open project.cmd').write_text('@echo off\nstart "" "'+str(python)+'" -B "'+str(launch)+'" "'+r['project']+'"\n',encoding='utf-8')
    print(HOME/'Open projects.cmd')


def gates():
    stage=read(REPORT/'ARITHMETIC_STAGE.json');assert sha(TARGET/'RUNTIME.json')==stage['runtime_sha256']
    for n,h in read(TARGET/'RUNTIME.json')['files'].items():assert sha(TARGET/n)==h,n
    for file in ['REGRESSION.json','ARITHMETIC_REPLAY.json','CONTROL_ARITHMETIC_EXACT.json']:
        assert read(REPORT/file)['status']=='passed',file
    suite=next(ET.parse(REPORT/'FOCUSED_CHECKS.xml').getroot().iter('testsuite'))
    assert int(suite.get('tests'))==66 and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    for name in ['wake4_gpu','wake8_gpu','wake4_cpu','zero4_gpu','zero8_corrected']:
        r=read(REPORT/name/'RECEIPT.json');assert r['status']=='complete' and r['errors']==[] and r['actual_run_button']
    assert all(read(REPORT/'METRICS.json')['aims']['published_2d_model'].values())
    previous=PARENT/'FINAL_MANIFEST.json'
    for n,h in read(previous)['files'].items():assert sha(ROOT/n)==h,n
    result=dict(status='passed',runtime_sha256=stage['runtime_sha256'],parent=stage['parent_runtime'],
        focused_checks=66,exact_hydro_bedload_AMR_restart_controls=4,failed_control_cold_replay_exact_through_s=250,
        previous_evidence_verified=True,physical_aims=read(REPORT/'METRICS.json')['aims'])
    save(REPORT/'GATES.json',result);return result


def install():
    g=read(REPORT/'GATES.json');assert g['status']=='passed'
    assert sha(TARGET/'RUNTIME.json')==g['runtime_sha256'] and sha(RUNTIME/'RUNTIME.json')==g['parent']
    query=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^pythonw?\.exe$|^BRAIDS') -and ($_.CommandLine -match 'Desktop[\\/]BRAIDS[\\/]desktop[\\/]launch\.py|Desktop[\\/]BRAIDS[\\/]desktop[\\/]runtime[\\/]')} | Select-Object -ExpandProperty ProcessId"
    active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
    if active:raise RuntimeError('Installed Desktop/worker still active: '+active)
    old=read(RUNTIME/'RUNTIME.json');new=read(TARGET/'RUNTIME.json');changed=set(read(REPORT/'ARITHMETIC_STAGE.json')['changed'])
    assert changed=={n for n,h in new['files'].items() if h!=old['files'].get(n)}
    for name in changed:assert sha(TARGET/name)==new['files'][name],name
    backup=OUT/'preview58_backup';backup.mkdir(exist_ok=False)
    for name in changed|{'RUNTIME.json'}:
        if (RUNTIME/name).exists():
            p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUNTIME/name,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for name in changed:shutil.copy2(TARGET/name,RUNTIME/name)
    shutil.copy2(TARGET/'RUNTIME.json',RUNTIME/'RUNTIME.json')
    shutil.copy2(TARGET/'braids_desktop/__init__.py',ROOT/'braids_desktop/__init__.py')
    d=read(ROOT/'desktop/PROVENANCE.json');report='validation/hydrokinetic_turbines/reports/nearwake60/REPORT.md'
    d.update(version='0.1.0-preview59',ui_qualification=report)
    d['solver'].update(source='desktop/qualification/nearwake60/candidate',runtime_manifest_sha256=g['runtime_sha256'],qualification=report,
        modification_scope='Cumulative FP64 water diagnostic allowance with unchanged per-increment guard, wall/stock checks, physical equations, native kernels and project v39. Wake improvement is a qualified project mesh correction.')
    save(ROOT/'desktop/PROVENANCE.json',d)
    record=read(HOME/'PROJECTS.json')['wake8_gpu']
    smoke=subprocess.run([sys.executable,'-B',str(ROOT/'desktop/launch.py'),record['project'],'--smoke'],cwd=ROOT,
        env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),capture_output=True,text=True,timeout=180)
    save(REPORT/'SMOKE.json',dict(exit_code=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr,real_project_argument=True))
    assert smoke.returncode==0,smoke.stdout+smoke.stderr
    save(REPORT/'INSTALL.json',dict(status='passed',version='0.1.0-preview59',runtime_sha256=g['runtime_sha256'],changed=sorted(changed),prior_runtime=g['parent']))
    print('Installed preview59 and smoke-tested real project launch')


def seal():
    metrics=read(REPORT/'METRICS.json');reopen=read(REPORT/'PROJECT_REOPEN_CHECK.json')
    assert reopen['status']=='passed'
    assert sha(RUNTIME/'RUNTIME.json')==read(REPORT/'ARITHMETIC_STAGE.json')['runtime_sha256']
    assert read(REPORT/'INSTALL.json')['status']=='passed'
    assert read(REPORT/'INSTALLED_IDENTITIES.json')['status']=='passed'
    from braids_core.river_launcher import verify
    verify()
    for name in ['wake4_gpu','wake8_gpu','wake4_cpu','zero4_gpu','zero8_corrected']:
        r=read(REPORT/name/'RECEIPT.json');assert r['status']=='complete' and r['errors']==[]
        source=Path(r['output']);dest=REPORT/name/'accepted_metadata';dest.mkdir(exist_ok=False)
        for file in ['PROJECT.json','RUN_CONTEXT.json','PROGRESS.json','STATUS.json','DIAGNOSTICS.json','TURBINE_PERFORMANCE.csv','TURBINE_PERFORMANCE.json','TURBINE_PERFORMANCE_INDEX.json']:
            shutil.copy2(source/file,dest/file)
        shutil.copy2(source/'frames/INDEX.json',dest/'FRAME_INDEX.json')
    refs=read(PARENT/'REFERENCE_IDENTITIES.json')
    for file,h in refs['files'].items():assert sha(ROOT/file)==h
    save(REPORT/'REFERENCE_IDENTITIES.json',refs)
    retained=REPORT/'references';retained.mkdir(exist_ok=False)
    for filename in ['nash_2015_figure6b.csv','nash_2015_figure6b.json','nash_2015_figure7.csv','nash_2015_figure7.json']:
        shutil.copy2(ROOT/'validation/hydrokinetic_turbines/measurements'/filename,retained/filename)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',REPORT/'DELIVERY_PROVENANCE.json')
    # Preserve only the metadata/figures and reproducible scripts in this seal;
    # the editable project outputs are not recopied into the repository.
    files=[p for p in REPORT.rglob('*') if p.is_file() and not {'replay_work','ui_settings'}&set(p.parts)]
    files.extend(ROOT/'desktop'/n for n in ['nearwake60.py','run_nearwake60_ui.py','audit_nearwake60.py',
        'analyze_nearwake60.py','ledger_nearwake60.py','figures_nearwake60.py','verify_nearwake60.py',
        'finalize_nearwake60.py','inspect_nearwake60.py','package_nearwake60.py','compare_mesh_controls60.py',
        'qualify_nearwake60.py','stage_nearwake60.py','test_nearwake60_budget.py','check_nearwake60_budget.py','run_nearwake60_tests.py',
        'document_nearwake60.py','verify_nearwake60_installed.py'])
    files.extend(TARGET/n for n in read(TARGET/'RUNTIME.json')['files'] if not n.startswith('.venv/'))
    files.append(TARGET/'RUNTIME.json')
    assert not (REPORT/'FINAL_MANIFEST.json').exists()
    save(REPORT/'FINAL_MANIFEST.json',dict(status='complete',runtime_sha256=sha(RUNTIME/'RUNTIME.json'),
        physical_solver_modified=False,cumulative_budget_diagnostic_corrected=True,installed_version='0.1.0-preview59',
        interpretation='Mesh correction qualified separately from measured near-wake validation. Failed physical aims remain explicit.',
        physical_aims=metrics['aims'],files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    dest=HOME/'comparisons';dest.mkdir(exist_ok=False)
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix in {'.md','.json','.png','.csv'}:shutil.copy2(p,dest/p.name)
    shutil.copytree(retained,dest/'references')
    print('Sealed audit and delivered comparison figures:',dest)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['package','gates','install','seal']);v=a.parse_args();globals()[v.action]()
