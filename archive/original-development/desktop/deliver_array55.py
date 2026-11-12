"""Gate/install/seal the qualified multi-rotor Desktop preview."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/array_hkt4d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')


def verify(root):
    d=read(root/'RUNTIME.json')
    for n,h in d['files'].items():assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return d


def gate():
    stage=read(OUT/'STAGE.json');assert sha(TARGET/'RUNTIME.json')==stage['runtime'];verify(TARGET)
    previous_runtime=ROOT/'desktop/qualification/decimal_mesh54/candidate';assert sha(TARGET/'PRECOMPILED.json')==sha(previous_runtime/'PRECOMPILED.json')
    for name,count in [('FINAL_REGRESSION.xml',100),('FINAL_UI.xml',8)]:
        suite=next(ET.parse(OUT/name).getroot().iter('testsuite'));assert int(suite.get('tests'))==count and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    data=read(REPORT/'METRICS.json');assert data['preparation']['4']['speedup']>1.
    assert max(r['maximum_depth_m'] for r in data['array_backend_startup'])<1e-3
    assert data['frozen_single_regression']['maximum_depth_m']<1e-3
    assert max(s['useful_power_W']['peak_to_peak_percent'] for s in data['stationarity_cpu_50_60_s'].values())<1.
    for name in ['candidate_array_cpu','candidate_array_gpu','baseline_single_gpu','candidate_single_gpu']:
        c=read(OUT/name/'CHECK.json');assert c['restart_exact'] and c['compiler_discovery_disabled'] and c['maximum_water_residual_m3']<1e-8
    assert all(read(OUT/'PERFORMANCE_RESTART_EXACT.json').values())
    assert all(r['weights_exact'] for r in read(OUT/'PROJECTED_ROTATION_QUADRATURE.json'))
    assert data['candidate_array_cpu']['end_s']==60. and data['candidate_array_gpu']['end_s']==6.
    assert data['remesh']['all_rotor_patches_finest'] and data['remesh']['fixed_hubs_preserved'] and data['remesh']['maximum_backend_depth_m']<1e-3
    for backend in ['cpu','gpu']:
        c=read(OUT/('remesh_'+backend)/'CHECK.json');assert c['restart_after_remesh_exact'] and c['changed'] and c['changed'][0]['time_s']>0
    previous=REPORT.parent/'power_mesh_hkt4c/FINAL_MANIFEST.json';manifest=read(previous)
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
    save(REPORT/'GATES.json',dict(status='passed',runtime=stage['runtime'],parent=stage['parent'],packaged_checks=108,native_bindings_unchanged=True,
        cpu_constant_inflow_s=60.,gpu_matched_startup_s=6.,both_backends_restart_exact=True,actual_two_rotor_remesh_and_restart=True,
        previous_manifest_sha256=sha(previous),previous_entries_verified=len(manifest['files'])))
    return stage


def install():
    stage=gate();old=verify(BASE);new=verify(TARGET);assert sha(BASE/'RUNTIME.json')==stage['parent']
    command=r"Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\/](launch|worker)\.py|braids_desktop\.app|desktop[\\/]runtime[\\/]braids_core[\\/]river_launcher\.py')} | Select-Object -ExpandProperty ProcessId"
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if processes:raise RuntimeError('Save and close BRAIDS Desktop/workers before installing: '+processes)
    changed={n for n,h in new['files'].items() if old['files'].get(n)!=h};assert changed==set(stage['files']) and set(old['files'])==set(new['files'])
    backup=OUT/'preview54_backup';backup.mkdir(exist_ok=False)
    for n in changed|{'RUNTIME.json'}:
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in changed:shutil.copy2(TARGET/n,BASE/n)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    path=ROOT/'desktop/PROVENANCE.json';d=read(path);report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    d.update(version='0.1.0-preview55',ui_qualification=report);d['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=stage['runtime'],qualification=report,
        modification_scope='Multiple independent covered rotors; shared geometry, gradients, ambient stress and summed pressure-balanced source. Exact grid-coincident face reuse and indexed mapping. Per-rotor editing, plots and selected/all CSV. Native kernels unchanged.')
    save(path,d);save(REPORT/'INSTALL.json',dict(version='0.1.0-preview55',runtime=stage['runtime'],parent=stage['parent'],files=sorted(changed),native_bindings_unchanged=True,independent_runtimes_modified=False));print('Installed preview55')


def seal():
    if (REPORT/'FINAL_MANIFEST.json').exists():raise ValueError('Sealed evidence cannot be overwritten')
    stage=gate();assert sha(BASE/'RUNTIME.json')==stage['runtime'];verify(BASE);assert read(REPORT/'SMOKE.json')['status']=='passed'
    files=[p for p in REPORT.iterdir() if p.is_file()]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(TARGET) and not p.is_relative_to(OUT/'preview54_backup'))
    files.append(TARGET/'RUNTIME.json')
    files.extend(ROOT/'desktop'/n for n in ['profile_turbine_array55.py','benchmark_geometry_threads55.py','check_array55.py','check_array_remesh55.py','analyze_array55.py','deliver_array55.py'])
    files.extend(ROOT/'tests'/n for n in ['test_turbine_array55.py','test_turbine_array_ui55.py'])
    save(REPORT/'FINAL_MANIFEST.json',dict(status='passed',runtime=stage['runtime'],candidate_manifest_authenticates_runtime_payload=True,editable_demo_excluded=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}));print('Sealed preview55 evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gate','install','seal']);getattr(__import__(__name__),p.parse_args().action)()
