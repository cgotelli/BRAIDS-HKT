"""Gate and install the decimal-cell and velocity-driven rotor update."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/decimal_mesh54';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_mesh_hkt4c'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')


def verify(root):
    d=json.loads((root/'RUNTIME.json').read_text())
    for n,h in d['files'].items():assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return d


def gate():
    stage=json.loads((OUT/'STAGE.json').read_text());assert sha(TARGET/'RUNTIME.json')==stage['runtime'];verify(TARGET)
    assert sha(TARGET/'PRECOMPILED.json')==sha(ROOT/'desktop/qualification/turbine_clearance53/candidate/PRECOMPILED.json')
    for name,count in [('PACKAGED_REGRESSION.xml',84),('POWER_UI.xml',6),('REFERENCE_REGRESSION.xml',92)]:
        suite=next(ET.parse(OUT/name).getroot().iter('testsuite'))
        assert int(suite.get('tests'))==count and all(suite.get(k)=='0' for k in ['failures','errors','skipped']),name
    d=json.loads((REPORT/'METRICS.json').read_text());assert d['mesh_after']['non_square']==0 and d['original_quarter_metre_hierarchy_exact']
    for name in ['off','on','rpm']:
        assert d[name]['maximum_depth_difference_m']<1e-4 and d[name]['minimum_area_m2']>=.01-1e-9
        for backend in ['cpu','gpu']:
            assert d[name][backend]['maximum_water_residual_m3']<1e-8
            if name!='off':assert d[name][backend]['restart_exact']
    # Every earlier sealed source, figure and result remains byte-identical.
    previous=REPORT.parent/'clearance_hkt4b/FINAL_MANIFEST.json';manifest=json.loads(previous.read_text())
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
    save(REPORT/'GATES.json',dict(status='passed',runtime=stage['runtime'],parent=stage['parent'],packaged_checks=90,additional_retained_research_checks=64,
        previous_manifest_sha256=sha(previous),native_bindings_unchanged=True,exact_restart=True,decimal_grid_fixed=True))
    return stage


def install():
    stage=gate();old=verify(BASE);new=verify(TARGET);assert sha(BASE/'RUNTIME.json')==stage['parent']
    command="Get-CimInstance Win32_Process | Where-Object {($_.Name -match '^python|^BRAIDS') -and ($_.CommandLine -match 'desktop[\\\\/]launch.py|braids_desktop.app|desktop[\\\\/]runtime[\\\\/]braids_core[\\\\/]river_launcher.py')} | Select-Object -ExpandProperty ProcessId"
    processes=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True).strip()
    if processes:raise RuntimeError('Save and close BRAIDS Desktop before installing: '+processes)
    changed={n for n,h in new['files'].items() if old['files'].get(n)!=h};assert changed==set(stage['files']) and set(old['files'])<=set(new['files'])
    backup=OUT/'preview53_backup';backup.mkdir(exist_ok=False)
    for n in (changed & set(old['files']))|{'RUNTIME.json'}:
        p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    shutil.copy2(ROOT/'desktop/PROVENANCE.json',backup/'PROVENANCE.json')
    for n in changed:
        p=BASE/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(TARGET/n,p)
    shutil.copy2(TARGET/'RUNTIME.json',BASE/'RUNTIME.json');verify(BASE)
    path=ROOT/'desktop/PROVENANCE.json';d=json.loads(path.read_text());report=str((REPORT/'REPORT.md').relative_to(ROOT)).replace('\\','/')
    d.update(version='0.1.0-preview54',ui_qualification=report);d['solver'].update(source=str(TARGET.relative_to(ROOT)).replace('\\','/'),runtime_manifest_sha256=stage['runtime'],qualification=report,
        modification_scope='Coordinate-ULP-aware area classification retains complete decimal-grid squares; genuine cut-cell merging unchanged. Explicit fixed-RPM turbine curves resolve Ct/Cp and upstream velocity together. Per-turbine CSV and plots; no blade-geometry predictor. Same native kernels.')
    save(path,d);save(REPORT/'INSTALL.json',dict(version='0.1.0-preview54',runtime=stage['runtime'],parent=stage['parent'],files=sorted(changed),native_bindings_unchanged=True,independent_runtimes_modified=False))
    print('Installed preview54')


def seal():
    if (REPORT/'FINAL_MANIFEST.json').exists():raise ValueError('Sealed evidence cannot be overwritten')
    stage=gate();assert sha(BASE/'RUNTIME.json')==stage['runtime'];verify(BASE)
    assert json.loads((REPORT/'SMOKE.json').read_text())['status']=='passed'
    files=[p for p in REPORT.iterdir() if p.is_file()]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(TARGET) and not p.is_relative_to(OUT/'preview53_backup'))
    files.append(TARGET/'RUNTIME.json')
    files.extend(ROOT/'desktop'/n for n in ['check_decimal_mesh54.py','analyze_power_mesh54.py','deliver_power_mesh54.py'])
    files.extend(ROOT/'tests'/n for n in ['test_decimal_mesh54.py','test_turbine_performance54.py','test_turbine_power_ui54.py'])
    save(REPORT/'FINAL_MANIFEST.json',dict(status='passed',runtime=stage['runtime'],candidate_manifest_authenticates_runtime_payload=True,
        editable_demo_excluded=True,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print('Sealed preview54 evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gate','install','seal']);getattr(__import__(__name__),p.parse_args().action)()
