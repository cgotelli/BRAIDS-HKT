"""Authenticate delivery, numerical-parent/UI-only diff and bounded evidence."""
import argparse,datetime,json,subprocess,sys,xml.etree.ElementTree as ET
from stage_hkt4 import ROOT,OUT,TARGET,BASE,sha,save,verify

REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_hkt4'


def main(seal=False):
    if seal and (REPORT/'FINAL_MANIFEST.json').exists():raise ValueError('Sealed evidence cannot be overwritten')
    current=verify(TARGET);installed=verify(BASE);assert current==installed
    parent=json.loads((OUT/'QUALIFIED_NUMERICAL_RUNTIME.json').read_text());difference=[n for n in parent['files'] if parent['files'][n]!=current['files'][n]]
    assert set(parent['files'])==set(current['files']) and difference==['braids_desktop/app.py']
    xmls={}
    for name,count in [('FINAL_CHECKS.xml',78),('UI_PACKAGED.xml',3)]:
        s=next(ET.parse(REPORT/name).getroot().iter('testsuite'));assert int(s.get('tests'))==count and all(s.get(k)=='0' for k in ['failures','errors','skipped']);xmls[name]=s.attrib
    metrics=json.loads((REPORT/'METRICS.json').read_text());assert metrics['ordinary_reference']['q_byte_identical']
    for kind in ['full','remesh']:
        m=metrics[kind];assert m['maximum_backend_depth_m']<1e-4 and m['minimum_area_m2']>=.125**2*(1-1e-10)
        assert all(m[b]['maximum_water_residual_m3']<1e-9 and m[b]['finite'] for b in ['cpu','gpu'])
    for b in ['cpu','gpu']:
        c=json.loads((OUT/('public_release_'+b)/'CHECK.json').read_text());assert c['runtime']==sha(OUT/'QUALIFIED_NUMERICAL_RUNTIME.json') and c['compiler_discovery_disabled'] and c['restart_exact'] and c['resume_cancel_exact'] and c['actual_remesh']
    assert sha(ROOT/'braids_core/sediment/river_turbine_flow.h')==sha(ROOT/'validation/hydrokinetic_turbines/terrain_hkt3l.h')
    smoke=json.loads((REPORT/'SMOKE.json').read_text());assert smoke['runtime']==sha(TARGET/'RUNTIME.json') and smoke['exit_code']==0
    prior_entries=0;prior={}
    for name in ['piano_hkt3g','loading_hkt3h','joly_source_hkt3i','otf_hkt3j','polygon_hkt3k','terrain_hkt3l']:
        p=REPORT.parent/name/'FINAL_MANIFEST.json';d=json.loads(p.read_text())
        for n,h in d['files'].items():assert sha(ROOT/n)==h,n;prior_entries+=1
        prior[str(p.relative_to(ROOT))]=sha(p)
    staged=json.loads((OUT/'STAGE.json').read_text())
    for n,h in staged['source_files'].items():
        assert current['files'][n]==h,n
        if not (REPORT/'FINAL_MANIFEST.json').exists():assert sha(ROOT/n)==h,n
    old=json.loads((OUT/'preview51_backup/RUNTIME.json').read_text());old_bindings=json.loads((OUT/'preview51_backup/PRECOMPILED.json').read_text());new_bindings=json.loads((TARGET/'PRECOMPILED.json').read_text())
    assert all(new_bindings[k]==v for k,v in old_bindings.items())
    added_bindings=sorted(set(new_bindings)-set(old_bindings));assert len(added_bindings)==4
    # Prove the UI-only launcher update has not reinterpreted prepared physics.
    code='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from braids_core.river_worker import load_prepared
for p in sys.argv[2:]:load_prepared(Path(p))
print("Final public prepared identities verified")'''
    done=subprocess.run([sys.executable,'-c',code,str(TARGET),str(OUT/'public_release_cpu/prepared'),str(OUT/'public_release_gpu/prepared')],text=True,capture_output=True,check=True)
    audit=dict(previous_runtime=sha(OUT/'preview51_backup/RUNTIME.json'),delivered_runtime=sha(TARGET/'RUNTIME.json'),
        changed={n:dict(before=old['files'].get(n),after=h) for n,h in current['files'].items() if old['files'].get(n)!=h},
        removed=sorted(set(old['files'])-set(current['files'])),prior_native_bindings_unchanged=True,new_bindings=added_bindings)
    gates=dict(status='Installed scoped Desktop turbine delivery passes',runtime=sha(TARGET/'RUNTIME.json'),numerical_parent=sha(OUT/'QUALIFIED_NUMERICAL_RUNTIME.json'),
        ui_only_difference=difference,tests=xmls,prior_manifest_entries_verified=prior_entries,prior_manifests=prior,prepared_check=done.stdout.strip(),
        ordinary_q_exact=True,restart_and_cancellation_exact=True,real_terrain_remesh=True,compiler_discovery_disabled=True,installed_smoke_pass=True)
    if (REPORT/'FINAL_MANIFEST.json').exists():
        assert json.loads((REPORT/'GATES.json').read_text())==gates
        for n,h in json.loads((REPORT/'FINAL_MANIFEST.json').read_text())['files'].items():assert sha(ROOT/n)==h,n
    else:
        save(REPORT/'SOURCE_AUDIT.json',audit);save(REPORT/'GATES.json',gates)
    if seal:
        files=[p for p in REPORT.iterdir() if p.is_file() and p.name!='FINAL_MANIFEST.json']
        files.extend(OUT/n for n in ['STAGE.json','PACKAGE.json','BINDINGS.json','QUALIFIED_NUMERICAL_RUNTIME.json','STARTUP_UI_FIX.json'])
        files.append(TARGET/'RUNTIME.json')
        for n in ['stage_hkt4.py','record_hkt4.py','package_hkt4.py','check_hkt4_packaged.py','check_hkt4_ordinary.py','check_hkt4_ui.py','analyze_hkt4.py','install_hkt4.py','qualify_hkt4.py']:files.append(ROOT/'desktop'/n)
        files.extend(ROOT/'tests'/n for n in ['test_hydrokinetic_desktop_hkt4.py','test_hydrokinetic_desktop_ui_hkt4.py'])
        files.append(ROOT/'validation/hydrokinetic_turbines/desktop_hkt4_case.py')
        for b in ['cpu','gpu']:
            folder=OUT/('public_release_'+b);files.extend(folder/n for n in ['CHECK.json','input.json'])
            prep=folder/'prepared';receipt=json.loads((prep/'PREPARED.json').read_text());files.append(prep/'PREPARED.json');files.extend(prep/n for n in receipt['files'])
            for run in ['full','restart','cancel','resume_cancel','remesh']:
                f=folder/run;files.extend(p for p in f.iterdir() if p.is_file());files.extend(p for p in (f/'frames').iterdir() if p.is_file())
        example=ROOT/'desktop/examples/turbine_hkt4';files.extend(p for p in example.rglob('*') if p.is_file())
        save(REPORT/'FINAL_MANIFEST.json',dict(**gates,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            runtime_manifest_transitively_authenticates_candidate=True,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print('All delivery gates pass;',prior_entries,'prior entries unchanged; 78 focused tests; final runtime',sha(TARGET/'RUNTIME.json'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
