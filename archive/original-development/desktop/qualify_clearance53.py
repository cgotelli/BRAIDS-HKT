"""Clearance thresholds, synthetic terrain, packaged trajectories and identity."""
import argparse,datetime,json,sys,xml.etree.ElementTree as ET
from turbine_clearance53 import ROOT,OUT,TARGET,BASE,REPORT,FILES,PARENT,sha,save,verify


def main(seal=False):
    if seal and (REPORT/'FINAL_MANIFEST.json').exists():raise ValueError('Sealed evidence cannot be overwritten')
    runtime=verify(TARGET);m=json.loads((REPORT/'METRICS.json').read_text());tests={}
    for name,count in [('REGRESSION.xml',92),('UI_PACKAGED.xml',4)]:
        d=next(ET.parse(REPORT/name).getroot().iter('testsuite'));assert int(d.get('tests'))==count and all(d.get(k)=='0' for k in ['failures','errors','skipped']);tests[name]=d.attrib
    assert m['dem']['shape']==[140,320] and m['dem']['all_finite'] and m['dem']['maximum_elevation_error_m']<1e-10
    for case,data in m['cases'].items():
        assert data['maximum_backend_depth_m']<1e-4 and data['minimum_area_m2']>=.25**2*(1-1e-10)
        for b in ['cpu','gpu']:
            assert data[b]['maximum_water_residual_m3']<1e-9
            d=data[b]['last'];assert d['active']==(case=='normal')
            if case!='normal':assert d['thrust_N']==d['useful_power_W']==d['hydraulic_work_W']==0.
    for b in ['cpu','gpu']:
        d=json.loads((OUT/('coupled_'+b)/'CHECK.json').read_text());assert d['restart_exact'] and d['changed_clearance_restart_rejected'] and d['compiler_discovery_disabled']
    stage=json.loads((OUT/'STAGE.json').read_text());assert stage['runtime']==sha(TARGET/'RUNTIME.json')
    for name,h in stage['files'].items():
        assert runtime['files'][name]==h
        if not (REPORT/'FINAL_MANIFEST.json').exists():assert sha(ROOT/name)==h
    parent_root=ROOT/'desktop/qualification/hkt4/candidate';parent=verify(parent_root);assert sha(parent_root/'RUNTIME.json')==PARENT
    assert sha(TARGET/'PRECOMPILED.json')==sha(parent_root/'PRECOMPILED.json')
    assert {n for n,h in parent['files'].items() if runtime['files'][n]!=h}==set(FILES)
    prior={};entries=0
    for name in ['piano_hkt3g','loading_hkt3h','joly_source_hkt3i','otf_hkt3j','polygon_hkt3k','terrain_hkt3l','desktop_hkt4']:
        p=REPORT.parent/name/'FINAL_MANIFEST.json';d=json.loads(p.read_text())
        for n,h in d['files'].items():assert sha(ROOT/n)==h,n;entries+=1
        prior[str(p.relative_to(ROOT))]=sha(p)
    gates=dict(status='passed',runtime=sha(TARGET/'RUNTIME.json'),parent=PARENT,native_bindings_unchanged=True,tests=tests,
        maximum_backend_depth_m=max(v['maximum_backend_depth_m'] for v in m['cases'].values()),restart_exact=True,changed_limits_restart_rejected=True,independent_limits_qualified=True,prior_manifest_entries_verified=entries,prior_manifests=prior)
    if (REPORT/'FINAL_MANIFEST.json').exists():
        assert json.loads((REPORT/'GATES.json').read_text())==gates
        for n,h in json.loads((REPORT/'FINAL_MANIFEST.json').read_text())['files'].items():assert sha(ROOT/n)==h,n
    else:save(REPORT/'GATES.json',gates)
    if seal:
        assert (REPORT/'REPORT.md').exists() and (REPORT/'INSTALL.json').exists() and (REPORT/'SMOKE.json').exists()
        assert sha(BASE/'RUNTIME.json')==sha(TARGET/'RUNTIME.json');verify(BASE)
        files=[p for p in REPORT.iterdir() if p.is_file() and p.name!='FINAL_MANIFEST.json']+[TARGET/'RUNTIME.json',OUT/'STAGE.json']
        files.extend(ROOT/'desktop'/n for n in ['turbine_clearance53.py','trapezoid_turbine53.py','check_clearance53.py','check_clearance53_ui.py','analyze_clearance53.py','qualify_clearance53.py'])
        files.extend(ROOT/'tests'/n for n in ['test_hydrokinetic_clearance_hkt4b.py','test_hydrokinetic_clearance_ui_hkt4b.py'])
        files.extend(p for p in (OUT/'channel_template').rglob('*') if p.is_file())
        for b in ['cpu','gpu']:
            folder=OUT/('coupled_'+b);files.append(folder/'CHECK.json')
            for case in ['normal','surface_off','bed_off']:
                f=folder/case;files.append(f/'input.json');files.append(f/'prepared/PREPARED.json')
                receipt=json.loads((f/'prepared/PREPARED.json').read_text());files.extend(f/'prepared'/n for n in receipt['files'])
                files.extend(p for p in (f/'run').iterdir() if p.is_file());files.extend(p for p in (f/'run/frames').iterdir() if p.is_file())
        save(REPORT/'FINAL_MANIFEST.json',dict(**gates,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_manifest_authenticates_runtime_payload=True,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}))
    print('96 focused tests, both coupled clearance checks, restart and DEM audit pass;',entries,'prior manifest entries unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');main(p.parse_args().seal)
