"""Additive Desktop turbine release; independent project runtimes stay frozen."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/hkt4';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
FILES='''braids_desktop/app.py
braids_core/river_turbines.py
braids_core/river_project.py
braids_core/river_worker.py
braids_core/hydrokinetic_turbines.py
braids_core/sediment/adaptive_river_worker.py
braids_core/sediment/river_amr_transfer.py
braids_core/sediment/portable_river_runtime.py
braids_core/sediment/river_turbine_flow.py
braids_core/sediment/river_turbine_flow.h
braids_core/sediment/river_turbine_geometry.py
braids_core/sediment/river_hydrokinetic_stress.py
braids_core/sediment/river_hydrokinetic_stress.h
braids_core/sediment/river_hydrokinetic_transport.py
braids_core/sediment/river_hydrokinetic_transport.h
braids_desktop/turbines.py
braids_desktop/forms.py
braids_desktop/shell.py
braids_desktop/map_view.py
braids_desktop/results.py
braids_desktop/services.py
braids_desktop/preparation.py
braids_desktop/__init__.py'''.splitlines()


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def verify(root):
    d=json.loads((root/'RUNTIME.json').read_text())
    for name,h in d['files'].items():assert (root/name).resolve().is_relative_to(root.resolve()) and sha(root/name)==h,name
    return d


def main():
    assert sha(BASE/'RUNTIME.json')=='a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1'
    old=verify(BASE);m=json.loads(json.dumps(old))
    if not TARGET.exists():
        for name in old['files']:
            p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,p)
    else:m=verify(TARGET)
    for name in FILES:
        p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p);m['files'][name]=sha(p)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'STAGE.json',dict(base=sha(BASE/'RUNTIME.json'),runtime=sha(TARGET/'RUNTIME.json'),source_files={n:sha(ROOT/n) for n in FILES},scope='One zero-yaw hydraulic rotor; explicit CPU/GPU; no sediment, yaw force, support drag or new turbulence.'))
    print(TARGET)


if __name__=='__main__':main()
