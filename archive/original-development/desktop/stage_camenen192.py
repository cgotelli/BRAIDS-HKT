"""Stage only authenticated payloads; preserve preview32 as the reference."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_grains191/candidate'
OUT=ROOT/'validation/_camenen192'
TARGET=OUT/'candidate'
BASE_HASH='23da55d1d9eab07f8a814cf101a3069cef78bd358e6bf80da66a6dec5a693384'
FILES=('braids_core/river_model.py','braids_core/river_project.py',
       'braids_core/sediment_method_catalog.py','braids_core/sediment/capacity_parameters.py',
       'braids_core/sediment/grain_diameter_contract.py','braids_core/sediment/river_scalar_law.h',
       'braids_core/sediment/river_capacity.py','braids_core/sediment/river_suspension.py',
       'braids_desktop/bedload_catalog.py','braids_desktop/river_physics.py','braids_desktop/__init__.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():
        assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return manifest
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    original=verify(BASE)
    if not TARGET.exists():
        for n in original['files']:
            dest=TARGET/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dest)
        save(TARGET/'RUNTIME.json',original)
    manifest=verify(TARGET)
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    save(TARGET/'RUNTIME.json',manifest);verify(TARGET)
    print('Staged',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
