"""Isolated strict-diameter Desktop candidate; native solver payload unchanged."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_formula190/release_explicit'
OUT=ROOT/'validation/_grains191'
TARGET=OUT/'candidate'
FILES=('braids_core/river_model.py','braids_core/river_project.py',
       'braids_core/sediment/grain_diameter_contract.py','braids_desktop/river_physics.py',
       'braids_desktop/__init__.py')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():
        assert (root/n).resolve().is_relative_to(root.resolve()) and sha(root/n)==h,n
    return manifest
def main():
    assert sha(BASE/'RUNTIME.json')=='c813a52fb8c48913d973d1020bfb5fb72b4b61bf667d98def37614e1a87087d1'
    manifest=verify(BASE)
    if not TARGET.exists():shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(TARGET)
    print('Staged',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
