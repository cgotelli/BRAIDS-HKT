"""Stage only audited Smart-Jaeggi changes on the verified preview28 baseline."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_smart183';TARGET=OUT/'candidate';BASE=ROOT/'desktop/runtime'
FILES=('braids_core/river_model.py','braids_core/river_project.py','braids_core/sediment_method_catalog.py',
       'braids_core/sediment/capacity_parameters.py',
       'braids_core/sediment/river_capacity.py','braids_core/sediment/river_scalar_law.h',
       'braids_core/sediment/river_smart_jaggi.py','braids_core/sediment/river_corrections.py',
       'braids_core/sediment/river_suspension.py')
BASE_HASH='c0e931896a988945d71f963c2e13e2a9e860f8a3293fb2a7be1726401eb371d5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for name,h in manifest['files'].items():assert sha(BASE/name)==h,name
    if not TARGET.exists():shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    save(TARGET/'RUNTIME.json',manifest)
    save(OUT/'STAGE.json',dict(baseline=BASE_HASH,files={f:sha(ROOT/f) for f in FILES},
                             staged=sha(TARGET/'RUNTIME.json'),hydraulics_changed=False,Delft3D_changed=False))
    print('Staged',sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
