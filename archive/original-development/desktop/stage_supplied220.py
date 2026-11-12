"""Stage supplied-suspension delivery over the immutable preview43 package."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='cdd68a8952cee7716c0d148841c6d65440e887ddc07dadfab17f1eb425173cd2'
OUT=ROOT/'validation/_supplied220';TARGET=OUT/'candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_supplied220'
FILES=('braids_core/river_project.py','braids_core/river_model.py','braids_core/river_worker.py',
    'braids_core/sediment/transport_setup.py','braids_core/sediment/river_suspension.py',
    'braids_core/sediment/river_supplied_suspension.py',
    'braids_desktop/__init__.py','braids_desktop/river_physics.py','braids_desktop/forms.py',
    'braids_desktop/sediment_workflow.py')

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    original=verify(BASE)
    if TARGET.exists():m=verify(TARGET)
    else:
        m=original
        for n in m['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst);m['files'][n]=sha(dst)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    assert sha(TARGET/'PRECOMPILED.json')==sha(BASE/'PRECOMPILED.json')
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES,native_bindings_unchanged=True))
    print('Staged supplied suspension',sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
