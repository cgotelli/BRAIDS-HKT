"""Isolated MF candidate; native payloads remain those of preview44."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='d32038c677ce5d70cb70f08c1977a9208f06a0fe0dbee1a0ee9b655b4d7042ea'
OUT=ROOT/'desktop/qualification/morfac221';TARGET=OUT/'candidate_final'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_morfac221'
FILES=('braids_core/river_project.py','braids_core/sediment/transport_setup.py',
    'braids_core/sediment/river_suspension.py','braids_core/sediment/river_morfac.py',
    'braids_core/sediment/adaptive_river_worker.py','braids_core/sediment/river_classic_suspension.py',
    'braids_core/sediment/river_supplied_suspension.py',
    'braids_desktop/__init__.py','braids_desktop/forms.py',
    'braids_desktop/river_physics.py','braids_desktop/sediment_workflow.py')

def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    m=verify(BASE)
    if TARGET.exists():verify(TARGET)
    else:
        for n in m['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst);m['files'][n]=sha(dst)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    assert sha(TARGET/'PRECOMPILED.json')==sha(BASE/'PRECOMPILED.json')
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES,native_bindings_unchanged=True))
    print('Staged preview45',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
