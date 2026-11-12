"""Isolated MF candidate; native payloads remain those of preview45."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='62cd0ff44584d79934c1c59ac55ab575cb214f125a97eb4a7015f9361feb9bc1'
OUT=ROOT/'desktop/qualification/collapse222';TARGET=OUT/'candidate_final'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_collapse222'
FILES=('braids_core/river_project.py','braids_core/sediment/transport_setup.py',
    'braids_core/sediment/collapse_setup.py','braids_core/sediment/adaptive_river_worker.py',
    'braids_desktop/__init__.py','braids_desktop/forms.py','braids_desktop/river_physics.py','braids_desktop/sediment_workflow.py')


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
    print('Staged preview46',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
