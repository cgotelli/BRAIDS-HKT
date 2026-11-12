"""Stage accepted-demand bank and toe composition with ordinary Exner inventory."""
from stage_grains212 import ROOT,sha,save,verify
import shutil
BASE=ROOT/'desktop/runtime'
BASE_HASH='569ad3147dbc3dc043df9505c7e32f8ecdca33be903a0d92ea529e61337f17e6'
OUT=ROOT/'desktop/qualification/banks224'
TARGET=OUT/('candidate_packaged' if (OUT/'BINDING_FIX.json').exists() else 'candidate_release')
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_banks224'
FILES=('braids_core/river_project.py','braids_core/sediment/bank_setup.py',
    'braids_core/sediment/collapse_setup.py','braids_core/sediment/adaptive_river_worker.py',
    'braids_core/sediment/river_physical_toe.py','braids_core/sediment/transport_setup.py',
    'braids_desktop/__init__.py','braids_desktop/river_physics.py',
    'braids_desktop/sediment_workflow.py')
def main():
    if (OUT/'BINDING_FIX.json').exists():
        raise RuntimeError('Preserve the qualified stages; the additive binding is recorded in BINDING_FIX.json')
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
    print('Staged preview48',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
