"""Stage S1 configuration UI over preview38; no numerical payload changes."""
from stage_grains212 import ROOT,sha,save,verify
import shutil

BASE=ROOT/'desktop/runtime'
BASE_HASH='cbebb461ab092a30bedb944d67380ca9e5832d55fdcf6d9bf73d366c9ed6e2b4'
OUT=ROOT/'validation/_suspended214'
TARGET=OUT/'candidate'
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_suspension214'
FILES=('braids_core/river_project.py','braids_core/sediment/transport_setup.py',
       'braids_desktop/__init__.py','braids_desktop/river_physics.py',
       'braids_desktop/sediment_workflow.py','braids_desktop/forms.py')


def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    original=verify(BASE)
    if TARGET.exists():
        current=verify(TARGET)
        assert set(current['files'])-set(original['files'])<=set(FILES)
        for n,h in original['files'].items():
            if n not in FILES:assert current['files'][n]==h,n
    else:
        for n in original['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
        original['files'][n]=sha(dst)
    save(TARGET/'RUNTIME.json',original);verify(TARGET)
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,
        changed=FILES,numerical_payloads_unchanged=True,suspension_execution_admitted=False))
    print('Staged preview39',sha(TARGET/'RUNTIME.json'),flush=True)


if __name__=='__main__':main()
