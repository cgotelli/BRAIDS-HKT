"""Separate published-law candidate over the accepted-controller experiment."""
import json,shutil,sys
from stage_step190 import ROOT,sha
BASE=ROOT/'validation/_step190/candidate'
OUT=ROOT/'validation/_formula190';TARGET=OUT/'candidate'
FILES=('braids_core/river_model.py','braids_core/river_project.py','braids_core/sediment_method_catalog.py',
       'braids_core/sediment/capacity_parameters.py','braids_core/sediment/river_capacity.py',
       'braids_core/sediment/river_scalar_law.h','braids_core/sediment/river_smart_jaggi.py',
       'braids_core/sediment/river_corrections.py','braids_core/sediment/river_suspension.py',
       'braids_desktop/bedload_catalog.py','braids_desktop/river_physics.py')
def main():
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    if not TARGET.exists():shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Staged',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
