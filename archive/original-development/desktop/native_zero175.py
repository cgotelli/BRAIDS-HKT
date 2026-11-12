"""Supported FLOW numerical-scheme contrast at exactly zero viscosity."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop'),str(ROOT/'validation/sediment/delft3d_current')]
from check_bedload161 import replace
from stage_wall172 import save,sha
import runtime


def main():
    parent=ROOT/'validation/_corner174/delft';out=ROOT/'validation/_zero175/native_cyclic'
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest
        shutil.copy2(parent/name,out/name)
    changes=dict(Momsol='#Cyclic#',Tstop='300',Flmap='0 1 300',Flhis='0 1 300')
    p=out/'sim_P057.mdf';code=p.read_text()
    for k,v in changes.items():code=replace(code,k,v)
    p.write_text(code)
    save(out/'INPUTS.json',dict(parent=str(parent),changes=changes,
        parent_manifest_sha256=sha(parent/'INPUTS.json'),Delft3D_source_modified=False,
        Delft3D_runtime_modified=False,inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=240)),flush=True)


if __name__=='__main__':main()
