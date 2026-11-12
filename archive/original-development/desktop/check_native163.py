"""Supported FLOW momentum-setting control; source and runtime are read-only."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop'))
from check_bedload161 import sha,save,replace
import runtime


def main():
    parent=ROOT/'validation/_hydraulics162/mobile/delft_mpm'
    out=ROOT/'validation/_groin163/delft_mpm_cyclic'
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in inputs['inputs'].items():
        assert sha(parent/name)==digest
        shutil.copy2(parent/name,out/name)
    path=out/'sim_P057.mdf'
    path.write_text(replace(path.read_text(),'Momsol','#Cyclic#'))
    save(out/'INPUTS.json',dict(parent=str(parent),parent_sha256=sha(parent/'INPUTS.json'),
        changes={'Momsol':'Cyclic'},source_modified=False,runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180),flush=True)


if __name__=='__main__':main()
