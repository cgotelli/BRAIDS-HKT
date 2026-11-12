"""Enable stock FLOW momentum output in a new deck; never edit engine sources."""
import json
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop'),str(ROOT/'validation/sediment/delft3d_current')]
from check_bedload161 import sha,save,replace
import runtime


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fine',action='store_true');args=ap.parse_args()
    parent=ROOT/'validation/_hydraulics162/mobile/delft_mpm'
    out=ROOT/('validation/_corner167/native_momentum_fine' if args.fine else 'validation/_corner167/native_momentum')
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest
        shutil.copy2(parent/name,out/name)
    changes=dict(MomTrm='#Y#')
    if args.fine:changes['Flmap']='39.9 0.01 40.1'
    path=out/'sim_P057.mdf';code=path.read_text()
    for key,value in changes.items():code=replace(code,key,value)
    path.write_text(code)
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=changes,Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    result=runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
