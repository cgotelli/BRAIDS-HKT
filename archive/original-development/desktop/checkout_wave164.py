"""Fresh preparation and direct maintained-source execution, not installation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_gpu.atomic_output import write_json_atomic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--formula',default='mpm',choices=['mpm','ashida_michiue_generalized','wang_fredsoe'])
    ap.add_argument('--tag',default='maintained')
    ap.add_argument('--until',type=float,default=60.)
    ap.add_argument('--restart')
    args=ap.parse_args()
    out=ROOT/f'validation/_wave164/{args.tag}/{args.formula}_{args.backend}'
    out.mkdir(parents=True,exist_ok=False)
    src=ROOT/f'validation/_hydraulics162/mobile/braids_{args.formula}_{args.backend}/project'
    doc=json.loads((src/'braids_project.json').read_text())
    dem=Path(doc['terrain']['dem_path'])
    if not dem.is_absolute():doc['terrain']['dem_path']=str((src/dem).resolve())
    write_json_atomic(out/'project.json',doc)
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    print(river_worker.run(prepared,out/'run',until=args.until,restart=args.restart),flush=True)


if __name__=='__main__':main()
