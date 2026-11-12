"""Preserved Test1 input, explicit backend clone and short/full regression run."""
from pathlib import Path
import sys,argparse,json
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from braids_core.river_project import load_project,validate_project
from braids_core.river_model import resolve_model
from braids_core.river_worker import prepare,run
from braids_gpu.atomic_output import write_json_atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','gpu'],required=True);p.add_argument('--until',type=float);a=p.parse_args()
    doc=deepcopy(load_project(a.source));doc['model']['request']['execution']['backend']=a.backend
    doc['model']=resolve_model(doc['model']['request']).document();doc=validate_project(doc)
    target=a.output.resolve();target.mkdir(parents=True,exist_ok=False)
    write_json_atomic(target/'PROJECT.json',doc)
    prepare(target/'PROJECT.json',target/'prepared')
    result=run(target/'prepared',target/'run',until=a.until)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
