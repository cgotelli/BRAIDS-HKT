"""Public semi-implicit Test1 mesh-policy timing; no numerical substitution."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_gpu.atomic_output import write_json_atomic
from tools.prepare_mesh_modes117 import current

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--mesh-mode',choices=('dynamic','static'),required=True)
    args,left=p.parse_known_args()
    original=river_worker.prepare
    def prepare(project,*a,**kw):
        write_json_atomic(project,current(json.loads(Path(project).read_text()),args.mesh_mode))
        return original(project,*a,**kw)
    river_worker.prepare=prepare
    from tools import profile_test1_amr
    sys.argv=[profile_test1_amr.__file__,*left,'--checkout']
    profile_test1_amr.main()

if __name__=='__main__':main()
