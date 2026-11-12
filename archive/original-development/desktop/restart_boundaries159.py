"""Current boundary identities and sampled forcing survive selected restart."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('BRAIDS_AUDIT_RUNTIME',str(ROOT/'desktop/runtime')))
sys.path[:0]=[str(RUNTIME),str(ROOT)];sys.dont_write_bytecode=True
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.sediment.adaptive_river_worker import read_live_frame
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    folder=args.root/'project';prepared=folder/'prepared/base';full=folder/'runs/audit'
    end=read_live_frame(full)[0]['time_s'];split=round(end/2)
    part=folder/'runs/restart_first';resume=folder/'runs/restart_resumed'
    assert river_worker.run(prepared,part,until=split)['status']=='segment_complete'
    assert river_worker.run(prepared,resume,restart=part/'checkpoint.npz')['status']=='complete'
    a=read_live_frame(full)[2];b=read_live_frame(resume)[2]
    differences={k:float(np.max(abs(a[k]-b[k]))) if a[k].size else 0. for k in a if a[k].dtype.kind in 'fiu'}
    assert max(differences.values())<1e-8,differences
    result=dict(split_time_s=split,end_time_s=end,maximum_absolute_difference=differences)
    (args.root/'RESTART.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
