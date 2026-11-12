"""Short authenticated restart check of the staged high-TI source."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RT=ROOT/'desktop/qualification/performance66/candidate';HOME=ROOT/'desktop/qualification/performance66/runs'
sys.path[:0]=[str(RT),str(ROOT)];sys.dont_write_bytecode=True
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core.sediment import portable_river_runtime
from braids_core import river_worker
from braids_desktop import preparation
from braids_desktop.results import ResultsStore
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main(backend):
    portable_river_runtime.install(read(RT/'PRECOMPILED.json'),RT)
    folder=HOME/f'restart_{backend}';folder.mkdir(exist_ok=False)
    doc=read(HOME/f'6D_{backend}_nu1/input.json');doc['settings'].update(end_time_s=1.,dt_max_s=.125,output_interval_s=.25,checkpoint_interval_s=.5)
    (folder/'input.json').write_text(json.dumps(doc),encoding='utf-8');prepared=preparation.prepare(folder/'input.json',folder/'prepared',asset_cache=HOME/'assets')
    river_worker.run(prepared,folder/'full');river_worker.run(prepared,folder/'part',until=.5)
    river_worker.run(prepared,folder/'restart',restart=folder/'part/checkpoint.npz')
    a=ResultsStore(folder/'full').read(-1);b=ResultsStore(folder/'restart').read(-1)
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(a.z,b.z)
    assert read(folder/'full/TURBINE_PERFORMANCE.json')[-2:]==read(folder/'restart/TURBINE_PERFORMANCE.json')[-2:]
    (folder/'RESTART_PASS.json').write_text(json.dumps(dict(backend=backend,split=.5,end=1.,exact_q_z_and_final_turbines=True),indent=2))
    print('Exact restart passed',backend)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['cpu','gpu']);main(p.parse_args().backend)
