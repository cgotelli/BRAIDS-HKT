"""Fresh-process lifecycle check and optional isolated timestep contrast."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output');p.add_argument('--half-dt',action='store_true')
a=p.parse_args();case=Path(a.case).resolve();runtime=ROOT/'validation/_startup208/candidate'
sys.path[:0]=[str(runtime),str(ROOT)]
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
import subprocess
original=subprocess.Popen
def no_compiler(command,*args,**kw):
    exe=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
    assert not any(s in exe for s in ('gfortran','g++','gcc','clang')),exe
    return original(command,*args,**kw)
subprocess.Popen=no_compiler
from braids_core.river_worker import run
from braids_desktop.preparation import prepare
from braids_desktop.results import ResultsStore
import numpy as np
if a.half_dt:
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    f=case/'project.json'
    if not f.exists():f=case/'flume.braids.json'
    d=json.loads(f.read_text());d['settings']['dt_max_s']/=2
    f=out/'project.json';f.write_text(json.dumps(d,indent=2))
    prep=prepare(f,out/'prepared',asset_cache=out/'assets')
    status=run(prep,out/'run',checkpoint_dir=out/'checkpoints')
    (out/'CHECK.json').write_text(json.dumps(status,indent=2));print(status)
else:
    checks=sorted((case/'checkpoints').glob('*.json'),key=lambda f:json.loads(f.read_text())['time_s'])
    checkpoint=checks[len(checks)//2].with_suffix('.npz')
    status=run(case/'prepared',case/'resumed',restart=checkpoint)
    f=ResultsStore(case/'run').read(-1);r=ResultsStore(case/'resumed').read(-1)
    exact={k:bool(np.array_equal(f.arrays[k],r.arrays[k])) for k in f.arrays}
    assert all(exact.values()),exact
    (case/'RESTART.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),exact=exact,status=status,compiler_free=True),indent=2))
    print(case.name,'exact fresh restart',status['time_s'])
