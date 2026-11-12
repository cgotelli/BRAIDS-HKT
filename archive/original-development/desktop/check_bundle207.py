"""Fresh-process checkpoint continuation and field equality for staged Desktop."""
import argparse,hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--case',required=True)
    a=p.parse_args();runtime=Path(a.runtime).resolve();case=Path(a.case).resolve();sys.path[:0]=[str(runtime),str(ROOT)]
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.river_worker import run
    from braids_desktop.results import ResultsStore
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
    original=subprocess.Popen
    def no_compiler(command,*args,**kw):
        exe=str(command[0] if isinstance(command,(list,tuple)) else command).lower()
        assert not any(s in exe for s in ('gfortran','g++','gcc','clang')),exe
        return original(command,*args,**kw)
    subprocess.Popen=no_compiler
    checks=sorted((case/'checkpoints').glob('*.json'),key=lambda p:json.loads(p.read_text())['time_s'])
    middle=checks[len(checks)//2].with_suffix('.npz')
    result=run(case/'prepared',case/'resumed',restart=middle)
    full=ResultsStore(case/'run').read(-1);resumed=ResultsStore(case/'resumed').read(-1)
    exact={k:bool(np.array_equal(full.arrays[k],resumed.arrays[k])) for k in full.arrays}
    assert all(exact.values()),exact
    diagnostics=json.loads((case/'run/DIAGNOSTICS.json').read_text())
    budget={k:max(abs(r[k]) for r in diagnostics) for k in ('solid_residual_m3','water_plus_pore_residual_m3')}
    assert max(budget.values())<1e-10,budget
    for field in ('mobile_bedload_concentration','secondary_intensity_m_s','bedload_capacity_m2_s','bedload_m2_s'):
        assert np.isfinite(full.values[field]).all(),field
    store=ResultsStore(case/'run');flows={}
    if 'inlet' in store.boundary_ids:
        water=store.boundary_flow('inlet',magnitude=False).values
        solid=store.boundary_flow('inlet',quantity='bedload',magnitude=False).values/2650.
        np.testing.assert_allclose(water+solid,.024,rtol=1e-10,atol=1e-14)
        np.testing.assert_allclose(solid/water,.001,rtol=1e-10,atol=1e-14)
        flows=dict(water_m3_s=water.tolist(),solid_m3_s=solid.tolist(),mixture_m3_s=(water+solid).tolist())
    record=dict(runtime=hashlib.sha256((runtime/'RUNTIME.json').read_bytes()).hexdigest(),checkpoint=str(middle),
        restart=result,exact=exact,budgets=budget,boundary_flows=flows,compiler_free=True)
    (case/'RESTART.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(case.name,'exact restart',budget,flush=True)
if __name__=='__main__':main()
