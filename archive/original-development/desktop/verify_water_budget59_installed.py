"""Verify installed routing and prepared identities without another simulation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_validation59'
sys.path.insert(0,str(BASE));sys.path.insert(1,str(ROOT))
from braids_core.river_launcher import verify
from braids_core.river_worker import load_prepared
from braids_core.sediment import shoreline_solver
assert Path(shoreline_solver.__file__).resolve()==(BASE/'braids_core/sediment/shoreline_solver.py').resolve()
verify();records=[]
for name in ['nash_cpu','nash_gpu','zero_cpu','zero_gpu']:
    r=json.loads((REPORT/name/('STARTED.json' if name.endswith('cpu') else 'RECEIPT.json')).read_text())
    output=Path(r['output']);history=output.parent.parent/'history'/(output.name+'.log')
    args=json.loads(history.open(encoding='utf-8').readline())['arguments'];prepared=Path(args[args.index('--input')+1])
    doc,receipt=load_prepared(prepared)
    records.append(dict(case=name,prepared=str(prepared),installed_runtime_accepts=True,runtime_id=receipt['runtime_id']))
old=json.loads((REPORT/'fixed_floor_trials/nash_cpu/STARTED.json').read_text());output=Path(old['output'])
args=json.loads((output.parent.parent/'history'/(output.name+'.log')).open(encoding='utf-8').readline())['arguments']
try:load_prepared(args[args.index('--input')+1])
except ValueError as e:
    assert 'Runtime changed' in str(e);rejection=str(e)
else:raise AssertionError('Old prepared runtime must be rejected')
result=dict(status='passed',runtime_sha256=hashlib.sha256((BASE/'RUNTIME.json').read_bytes()).hexdigest(),
    current_projects=records,old_preparation_rejected=rejection,arithmetic_source=str(shoreline_solver.__file__))
(REPORT/'INSTALLED_IDENTITIES.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('Installed runtime accepts all four current preparations and rejects old runtime identity',flush=True)
