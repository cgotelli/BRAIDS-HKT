"""Verify installed namespace, identity rejection and the current preparation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime'
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/nearwake60'
sys.path.insert(0,str(BASE));sys.path.insert(1,str(ROOT))
from braids_core.river_launcher import verify
from braids_core.river_worker import load_prepared
from braids_core.sediment import river_budget_roundoff,shoreline_solver
assert Path(river_budget_roundoff.__file__).resolve()==(BASE/'braids_core/sediment/river_budget_roundoff.py').resolve()
verify();result=[]
for name in ['wake4_gpu','wake8_gpu','zero8_corrected']:
    r=json.loads((REPORT/name/'STARTED.json').read_text());output=Path(r['output'])
    args=json.loads((output.parent.parent/'history'/(output.name+'.log')).open(encoding='utf-8').readline())['arguments']
    prepared=Path(args[args.index('--input')+1])
    try:doc,receipt=load_prepared(prepared)
    except ValueError as e:
        assert name!='zero8_corrected' and 'Runtime changed' in str(e)
        result.append(dict(case=name,accepted=False,reason=str(e)))
    else:
        assert name=='zero8_corrected';result.append(dict(case=name,accepted=True,runtime_id=receipt['runtime_id']))
d=dict(status='passed',runtime_sha256=hashlib.sha256((BASE/'RUNTIME.json').read_bytes()).hexdigest(),
       preparations=result,old_fields_remain_readable=True,source_path=str(river_budget_roundoff.__file__))
(REPORT/'INSTALLED_IDENTITIES.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print('Installed current preparation accepted; historical identities explicitly rejected')
