"""Reject controller tampering and physical changes before continuation."""
import io,json,os,sys
from copy import deepcopy
from types import SimpleNamespace
from stage_formula190 import ROOT,TARGET,OUT
from stage_secondary185 import save
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
from braids_core import river_worker
from braids_core.sediment.river_field_policy import AutomaticRiver,FieldSettings

def main():
    folder=OUT/'restart_rejection';folder.mkdir(exist_ok=False)
    case=OUT/'runs/dry/recking_2010_cpu'
    project,receipt=river_worker.load_prepared(case/'prepared')
    identity=river_worker.checkpoint_identity(project,receipt)
    payload=river_worker.read_checkpoint(case/'cancelled/checkpoint.npz',identity)
    with np.load(io.BytesIO(payload),allow_pickle=False) as f:
        fields={key:f[key].copy() for key in f.files}
    meta=json.loads(str(fields['metadata']));checks={}
    assert meta['sediment_clock']['hint_s']>0
    for name,change,expected in (
        ('positive_hint_tamper',lambda m:m['sediment_clock'].update(hint_s=m['sediment_clock']['hint_s']*1.1),'history checksum'),
        ('negative_hint',lambda m:m['sediment_clock'].update(hint_s=-1.),'Invalid sediment'),
        ('boolean_hint',lambda m:m['sediment_clock'].update(hint_s=True),'Invalid sediment'),
        ('controller_method',lambda m:m['sediment_clock'].update(method='other'),'controller identity'),
        ('controller_source',lambda m:m['sediment_clock'].update(code_sha256='0'*64),'controller identity')):
        altered=deepcopy(meta);change(altered);arrays=dict(fields,metadata=np.array(json.dumps(altered)))
        path=folder/(name+'.npz');np.savez(path,**arrays)
        try:
            AutomaticRiver.restart(path,SimpleNamespace(count=len(fields['reference'])),None,
                                   fields['reference'],FieldSettings(**meta['settings']))
        except ValueError as exc:
            assert expected in str(exc),(name,str(exc));checks[name]=str(exc)
        else:raise AssertionError(name+' was accepted')
    # Public continuation must reject new physical parameters, even with the
    # same method and mesh. An end-time-only extension remains admitted.
    extended=deepcopy(project);extended['settings']['end_time_s']=20.
    assert river_worker.validate_continuation(project,extended)==20.
    for parameter,value in [('ReachSlope',.02),('ACal',.8)]:
        altered=deepcopy(project)
        altered['model']['physics']['sediment']['capacity']['parameters'][parameter]=value
        try:river_worker.validate_continuation(project,altered)
        except ValueError as exc:checks['changed_'+parameter]=str(exc)
        else:raise AssertionError('Changed '+parameter+' was accepted')
    bad=deepcopy(identity);bad['runtime_id']={'different':'runtime'}
    try:river_worker.read_checkpoint(case/'cancelled/checkpoint.npz',bad)
    except ValueError as exc:checks['changed_public_runtime']=str(exc)
    else:raise AssertionError('Different runtime was accepted')
    save(folder/'CHECK.json',dict(checks=checks,end_time_only_extension_allowed=True,
                               original_checkpoint_untouched=True))
    print('PASS',len(checks),'restart rejection checks')
if __name__=='__main__':main()
