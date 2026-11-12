"""One-state comparison with the frozen source-consistent coupled reference."""
import subprocess,sys,json
from coupled_wake76 import ROOT,TARGET,REPORT,read,save
import numpy as np

def sample(runtime,test_module):
    script=f'''
import sys,json
sys.path[:0]=[{str(runtime)!r},{str(ROOT)!r}]
from {test_module} import fixture
import numpy as np
a,q,z,g=fixture();xy=a.rotors[0].xy
q[1]=q[0]*(.8+.02*(xy[:,1]-2)**2+.01*xy[:,0])
rate=np.zeros_like(q);rows=a.apply(q,z,g,rate)
keys=['force_per_density_m4_s2','useful_power_per_density_m5_s3','rotor_mean_velocity_m_s','resolved_disk_minus_source_m_s']
print(json.dumps(dict(rows=[[r[k] for k in keys] for r in rows],rate=rate.tolist())))
'''
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',script],text=True,cwd=ROOT))

old=sample(ROOT/'desktop/qualification/projection71/candidate','tests.test_staggered69')
new=sample(TARGET,'tests.test_coupled_wake76')
gaps={key:float(np.max(abs(np.array(old[key])-new[key]))) for key in old}
assert all(v<1e-12 for v in gaps.values()),gaps
base=read(ROOT/'desktop/runtime/RUNTIME.json');candidate=read(TARGET/'RUNTIME.json')
unchanged=['braids_core/sediment/river_turbine_flow.py','braids_core/sediment/river_turbine_source.py']
unchanged=[k for k in unchanged if k in candidate['files']]
assert unchanged and all(candidate['files'][k]==base['files'][k] for k in unchanged)
save(REPORT/'REFERENCE.json',dict(passed=True,reference='projection71/candidate',max_absolute_differences=gaps,unchanged_source_files=unchanged,scope='Identical coupled source/moments on a nonuniform shared state; no new experimental accuracy claim.'))
print(gaps)
