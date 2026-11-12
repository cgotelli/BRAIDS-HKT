"""Inspect the saved failing faces without running hydraulic timesteps."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_amr150'
sys.path.insert(0,str(ROOT/('validation/_amr150/candidate' if len(sys.argv)>1 else 'desktop/runtime')))
from braids_core.sediment.river_amr_transfer import boundary_map
with np.load(Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'baseline/FAILED_FACES.npz') as f:
    old={k[4:]:f[k] for k in f.files if k.startswith('old_')}
    new={k[4:]:f[k] for k in f.files if k.startswith('new_')}
    old['nf']=int(old['nf']);new['nf']=int(new['nf'])
saved={}
def trace(frame,event,arg):
    if frame.f_code is boundary_map.__code__:
        if event=='exception':saved.update(frame.f_locals)
        return trace
sys.settrace(trace)
try:boundary_map(old,new)
except ValueError:pass
finally:sys.settrace(None)
def detail(a,i):
    return dict(face=int(i),bid=int(a['bid'][i]),mid=a['mid'][:,i].tolist(),normal=a['normal'][:,i].tolist(),length=float(a['length'][i]))
report={}
for label,a,b,cov in [('source',old,new,saved['coverage']),('target',new,old,saved['target_coverage'])]:
    ids=np.flatnonzero((a['neighbour']<0)&(abs(cov-1)>1e-9));rows=[]
    for i in ids:
        near=np.flatnonzero((b['neighbour']<0)&(b['bid']==a['bid'][i]))
        dist=np.linalg.norm(b['mid'][:,near]-a['mid'][:,i,None],axis=0)
        row=dict(**detail(a,i),coverage=float(cov[i]),near=[])
        for j in near[np.argsort(dist)[:5]]:
            shift=b['mid'][:,j]-a['mid'][:,i];n=a['normal'][:,i];t=np.array([-n[1],n[0]]);center=shift@t
            overlap=max(0.,min(.5*a['length'][i],center+.5*b['length'][j])-max(-.5*a['length'][i],center-.5*b['length'][j]))
            row['near'].append(dict(**detail(b,j),line_distance=float(abs(shift@n)),normal_dot=float(n@b['normal'][:,j]),overlap=float(overlap)))
        rows.append(row)
    report[label]=rows
(OUT/('SWEEP_DIAGNOSIS.json' if len(sys.argv)>1 else 'DIAGNOSIS.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
