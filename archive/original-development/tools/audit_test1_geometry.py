"""Read-only diagnostic for Test1's prepared polygons and AMR boundary maps."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from braids_gpu.general_live_transport import read_records

root=Path(sys.argv[1]);folders=sorted(p for p in (root/'core/prepared/cache').iterdir() if (p/'case.data').exists())
rows=[(p,read_records(p)) for p in folders];rows.sort(key=lambda r:-len(r[1]['area']))
old,new=rows[0][1],rows[-1][1]
src=np.flatnonzero(old['neighbour']<0);dst=np.flatnonzero(new['neighbour']<0)
coverage=np.zeros(old['nf']);target_coverage=np.zeros(new['nf']);examples=[]
def key(d,i):return tuple([int(d['bid'][i]),*d['normal'][:,i],*d['mid'][:,i],d['length'][i]])
print('owners',[(str(p),len(d['area'])) for p,d in rows]);print('boundary_counts',len(src),len(dst))
print('exact_keys_equal',set(key(old,i) for i in src)==set(key(new,j) for j in dst))
for j in dst:
    for i in src:
        if old['bid'][i]!=new['bid'][j] or np.dot(old['normal'][:,i],new['normal'][:,j])<1-1e-10:continue
        normal=old['normal'][:,i];shift=new['mid'][:,j]-old['mid'][:,i]
        if abs(np.dot(shift,normal))>1e-10*max(1.,old['length'][i],new['length'][j]):continue
        tangent=np.array([-normal[1],normal[0]]);center=np.dot(shift,tangent)
        lo=max(-.5*old['length'][i],center-.5*new['length'][j]);hi=min(.5*old['length'][i],center+.5*new['length'][j]);overlap=max(0.,hi-lo)
        if overlap>0:
            coverage[i]+=overlap/old['length'][i];target_coverage[j]+=overlap/new['length'][j]
            if key(old,i)!=key(new,j):examples.append(dict(old=int(i),new=int(j),overlap=overlap,old_length=old['length'][i],new_length=new['length'][j]))
print('old_bad',[(int(i),float(coverage[i])) for i in src if abs(coverage[i]-1)>1e-9][:10])
print('new_bad',[(int(j),float(target_coverage[j])) for j in dst if abs(target_coverage[j]-1)>1e-9][:10])
print('nonidentical_overlaps',len(examples),json.dumps(examples[:10]))
