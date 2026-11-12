"""Compare same-mesh qualified states, retaining numerical and budget evidence."""
import argparse,json,sys
from stage_secondary185 import ROOT,OUT,save
sys.path.insert(0,str(ROOT))
import numpy as np
from braids_desktop.results import ResultsStore

def compare(a,b):
    a=ResultsStore(a).read(-1);b=ResultsStore(b).read(-1)
    assert a.time_s==b.time_s
    rows={}
    for k in ('q','z','W','mass'):
        d=b.arrays[k]-a.arrays[k]
        rows[k]=dict(max_abs=float(abs(d).max()),rms=float(np.sqrt(np.mean(d*d))))
    d=b.arrays['q'][0]-a.arrays['q'][0]
    rows['depth']=dict(max_abs_m=float(abs(d).max()),rms_m=float(np.sqrt(np.mean(d*d))),
                       peak_cell=int(np.argmax(abs(d))))
    return dict(time_s=a.time_s,fields=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v6');a=p.parse_args()
    rows=dict(backend=compare(OUT/f'signed_cpu_{a.version}/run',OUT/f'signed_gpu_{a.version}/run'),
              half_step=compare(OUT/f'signed_cpu_{a.version}/run',OUT/f'signed_half_cpu_{a.version}/run'))
    save(OUT/f'COMPARE_{a.version}.json',rows);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
