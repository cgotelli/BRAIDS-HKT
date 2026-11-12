"""Native-cell paired differences; no image smoothing or hidden interpolation."""
import sys,json
from stage_step190 import ROOT,TARGET
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
from braids_desktop.results import ResultsStore
from stage_secondary185 import save
BASE=ROOT/'validation/_step190'
def main():
    rows={}
    for law in ('mpm','ashida_michiue_generalized','wang_fredsoe','smart_jaggi'):
        for backend in ('cpu','gpu'):
            label=law+'_'+backend;c=BASE/'runs/candidate'/label
            if not (c/'CHECK.json').exists():continue
            b=(BASE/'runs/baseline'/label if law=='wang_fredsoe' else ROOT/'validation/_bend189/dry/groin'/label)
            if not (b/'CHECK.json').exists():continue
            old,new=ResultsStore(b/'full').read(-1),ResultsStore(c/'full').read(-1)
            for k in ('vertices','offsets','levels'):np.testing.assert_array_equal(old.mesh[k],new.mesh[k])
            diff=new.z-old.z;dh=new.q[0]-old.q[0];area=new.arrays['area']
            bs,cs=[json.loads((p/'CHECK.json').read_text()) for p in (b,c)]
            rows[label]=dict(bed_max_mm=float(abs(diff).max()*1000),bed_rms_mm=float(np.sqrt(np.average(diff*diff,weights=area))*1000),
                depth_max_mm=float(abs(dh).max()*1000),old_rejected=len(bs['status']['rejected_trials']),
                new_rejected=len(cs['status']['rejected_trials']),old_steps=bs['status']['accepted_steps'],new_steps=cs['status']['accepted_steps'],
                old_wall_s=bs['status']['wall_s'],new_wall_s=cs['status']['wall_s'],
                water_residual_m3=cs['water_residual_m3'],sediment_residual_kg=cs['sediment_residual_kg'])
            print(label,rows[label],flush=True)
    save(BASE/'COMPARISON.json',rows)
if __name__=='__main__':main()
