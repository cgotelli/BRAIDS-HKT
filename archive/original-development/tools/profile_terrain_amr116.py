"""New private terrain-policy projects from frozen Test1 physical input.

No old project migration or installed-plugin modification. The explicit method
remains research-only. Unchanged-domain contrasts isolate caching and terrain.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--terrain',choices=('on','off'),default='on')
    ap.add_argument('--trace-cfl',action='store_true')
    ap.add_argument('--remainder-clock',action='store_true')
    extra,remaining=ap.parse_known_args()
    out=Path(remaining[remaining.index('--output')+1]).resolve()
    from braids_core import river_worker
    from braids_core.river_project import SCHEMA
    from braids_core.sediment.river_field_policy import FieldSettings
    from braids_core.sediment.river_terrain_refinement import terrain_features
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.bed_columns_v101 import host
    from braids_gpu.atomic_output import write_json_atomic,save_arrays_atomic
    prepare=river_worker.prepare
    def current(project,*args,**kwargs):
        d=json.loads(Path(project).read_text());d['schema']=SCHEMA
        d['settings']['adaptation']={**asdict(FieldSettings()),**d['settings']['adaptation'],
            'terrain_enabled':extra.terrain=='on'}
        write_json_atomic(project,d)
        result=prepare(project,*args,**kwargs)
        p=json.loads((result/'core/prepared/profile.json').read_text())
        with np.load(result/'core/prepared/initial.npz') as f:bed=f['rock']+f['thickness'].sum(0)
        t=core.hierarchy(p);start=time.perf_counter()
        features=terrain_features(t,bed,FieldSettings(**p['adaptation']),threads=p['controls']['threads'])
        write_json_atomic(out/'TERRAIN.json',dict(features['identity'],assessment_s=time.perf_counter()-start))
        save_arrays_atomic(out/'TERRAIN.npz',bed=bed,xy=np.array([c.centroid_xy for c in t.finest.cells]),
            **{k:features[k] for k in ('nodes','tag','angle_deg','detail_m')})
        return result
    river_worker.prepare=current
    from validation.hydraulics.merged_fwave115 import explicit
    if extra.remainder_clock:
        from validation.hydraulics.terrain_amr116 import explicit_clock
        explicit.Explicit=explicit_clock.Explicit;explicit.install=explicit_clock.install;explicit.METHOD=explicit_clock.METHOD
    rate=explicit.Explicit.rate;traces=[]
    if extra.trace_cfl:
        def trace(self,q,dt,at):
            try:return rate(self,q,dt,at)
            finally:
                frequency=host(self.frequency);i=int(np.argmax(frequency));a=self.solver;g=a.physics.g
                faces=np.flatnonzero((a.data['owner']==i)|(a.data['neighbour']==i))
                h=host(q[:,i]);area=host(g['area']);length=host(g['length'])
                traces.append(dict(time=at,trial_dt=dt,cell=i,frequency=float(frequency[i]),
                    bound_s=float(self.cfl/frequency[i]) if frequency[i]>0 else None,
                    area_m2=float(area[i]),xy=host(g['xy'])[:,i].tolist(),q=h.tolist(),
                    adjacent_exterior=[str(a.data['bid'][j]) for j in faces if a.data['neighbour'][j]<0],
                    face_speed=host(self.speed)[faces].tolist(),face_length=length[faces].tolist()))
        explicit.Explicit.rate=trace
    from tools import profile_merged_fwave115
    sys.argv=[str(Path(profile_merged_fwave115.__file__)),*remaining]
    try:profile_merged_fwave115.main()
    finally:
        if out.exists():
            write_json_atomic(out/'TERRAIN_EXPERIMENT.json',dict(settings=vars(extra),
                current_source_identity=river_worker.source_identity(),plugin_changed=False,
                explicit_is_research=True,cfl_traces=traces))
            ep=out/'EXPERIMENT.json'
            if ep.exists():
                d=json.loads(ep.read_text());d['production_and_plugin_changed']=False
                d['production_source_changed']=True;d['plugin_changed']=False
                d['scope']='current terrain/caching implementation with isolated integrator, not release qualification'
                write_json_atomic(ep,d)

if __name__=='__main__':main()
