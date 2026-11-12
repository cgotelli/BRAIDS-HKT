"""Read-only solver instrumentation on private replays of the user's aaa inputs.

Uses the installed, authenticated alpha127 operators unchanged. This file is
not imported by the plugin. Additional host copies are diagnostic overhead.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
sys.dont_write_bytecode = True


def main():
    from braids_core.sediment.river_explicit import RiverExplicit
    from braids_core.sediment.bed_columns_v101 import host
    from tools import profile_test1_amr

    out = Path(sys.argv[sys.argv.index('--output')+1])
    samples, singles = [], []
    original_rate, original_single = RiverExplicit.rate, RiverExplicit._single
    next_sample = [0.]
    diagnostic_time = [0.]

    def rate(self, q, dt, at):
        try:
            return original_rate(self, q, dt, at)
        finally:
            if at + 1e-10 >= next_sample[0]:
                began = time.perf_counter()
                p = self.solver.physics
                d = self.solver.data
                f = host(self.frequency).copy()
                state = host(q).copy()
                speeds = host(self.speed).copy()
                ext = self.solver.exterior_bed
                trace, mask = host(ext.trace), host(ext.selected)
                records = []
                for i in np.argsort(f)[-5:][::-1]:
                    i = int(i)
                    faces = np.flatnonzero((d['owner']==i)|(d['neighbour']==i))
                    contributions = []
                    for j in faces:
                        j = int(j); o, n = int(d['owner'][j]), int(d['neighbour'][j])
                        s = float(speeds[j])
                        normal = d['normal'][:,j]
                        if mask[j]:
                            def wave(x):
                                h = x[0]
                                return abs(float(x[1:] @ normal))/h + np.sqrt(p.parameters.gravity*h) if h > p.parameters.dry else 0.
                            s = max(wave(state[:,o]),wave(trace[:,j]))
                        contributions.append(dict(face=j,owner=o,neighbour=n,bid=int(d['bid'][j]),
                            normal=normal.tolist(),length_m=float(d['length'][j]),speed_m_s=s,
                            contribution_s_inverse=float(d['length'][j]*s/d['area'][i]),
                            other_depth_m=float(state[0,n if o==i and n>=0 else o]),
                            other_velocity_m_s=(state[1:,n if o==i and n>=0 else o]/max(state[0,n if o==i and n>=0 else o],1e-30)).tolist(),
                            exterior_selected=bool(mask[j]),trace=trace[:,j].tolist() if mask[j] else None))
                    records.append(dict(cell=i,xy=d['xy'][:,i].tolist(),area_m2=float(d['area'][i]),
                        depth_m=float(state[0,i]),velocity_m_s=(state[1:,i]/max(state[0,i],1e-30)).tolist(),
                        frequency_s_inverse=float(f[i]),cfl_limit_s=float(self.cfl/f[i]) if f[i]>0 else None,
                        faces=contributions))
                samples.append(dict(time_s=float(at),trial_dt_s=float(dt),cells=p.n,cfl=self.cfl,
                    stage_max_cfl=float(dt*f.max()),top=records))
                if len(samples) == 1 or at >= 10 and int(at) % 10 == 0:
                    np.savez_compressed(out/f'SAMPLE_{len(samples):04d}.npz',q=state,frequency=f,xy=d['xy'],area=d['area'])
                next_sample[0] = .1 if at < .1-1e-10 else (1. if at < 1.-1e-10 else float(np.floor(at)+1.))
                diagnostic_time[0] += time.perf_counter()-began

    def single(self, q, dt, at):
        began = time.perf_counter(); error = None
        try:
            return original_single(self, q, dt, at)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            singles.append(dict(time_s=float(at),dt_s=float(dt),frequency=float(self.frequency_peak),
                                wall_s=time.perf_counter()-began,error=error))

    RiverExplicit.rate, RiverExplicit._single = rate, single
    try:
        profile_test1_amr.main()
    finally:
        if out.exists():
            (out/'LIMITERS.json').write_text(json.dumps(dict(samples=samples,singles=singles,
                diagnostic_host_copy_s=diagnostic_time[0],scope='Unchanged installed operators; sampled extra host copies; timings include instrumentation'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
