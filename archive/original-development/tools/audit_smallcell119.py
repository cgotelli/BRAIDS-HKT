"""Read-only saved-run audit and bounded private midpoint trace for the 1 m case.

No installed source, user input, reference result or hydraulic operator changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'validation/_smallcell119'
PREPARED = Path('C:/Users/STREEM/Documents/BRAIDS/runs/prepare_20261022_114735_df4b84')
RUN = Path('C:/Users/STREEM/Documents/BRAIDS/runs/run_20261022_114753_046840')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def saved():
    from braids_gpu.general_live_transport import read_records
    cache = next((PREPARED/'core/prepared/cache').glob('*/case.data')).parent
    d = read_records(cache)
    idx = json.loads((RUN/'frames/INDEX.json').read_text())
    with np.load(RUN/'frames'/idx['frames'][-1]['epoch']) as f:
        geom = {k:f[k] for k in f.files}
    with np.load(RUN/'frames'/idx['frames'][-1]['file']) as f:
        state = {k:f[k] for k in ('q','z','water')}
    area=geom['area'];nominal=16./4.**geom['levels'];fraction=area/nominal
    owner=d['owner'];neighbor=d['neighbour'];bid=d['bid'];xy=d['xy']
    def stats(cells):
        cells=np.unique(cells);low=cells[np.argmin(area[cells])]
        return dict(cells=len(cells),minimum_area_m2=float(area[low]),minimum_cell=int(low),
            minimum_fraction=float(fraction[cells].min()),below_half=int((fraction[cells]<.5).sum()),
            wet_cells=int((state['q'][0,cells]>1e-6).sum()),max_depth_m=float(state['q'][0,cells].max()))
    boundaries={}
    for b in sorted(set(bid[bid>=0])):
        faces=np.where((bid==b)&(neighbor<0))[0];cells=owner[faces]
        boundaries[str(int(b))]=dict(**stats(cells),faces=faces.tolist(),owners=cells.tolist(),
            net_inward_volume_m3=float(np.sum(state['water'][faces])),
            average_inward_discharge_m3_s=float(np.sum(state['water'][faces]))/idx['frames'][-1]['time_s'])
    record=dict(saved_status=json.loads((RUN/'STATUS.json').read_text()),
        project_sha256=hashlib.sha256(Path('C:/Users/STREEM/Documents/BRAIDS/1m.json').read_bytes()).hexdigest(),
        prepared=str(PREPARED),run=str(RUN),cells=len(area),nominal_sizes_m=sorted(set(np.sqrt(nominal))),
        all_cells=stats(np.arange(len(area))),boundaries=boundaries,
        wet_cells=int((state['q'][0]>1e-6).sum()),water_volume_m3=float(area@state['q'][0]),
        data_keys=list(d),xy_shape=list(xy.shape))
    write(BASE/'SAVED_AUDIT.json',record)
    np.savez(BASE/'GEOMETRY.npz',area=area,fraction=fraction,nominal=nominal,xy=xy,
        owner=owner,neighbour=neighbor,bid=bid,length=d['length'],q=state['q'],z=state['z'],
        offsets=geom['offsets'],vertices=geom['vertices'])
    print(json.dumps({k:v for k,v in record.items() if k!='data_keys'},indent=2))

def trace(backend,until,preflight=False,label=None):
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.bed_columns_v101 import host
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    out=BASE/(backend+'_'+(label or ('preflight' if preflight else 'baseline')))
    records=[];advanced=[];original=RiverMidpoint._solve_single;advance=RiverMidpoint.advance
    def measured(self,base,dt,at):
        began=time.perf_counter();row=dict(time=float(at),dt=float(dt))
        try:
            result=original(self,base,dt,at)
            row.update(ok=True,iterations=result[2]['nonlinear_iterations'])
            return result
        except Exception as exc:
            row.update(ok=False,error=str(exc));tb=exc.__traceback__;local=None
            while tb:
                if tb.tb_frame.f_code.co_name=='_solve_single':local=tb.tb_frame.f_locals
                tb=tb.tb_next
            if local is not None and 'defect' in local:
                defect=host(local['defect']);indices=np.argsort(np.max(abs(defect),axis=0))[-5:][::-1]
                d=self.solver.data;p=self.solver.physics;area=host(p.g['area']);xy=host(p.g['xy'])
                mid=host(local['mid']);z=host(self.solver.z)
                row['iteration']=int(local.get('iteration',-1));row['cells']=[]
                for i in indices:
                    faces=np.where((d['owner']==i)|(d['neighbour']==i))[0]
                    exterior=faces[d['neighbour'][faces]<0]
                    row['cells'].append(dict(cell=int(i),xy=xy[:,i].tolist(),area_m2=float(area[i]),
                        depth_m=float(mid[0,i]),bed_m=float(z[i]),defect=defect[:,i].tolist(),
                        boundary_ids=d['bid'][exterior].tolist()))
            raise
        finally:
            row['wall_s']=time.perf_counter()-began;records.append(row)
    def advance_trace(self,q,dt,at):
        before=len(records);start=time.perf_counter();ok=False
        try:
            hint=self.step_hint
            # Research-only fail-fast guard: the old route deterministically
            # rejects this same interval after spending up to 128 valid solves.
            if preflight and hint is not None:
                depth=max(0,int(np.ceil(np.log2(dt/hint)-1e-10)))
                if depth>7:raise BendTrialRejected('Implicit hydraulic subdivision limit (known hint preflight)')
            result=advance(self,q,dt,at);ok=True;return result
        finally:
            advanced.append(dict(time=float(at),dt=float(dt),accepted=ok,
                calls=len(records)-before,wall_s=time.perf_counter()-start))
    RiverMidpoint._solve_single=measured;RiverMidpoint.advance=advance_trace
    from tools import profile_test1_amr
    sys.argv=[profile_test1_amr.__file__,'--source',str(PREPARED),'--output',str(out),
        '--backend',backend,'--until',str(until),'--checkout']
    try:profile_test1_amr.main()
    finally:
        write(out/'TRACE.json',dict(singles=records,outer=advanced,preflight=preflight,
            note='Process-local instrumentation; preflight changes rejection timing only when explicitly enabled.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('saved','cpu','gpu'))
    ap.add_argument('--until',type=float,default=.75);ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--label');args=ap.parse_args()
    if args.action=='saved':saved()
    else:trace(args.action,args.until,args.preflight,args.label)
