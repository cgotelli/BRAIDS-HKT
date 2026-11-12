"""Evaluate saved inlet states with unchanged installed CPU/GPU operators.

No time evolution or numerical correction is installed. The wall-pressure
comparison is an explicitly labelled algebraic diagnostic, not a new solver.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime')
sys.path.insert(0,str(RUNTIME));sys.dont_write_bytecode=True


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    args=ap.parse_args();out=ROOT/'validation/_aaa128'/('boundary_'+args.backend)
    out.mkdir(parents=True,exist_ok=False)
    from braids_core.river_launcher import verify
    verify()
    from braids_core.sediment.portable_river_runtime import install
    install(read(RUNTIME/'PRECOMPILED.json'),RUNTIME)
    from braids_core import river_worker
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.bed_columns_v101 import host
    from braids_core.river_boundary import carrier_schedules
    from braids_core.river_model import resolve_model
    prepared=ROOT/'validation/_aaa128/gpu_diagnostic/prepared'
    project,receipt=river_worker.load_prepared(prepared)
    p,initial,t=core.load_prepared(prepared/'core/prepared')
    req=project['model']['request'];req['execution']['backend']=args.backend
    resolved=resolve_model(req).document()
    params=dict(slope=project['slope_parameters'],bank_survey=read(prepared/'BANK_SURVEY.json'),
        boundaries=dict(hydraulics=project['settings']['hydraulics'],states=project['boundary_states'],resolved=project['boundary_resolution']),
        sediment_schedules=carrier_schedules(project['sediment_schedules'],project['settings']['sediment_boundaries'],project['settings']['columns']),
        transport_boundaries=project['settings']['sediment_boundaries'])
    for key in ('hiding','roughness','skin_shear','wet_avalanche','van_rijn_reference','dependent_layers'):
        params[key]=project[('dependent_layer' if key=='dependent_layers' else key)+'_parameters']
    cache=core.cache_for(prepared/'core/prepared/cache',p,initial,t)
    make=core.factory(args.backend,p,t,cache,out/'runtime',resolved_model=resolved,process_parameters=params)
    leaves=core.initial_leaves(p,initial,t,make.required_leaves)
    a=make(leaves,t.select(leaves));d=a.data;phys=a.physics;m=a.solver.midpoint
    frames=Path('C:/Users/STREEM/Desktop/aaa/runs/run_20261022_204331_367153/frames')
    result=[]
    for row in read(frames/'INDEX.json')['frames']:
        if row['time_s'] not in (10.,30.,100.):continue
        saved=dict(np.load(frames/row['file']));q=a.xp.asarray(saved['q']);a.z[:]=a.xp.asarray(saved['z'])
        m.interval_base=q;m.minimum_theta=1.;m.frequency_peak=0.
        rate,water=m.rate(q,1e-7,row['time_s'])
        ext=a.exterior_bed;trace=host(ext.trace);flux=host(ext.flux);rate=host(rate)
        exbed=host(phys.exterior_bed);z=host(a.z);qh=host(q)
        rs=[]
        for i in (983,988,989):
            faces=np.flatnonzero((d['owner']==i)&(d['neighbour']<0)&(d['bid']==0))
            assert len(faces)==1
            j=int(faces[0]);normal=d['normal'][:,j];h=float(qh[0,i])
            wall=.5*phys.parameters.gravity*h*h*normal
            delta=-d['length'][j]/d['area'][i]*(wall-flux[1:,j])
            qdrag=q.copy();phys.friction(qdrag,.001)
            rs.append(dict(cell=i,face=j,xy=d['xy'][:,i].tolist(),h_m=h,z_m=float(z[i]),exterior_bed_m=float(exbed[j]),
                velocity_m_s=(qh[1:,i]/h).tolist(),trace=trace[:,j].tolist(),boundary_flux=flux[:,j].tolist(),
                spatial_rate=rate[:,i].tolist(),drag_momentum_rate=((host(qdrag)[1:,i]-qh[1:,i])/.001).tolist(),
                closed_face_hydrostatic_momentum_flux=wall.tolist(),
                hypothetical_closed_face_rate_delta=delta.tolist(),
                note='Wall-pressure delta is a diagnostic only; it is relevant to zero-water dry exterior portions, not a replacement for active inflow.'))
        result.append(dict(time_s=row['time_s'],cells=rs,
            inward_discharge_m3_s=-float(np.sum(flux[0,d['bid']==0]*d['length'][d['bid']==0]))))
    (out/'BOUNDARY.json').write_text(json.dumps(dict(backend=args.backend,runtime=river_worker.source_identity(),
        source_prepared_receipt=receipt,samples=result),indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
