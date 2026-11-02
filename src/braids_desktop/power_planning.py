"""Authenticated, explicitly separate power estimates from ambient run frames."""
import csv,hashlib,io,json
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from braids_core.river_folder import _atomic_json
from braids_core.river_turbine_power_plan import validate,evaluate,METHOD


def calculate(store,progress=None,cancelled=None):
    project=store.project
    if not project or not project.get('power_plan') or project['turbines'] is not None or store.context.get('turbines') is not None:
        raise ValueError('Open an authenticated power-planning run with turbine-free hydraulics. Disturbed coupled turbine results cannot be reused as ambient flow.')
    validate(project['power_plan'],project);times=store.times.copy()
    if not len(times):raise ValueError('Wait for the first saved ambient-flow frame.')
    records=[];reference=None;owner_cache={}
    for k,time in enumerate(times):
        if cancelled and cancelled():raise InterruptedError('Power calculation cancelled; previous published results retained.')
        f=store.read(k)
        def sample(xy):
            key=(f.epoch,np.asarray(xy).tobytes())
            if key not in owner_cache:
                ids=[store._owner(f,Point(p)) for p in xy]
                if any(i is None for i in ids):raise ValueError('A rotor extends outside the ambient mesh. Move it inside the domain.')
                owner_cache[key]=np.array(ids,int)
            ids=owner_cache[key];h=f.q[0,ids];u=np.divide(f.q[1,ids],h,out=np.zeros_like(h),where=h>1e-8);v=np.divide(f.q[2,ids],h,out=np.zeros_like(h),where=h>1e-8)
            return h,f.z[ids],u,v
        if reference is None:
            reference={r['id']:float(sample([r['center_xy']])[1][0]) for r in project['power_plan']['layout']['turbines']}
        records.extend(evaluate(project,sample,time,reference))
        if progress:progress(100*(k+1)/len(times),f'Power estimates at {time:g} s')
    # Separate names prevent any consumer mistaking this for accepted actuator
    # diagnostics. Publication is atomic; an old index never authenticates new data.
    root=Path(store.output);_atomic_json(root/'POWER_PLANNING.json',records)
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    data=buf.getvalue().encode('utf-8');tmp=root/'POWER_PLANNING.csv.tmp';tmp.write_bytes(data);tmp.replace(root/'POWER_PLANNING.csv')
    index=dict(schema=METHOD,project_sha256=store.context['project_sha256'],frames=store.index['frames'][:len(times)],rows=len(records),
        files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['POWER_PLANNING.json','POWER_PLANNING.csv']},
        assumptions='Turbine-free ambient SWE; uniform vertical ambient profile; quasi-steady Gaussian deficit; finite rectangular section; linear overlap; no force feedback; no electrical prediction without efficiency')
    _atomic_json(root/'POWER_PLANNING_INDEX.json',index)
    return records


def read(store):
    root=Path(store.output)
    try:raw=(root/'POWER_PLANNING_INDEX.json').read_bytes()
    except FileNotFoundError:raise ValueError('Click Calculate / refresh power estimates to evaluate this turbine-free run.')
    index=json.loads(raw)
    if index.get('schema')!=METHOD or index.get('project_sha256')!=store.context.get('project_sha256') or index.get('frames')!=store.index['frames'][:len(index.get('frames',[]))]:
        raise ValueError('Power estimate provenance differs from this run. Recalculate from authenticated ambient frames.')
    if set(index.get('files',{}))!={'POWER_PLANNING.json','POWER_PLANNING.csv'}:raise ValueError('Invalid power estimate index.')
    blobs={name:(root/name).read_bytes() for name in index['files']}
    if any(hashlib.sha256(blobs[name]).hexdigest()!=h for name,h in index['files'].items()) or (root/'POWER_PLANNING_INDEX.json').read_bytes()!=raw:raise ValueError('Power estimates are updating or damaged; refresh after publication.')
    rows=json.loads(blobs['POWER_PLANNING.json'])
    if len(rows)!=index['rows'] or not rows:raise ValueError('No complete power samples.')
    return rows,blobs['POWER_PLANNING.csv']
