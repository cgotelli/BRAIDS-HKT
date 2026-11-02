"""Authenticated per-frame hub-plane reconstruction for maps/profiles/exports."""
import hashlib,json,math
import numpy as np
from braids_core.river_turbine_hub_field import plane_field

PREFIX='hub_velocity_m_s:'

def variables(project):
    rows=((project or {}).get('turbines') or {}).get('turbines',[])
    if not rows or any(r['control']['mode']!='fixed_rpm_surface' for r in rows):return []
    return [PREFIX+r['id'] for r in rows]

def info(key):
    if not key.startswith(PREFIX) or not key[len(PREFIX):]:raise ValueError('Unknown hub-height plane')
    return ('Hub-height speed (approx.) - '+key[len(PREFIX):], 'm/s','Reds')

def frame_samples(store,time_s):
    root=store.output;error=None
    for _ in range(2):
        try:
            raw=(root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes();index=json.loads(raw)
            if index.get('schema')!='braids.turbine-performance.v1':raise ValueError('Current turbine performance index required.')
            data=(root/'TURBINE_PERFORMANCE.json').read_bytes()
            if hashlib.sha256(data).hexdigest()!=index['files']['TURBINE_PERFORMANCE.json'] or (root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes()!=raw:
                raise ValueError('Turbine samples are updating or their checksum differs; refresh after publication.')
            all_rows=json.loads(data)
            if len(all_rows)!=index['rows']:raise ValueError('Turbine sample count differs.')
            rows=[r for r in all_rows if math.isclose(r['time_s'],time_s,rel_tol=0,abs_tol=1e-8)]
            if not rows:raise ValueError('No simultaneous turbine sample for this frame yet; refresh after publication.')
            return rows
        except (OSError,ValueError,KeyError) as exc:error=exc
    raise error

def reconstruct(store,frame,key):
    if key not in variables(store.project):raise ValueError('The loaded run does not support this hub plane.')
    rows=frame_samples(store,frame.time_s);rotors=store.project['turbines']['turbines'];name=key[len(PREFIX):]
    rotor=next(r for r in rotors if r['id']==name);sample=next(r for r in rows if r['turbine_id']==name)
    elevation=sample['rotor_top_elevation_m']-rotor['rotor_diameter_m']/2
    xy=np.array([[p.centroid.x,p.centroid.y] for p in frame.polygons])
    uv=np.column_stack((frame.values['velocity_x_m_s'],frame.values['velocity_y_m_s']))
    return plane_field(xy,frame.z,frame.q[0],uv,rotors,rows,elevation)
