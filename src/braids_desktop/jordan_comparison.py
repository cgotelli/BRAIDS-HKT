"""Paired ambient / turbine-on comparison; separate, authenticated exports."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import csv, hashlib, io, json, uuid
import numpy as np
from shapely.geometry import Point, Polygon
from braids_core.river_turbine_jordan import METHOD, WakeParameters, evaluate
from braids_core.river_model import resolve_model
from braids_core.river_folder import _atomic_json


def make_pair(project):
    """Return two documents; retain all turbine forces in the coupled member.

    Move the same viscosity to Advanced hydraulics for the turbine-free member.
    This is necessary: simply disabling the extension would also remove mixing.
    No sediment or secondary-flow qualification is implied.
    """
    on=deepcopy(project);layout=on.get('turbines')
    if not layout or not layout['turbines']:raise ValueError('Place turbines before creating a Jordan pair.')
    if on['model']['request']['sediment']['enabled']:raise ValueError('Jordan comparison currently requires hydrodynamics only.')
    if on['model']['request']['hydrodynamics']['secondary_flow']!='secondary.off':raise ValueError('Turn secondary flow off for this non-yawed comparison.')
    if on['settings']['adaptive']:raise ValueError('Use a static mesh for the first Jordan paired comparison.')
    layout.pop('power_estimation',None)  # coupled diagnostic remains literal Thetis
    on['name']+=' - turbines on (Jordan pair)'
    off=deepcopy(on);off['name']=project['name']+' - ambient (Jordan pair)';off['turbines']=None
    h=off['model']['request']['hydrodynamics'];nu=float(layout['ambient_viscosity_m2_s'])
    if h['turbulence']!='off':nu+=off['settings']['controls']['viscosity']
    h['turbulence']='constant_eddy_viscosity' if nu>0 else 'off'
    off['settings']['controls']['viscosity']=nu
    off['model']=resolve_model(off['model']['request']).document()
    return off,on


def pair_contract(ambient,coupled):
    a,b=ambient.project,coupled.project
    if not a or not b or a.get('turbines') is not None or not b.get('turbines'):
        raise ValueError('Choose turbine-free ambient results and a turbine-on result. Coupled fields cannot be reused as ambient input.')
    if a['model']['request']['sediment']['enabled'] or b['model']['request']['sediment']['enabled']:
        raise ValueError('This comparison is hydraulic only.')
    if a['settings']['adaptive'] or b['settings']['adaptive']:
        raise ValueError('Use static meshes for this first qualified Jordan comparison.')
    for key in ['initial','roughness_parameters','material_zones','boundary_states']:
        if a.get(key)!=b.get(key):raise ValueError('Paired input differs: '+key+'. Generate both runs from the same hydraulic configuration.')
    for key in ['domain','solids','hydraulics','assignments']:
        if a['settings'][key]!=b['settings'][key]:raise ValueError('Paired input differs: '+key)
    if a['model']['request']['hydrodynamics']['roughness']!=b['model']['request']['hydrodynamics']['roughness'] or a['settings']['controls']['chezy']!=b['settings']['controls']['chezy']:
        raise ValueError('Ambient and turbine-on runs must use the same bed resistance.')
    if a['settings']['solids']:raise ValueError('Analytical obstacle-wake interaction is not qualified in this straight-reach comparison.')
    def nu(q):
        h=q['model']['request']['hydrodynamics']
        if h['secondary_flow']!='secondary.off':raise ValueError('Secondary flow is outside this comparison.')
        return (q['settings']['controls']['viscosity'] if h['turbulence']!='off' else 0.)+(q.get('turbines') or {}).get('ambient_viscosity_m2_s',0.)
    if abs(nu(a)-nu(b))>1e-12:raise ValueError('Ambient and turbine-on runs must retain the same horizontal viscosity.')
    # DEM may have been copied to a portable project; compare bytes rather than paths.
    def terrain(q,store):
        v=q['terrain'];p=Path(v['dem_path'])
        if not p.is_absolute():
            candidates=[store.output/p]+[parent/p for parent in store.output.parents if (parent/'BRAIDS_FOLDER.json').is_file()]
            found={x.resolve() for x in candidates if x.is_file()}
            if len(found)!=1:raise ValueError('Retain the portable project folder and its terrain inputs for the paired comparison.')
            p=found.pop()
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if terrain(a,ambient)!=terrain(b,coupled):raise ValueError('Paired terrain data differs.')


def compare(ambient,coupled,config,*,destination=None,progress=None,cancelled=None):
    pair_contract(ambient,coupled)
    if config.get('schema')!=METHOD:raise ValueError('Choose a Jordan analytical configuration file with calibration provenance.')
    params=WakeParameters(**config['wake_parameters']);rho=float(config['water_density_kg_m3'])
    layout=deepcopy(coupled.project['turbines']);controls=config.get('analytical_controls',{})
    if set(controls)-{r['id'] for r in layout['turbines']}:raise ValueError('Analytical control table names must match turbine identifiers in the saved run.')
    for r in layout['turbines']:
        if r['id'] in controls:r['control']=controls[r['id']]
    # The borrowed depth-mean ambient is vertically uniform. Straight connected
    # reach only: do not send a Gaussian wake through a wall or around a bend.
    from braids_core.river_turbines import frame
    n,t=frame(layout['turbines'][0]);xy=np.asarray(coupled.project['settings']['domain']['vertices_xy'])
    if abs(Polygon(xy).area-np.ptp(xy@n)*np.ptp(xy@t))>1e-7*max(1.,Polygon(xy).area):
        raise ValueError('This first Jordan workflow supports straight rectangular reaches aligned with the rotors.')
    common=sorted(set(ambient.times)&set(coupled.times))
    if not common:raise ValueError('No matching saved times. Use the same output interval in both runs.')
    root=Path(coupled.output);index_raw=(root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes();idx=json.loads(index_raw)
    data=(root/'TURBINE_PERFORMANCE.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=idx['files']['TURBINE_PERFORMANCE.json'] or (root/'TURBINE_PERFORMANCE_INDEX.json').read_bytes()!=index_raw:
        raise ValueError('Turbine diagnostics are updating or damaged. Retry after the next complete output.')
    hydraulic={(float(r['time_s']),r['turbine_id']):r for r in json.loads(data)}
    rows=[];owners={};reference={}
    for k,time in enumerate(common):
        if cancelled and cancelled():raise InterruptedError('Jordan comparison cancelled before publishing.')
        f=ambient.read(int(np.where(ambient.times==time)[0][0]))
        def sample(points):
            key=(f.epoch,np.asarray(points).tobytes())
            if key not in owners:
                ids=[ambient._owner(f,Point(p)) for p in points]
                if any(i is None for i in ids):raise ValueError('A rotor extends beyond the ambient mesh.')
                owners[key]=np.asarray(ids,int)
            ids=owners[key];h=f.q[0,ids];uv=np.divide(f.q[1:3,ids],h,out=np.zeros((2,len(ids))),where=h>1e-8)
            return h,f.z[ids],uv[0],uv[1]
        if not reference:
            first=ambient.read(0)
            for r in layout['turbines']:
                i=ambient._owner(first,Point(r['center_xy']))
                if i is None:raise ValueError('Rotor centre is outside the ambient mesh.')
                reference[r['id']]=float(first.z[i])
        out,_=evaluate(layout,sample,params,time_s=time,rho=rho,reference_bed=reference)
        for r in out:
            key=(time,r['turbine_id'])
            if key not in hydraulic:raise ValueError('A matching hydraulic turbine sample is missing.')
            h=hydraulic[key]
            if h.get('power_estimator','thetis')!='thetis':
                raise ValueError('Select stock Thetis power in the coupled member; do not compare two analytical overlays.')
            r['coupled_power_W']=float(h['useful_power_W'])*rho/1000.
            r['coupled_local_velocity_m_s']=float(h['local_velocity_m_s'])
            r['coupled_active']=h['active'];r['rho_kg_m3']=rho
            rows.append(r)
        if progress:progress((k+1)/len(common))
    totals={};last={}
    for row in rows:
        name=row['turbine_id'];totals.setdefault(name,[0.,0.])
        if name in last:
            prev=last[name];dt=row['time_s']-prev['time_s']
            for k,field in enumerate(['analytical_power_W','coupled_power_W']):totals[name][k]+=.5*dt*(prev[field]+row[field])/3600.
        row['analytical_energy_Wh'],row['coupled_energy_Wh']=totals[name]
        last[name]=row
    # Publish to a new comparison folder. Existing run diagnostics are untouched.
    json.dumps(rows,allow_nan=False)  # fail before creating any partial publication
    dest=Path(destination) if destination else root/'comparisons'/('jordan_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True,exist_ok=False)
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (dest/'POWER.csv').write_text(buf.getvalue(),encoding='utf8',newline='')
    _atomic_json(dest/'POWER.json',rows)
    _atomic_json(dest/'CONFIG.json',config)
    _atomic_json(dest/'PROVENANCE.json',dict(method=METHOD,ambient=str(ambient.output),coupled=str(coupled.output),
        ambient_project_sha256=ambient.context['project_sha256'],coupled_project_sha256=coupled.context['project_sha256'],
        ambient_frames=ambient.index['frames'],coupled_frames=coupled.index['frames'],times_s=common,
        turbine_performance_sha256=hashlib.sha256(data).hexdigest(),
        note='Analytical screening and hydraulic assessment are distinct. No analytical deficit is inserted in the SWE. Uniform vertical ambient; quasi-steady Gaussian/FLS; original FLORIS220 5x5/half-radius rotor sampling and 0.05m/s TI-overlap rule. Site calibration required. No layout optimizer.'))
    return dest,rows
