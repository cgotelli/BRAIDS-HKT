"""One named boundary contract for public projects, QGIS and CPU/CUDA owners."""
from copy import deepcopy
import math
import numpy as np

# label, role, active fields, time-table fields. Order is also the editor order.
HYDRAULICS={
    'inlet.total_discharge':('Total discharge; free section stage','inlet',('discharge_m3_s',),('discharge_m3_s',)),
    'inlet.stage_discharge':('Discharge; water-elevation reference','inlet',('discharge_m3_s','stage_m'),('discharge_m3_s','stage_m')),
    'inlet.depth_discharge':('Discharge; water-depth reference (fixed datum)','inlet',('discharge_m3_s','depth_m','datum_m'),('discharge_m3_s','depth_m')),
    'inlet.critical_discharge':('Discharge; critical-depth reference','inlet',('discharge_m3_s',),('discharge_m3_s',)),
    'inlet.normal_discharge':('Discharge; normal-depth reference (automatic mesh slope)','inlet',('discharge_m3_s','energy_slope'),('discharge_m3_s',)),
    'outlet.stage':('Water-surface elevation','outlet',('stage_m',),('stage_m',)),
    'outlet.normal_depth':('Normal depth (manual mean bed slope)','outlet',('bed_slope',),()),
    'outlet.rating_curve':('Q–H rating curve','outlet',('rating_curve',),()),
    'outlet.critical':('Free outflow (critical control / supercritical export)','outlet',(),()),
    'outlet.supercritical':('Supercritical extrapolation','outlet',(),()),
    'outlet.free_weir':('Free undrowned weir (structure; no reverse flow)','outlet',('crest_m','coefficient_m_half_s','effective_width_m'),()),
}
# Retain original operators for frozen comparisons and component verification,
# not as aliases or compatibility execution routes for public projects.
RETIRED_PUBLIC_HYDRAULICS = {
    'inlet.depth_discharge': 'Use inlet.stage_discharge (water-elevation reference); explicitly enter stage = fixed datum + reference depth. No automatic conversion is performed.',
    'outlet.supercritical': 'Use outlet.critical (free outflow); it already extrapolates outward supercritical flow and also handles subcritical flow. No automatic conversion is performed.',
}
PUBLIC_HYDRAULICS = {key: value for key, value in HYDRAULICS.items() if key not in RETIRED_PUBLIC_HYDRAULICS}


def require_public_hydraulic(method):
    if method in RETIRED_PUBLIC_HYDRAULICS:
        raise ValueError('Retired duplicate boundary method '+method+'. '+RETIRED_PUBLIC_HYDRAULICS[method])
    if method not in PUBLIC_HYDRAULICS:
        raise ValueError('Unsupported public hydraulic method: '+str(method))

PARAMETERS={
    'discharge_m3_s':('Inward discharge (m³/s)','0'),
    'stage_m':('Water-surface elevation (m, DEM datum)','0'),
    'depth_m':('Depth above fixed datum (m)','0.1'),
    'datum_m':('Fixed depth datum (m, DEM datum)','0'),
    'energy_slope':('Normal-flow slope source','mesh'),
    'bed_slope':('Mean downstream bed slope (m/m)','0.001'),
    'rating_curve':('Discharge–elevation rating curve','[["", ""], ["", ""]]'),
    'crest_m':('Weir crest (m, DEM datum)','0'),
    'coefficient_m_half_s':('Weir coefficient C (m½/s; includes gravity factors)','1.7'),
    'effective_width_m':('Effective weir width (m)','1'),
    'elevation_m':('Exterior bed elevation (m, DEM datum)','0'),
    'elevation_rate_m_s':('Upward-positive bed rate (m/morphological s)','0'),
}
BED_METHODS={'bed.free':'Free bed increment','bed.fixed':'Fixed exterior bed',
    'bed.elevation':'Prescribed exterior elevation','bed.elevation_rate':'Prescribed elevation rate'}
COMPOSITION_METHODS={'composition.free':'Free column from adjacent cell','composition.fixed':'Fixed exterior column',
    'composition.mass_fraction':'Prescribed mass fractions','composition.reference_volume_fraction':'Prescribed reference-volume fractions'}
STAGE_POLICIES={'stage.outflow_only':'Outflow only; close inward portions',
    'stage.subcritical_backflow':'Permit subcritical backflow; zero incoming tangent'}
INITIALS={'initial.prepared_bed':'Prepared bed at the physical boundary face','initial.elevation':'Specified exterior elevation'}
APPLICABILITY={
    'inlet.total_discharge':'Requires a wet section for nonzero discharge; use a prescribed stage/depth inlet for dry-start inundation.',
    'inlet.stage_discharge':'Net inward discharge has priority and is imposed exactly. Elevation defines a common exterior reference water line, not a simultaneous strong stage condition. Boundary Riemann solutions admit shocks, rarefactions and backwater; actual face elevations may differ from the reference and from each other. Dry subcritical-reference admission is sonic. Local outward portions remain visible in the signed flux budget; the total remains the specified inward discharge. Incoming tangent is zero.',
    'inlet.depth_discharge':'Depth above the explicit fixed datum (not the evolving bed) defines a common exterior reference elevation. Net inward discharge has priority and is exact; actual depth is not simultaneously clamped. Boundary Riemann solutions handle shocks, rarefactions and backwater, with potentially different actual face elevations. Dry subcritical-reference admission is sonic. Signed local backflow is retained while total discharge remains inward. Incoming tangent is zero.',
    'inlet.critical_discharge':'The exterior reference section is critical on actual exterior geometry (section Froude number 1, not every local face). Net inward discharge is exact. Boundary Riemann solutions handle shocks, rarefactions and backwater; actual face water lines need not equal the common reference or remain critical. Dry subcritical-reference admission is sonic locally, not necessarily section-critical on an uneven bed. Signed local backflow is retained in the exact net section discharge.',
    'inlet.normal_discharge':'Normal-depth reference uses selected resistance and signed inward bed fall fitted automatically on the prepared finest mesh. This bed slope is an energy-slope proxy under the uniform-flow assumption, frozen across AMR/restart; flat/adverse slope is rejected. The fit is saved in NORMAL_SLOPES.json. Net inward discharge is exact. Boundary Riemann solutions retain a compatible uniform normal flow but admit shocks and backwater instead of forcing normal depth. Actual face water lines can differ from the common reference; dry subcritical-reference admission is sonic. Signed local backflow remains visible.',
    'outlet.stage':'Outflow-only closes inward faces. Optional backflow is subcritical with zero incoming tangent; wet-stage inundation over a dry exterior trace is rejected.',
    'outlet.normal_depth':'Uses the entered positive mean bed slope as an energy-slope approximation for normal flow and the selected hydraulic roughness. Disconnected physical openings are computed separately. This is a uniform-flow assumption, not a measured downstream water level.',
    'outlet.rating_curve':'Enter section discharge (m³/s) and water-surface elevation (m, DEM datum). Values must increase strictly in both columns. The curve represents the whole named outlet and is linearly interpolated without extrapolation. Outflow only.',
    'outlet.critical':'Outgoing-characteristic critical section; local inward flow closes and outward supercritical flow bypasses the critical target.',
    'outlet.supercritical':'Wet outward supercritical flow only. Wet subcritical or reverse flow is rejected; dry faces remain closed.',
    'outlet.free_weir':'Free undrowned weir Q=C W H^1.5, solved on the subcritical control branch with shock/rarefaction compatibility. Local recirculation closes without importing water. A crest that cannot control the flow is rejected; no submerged-weir substitution.',
}


def exact(row,keys,label):
    if not isinstance(row,dict) or set(row)!=set(keys):raise ValueError(label+': required fields '+', '.join(sorted(keys)))


def finite(value,label):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(label+': finite numeric value required')
    return float(value)


def rating_curve(value):
    if not isinstance(value,list) or len(value)<2:raise ValueError('Q–H curve requires at least two discharge/elevation rows')
    rows=[]
    for row in value:
        if not isinstance(row,list) or len(row)!=2:raise ValueError('Each Q–H row requires discharge and water elevation')
        rows.append([finite(row[0],'rating discharge'),finite(row[1],'rating elevation')])
    if any(r[0]<0 for r in rows) or any(b[0]<=a[0] or b[1]<=a[1] for a,b in zip(rows,rows[1:])):
        raise ValueError('Q–H discharge must be nonnegative and both discharge and elevation must increase strictly')
    return rows


def hydraulic_controls(data,controls):
    """Shared method validation; None is only the private component-test default."""
    nb=len(data['mode'])
    if controls is None:
        controls={b:({'method':'inlet.total_discharge','discharge_m3_s':float(data['forcing'][0,b])}
                    if m==6 else {'method':'outlet.stage','stage_m':float(data['forcing'][1,b])}) for b,m in enumerate(data['mode'])}
    if not isinstance(controls,dict) or set(controls)!=set(range(nb)):
        raise ValueError('Named hydraulic control for every exterior boundary required')
    mode=np.empty(nb,np.int32);forcing=np.zeros((2,nb));resolved={}
    for b,row in controls.items():
        method=row.get('method') if isinstance(row,dict) else None
        if method not in HYDRAULICS:raise ValueError('Unsupported hydraulic method')
        exact(row,('method',*HYDRAULICS[method][2]),'Hydraulic control')
        values={k:(rating_curve(row[k]) if k=='rating_curve' else 'mesh' if k=='energy_slope' and row[k]=='mesh' else finite(row[k],k)) for k in HYDRAULICS[method][2]}
        if any(values.get(k,0)<0 for k in ('discharge_m3_s','depth_m')):raise ValueError('Discharge and prescribed depth must be nonnegative')
        for k in ('energy_slope','bed_slope','coefficient_m_half_s','effective_width_m'):
            if k in values and values[k]!='mesh' and values[k]<=0:raise ValueError(k+': positive value required')
        mode[b]={'inlet.total_discharge':6,'outlet.stage':2,'outlet.supercritical':3,
                 'outlet.critical':4,'outlet.free_weir':5,'outlet.normal_depth':10,'outlet.rating_curve':11}.get(method,9)
        forcing[0,b]=values.get('discharge_m3_s',0.)
        forcing[1,b]=values.get('stage_m',values.get('datum_m',0.)+values.get('depth_m',0.))
        if not np.isfinite(forcing[:,b]).all():raise ValueError('Resolved hydraulic control is not finite')
        resolved[str(b)]=dict(method=method,**values)
    return mode,forcing,resolved


def new_state(method,*,sediment=False):
    """Creation defaults only; never fill fields in a loaded public project."""
    from .river_schedule import CONSTANT
    return dict(stage_policy='stage.outflow_only' if method=='outlet.stage' else None,
        secondary_intensity_m_s=0.,
        hydraulic_schedule=dict(method=CONSTANT),
        bed=dict(method='bed.free' if sediment else 'bed.fixed',initial=dict(method='initial.prepared_bed')),
        bed_schedule=dict(method=CONSTANT),
        composition=dict(method='composition.free') if sediment else None,
        composition_schedule=dict(method=CONSTANT) if sediment else None)


def resolve(hydraulics,states,kinds,*,sediment,ng,end_time):
    """Validate before preparation and return explicit resolved method identities."""
    from .river_hydraulic_schedule import validate as validate_hydraulic
    from .river_exterior_schedule import ExteriorForcing,BED_FIELDS,COMPOSITION_FIELDS
    from .river_schedule import CONSTANT
    if not isinstance(hydraulics,dict) or not isinstance(states,dict) or set(hydraulics)!=set(kinds) or set(states)!=set(kinds):
        raise ValueError('Hydraulic and exterior definitions must match every assigned open boundary')
    result={}
    for name in sorted(kinds):
        if type(name) is not str or not name:raise ValueError('Nonempty named boundary required')
        control=hydraulics[name];_,_,validated=hydraulic_controls(dict(mode=[0]),{0:control})
        require_public_hydraulic(control['method'])
        if control['method']=='inlet.normal_discharge' and control['energy_slope']!='mesh':
            raise ValueError(name+': public normal depth requires automatic mesh slope (energy_slope="mesh"); manual historical slopes are not supported')
        if HYDRAULICS[control['method']][1]!=kinds[name]:raise ValueError(name+': hydraulic method does not match inlet/outlet assignment')
        row=states[name]
        exact(row,('stage_policy','secondary_intensity_m_s','hydraulic_schedule','bed','bed_schedule','composition','composition_schedule'),name)
        finite(row['secondary_intensity_m_s'],name+' incoming secondary intensity')
        if control['method']=='outlet.stage':
            if row['stage_policy'] not in STAGE_POLICIES:raise ValueError(name+': explicit stage reversal policy required')
        elif row['stage_policy'] is not None:raise ValueError(name+': reversal policy only belongs to stage outlets')
        validate_hydraulic(validated,{0:row['hydraulic_schedule']},end_time)
        bed=row['bed'];method=bed.get('method') if isinstance(bed,dict) else None
        if method not in BED_FIELDS:raise ValueError(name+': named bed method required')
        field=BED_FIELDS[method];exact(bed,('method','initial',*([field] if field else [])),name+' bed')
        initial=bed['initial'];im=initial.get('method') if isinstance(initial,dict) else None
        if im not in INITIALS:raise ValueError(name+': explicit exterior bed initialization required')
        exact(initial,('method',*(['elevation_m'] if im=='initial.elevation' else [])),name+' initial bed')
        if im=='initial.elevation':finite(initial['elevation_m'],'initial elevation')
        v=finite(bed[field],field) if field else 0.
        if method=='bed.elevation' and initial!=dict(method='initial.elevation',elevation_m=v):
            raise ValueError(name+': prescribed elevation must also be the explicit initial exterior elevation')
        ExteriorForcing(np,{0:method},{0:row['bed_schedule']},end_time,np.array([0]),np.array([v]))
        comp=row['composition'];reason=None;resolved_comp=None
        if sediment is None:
            if (bed!=dict(method='bed.fixed',initial=dict(method='initial.prepared_bed')) or
                row['bed_schedule']!=dict(method=CONSTANT) or comp is not None or row['composition_schedule'] is not None):
                raise ValueError(name+': sediment off requires fixed prepared exterior bed and no active composition forcing')
        else:
            cm=comp.get('method') if isinstance(comp,dict) else None
            if cm not in COMPOSITION_FIELDS:raise ValueError(name+': named composition method required')
            prescribed=COMPOSITION_FIELDS[cm] is not None
            exact(comp,('method',*(['fractions'] if prescribed else [])),name+' composition')
            fractions=comp['fractions'] if prescribed else [1./ng]*ng
            if (not isinstance(fractions,list) or len(fractions)!=ng or
                any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in fractions) or
                abs(math.fsum(fractions)-1)>1e-12):raise ValueError(name+': normalized nonnegative ordered fractions required')
            ExteriorForcing(np,{0:cm},{0:row['composition_schedule']},end_time,np.array([0]),np.array(fractions)[:,None],composition=True)
            resolved_comp=cm
            if cm=='composition.fixed' and sediment['bed']['coarse_layer']:
                resolved_comp='composition.free';reason='FLOW fixed composition with a coarse layer copies the full adjacent column'
        result[name]=dict(hydraulic=control['method'],stage_policy=row['stage_policy'],bed=method,
            secondary=dict(method='prescribed_incoming_secondary_intensity_v1',
                intensity_m_s=row['secondary_intensity_m_s'],active_when='secondary.prognostic and inward water flux'),
            composition=resolved_comp,composition_reason=reason,
            applicability=APPLICABILITY[control['method']],
            table_clock='physical seconds',rate_clock='morphological seconds',ownership='zero-area exterior; no interior clamp')
    return result


def carrier(hydraulics):
    """Inert export records, not an alternative public format or physical choice."""
    return {name:(dict(mode='total_discharge',discharge_m3_s=0.) if row['method'].startswith('inlet.') else
                  dict(mode='stage',stage_m=0.)) for name,row in hydraulics.items()}


def install(a,definition,folder):
    from .sediment.bed_columns_v101 import host
    from .sediment.river_exterior_bed import ExteriorBed
    from .sediment.river_exterior_sediment import ExteriorSediment
    from .sediment.bend_methods_v1 import digest
    hydraulics=definition['hydraulics'];states=definition['states'];names=sorted(hydraulics)
    owner=a
    while 'solver' in vars(owner):owner=owner.solver
    identity=dict(source_sha256=digest(__file__),definition=deepcopy(definition),
        execution='one exterior owner on the selected backend; no carrier physics',public_support=True)
    if not names:
        if len(a.data['mode']):raise ValueError('Empty public boundaries differ from prepared open faces')
        identity['execution']='closed exterior; wall fluxes only; no open-boundary owner'
        owner.identity['public_boundaries']=identity
        return
    d=a.data;p=a.physics;outside=(d['neighbour']<0)&(d['bid']>=0);faces=np.flatnonzero(outside);o=d['owner'][faces]
    clock=getattr(p,'_secondary_clock',None)
    if clock is not None:
        incoming=np.zeros(p.nf)
        for b,name in enumerate(names):
            incoming[outside&(d['bid']==b)]=finite(states[name]['secondary_intensity_m_s'],name+' incoming secondary intensity')
        clock.boundary_values[:]=p.xp.asarray(incoming)
        clock.identity['open_boundary']['incoming_intensity_by_boundary_m_s']={name:states[name]['secondary_intensity_m_s'] for name in names}
    from .sediment import river_open_bed_trace
    z=river_open_bed_trace.prepared_face_bed(p,host(a.z));values=np.zeros(p.nf)
    identity['prepared_bed_initialization']=dict(method='bounded_one_sided_prepared_plane_at_physical_face_v1',
        source_sha256=digest(river_open_bed_trace.__file__),
        geometry='actual cell centroids and physical open-face midpoints; distinct-neighbour weighted LS',
        bounds='one observed neighbour elevation span; isolated cells retain the prepared value',
        overrides='explicit initial.elevation is unchanged')
    for b,name in enumerate(names):
        row=states[name]['bed'];f=outside&(d['bid']==b)
        if row['initial']['method']=='initial.elevation':z[f]=row['initial']['elevation_m']
        values[f]=row.get('elevation_m',row.get('elevation_rate_m_s',0.))
    ext=ExteriorBed(a,methods={b:states[name]['bed']['method'] for b,name in enumerate(names)},initial=z,value=values,
        hydraulics={b:hydraulics[name] for b,name in enumerate(names)},
        stage_policies={b:states[name]['stage_policy'] for b,name in enumerate(names) if hydraulics[name]['method']=='outlet.stage'},
        hydraulic_schedules={b:states[name]['hydraulic_schedule'] for b,name in enumerate(names)},
        bed_schedules={b:states[name]['bed_schedule'] for b,name in enumerate(names)})
    from .sediment.river_explicit import RiverExplicit
    if isinstance(getattr(owner,'midpoint',None),RiverExplicit) and owner.midpoint.explicit_boundary_budget is None:
        owner.midpoint.bind_boundary_budget(ext)
    if a.morphology:
        mass=host(a.columns.mass)[:,:,o];sv=host(a.columns.sv)[:,o]
        fractions=np.full((mass.shape[1],len(faces)),1./mass.shape[1])
        for b,name in enumerate(names):
            row=states[name]['composition']
            if 'fractions' in row:fractions[:,d['bid'][faces]==b]=np.asarray(row['fractions'])[:,None]
        ExteriorSediment(ext,methods={b:states[name]['composition']['method'] for b,name in enumerate(names)},
            initial_mass=mass,initial_sv=sv,fractions=fractions,folder=folder,
            composition_schedules={b:states[name]['composition_schedule'] for b,name in enumerate(names)})
    for key in ('exterior_bed','exterior_sediment'):
        if key in owner.identity:
            owner.identity[key].update(public_support=True,admission='strict named river-project boundary definition')
    from .sediment.river_open_bed_trace import install as install_bed_trace
    install_bed_trace(a)
    owner.identity['public_boundaries']=identity
