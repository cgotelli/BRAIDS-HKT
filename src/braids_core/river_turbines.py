"""Strict Desktop turbine contract, placement geometry and preparation mapping."""
from copy import deepcopy
import math
import numpy as np
from shapely.geometry import Polygon,Point,LineString

METHOD='actuator.thetis_depth_averaged.v1'
WAKE_WIDTH_DIAMETERS=5.


def definition(center,direction,*,name='Turbine 1',diameter=0.7,hub_height=1.,minimum_bed_clearance=.1,minimum_surface_clearance=.1):
    return dict(id=name,center_xy=list(map(float,center)),river_direction_xy=list(map(float,direction)),
        rotor_diameter_m=float(diameter),hub_height_m=float(hub_height),minimum_bed_clearance_m=float(minimum_bed_clearance),minimum_surface_clearance_m=float(minimum_surface_clearance),
        control=dict(mode='constant_ct',thrust_coefficient=.8,power_coefficient=.4),
        wake_refinement=dict(enabled=True,length_diameters=10.))


def default():return dict(method=METHOD,turbines=[],ambient_viscosity_m2_s=.006)


def coefficients(row):
    from .sediment.river_turbine_thetis import validate_control
    control=validate_control(row['control'])
    return control.get('thrust_coefficient'),control.get('power_coefficient'),control


def orientation(row):
    value=row.get('orientation',{'mode':'flow_following'})
    if value=={'mode':'flow_following'}:return dict(value)
    if (not isinstance(value,dict) or set(value)!={'mode','yaw_deg'} or value['mode']!='fixed_axis'
        or type(value['yaw_deg']) not in (int,float) or not math.isfinite(value['yaw_deg']) or not -90<=value['yaw_deg']<=90):
        raise ValueError('Choose flow-following loading, or fixed-axis loading with yaw_deg between -90 and 90. Positive yaw rotates the rotor normal counterclockwise from the placed river direction.')
    return dict(value)


def river_frame(row):
    d=np.asarray(row['river_direction_xy'],float);n=d/np.linalg.norm(d);return n,np.array([-n[1],n[0]])


def frame(row):
    n,t=river_frame(row);o=orientation(row)
    if o['mode']=='fixed_axis':
        angle=math.radians(o['yaw_deg']);n=math.cos(angle)*n+math.sin(angle)*t
    return n,np.array([-n[1],n[0]])


def rotor_line(row):
    n,t=frame(row);c=np.asarray(row['center_xy']);r=row['rotor_diameter_m']/2
    return [(c-r*t).tolist(),(c+r*t).tolist()]


def footprint(row):
    # Stock Thetis actuator density is axis aligned, independent of flow angle.
    from shapely.geometry import box
    x,y=row['center_xy'];r=row['rotor_diameter_m']/2
    if orientation(row)['mode']=='flow_following':return box(x-r,y-r,x+r,y+r)
    n,t=frame(row);c=np.array([x,y])
    return Polygon([c+a*n+b*t for a,b in [(-r,-r),(r,-r),(r,r),(-r,r)]])


def wake_footprint(row):
    """Mesh protection only, along the user's placed downstream direction."""
    setting=row['wake_refinement']
    if not setting['enabled']:return None
    n,t=river_frame(row);c=np.asarray(row['center_xy']);D=row['rotor_diameter_m']
    half=.5*WAKE_WIDTH_DIAMETERS*D;length=setting['length_diameters']*D
    return Polygon([c+x*n+y*t for x,y in [(0.,-half),(length,-half),(length,half),(0.,half)]])


def validate(value,project=None,*,allow_empty=False):
    if value is None:return None
    if not isinstance(value,dict) or set(value)-{'power_estimation'}!={'method','turbines','ambient_viscosity_m2_s'} or value['method']!=METHOD:raise ValueError('Choose the current Thetis turbine method. Old Gaussian/TSR turbine projects must be reconfigured with constant coefficients or a speed / Ct / Cp table.')
    from .sediment.river_turbine_power import validate as validate_power
    validate_power(value.get('power_estimation'))
    nu=value['ambient_viscosity_m2_s']
    if type(nu) not in (float,int) or not math.isfinite(nu) or nu<0:raise ValueError('Set a finite nonnegative turbine ambient viscosity (m²/s).')
    rows=value['turbines']
    if not isinstance(rows,list) or (not rows and not allow_empty):raise ValueError('Place at least one turbine with Add turbine.')
    names=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)-{'orientation'}!={'id','center_xy','river_direction_xy','rotor_diameter_m','hub_height_m','minimum_bed_clearance_m','minimum_surface_clearance_m','control','wake_refinement'}:raise ValueError('Current turbine fields with explicit bed and surface clearance limits and wake refinement are required; create a new turbine.')
        o=orientation(row)
        if value.get('power_estimation',{}).get('method') in ('power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1') and o['mode']=='fixed_axis':
            table=value['power_estimation']['device']
            if abs(row['rotor_diameter_m']-table['diameter_m'])>1e-12:raise ValueError('The device yaw table diameter must match the fixed-axis rotor.')
            maximum=min(32.,table['yaw_deg'][-1]) if value['power_estimation']['method']=='power.dou2019_device_yaw.v1' else table['yaw_deg'][-1]
            if abs(o['yaw_deg'])>maximum:
                raise ValueError(f'Turbine "{row["id"]}": selected yaw exceeds the available {maximum:g}-degree range. Use an angle covered by both the device table and wake model. Actual local yaw is also checked during the run.')
        if value.get('power_estimation',{}).get('method')=='power.device_bem_yaw_ratio.v1':
            if o['mode']!='fixed_axis':raise ValueError('The device yaw table requires explicit fixed-axis rotors, including zero yaw.')
            if abs(row['rotor_diameter_m']-value['power_estimation']['diameter_m'])>1e-12:raise ValueError('The yaw table diameter must match the actual rotor. Do not rescale a device implicitly.')
        if o['mode']=='fixed_axis':
            if value.get('power_estimation',{'method':'thetis'}).get('method') not in ('thetis','power.device_bem_yaw_ratio.v1','power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1'):
                raise ValueError('Fixed-axis turbines cannot use the non-yawed Gaussian estimator. Select Dou (2019) array power and import matching device yaw data, or retain resolved-field Thetis power with its incoming-normal convention.')
            if row['control'].get('mode')=='thetis_speed_table':
                raise ValueError('Fixed-axis turbines require zero-yaw incoming-speed Ct/Cp curves or constant coefficients. Supply actual device data at the intended controller/RPM. Curves use incoming speed magnitude; the cosine yaw reduction is applied once. The mixed local-speed stock table cannot be relabelled as an incoming-speed curve.')
        if not isinstance(row['id'],str) or not row['id'].strip():raise ValueError('Give the turbine a nonempty name.')
        if row['id'] in names:raise ValueError('Turbine names must be unique so each rotor has its own settings and results.')
        names.add(row['id'])
        for k in ['center_xy','river_direction_xy']:
            a=np.asarray(row[k],float)
            if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError('Turbine position and river direction require two finite coordinates.')
        if np.linalg.norm(row['river_direction_xy'])<1e-10:raise ValueError('Click a second point along the river, different from the turbine centre.')
        for k in ['rotor_diameter_m','hub_height_m']:
            if type(row[k]) not in (float,int) or not math.isfinite(row[k]) or row[k]<=0:raise ValueError(f'Turbine "{row["id"]}": {k} must be a positive finite value in metres.')
        if row['hub_height_m']<=row['rotor_diameter_m']/2:raise ValueError(f'Turbine "{row["id"]}": hub height {row["hub_height_m"]:g} m must be greater than rotor radius {row["rotor_diameter_m"]/2:g} m; otherwise the rotor touches/intersects the bed. Raise the hub, also allowing the required bed clearance.')
        for k in ['minimum_bed_clearance_m','minimum_surface_clearance_m']:
            if type(row[k]) not in (float,int) or not math.isfinite(row[k]) or row[k]<0:raise ValueError('Minimum bed clearance and minimum surface cover must be finite nonnegative metres.')
        if row['hub_height_m']-row['rotor_diameter_m']/2 < row['minimum_bed_clearance_m']-1e-12:
            raise ValueError(f'Turbine "{row["id"]}": bottom clearance is {row["hub_height_m"]-row["rotor_diameter_m"]/2:g} m, below the required {row["minimum_bed_clearance_m"]:g} m. Increase hub height to at least {row["rotor_diameter_m"]/2+row["minimum_bed_clearance_m"]:g} m (radius + clearance), or revise the specified clearance.')
        try:coefficients(row)
        except (ValueError,TypeError,KeyError) as exc:raise ValueError(f'Turbine "{row["id"]}": {exc}') from exc
        wake=row['wake_refinement']
        if (not isinstance(wake,dict) or set(wake)!={'enabled','length_diameters'} or type(wake['enabled']) is not bool
            or type(wake['length_diameters']) not in (int,float) or not math.isfinite(wake['length_diameters'])
            or not 0<wake['length_diameters']<=1000):raise ValueError('Wake refinement needs an on/off choice and a positive length up to 1000 rotor diameters.')
    # Rotor projections must not intersect. Vertical stacking is not an admitted
    # layout. Distinguish this physical placement check from numerical support.
    tolerance=128*np.finfo(float).eps*max(1.,max((abs(x) for r in rows for x in r['center_xy']),default=1.))
    for j,b in enumerate(rows):
        cb=np.asarray(b['center_xy']);tb=frame(b)[1]*b['rotor_diameter_m']/2
        lb=LineString([cb-tb,cb+tb])
        for a in rows[:j]:
            ca=np.asarray(a['center_xy']);ta=frame(a)[1]*a['rotor_diameter_m']/2
            la=LineString([ca-ta,ca+ta])
            if la.distance(lb)<=tolerance:
                raise ValueError(f'Turbines "{a["id"]}" and "{b["id"]}" overlap or touch in their rotor plan projections. Separate the rotor tips; vertically stacked rotors are not supported.')
            overlap=footprint(a).intersection(footprint(b)).area
            if overlap>max(1e-10*min(a['rotor_diameter_m'],b['rotor_diameter_m'])**2,tolerance*max(a['rotor_diameter_m'],b['rotor_diameter_m'])):
                raise ValueError(f'Turbines "{a["id"]}" and "{b["id"]}" have overlapping D-by-D numerical actuator footprints ({overlap:.6g} m²). This is a source-support restriction, distinct from a rotor collision or overlapping downstream wakes. Increase their separation until the force footprints no longer overlap.')
    if project is not None:
        from .river_jiang_cao import active as jiang_cao_active
        if jiang_cao_active(project['model']):
            raise ValueError('Jiang-Cao uses a separate mixture solver without turbine momentum coupling. Select an ordinary sediment transport model when turbines are active; otherwise turbine forces would be omitted.')
        # Ordinary sediment combinations are governed by their own model and
        # integrator contracts, not by a turbine-specific formula whitelist.
        hydro=project['model']['request']['hydrodynamics']
        if hydro['turbulence']!='off' and project['settings']['controls']['viscosity']>0. and nu>0.:
            raise ValueError('Ambient turbulent viscosity is specified twice. Set Hydrokinetic turbines ambient viscosity to 0 to use the Advanced hydrodynamics value, or turn off Advanced horizontal turbulent viscosity. Secondary flow can remain enabled.')
        s=project['settings'];dx=s['grid']['dx_m']/2**(s['levels']-1)
        domain=Polygon(s['domain']['vertices_xy'])
        for solid in s['solids']:domain=domain.difference(Polygon(solid['points_xy']))
        for row in rows:
            if not domain.covers(footprint(row)):raise ValueError(f'Turbine "{row["id"]}": move its entire D-by-D force footprint inside the fluid domain and away from solids.')
    return deepcopy(value)


def constraints(value):
    return [] if value is None else [dict(id='__turbine_'+r['id'],closed=False,points_xy=rotor_line(r)) for r in value['turbines']]


def mesh_regions(value):
    """Derived local resolution; the user's river minimum remains unchanged."""
    regions=[]
    for r in (() if value is None else value['turbines']):
        for kind,shape in [('rotor',footprint(r)),('wake',wake_footprint(r))]:
            if shape is not None:regions.append(dict(id=r['id']+'/'+kind,
                maximum_size_m=r['rotor_diameter_m']/2,points_xy=list(map(list,shape.exterior.coords))))
    return regions


def prepare(value,tree,bed,*,cancelled=lambda:False):
    if value is None:return None
    cells=tree.finest.cells;bed=np.asarray(bed);base=tree.offsets[-2];required=set();rows=[]
    from .sediment.river_turbine_geometry import GeometryIndex
    geometry=GeometryIndex(cells,cancelled=cancelled)
    # These permanent subcells include the graded transition, which must not
    # disappear during depth-driven adaptation outside the wet corridor.
    ordinary_area=tree.finest.grid.dx_m*tree.finest.grid.dy_m
    local={i for i,c in enumerate(cells) if c.nominal_area_m2<ordinary_area*(1-1e-12)}
    required.update(base+i for i in local)
    for face in tree.finest.faces:
        if face.neighbour>=0 and (face.owner in local or face.neighbour in local):
            required.update((base+face.owner,base+face.neighbour))
    for row in value['turbines']:
        patch=footprint(row);ids=[]
        if cancelled():raise InterruptedError('Turbine preparation cancelled')
        for i in geometry.query(patch):
            if geometry.polygons[i].intersection(patch).area>1e-14*patch.area:ids.append(i)
        center=Point(row['center_xy']);covers=[i for i in geometry.query(center) if geometry.polygons[i].covers(center)]
        if not covers:raise ValueError('Turbine centre is outside the prepared fluid mesh.')
        hub=float(max(bed[i] for i in covers)+row['hub_height_m']);required.update(base+i for i in ids)
        wake=wake_footprint(row);wake_ids=[]
        if wake is not None:
            for i in geometry.query(wake):
                if cancelled():raise InterruptedError('Wake refinement cancelled')
                if geometry.polygons[i].intersection(wake).area>1e-14*min(wake.area,geometry.polygons[i].area):wake_ids.append(i)
            required.update(base+i for i in wake_ids)
        ct,cp,control=coefficients(row);n,tangent=frame(row)
        actual=max(math.sqrt(cells[i].nominal_area_m2) for i in ids)
        if actual>row['rotor_diameter_m']/2*(1+1e-10):raise ValueError('Turbine local mesh is unresolved; prepare the mesh with automatic turbine refinement.')
        rows.append(dict(name=row['id'],D=row['rotor_diameter_m'],X=row['center_xy'][0],Y=row['center_xy'][1],normal_xy=n.tolist(),
            orientation=orientation(row),river_direction_xy=river_frame(row)[0].tolist(),
            hub_elevation_m=hub,hub_height_m=row['hub_height_m'],hub_reference_finest_cells=covers,
            minimum_bed_clearance_m=row['minimum_bed_clearance_m'],minimum_surface_clearance_m=row['minimum_surface_clearance_m'],
            ct=ct,cp=cp,control=control,rho=1000.,nu=value['ambient_viscosity_m2_s'],power_estimation=deepcopy(value.get('power_estimation',{'method':'thetis'})),
            local_mesh=dict(maximum_size_m=row['rotor_diameter_m']/2,actual_rotor_size_m=actual,
                ordinary_size_m=math.sqrt(ordinary_area),automatic=actual<math.sqrt(ordinary_area)*(1-1e-12)),
            wake_refinement=dict(**row['wake_refinement'],width_diameters=WAKE_WIDTH_DIAMETERS,finest_cells=len(wake_ids))))
    return dict(method=METHOD,definition=deepcopy(value),required_leaves=sorted(required),rotors=rows,
        hub_reference='maximum incident prepared cell-mean bed at rotor centre; fixed absolute elevation thereafter')
