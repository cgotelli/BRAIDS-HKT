"""Non-yawed, power-only vertical wake estimator.

The published Gaussian recovered-pressure momentum relation (Bastankhah &
Porte-Agel, as used by Jordan et al. 2022, doi:10.1007/s40722-022-00225-2)
sets the amplitude. Wake width inputs require independent provenance. The
analytical power field uses a first-row incoming reference in a quasi-uniform
straight reach. It is not added to, or substituted for, the SWE velocity field.
This adaptation is not stock Thetis or a three-dimensional flow solution.
"""
import math
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

METHOD = 'power.gaussian_upstream_reference.v1'


def validate(setting):
    if setting is None:
        return {'method': 'thetis'}
    if not isinstance(setting, dict):
        raise ValueError('Select a turbine power estimator.')
    if setting == {'method': 'thetis'}:
        return dict(setting)
    if setting.get('method') in ('power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1'):
        from .river_turbine_combined_power import validate as combined_validate
        return combined_validate(setting)
    if setting.get('method')=='power.device_bem_yaw_ratio.v1':
        from .river_turbine_device_yaw import validate as device_validate
        return device_validate(setting)
    names = {'method', 'growth', 'initial_width', 'minimum_distance_D', 'provenance'}
    if set(setting) != names or setting['method'] != METHOD:
        raise ValueError('Gaussian power requires growth k, initial width epsilon, minimum spacing and the source of these inputs.')
    for key, lo, hi in [('growth', 0., 1.), ('initial_width', .05, 2.), ('minimum_distance_D', 3., 100.)]:
        v = setting[key]
        if type(v) not in (float, int) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'Gaussian power: {key} must be finite in [{lo:g}, {hi:g}].')
    if not isinstance(setting['provenance'], str) or not setting['provenance'].strip():
        raise ValueError('Gaussian power: state the source/calibration of wake growth and width. They are not universal constants or the mesh-refinement length.')
    return dict(setting)


def disk_nodes(diameter, n=24):
    v, w = np.polynomial.legendre.leggauss(n)
    theta = v * math.pi / 2
    y = diameter / 2 * np.cos(theta)[:, None] * v[None, :]
    z = np.broadcast_to((diameter / 2 * np.sin(theta))[:, None], y.shape)
    weights = w[:, None] * w[None, :] * np.cos(theta)[:, None] ** 2 / 2
    return y.ravel(), z.ravel(), weights.ravel()


def vertical_mean(H, hub, sigma, xp=np):
    erf = __import__('scipy.special', fromlist=['erf']).erf if xp is np else __import__('cupyx.scipy.special', fromlist=['erf']).erf
    return sigma * math.sqrt(math.pi / 2) / H * (erf((H-hub)/(math.sqrt(2)*sigma)) + erf(hub/(math.sqrt(2)*sigma)))


def anomaly(H, z, hub, sigma, amplitude, xp=np):
    return amplitude * (vertical_mean(H, hub, sigma, xp) - xp.exp(-.5*((z-hub)/sigma)**2))


def cp_values(control, speed, xp=np):
    from .river_turbine_thetis import lookup, curve_rows
    if control['mode'] != 'constant_ct':
        rows = curve_rows(control,'power_coefficient')
        if bool(xp.any((speed < rows[0]['speed_m_s']) | (speed >= rows[-1]['speed_m_s']))):
            raise ValueError('Reconstructed rotor speed is outside the supplied Cp curve. Extend it with supported device data; no extrapolation or clipping is applied.')
    return lookup(control, speed, 'power_coefficient', xp)


class PowerEstimator:
    """Stateless diagnostic on accepted states; never contributes a force.

    Local disk interpolation is rebuilt with each hydraulic mesh. A fixed
    placement axis defines each non-yawed disk; material misalignment is an
    explicit error. Reversed aligned flow uses the reversed parent ordering.
    """
    def __init__(self, rotors, geometry):
        self.rotors = rotors
        self.setting = validate(rotors[0].c.get('power_estimation'))
        if any(validate(r.c.get('power_estimation')) != self.setting for r in rotors):
            raise ValueError('Use one explicit shared power-estimation configuration for the turbine array.')
        self.combined = self.setting['method'] in ('power.gaussian_device_yaw.v1','power.dou2019_device_yaw.v1')
        self.yaw_wake=self.setting['method']=='power.dou2019_device_yaw.v1'
        self.identity_setting = self.setting
        if self.combined:self.setting=self.setting['wake']
        self.enabled = self.setting['method'] == METHOD
        self.geometry = []
        self.paths = {}
        self.domain_safe=unary_union(geometry.polygons) if self.yaw_wake else None
        if not self.enabled:
            return
        for r in rotors:
            xp = r.xp; c = r.c; centre = np.array([c['X'], c['Y']])
            n = np.asarray(r.original.get('normal_xy', [1., 0.])); n = n/np.linalg.norm(n)
            y, zz, w = disk_nodes(c['D']); tangent = np.array([-n[1], n[0]])
            points = y[:, None] * tangent
            ids = np.asarray(geometry.query(Point(centre).buffer(2*c['D'])), dtype=int)
            xy = geometry.xy[ids] - centre
            if len(ids) < 3:
                raise ValueError(f'Turbine "{c["name"]}": refine the rotor patch for circular power sampling.')
            tri = Delaunay(xy); si = tri.find_simplex(points)
            if np.any(si < 0):
                raise ValueError(f'Turbine "{c["name"]}": circular power samples fall outside wet-mesh centroids; refine or move the rotor away from the wall.')
            b = np.einsum('nij,nj->ni', tri.transform[si, :2], points-tri.transform[si, 2])
            nodes = ids[tri.simplices[si]]
            bary = np.c_[b, 1-b.sum(1)]
            self.geometry.append((xp.asarray(nodes), xp.asarray(bary), xp.asarray(y), xp.asarray(zz), xp.asarray(w), n))
        tolerance = 128*np.finfo(float).eps*max(1., float(abs(geometry.xy).max()))
        for j, r in enumerate(rotors):
            for i, parent in enumerate(rotors):
                if i == j: continue
                path = LineString([(r.c['X'], r.c['Y']), (parent.c['X'], parent.c['Y'])])
                polys = [geometry.polygons[k] for k in geometry.query(path)]
                self.paths[i,j] = bool(polys and unary_union(polys).buffer(tolerance).covers(path))

    def correct(self, samples, q, bed, disabled_sources=()):
        if not self.enabled:
            return samples
        from .river_turbine_thetis import cell_terms
        xp = self.rotors[0].xp
        from . import river_turbine_combined_power as combined
        states=[dict(s) for s in samples]
        for i in disabled_sources:states[i]['active']=False
        axes=[combined.flow_axis(r,g[-1]) if self.combined else g[-1] for r,g in zip(self.rotors,self.geometry)]
        stream_ct=[combined.stream_ct(r,s,axis) if self.combined else s['thrust_coefficient'] for r,s,axis in zip(self.rotors,states,axes)]
        from . import river_turbine_dou as dou
        yaw_data=[dou.source_parameters(r,s,c) if s['active'] and c>0 else (0.,0.) for r,s,c in zip(self.rotors,states,stream_ct)] if self.yaw_wake else None
        signs = []
        for r, geometry, s in zip(self.rotors, self.geometry, samples):
            n = axes[len(signs)]; ids = r.ids
            h = q[0, ids]; uv = q[1:3, ids]/xp.maximum(h, r.dry)
            vec = xp.sum(uv*r.normalized[ids], axis=1)
            if xp is not np: vec = vec.get()
            along = float(np.dot(vec, n)); norm = float(np.linalg.norm(vec))
            if s['active'] and norm > 1e-8 and abs(along)/norm < math.cos(math.radians(10)):
                raise ValueError(f'Turbine "{r.c["name"]}": flow differs by more than 10 degrees from its placement axis. Reorient it with the river; Gaussian power is qualified for non-yawed flow only.')
            signs.append(1 if along >= 0 else -1)
        def exposed(i,x,lateral,radius):
            up=self.rotors[i]
            if self.yaw_wake and yaw_data[i][1]!=0:
                sy,sz,offset,a,m,_=dou.parameters(up.c['D'],x,*yaw_data[i],self.setting['growth'],self.setting['initial_width'])
                return abs(lateral-float(offset-sy*m))<=4*float(sy)+radius
            sigma=up.c['D']*self.setting['initial_width']+self.setting['growth']*x
            return abs(lateral)<=4*sigma+radius
        parent_map=[]
        for j,r in enumerate(self.rotors):
            parents=[]
            for i,(up,state) in enumerate(zip(self.rotors,states)):
                if i==j or not state['active'] or state['thrust_coefficient']<=0:continue
                un=axes[i]*signs[i];d=np.array([r.c['X']-up.c['X'],r.c['Y']-up.c['Y']]);x=float(d@un)
                if x<=1e-8*up.c['D']:continue
                sigma=up.c['D']*self.setting['initial_width']+self.setting['growth']*x
                if exposed(i,x,float(d@np.array([-un[1],un[0]])),r.c['D']/2):parents.append(i)
            parent_map.append(parents)
        def roots(j,seen=()):
            if j in seen:raise ValueError('Cyclic/opposing wake directions are outside the aligned analytical power model.')
            return {j} if not parent_map[j] else set().union(*(roots(i,(*seen,j)) for i in parent_map[j]))
        root_map=[sorted(roots(j)) for j in range(len(samples))]
        for j, (r, geo, s) in enumerate(zip(self.rotors, self.geometry, samples)):
            original = s['useful_power_per_density_m5_s3']
            s.update(power_estimator=(self.identity_setting['method'] if self.combined else METHOD), uncorrected_power_W=original*r.c['rho'], power_correction_factor=1.,
                     rotor_power_velocity_m_s=s['upstream_velocity_m_s'], active_upstream_wakes=0,
                     reconstruction_active=False, wake_growth=self.setting['growth'], wake_initial_width=self.setting['initial_width'],
                     wake_parameter_source=self.setting['provenance'],
                     ambient_reference_velocity_m_s=s['upstream_velocity_m_s'],inferred_ambient_discrepancy_percent=0.,
                     power_assumptions=(dou.ASSUMPTIONS if self.yaw_wake else combined.ASSUMPTIONS if self.combined else 'quasi-steady, straight quasi-uniform ambient reach; published Gaussian deficits on first-row incoming reference; mechanical power; unchanged Thetis force'))
            if not s['active']: continue
            nodes,bary,y,zrel,w,n = geo
            if bool(xp.any(q[0,nodes] <= r.dry)):
                raise ValueError(f'Turbine "{r.c["name"]}": power quadrature touches dry cells; refine/move the rotor.')
            def interp(a): return xp.sum(a[nodes]*bary, axis=1)
            h = interp(q[0]); u = interp(q[1]/xp.maximum(q[0],r.dry)); v = interp(q[2]/xp.maximum(q[0],r.dry))
            zbed = interp(bed); z = r.c['hub_elevation_m']+zrel-zbed
            if self.combined:incident,angle=combined.incoming(r,h,u,v,n,xp)
            else:_,_,incident,_,_ = cell_terms(r.control,r.c['D'],h,u,v,xp)
            references=[states[i]['upstream_velocity_m_s'] for i in root_map[j]]
            reference=float(np.mean(references))
            if max(references)-min(references)>.1*reference:
                raise ValueError(f'Turbine "{r.c["name"]}": first-row reference speeds differ by more than 10%; this quasi-uniform ambient power model is not applicable. Use stock Thetis power or subdivide into independently qualified reaches.')
            if parent_map[j] and bool(xp.any(h<2*r.c['D'])):
                raise ValueError(f'Turbine "{r.c["name"]}": Gaussian power requires depth/D at least 2 in this qualified scope. Use stock Thetis power for shallower conditions.')
            deficit = xp.zeros_like(h); mean_deficit = xp.zeros_like(h); parents = 0
            contributions=[]
            for i, (up, state) in enumerate(zip(self.rotors,states)):
                if i == j or not state['active'] or state['thrust_coefficient'] <= 0: continue
                un = axes[i]*signs[i]; tangent = np.array([-un[1],un[0]])
                displacement = np.array([r.c['X']-up.c['X'],r.c['Y']-up.c['Y']])
                distance = float(displacement@un)
                if distance <= 1e-8*up.c['D']: continue
                sigma = up.c['D']*self.setting['initial_width']+self.setting['growth']*distance
                lateral = float(displacement@tangent)
                if not exposed(i,distance,lateral,r.c['D']/2):continue
                if distance/up.c['D'] < self.setting['minimum_distance_D']:
                    raise ValueError(f'Turbine "{r.c["name"]}" is only {distance/up.c["D"]:.3g}D behind "{up.c["name"]}"; this far-wake power model requires at least {self.setting["minimum_distance_D"]:g}D. Increase spacing or select uncorrected Thetis power.')
                if not self.paths[i,j] or abs(float(un@(axes[j] if self.combined else n))) < math.cos(math.radians(10)):
                    raise ValueError(f'Wake from "{up.c["name"]}" to "{r.c["name"]}" crosses a wall or the placed river directions differ by more than 10 degrees. This analytical power model requires a straight reach; rotor yaw is a separate setting.')
                node_distance=distance
                if self.combined and r.original.get('orientation',{}).get('mode')=='fixed_axis':
                    # Integrate the actual tilted receiver plane, not a normal plane at its centre.
                    node_distance=distance+y*float(np.array([-n[1],n[0]])@un)
                    if bool(xp.any(node_distance/up.c['D']<self.setting['minimum_distance_D'])):raise ValueError('The tilted rotor enters the declared near-wake exclusion.')
                    sigma=up.c['D']*self.setting['initial_width']+self.setting['growth']*node_distance
                if self.yaw_wake and yaw_data[i][1]!=0:
                    yy=lateral+y*float(np.array([-n[1],n[0]])@tangent)
                    shape,sigma=dou.lateral(up.c['D'],node_distance,yy,*yaw_data[i],self.setting['growth'],self.setting['initial_width'],xp=xp)
                    amp=state['upstream_velocity_m_s']*shape
                    # Check the deflected centre path against solid/domain walls.
                    xx=np.linspace(max(1e-8,up.c['D']*.01),distance,17)
                    sy,sz,off,aa,mm,am=dou.parameters(up.c['D'],xx,*yaw_data[i],self.setting['growth'],self.setting['initial_width'])
                    pp=np.array([up.c['X'],up.c['Y']])+xx[:,None]*un+off[:,None]*tangent
                    if not self.domain_safe.covers(LineString(pp)):
                        raise ValueError('Deflected analytical wake crosses a domain/solid wall; this unconfined profile is not applicable.')
                else:
                    rad = 1-stream_ct[i]*up.c['D']**2/(8*sigma**2)
                    if bool(xp.any(rad <= 0)):
                        raise ValueError(f'Wake from "{up.c["name"]}": width is too small for the Gaussian momentum relation. Check its independently calibrated width/growth and spacing.')
                    yy = lateral+y*float(np.array([-n[1],n[0]])@tangent)
                    amp = state['upstream_velocity_m_s']*(1-(xp.sqrt(rad) if self.combined else math.sqrt(rad)))*xp.exp(-.5*(yy/sigma)**2)
                hub = up.c['hub_elevation_m']-zbed
                if bool(xp.any((hub <= 0)|(hub >= h))):
                    raise ValueError(f'Wake axis from "{up.c["name"]}" is outside the receiving water column. This profile is not qualified across this bed/depth change.')
                part=amp*xp.exp(-.5*((z-hub)/sigma)**2)
                contributions.append(dict(turbine_id=state['turbine_id'],distance_D=distance/up.c['D'],rotor_mean_deficit_m_s=float(xp.sum(w*part)),source_incoming_m_s=state['upstream_velocity_m_s']))
                deficit += part
                mean_deficit += amp*vertical_mean(h,hub,sigma,xp); parents += 1
            if not parents: continue
            reconstructed = reference-deficit
            if bool(xp.any(~xp.isfinite(reconstructed)|(reconstructed <= 0))):
                raise ValueError(f'Turbine "{r.c["name"]}": overlapping wake profiles predict nonpositive rotor flow; no clipping is applied.')
            if self.combined:
                cp=combined.effective_cp(r,reconstructed,angle,xp)
                if cp is None:
                    combined.stop(s,parents);continue
            else:cp = cp_values(r.control,reconstructed,xp)
            corrected = xp.sum(w*cp*reconstructed**3)
            stats = xp.stack((xp.sum(w*(incident+mean_deficit)),corrected,xp.sum(w*reconstructed**3),xp.sum(w*cp)))
            if xp is not np: stats = stats.get()
            inferred,new,cube,cpmean = map(float,stats)
            discrepancy=100*(inferred/reference-1)
            if abs(discrepancy)>10:
                raise ValueError(f'Turbine "{r.c["name"]}": inferred local ambient speed differs from its first-row reference by {discrepancy:.2f}%. River acceleration, blockage or mixing is outside this quasi-uniform power scope; use stock Thetis power or a qualified ambient field.')
            if original <= 0: raise ValueError(f'Turbine "{r.c["name"]}": zero reference power inside the active analytical range.')
            power=.5*math.pi*r.c['D']**2/4*new;ratio=power/original
            if power > s['hydraulic_power_removed_per_density_m5_s3']*(1+1e-8):
                raise ValueError(f'Turbine "{r.c["name"]}": estimated shaft power exceeds hydraulic source work. Check curve reference and wake inputs.')
            s.update(useful_power_per_density_m5_s3=power,power_correction_factor=ratio,
                     rotor_power_velocity_m_s=cube**(1/3),power_coefficient=new/cube,
                     power_basis='Circular rotor integral of Cp(U) U^3 on analytical incoming wake field; mechanical power; generator losses excluded',
                     coefficient_basis=s['coefficient_basis']+'; reported Cp is rotor U^3-weighted for analytical power; Ct still owns the hydraulic source',
                     ambient_reference_velocity_m_s=reference,inferred_ambient_discrepancy_percent=discrepancy,
                     active_upstream_wakes=parents,reconstruction_active=True,upstream_wake_contributions=contributions)
            if self.yaw_wake:s.update(wake_profile='Dou2019 Eqs1/3/12/14/15 with aligned Gaussian vertical completion')
            if self.combined:s.update(power_basis='Circular integral of Cp0(U) U^3 times device BEM yaw ratio; Gaussian incoming magnitude and local incident direction',power_assumptions=(dou.ASSUMPTIONS if self.yaw_wake else combined.ASSUMPTIONS),streamwise_wake_Ct=stream_ct[j])
        return samples
