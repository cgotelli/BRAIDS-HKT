"""Thetis discrete-turbine equations on BRAIDS finite-volume cells.

Adapted from thetisproject/thetis 162c605079611dfce69b4d1182181a421116b141,
thetis/turbines.py (MIT; see THIRD_PARTY_LICENSES/Thetis.txt).
The hydraulic source has no analytical wake overlay, TSR resolver or yaw force.
A separately selected analytical estimator can change accepted power diagnostics.
Cell-local total water depth is used in both drag and power; the historical
Thetis diagnostic callback's bathymetry argument is not copied.
"""
import math
import hashlib
from pathlib import Path
import numpy as np
from .river_turbine_flow import CoveredRotor
from .river_turbine_geometry import GeometryIndex

METHOD = 'actuator.thetis_depth_averaged.v1'
REFERENCE = '162c605079611dfce69b4d1182181a421116b141'
_TABLES = {}
_INCIDENT_KERNEL = None
_INCIDENT_CURVES_KERNEL = None


def curve_rows(control, field):
    """Independent device data; shared legacy tables retain their exact lookup."""
    if control['mode'] == 'incident_speed_curves':
        return control['thrust_curve' if field == 'thrust_coefficient' else 'power_curve']
    return control['performance']


def curve_arrays(control, field, xp):
    key=(id(control),xp.__name__,field)
    if key not in _TABLES or _TABLES[key][0] is not control:
        if len(_TABLES)>=64:_TABLES.pop(next(iter(_TABLES)))
        rows=curve_rows(control,field)
        _TABLES[key]=(control,xp.asarray([r['speed_m_s'] for r in rows],dtype=float),
                     xp.asarray([r[field] for r in rows],dtype=float))
    return _TABLES[key][1:]


def table_arrays(control,xp):
    # Validated operating controls are immutable throughout a run. Keep their
    # host/device arrays rather than rebuilding a device table at every stage.
    key=(id(control),xp.__name__)
    if key not in _TABLES or _TABLES[key][0] is not control:
        if len(_TABLES)>=64:_TABLES.pop(next(iter(_TABLES)))
        rows=control['performance']
        _TABLES[key]=(control,xp.asarray([r['speed_m_s'] for r in rows]),
            xp.asarray([r['thrust_coefficient'] for r in rows]),xp.asarray([r['power_coefficient'] for r in rows]))
    return _TABLES[key][1:]


def incident_gpu(control,diameter,depth,u,v,xp):
    global _INCIDENT_KERNEL, _INCIDENT_CURVES_KERNEL
    if _INCIDENT_KERNEL is None:
        _INCIDENT_KERNEL=xp.ElementwiseKernel(
            'float64 diameter, float64 depth, float64 u, float64 v, raw float64 xs, raw float64 cts, raw float64 cps, int32 count',
            'float64 K, float64 P, float64 U, float64 Ct, float64 Cp',
            r'''
            double s=hypot(u,v), c=3.14159265358979323846*diameter/(4*depth);
            double lo=xs[0],hi=xs[count-1];
            double smin=lo*.5*(1+sqrt(1-c*cts[0])),smax=hi*.5*(1+sqrt(1-c*cts[count-1]));
            bool active=s>=smin && s<smax;
            for(int it=0;it<44;it++){
                double mid=.5*(lo+hi);int a=0,b=count-1;
                while(b-a>1){int m=(a+b)/2;if(xs[m]<=mid)a=m;else b=m;}
                double ct=cts[a]+(cts[b]-cts[a])*(mid-xs[a])/(xs[b]-xs[a]);
                double local=mid*.5*(1+sqrt(1-c*ct));
                if(local<s)lo=mid;else hi=mid;
            }
            U=active?.5*(lo+hi):s;Ct=0;Cp=0;
            if(active){int a=0,b=count-1;while(b-a>1){int m=(a+b)/2;if(xs[m]<=U)a=m;else b=m;}
                double t=(U-xs[a])/(xs[b]-xs[a]);Ct=cts[a]+(cts[b]-cts[a])*t;Cp=cps[a]+(cps[b]-cps[a])*t;}
            double alpha=.5*(1+sqrt(1-c*Ct)),A=3.14159265358979323846*diameter*diameter/4;
            K=A*Ct/(2*alpha*alpha);P=.5*A*Cp*U*U*U;
            ''','braids_incident_curve_induction_v1')
    if control['mode']=='incident_speed_curves':
        if _INCIDENT_CURVES_KERNEL is None:
            # Identical Ct inversion and hydraulic source, independent Cp knots.
            operation=_INCIDENT_KERNEL.operation.replace(
                'Cp=cps[a]+(cps[b]-cps[a])*t;',
                'int pa=0,pb=pcount-1;while(pb-pa>1){int m=(pa+pb)/2;if(pxs[m]<=U)pa=m;else pb=m;} '
                'Cp=cps[pa]+(cps[pb]-cps[pa])*(U-pxs[pa])/(pxs[pb]-pxs[pa]);')
            _INCIDENT_CURVES_KERNEL=xp.ElementwiseKernel(
                'float64 diameter, float64 depth, float64 u, float64 v, raw float64 xs, raw float64 cts, raw float64 cps, int32 count, raw float64 pxs, int32 pcount',
                'float64 K, float64 P, float64 U, float64 Ct, float64 Cp',operation,
                'braids_independent_incident_curves_v1')
        x,ct=curve_arrays(control,'thrust_coefficient',xp)
        px,cp=curve_arrays(control,'power_coefficient',xp)
        return _INCIDENT_CURVES_KERNEL(diameter,depth,u,v,x,ct,cp,len(x),px,len(px))
    x,ct,cp=table_arrays(control,xp)
    return _INCIDENT_KERNEL(diameter,depth,u,v,x,ct,cp,len(control['performance']))


def validate_control(control):
    if not isinstance(control, dict):
        raise ValueError('Choose Thetis constant coefficients or a speed / Ct / Cp table.')
    c = dict(control)
    mode = c.get('mode')
    if mode == 'constant_ct':
        if set(c) != {'mode', 'thrust_coefficient', 'power_coefficient'}:
            raise ValueError('Constant input requires Ct and Cp (null or zero selects Thetis ideal-disk Cp).')
        pairs = [(c['thrust_coefficient'], c['power_coefficient'])]
    elif mode == 'incident_speed_curves':
        if set(c) != {'mode','model_name','thrust_curve','power_curve'} or not isinstance(c['model_name'],str) or not c['model_name'].strip():
            raise ValueError('Name the device curves and supply separate thrust_curve and power_curve tables, both referenced to incoming speed.')
        for key,field,label in [('thrust_curve','thrust_coefficient','Ct'),('power_curve','power_coefficient','Cp')]:
            rows=c[key]
            if not isinstance(rows,list) or len(rows)<2:
                raise ValueError(f'{label} curve requires at least two speed / {label} rows.')
            speeds=[]
            for row in rows:
                if not isinstance(row,dict) or set(row)!={'speed_m_s',field}:
                    raise ValueError(f'{label} rows require only speed_m_s and {field}.')
                u=row['speed_m_s'];value=row[field]
                if type(u) not in (int,float) or not math.isfinite(u) or u<0:
                    raise ValueError(f'{label} speeds must be finite and nonnegative.')
                if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (value>=1 if label=='Ct' else value>1):
                    raise ValueError(f'{label} must satisfy '+('0 <= Ct < 1.' if label=='Ct' else '0 <= Cp <= 1.'))
                speeds.append(u)
            if any(b<=a for a,b in zip(speeds,speeds[1:])):
                raise ValueError(f'{label} speeds must be strictly increasing; remove duplicates and sort the rows.')
        if c['power_curve'][0]['speed_m_s']>c['thrust_curve'][0]['speed_m_s'] or c['power_curve'][-1]['speed_m_s']<c['thrust_curve'][-1]['speed_m_s']:
            raise ValueError('The Cp speed range must cover the entire Ct operating range. Supply supported Cp data; no extrapolation is applied. Additional Cp coverage can support wake-affected rotor sampling without extending Ct operation.')
        pairs=[]
    elif mode in ('thetis_speed_table', 'incident_speed_table'):
        if set(c) != {'mode', 'model_name', 'performance'} or not isinstance(c['model_name'], str) or not c['model_name'].strip():
            raise ValueError('Name the turbine speed table and provide speed, Ct and Cp rows.')
        rows = c['performance']
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError('Thetis needs at least two speed / Ct / Cp rows.')
        speeds = []
        for r in rows:
            if not isinstance(r, dict) or set(r) != {'speed_m_s', 'thrust_coefficient', 'power_coefficient'}:
                raise ValueError('Each Thetis table row requires speed_m_s, thrust_coefficient and power_coefficient.')
            u = r['speed_m_s']
            if type(u) not in (int, float) or not math.isfinite(u) or u < 0:
                raise ValueError('Table speeds must be finite and nonnegative.')
            speeds.append(u)
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError('Table speeds must be strictly increasing; remove duplicates and sort the rows.')
        pairs = [(r['thrust_coefficient'], r['power_coefficient']) for r in rows]
    else:
        raise ValueError('Choose constant Ct/Cp or an explicitly referenced speed / Ct / Cp table. Old TSR/speed-surface controls are unsupported. The optional array-power estimator is selected separately.')
    for ct, cp in pairs:
        if type(ct) not in (int, float) or not math.isfinite(ct) or not 0 <= ct < 1:
            raise ValueError('Ct must be finite, with 0 <= Ct < 1.')
        if cp is None and mode == 'constant_ct':
            continue
        if type(cp) not in (int, float) or not math.isfinite(cp) or not 0 <= cp <= 1:
            raise ValueError('Cp must be finite and between 0 and 1.')
    if mode in ('incident_speed_table','incident_speed_curves'):
        # H>=D for a fully covered rotor. At H=D this is the strongest
        # induction in that envelope; check both endpoints of each linear Ct.
        rows=curve_rows(c,'thrust_coefficient')
        for left,right in zip(rows,rows[1:]):
            slope=(right['thrust_coefficient']-left['thrust_coefficient'])/(right['speed_m_s']-left['speed_m_s'])
            for row in (left,right):
                root=math.sqrt(1-math.pi/4*row['thrust_coefficient'])
                derivative=.5*(1+root)-row['speed_m_s']*math.pi/4*slope/(4*root)
                if derivative<=0:
                    raise ValueError('Incident-speed Ct table gives a nonunique induction-corrected operating point. Check the curve/units; local speed must increase with incident speed throughout H>=D.')
    return c


def lookup(control, speed, field, xp=np):
    if control['mode'] == 'constant_ct':
        ct = control['thrust_coefficient']
        # Exactly Thetis ConstantThrustTurbine: a falsy Cp selects ideal disk.
        value = ct if field == 'thrust_coefficient' else control['power_coefficient'] or .5 * ct * (1 + math.sqrt(1 - ct))
        return xp.full_like(speed, value)
    if control['mode']=='incident_speed_curves':
        x,y=curve_arrays(control,field,xp)
    else:
        x,ct,cp = table_arrays(control,xp)
        y = ct if field=='thrust_coefficient' else cp
    result = xp.interp(speed, x, y)
    # Stock recursive conditional has a strict upper endpoint, including Ct.
    return xp.where((speed < x[0]) | (speed >= x[-1]), 0., result)


def cell_terms(control, diameter, depth, u, v, xp=np):
    """Per unit turbine density; F/rho vector=-K*|u|u, P/rho below."""
    speed = xp.hypot(u, v)
    if control['mode'] in ('incident_speed_table','incident_speed_curves'):
        # Explicit adapter for device curves referenced to undisturbed speed.
        # Solve s=U*alpha(Ct(U),H) at current cell depth, not a fixed Href.
        rows=curve_rows(control,'thrust_coefficient');first=rows[0];last=rows[-1]
        c=math.pi*diameter/(4*depth)
        if bool(xp.any(depth<diameter*(1-1e-10))):
            raise ValueError('Incident-speed device curves require a fully covered rotor (local depth at least D).')
        if xp is not np:
            return incident_gpu(control,diameter,depth,u,v,xp)
        low=xp.full_like(speed,first['speed_m_s']);high=xp.full_like(speed,last['speed_m_s'])
        smin=low*.5*(1+xp.sqrt(1-c*first['thrust_coefficient']))
        smax=high*.5*(1+xp.sqrt(1-c*last['thrust_coefficient']))
        active=(speed>=smin)&(speed<smax)
        for _ in range(44):
            mid=.5*(low+high);ct=lookup(control,mid,'thrust_coefficient',xp)
            local=mid*.5*(1+xp.sqrt(1-c*ct))
            below=local<speed;low=xp.where(below,mid,low);high=xp.where(below,high,mid)
        incident=xp.where(active,.5*(low+high),speed)
        ct=xp.where(active,lookup(control,incident,'thrust_coefficient',xp),0.)
        cp=xp.where(active,lookup(control,incident,'power_coefficient',xp),0.)
        alpha=.5*(1+xp.sqrt(1-c*ct));area=math.pi*diameter**2/4
        return area*ct/(2*alpha**2),.5*area*cp*incident**3,incident,ct,cp
    ct = lookup(control, speed, 'thrust_coefficient', xp)
    area = math.pi * diameter**2 / 4
    radicand = 1 - area * ct / (diameter * depth)
    if bool(xp.any(radicand < 0)):
        raise ValueError('Thetis induction correction is outside its real domain: Ct*A exceeds rotor diameter times local water depth. Check turbine diameter, depth and Ct data.')
    alpha = .5 * (1 + xp.sqrt(radicand))
    incident = speed / alpha
    cp = lookup(control, incident, 'power_coefficient', xp)
    coefficient = area * ct / (2 * alpha**2)
    return coefficient, .5 * area * cp * incident**3, incident, ct, cp


def stop_below_cut_in(diagnostic, control, diameter, depth, speed, xp, *, below_cut_in=None):
    """Whole-rotor cut-in at the lower device-curve bound; no extrapolation.

    Incoming-speed curves require conversion to local speed at current depth.
    Every contributing support cell must be in range before the rotor restarts.
    This is stateless so rejected trials cannot change the operating policy.
    """
    if diagnostic.get('inactive_reason') == 'no_incoming_flow':
        diagnostic['rotational_speed_rpm'] = 0.
    if not diagnostic['active'] or control['mode'] == 'constant_ct':
        return
    if below_cut_in is None:
        first = curve_rows(control, 'thrust_coefficient')[0]
        minimum = first['speed_m_s']
        if control['mode'] in ('incident_speed_table', 'incident_speed_curves'):
            minimum = minimum * .5 * (1 + xp.sqrt(1 - np.pi * diameter * first['thrust_coefficient'] / (4 * depth)))
        below_cut_in = bool(xp.any(speed < minimum))
    if below_cut_in:
        diagnostic.update(active=False, inactive_reason='below_cut_in_speed',
            rotational_speed_rpm=0., thrust_coefficient=0., power_coefficient=0.,
            force_per_density_m4_s2=0., useful_power_per_density_m5_s3=0.,
            hydraulic_power_removed_per_density_m5_s3=0.)


class ThetisRotor(CoveredRotor):
    def __init__(self, cells, c, backend, **kwargs):
        original = dict(c)
        control = validate_control(c['control'])
        # Thetis uses a global-axis-aligned D-by-D source bump. Placement
        # direction is retained separately for the drawn rotor/wake refinement.
        neutral = dict(c, normal_xy=[1., 0.], ct=0., cp=0., control={'mode': 'constant_ct'})
        super().__init__(cells, neutral, backend, **kwargs)
        self.control = control
        self.original = original
        self.density_integrals = self.weights / ((c['D'] / 2)**2 * 1.45661)
        self.identity.update(method=METHOD, reference_commit=REFERENCE, physical=original,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            force='cell-local -A Ct/(2 alpha(H)^2) density |u|u',
            power='integral .5 rho A Cp(|u|/alpha) (|u|/alpha)^3 density',
            support='axis-aligned D-by-D exponential bump / ((D/2)^2 * 1.45661)',
            wake='resolved SWE only; no analytical reconstruction or parent detection',
            options='upwind correction enabled; support drag and prescribed shear profile disabled',
            direction='flow-directed, including reversed flow; no yaw model')

    def sample(self, q, z, grad):
        xp = self.xp; ids = self.ids; h = q[0, ids]
        u = q[1, ids] / xp.maximum(h, self.dry)
        v = q[2, ids] / xp.maximum(h, self.dry)
        speed = xp.hypot(u, v); w = self.normalized[ids]
        row = self.plane; eta = q[0, row] + z[row]
        surface = eta + grad[4, row] * self.delta[0] + grad[5, row] * self.delta[1]
        bed = z[row] + grad[6, row] * self.delta[0] + grad[7, row] * self.delta[1]
        stats = xp.stack((xp.min(xp.minimum(surface, eta)), xp.max(xp.maximum(bed, z[row])),
                          xp.any(h <= self.dry), xp.sum(w * speed), xp.sum(w * h)))
        if xp is not np: stats = stats.get()
        # Preserve the explicit user-required full coverage/bed-clearance gate.
        # Neutral base arithmetic supplies only its standard diagnostic schema.
        d, _, _ = self._operating_state(tuple(map(float, stats)))
        stop_below_cut_in(d, self.control, self.c['D'], h, speed, xp)
        k = xp.zeros_like(h)
        if d['active']:
            K, power, incident, ct, cp = cell_terms(self.control, self.c['D'], h, u, v, xp)
            density = self.density_integrals[ids]
            k = K * density / self.area[ids]
            values = xp.stack((xp.sum(K * density * speed**2), xp.sum(K * density * speed**3),
                xp.sum(power * density), xp.sum(w * incident), xp.sum(w * ct), xp.sum(w * cp)))
            if xp is not np: values = values.get()
            F, work, P, incoming, ct_mean, cp_mean = map(float, values)
            d.update(force_per_density_m4_s2=F, hydraulic_power_removed_per_density_m5_s3=work,
                     useful_power_per_density_m5_s3=P, upstream_velocity_m_s=incoming,
                     thrust_coefficient=ct_mean, power_coefficient=cp_mean)
        d.update(method=METHOD, control_mode=self.control['mode'],
            performance_model=self.control.get('model_name', 'Thetis constant coefficients'),
            requested_Ct=self.control.get('thrust_coefficient'), requested_Cp=self.control.get('power_coefficient'),
            power_basis='Thetis rotor mechanical power; generator losses excluded',
            coefficient_basis=('Ct and Cp: incident-speed device curves, implicit current-depth induction adapter' if self.control['mode'] in ('incident_speed_table','incident_speed_curves') else 'Ct: local depth-mean speed; Cp: self-induction-corrected speed; footprint-weighted output'),
            reconstruction_active=False, active_upstream_wakes=0,
            electrical_efficiency=None, electrical_power_W=None)
        return d, k, u, v


class ThetisArray:
    def __init__(self, cells, rows, backend):
        geometry = GeometryIndex(cells)
        from .river_turbine_yaw import FixedAxisRotor
        from .river_turbine_yaw_legacy import FixedAxisRotor as LegacyFixedAxisRotor
        def rotor_type(c):
            if c.get('orientation',{}).get('mode')!='fixed_axis':return ThetisRotor
            # Existing projects explicitly use the incoming-NORMAL convention.
            # Dou uses its qualified incident-MAGNITUDE/device response instead.
            # A new estimator must not silently reinterpret saved device curves.
            return LegacyFixedAxisRotor if c.get('power_estimation',{'method':'thetis'})=={'method':'thetis'} else FixedAxisRotor
        self.rotors = tuple(rotor_type(c)(cells, c, backend, geometry=geometry) for c in rows)
        from .river_turbine_power import PowerEstimator
        self.power_estimator = PowerEstimator(self.rotors, geometry)
        self.identity = dict(method=METHOD, reference_commit=REFERENCE,
                             rotors=[r.identity for r in self.rotors],
                             coupling='all turbine sinks on the same resolved field; no analytical wake overlay')
        self.identity['power_estimation'] = self.power_estimator.identity_setting

    def _raw_samples(self,q,z,grad):
        from .turbine_constant_batch import eligible,ConstantRotorBatch
        from .turbine_yaw_batch import eligible as yaw_eligible,YawRotorBatch
        batched={}
        for name,predicate,kind in (('constant',eligible,ConstantRotorBatch),('yaw',yaw_eligible,YawRotorBatch)):
            selected=tuple(i for i,r in enumerate(self.rotors) if predicate(r))
            if not selected:continue
            # Identity guards against replacement at the same array indices.
            key=tuple(id(self.rotors[i]) for i in selected)
            if getattr(self,'_'+name+'_selection',None)!=key:
                setattr(self,'_'+name+'_selection',key)
                setattr(self,'_'+name+'_batch',kind(tuple(self.rotors[i] for i in selected)))
            batched.update(zip(selected,getattr(self,'_'+name+'_batch').evaluate(q,z,grad)))
        return [batched[i] if i in batched else r.sample(q,z,grad) for i,r in enumerate(self.rotors)]

    def sample(self, q, z, grad):
        if self.power_estimator.combined:return self._combined(q,z,grad)[0]
        return self.power_estimator.correct([v[0] for v in self._raw_samples(q,z,grad)], q, z)

    def _combined(self,q,z,grad):
        raw=self._raw_samples(q,z,grad);off=set()
        # Propagate operation through the acyclic wake graph at a frozen state.
        # Re-evaluate receivers: removing a stopped upstream rotor may restart
        # a farther downstream rotor. No ordering dependence or saved hysteresis.
        for _ in range(len(raw)+1):
            rows=self.power_estimator.correct([dict(v[0]) for v in raw],q,z,disabled_sources=off)
            new={j for j,s in enumerate(rows) if s.get('inactive_reason')=='below_cut_in_speed'}
            if new==off:return rows,raw
            off=new
        raise RuntimeError('Operating-state graph did not settle.')

    def apply(self, q, z, grad, rate):
        if self.power_estimator.combined:
            rows,raw=self._combined(q,z,grad)
            for rotor,s,(_,k,u,v) in zip(self.rotors,rows,raw):
                if not s['active']:continue
                speed=rotor.xp.hypot(u,v)
                rate[1,rotor.ids]-=k*speed*u;rate[2,rotor.ids]-=k*speed*v
            return rows
        raw=self._raw_samples(q,z,grad)
        for rotor,(d,k,u,v) in zip(self.rotors,raw):
            speed=rotor.xp.hypot(u,v)
            rate[1,rotor.ids]-=k*speed*u;rate[2,rotor.ids]-=k*speed*v
        return [v[0] for v in raw]

    def low_speed(self,q,z,grad,refresh):
        """Cheap accepted-state screen, shared by HLLC and f-wave owners."""
        if self.power_estimator.combined:
            refresh()
            return [s['turbine_id'] for s in self._combined(q,z,grad)[0] if s.get('inactive_reason') in ('below_cut_in_speed','no_incoming_flow')]
        candidates=[]
        for rotor in self.rotors:
            xp=rotor.xp;ids=rotor.ids;h=q[0,ids];u=q[1,ids];v=q[2,ids]
            speed=(u*rotor.normal[0]+v*rotor.normal[1] if
                   rotor.original.get('orientation',{}).get('mode')=='fixed_axis' else xp.hypot(u,v))
            first=0. if rotor.control['mode']=='constant_ct' else curve_rows(rotor.control,'thrust_coefficient')[0]['speed_m_s']
            if getattr(rotor,'device_yaw',False):first=max(first,rotor.device_setting['speed_m_s'][0])
            if bool(xp.any(speed<=max(first,1e-10)*xp.maximum(h,rotor.dry))):candidates.append(rotor)
        if not candidates:return []
        refresh()
        return [d['turbine_id'] for r in candidates for d in [r.sample(q,z,grad)[0]]
                if d.get('inactive_reason') in ('below_cut_in_speed','no_incoming_flow')]
