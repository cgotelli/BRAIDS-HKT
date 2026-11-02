"""Explicitly named shallow extension of the JC research composition.

NOT the authors' wet/dry implementation, a public model, or an AMR admission.
The continuation changes constitutive coefficients, never the physical depth.
All cell/face arithmetic stays on the explicitly selected CPU or GPU backend.
"""
from dataclasses import dataclass, asdict
from types import MethodType
import numpy as np
from .bend_methods_v1 import BendTrialRejected, digest
from .jiang_cao_capacity import CapacityEvaluation
from .jiang_cao_coupled import JiangCaoCoupled
from .jiang_cao_secondary import SecondaryOperator, SecondaryApplicabilityError
from .jiang_cao_roughness import RoughnessOperator
from .jiang_cao_collapse import JiangCaoCollapse, ReposeCollapse
from .jiang_cao_shoreline import ShorelineOperators


@dataclass(frozen=True)
class ShallowTreatment:
    """Research assumption: matching depth 10 D90, full activation at 20 D90."""
    reference_submergence: float = 10.
    method: str = 'shallow.matched_manning_deposition201_v1'
    independent_bedload = False

    def __post_init__(self):
        if (isinstance(self.reference_submergence, bool) or
            not np.isfinite(self.reference_submergence) or self.reference_submergence < 10 or
            self.method != 'shallow.matched_manning_deposition201_v1'):
            raise ValueError('Named shallow treatment and reference submergence >= 10 required')

    def weight(self, h, d90, xp):
        hr = self.reference_submergence*d90
        t = xp.clip((h-hr)/hr, 0., 1.)
        return t*t*(3.-2.*t), 6.*t*(1.-t)/hr

    def bedload_weight(self, h, d90, dry, xp):
        return self.weight(h, d90, xp)

    def resistance(self, op, q):
        xp = op.xp; h = q[0]; p = op.parameters
        if not bool(xp.all(xp.isfinite(q))) or bool(xp.any(h < 0)):
            raise ValueError('Finite nonnegative physical depth required')
        hr = self.reference_submergence*p.d90_m
        if bool(xp.all(h >= hr)):return op.evaluate(q)[9]
        wet = h > p.material.capacity_material.dry_depth_m
        ref = q.copy(); ref[0] = xp.maximum(h, hr)
        ref[1:] = xp.where(h >= hr, q[1:], xp.where(wet, q[1:]/xp.where(wet, h, 1.)*ref[0], 0.))
        # Same speed at reference depth; only n is continued into shallow cells.
        # No reference depth enters the state, capacity shear, flux or volume.
        return op.evaluate(ref)[9]


@dataclass(frozen=True)
class GrainScaleBedloadTreatment(ShallowTreatment):
    """Named BRAIDS continuation, not a published Jiang-Cao equation.

    New entrainment requires a submerged grain: zero at h <= D90, smoothstep
    from D90 to 2D90, unchanged capacity at h >= 2D90. Existing mobile sediment
    still transports/deposits below D90. This does not regularize unbounded velocities or
    permit packing, invalid secondary profiles, or zero-depth phase fluxes.
    Resistance and secondary applicability retain the separately qualified
    10D90/20D90 continuation. There is no calibrated cutoff parameter.
    """
    method: str = 'shallow.submerged_grain_bedload_separate_profile209_v2'
    independent_bedload = True

    def __post_init__(self):
        if (type(self.reference_submergence) not in (int,float) or
            self.reference_submergence != 10. or
            self.method != 'shallow.submerged_grain_bedload_separate_profile209_v2'):
            raise ValueError('Named grain-scale bedload treatment with fixed 10D90 profile reference required')

    def bedload_weight(self, h, d90, dry, xp):
        if not np.isfinite(d90) or not 0 <= dry < d90:
            raise ValueError('D90 must exceed the hydraulic dry tolerance')
        t = xp.clip((h-d90)/d90, 0., 1.)
        return t*t*(3.-2.*t), 6.*t*(1.-t)/d90


class JiangCaoShoreline(JiangCaoCollapse):
    def __init__(self, hydraulics, material, *, roughness_parameters,
                 shallow=ShallowTreatment(), secondary_enabled=True,
                 collapse_enabled=True, bedforms_enabled=True, direction_resolution=None,
                 secondary_adjustment=None, **kwargs):
        if roughness_parameters.material != material:
            raise ValueError('Roughness and transport bulk material identities must match')
        if not isinstance(shallow, ShallowTreatment):raise TypeError('Explicit shallow treatment required')
        if any(type(v) is not bool for v in (secondary_enabled, collapse_enabled, bedforms_enabled)):
            raise ValueError('Explicit boolean process switches required')
        self.shallow = shallow
        self.roughness = RoughnessOperator(hydraulics.backend, roughness_parameters,
            bedforms_enabled=bedforms_enabled, threads=hydraulics.physics.flow.threads)
        n = shallow.resistance(self.roughness, hydraulics.q)
        JiangCaoCoupled.__init__(self, hydraulics, material, manning=n, **kwargs)
        self.front = ShorelineOperators(self.mesh, self.backend, threads=hydraulics.physics.flow.threads)
        self.secondary_enabled = secondary_enabled
        self.secondary = SecondaryOperator(self.backend, self.xp, hydraulics.physics.flow.threads)
        self.direction = None
        if direction_resolution is not None:
            from .jiang_cao_direction import DirectionOperator, METHOD
            if direction_resolution != METHOD:
                raise ValueError('Explicit named research direction-resolution method required')
            self.direction = DirectionOperator(self.mesh, self.backend, threads=hydraulics.physics.flow.threads)
        self.adjustment = None
        if secondary_adjustment is not None:
            from .jiang_cao_adjustment import AdjustmentOperator, METHOD as ADJUSTMENT
            from .jiang_cao_direction import DirectionOperator
            if secondary_adjustment != ADJUSTMENT or direction_resolution is not None or not secondary_enabled:
                raise ValueError('Named finite adjustment requires secondary ON and no algebraic resolution weighting')
            self.direction = DirectionOperator(self.mesh,self.backend,threads=hydraulics.physics.flow.threads)
            self.adjustment = AdjustmentOperator(self)
            self.state = self.xp.vstack((self.state,self.xp.zeros(self.n)))
        self.collapse_enabled = collapse_enabled
        self.collapse = ReposeCollapse(self.mesh, material, self.backend, threads=hydraulics.physics.flow.threads)
        self.collapse_integral = self.xp.zeros(self.nf); self.collapse_gross = 0.
        self._pending = self.xp.zeros(self.nf); self._gross = 0.
        self._iterations = 0; self._virtual_iterations = 0
        def friction(owner, q, dt, *args):self.apply_friction(q, dt)
        hydraulics.physics.friction = MethodType(friction, hydraulics.physics)
        self.identity.pop('manning_sha256')
        self.identity.update(method='coupled.jc2026_p15_shoreline201_v1',
            wrapper_shoreline_sha256=digest(__file__), shallow=asdict(shallow),
            roughness=self.roughness.identity, secondary=self.secondary.identity,
            collapse=self.collapse.identity, phase_transport=self.front.identity,
            secondary_enabled=secondary_enabled, collapse_enabled=collapse_enabled,
            activation='smoothstep capacity and secondary covariance/drift from h_ref to 2*h_ref; zero new entrainment below h_ref',
            shallow_resistance='Manning n matched at h_ref at same speed; use actual depth for drag',
            rest_direction='secondary direction unresolved when Fr <= sqrt(float64 epsilon); zero secondary only, never zero physical momentum',
            drying='deposit remaining mobile solid at hydraulic dry tolerance; conserve solid, free surface and water+pore water',
            limits='research fixed wet/dry saturated alluvial mesh; one grain MF1; mild curvature; no restart/AMR/public admission')
        if self.direction is not None:
            self.identity['direction_resolution'] = self.direction.identity
        if self.adjustment is not None:
            self.identity['secondary_adjustment'] = self.adjustment.identity
        if shallow.independent_bedload:
            self.identity['activation'] = ('no new entrainment below D90; bedload smoothstep from D90 to 2D90; '
                'unchanged capacity above 2D90; separate secondary covariance/steering activation '
                'from 10D90 to 20D90; retain gravitational steering below secondary activation')

    def source(self,s,dt):
        if self.adjustment is None:return super().source(s,dt)
        out=s.copy();out[:6]=super().source(s[:6],dt)
        # Exchange changes h, not the advected intensity of remaining water.
        out[6]=s[6]*out[0]/self.xp.where(s[0]>self.dry,s[0],1.)
        midpoint=.5*(s+out)
        return self.adjustment.relax(out,dt,midpoint)

    @property
    def dry(self):return self.exchange.material.capacity_material.dry_depth_m

    def _validate(self, s):
        xp = self.xp; h, m = s[0], s[3]
        if (not bool(xp.all(xp.isfinite(s))) or
            bool(xp.any((h < 0)|(m < 0)|(s[4] < 0)|((h > 0)&(m >= self.B*h))|((h == 0)&(m != 0)))) or
            bool(xp.any((h == 0)&((s[1] != 0)|(s[2] != 0))))):
            raise BendTrialRejected('Wet/dry phase bounds violated; no inventory clipping')
        if s.shape[0]==7 and bool(xp.any((h==0)&(s[6]!=0))):
            raise BendTrialRejected('Zero-depth cell carries secondary intensity')

    def clean_dry(self, s):
        """Private-stage conservative terminal deposition, not deletion."""
        xp = self.xp; out = s.copy(); dry = out[0] <= self.dry
        deposit = xp.where(dry, out[3], 0.)
        out[0] -= deposit/self.B; out[5] += deposit/self.B
        out[4] += deposit; out[3] -= deposit
        out[1:3, dry] = 0.
        if out.shape[0]==7:out[6,dry]=0.
        self._validate(out)
        return out

    def gradient(self, v, active, *, limited=False):
        # Retain the frozen deep-water stencil when every neighbour is wet.
        if bool(self.xp.all(active)):return self.mesh.gradient(v, limited=limited)
        return self.front.gradient(v, active, limited=limited)

    def target(self, s):
        xp = self.xp; h = s[0]; d90 = self.roughness.parameters.d90_m
        w, dw = self.shallow.bedload_weight(h, d90, self.dry, xp); active = w > 0
        q = self.mesh.capacity_state(s[:3])
        n = self.shallow.resistance(self.roughness, q)
        values = xp.zeros((23, self.n)); status = xp.ones(self.n, dtype=xp.int32)
        values[9] = xp.where(q[0] > 0, q[0]**(1/6)/n, 0.)
        if self.shallow.independent_bedload:
            # At exact rest there is no shear-driven entrainment and no flow
            # direction. Do not evaluate directional slope factors on an
            # initial unstable bank before its conservative collapse step.
            # No small-speed cutoff: every nonzero momentum still uses the law.
            at_rest = active & (xp.hypot(s[1],s[2]) == 0.)
            status[at_rest] = 2
            active &= ~at_rest
        if bool(xp.any(active)):
            c = s[3,active]/h[active]
            r = self.exchange.capacity.evaluate(q[:,active], self.mesh.capacity_gradient(s[5],s[:3])[:,active],
                                                n[active], concentration=c)
            v = r.values.copy(); raw = v[:3].copy(); ww = w[active]
            v[:3] *= ww; v[12:22] *= ww
            for row, component in ((13,0),(16,1),(19,2)):v[row] += dw[active]*raw[component]
            values[:,active] = v; status[active] = r.status
        discharge = xp.hypot(q[1],q[2])
        ce = values[0]/xp.where(discharge > 0, discharge, 1.)
        if bool(xp.any(~xp.isfinite(ce)|(ce<0.))):
            raise BendTrialRejected('Shallow capacity target must be finite and nonnegative')
        return ce, CapacityEvaluation(values, status)

    def apply_friction(self, q, dt):
        dt = self.exchange._dt(dt)
        if not dt:return
        gravity = self.exchange.material.capacity_material.gravity_m_s2
        def drag(original, n, duration):
            result = original.copy()
            self.front.engine.launch('jc201_manning',self.n,self.n,float(duration),gravity,self.dry,n,result)
            return result
        pred = drag(q, self.shallow.resistance(self.roughness, q), .5*dt)
        q[...] = drag(q, self.shallow.resistance(self.roughness, pred), dt)

    def relative_discharge(self, s, cap):
        xp = self.xp; w,_ = self.shallow.weight(s[0], self.roughness.parameters.d90_m, xp)
        active = w > 0; fields = xp.zeros((9,self.n))
        if bool(xp.any(active)):
            if self.secondary_enabled:
                # Dry/unsupported neighbour velocities are not fictitious zero measurements.
                velocity = s[1:3]/xp.where(s[0] > self.dry, s[0], 1.)
                if self.adjustment is not None:
                    intensity=s[6]/xp.where(s[0]>self.dry,s[0],1.)
                    # Rigid-rotation Jacobian encodes k_eff=I/(U*h), so the
                    # same profile/covariance/drift kernel uses A=I/kappa^2.
                    omega=intensity/xp.where(s[0]>self.dry,s[0],1.)
                    gradient=xp.ascontiguousarray(xp.vstack((xp.zeros(self.n),-omega,omega,xp.zeros(self.n))))
                elif self.direction is None:
                    gradient = xp.vstack([self.gradient(v, active) for v in velocity])
                else:
                    resolved = self.direction.evaluate(velocity, active)
                    gradient = xp.ascontiguousarray(resolved[:4]*resolved[5])
                gravity=self.exchange.material.capacity_material.gravity_m_s2
                # Curvature divides a velocity Jacobian by speed. A direction
                # inferred only from lake-at-rest cancellation is not resolved:
                # kinetic energy below float64 epsilon times g*h is roundoff.
                moving=active&(xp.hypot(velocity[0],velocity[1]) > xp.sqrt(np.finfo(float).eps*gravity*s[0]))
                if bool(xp.any(moving)):
                    n = s[0,moving]**(1/6)/cap.values[9,moving]
                    try:
                        fields[:,moving] = self.secondary.evaluate(s[:3,moving], gradient[:,moving], n,
                            cap.values[:,moving], gravity)
                    except SecondaryApplicabilityError as exc:
                        i=int(xp.flatnonzero(moving)[exc.index]);xy=self.mesh.g['xy'][:,i]
                        exc.args=(f'{exc} Physical mesh cell {i}, centroid ({float(xy[0]):.9g}, {float(xy[1]):.9g}) m; '
                                  f'last accepted time {self.time:.9g} s. Trial state was not accepted.',)
                        raise
            else:
                fields[7:] = JiangCaoCoupled.relative_discharge(self, s, cap)
            fields[3:6] *= w
            fields[7:] *= w
        if self.shallow.independent_bedload:
            # The full secondary drift already includes gravitational steering.
            # Add only its missing fraction, never a second complete slope term.
            gravity_drift = JiangCaoCoupled.relative_discharge(self, s, cap)
            if self.secondary_enabled:
                fields[7:] += (1.-w)*gravity_drift
            else:
                fields[7:] = gravity_drift
        self.secondary_fields = fields
        return fields[7:].copy()

    def momentum_correction(self, s, rate):
        if not self.secondary_enabled:return
        xp = self.xp; g = self.mesh.g; n,nf = self.n,self.nf; wet = s[0] > self.dry
        tensor = xp.ascontiguousarray(s[0]*self.secondary_fields[3:6])
        grad = xp.vstack([self.gradient(t,wet) for t in tensor]); flux = xp.empty((2,nf))
        e = self.secondary.engine
        e.launch('jc_dispersion_faces', nf,n,nf,g['owner'],g['neighbour'],g['kind'],g['length'],
                 g['normal'],g['xy'],g['mid'],tensor,grad,flux)
        o,j = g['owner'],g['neighbour']; common = wet[o]&((j<0)|wet[xp.maximum(j,0)])
        jj=xp.maximum(j,0)
        overlap=xp.minimum(s[0,o]+s[5,o],s[0,jj]+s[5,jj])-xp.maximum(s[5,o],s[5,jj])
        common &= (j<0)|(overlap>self.dry)
        flux[:,~common] = 0.
        e.launch('jc_dispersion_divergence',n,n,nf,g['offset'],g['faces'],g['owner'],g['area'],flux,rate)
        self.dispersion_flux = flux

    def rate(self, s, dt, time, inlet_concentration):
        self._validate(s); a=self.a; xp=self.xp; g=self.mesh.g; n,nf=self.n,self.nf
        a.check(); a.z[...] = s[5]; a.physics.z0[...] = s[5]; a.midpoint.interval_base = s[:3]
        r,Q = a.midpoint.rate(s[:3],dt,time); Q = Q*g['length']
        wet = s[0] > self.dry; c = s[3]/xp.where(s[0] > 0,s[0],1.)
        grad = self.gradient(c,wet,limited=True); cp=self.exchange.material.capacity_material
        self.engine.launch('jc_density_source',n,n,cp.gravity_m_s2,cp.water_density_kg_m3,
                           cp.solid_density_kg_m3,s[:3],c,grad,r)
        _,cap = self.target(s); delta=self.relative_discharge(s,cap); self.momentum_correction(s,r)
        F=xp.empty(nf); T=xp.empty(nf); dm=xp.empty(n)
        dgrad=xp.vstack([self.gradient(v,wet) for v in delta])
        cin=inlet_concentration(time) if callable(inlet_concentration) else inlet_concentration
        cin=xp.ascontiguousarray(xp.broadcast_to(xp.asarray(cin,dtype=xp.float64),(nf,)))
        self.engine.launch('jc_mobile_faces',nf,n,nf,g['owner'],g['neighbour'],g['kind'],g['length'],
            g['normal'],g['xy'],g['mid'],Q,c,grad,delta,dgrad,cin,F,T)
        # Relative bed drift needs a common wetted face. Hydraulic wetting flux
        # remains untouched; at that face use its conservative donor phase flux.
        o,j=g['owner'],g['neighbour']; jj=xp.maximum(j,0)
        overlap=xp.minimum(s[0,o]+s[5,o],s[0,jj]+s[5,jj])-xp.maximum(s[5,o],s[5,jj])
        common=(j>=0)&wet[o]&wet[jj]&(overlap>self.dry)
        donor=xp.where(Q>=0,o,jj); F=xp.where(common,F,Q*c[donor])
        _,_,F,theta=self.front.flux(s[0],s[3],Q,F,dt,packing=self.B,inlet_concentration=cin)
        self.engine.launch('jc_mobile_divergence',n,n,g['offset'],g['faces'],g['owner'],g['area'],F,dm)
        self.phase_minimum_theta=min(self.phase_minimum_theta,float(xp.min(theta)))
        out=xp.zeros_like(s);out[:3]=r;out[3]=dm
        if self.adjustment is not None:out[6]=self.adjustment.rate(s,Q,r[0],dt)
        return out,Q,F

    def step(self,dt,*,inlet_concentration=0.):
        dt=self.exchange._dt(dt)
        if dt<=0:raise ValueError('Positive coupled timestep required')
        a=self.a;xp=self.xp;oldq=a.q.copy();oldz=a.z.copy();saved=a.snapshot()
        self._pending.fill(0.);self._gross=0.;self._iterations=0;self._virtual_iterations=0
        self.phase_minimum_theta=1.;a.midpoint.minimum_theta=1.;a.midpoint.frequency_peak=0.
        try:
            s=self.clean_dry(self.source(self.clean_dry(self.state),.5*dt));a.physics.friction(s[:3],.5*dt)
            r,Q0,F0=self.rate(s,dt,self.time,inlet_concentration);s1=self.clean_dry(s+dt*r)
            r,Q1,F1=self.rate(s1,dt,self.time+dt,inlet_concentration)
            end=self.clean_dry(.5*s+.5*(s1+dt*r));a.physics.friction(end[:3],.5*dt)
            end=self.clean_dry(self.source(end,.5*dt));self._validate(end)
            outside=self.mesh.g['neighbour']<0;Q=.5*(Q0+Q1);F=.5*(F0+F1)
            si=self.solid_integral+dt*F*outside;wi=self.water_integral+dt*(Q-F)*outside
            solid,water=self.totals(end);sr=solid+float(xp.sum(si))-self.initial_solid
            wr=water+float(xp.sum(wi))-self.initial_water
            if max(abs(sr),abs(wr))>1e-10*max(1.,abs(self.initial_solid),abs(self.initial_water)):
                raise BendTrialRejected('Coupled wet/dry phase conservation failed')
            a.check();self.state=end;self.time+=dt;self.steps+=1
            self.solid_integral=si;self.water_integral=wi;self.collapse_integral+=self._pending;self.collapse_gross+=self._gross
            a.q[...]=end[:3];a.z[...]=end[5];a.physics.z0[...]=end[5]
            self.last=dict(time_s=self.time,solid_residual_m3=sr,water_plus_pore_residual_m3=wr,
                hydraulic_cfl=dt*a.midpoint.frequency_peak,phase_minimum_theta=self.phase_minimum_theta,
                collapse_sweeps=self._iterations,virtual_collapse_sweeps=self._virtual_iterations,
                cumulative_collapse_gross_solid_m3=self.collapse_gross,dt_s=float(dt))
            return dict(self.last)
        except BaseException:
            a.q[...]=oldq;a.z[...]=oldz;a.restore(saved)
            self._pending.fill(0.);self._gross=0.;self._iterations=0;self._virtual_iterations=0
            raise
