"""Identified Sun/van Rijn reconstruction, research only, no public admission.

Equilibrium SUBGRID dunes affect resistance, not geometric bed elevation or
sediment inventory. R_b=h is the explicit depth-averaged wide-channel mapping.
The Shields curve for dune mobility is distinct from JC's 0.047 slope threshold.
"""
from dataclasses import dataclass,asdict
from numbers import Real
from pathlib import Path
from types import MethodType
import numpy as np
from .jiang_cao_exchange import ExchangeMaterial
from .jiang_cao_secondary import JiangCaoSecondary
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='roughness.sun2015_van_rijn1984_reconstruction_v1'
FIELDS=('grain_chezy','grain_ustar_squared','transport_stage','height_m','length_m',
        'grain_ks_m','form_ks_m','total_ks_m','chezy','manning','total_ustar_squared')


def critical_shields(dstar):
    """van Rijn five-branch Shields approximation, NOT JC capacity threshold."""
    if not np.isfinite(dstar) or dstar<1:raise ValueError('Noncohesive dune closure requires Dstar >= 1')
    if dstar<=4:return .24/dstar
    if dstar<=10:return .14*dstar**(-.64)
    if dstar<=20:return .04*dstar**(-.1)
    if dstar<=150:return .013*dstar**.29
    return .055


@dataclass(frozen=True)
class RoughnessParameters:
    material: ExchangeMaterial
    d90_m: float  # required: never inferred as 1.5 D50
    diameter_definition: str='d90'

    def __post_init__(self):
        if not isinstance(self.material,ExchangeMaterial):raise TypeError('Explicit ExchangeMaterial required')
        d=self.d90_m
        if isinstance(d,(bool,np.bool_)) or not isinstance(d,Real) or not np.isfinite(d) or d<self.material.capacity_material.d50_m:
            raise ValueError('Explicit finite D90 >= D50 required')
        if self.diameter_definition!='d90':raise ValueError('Roughness requires D90, not D50/D84/Dmean under a different label')
        if not np.isfinite(self.dstar):raise ValueError('Finite Dstar required')
        critical_shields(self.dstar)

    @property
    def dstar(self):
        p=self.material.capacity_material
        return p.d50_m*((p.solid_density_kg_m3/p.water_density_kg_m3-1)*p.gravity_m_s2/
                       self.material.kinematic_viscosity_m2_s**2)**(1/3)


class RoughnessOperator:
    def __init__(self,backend,parameters,*,bedforms_enabled=True,threads=4):
        if not isinstance(parameters,RoughnessParameters):raise TypeError('Explicit RoughnessParameters required')
        if type(bedforms_enabled) is not bool:raise ValueError('Explicit boolean bedform switch required')
        if backend=='gpu':import cupy as xp
        elif backend=='cpu':xp=np
        else:raise ValueError('Explicit cpu or gpu required; no fallback')
        self.xp,self.parameters,self.bedforms_enabled=xp,parameters,bedforms_enabled
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=('jc_roughness',))
        p=parameters.material.capacity_material
        tc=critical_shields(parameters.dstar)
        uc2=p.gravity_m_s2*(p.solid_density_kg_m3/p.water_density_kg_m3-1)*p.d50_m*tc
        self.wire=xp.asarray([p.d50_m,parameters.d90_m,p.gravity_m_s2,uc2])
        self.identity=dict(method=METHOD,parameters=asdict(parameters),bedforms_enabled=bedforms_enabled,
            dstar=parameters.dstar,bedform_critical_shields=tc,critical_ustar_squared=uc2,
            runtime=self.engine.identity,wrapper_sha256=digest(__file__),fields=FIELDS,
            stage='grain-only Chezy; clear-water density contrast; Shields van Rijn five-branch curve',
            resistance='Sun Eq1d; instantaneous equilibrium dunes; R_b=h; no sidewall correction or ripple/antidune model',
            capacity='JC total resistance; no extra form-drag/Bagnold factor; derivatives conditional on supplied roughness',
            public_support=False)

    def evaluate(self,q):
        xp=self.xp;q=xp.ascontiguousarray(xp.asarray(q,dtype=xp.float64))
        if q.ndim!=2 or q.shape[0]!=3 or q.shape[1]==0 or q.shape[1]>np.iinfo(np.int32).max:
            raise ValueError('Nonempty q(3,n) required')
        p=self.parameters;dry=p.material.capacity_material.dry_depth_m
        if not bool(xp.all(xp.isfinite(q))) or bool(xp.any(q[0]<=max(dry,p.d90_m/4))):
            raise ValueError('Fully wet finite state and 12 h > 3 D90 required; no hidden depth or roughness floor')
        out=xp.empty((len(FIELDS),q.shape[1]))
        self.engine.launch('jc_roughness',q.shape[1],q.shape[1],int(self.bedforms_enabled),self.wire,q,out)
        if not bool(xp.all(xp.isfinite(out))) or bool(xp.any(out[8:10]<=0)):
            raise BendTrialRejected('Nonpositive/nonfinite total Chezy resistance; outside log-law range, no clipping')
        return out


class JiangCaoRoughness(JiangCaoSecondary):
    def __init__(self,hydraulics,material,*,roughness_parameters,bedforms_enabled=True,**kwargs):
        if roughness_parameters.material!=material:raise ValueError('Roughness and transport material identities must match')
        if 'manning' in kwargs:raise ValueError('Manning is resolved by Sun/van Rijn, not a second prescribed input')
        op=RoughnessOperator(hydraulics.backend,roughness_parameters,bedforms_enabled=bedforms_enabled,
                             threads=hydraulics.physics.flow.threads)
        initial=op.evaluate(hydraulics.q)
        super().__init__(hydraulics,material,manning=initial[9],**kwargs)
        self.roughness=op
        def friction(owner,q,dt,*args):self.apply_friction(q,dt)
        hydraulics.physics.friction=MethodType(friction,hydraulics.physics)
        self.identity.pop('manning_sha256')
        self.identity=dict(self.identity,method='coupled.jc2026_p15_roughness199_v1',roughness=op.identity,
            wrapper_roughness_sha256=digest(__file__),grain_policy='one bulk material; explicit measured D50 and D90',
            friction='midpoint-predicted resistance with exact frozen-coefficient quadratic drag',
            resistance_refresh='source predictor, target, hydraulic friction half-stages; secondary uses same-stage capacity Chezy',
            limits='fully wet fixed mesh, one grain MF1, mild curvature, alluvial bed; no collapse/restart/AMR')

    def target(self,s):
        q=self.mesh.capacity_state(s[:3]);r=self.roughness.evaluate(q)
        return self.exchange.capacity_target(s,self.mesh.capacity_gradient(s[5],s[:3]),r[9],capacity_q=q)

    def apply_friction(self,q,dt):
        dt=self.exchange._dt(dt)
        if not dt:return
        # For variable K(U)=g/C(U)^2/h, freezing K at the initial velocity is
        # first order. Predict a half-step, then use its K in the exact
        # quadratic drag formula evaluated with the ORIGINAL speed.
        n0=self.roughness.evaluate(q)[9];pred=q.copy()
        gravity=self.exchange.material.capacity_material.gravity_m_s2
        self.engine.launch('jc_manning',self.n,self.n,.5*dt,gravity,n0,pred)
        nm=self.roughness.evaluate(pred)[9]
        self.engine.launch('jc_manning',self.n,self.n,dt,gravity,nm,q)

    def relative_discharge(self,s,cap):
        if not self.secondary_enabled:return super().relative_discharge(s,cap)
        velocity=s[1:3]/s[0];gradient=self.xp.vstack([self.mesh.gradient(v) for v in velocity])
        # Capacity already contains the freshly resolved SAME-stage Chezy.
        # Do not read self.manning: it is only the constructor's initial value.
        n=s[0]**(1/6)/cap.field('chezy')
        self.secondary_fields=self.secondary.evaluate(s[:3],gradient,n,cap.values,
            self.exchange.material.capacity_material.gravity_m_s2)
        return self.secondary_fields[7:].copy()
