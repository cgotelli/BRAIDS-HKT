"""Explicit simple hydraulic/bedload contract.

The simple hydraulic route may retain the already selected conservative
horizontal-viscosity operator.  It still disables secondary flow, suspension,
bedforms and every morphodynamic auxiliary.  Keeping the existing momentum
callback is important: :mod:`shoreline_viscosity` is installed earlier in the
common owner lifecycle and used to be silently replaced here.
"""
from types import MethodType
from .bend_methods_v1 import digest


def install(a,mode):
    if mode not in ('hydrodynamics','bedload'):raise ValueError('Simple river physics mode required')
    p=a.physics
    if a.time or a.steps or p._open:raise ValueError('Select physics before execution')
    previous=p.local
    def local(owner,*args):
        # Preserve the shared hydraulic/roughness input mapping, but disable
        # ALL secondary-current outputs, including source/boundary equilibrium.
        # Betac alone only switches momentum dispersion, not secondary flow.
        out=previous(*args)
        out[:9]=0.;out[10:]=0.
        return out
    p.local=MethodType(local,p)
    # ``shoreline_viscosity.install`` owns the conservative face operator and
    # wraps the previous momentum callback before this selector is reached.
    # Preserve that wrapper only when it records a nonzero selected viscosity;
    # otherwise retain the historical no-auxiliary no-op exactly.
    viscosity=p.identity.get('conservative_viscosity',{})
    viscosity_on=float(viscosity.get('coefficient_m2_s',0.))>0.
    if not viscosity_on:
        def momentum(owner,q,water,rate):pass
        p.add_momentum=MethodType(momentum,p)
    a.morphology=mode=='bedload'
    a.identity['physics_mode']=mode
    p.identity['simple_physics']=dict(mode=mode,source_sha256=digest(__file__),secondary_flow=False,
        viscosity=viscosity_on,suspension=False,bedforms=False,banks=False,collapse=False,
        bedload=mode=='bedload',exner=mode=='bedload')
    return a
