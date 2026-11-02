"""Explicit auxiliary-process OFF contract for hydraulic/bedload-only runs."""
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
    def momentum(owner,q,water,rate):pass
    p.add_momentum=MethodType(momentum,p)
    a.morphology=mode=='bedload'
    a.identity['physics_mode']=mode
    p.identity['simple_physics']=dict(mode=mode,source_sha256=digest(__file__),secondary_flow=False,
        viscosity=False,suspension=False,bedforms=False,banks=False,collapse=False,
        bedload=mode=='bedload',exner=mode=='bedload')
    return a
