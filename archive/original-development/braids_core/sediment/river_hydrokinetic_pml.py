"""Named source-complete algebraic 2D mixing for the Nash reconstruction.

nu_t = a u_* H (Olbert et al., 2016/2017, Eq. 1); symmetric stress as
Nash et al. (2015), Eq. 2/9.  No turbine-generated k source is present in
this published model. The coefficient must be explicit: the turbine paper
does not report it. CPU and GPU run the same native formulas.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
from pathlib import Path
from types import MethodType

from .bend_methods_v1 import digest
from .river_hydrokinetic_stress import FullStressEngine


@dataclass(frozen=True)
class PrandtlMixing:
    coefficient:float
    chezy_m_half_s:float
    no_slip_walls:bool=True

    def validate(self):
        if not (math.isfinite(self.coefficient) and self.coefficient>0 and
                math.isfinite(self.chezy_m_half_s) and self.chezy_m_half_s>0):
            raise ValueError('Positive explicit Prandtl coefficient and Chezy required')
        return self


class PrandtlEngine(FullStressEngine):
    SOURCE=Path(__file__).with_suffix('.h')
    NAMES=('hkt_pml_nu','hkt_pml_walls')


def install(owner,config):
    config.validate();p=owner.physics;xp=p.xp
    if owner.time or owner.steps or p._open or float(p.nu)!=0:
        raise ValueError('Install PML on a fresh owner with zero separate viscosity')
    if hasattr(owner,'hydrokinetic_turbulence'):
        raise ValueError('PML and transported turbine turbulence cannot both own momentum mixing')
    interior=FullStressEngine(owner.backend,p.flow.threads)
    algebra=PrandtlEngine(owner.backend,p.flow.threads)
    p.hkt_pml_nu=xp.zeros(p.n);p.hkt_pml_grad=xp.zeros((10,p.n))
    p.hkt_pml_face=xp.zeros((2,p.nf));p.hkt_pml_work=xp.zeros((3,p.n))
    p.hkt_pml_unit=xp.ones(p.n)
    previous=p.add_momentum
    def add_momentum(self,q,water,rate):
        previous(q,water,rate);g=p.g
        algebra.launch('hkt_pml_nu',p.n,p.n,float(p.parameters.dry),
            float(p.parameters.gravity),float(config.chezy_m_half_s),float(config.coefficient),q,p.hkt_pml_nu)
        p._gradients(q,p.W,p.hkt_pml_grad)
        interior.launch('hkt_full_strain_faces',p.nf,p.nf,p.n,float(p.parameters.dry),
            q,p.z0,p.hkt_pml_grad,p.hkt_pml_nu,g['owner'],g['neighbour'],g['normal'],
            g['distance'],g['xy'],p.hkt_pml_face)
        if config.no_slip_walls:
            algebra.launch('hkt_pml_walls',p.nf,p.nf,p.n,float(p.parameters.dry),q,p.hkt_pml_nu,
                g['owner'],g['neighbour'],g['bid'],g['normal'],g['distance'],p.hkt_pml_face)
        p.hkt_pml_work.fill(0.)
        p.flow.launch('flow_momentum_gather',p.n,p.n,p.nf,p.hkt_pml_unit,p.hkt_pml_face,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],p.hkt_pml_work)
        rate[1:]+=p.hkt_pml_work[1:]
    p.add_momentum=MethodType(add_momentum,p)
    identity=dict(method='turbulence.olbert_prandtl_depth_average_v1',configuration=asdict(config),
        formula='nu_t=a*sqrt(g)/C*|U|*H',stress='h nu_t (grad U + grad U.T)',
        coefficient_provenance='explicit reconstruction bracket; not specified by Nash 2015',
        references=['10.3390/en81212380','10.1080/00221686.2016.1252799'],
        source_sha256=digest(__file__),algebra_runtime=algebra.identity,stress_runtime=interior.identity,
        turbine_tke_source=False,desktop_admission=False)
    owner.identity['hydrokinetic_pml']=identity;p.identity['hydrokinetic_pml']=identity
    return owner
