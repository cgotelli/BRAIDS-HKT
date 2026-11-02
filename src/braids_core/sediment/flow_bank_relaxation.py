"""Ordered conservative bank supply and relaxation for shoreline research.

Transport -> FLOW dry-bank sharing -> slope redistribution. Each transfer
has a separate transactional ledger; only transport enters boundary budgets.
The caller retains the explicit CPU/GPU and hydraulic integrator choice.
"""
from types import MethodType
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from .flow_dry_bank import install_shoreline_bank
from .flow_slope_relaxation import install_shoreline_relaxation


def install_shoreline_bank_relaxation(solver,xy,*,theta=.5,slope=.5,
                                     relaxation=True,hmaxth=0.,sedthr=.001):
    p=solver.physics;xp=p.xp
    if p.time or p.steps or p._open or any(hasattr(p,n) for n in
            ('bank_engine','relaxation_engine','transport_cumulative_mass')):
        raise ValueError('Compose bank and relaxation once, before execution')
    if not all(np.isfinite(v) for v in (theta,slope,hmaxth,sedthr)) or not (
            0<=theta<=1 and slope>0 and hmaxth>=0 and sedthr>=.001):
        raise ValueError('Invalid bank/relaxation controls')
    if not isinstance(relaxation,bool):raise ValueError('Explicit relaxation toggle required')
    initial_finish=p.finish;initial_commit=p.commit
    p.transport_trial_mass=xp.zeros(p.n);p.transport_cumulative_mass=xp.zeros(p.n)
    # Integrated gross transfer, rather than positive part of the final net
    # field: transfers that later reverse must not disappear from this budget.
    p.process_gross_mass=xp.zeros(2)
    def transport_finish(owner,qold,qnew,water,*,morphology=True):
        result=initial_finish(qold,qnew,water,morphology=morphology)
        owner.transport_trial_mass[:]=1590.*(owner.ztrial-owner.z0)*owner.g['area']
        return result
    def transport_commit(owner):
        initial_commit();owner.transport_cumulative_mass+=owner.transport_trial_mass
    p.finish=MethodType(transport_finish,p);p.commit=MethodType(transport_commit,p)
    install_shoreline_bank(solver,xy,theta=theta,hmaxth=hmaxth,sedthr=sedthr)
    install_shoreline_relaxation(solver,xy,slope=slope,transport=True,
        method='classic_scour_bundle' if relaxation else 'off')
    composed_finish=p.finish;composed_commit=p.commit
    def finish(owner,qold,qnew,water,*,morphology=True):
        result=composed_finish(qold,qnew,water,morphology=morphology)
        total=1590.*(owner.ztrial-owner.z0)*owner.g['area']
        terms=owner.transport_trial_mass+owner.bank_trial_mass+owner.relaxation_trial_mass
        residual=float(xp.max(xp.abs(total-terms)))
        if residual>1e-9 or bool(xp.any(owner.ztrial<owner.rock-1e-13)):
            raise BendTrialRejected('Composed bank/collapse transfer ledger or stock failed')
        owner.last['process_cell_closure_max_kg']=residual
        return result
    def commit(owner):
        composed_commit()
        owner.process_gross_mass[0]+=xp.sum(xp.maximum(owner.bank_trial_mass,0.))
        owner.process_gross_mass[1]+=xp.sum(xp.maximum(owner.relaxation_trial_mass,0.))
    p.finish=MethodType(finish,p);p.commit=MethodType(commit,p)
    p.identity['bank_relaxation_composition']=dict(source_sha256=digest(__file__),
        order=['transport','dry_bank','relaxation'],one_owner_per_process=True,
        boundary_ledger='transport_only',scope='explicit_uniform_grid_single_grain_shoreline',
        mechanical_block_failure=False)
    return solver
