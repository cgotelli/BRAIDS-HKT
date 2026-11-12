"""One grain-resolved bank/collapse transaction on the selected river solver.

FLOW bott3d bank sharing retains its net-erosion and donor-composition rule.
Legacy avalan provides only slope geometry: BRAIDS conservatively peels finite
donor layers, a necessary extension because avalan does not track composition.
"""
from pathlib import Path
from types import MethodType
import numpy as np
from .river_mixtures import RiverMixtureSolver
from .bed_columns_v101 import host
from .bend_methods_v1 import digest,BendTrialRejected
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from validation.sediment.river_morphodynamics import wet_bank_handoff


class LayeredScour:
    def __init__(self,backend,threads=4):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU/GPU backend required')
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_name('river_layered_scour.h'),
            names=('layered_scour_rows','layered_scour_columns'))
        self.identity=self.engine.identity

    def apply(self,mass,rock,area,*,dx,slope):
        if not np.isfinite([dx,slope]).all() or min(dx,slope)<=0:raise ValueError('Positive spacing/slope required')
        xp=np
        if self.engine.backend=='gpu':
            import cupy as xp
        mass=xp.ascontiguousarray(mass,dtype=xp.float64)
        rock=xp.ascontiguousarray(rock,dtype=xp.float64)
        area=xp.ascontiguousarray(area,dtype=xp.float64)
        if mass.ndim!=4:raise ValueError('Expected layers,classes,y,x')
        nl,ng,ny,nx=mass.shape
        if min(nl,ng,ny,nx)<1:raise ValueError('Nonempty layered collapse fields required')
        if rock.shape!=(ny,nx) or area.shape!=(ny,nx):raise ValueError('Layered collapse geometry differs')
        if (not all(bool(xp.all(xp.isfinite(a))) for a in (mass,rock,area))
                or bool(xp.any(mass<0)) or bool(xp.any(area<=0))):raise ValueError('Invalid layered collapse inputs')
        work=xp.ascontiguousarray(mass.copy());rock=xp.ascontiguousarray(rock);area=xp.ascontiguousarray(area)
        moved=xp.zeros((ng,ny,nx))
        for name,count in [('layered_scour_rows',ny),('layered_scour_columns',nx)]:
            self.engine.launch(name,count,nx,ny,ng,nl,float(dx),float(slope),area,rock,work,moved)
        if (bool(xp.any(work<0)) or not bool(xp.all(xp.isfinite(work)))
                or float(xp.max(xp.abs(moved.sum(axis=(1,2)))))>1e-9):
            raise BendTrialRejected('Layered collapse class budget/stock failed')
        closure=(work-mass).sum(axis=0)*area-moved
        if float(xp.max(xp.abs(closure)))>1e-9:raise BendTrialRejected('Layered collapse local ledger failed')
        return work,moved


class RiverMixtureBankSolver(RiverMixtureSolver):
    def __init__(self,*args,bank_theta=.5,collapse=False,collapse_slope=.5,**kwargs):
        if not np.isfinite(bank_theta) or not 0<=bank_theta<=1:raise ValueError('Bank fraction must be in [0,1]')
        if type(collapse) is not bool:raise ValueError('Explicit collapse control required')
        if not np.isfinite(collapse_slope) or collapse_slope<=0:raise ValueError('Positive collapse slope required')
        super().__init__(*args,**kwargs)
        self.bank_theta=float(bank_theta);self.collapse=collapse;self.collapse_slope=float(collapse_slope)
        p=self.physics;xp=self.xp;ng=len(self.capacity)
        self.scour=LayeredScour(self.backend,p.flow.threads) if collapse else None
        # The scalar adapter is not called; reuse only its verified geometry,
        # parallel bank operator and selected wet-portion eligibility rule.
        p.identity['dry_bank']['theta']=self.bank_theta
        if bank_theta>0:wet_bank_handoff.install(self.solver)
        p.class_bank_trial=xp.zeros((ng,p.n));p.class_bank_cumulative=xp.zeros((ng,p.n))
        p.class_collapse_trial=xp.zeros((ng,p.n));p.class_collapse_cumulative=xp.zeros((ng,p.n))
        p.class_bank_gross=xp.zeros(ng);p.class_collapse_gross=xp.zeros(ng)
        original=p.commit
        def commit(owner):
            original()
            owner.class_bank_cumulative+=owner.class_bank_trial
            owner.class_collapse_cumulative+=owner.class_collapse_trial
            owner.bank_cumulative_mass+=owner.bank_trial_mass
            owner.relaxation_cumulative_mass+=owner.relaxation_trial_mass
            owner.class_bank_gross+=xp.maximum(owner.class_bank_trial,0.).sum(axis=1)
            owner.class_collapse_gross+=xp.maximum(owner.class_collapse_trial,0.).sum(axis=1)
        p.commit=MethodType(commit,p)
        p.identity['classwise_bank_collapse']=dict(source_sha256=digest(__file__),bank_theta=bank_theta,
            collapse=collapse,collapse_slope=collapse_slope,scour_runtime=self.scour.identity if collapse else None,
            sequence='transport; net-erosion bank sharing using pre-update surface composition; layer update; layered collapse; layer rebalance; one commit',
            bank_stock='post-transport active stock; finite donor and per-class shared limiter',
            collapse_stock='top-down donor strata; deposited into recipient active layer; finite rock',
            collapse_provenance='legacy FLOW geometry extended conservatively; native avalan ignores class inventories',
            reference='Delft3D4 FLOW 2DH bott3d.f90 and avalan.f90; unmodified runtime',public_support=False)
        self.identity['selected_river_bundle']['scope']='fixed_grid_classwise_EH_layers_bank_collapse_implicit_midpoint'
        p.identity['scope']='fixed_grid_classwise_EH_layers_bank_collapse_shoreline_development'
        p.identity['legacy_scalar_bank_collapse_adapter_called']=False

    def finish(self,qold,qnew,water,*,morphology=True):
        p=self.physics;xp=self.xp;c=self.columns;g=p.g
        super().finish(qold,qnew,water,morphology=morphology)
        p.class_bank_trial.fill(0);p.class_collapse_trial.fill(0)
        p.bank_trial_mass.fill(0);p.relaxation_trial_mass.fill(0)
        if not morphology:return p.last
        shape=tuple(p.identity['dry_bank']['shape']);ng=len(self.capacity)
        order=p.bank_order;back=p.bank_back
        field=lambda v:xp.ascontiguousarray(v[...,order].reshape((*v.shape[:-1],*shape)))
        if self.bank_theta>0:
            active=c.available+p.class_delta
            if bool(xp.any(active<-1e-12)):raise BendTrialRejected('Negative post-transport active stock')
            _,move=p.bank_engine.apply(field(p.z0),field(qnew[0]),field(g['area']),xp.full(ng,1590.),
                field(qnew[0]>p.sediment_depth),xp.ones(shape,xp.int32),field(p.class_delta),
                field(c.surface_fraction),xp.ones((ng,*shape)),field(xp.maximum(active,0.)),
                theta=self.bank_theta,hmaxth=0.,sedthr=p.sediment_depth)
            p.class_bank_trial[:]=move.reshape(ng,-1)[:,back]
            c.rollback()
            c.begin(p.class_delta+p.class_bank_trial/g['area'],qnew[0],p._dt)
        if self.collapse:
            mass,sv,ext,basal,dt,_=c.pending
            work,move=self.scour.apply(field(mass),field(c.rock),field(g['area']),dx=float(self.data['dx']),slope=self.collapse_slope)
            p.class_collapse_trial[:]=move.reshape(ng,-1)[:,back]
            work=xp.ascontiguousarray(work.reshape(mass.shape[0],ng,-1)[:,:,back])
            if bool(xp.any(move!=0)):
                # Normalize layer thicknesses AFTER moving the actual strata.
                # Zero external increment: do not apply the transfer twice.
                targets=c.runtime.targets(qnew[0])
                work,nextsv,base=c.runtime.evaluate(work,sv,xp.zeros_like(p.class_delta),targets,p._dt)
                if bool(xp.any(base!=0)):raise BendTrialRejected('Collapse cannot import basal stock')
                after=host((work*g['area']).sum(axis=(0,2)))
                before=host((c.mass*g['area']).sum(axis=(0,2)))
                residual=after-before-ext-basal
                if abs(residual).max()>1e-9:raise BendTrialRejected('Composed class stock budget failed')
                c.pending=(work,nextsv,ext,basal,dt,residual);c.seal_pending()
        mass,sv,ext,basal,dt,residual=c.pending
        if np.any(basal):raise BendTrialRejected('Bank/collapse cannot import basal stock')
        p.ztrial[:]=c.rock+(mass/1590.).sum(axis=(0,1))
        p.bank_trial_mass[:]=p.class_bank_trial.sum(axis=0)
        p.relaxation_trial_mass[:]=p.class_collapse_trial.sum(axis=0)
        actual=(mass-c.mass).sum(axis=0)*g['area']
        expected=p.class_delta*g['area']+p.class_bank_trial+p.class_collapse_trial
        local=float(xp.max(xp.abs(actual-expected)))
        budget=float(xp.max(xp.abs(expected.sum(axis=1)-p.class_delta.dot(g['area']))))
        if local>1e-9 or budget>1e-9:raise BendTrialRejected('Composed bank/collapse process ledgers failed')
        p.last.update(class_process_closure_max_kg=local,class_internal_budget_max_kg=budget,
            bank_transfer_mass_kg=float(xp.maximum(p.class_bank_trial,0.).sum()),
            collapse_transfer_mass_kg=float(xp.maximum(p.class_collapse_trial,0.).sum()),
            class_column_budget_max_kg=float(abs(residual).max()),
            maximum_relative_total_bed_increment=float(xp.max(xp.abs(p.ztrial-p.z0)/xp.maximum(qnew[0],p.sediment_depth))))
        return p.last
