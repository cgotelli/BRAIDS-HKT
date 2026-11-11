"""Conservative signed secondary transport on accepted hydraulic SSP stages.

No averaged outer-step water flux is used to advect a dry-start scalar. The
same stage depths and face water fluxes define a conservative dense hydraulic
path for Heun transport. A separate bound shortens a scalar substep,
never change the user backend, physics, output clock or outer sediment clock.
"""
from pathlib import Path
import numpy as np
from . import secondary_vector_ops as vectors
from .bend_methods_v1 import BendTrialRejected, digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('secondary_metric','secondary_frequency','secondary_signed_stress','secondary_open_faces')


class SecondaryClock:
    def __init__(self, solver, cfl):
        self.solver=solver;self.p=p=solver.physics;self.xp=xp=p.xp
        self.cfl=float(cfl);self.engine=None;self.active=False
        # These arrays belong to the parent snapshot/checkpoint transaction.
        p.secondary_pending=xp.zeros(p.n)
        p.secondary_pending_flux=xp.zeros(p.nf)
        p.secondary_pending_source=xp.zeros(p.n)
        p.secondary_clock_stats=xp.zeros(5)  # elapsed, steps, max CFL, dry sink, min dt
        p.secondary_frequency=xp.zeros(p.n)
        # Immutable mesh-owned metric is not trial state. In particular, a
        # first-step rollback must not zero a lazily constructed metric while
        # keeping its engine alive.
        self.metric=xp.zeros(p.nf)
        self.boundary_values=xp.zeros(p.nf)
        self.identity=dict(method='hydraulic_stage_consistent_secondary_ssprk22_v1',
            wrapper_sha256=digest(__file__),kernel_sha256=digest(Path(__file__).with_suffix('.h')),
            hydraulic_path='SSPRK22 quadratic depth, linear shared water flux; Euler scalar stage uses matching Euler water inventory',
            momentum_feedback='SSPRK22 stage-consistent tensor; nonpublishing stable secondary predictor for stage two',
            betac_limiter='BRAIDS sign-preserving magnitude cap; native FLOW negative-overlimit sign reversal is not retained',
            cfl=self.cfl,source='unchanged FLOW old-depth source and new-depth implicit sink per substep',
            stability='limited advection donor bound plus absolute nonorthogonal diffusion row bound',
            budget_tolerance=1e-10,backend=solver.backend)
        self.identity['open_boundary']=dict(method='prescribed_incoming_secondary_intensity_v1',
            incoming='independent signed I in m/s per named open boundary, also on dry receivers',
            outgoing='owner transported signed I',walls='zero scalar flux',
            component_default_m_s=0.,
            reason='Receiving-cell curvature is not upstream forcing; retain interior source and relaxation')
        p.identity['boundaries']=self.identity['open_boundary']

    def enabled(self):
        p=self.p
        resolved=p.identity.get('resolved_river_model',{}).get('model',{})
        h=resolved.get('physics',{}).get('hydrodynamics',{})
        return (h.get('secondary_flow')!='secondary.off'
            and not p.identity.get('simple_physics'))

    def begin(self):
        p=self.p;self.active=self.enabled()
        p.secondary_pending[:]=p.W;p.secondary_pending_flux.fill(0)
        p.secondary_pending_source.fill(0);p.secondary_clock_stats.fill(0)
        if not self.active:return
        self.prepare()

    def prepare(self):
        p=self.p
        if self.engine is None:
            self.engine=WaveEngine(self.solver.backend,p.flow.threads,
                source=Path(__file__).with_suffix('.h'),names=NAMES)
            g=p.g
            self.engine.launch('secondary_metric',p.nf,p.n,p.nf,
                int(getattr(p,'correct_diffusion',False)),g['owner'],g['neighbour'],
                g['offset'],g['faces'],g['xy'],g['normal'],g['distance'],g['inverse'],self.metric)
            if not bool(self.xp.all(self.xp.isfinite(self.metric))):
                raise ValueError('Non-finite secondary diffusion geometry')

    def correct_stress(self):
        if not self.enabled():return
        self.prepare();p=self.p
        self.engine.launch('secondary_signed_stress',p.n,p.n,p.inputs,p.outputs)

    def frequency(self,q,water):
        p=self.p;g=p.g
        self.engine.launch('secondary_frequency',p.n,p.n,p.nf,p.parameters.dry,p.diff,
            q,water,self.metric,g['owner'],g['neighbour'],g['bid'],g['offset'],
            g['faces'],g['length'],g['area'],p.secondary_frequency)
        return float(self.xp.max(p.secondary_frequency))

    def scalar_faces(self,q,hi,water):
        """Interior FLOW transport, with independent upstream circulation.

        Prescribe upstream I, not the receiving cell's source/sink equilibrium
        or evolving I. Both create a feedback through boundary curvature.
        This is a BRAIDS boundary discretization, not the literal FLOW secbou.
        No interior source, stress, viscosity or sediment formula is changed.
        """
        p=self.p;g=p.g
        p.scalar_faces(q,hi,water)
        self.engine.launch('secondary_open_faces',p.nf,p.nf,p.parameters.dry,
            q,hi,water,g['owner'],g['neighbour'],g['bid'],self.boundary_values,p.scalarflux)

    def trial(self,q0,q1,q2,w0,w1,dt):
        """Return a pending update; a failed trial cannot publish any ledger."""
        if not self.active:return None
        p=self.p;xp=self.xp;g=p.g;n=p.n
        old=p.secondary_pending;new=old.copy();flux=xp.zeros(p.nf);source=xp.zeros(n)
        removed=0.;elapsed=0.;steps=0;peak=0.;minimum=dt;hint=dt
        r0=(q1-q0)/dt;r1=(2.*q2-q0-q1)/dt
        def state(f):return vectors.state(xp,q0,r0,r1,dt,f)
        while elapsed<dt-1e-14*dt:
            self.solver.check()
            f=elapsed/dt;start=state(f);wa=vectors.blend(xp,w0,w1,f)
            frequency=self.frequency(start,wa)
            delta=min(hint,dt-elapsed,.98*self.cfl/frequency if frequency>0 else dt)
            for retry in range(40):
                end_fraction=(elapsed+delta)/dt
                end=state(end_fraction)
                # Matching Euler depth preserves uniform intensity exactly.
                stage=vectors.stage(xp,start,r0,r1,delta,f)
                wb=vectors.blend(xp,w0,w1,end_fraction)
                frequency=max(frequency,self.frequency(stage,wb))
                valid=(np.isfinite(frequency) and delta*frequency<=self.cfl*(1+1e-12)
                       and not bool(xp.any(stage[0]<0)))
                if valid:break
                delta=min(.5*delta,.98*self.cfl/frequency if frequency>0 else .5*delta)
                if delta<dt*2**-35:raise BendTrialRejected('Secondary transport timestep underflow')
            else:raise BendTrialRejected('Secondary transport stage stability failed')
            p.local(end,p.z0,start[0],new)
            self.scalar_faces(start,new,wa);part=.5*delta*p.scalarflux
            p.flow.launch('flow_scalar_gather',n,n,delta,new,p.scalarflux,
                g['offset'],g['faces'],g['signs'],g['length'],g['area'],p.work)
            self.scalar_faces(stage,p.work,wb);part+=.5*delta*p.scalarflux
            p.flow.launch('flow_scalar_gather',n,n,delta,new,p.scalarflux,
                g['offset'],g['faces'],g['signs'],g['length'],g['area'],p.work2)
            transported,new=vectors.update(xp,p.work,p.work2,start[0],p.outputs[3],p.outputs[4],delta)
            dry=end[0]<=p.parameters.dry
            removed+=float(xp.sum(xp.where(dry,new,0.)*g['area']))
            new[dry]=0.;source+=new-transported;flux+=part
            elapsed+=delta;steps+=1;peak=max(peak,delta*frequency);minimum=min(minimum,delta)
            hint=2.*delta
            if steps>8192:raise BendTrialRejected('Secondary transport substep limit')
        outside=(g['neighbour']<0)&(g['bid']>=0)
        residual=xp.sum((new-old)*g['area'])+xp.sum(flux*g['length']*outside)-xp.sum(source*g['area'])
        if not bool(xp.all(xp.isfinite(new))) or abs(float(residual))>1e-10:
            raise BendTrialRejected('FLOW signed secondary substep budget failed')
        return new,flux,source,removed,peak,steps,minimum

    def accept(self,trial,dt):
        if trial is None:return
        p=self.p;new,flux,source,removed,cfl,steps,minimum=trial
        p.secondary_pending[:]=new;p.secondary_pending_flux+=flux
        p.secondary_pending_source+=source
        p.secondary_clock_stats[0]+=dt;p.secondary_clock_stats[1]+=steps
        p.secondary_clock_stats[2]=max(float(p.secondary_clock_stats[2]),cfl)
        p.secondary_clock_stats[3]+=removed
        previous=float(p.secondary_clock_stats[4])
        p.secondary_clock_stats[4]=min(previous,minimum) if previous else minimum

    def finish(self):
        p=self.p;xp=self.xp;g=p.g
        if abs(float(p.secondary_clock_stats[0])-p._dt)>1e-12*max(1.,p._dt):
            raise BendTrialRejected('Secondary and hydraulic clocks differ')
        p.Wtrial[:]=p.secondary_pending;p.scalarflux_integral[:]=p.secondary_pending_flux
        p.source_increment[:]=p.secondary_pending_source
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=-xp.sum(p.scalarflux_integral*g['length']*outside)
        source=xp.sum(p.source_increment*g['area'])
        residual=xp.sum((p.Wtrial-p.W)*g['area'])-boundary-source
        if not bool(xp.all(xp.isfinite(p.Wtrial))) or abs(float(residual))>1e-10:
            raise BendTrialRejected('FLOW signed secondary budget failed')
        return float(boundary),float(source),float(residual)
