"""Existing FLOW secondary physics on the conservative Hancock water path.

The local Hancock predictor is not a conservative scalar-transport path.
Use accepted face fluxes and linear water inventories, preserving uniform I,
including dry receivers. Half-time circulation is private predictor data;
publish the full interval only after the complete hydraulic trial succeeds.
"""
from copy import deepcopy
from pathlib import Path
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment.river_secondary_clock import SecondaryClock
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class HancockAdvanced(SecondaryClock):
    def __init__(self,solver,cfl,*,secondary,viscosity):
        p=solver.physics;previous=getattr(p,'_secondary_clock',None)
        boundary=None if previous is None else (previous.boundary_values.copy(),deepcopy(previous.identity['open_boundary']))
        super().__init__(solver,cfl)
        if boundary is not None:
            self.boundary_values[:]=boundary[0];self.identity['open_boundary']=boundary[1]
            p.identity['boundaries']=self.identity['open_boundary']
        p._secondary_clock=self
        self.secondary_enabled=bool(secondary);self.viscosity=float(viscosity)
        self.stability=None;self.viscous_frequency=self.xp.zeros(p.n);self.viscous_half=self.xp.zeros(p.n)
        self.identity.update(method='hancock_conservative_water_secondary_v1',
            hancock_wrapper_sha256=digest(__file__),
            hydraulic_path='linear accepted water inventory; constant actual Hancock shared face flux',
            momentum_feedback='accepted-start tensor; private half-interval secondary predictor at Hancock velocity; full interval published after success',
            half_intensity_mapping='preserve I when local Hancock depth differs from conservative half-interval depth',
            secondary_enabled=self.secondary_enabled,viscosity_m2_s=self.viscosity,
            viscous_stability='absolute nonorthogonal velocity-diffusion row bound added to hydraulic face frequency')
        if self.viscosity>0.:
            self.prepare()
            self.stability=WaveEngine(self.solver.backend,self.p.flow.threads,
                source=Path(__file__).with_name('hllc_advanced.h'),names=('hllc_advanced_frequency',))
            self.identity['viscous_runtime']=self.stability.identity

    def enabled(self):return self.secondary_enabled

    def begin(self):
        super().begin()

    def start(self,q):
        if self.active:self.p.refresh_secondary_momentum(q,self.p.secondary_pending)

    def half(self,q0,qnew,qhalf,water,dt):
        if not self.active:return
        xp=self.xp;p=self.p;dry=p.parameters.dry
        # The water path derives from the same single shared flux as the
        # accepted hydraulic update. Use Hancock's half-time velocity for the
        # source/relaxation, without substituting its local depth inventory.
        end=.5*(q0+qnew)
        end[1:]=qhalf[1:]*xp.where(qhalf[0]>dry,end[0]/xp.maximum(qhalf[0],dry),0.)
        trial=self.trial(q0,end,end,water,water,.5*dt)
        intensity=xp.where(end[0]>dry,trial[0]/xp.maximum(end[0],dry),0.)
        p.refresh_secondary_momentum(qhalf,qhalf[0]*intensity)

    def end(self,q0,qnew,water,dt):
        self.accept(self.trial(q0,qnew,qnew,water,water,dt),dt)

    def add_frequency(self,q,frequency,half=None):
        if self.viscosity<=0.:return
        p=self.p;g=p.g
        self.stability.launch('hllc_advanced_frequency',p.n,p.n,p.nf,
            p.parameters.dry,self.viscosity,q,self.metric,g['owner'],g['neighbour'],
            g['offset'],g['faces'],g['length'],g['area'],self.viscous_frequency)
        if half is not None:
            self.stability.launch('hllc_advanced_frequency',p.n,p.n,p.nf,
                p.parameters.dry,self.viscosity,half,self.metric,g['owner'],g['neighbour'],
                g['offset'],g['faces'],g['length'],g['area'],self.viscous_half)
            self.xp.maximum(self.viscous_frequency,self.viscous_half,out=self.viscous_frequency)
        frequency+=self.viscous_frequency
