"""Compact second-order HLLC controller for the current hydraulic owner."""
from pathlib import Path
import numpy as np
from .hllc import HLLCHydraulics
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment.river_explicit import RiverExplicit

METHOD='explicit_hllc_muscl_hancock_river_v1'


class RiverHLLC:
    hydraulic_pressure='hydrostatic_hllc'

    def __init__(self,solver,mesh,cfl):
        self.solver=solver;self.cfl=float(cfl);self.easy_steps=0;p=solver.physics
        def exterior(q,z,time):
            boundary=getattr(solver,'exterior_bed',None)
            if boundary is None:raise ValueError('HLLC requires the current exterior boundary owner')
            boundary.residual_token=object()
            try:boundary.evaluate(q,z,time=time)
            finally:boundary.residual_token=None
            return boundary.flux,boundary.trace
        boundary=getattr(solver,'exterior_bed',None)
        self.inlet_feasibility=None;self.outlet_feasibility=None
        if boundary is not None:
            from .outlet_feasibility import OutletFeasibility
            self.outlet_feasibility=OutletFeasibility(boundary)
        if boundary is not None and boundary.inlet_compatibility is not None:
            from .inlet_feasibility import InletFeasibility
            self.inlet_feasibility=InletFeasibility(boundary)
        def feasible_boundary(q,dt,flux,trace):
            if self.outlet_feasibility is not None:self.outlet_feasibility(q,dt,flux,trace)
            if self.inlet_feasibility is not None:self.inlet_feasibility(q,dt,flux,trace)
        turbine='hkt_terrain' in solver.identity
        hydro=p.identity.get('resolved_river_model',{}).get('model',{}).get('physics',{}).get('hydrodynamics',{})
        secondary=hydro.get('secondary_flow')=='secondary.prognostic'
        # The conservative wet-face owner disables the inherited nu to avoid
        # double stress. Its recorded coefficient is the selected viscosity.
        nu=float(p.identity.get('conservative_viscosity',{}).get('coefficient_m2_s',p.nu))
        self.secondary=None
        if secondary or nu>0.:
            from .advanced import HancockAdvanced
            self.secondary=HancockAdvanced(solver,cfl,secondary=secondary,viscosity=nu)
        def momentum(q,water,rate):
            # Momentum stress is a velocity acceleration times the actual
            # evaluation-stage water depth, never a frozen outer-step depth.
            p.hold[:]=q[0];p.add_momentum(q,water,rate)
        self.operator=HLLCHydraulics(mesh,backend=solver.backend,threads=p.flow.threads,
             gravity=p.parameters.gravity,dry=p.parameters.dry,cfl=cfl,
             exterior=exterior if any(f.kind in ('inlet','outlet') for f in mesh.faces) else None,
             friction=p.friction,check=solver.check,integration='hancock',
             momentum=momentum if secondary or nu>0. or turbine else None,
             boundary_correction=feasible_boundary if boundary is not None else None,
             face_correction=getattr(solver,'hkt_hllc_pressure',None),advanced=self.secondary)
        self.identity=dict(method=METHOD,controller_sha256=digest(__file__),cfl=self.cfl,
             spatial=self.operator.identity,
             friction='Strang drag around every actual Hancock substep',
             cfl_definition='dt max_cell(sum_face(length * signal_speed) / actual_area)',
             state='accepted scalar timestep hint; every new-mesh stage recomputes frequency',
             momentum=dict(secondary=secondary,advanced_viscosity_m2_s=nu,turbines=turbine),
             explicit_fallback=False)
        self.identity['inlet_feasibility']=None if self.inlet_feasibility is None else self.inlet_feasibility.identity
        self.identity['outlet_feasibility']=None if self.outlet_feasibility is None else self.outlet_feasibility.identity
        if self.secondary is not None:self.identity['advanced_hydrodynamics']=self.secondary.identity
        selected=p.identity.get('resolved_river_model',{}).get('model',{}).get('physics',{}).get('sediment')
        if selected is not None:
            from .bedload import identity, validate_selection
            validate_selection(selected,hydro)
            self.identity['bedload']=identity(solver)

    @property
    def step_hint(self):return self.operator.step_hint

    @step_hint.setter
    def step_hint(self,value):self.operator.step_hint=value

    def state(self):return dict(step_hint=self.step_hint,easy_steps=0)

    validate_state=staticmethod(RiverExplicit.validate_state)

    def restore(self,state):
        self.validate_state(state)
        if state['easy_steps']!=0:raise ValueError('Invalid HLLC explicit clock state')
        self.step_hint=state['step_hint'];self.easy_steps=0

    def advance(self,base,dt,time):
        q,water,info=self.operator.advance(base,self.solver.z,dt,time)
        return q,water,dict(integrator=METHOD,hydraulic_substeps=info['hydraulic_substeps'],
             hydraulic_minimum_dt=info['minimum_dt_s'],hydraulic_maximum_dt=info['maximum_dt_s'],
             hydraulic_endpoint_remainders=0,hydraulic_rejections=info['rejections'],hydraulic_preempted_retries=0,
             fwave_evaluations=0,hllc_evaluations=info['spatial_evaluations'],linear_matvecs=0,nonlinear_iterations=0,
             accepted_cfl_max=info['accepted_cfl_max'],hydraulic_next_substep_hint_s=self.step_hint,explicit_fallback=False)


def install(owner,mesh,cfl):
    solver=getattr(owner,'solver',owner)
    if solver.time or solver.steps or solver.physics._open:raise ValueError('Fresh hydraulic owner required')
    controller=RiverHLLC(solver,mesh,cfl);solver.midpoint=controller
    solver.identity['integrator']=METHOD;solver.identity.pop('river_explicit',None)
    solver.identity['river_hllc']=controller.identity
    return owner
