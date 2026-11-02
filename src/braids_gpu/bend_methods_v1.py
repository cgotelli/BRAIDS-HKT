"""Explicit experimental bend bundle on the reconstructed production GPU path.

The native input is a hydraulic-only carrier, never re-labelled as a historical
Delft sediment law. A distinct receipt/checkpoint records the additional owner.
This adapter is deliberately not exposed in QGIS before experiment qualification.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from braids_core.sediment.bend_methods_v1 import BendPhysics, BendTrialRejected, digest


class ProductionBendOwner:
    def __init__(self,solver,*,morphology=True,**options):
        import cupy as cp
        from .implicit_midpoint_fwave import ImplicitStepRejected
        self.cp=cp;self.solver=solver;self.reject=ImplicitStepRejected
        d=solver.data;c=solver.implicit_midpoint
        if c is None or c.accepted_steps or getattr(solver,'bend_bundle_v1',None) is not None:
            raise ValueError('Bind bend bundle to a fresh implicit solver only')
        for key in ('bedlaw','exchange_mode','secondary_flow_model','secondary_flow','bed_slope',
                    'horizontal_viscosity','bedload_adaptation','bedform_roughness','general_avalanche',
                    'amr_enabled','bank_failure','bank_toe_erosion_mode','morphological_acceleration'):
            if np.any(d.get(key,0)): raise ValueError('Bend bundle forbids duplicate/incompatible '+key)
        if int(d['nfrac'])!=1 or int(d['nbed_layers'])!=1 or np.any(d['state'][3]):
            raise ValueError('Bend bundle requires one grain, one bed inventory and zero suspended mass')
        if float(d['morphological_factor'])!=1 or np.any(d['nhydro_series']) or np.any(d['nseries']):
            raise ValueError('First bend integration requires fixed forcing and MF=1')
        for key in ('fraction_bed_input','bc_bed_input','fraction_concentration',
                    'series_bed_input','fraction_series_bed_input','secondary_mass'):
            if np.any(d.get(key,0)):
                raise ValueError('Bend bundle cannot ignore an existing '+key)
        if np.any(np.asarray(d['forcing'])[2:4]):
            raise ValueError('Bend feed must be supplied through its explicit owner')
        self.physics=BendPhysics(d,backend='gpu',**options)
        p=self.physics.parameters
        expected=(float(d['grav']),float(d['dry']),float(d['d50_fraction'][0]),
                  float(d['rhos_fraction'][0])/float(d['rho']))
        if not np.allclose(expected,(p.gravity,p.dry,p.diameter,p.relative_density),rtol=1e-12,atol=0):
            raise ValueError('Bend physics differs from hydraulic material')
        if not np.allclose(d['manning_cell'],p.manning,rtol=1e-12,atol=0):
            raise ValueError('Bend shear and hydraulic Manning must agree')
        self.rhos=float(d['rhos_fraction'][0]); self.bulk=self.rhos*(1-p.porosity)
        if not np.allclose(d['bulk_fraction'],self.bulk): raise ValueError('Porosity differs')
        self.layers_trial=cp.zeros_like(solver.state['layers'])
        self.momentum_rate=cp.zeros_like(solver.state['q'])
        self.bed_ledger=cp.zeros_like(solver.ledger['bed'])
        self.morphology=bool(morphology)
        self.physics.initialize_secondary(solver.state['q'],solver.bed())
        # Transaction enumerator only traverses GPU-owned classes, so expose a
        # stable dictionary of EVERY shared-owner device allocation explicitly.
        self.resident={k:v for k,v in vars(self.physics).items() if isinstance(v,cp.ndarray)}
        self.resident['geometry']=self.physics.g
        self.identity=dict(component=self.physics.identity,adapter_sha256=digest(__file__),
            hydraulic_integrator='implicit_fwave_midpoint',hydraulic_receipt=solver.candidate_receipt(),
            initial_input_sha256=hashlib.sha256(b''.join(
                k.encode()+np.asarray(v).tobytes() for k,v in sorted(d.items()) if isinstance(v,np.ndarray))).hexdigest(),
            public_support=False,complete_author_bundle=False)
        self.identity['implicit_controls']={k:getattr(c,k) for k in ('nonlinear_rtol','nonlinear_atol',
            'nonlinear_maxiter','linear_rtol','linear_atol','linear_maxiter','advective_cfl_limit','froude_limit')}
        solver.bend_bundle_v1=self
        c.bind_bend_physics_v1(self)

    def _call(self,fn,*args,**kwargs):
        try: return fn(*args,**kwargs)
        except BendTrialRejected as exc: raise self.reject(str(exc)) from exc

    def begin(self,q,z,dt,time): self._call(self.physics.begin,q,z,dt,time)
    def add_momentum_rate(self,q,water,rate):
        self.momentum_rate.fill(0)
        self._call(self.physics.add_momentum,q,water,self.momentum_rate)
        rate += self.momentum_rate

    def finish(self,qold,qnew,water):
        p=self.physics
        self._call(p.finish,qold,qnew,water,morphology=self.morphology)
        self.layers_trial[0,0]=self.bulk*(p.ztrial-p.rock)
        # Solver bed ledger is mass; component ledger is solid volume. Lai's
        # combined stock includes M, which is explicitly included in fields.
        self.bed_ledger.fill(0)
        for bid in range(self.solver.nb):
            self.bed_ledger[0,bid]=-self.rhos*self.cp.sum(self.cp.where(
                p.g['bid']==bid,p.face_integral*p.g['length'],0.))

    def trial_metadata(self):
        p=self.physics
        return copy.deepcopy(dict(time=p.time,steps=p.steps,last=p.last,open=p._open,
                                   dt=getattr(p,'_dt',None)))

    def restore_trial_metadata(self,value):
        p=self.physics;p.time=value['time'];p.steps=value['steps'];p.last=value['last'];p._open=value['open']
        if value['dt'] is not None:p._dt=value['dt']
        elif hasattr(p,'_dt'):del p._dt

    def fields(self):
        p=self.physics;c=self.cp
        return dict(bend_vorticity_inventory=c.asnumpy(p.W),bend_vns_m_s=c.asnumpy(.25*p.W),
            bend_moving_solid_m=c.asnumpy(p.M),bend_solid_boundary_m3=c.asnumpy(p.boundary_ledger),
            bend_secondary_boundary=c.asnumpy(p.secondary_boundary_ledger),
            bend_secondary_source=c.asnumpy(p.secondary_source_ledger),
            bend_secondary_projection=c.asnumpy(p.secondary_projection_ledger))

    def checkpoint(self,path):
        from .a1_full_array_transaction import _unique_arrays
        from .atomic_output import save_arrays_atomic
        if self.physics._open:raise ValueError('Checkpoint requires accepted state')
        arrays=_unique_arrays(self.solver);names=sorted(arrays)
        values={f'a{i}':self.cp.asnumpy(arrays[name]) for i,name in enumerate(names)}
        c=self.solver.implicit_midpoint
        metadata=dict(identity=self.identity,names=names,physics=self.trial_metadata(),
            morphology=self.morphology,time=self.solver._glt_accepted_time,
            array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in values.items()},
            controller={k:getattr(c,k) for k in ('have_predictor','accepted_steps','rejected_steps',
                                              'fwave_evaluations','linear_matvecs')})
        save_arrays_atomic(Path(path),metadata=np.array(json.dumps(metadata,sort_keys=True)),
            **values)

    def restart(self,path):
        from .a1_full_array_transaction import _unique_arrays
        if self.physics._open:raise ValueError('Restart requires accepted state')
        arrays=_unique_arrays(self.solver);names=sorted(arrays)
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']))
            if meta['identity']!=self.identity or meta['names']!=names:
                raise ValueError('Bend checkpoint method/runtime/geometry identity differs')
            values=[f[f'a{i}'].copy() for i in range(len(names))]
            for i,(name,a) in enumerate(zip(names,values)):
                if a.shape!=arrays[name].shape or a.dtype!=arrays[name].dtype:
                    raise ValueError('Bend checkpoint array differs: '+name)
                if hashlib.sha256(a.tobytes()).hexdigest()!=meta['array_sha256'][f'a{i}']:
                    raise ValueError('Bend checkpoint array checksum differs: '+name)
            if (not np.isfinite(meta['time']) or meta['time']<0 or meta['physics']['open'] or
                    meta['physics']['time']!=meta['time']):raise ValueError('Invalid checkpoint clock')
        for name,a in zip(names,values): arrays[name].set(a)
        self.restore_trial_metadata(meta['physics']);self.morphology=meta['morphology']
        self.solver._glt_accepted_time=meta['time']
        for k,v in meta['controller'].items():setattr(self.solver.implicit_midpoint,k,v)
        self.solver.implicit_midpoint.equilibrium_candidate=False
