"""Experimental full native-CPU bend trajectory with reconstructed hydraulics.

Augmented f-wave space operator from current Fortran; implicit midpoint solved
by the GPU controller's frozen acoustic iteration, with CPU BiCGSTAB. Native OpenMP bend
operators share the CUDA equations. No production selector/default is changed.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendPhysics,Parameters,BendTrialRejected,digest
from .bend_native_hydraulics import NativeHydraulics


class CPUBendSolver:
    def __init__(self,deck,data,*,method,parameters=None,threads=4,morphology=True,
                 nonlinear_atol=1e-14,nonlinear_rtol=1e-6,**options):
        p=parameters or Parameters()
        for key in ('bedlaw','exchange_mode','secondary_flow_model','secondary_flow','bed_slope',
                    'horizontal_viscosity','bedload_adaptation','bedform_roughness','general_avalanche',
                    'amr_enabled','bank_failure','bank_toe_erosion_mode','morphological_acceleration'):
            if np.any(data.get(key,0)):raise ValueError('CPU bend forbids duplicate/incompatible '+key)
        if int(data['nfrac'])!=1 or int(data['nbed_layers'])!=1 or np.any(data['state'][3]):
            raise ValueError('CPU bend requires one grain/bed and zero suspension')
        if float(data['morphological_factor'])!=1 or np.any(data['nhydro_series']) or np.any(data['nseries']):
            raise ValueError('CPU bend requires fixed forcing and MF=1')
        for key in ('fraction_bed_input','bc_bed_input','fraction_concentration',
                    'series_bed_input','fraction_series_bed_input','secondary_mass'):
            if np.any(data.get(key,0)):raise ValueError('CPU bend cannot ignore '+key)
        if np.any(np.asarray(data['forcing'])[2:4]):raise ValueError('CPU bend feed requires explicit owner')
        if not np.allclose((data['grav'],data['dry'],data['d50_fraction'][0],data['rhos_fraction'][0]/data['rho']),
                           (p.gravity,p.dry,p.diameter,p.relative_density),rtol=1e-12,atol=0):
            raise ValueError('CPU bend hydraulic material differs')
        if not np.allclose(data['manning_cell'],p.manning,rtol=1e-12,atol=0):raise ValueError('CPU shear/Manning differs')
        if not np.allclose(data['bulk_fraction'],data['rhos_fraction'][0]*(1-p.porosity)):
            raise ValueError('CPU bend porosity differs')
        if not np.isfinite(nonlinear_atol) or not 0<nonlinear_atol<=1e-6:
            raise ValueError('Invalid CPU nonlinear tolerance')
        if not np.isfinite(nonlinear_rtol) or not 0<nonlinear_rtol<1:
            raise ValueError('Invalid CPU nonlinear relative tolerance')
        self.physics=BendPhysics(data,backend='cpu',method=method,parameters=p,threads=threads,**options)
        self.hydraulics=NativeHydraulics(deck,data,threads)
        self.q=np.array(data['state'][:3],order='C',copy=True)
        self.z=np.array(data['rock']+data['state'][4]/data['bulk_fraction'][0])
        self.physics.initialize_secondary(self.q,self.z)
        self.time=float(data.get('time',0));self.steps=0;self.water_ledger=np.zeros(data['nf'])
        self.morphology=bool(morphology);self.nonlinear_atol=float(nonlinear_atol)
        self.nonlinear_rtol=float(nonlinear_rtol)
        self.chezy_min=float(data.get('manning_friction_chezy_minimum_depth',0.))
        self.last={};self.cancelled=lambda:False
        self.predictor=np.zeros_like(self.q);self.have_predictor=False
        self.identity=dict(physics=self.physics.identity,hydraulics=self.hydraulics.identity,
            integrator='native_reconstructed_implicit_midpoint_acoustic_bicgstab_v1',
            controller_sha256=digest(__file__),nonlinear_atol=self.nonlinear_atol,
            nonlinear_rtol=self.nonlinear_rtol,
            public_support=False,complete_author_bundle=False)

    def _check(self):
        if self.cancelled():raise BendTrialRejected('CPU bend cancelled')

    def _friction(self,q,halfdt):
        # Same exact uniform-drag map and depth floor as GPU ti_manning.
        p=self.physics
        p.engine.launch('bend_manning',p.n,p.n,float(halfdt),self.chezy_min,p.p,q)

    def _envelope(self,q,dt):
        p=self.physics;g=p.g
        p.engine.launch('bend_hydraulic_envelope',p.n,p.n,p.nf,float(dt),p.p,q,
            g['offset'],g['faces'],g['owner'],g['neighbour'],g['normal'],g['length'],g['area'],p.metric,p.status)
        if np.any(p.status):raise BendTrialRejected('CPU midpoint wet/low-Froude/advective envelope exceeded')

    def _snapshot(self):
        p=self.physics
        return ({k:v.copy() for k,v in vars(p).items() if isinstance(v,np.ndarray)},
                copy.deepcopy({k:v for k,v in vars(p).items() if k in ('time','steps','last','_open','_dt')}))

    def _restore(self,snapshot):
        arrays,meta=snapshot;p=self.physics
        for k,v in arrays.items():getattr(p,k)[...]=v
        for k in ('time','steps','last','_open','_dt'):
            if k in meta:setattr(p,k,meta[k])
            elif hasattr(p,k):delattr(p,k)

    def step(self,dt,time=None,*,commit=True):
        from scipy.sparse.linalg import LinearOperator,bicgstab
        if time is None:time=self.time
        if time!=self.time or not np.isfinite(dt) or dt<=0:raise ValueError('CPU bend trial clock differs')
        snapshot=self._snapshot();p=self.physics;before_calls=self.hydraulics.calls
        try:
            self._check();base=self.q.copy();self._envelope(base,dt);self._friction(base,.5*dt)
            p.begin(base,self.z,dt,time)
            tau=.5*dt;last={}
            def residual(flat):
                self._check();mid=flat.reshape(base.shape)
                if not np.isfinite(mid).all() or np.min(mid[0])<=20*p.parameters.dry:
                    raise BendTrialRejected('CPU implicit midpoint left fully-wet envelope')
                self._envelope(mid,dt)
                rate,water=self.hydraulics.rates(mid,self.z,dt,time+tau)
                p.add_momentum(mid,water,rate)
                last.update(rate=rate,water=water)
                return (mid-base-tau*rate).ravel()
            midpoint=base+tau*self.predictor if self.have_predictor else base.copy()
            work=np.empty_like(base);g=p.g;linear_calls=[0]
            def matvec(x):
                self._check();x=np.ascontiguousarray(x)
                p.engine.launch('bend_linear_acoustic',p.n,p.n,p.nf,float(p.parameters.gravity),base[0],x,
                    g['offset'],g['faces'],g['signs'],g['owner'],g['neighbour'],g['bid'],
                    g['normal'],g['length'],g['area'],work)
                linear_calls[0]+=1
                return x-tau*work.ravel()
            matrix=LinearOperator((base.size,base.size),matvec=matvec,dtype=np.float64)
            for iteration in range(31):
                defect=residual(midpoint.ravel());norm=float(np.linalg.norm(defect))
                scale=max(np.linalg.norm(midpoint-base),tau*np.linalg.norm(last['rate']))
                tolerance=max(self.nonlinear_atol*np.sqrt(base.size),self.nonlinear_rtol*scale)
                if norm<=tolerance:break
                if iteration==30:raise BendTrialRejected('CPU implicit midpoint did not converge')
                target=.2*tolerance
                delta,info=bicgstab(matrix,-defect,rtol=min(.01,target/norm),atol=min(1e-10,target),maxiter=400)
                if info!=0 or np.linalg.norm(matvec(delta)+defect)>max(target*1.05,1e-15):
                    raise BendTrialRejected('CPU acoustic linear iteration did not converge')
                midpoint+=delta.reshape(base.shape)
            # Build endpoint from authoritative shared faces, not unconverged iterate.
            qnew=base+dt*last['rate'];water=last['water'];self._friction(qnew,.5*dt)
            if np.min(qnew[0])<=20*p.parameters.dry:raise BendTrialRejected('CPU endpoint left wet envelope')
            face_ledger=np.where((p.mesh['neighbour']<0)&(p.mesh['bid']>=0),-dt*water*p.mesh['length'],0.)
            water_error=float(np.dot(p.mesh['area'],qnew[0]-base[0])-sum(face_ledger))
            if abs(water_error)>1e-10:raise BendTrialRejected('CPU water budget failed')
            p.finish(base,qnew,water,morphology=self.morphology);self._check()
            diagnostics=dict(**p.last,water_residual_m3=water_error,
                nonlinear_max_residual=float(np.max(abs(defect))),
                nonlinear_norm_residual=norm,nonlinear_tolerance=float(tolerance),
                nonlinear_iterations=iteration,linear_matvecs=linear_calls[0],
                fwave_evaluations=self.hydraulics.calls-before_calls)
            result=dict(q=qnew.copy(),z=p.ztrial.copy(),W=p.Wtrial.copy(),M=p.Mtrial.copy(),
                        base=base.copy(),midpoint=midpoint.copy(),water=water.copy(),diagnostics=diagnostics)
            if commit:
                p.commit();self.q[...]=qnew;self.z[...]=result['z'];self.water_ledger+=face_ledger
                self.predictor[...]=last['rate'];self.have_predictor=True
                self.time+=dt;self.steps+=1;self.last=diagnostics
            else:self._restore(snapshot)
            return result
        except BaseException:
            self._restore(snapshot);raise

    def checkpoint(self,path):
        from braids_gpu.atomic_output import save_arrays_atomic
        if self.physics._open:raise ValueError('Checkpoint requires accepted state')
        arrays,meta=self._snapshot()
        arrays.update(cpu_q=self.q,cpu_z=self.z,cpu_water_ledger=self.water_ledger,cpu_predictor=self.predictor)
        header=dict(identity=self.identity,physics=meta,time=self.time,steps=self.steps,
            morphology=self.morphology,last=self.last,have_predictor=self.have_predictor,
            hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
        save_arrays_atomic(Path(path),metadata=np.array(json.dumps(header,sort_keys=True)),**arrays)

    def restart(self,path):
        if self.physics._open:raise ValueError('Restart requires accepted state')
        expected,_=self._snapshot();expected.update(cpu_q=self.q,cpu_z=self.z,cpu_water_ledger=self.water_ledger,cpu_predictor=self.predictor)
        with np.load(path,allow_pickle=False) as f:
            header=json.loads(str(f['metadata']))
            if header['identity']!=self.identity:raise ValueError('CPU bend checkpoint identity differs')
            if set(f.files)!=set(expected)|{'metadata'}:raise ValueError('CPU checkpoint fields differ')
            arrays={k:f[k].copy() for k in expected}
            for k,v in arrays.items():
                if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype:raise ValueError('CPU checkpoint shape differs')
                if hashlib.sha256(v.tobytes()).hexdigest()!=header['hashes'][k]:raise ValueError('CPU checkpoint checksum differs')
            if header['time']!=header['physics']['time'] or header['physics']['_open']:
                raise ValueError('CPU checkpoint clock differs')
        self.q[...]=arrays.pop('cpu_q');self.z[...]=arrays.pop('cpu_z');self.water_ledger[...]=arrays.pop('cpu_water_ledger')
        self.predictor[...]=arrays.pop('cpu_predictor');self.have_predictor=header['have_predictor']
        self._restore((arrays,header['physics']))
        self.time=header['time'];self.steps=header['steps'];self.morphology=header['morphology'];self.last=header['last']
