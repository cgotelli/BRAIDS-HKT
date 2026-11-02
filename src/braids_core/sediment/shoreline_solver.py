"""Paired shoreline qualification of the maintained FLOW sediment bundle.

Shared explicit SSPRK22 with native/CUDA augmented f-wave spatial residuals.
Wet/dry and CFL rejection reduce the explicit timestep; no method fallback.
All fluid, bed and signed-secondary state stay on the selected backend.
"""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from braids_core.windows_paths import filesystem_path

from .bend_methods_v1 import BendTrialRejected, digest
from .flow2dh_current_bed import Flow2DHCurrentBed
from .flow2dh_fixed_bed import Flow2DHFixedBed
from . import bend_native_hydraulics as native
from . import wall_bed_hydraulics as wall

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def water_increment_roundoff_limit(inventory_m3,boundary_m3):
    """FP64 subtraction/reduction bound, consistent across domain scales.

    This is an arithmetic allowance, not a water source or a physical error
    tolerance. The fixed floor retains the historical small-domain screen.
    """
    values=(float(inventory_m3),float(boundary_m3))
    if not all(np.isfinite(value) for value in values):
        raise BendTrialRejected('Nonfinite water budget scale')
    scale=max(1.,*(abs(value) for value in values))
    return max(1.e-10,64*np.finfo(float).eps*scale)


def build_native():
    library,parent,compiler=wall.build()
    bridge=HERE/'shoreline_native_bridge.inc'
    from .river_exterior_bed import native_source
    code=native_source((library.parent/'cutcell_wall_bed.f90').read_text())
    # Export already-computed wave speeds for the explicit face-sum CFL.
    # Do not change the augmented Riemann operator or recompute its flux.
    marker='\ncontains\n'
    if code.count(marker)!=1:raise ValueError('Native module marker differs')
    code=code.replace(marker,'\n real(8),allocatable :: river_face_speed(:)\n'+marker)
    marker='  next=0.d0;accepted=0.d0'
    if code.count(marker)!=1:raise ValueError('Native wave speed marker differs')
    code=code.replace(marker,'  if(.not.allocated(river_face_speed)) allocate(river_face_speed(nf))\n'
        '  river_face_speed=speed\n'+marker)
    end=code.rindex('\nend module')
    code=code[:end]+'''\nsubroutine river_get_face_speed(values) bind(C)
 use iso_c_binding
 real(c_double),intent(out)::values(nf)
 values=river_face_speed
end subroutine
'''+code[end:]
    transform_sha=hashlib.sha256(code.encode()).hexdigest()
    key=hashlib.sha256((digest(library)+digest(bridge)+transform_sha).encode()).hexdigest()[:20]
    dst=ROOT/'build/shoreline_hydraulics'/key;dst.mkdir(parents=True,exist_ok=True)
    output=dst/library.name
    if not output.exists():
        from tools.build_cutcell import SOURCES
        end=code.rindex('\nend module')
        generated=dst/'shoreline.f90'
        generated.write_text(code[:end]+'\n'+bridge.read_text()+'\nend module\n')
        deps=[library.parent/'source'/Path(p).name for p in SOURCES[:-1]]
        command=[compiler,'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
                 *map(str,deps),str(generated),'-o',str(output)]
        if os.name=='nt':command+=['-Wl,--export-all-symbols']
        result=subprocess.run(command,cwd=dst,capture_output=True,text=True,timeout=180)
        (dst/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(dst/'build.log'))
        (dst/'identity.json').write_text(json.dumps(dict(parent=parent,bridge_sha256=digest(bridge),exterior_transform_sha256=transform_sha,
            generated_sha256=digest(generated),library_sha256=digest(output),command=command),indent=2))
    identity=json.loads((dst/'identity.json').read_text())
    if digest(output)!=identity['library_sha256']:raise ValueError('Shoreline library changed')
    return output,identity,compiler


class NativeShoreline(native.NativeHydraulics):
    def __init__(self,deck,data,threads):
        with native._lock:
            old=native.build
            try:
                native.build=build_native
                super().__init__(deck,data,threads)
            finally:native.build=old
        self.lib.shoreline_native_euler.argtypes=self.lib.bend_native_rates.argtypes
        self.lib.shoreline_native_euler.restype=None

    def euler(self,q,z,dt,time):
        q=np.ascontiguousarray(q);z=np.ascontiguousarray(z)
        result=np.empty_like(q);water=np.empty(self.nf);code=ctypes.c_int()
        with native._lock:
            self.lib.shoreline_native_euler(q.ctypes.data,z.ctypes.data,float(dt),float(time),
                self.threads,result.ctypes.data,water.ctypes.data,ctypes.byref(code))
        self.calls+=1
        if code.value:raise BendTrialRejected('Native shoreline Euler CFL/positivity rejection')
        return result,water


class ShorelinePhysics(Flow2DHCurrentBed):
    """Dry-state extension; wet interior retains every accepted correction.

    Division-safe scratch is never water state. Inactive cells have zero
    velocity, secondary source and sediment capacity. With ThetSD=0 no bed
    material crosses an inactive sediment face. No bank/collapse transfer.
    """
    def __init__(self,*args,sediment_depth=.001,**kwargs):
        if not np.isfinite(sediment_depth) or sediment_depth<.001:
            raise ValueError('Explicit sediment depth >= 1 mm required')
        super().__init__(*args,allow_nonflat=True,**kwargs)
        self.sediment_depth=float(sediment_depth)
        self.effective_wet_weights=self.xp.zeros_like(self.effective_weights)
        self.identity.update(shoreline_sha256=digest(__file__),scope='fixed_grid_single_grain_EH_shoreline_research',
            sediment_depth_m=sediment_depth,shoreline_transfer='zero_at_inactive_face_ThetSD0',
            viscous_shoreline='free_slip_zero_viscous_exchange_across_hydraulically_dry_faces',
            secondary_dry_sink='explicitly_budgeted_hI_reset',public_support=False,
            sediment_roughness_connectivity='physical wet-neighbour surface gradients and face weights; dry scratch excluded')

    def effective_inputs(self,q,z):
        # Constitutive q may contain positive scratch in inactive sediment
        # cells. Hydraulic connectivity and free-surface slopes must not.
        raw=getattr(self,'_effective_physical_q',q);g=self.g;n=self.n;nf=self.nf
        self.effective.launch('effective_wet_gradient',n,n,self.parameters.dry,raw,z,
            g['offset'],g['faces'],g['owner'],g['neighbour'],g['xy'],self.effective_gradient)
        self.effective.launch('effective_face_depth',nf,nf,n,raw,z,self.effective_gradient,
            g['owner'],g['neighbour'],g['bid'],g['distance'],g['displacement'],self.effective_face)
        self.effective.launch('effective_wet_weights',nf,nf,n,self.parameters.dry,raw,
            g['owner'],g['neighbour'],g['bid'],self.effective_weights,self.effective_wet_weights)
        self.effective.launch('effective_cell_shear',n,n,nf,self.chezy,q,self.effective_face,self.effective_wet_weights,
            g['offset'],g['faces'],g['owner'],g['neighbour'],g['bid'],self.effective_fields,self.effective_q)
        if not bool(self.xp.all(self.xp.isfinite(self.effective_fields))):raise ValueError('Invalid shoreline effective shear')

    def safe(self,q,threshold=None):
        threshold=self.parameters.dry if threshold is None else threshold
        wet=q[0]>threshold
        result=q.copy();result[0]=self.xp.where(wet,q[0],.001001)
        result[1:]=self.xp.where(wet,q[1:],0.)
        return result,wet

    def local(self,q,z,olddepth,hi):
        safe,wet=self.safe(q)
        out=super().local(safe,z,self.xp.maximum(olddepth,.001001),self.xp.where(wet,hi,0.))
        out[:,~wet]=0.
        return out

    def friction(self,q,dt,*args):
        safe,wet=self.safe(q)
        super().friction(safe,dt)
        q[1:]=self.xp.where(wet,safe[1:],0.)

    def begin(self,q,z,dt,time):
        if self._open or time!=self.time or not np.isfinite(dt) or dt<=0:
            raise ValueError('Invalid shoreline trial clock')
        self._dt=float(dt);self.z0[:]=z;self.hold[:]=q[0];self.Wbase[:]=self.W
        self.refresh_secondary_momentum(q,self.W)
        self._open=True

    def refresh_secondary_momentum(self,q,hi):
        """Same physical tensor, refreshed at the actual hydraulic clock."""
        self.hold[:]=q[0]
        self.local(q,self.z0,q[0],hi)
        clock=getattr(self,'_secondary_clock',None)
        if clock is not None:clock.correct_stress()
        self.frozenstress[:]=self.outputs[13:15]
        safe,_=self.safe(q)
        self.mobile.launch('flow_directional_geometric',self.n,self.n,safe,self.outputs,self.directional_B,self.directional_source)
        self.geometry_Bstate[:]=q[0]*self.directional_B
        self._gradients(safe,self.geometry_Bstate,self.geometry_Bgrad)

    def add_momentum(self,q,water,rate):
        safe,wet=self.safe(q);scratch=self.xp.zeros_like(rate)
        super().add_momentum(safe,water,scratch)
        if self.nu and not bool(self.xp.all(wet)):
            # Division-safe dry scratch is not a stagnant water column.
            # The fully wet operator's velocity difference otherwise imposes
            # unintended no-slip drag on an internal dry-bank interface.
            # Remove ONLY this viscous contribution. Keep the established
            # secondary stress/source and all wet-wet geometry unchanged.
            xp=self.xp;g=self.g;o=g['owner'];j=g['neighbour']
            closed=(~wet[o])|((j>=0)&(~wet[xp.maximum(j,0)]))
            viscous=xp.empty_like(self.momentumflux)
            self.geometry.launch('geometry_momentum_faces',self.nf,self.nf,self.n,
                0,int(self.correct_viscosity),self.nu,safe,xp.zeros(self.n),
                self.geometry_grad,self.geometry_Bgrad,o,j,g['normal'],g['distance'],
                g['xy'],g['displacement'],viscous)
            viscous*=xp.where(closed,-1.,0.)
            self.flow.launch('flow_momentum_gather',self.n,self.n,self.nf,
                self.hold,viscous,g['offset'],g['faces'],g['signs'],g['length'],g['area'],scratch)
        rate[1:]+=self.xp.where(wet,scratch[1:],0.)

    def scalar_faces(self,q,hi,water):
        safe,wet=self.safe(q)
        super().scalar_faces(safe,self.xp.where(wet,hi,0.),water)
        # Do not diffuse signed intensity across a dry interface; advection
        # into newly inundated cells is handled by the shared water flux.
        g=self.g
        self.flow.launch('flow_shoreline_scalar_faces',self.nf,self.nf,self.parameters.dry,
            q,safe,hi,self.outputs[5],water,g['owner'],g['neighbour'],g['bid'],self.scalarflux)

    def bed_faces(self,q,z,hi):
        safe,wet=self.safe(q,self.sediment_depth)
        self._effective_physical_q=q
        try:super().bed_faces(safe,z,self.xp.where(wet,hi,0.))
        finally:del self._effective_physical_q
        g=self.g;o=g['owner'];j=g['neighbour']
        active=wet[o]&self.xp.where(j>=0,wet[self.xp.maximum(j,0)],True)
        self.aux[2]*=active

    def finish(self,qold,qnew,water,*,morphology=True):
        Flow2DHFixedBed.finish(self,qold,qnew,water,morphology=False)
        xp=self.xp;g=self.g;dry=qnew[0]<=self.parameters.dry
        removed=float(xp.sum(xp.where(dry,self.Wtrial,0.)*g['area']))
        self.Wtrial[dry]=0.;self.last['secondary_source']-=removed
        self.last['secondary_dry_sink']=removed
        if not morphology:return self.last
        dt=self._dt*self.morfac;phi=1-self.parameters.porosity
        self.bed_faces(qnew,self.z0,self.Wtrial)
        self.available[:]=phi*(self.z0-self.rock)
        donor=self.limit(self.available,dt)
        self.face_integral[:]=dt*self.aux[2]
        self.ztrial[:]=self.z0+(dt/phi)*self.divergence(2)
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=-xp.sum(self.face_integral*g['length']*outside)
        residual=xp.sum(phi*(self.ztrial-self.z0)*g['area'])-boundary
        change=float(xp.max(xp.abs(self.ztrial-self.z0)/xp.maximum(qnew[0],self.sediment_depth)))
        if (not bool(xp.all(xp.isfinite(self.ztrial))) or bool(xp.any(self.ztrial<self.rock-1e-13))
                or abs(float(residual))>1e-10 or change>.05):
            raise BendTrialRejected('Shoreline sediment budget/positivity/increment rejection')
        self.last.update(morphology=True,solid_volume_residual_m3=float(residual),solid_boundary_m3=float(boundary),
            donor_factor_min=donor,maximum_relative_bed_increment=change,morphfac=self.morfac)
        return self.last


class ShorelineSolver:
    def __init__(self,deck,data,*,backend,controls,bedbc,feed,gpu_solver_type=None,source_folder=None,
                 native_hydraulics=None):
        from .current_river_solver import validate_current_river_input
        # Reuse all duplicate-physics/material checks. The ONLY different
        # admissions are non-flat bed and nonnegative depth, validated below.
        projected=dict(data,bed=np.zeros_like(data['bed']),state=np.array(data['state'],copy=True))
        projected['state'][0]=np.maximum(projected['state'][0],1.)
        validate_current_river_input(projected,controls)
        if (not np.isfinite(data['state']).all() or np.any(data['state'][0]<0)
                or not np.isfinite(data['bed']).all()):raise ValueError('Invalid shoreline input')
        if backend not in ('cpu','gpu'):raise ValueError('Choose CPU or GPU explicitly')
        self.backend=backend;self.data=data;self.controls=controls
        self.physics=ShorelinePhysics(data,backend=backend,chezy=controls.chezy,diameter=controls.diameter,
            bedbc=bedbc,feed=feed,morfac=controls.morfac,betac=controls.betac,
            viscosity=controls.viscosity,diffusivity=controls.diffusivity,threads=controls.threads)
        self.xp=self.physics.xp;xp=self.xp
        self.q=xp.asarray(data['state'][:3]).copy();self.z=xp.asarray(data['bed']).copy()
        if backend=='cpu':
            constructor=NativeShoreline if native_hydraulics is None else native_hydraulics
            self.hydraulics=constructor(deck,data,controls.threads);hydroid=self.hydraulics.identity
        else:
            if native_hydraulics is not None:raise ValueError('Native constructor cannot replace the explicit GPU backend')
            if gpu_solver_type is None or source_folder is None:raise ValueError('Authenticated GPU sources required')
            self.hydraulics=gpu_solver_type(data)
            # Retain the authenticated one-stage operator for spatial checks.
            # Its carrier's SSPRK33 clock is NEVER used by this owner.
            core=[c for c in type(self.hydraulics).__mro__ if c.__name__=='CoupledSolver']
            if len(core)!=1:raise ValueError('Unknown CUDA Euler interface')
            self.gpu_euler=core[0].euler
            class Holder:identity={}
            holder=Holder();wall.install(self.hydraulics,holder,'gpu',deck,data,source_folder)
            hydroid=dict(source_sha256=digest(Path(source_folder)/'general_live_transport_sources.json'),**holder.identity)
        self.time=0.;self.steps=0;self.morphology=False;self.cancelled=lambda:False;self.last={}
        self.water_ledger=xp.zeros(data['nf'])
        self.identity=dict(schema=2,
            physics=self.physics.identity,hydraulics=hydroid,source_sha256=digest(__file__),
            input_sha256={p.name:digest(p) for p in sorted(filesystem_path(deck).parent.glob('case.data*'))},
            backend=backend,hydraulic_qualification='explicit_qualification125_and_resistance126',public_qgis_admission=False)
        from .river_explicit import install
        install(self,controls.cfl)

    def host(self,a):return np.asarray(a) if self.backend=='cpu' else self.xp.asnumpy(a)

    def snapshot(self):
        p=self.physics;xp=self.xp
        return ({k:v.copy() for k,v in vars(p).items() if isinstance(v,xp.ndarray)},
            dict(copy.deepcopy({k:getattr(p,k) for k in ('time','steps','last','_open','_dt') if hasattr(p,k)}),
                 midpoint_state=self.midpoint.state()))

    def _trial_snapshot(self):
        """Private single-step workspace; public snapshots remain detached.

        Copy every physics array, including newly installed optional features.
        Reuse allocation only, never an old value. Shapes/dtypes may change on
        a rebuilt owner; no view into this workspace escapes through step().
        """
        if getattr(self,'_trial_snapshot_in_use',False):
            raise RuntimeError('A shoreline trial is already active')
        xp=self.xp;p=self.physics;pool=getattr(self,'_trial_snapshot_arrays',{})
        arrays={}
        for k,v in vars(p).items():
            if not isinstance(v,xp.ndarray):continue
            saved=pool.get(k)
            if saved is None or saved.shape!=v.shape or saved.dtype!=v.dtype:
                saved=xp.empty_like(v)
            saved[...]=v;arrays[k]=saved
        meta=copy.deepcopy({k:getattr(p,k) for k in ('time','steps','last','_open','_dt') if hasattr(p,k)})
        meta['midpoint_state']=self.midpoint.state()
        self._trial_snapshot_arrays=arrays;self._trial_snapshot_in_use=True
        return arrays,meta

    def restore(self,snapshot):
        arrays,meta=snapshot
        self.midpoint.validate_state(meta['midpoint_state'])
        for k,v in arrays.items():getattr(self.physics,k)[...]=v
        for k,v in meta.items():
            if k!='midpoint_state':setattr(self.physics,k,v)
        self.midpoint.restore(meta['midpoint_state'])
        if '_dt' not in meta and hasattr(self.physics,'_dt'):delattr(self.physics,'_dt')

    def check(self):
        if self.cancelled():raise InterruptedError('Shoreline trial cancelled')

    def euler(self,q,dt,time):
        self.check();xp=self.xp;h=self.hydraulics
        if self.backend=='cpu':nextq,water=h.euler(q,self.z,dt,time)
        else:
            state=h.stepper.stage
            for k,v in h.state.items():state[k][...]=v
            state['q'][...]=q;state['layers'][0,0]=1590.*(self.z-self.physics.rock)
            h.boundary.begin_trial()
            trial=self.gpu_euler(h,state,dt,time)
            if bool(xp.any(trial['status'])):raise BendTrialRejected('CUDA shoreline Euler rejected')
            nextq=trial['state']['q'].copy();water=h.flux.buffers['fl'][0].copy()
        rate=xp.zeros_like(q);self.physics.add_momentum(q,water,rate);nextq+=dt*rate
        nextq[1:,nextq[0]<=self.physics.parameters.dry]=0.
        if not bool(xp.all(xp.isfinite(nextq))) or bool(xp.any(nextq[0]<0)):
            raise BendTrialRejected('Shoreline Euler invalid')
        return nextq,water

    def step(self,dt,time=None,*,commit=True):
        if time is not None and time!=self.time:raise ValueError('Shoreline clock differs')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive timestep required')
        p=self.physics;xp=self.xp;snapshot=self._trial_snapshot()
        try:
            # Predict accepted-start roughness/bedforms once. Actual drag uses
            # the hydraulic substep clock, including temporal-error trials.
            self.check();base=self.q.copy();p.friction(base,0.);p.begin(base,self.z,dt,self.time)
            nextq,water,integration=self.midpoint.advance(base,dt,self.time)
            outside=(p.g['neighbour']<0)&(p.g['bid']>=0)
            ledger=-dt*water*p.g['length']*outside
            residual=float(xp.sum((nextq[0]-self.q[0])*p.g['area'])-xp.sum(ledger))
            inventory=float(xp.sum(xp.maximum(self.q[0],nextq[0])*p.g['area']))
            roundoff=water_increment_roundoff_limit(inventory,float(xp.sum(xp.abs(ledger))))
            if not np.isfinite(residual) or abs(residual)>roundoff:
                raise BendTrialRejected(f'Shoreline water budget failed: residual={residual:.6e} m3, roundoff={roundoff:.6e} m3')
            p.finish(base,nextq,water,morphology=self.morphology);self.check()
            result=dict(q=nextq.copy(),z=p.ztrial.copy(),hI=p.Wtrial.copy(),
                diagnostics=dict(p.last,water_residual_m3=residual,water_increment_roundoff_limit_m3=roundoff,**integration))
            if commit:
                p.commit();self.q[...]=nextq;self.z[...]=result['z'];self.water_ledger+=ledger
                self.time+=dt;self.steps+=1;self.last=result['diagnostics']
            else:self.restore(snapshot)
            return result
        except BaseException:
            self.restore(snapshot);raise
        finally:self._trial_snapshot_in_use=False

    def checkpoint_bytes(self):
        from braids_gpu.atomic_output import arrays_to_bytes
        if self.physics._open:raise ValueError('Checkpoint needs accepted state')
        arrays,meta=self.snapshot();arrays={k:self.host(v) for k,v in arrays.items()}
        arrays.update(saved_q=self.host(self.q),saved_z=self.host(self.z),water_ledger=self.host(self.water_ledger))
        header=dict(identity=self.identity,physics=meta,time=self.time,steps=self.steps,morphology=self.morphology,
                    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
        return arrays_to_bytes(metadata=np.array(json.dumps(header,sort_keys=True)),**arrays)

    def checkpoint(self,path):
        from braids_gpu.atomic_output import write_bytes_atomic
        write_bytes_atomic(path,self.checkpoint_bytes())

    def restart(self,path):
        if self.physics._open:raise ValueError('Restart needs accepted state')
        arrays,meta=self.snapshot();arrays.update(saved_q=self.q,saved_z=self.z,water_ledger=self.water_ledger)
        with np.load(path,allow_pickle=False) as saved:
            header=json.loads(str(saved['metadata']))
            if header['identity']!=self.identity or set(saved.files)!=set(arrays)|{'metadata'}:
                raise ValueError('Shoreline checkpoint identity/schema differs')
            self.midpoint.validate_state(header['physics']['midpoint_state'])
            for k,v in arrays.items():
                a=saved[k]
                if (a.shape!=v.shape or a.dtype!=v.dtype or
                        hashlib.sha256(a.tobytes()).hexdigest()!=header['hashes'][k]):
                    raise ValueError('Shoreline checkpoint array differs')
            # Validate all arrays before mutation.
            for k,v in arrays.items():v[...]=self.xp.asarray(saved[k])
        # snapshot() returns detached copies of the physics arrays. Loading
        # those copies above does not restore the live secondary state,
        # transfer histories or ledgers. Publish every authenticated physics
        # array, not just its scalar metadata, on either execution backend.
        self.restore(({k:v for k,v in arrays.items()
                       if k not in ('saved_q','saved_z','water_ledger')},header['physics']))
        self.time=header['time'];self.steps=header['steps'];self.morphology=header['morphology']
