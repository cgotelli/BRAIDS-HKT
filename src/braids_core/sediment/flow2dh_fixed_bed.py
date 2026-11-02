"""Explicit classic FLOW fixed-bed owner on the paired implicit f-wave drivers.

Hydraulic qualification route, not a morphodynamic or QGIS default. Local
constitutive equations reuse the existing Fortran/CUDA FLOW operators. New
cell/face glue is compiled OpenMP or CUDA, with no automatic backend fallback.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .bend_methods_v1 import BendPhysics, Parameters, BendTrialRejected, digest, OPTIONS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAMES=('flow_inputs','flow_drag','flow_momentum_faces','flow_momentum_gather',
       'flow_scalar_faces','flow_scalar_gather','flow_shoreline_scalar_faces')


class FlowEngine:
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        kernel=HERE/'flow2dh_fixed_bed_kernels.h'
        self.identity=dict(backend=backend,threads=threads,kernels_sha256=digest(kernel))
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=kernel.read_text(),options=OPTIONS,name_expressions=NAMES)
            self.fn={name:self.module.get_function(name) for name in NAMES}
            from braids_gpu.flow2dh_secondary_operator import Flow2DHSecondaryCUDA
            self.operator=Flow2DHSecondaryCUDA()
            self.identity['operator_sha256']=digest(ROOT/'braids_gpu/flow2dh_secondary_operator.cu')
            return
        if backend!='cpu':raise ValueError('Explicit CPU or GPU backend required')
        cpp=shutil.which('g++');fc=shutil.which('gfortran')
        if not cpp or not fc:raise RuntimeError('Native C++ and Fortran compilers required')
        sources=[ROOT/'solver/wave/braids_flow2dh_secondary_operator_module.f90',
                 ROOT/'solver/wave/braids_flow2dh_secondary_c_api.f90']
        versions=[subprocess.check_output([c,'--version'],text=True).splitlines()[0] for c in (cpp,fc)]
        flags=['-O3','-fopenmp','-shared','-ffp-contract=off']
        key=hashlib.sha256((str(versions)+str(flags)+kernel.read_text()+''.join(digest(s) for s in sources)).encode()).hexdigest()[:20]
        folder=ROOT/'build/flow2dh_fixed_bed'/key;folder.mkdir(parents=True,exist_ok=True)
        self.dirs=[os.add_dll_directory(str(Path(fc).parent))] if os.name=='nt' else []
        lib=folder/'flow_glue.dll';native=folder/'flow_local.dll'
        if not lib.exists():subprocess.run([cpp,'-x','c++','-std=c++11',*flags,'-static-libgcc','-static-libstdc++',str(kernel),'-o',str(lib)],capture_output=True,check=True,timeout=120)
        if not native.exists():subprocess.run([fc,*flags,'-ffree-line-length-none',*[str(s) for s in sources],'-J',str(folder),'-o',str(native)],cwd=folder,capture_output=True,check=True,timeout=120)
        self.lib=ctypes.CDLL(str(lib));self.local=ctypes.CDLL(str(native))
        self.fn={name:getattr(self.lib,name) for name in NAMES}
        for fn in self.fn.values():fn.restype=None
        self.local.braids_flow2dh_batch.restype=None
        self.identity.update(compiler=versions,flags=flags,local_sources={str(s.relative_to(ROOT)):digest(s) for s in sources},
                             libraries={p.name:digest(p) for p in (lib,native)})

    def launch(self,name,n,*args):
        if self.backend=='gpu':
            args=tuple(np.int32(a) if isinstance(a,int) else np.float64(a) if isinstance(a,float) else a for a in args)
            self.fn[name](((n+127)//128,),(128,),args)
        else:
            self.lib.flow_threads(ctypes.c_int(self.threads))
            args=tuple(ctypes.c_int(a) if isinstance(a,int) else ctypes.c_double(a) if isinstance(a,float) else ctypes.c_void_p(a.ctypes.data) for a in args)
            self.fn[name](*args)


class Flow2DHFixedBed(BendPhysics):
    def __init__(self,data,*,backend,chezy,betac=0.,viscosity=0.,diffusivity=0.,threads=4,parameters=None,allow_nonflat=False):
        if not np.isfinite([chezy,betac,viscosity,diffusivity]).all() or chezy<1.e-6 or min(viscosity,diffusivity)<0:
            raise ValueError('Invalid classic FLOW fixed-bed coefficients')
        if not 0<=betac<=1:raise ValueError('FLOW Betac must be in [0,1]')
        if diffusivity!=0:raise ValueError('Nonzero scalar diffusion is not qualified on this fixed-bed route')
        if (np.any(data['bed']!=0) and not allow_nonflat) or data['amr_enabled']:
            raise ValueError('Current qualification route requires fixed flat bed and fixed grid')
        super().__init__(data,backend=backend,method='lv_odgaard',threads=threads,
            parameters=parameters or Parameters(manning=float(data['manning_cell'][0]),ripple_factor=0.,wall_sand_roughness=0.),
            scalar_diffusion=False)
        self.flow=FlowEngine(backend,threads);xp=self.xp;n=self.n
        self.chezy=float(chezy);self.betac=float(betac);self.nu=float(viscosity);self.diff=float(diffusivity)
        self.dx=float(data['dx']);self.inputs=xp.zeros((12,n));self.outputs=xp.zeros((15,n))
        self.flags=xp.zeros((5,n),dtype=xp.int32);self.active=xp.ones(n,dtype=xp.uint8)
        self.controls=xp.array([self.parameters.dry,2.5,20.,self.parameters.gravity,.41,1.,betac,np.finfo(float).eps])
        self.hold=xp.zeros(n);self.scalarflux=xp.zeros(self.nf);self.scalarflux_integral=xp.zeros(self.nf)
        self.momentumflux=xp.zeros((2,self.nf));self.frozenstress=xp.zeros((2,n))
        self.source_increment=xp.zeros(n)
        self.gpu_result=None
        if backend=='gpu':
            self.gpu_result=self.flow.operator.allocate_result(n)
            # Expose outputs to the solver's transaction inventory.
            for name in self.flow.operator._OUTPUT_NAMES:setattr(self,'local_'+name,getattr(self.gpu_result,name))
            for name in self.flow.operator._FLAG_NAMES:setattr(self,'local_'+name,getattr(self.gpu_result,name))
            self.local_status=self.gpu_result.status
        self.identity=dict(schema=3,method='classic_flow2dh_fixed_bed_v3',scope='fully_wet_fixed_flat_bed',
            backend=backend,runtime=self.flow.identity,wrapper_sha256=digest(__file__),chezy=self.chezy,
            betac=self.betac,viscosity_m2_s=self.nu,diffusivity_m2_s=self.diff,curvature='clockwise_positive_velocity_LS',
            source='existing_native_and_cuda_classic_FLOW_operator',state='signed_hI_m2_s',
            transport='limited_linear_Heun_shared_water_faces_then_old_depth_source_new_depth_BE_sink',
            stress='FLOW_manual_9.151-153_objective_Cartesian_tensor; vihsec_velocity_acceleration_times_accepted_start_depth',
            boundaries='adjacent equilibrium on incoming open faces; owner outgoing; zero wall flux',
            momentum_boundary='owner-extrapolated secondary stress following nonzero vihsec boundary stress; free-slip viscous flux',
            morphology=False,public_support=False)

    def local(self,q,z,olddepth,hi):
        e,g,n=self.engine,self.g,self.n
        e.launch('gradients',n,n,q,z,hi,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.grad)
        self.flow.launch('flow_inputs',n,n,self.dx,self.chezy,q,olddepth,hi,self.grad,self.inputs)
        if self.engine.backend=='cpu':
            self.flow.local.braids_flow2dh_batch(ctypes.c_int(n),ctypes.c_int(self.flow.threads),
                ctypes.c_void_p(self.inputs.ctypes.data),ctypes.c_void_p(self.controls.ctypes.data),
                ctypes.c_void_p(self.outputs.ctypes.data),ctypes.c_void_p(self.flags.ctypes.data))
            if np.any(self.flags[4]):raise BendTrialRejected('Invalid FLOW local state')
        else:
            op=self.flow.operator
            result=op.evaluate(self.active,*list(self.inputs),dryflc=self.parameters.dry,rmincf=2.5,chzmin=20.,
                gravity=self.parameters.gravity,von_karman=.41,espir=1.,betac=self.betac,
                epsilon_value=np.finfo(float).eps,out=self.gpu_result)
            for k,name in enumerate(op._OUTPUT_NAMES):self.outputs[k]=getattr(result,name)
        return self.outputs

    def friction(self,q,dt,*args):
        self.flow.launch('flow_drag',self.n,self.n,float(dt),self.parameters.gravity,self.chezy,q)

    def initialize_secondary(self,q,z):pass

    def begin(self,q,z,dt,time):
        if self._open or time!=self.time or not np.isfinite(dt) or dt<=0:raise ValueError('Invalid FLOW trial clock')
        if bool(self.xp.any(q[0]<=20*self.parameters.dry)):raise BendTrialRejected('Fully wet FLOW scope only')
        self._dt=float(dt);self.z0[:]=z;self.hold[:]=q[0];self.Wbase[:]=self.W
        self.local(q,z,q[0],self.W);self.frozenstress[:]=self.outputs[13:15];self._open=True

    def add_momentum(self,q,water,rate):
        g=self.g;n=self.n;nf=self.nf
        self.flow.launch('flow_momentum_faces',nf,nf,n,self.nu,q,*self.frozenstress,
            g['owner'],g['neighbour'],g['normal'],g['distance'],self.momentumflux)
        self.flow.launch('flow_momentum_gather',n,n,nf,self.hold,self.momentumflux,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],rate)

    def scalar_faces(self,q,hi,water):
        e,g,n,nf=self.engine,self.g,self.n,self.nf
        e.launch('gradients',n,n,q,self.z0,hi,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.scalar_grad)
        self.flow.launch('flow_scalar_faces',nf,nf,n,self.diff,q,hi,self.scalar_grad,self.outputs[5],water,
            g['owner'],g['neighbour'],g['bid'],g['distance'],g['displacement'],self.scalarflux)

    def finish(self,qold,qnew,water,*,morphology=False):
        if morphology:raise ValueError('Classic FLOW fixed-bed qualification forbids morphology')
        if not self._open:raise ValueError('No open FLOW trial')
        xp=self.xp;n=self.n;g=self.g;dt=self._dt
        clock=getattr(self,'_secondary_clock',None)
        if clock is not None and clock.active:
            boundary,source,residual=clock.finish()
            self.ztrial[:]=self.z0;self.Mtrial[:]=self.M;self.face_integral.fill(0)
            self.last=dict(morphology=False,solid_volume_residual_m3=0.,solid_boundary_m3=0.,
                secondary_residual=residual,secondary_source=source,secondary_projection=0.,
                secondary_boundary=boundary,secondary_substeps=int(self.secondary_clock_stats[1]),
                secondary_cfl_max=float(self.secondary_clock_stats[2]),
                secondary_minimum_dt=float(self.secondary_clock_stats[4]),
                secondary_substep_dry_sink=float(self.secondary_clock_stats[3]))
            return self.last
        self.local(qnew,self.z0,qold[0],self.W)
        self.scalar_faces(qold,self.W,water);self.scalarflux_integral[:]=.5*dt*self.scalarflux
        self.flow.launch('flow_scalar_gather',n,n,dt,self.W,self.scalarflux,g['offset'],g['faces'],g['signs'],g['length'],g['area'],self.work)
        self.scalar_faces(qnew,self.work,water);self.scalarflux_integral[:]+=.5*dt*self.scalarflux
        self.flow.launch('flow_scalar_gather',n,n,dt,self.W,self.scalarflux,g['offset'],g['faces'],g['signs'],g['length'],g['area'],self.work2)
        self.work2[:]=.5*(self.work+self.work2)
        self.Wtrial[:]=(self.work2+dt*qold[0]*self.outputs[3])/(1.+dt*self.outputs[4])
        self.source_increment[:]=self.Wtrial-self.work2
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=-xp.sum(self.scalarflux_integral*g['length']*outside)
        source=xp.sum(self.source_increment*g['area'])
        residual=xp.sum((self.Wtrial-self.W)*g['area'])-boundary-source
        if not bool(xp.all(xp.isfinite(self.Wtrial))) or abs(float(residual))>1e-10:
            raise BendTrialRejected('FLOW signed secondary budget failed')
        self.ztrial[:]=self.z0;self.Mtrial[:]=self.M;self.face_integral.fill(0)
        self.last=dict(morphology=False,solid_volume_residual_m3=0.,solid_boundary_m3=0.,
            secondary_residual=float(residual),secondary_source=float(source),secondary_projection=0.,
            secondary_boundary=float(boundary))
        return self.last

    def commit(self):
        self.W[:]=self.Wtrial
        outside=(self.g['neighbour']<0)&(self.g['bid']>=0)
        self.secondary_boundary_ledger[:]-=self.scalarflux_integral*self.g['length']*outside
        self.secondary_source_ledger[...]+=self.last['secondary_source']
        self.time+=self._dt;self.steps+=1;self._open=False


def bind(solver,owner,physics,backend):
    owner.physics=physics
    if backend=='cpu':
        solver._friction=physics.friction;solver.identity['physics']=physics.identity
    else:
        owner.resident={k:v for k,v in vars(physics).items() if isinstance(v,physics.xp.ndarray)}
        owner.resident['geometry']=physics.g;owner.identity['component']=physics.identity
        solver.implicit_midpoint._friction_half_step=physics.friction
