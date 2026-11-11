"""Isolated paired classic FLOW secondary + EH + KF + conservative Exner owner.

Starts from a flat bed, then evolves it on a fixed mesh. All physical states
are computed by BRAIDS, never supplied by a reference trajectory. Not QGIS.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .bend_methods_v1 import Parameters, BendTrialRejected, digest, OPTIONS
from .flow2dh_fixed_bed import Flow2DHFixedBed
from .flow2dh_eh_operator import ClassicFlowEH

ROOT=Path(__file__).resolve().parents[2]
NAMES=('flow_bed_face_inputs','flow_bed_flux','flow_directional_faces','flow_directional_geometric')


class MobileEngine:
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        path=Path(__file__).with_name('flow2dh_mobile_kernels.h')
        self.identity=dict(source_sha256=digest(path),threads=threads,backend=backend)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=path.read_text(),options=OPTIONS,name_expressions=NAMES)
            self.fn={k:self.module.get_function(k) for k in NAMES}
        else:
            fc=shutil.which('g++')
            if not fc:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([fc,'--version'],text=True).splitlines()[0]
            key=hashlib.sha256((version+path.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_mobile'/key;folder.mkdir(parents=True,exist_ok=True)
            lib=folder/'mobile.dll'
            if not lib.exists():subprocess.run([fc,'-x','c++','-std=c++11','-O3','-fopenmp','-shared',
                '-static-libgcc','-static-libstdc++','-ffp-contract=off',str(path),'-o',str(lib)],capture_output=True,check=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(fc).parent))] if os.name=='nt' else []
            self.lib=ctypes.CDLL(str(lib));self.fn={k:getattr(self.lib,k) for k in NAMES}
            for f in self.fn.values():f.restype=None
            self.identity.update(compiler=version,library_sha256=digest(lib))

    def launch(self,name,n,*args):
        if self.backend=='gpu':
            self.fn[name](((n+127)//128,),(128,),tuple(np.int32(a) if isinstance(a,int)
                else np.float64(a) if isinstance(a,float) else a for a in args))
        else:
            self.lib.mobile_threads(ctypes.c_int(self.threads))
            self.fn[name](*(ctypes.c_int(a) if isinstance(a,int) else ctypes.c_double(a)
                if isinstance(a,float) else ctypes.c_void_p(a.ctypes.data) for a in args))


class Flow2DHMobileBed(Flow2DHFixedBed):
    def __init__(self,data,*,backend,chezy,diameter,bedbc,feed,morfac=1.,betac=1.,
                 viscosity=0.,diffusivity=0.,espir=1.,threads=4,allow_nonflat=False):
        if not np.isfinite([diameter,morfac,diffusivity]).all() or min(diameter,morfac)<=0 or diffusivity<0:
            raise ValueError('Invalid mobile-bed controls')
        p=Parameters(manning=float(data['manning_cell'][0]),diameter=diameter,ripple_factor=0.,wall_sand_roughness=0.)
        super().__init__(data,backend=backend,chezy=chezy,betac=betac,viscosity=viscosity,
            diffusivity=0.,espir=espir,threads=threads,parameters=p,allow_nonflat=allow_nonflat)
        self.diff=float(diffusivity);self.morfac=float(morfac)
        bc=np.asarray(bedbc,dtype=np.int32);f=np.asarray(feed,dtype=float)
        outside=(self.mesh['neighbour']<0)&(self.mesh['bid']>=0)
        if (bc.shape!=(self.nf,) or f.shape!=(self.nf,) or not np.isfinite(f).all()
            or np.any(f<0) or np.any(~np.isin(bc,[0,1,2])) or np.any(bc[~outside])
            or np.any(bc[outside]==0) or np.any(f[bc!=1])):
            raise ValueError('Explicit feed/outflow definitions required on every open boundary')
        self.bedbc=self.engine.array(bc);self.feed=self.engine.array(f)
        self.eh=ClassicFlowEH(backend,diameter=diameter,chezy=chezy,threads=threads)
        self.mobile=MobileEngine(backend,threads);xp=self.xp
        self.bedcell=xp.zeros((6,self.n));self.bedface=xp.zeros((10,self.nf))
        self.bedcorrected=xp.zeros((2,self.nf))
        self.directional_B=xp.zeros(self.n);self.directional_source=xp.zeros((2,self.n))
        self.identity=dict(self.identity,schema=2,method='classic_flow2dh_eh_mobile_v2',
            scope='fully_wet_single_grain_fixed_grid_mobile_bed',morphology=True,
            diffusivity_m2_s=self.diff,diameter_m=diameter,morfac=morfac,
            bed_runtime=self.mobile.identity,capacity_runtime=self.eh.identity,
            mobile_wrapper_sha256=digest(__file__),
            stress='released_FLOW_directional_streamline_frame: Fs=d_n(B)+2*k*B, Fn=-d_s(B); B=beta*speed^2',
            momentum_mapping='objective_local_streamline_frame_extension_on_Cartesian_cells; native_one_step_and_curved_metric_checks',
            bed_face='FLOW_sign_upwind_normal; geometric_tangent; affine_exact_secant_slope',
            bed_time='explicit_Exner_once_per_accepted_midpoint_step; morphfac_applied_once',
            capacity_forcing='independent_cell_depth_average_speed_constant_Chezy',
            slope_stress='absolute_Shields_times_sqrt(1+spiral_alpha_squared)',
            bed_boundary='prescribed_solid_volume_feed; owner_capacity_zero_normal_slope_outflow; zero_wall_flux',
            bedbc_sha256=hashlib.sha256(bc.tobytes()+f.tobytes()).hexdigest())

    def begin(self,q,z,dt,time):
        super().begin(q,z,dt,time)
        self.mobile.launch('flow_directional_geometric',self.n,self.n,q,self.outputs,self.directional_B,self.directional_source)

    def add_momentum(self,q,water,rate):
        g=self.g;n=self.n;nf=self.nf
        self.mobile.launch('flow_directional_faces',nf,nf,n,self.nu,q,self.directional_B,
            g['owner'],g['neighbour'],g['normal'],g['distance'],self.momentumflux)
        self.flow.launch('flow_momentum_gather',n,n,nf,self.hold,self.momentumflux,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],rate)
        rate[1:]+=self.hold*self.directional_source

    def bed_faces(self,q,z,hi):
        g=self.g;e=self.engine;n=self.n;nf=self.nf
        e.launch('gradients',n,n,q,z,hi,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.grad)
        self.bedcell[:]=self.eh.evaluate(q,hi)
        self.mobile.launch('flow_bed_face_inputs',nf,nf,n,self.parameters.diameter,q,z,self.bedcell,self.grad,
            g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],self.bedface)
        self.bedcorrected[:]=self.eh.slope(self.bedface)
        self.mobile.launch('flow_bed_flux',nf,nf,2650.,self.bedcorrected,g['neighbour'],
            self.bedbc,g['normal'],self.feed,self.aux)

    def finish(self,qold,qnew,water,*,morphology=True):
        super().finish(qold,qnew,water,morphology=False)
        if not morphology:return self.last
        xp=self.xp;g=self.g;dt=self._dt*self.morfac;phi=1-self.parameters.porosity
        self.bed_faces(qnew,self.z0,self.Wtrial)
        self.available[:]=phi*(self.z0-self.rock)
        donor=self.limit(self.available,dt)
        self.face_integral[:]=dt*self.aux[2]
        self.ztrial[:]=self.z0+(dt/phi)*self.divergence(2)
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary=-xp.sum(self.face_integral*g['length']*outside)
        residual=xp.sum(phi*(self.ztrial-self.z0)*g['area'])-boundary
        change=float(xp.max(xp.abs(self.ztrial-self.z0)/qnew[0]))
        if (not bool(xp.all(xp.isfinite(self.ztrial))) or bool(xp.any(self.ztrial<self.rock-1e-13))
            or abs(float(residual))>1e-10):raise BendTrialRejected('Classic mobile sediment budget failed')
        if change>.05:raise BendTrialRejected('Classic mobile bed change exceeds 5% of depth per trial')
        self.last.update(morphology=True,solid_volume_residual_m3=float(residual),solid_boundary_m3=float(boundary),
            donor_factor_min=donor,maximum_relative_bed_increment=change,morphfac=self.morfac)
        return self.last

    def commit(self):
        if not self._open:raise ValueError('No open classic mobile trial')
        outside=(self.g['neighbour']<0)&(self.g['bid']>=0)
        self.boundary_ledger[:]-=self.face_integral*self.g['length']*outside
        super().commit()
