"""Explicit, experimental Lai/LV component owner for native CPU and CUDA.
Not a public method selector. Hydraulic integration is supplied by its caller;
this module never implements or replaces the augmented f-wave solver.
"""
from __future__ import annotations
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from braids_core.windows_paths import filesystem_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = {'lai_prescribed_beta': 1, 'lai_bedload': 1, 'lv_bernard_schneider': 2,
           'lv_odgaard': 3, 'lv_kalkwijk_booij': 4,
           'lv_bs_torque_reconstruction_v1': 5}
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
KERNELS = ('gradients', 'constitutive', 'auxiliary_faces', 'divergence',
           'donor_factors', 'limit_flux', 'exchange', 'bend_vorticity_source',
           'bend_open_faces', 'bend_scalar_diffusion', 'bend_add_momentum',
           'bend_diffusion_metric', 'bend_lai_adaptation', 'bend_manning', 'bend_hydraulic_envelope',
           'bend_linear_acoustic', 'bend_viero_eddy', 'bend_lv_face_slope',
           'bend_tensor_gradients', 'bend_linear_momentum_faces', 'bend_constitutive_instantaneous',
           'bend_eq29_momentum_faces', 'bend_backtracked_curvature', 'bend_constitutive_curvature',
           'bend_lv_cell_viscosity','bend_surface_velocity_faces','bend_surface_velocity_diffusion')


class BendTrialRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class LaiAdaptation:
    """Explicit published alternatives; never extrapolate outside printed domains.

    Constant beta/Lb remains the saved-method default. Eq10abc requires an
    explicit fall velocity and Z<1; no suspended inventory is enabled by it.
    """
    velocity: str='prescribed'
    length: str='prescribed'
    settling_velocity: float=0.

    def codes(self):
        velocities={'prescribed':0,'lai_bed_eq10b':1,'lai_max_eq10abc':2}
        lengths={'prescribed':0,'phillips_sutherland_eq15':1}
        if self.velocity not in velocities or self.length not in lengths:
            raise ValueError('Unknown Lai adaptation choice')
        if not np.isfinite(self.settling_velocity) or self.settling_velocity<0:
            raise ValueError('Invalid Lai settling velocity')
        if self.velocity=='lai_max_eq10abc' and self.settling_velocity<=0:
            raise ValueError('Lai max velocity requires explicit positive settling velocity')
        if self.velocity!='lai_max_eq10abc' and self.settling_velocity!=0:
            raise ValueError('Settling velocity is unused by selected Lai adaptation')
        return velocities[self.velocity],lengths[self.length]


@dataclass(frozen=True)
class Parameters:
    gravity: float=9.81
    dry: float=1e-6
    manning: float=1/36
    diameter: float=.00078
    relative_density: float=2.65
    porosity: float=.4
    critical_shields: float=.047
    shape_factor: float=.75
    spiral_factor: float=1.
    bs_production: float=5.
    bs_decay: float=.5
    adaptation_length: float=.1
    sediment_velocity_ratio: float=1.
    longitudinal_capacity_factor: float=1.
    turbulence_coefficient: float=.7
    ripple_factor: float=1.
    longitudinal_slope_coefficient: float=0.
    wall_sand_roughness: float=0.
    molecular_viscosity: float=1e-6

    def array(self):
        a=np.array(list(asdict(self).values()),dtype=np.float64)
        if not np.isfinite(a).all() or np.any(a[:5]<=0) or self.relative_density<=1 or not 0<=self.porosity<1:
            raise ValueError('Invalid physical parameters')
        if np.any(a[6:]<0) or np.any(a[[6,11,12]]<=0):
            raise ValueError('Invalid closure parameters')
        if self.ripple_factor>1 or self.molecular_viscosity<=0:
            raise ValueError('Ripple factor must be in [0,1] and viscosity positive')
        return a


def digest(path):
    from .river_runtime_lease import digest as runtime_digest
    return runtime_digest(filesystem_path(path))


def prepared_geometry(data):
    """Auxiliary LS metric on the exact native hydraulic mesh, not a new mesh."""
    names = ('xy', 'area', 'owner', 'neighbour', 'normal', 'length', 'mid',
             'offset', 'faces', 'signs', 'distance', 'bid')
    integer = {'owner', 'neighbour', 'offset', 'faces', 'signs', 'bid'}
    g = {k: np.ascontiguousarray(data[k], dtype=np.int32 if k in integer else np.float64)
         for k in names}
    n = len(g['area']); nf = len(g['owner']); off = g['offset']
    if (np.any(g['area'] <= 0) or off.shape != (n+1,) or
            np.any(g['length'] <= 0) or g['xy'].shape != (2,n) or
            np.any(g['owner'] < 0) or np.any(g['owner'] >= n) or
            np.any(g['neighbour'] >= n)):
        raise ValueError('Invalid prepared bend geometry')
    delta = np.zeros((2,len(g['faces']))); inverse = np.zeros((3,n))
    for i in range(n):
        a = np.zeros((2,2))
        for k in range(off[i],off[i+1]):
            f = g['faces'][k]
            j = g['neighbour'][f] if g['owner'][f] == i else g['owner'][f]
            if j < 0: continue
            d = g['xy'][:,j]-g['xy'][:,i]; rr = d@d
            if rr <= 0: raise ValueError('Coincident cell centres')
            delta[:,k] = d; a += np.outer(d,d)/rr
        v = np.linalg.pinv(a); inverse[:,i] = v[0,0],v[0,1],v[1,1]
    disp = np.zeros((4,nf)); disp[:2] = g['mid']-g['xy'][:,g['owner']]
    inside = g['neighbour'] >= 0
    disp[2:,inside] = g['mid'][:,inside]-g['xy'][:,g['neighbour'][inside]]
    g.update(dxdy=delta,inverse=inverse,displacement=disp)
    return g


class NativeEngine:
    """Identical compiled cell/face bodies; backend is never inferred."""
    def __init__(self, backend, threads=4):
        if backend not in ('cpu','gpu') or not isinstance(threads,int) or threads < 1:
            raise ValueError('Choose cpu/gpu and a positive thread count')
        self.backend, self.threads = backend, threads
        path = HERE/'bend_methods_v1_kernels.h'; source = path.read_text()
        self.identity = dict(backend=backend, threads=threads, kernel_sha256=digest(path),
                             orchestration_sha256=digest(__file__))
        if backend == 'gpu':
            import cupy as cp
            self.xp = cp
            self.module = cp.RawModule(code=source,options=OPTIONS,name_expressions=KERNELS)
            self.kernels = {k:self.module.get_function(k) for k in KERNELS}
            self.identity['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        else:
            self.xp = np; cpp = shutil.which('g++')
            if not cpp: raise RuntimeError('Native C++/OpenMP compiler required')
            version = subprocess.check_output([cpp,'--version'],text=True).splitlines()[0]
            key = hashlib.sha256((source+version).encode()).hexdigest()[:20]
            build = ROOT/'build/bend_methods_v1'/key; build.mkdir(parents=True,exist_ok=True)
            lib = build/('bend.dll' if os.name == 'nt' else 'bend.so')
            if not lib.exists():
                subprocess.run([cpp,'-x','c++','-std=c++11','-O3','-fopenmp','-shared',
                    '-static-libgcc','-static-libstdc++','-ffp-contract=off',str(path),'-o',str(lib)],
                    check=True,capture_output=True,timeout=120)
            self.dll_dirs = [os.add_dll_directory(str(Path(cpp).parent))] if os.name == 'nt' else []
            self.lib = ctypes.CDLL(str(lib)); self.kernels = {k:getattr(self.lib,k) for k in KERNELS}
            for fun in self.kernels.values(): fun.restype = None
            self.identity.update(compiler=version,library_sha256=digest(lib))

    def array(self, a): return self.xp.asarray(np.ascontiguousarray(a))
    def host(self, a): return a.get() if self.backend == 'gpu' else a.copy()

    def launch(self, name, n, *args):
        if self.backend == 'gpu':
            self.kernels[name](((n+127)//128,),(128,),tuple(
                np.int32(a) if isinstance(a,int) else np.float64(a) if isinstance(a,float) else a for a in args))
        else:
            self.lib.bend_threads(self.threads)
            self.kernels[name](*(ctypes.c_int(a) if isinstance(a,int) else ctypes.c_double(a)
                if isinstance(a,float) else ctypes.c_void_p(a.ctypes.data) for a in args))


class BendPhysics:
    """One transactional secondary/mobile-solid/bed-volume owner.

    LV defaults to h*omega, not historical Delft hI. The explicitly identified
    Eq10 research alternative transports h*(4*v_ns), but exposes W=4*v_ns at
    all constitutive/checkpoint interfaces. Budgets name their actual inventory.
    Lai retains prescribed beta/Lb by default; alternative published choices
    are explicit, checkpointed and bounded by their printed applicability.
    Open feeds are solid volume/width, never bulk mass. All work arrays resident.
    """
    def __init__(self, data, *, backend, method, parameters=None, saturation=False,
                 scalar_diffusion=True, schmidt=1., threads=4, feed=None, omega_in=None,
                 lai_adaptation=None, turbulence_closure='legacy_parabolic',
                 lv_slope_reconstruction='cell_average', saturation_domain='strict',
                 auxiliary_reconstruction='legacy_two_point',
                 secondary_diffusion='full', secondary_evolution='transported', curvature_method='spatial_ls',
                 momentum_viscosity='conservative_face',secondary_transport='vorticity'):
        if method not in METHODS: raise ValueError('Unknown bend method')
        self.method = METHODS[method]; self.parameters = parameters or Parameters()
        p = self.parameters; pa = p.array()
        if self.method==5 and p.bs_production!=5.:
            raise ValueError('Power-law torque determines production; custom bs_production is unused')
        if secondary_evolution not in ('transported','instantaneous'):
            raise ValueError('Unknown secondary evolution')
        self.secondary_evolution=secondary_evolution
        self.instantaneous=secondary_evolution=='instantaneous'
        if secondary_transport not in ('vorticity','surface_velocity_eq10'):
            raise ValueError('Unknown secondary transport variable')
        if secondary_transport!='vorticity' and (self.method==1 or self.instantaneous):
            raise ValueError('Surface-velocity transport requires transported LV physics')
        self.secondary_transport=secondary_transport
        self.surface_transport=secondary_transport=='surface_velocity_eq10'
        if curvature_method not in ('spatial_ls','backtracked_spatial_v1','backtracked_previous_v1'):
            raise ValueError('Unknown bend curvature method')
        if self.method==1 and curvature_method!='spatial_ls':
            raise ValueError('LV curvature research cannot alter Lai')
        self.curvature_method=curvature_method
        if self.instantaneous:
            if self.method==1:raise ValueError('LV instantaneous adaptation cannot alter Lai')
            if scalar_diffusion:raise ValueError('Instantaneous adaptation requires scalar_diffusion=False; no vorticity PDE')
            if self.method in (2,5) and p.bs_decay<=0:raise ValueError('Instantaneous adaptation requires positive decay')
        if turbulence_closure not in ('legacy_parabolic', 'viero2019_smagorinsky_elder'):
            raise ValueError('Unknown bend turbulence closure')
        if self.method == 1 and turbulence_closure != 'legacy_parabolic':
            raise ValueError('2DEF turbulence cannot silently alter the Lai bundle')
        self.turbulence_closure = turbulence_closure
        if lv_slope_reconstruction not in ('cell_average', 'face_secant'):
            raise ValueError('Unknown LV slope reconstruction')
        if self.method == 1 and lv_slope_reconstruction != 'cell_average':
            raise ValueError('LV slope reconstruction cannot alter the Lai bundle')
        self.lv_slope_reconstruction = lv_slope_reconstruction
        if auxiliary_reconstruction not in ('legacy_two_point','linear_exact_v1','lv_eq29_v1'):
            raise ValueError('Unknown auxiliary momentum reconstruction')
        if self.method==1 and auxiliary_reconstruction!='legacy_two_point':
            raise ValueError('LV auxiliary reconstruction cannot alter Lai')
        self.auxiliary_reconstruction=auxiliary_reconstruction
        if momentum_viscosity not in ('conservative_face','lv_eq26_cell','lv_eq27_literal'):
            raise ValueError('Unknown momentum viscosity operator')
        if momentum_viscosity!='conservative_face' and (self.method==1 or auxiliary_reconstruction=='legacy_two_point'):
            raise ValueError('LV cell viscosity research requires LV nonorthogonal auxiliary faces')
        self.momentum_viscosity=momentum_viscosity
        # Research contrasts, not alternative authenticated author bundles.
        # The legacy full tensor remains the default on both backends.
        diffusion_codes={'full':7,'turbulent_longitudinal':3,'turbulent_only':1,'none':0}
        if secondary_diffusion not in diffusion_codes:
            raise ValueError('Unknown secondary diffusion decomposition')
        if secondary_diffusion!='full' and (self.method==1 or not scalar_diffusion):
            raise ValueError('Secondary diffusion contrast requires enabled LV scalar diffusion')
        self.secondary_diffusion=secondary_diffusion
        self.diffusion_code=diffusion_codes[secondary_diffusion]
        if saturation_domain not in ('strict','positive_interaction_v1'):
            raise ValueError('Unknown saturation domain policy')
        if saturation_domain!='strict' and (not saturation or self.method==1):
            raise ValueError('Saturation domain extension requires enabled LV saturation')
        self.saturation_domain = saturation_domain
        self.saturation_code = (2 if saturation_domain=='positive_interaction_v1' else 1) if saturation else 0
        self.lai_adaptation=lai_adaptation or LaiAdaptation()
        self.lai_velocity,self.lai_length=self.lai_adaptation.codes()
        if self.method!=1 and self.lai_adaptation!=LaiAdaptation():
            raise ValueError('Lai adaptation cannot alter LV')
        if self.method == 1 and (saturation or p.ripple_factor != 1 or
                p.longitudinal_slope_coefficient != 0 or p.wall_sand_roughness != 0):
            raise ValueError('LV controls cannot silently alter Lai')
        if not np.isfinite(schmidt) or schmidt <= 0: raise ValueError('Positive Schmidt required')
        self.engine = NativeEngine(backend,threads); self.xp = self.engine.xp
        xp = self.xp; self.mesh = prepared_geometry(data)
        self.g = {k:self.engine.array(v) for k,v in self.mesh.items()}
        self.n = n = len(self.mesh['area']); self.nf = nf = len(self.mesh['owner'])
        self.saturation = bool(saturation); self.scalar_diffusion = bool(scalar_diffusion)
        self.schmidt = float(schmidt); self.p = self.engine.array(pa)
        def boundary_values(value):
            a = np.zeros(nf) if value is None else np.asarray(value,dtype=float)
            if a.shape != (nf,) or not np.isfinite(a).all(): raise ValueError('Boundary array differs')
            closed = (self.mesh['neighbour'] >= 0) | (self.mesh['bid'] < 0)
            if np.any(a[closed]): raise ValueError('Boundary values must be zero on internal/wall faces')
            return self.engine.array(a)
        self.feed = boundary_values(feed); self.omega_in = boundary_values(omega_in)
        if bool(xp.any(self.feed < 0)): raise ValueError('Incoming feed must be nonnegative')
        if self.instantaneous and bool(xp.any(self.omega_in)):
            raise ValueError('Instantaneous adaptation cannot use prescribed incoming vorticity')
        self.W = xp.zeros(n); self.M = xp.zeros(n)
        self.local_W = xp.zeros(n)
        self.curvature_reference_q=xp.zeros((3,n));self.curvature_reference_grad=xp.zeros((10,n))
        self.curvature_work=xp.zeros(n);self.curvature_status=xp.zeros(n,dtype=xp.int32)
        self.Wbase = xp.zeros(n); self.Wtrial = xp.zeros(n); self.Mtrial = xp.zeros(n)
        self.S = xp.zeros(n); self.grad = xp.zeros((10,n)); self.op = xp.zeros((15,n))
        self.status = xp.zeros(n,dtype=xp.int32); self.aux = xp.zeros((4,nf))
        self.saturation_extension_used = xp.zeros(n,dtype=xp.int32)
        self.scalar_grad = xp.zeros((10,n)); self.conductance = xp.zeros(nf)
        self.scalar_inventory = xp.zeros(n)
        self.face_bed = xp.zeros(n)
        self.tensor_grad = xp.zeros((6,n))
        self.viscosity_correction = xp.zeros((2,n))
        self.metric = xp.zeros(n); self.factors = xp.zeros(n); self.rate = xp.zeros(n)
        self.work = xp.zeros(n); self.work2 = xp.zeros(n); self.available = xp.zeros(n)
        self.z0 = xp.zeros(n); self.ztrial = xp.zeros(n); self.rock = self.engine.array(data['rock'])
        self.face_integral = xp.zeros(nf); self.boundary_ledger = xp.zeros(nf)
        self.secondary_boundary_ledger = xp.zeros(nf); self.secondary_source_ledger = xp.zeros(())
        self.secondary_projection_ledger = xp.zeros(())
        self.secondary_face_integral = xp.zeros(nf)
        self.source_first = xp.zeros(n); self.source_second = xp.zeros(n)
        self._open = False; self.last = {}; self.time = float(data.get('time',0.)); self.steps = 0
        self.identity = dict(schema=1,method=method,parameters=asdict(p),saturation=self.saturation,
            saturation_domain=self.saturation_domain,
            scalar_diffusion=self.scalar_diffusion,schmidt=self.schmidt,secondary='h_omega_equals_4_vns',
            secondary_diffusion=self.secondary_diffusion,
            secondary_evolution=self.secondary_evolution,
            curvature_method=self.curvature_method,
            secondary_budget=('algebraic_projection_not_transport' if self.instantaneous else 'transported_conservation'),
            mesh_sha256=hashlib.sha256(b''.join(v.tobytes() for k,v in sorted(self.mesh.items()))).hexdigest(),
            feed_sha256=hashlib.sha256(self.engine.host(self.feed).tobytes()).hexdigest(),
            omega_in_sha256=hashlib.sha256(self.engine.host(self.omega_in).tobytes()).hexdigest(),
            lai_adaptation=asdict(self.lai_adaptation),lv_slope_reconstruction=lv_slope_reconstruction,
            auxiliary_reconstruction=auxiliary_reconstruction,
            runtime=self.engine.identity,scope='experimental fixed-grid safely-wet single-grain')
        self.identity['momentum_viscosity']=dict(method=momentum_viscosity,
            provenance='LV2023 Eqs25-27 contrast; author executable unavailable',
            conservative_internal_transfer=momentum_viscosity=='conservative_face')
        if self.surface_transport:
            self.identity['secondary_transport']=dict(method=secondary_transport,
                equation='LV2023 Eq10 material surface-velocity equation; slowly-varying-depth reduction',
                stored='W=4*v_ns=h*omega; NOT the transported inventory on this route',
                budget_inventory='h*W=4*h*v_ns',
                diffusion='same selected tensor, gradient of 4*v_ns instead of omega; research extension if enabled',
                inlet='omega_in retains vorticity units; incoming 4*v_ns=h_owner*omega_in')
            self.identity['secondary_budget']='transported_surface_velocity_inventory'
        if self.method==5:
            self.identity['secondary_source_reconstruction']=dict(
                scope='Derived research contrast, NOT authenticated LV power-law coefficients',
                production='3*(m+1)/(2*m*(m+2)); exact power-law torque with Table1 normalization',
                decay='Ds*kappa/m; unchanged empirical decay',
                reference='Finnie1999 Eqs35-45; LV2023 Eq4 and Table1',
                bs_production='not used; default placeholder retained for schema compatibility')
        # Keep historical choice explicit; checkpoint identity forbids silent
        # switching. Existing frozen runtimes remain the restart reference.
        self.identity['turbulence'] = dict(method=self.turbulence_closure,
            coefficients=({'Ct': p.turbulence_coefficient} if turbulence_closure == 'legacy_parabolic'
                          else {'Cs': .2, 'CE': .07}),
            area='actual BRAIDS control-volume area',
            provenance=('historical reconstruction assumption' if turbulence_closure == 'legacy_parabolic'
                        else 'Viero 2019 Eq4, doi:10.1016/j.jhydrol.2018.10.055; LFM deck not recovered'))

    def operators(self,q,z,W,M,*,curvature_dt=0.):
        if self.lv_slope_reconstruction == 'face_secant':
            self.face_bed[...] = z
        e,g,n = self.engine,self.g,self.n
        if self.instantaneous:self.S.fill(0)  # Never consume a lagged secondary state.
        else:self.S[...] = .25*q[0]*W
        e.launch('gradients',n,n,q,z,self.S,M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.grad)
        args=(n,self.method,self.saturation_code,self.p,q,self.S,self.grad,self.op,self.status)
        self.curvature_status.fill(0)
        if self.curvature_method!='spatial_ls' and curvature_dt>0:
            if not self._open:raise ValueError('Backtracked curvature requires a current trial reference')
            oldq,oldgrad=(self.curvature_reference_q,self.curvature_reference_grad) if self.curvature_method=='backtracked_previous_v1' else (q,self.grad)
            e.launch('bend_backtracked_curvature',n,n,float(curvature_dt),q,oldq,oldgrad,g['xy'],g['area'],
                g['mid'],g['normal'],self.nf,g['offset'],g['faces'],g['owner'],g['neighbour'],self.curvature_work,self.curvature_status)
            if bool(self.xp.any(self.curvature_status>1)):
                raise BendTrialRejected('Backtracked curvature footprint rejected')
            e.launch('bend_constitutive_curvature',n,n,self.method,self.saturation_code,int(self.instantaneous),
                self.p,q,self.S,self.grad,self.curvature_work,self.op,self.status,self.local_W)
            if self.instantaneous:W=self.local_W
        elif self.instantaneous:
            e.launch('bend_constitutive_instantaneous',n,*args,self.local_W)
            W=self.local_W
        else:e.launch('constitutive',n,*args)
        self.saturation_extension_used[...] = (self.status==6)
        if self.turbulence_closure == 'viero2019_smagorinsky_elder':
            e.launch('bend_viero_eddy',n,n,self.p,q,g['area'],self.grad,self.op)
        if self.method==1:
            e.launch('bend_lai_adaptation',n,n,self.lai_velocity,self.lai_length,
                float(self.lai_adaptation.settling_velocity),self.p,q,self.op,self.status)
        invalid=(self.status!=0)&((self.status!=6) if self.saturation_code==2 else True)
        if bool(self.xp.any(invalid)) or not bool(self.xp.all(self.xp.isfinite(self.op))):
            raise BendTrialRejected('Bend constitutive domain rejected: '+str(np.unique(e.host(self.status)).tolist()))
        # The gradient routine divides its scalar inventory by h. Constitutive
        # W remains 4*v_ns on BOTH routes; only its transport inventory changes.
        if self.surface_transport:self.scalar_inventory[...] = q[0]*W
        scalar=self.scalar_inventory if self.surface_transport else W
        e.launch('gradients',n,n,q,z,scalar,M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.scalar_grad)

    def faces(self,q,W,M,water,dt=0.):
        e,g,n,nf = self.engine,self.g,self.n,self.nf
        if self.instantaneous:W=self.local_W
        # Old face routine divides the supplied inventory by depth. Supplying W
        # and its gradients therefore reconstructs omega for advection.
        e.launch('auxiliary_faces',nf,nf,n,self.method,self.p,q,W,M,self.op,self.scalar_grad,water,
            g['owner'],g['neighbour'],g['normal'],g['distance'],g['displacement'],self.aux)
        if self.auxiliary_reconstruction=='linear_exact_v1':
            # Recompute here so diagnostic tensor decompositions use their
            # actual current tensor; do not cache a stale constitutive field.
            e.launch('bend_tensor_gradients',n,n,self.op,g['offset'],g['faces'],g['owner'],g['neighbour'],
                     g['dxdy'],g['inverse'],self.tensor_grad)
            e.launch('bend_linear_momentum_faces',nf,nf,n,q,self.op,self.grad,self.tensor_grad,
                     g['owner'],g['neighbour'],g['normal'],g['distance'],g['displacement'],self.aux)
        elif self.auxiliary_reconstruction=='lv_eq29_v1':
            e.launch('bend_eq29_momentum_faces',nf,nf,n,q,self.op,self.grad,
                     g['owner'],g['neighbour'],g['normal'],g['distance'],g['displacement'],self.aux)
        if self.lv_slope_reconstruction == 'face_secant':
            e.launch('bend_lv_face_slope',nf,nf,n,self.p,q,self.face_bed,W,self.op,self.grad,
                g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],self.aux)
        e.launch('bend_open_faces',nf,nf,n,self.method,self.p,q,W,M,self.op,water,g['owner'],g['neighbour'],g['bid'],
            g['normal'],self.feed,self.omega_in,self.aux)
        if self.surface_transport:
            e.launch('bend_surface_velocity_faces',nf,nf,n,q,W,self.scalar_grad,water,g['owner'],g['neighbour'],
                     g['bid'],g['displacement'],self.omega_in,self.aux)
        if self.instantaneous:
            # Eq11 has no transported secondary flux or diffusion CFL bound.
            # Momentum and bedload faces above still use the resolved local W.
            self.aux[3].fill(0);self.conductance.fill(0)
        elif self.method != 1 and self.scalar_diffusion and self.diffusion_code:
            kernel='bend_surface_velocity_diffusion' if self.surface_transport else 'bend_scalar_diffusion'
            e.launch(kernel,nf,nf,n,self.diffusion_code,self.schmidt,self.p,q,W,self.op,self.scalar_grad,
                g['owner'],g['neighbour'],g['normal'],g['xy'],self.aux,self.conductance)
            if dt:
                e.launch('bend_diffusion_metric',n,n,float(dt),q[0],self.conductance,g['length'],g['area'],g['offset'],g['faces'],self.metric)
                if float(self.xp.max(self.metric)) > .45:
                    raise BendTrialRejected('Secondary diffusion timestep bound exceeded')
        if not bool(self.xp.all(self.xp.isfinite(self.aux))): raise BendTrialRejected('Nonfinite auxiliary face flux')

    def divergence(self,component):
        g=self.g
        self.engine.launch('divergence',self.n,self.n,self.nf,component,self.aux,g['length'],g['area'],g['offset'],g['faces'],g['signs'],self.rate)
        return self.rate

    def source(self,q,z,W,M,dt,out,*,curvature_dt=0.):
        if self.instantaneous:raise ValueError('Instantaneous adaptation has no time-integrated vorticity source')
        self.operators(q,z,W,M,curvature_dt=curvature_dt)
        self.engine.launch('bend_vorticity_source',self.n,self.n,float(dt),self.op,W,out)

    def begin(self,q,z,dt,time):
        if self._open: raise ValueError('Bend trial already open')
        if not np.isfinite([dt,time]).all() or dt <= 0 or time != self.time:
            raise ValueError('Bend trial clock differs')
        if bool(self.xp.any(q[0] <= 20*self.parameters.dry)): raise BendTrialRejected('Bend requires fully wet cells')
        self.z0[...] = z; self._dt = float(dt)
        if self.curvature_method!='spatial_ls':
            self.curvature_reference_q[...] = q
            self.engine.launch('gradients',self.n,self.n,q,z,self.W,self.M,self.g['offset'],self.g['faces'],
                self.g['owner'],self.g['neighbour'],self.g['dxdy'],self.g['inverse'],self.curvature_reference_grad)
        if self.method == 1 or self.instantaneous: self.Wbase[...] = self.W
        else: self.source(q,z,self.W,self.M,.5*dt,self.Wbase)
        self.source_first[...] = self.Wbase-self.W
        self._open = True

    def add_momentum(self,q,water,rate):
        if not self._open: raise ValueError('Bend trial is not open')
        self.operators(q,self.z0,self.Wbase,self.M,curvature_dt=.5*self._dt); self.faces(q,self.Wbase,self.M,water)
        g=self.g
        self.engine.launch('bend_add_momentum',self.n,self.n,self.nf,self.aux,g['length'],g['area'],g['offset'],g['faces'],g['signs'],rate)
        if self.momentum_viscosity!='conservative_face':
            self.cell_viscosity_correction(q)
            rate[1:3] += self.viscosity_correction

    def cell_viscosity_correction(self,q):
        """Research difference from conservative mixing; not an additional stress."""
        if self.momentum_viscosity=='conservative_face':
            self.viscosity_correction.fill(0)
        else:
            g=self.g;factor=2. if self.momentum_viscosity=='lv_eq27_literal' else 1.
            self.engine.launch('bend_lv_cell_viscosity',self.n,self.n,self.nf,factor,q,
                self.op,self.grad,g['offset'],g['faces'],g['owner'],g['neighbour'],
                g['normal'],g['distance'],g['displacement'],g['length'],g['area'],self.viscosity_correction)
        return self.viscosity_correction

    def limit(self,available,dt):
        g=self.g;e=self.engine
        e.launch('donor_factors',self.n,self.n,self.nf,2,float(dt),available,self.aux,g['length'],g['area'],g['offset'],g['faces'],g['signs'],self.factors)
        e.launch('limit_flux',self.nf,self.nf,2,g['owner'],g['neighbour'],self.factors,self.aux)
        return float(self.xp.min(self.factors))

    def finish(self,qold,qnew,water,*,morphology=True):
        if not self._open: raise ValueError('Bend trial is not open')
        xp=self.xp;e=self.engine;dt=self._dt;p=self.parameters;g=self.g
        self.source_second.fill(0); self.secondary_face_integral.fill(0)
        if self.instantaneous:
            self.operators(qnew,self.z0,self.W,self.M,curvature_dt=dt)
            self.Wtrial[...] = self.local_W
        elif self.method != 1:
            self.operators(qold,self.z0,self.Wbase,self.M); self.faces(qold,self.Wbase,self.M,water,dt)
            self.secondary_face_integral[...] = .5*dt*self.aux[3]
            if self.surface_transport:
                self.work[...] = (qold[0]*self.Wbase+dt*self.divergence(3))/qnew[0]
            else:self.work[...] = self.Wbase+dt*self.divergence(3)
            self.operators(qnew,self.z0,self.work,self.M,curvature_dt=dt); self.faces(qnew,self.work,self.M,water,dt)
            self.secondary_face_integral[...] += .5*dt*self.aux[3]
            if self.surface_transport:
                self.work2[...] = .5*(qold[0]*self.Wbase+qnew[0]*self.work+dt*self.divergence(3))/qnew[0]
            else:self.work2[...] = .5*(self.Wbase+self.work+dt*self.divergence(3))
            self.source(qnew,self.z0,self.work2,self.M,.5*dt,self.Wtrial,curvature_dt=dt)
            self.source_second[...] = self.Wtrial-self.work2
        else: self.Wtrial[...] = self.W
        self.operators(qnew,self.z0,self.Wtrial,self.M,curvature_dt=dt); self.faces(qnew,self.Wtrial,self.M,water)
        self.ztrial[...] = self.z0; self.Mtrial[...] = self.M; self.face_integral.fill(0)
        donor = 1.
        if morphology and self.method == 1:
            donor=self.limit(self.M,dt);self.face_integral[...] = .5*dt*self.aux[2]
            self.work[...] = self.M+dt*self.divergence(2)
            self.operators(qnew,self.z0,self.Wtrial,self.work,curvature_dt=dt);self.faces(qnew,self.Wtrial,self.work,water)
            donor=min(donor,self.limit(self.work,dt));self.face_integral[...] += .5*dt*self.aux[2]
            self.work2[...] = .5*(self.M+self.work+dt*self.divergence(2))
            e.launch('exchange',self.n,self.n,float(dt),self.p,self.op,self.work2,self.z0,self.rock,
                self.Mtrial,self.ztrial,self.available)
        elif morphology:
            self.available[...] = (1-p.porosity)*(self.z0-self.rock)
            donor=self.limit(self.available,dt);self.face_integral[...] = dt*self.aux[2]
            self.ztrial[...] = self.z0+dt/(1-p.porosity)*self.divergence(2)
        outside=(g['neighbour'] < 0)&(g['bid'] >= 0)
        boundary=-xp.sum(xp.where(outside,self.face_integral*g['length'],0.))
        residual=xp.sum(g['area']*((1-p.porosity)*(self.ztrial-self.z0)+self.Mtrial-self.M))-boundary
        secondary_boundary=-xp.sum(xp.where(outside,self.secondary_face_integral*g['length'],0.))
        if self.surface_transport:
            secondary_source=xp.sum(g['area']*(qold[0]*self.source_first+qnew[0]*self.source_second))
            secondary_change=xp.sum(g['area']*(qnew[0]*self.Wtrial-qold[0]*self.W))
        else:
            secondary_source=xp.sum(g['area']*(self.source_first+self.source_second))
            secondary_change=xp.sum(g['area']*(self.Wtrial-self.W))
        secondary_projection=secondary_change if self.instantaneous else 0.
        secondary_residual=secondary_change-secondary_boundary-secondary_source-secondary_projection
        if (abs(float(residual)) > 1e-10 or abs(float(secondary_residual)) > 1e-10 or
                not all(bool(xp.all(xp.isfinite(a))) for a in (self.ztrial,self.Wtrial,self.Mtrial)) or
                bool(xp.any(self.ztrial < self.rock-1e-13)) or bool(xp.any(self.Mtrial < -1e-13))):
            raise BendTrialRejected('Bend stock/secondary budget failed')
        self.last=dict(solid_volume_residual_m3=float(residual),solid_boundary_m3=float(boundary),
            curvature_boundary_extension_cells=int(xp.sum(self.curvature_status==1)),
            secondary_residual=float(secondary_residual),secondary_boundary=float(secondary_boundary),
            secondary_source=float(secondary_source),secondary_projection=float(secondary_projection),
            secondary_budget=self.identity['secondary_budget'],
            local_equilibrium_residual=(float(xp.max(xp.abs(self.op[5]-.25*self.op[6]*self.Wtrial))) if self.instantaneous else None),
            donor_factor_min=donor,morphology=bool(morphology))
        return self.last

    def commit(self):
        if not self._open: raise ValueError('Bend trial is not open')
        self.W[...] = self.Wtrial;self.M[...] = self.Mtrial
        outside=(self.g['neighbour'] < 0)&(self.g['bid'] >= 0)
        self.boundary_ledger[...] -= self.xp.where(outside,self.face_integral*self.g['length'],0.)
        self.secondary_boundary_ledger[...] -= self.xp.where(outside,self.secondary_face_integral*self.g['length'],0.)
        self.secondary_source_ledger[...] += self.last['secondary_source']
        self.secondary_projection_ledger[...] += self.last['secondary_projection']
        self.time += self._dt;self.steps += 1;self._open=False

    def abort(self): self._open=False

    def initialize_secondary(self,q,z):
        """Resolve Eq11 for a fresh state; never reinterpret a saved restart."""
        if self._open or self.steps:raise ValueError('Secondary initialization requires a fresh owner')
        if self.instantaneous:
            self.operators(q,z,self.W,self.M)
            self.W[...] = self.local_W
