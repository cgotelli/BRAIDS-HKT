"""Resident CUDA Q5a columns; CPU/GPU choice is explicit, with no fallback."""
from pathlib import Path
import hashlib
import numpy as np
from braids_core.sediment.bed_columns_v101 import (
    HEADER, digest, array_sha, target_controls, column_grain_phi)


class ResidentColumnRuntimeV101:
    backend = 'implicit_fwave_midpoint_gpu'

    def __init__(self, methods, nc, nl, *, phi=None, sigma=None,
                 diffusion_depths=(), diffusion_coefficients=None):
        import cupy as cp
        self.cp = cp; self.methods, self.nc, self.nl = methods, nc, nl
        self.target_controls = target_controls(methods, nl)
        # Constructor validation is shared, but never calls/builds the CPU numerical runtime.
        ng = len(methods.fraction_ids)
        self.rho = np.ascontiguousarray(methods.density)
        p = np.broadcast_to(column_grain_phi(methods)[:, None]
                            if phi is None else phi, (ng, nc)).copy()
        s = np.broadcast_to(np.asarray(methods.phi_std or (0.,)*ng)[:, None]
                            if sigma is None else sigma, (ng, nc)).copy()
        z = np.ascontiguousarray(diffusion_depths, dtype=float)
        k = np.empty((0, nc)) if diffusion_coefficients is None else np.ascontiguousarray(diffusion_coefficients, dtype=float)
        if (not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s < 0)
                or z.ndim != 1 or k.shape != (len(z), nc) or not np.isfinite(z).all()
                or not np.isfinite(k).all() or np.any(z < 0) or np.any(np.diff(z) <= 0) or np.any(k < 0)):
            raise ValueError('Invalid CUDA distributions or bed mixing inputs')
        self.phi, self.sigma, self.zd, self.kd = (cp.asarray(a) for a in (p, s, z, k))
        self.device_rho = cp.asarray(self.rho); self.mud = cp.zeros(ng, cp.int32)
        path = Path(__file__).with_suffix('.cu')
        source = '#define BRAIDS_HD __device__\n'+HEADER.read_text()+path.read_text()
        options = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false')
        self.module = cp.RawModule(code=source, options=options, name_expressions=(
            'braids_bed_columns_v101', 'braids_bed_limits_v101', 'braids_bed_remap_v101',
            'braids_bed_characteristics_v101', 'braids_bed_fixfac_v101', 'braids_bed_integrate_v101',
            'braids_bed_tracer_v101', 'braids_bed_targets_v101', 'braids_bed_columns_coarse_v101'))
        self.kernel = self.module.get_function('braids_bed_columns_v101')
        self.receipt = {'equations': hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            'entry': hashlib.sha256(path.read_bytes()).hexdigest(), 'options': options,
            'device': cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(), 'cupy': cp.__version__}
        self.identity = digest({'backend': self.backend, 'methods': methods.fingerprint,
            'runtime': self.receipt, 'fields': array_sha(p, s, z, k), 'nc': nc, 'nl': nl})

    def targets(self, depth):
        cp = self.cp
        if (not isinstance(depth, cp.ndarray) or depth.shape != (self.nc,)
                or depth.dtype != cp.float64 or not depth.flags.c_contiguous
                or not bool(cp.isfinite(depth).all()) or bool((depth < 0).any())):
            raise ValueError('Resident finite nonnegative column depths required')
        out = cp.empty((self.nl, self.nc))
        mode, nlag, *values = self.target_controls
        self.module.get_function('braids_bed_targets_v101')(((self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(self.nl), np.int32(mode), np.int32(nlag),
             *map(np.float64, values), depth, out))
        return out

    def evaluate(self, mass, sv, delta, targets, dt, *, coarse_mobility=None):
        cp = self.cp; ng = len(self.methods.fraction_ids)
        shapes = ((self.nl, ng, self.nc), (self.nl, self.nc), (ng, self.nc), (self.nl, self.nc))
        for a, shape in zip((mass, sv, delta, targets), shapes):
            if not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != cp.float64 or not a.flags.c_contiguous:
                raise ValueError('Column solver requires resident contiguous FP64 arrays')
            if not bool(cp.isfinite(a).all()): raise ValueError('Nonfinite CUDA column')
        if (bool((mass < 0).any()) or bool(((sv <= 0) | (sv > 1)).any())
                or bool((targets <= 0).any()) or not np.isfinite(dt) or dt < 0):
            raise ValueError('Invalid CUDA column domain')
        out = cp.empty_like(mass); solid = cp.empty_like(sv); source = cp.empty_like(delta)
        work = cp.empty((max(7*self.nl+2*self.nl*ng, ng), self.nc), cp.float64); status = cp.empty(self.nc, cp.int32)
        m = self.methods
        kernel, extra = self.kernel, ()
        if coarse_mobility is not None:
            a = coarse_mobility
            if (not isinstance(a, cp.ndarray) or a.shape != (ng, self.nc)
                    or a.dtype != cp.float64 or not a.flags.c_contiguous
                    or not bool(cp.isfinite(a).all()) or bool(((a < 0) | (a > 1)).any())
                    or self.nl < 3 + int(m.exchange_thickness_m > 0) + m.lagrangian_underlayers):
                raise ValueError('Invalid resident coarse mobility or independent basal layer')
            kernel = self.module.get_function('braids_bed_columns_coarse_v101')
            extra = (a,)
        kernel(((self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(ng), np.int32(self.nl), np.int32(m.top_update),
             np.int32(m.base_update), np.int32(m.porosity), np.int32(m.lagrangian_underlayers),
             np.int32(m.exchange_thickness_m > 0), np.float64(dt), mass, sv, delta,
             targets, self.device_rho, self.phi, self.sigma, self.mud,
             np.float64(m.solid_fraction), np.float64(m.sand_solid_fraction), np.float64(m.mud_solid_fraction),
             np.int32(len(self.zd)), self.zd, self.kd, out, solid, source, work, status, *extra))
        if bool((status != 0).any()): raise ValueError('CUDA column finite stock overdraw or nonphysical state')
        return out, solid, source

    def limits(self, active, water, depth, bedrate, waterrate, *, morfac,
               max_bed_change_m=0., max_concentration_kg_m3=0., entrainment_fraction=0.):
        cp = self.cp; ng = len(self.methods.fraction_ids)
        shapes = ((ng, self.nc), (ng, self.nc), (self.nc,), (ng, self.nc), (ng, self.nc))
        arrays = (active, water, depth, bedrate, waterrate)
        for a, s in zip(arrays, shapes):
            if not isinstance(a, cp.ndarray) or a.shape != s or a.dtype != cp.float64 or not a.flags.c_contiguous or not bool(cp.isfinite(a).all()):
                raise ValueError('Limiter requires resident FP64 inputs')
        controls = (morfac, max_bed_change_m, max_concentration_kg_m3, entrainment_fraction)
        if not np.isfinite(controls).all() or min(controls) < 0 or any(bool((a < 0).any()) for a in arrays[:3]):
            raise ValueError('Limiter domain differs')
        out = cp.empty((4, self.nc), cp.float64)
        self.module.get_function('braids_bed_limits_v101')(((self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(ng), np.float64(morfac), *arrays, self.device_rho,
             np.float64(max_bed_change_m), np.float64(max_concentration_kg_m3), np.float64(entrainment_fraction), out))
        return out

    def remap(self, mass, sv, rock, offsets, indices, weights, newnc):
        cp = self.cp; ng = len(self.methods.fraction_ids)
        out = cp.empty((self.nl, ng, newnc), cp.float64); solid = cp.empty((self.nl, newnc), cp.float64); datum = cp.empty(newnc, cp.float64)
        self.module.get_function('braids_bed_remap_v101')(((newnc+127)//128,), (128,),
            (np.int32(newnc), np.int32(self.nc), np.int32(ng), np.int32(self.nl),
             cp.asarray(offsets), cp.asarray(indices), cp.asarray(weights), mass, sv, rock,
             self.device_rho, out, solid, datum))
        return out, solid, datum

    def integrate_rate(self, rate, dt, enabled=()):
        cp = self.cp; ng = len(self.methods.fraction_ids)
        mask = np.ascontiguousarray(enabled or (True,)*ng, dtype=np.int32)
        if (not isinstance(rate, cp.ndarray) or rate.shape != (ng, self.nc)
                or rate.dtype != cp.float64 or not rate.flags.c_contiguous
                or not bool(cp.isfinite(rate).all()) or not np.isfinite(dt) or dt < 0
                or mask.shape != (ng,) or np.any((mask != 0) & (mask != 1))):
            raise ValueError('Composition rate must be resident FP64 with valid fraction switches')
        out = cp.empty_like(rate)
        self.module.get_function('braids_bed_integrate_v101')(((ng*self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(ng), np.float64(dt), cp.asarray(mask), rate, out))
        return out
