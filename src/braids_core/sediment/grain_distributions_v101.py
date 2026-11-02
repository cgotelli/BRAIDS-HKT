"""Prepared FLOW grain distributions and native/resident mixture characteristics."""
from dataclasses import dataclass
import ctypes
import numpy as np
from .bed_columns_v101 import native_runtime_v101, array_sha, digest, host


@dataclass(frozen=True)
class GrainDistributionV101:
    kind: str
    diameter_m: tuple[float, ...]
    percentiles: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in ('log_uniform', 'lognormal', 'piecewise_log_uniform', 'legacy_single_d50_piecewise_log_uniform'):
            raise ValueError('Unsupported FLOW grain distribution')
        if (len(self.diameter_m) < 2 or len(self.percentiles) != len(self.diameter_m)
                or not np.isfinite(self.diameter_m).all() or min(self.diameter_m) <= 0
                or any(b < a for a, b in zip(self.diameter_m, self.diameter_m[1:]))
                or self.percentiles[0] != 0. or self.percentiles[-1] != 1.
                or any(b <= a for a, b in zip(self.percentiles, self.percentiles[1:]))):
            raise ValueError('Ordered positive diameters and full CDF endpoints required')

    @classmethod
    def log_uniform(cls, minimum, maximum):
        return cls('log_uniform', (minimum, maximum), (0., 1.))

    @classmethod
    def lognormal(cls, d50, geometric_sigma):
        if not np.isfinite([d50, geometric_sigma]).all() or d50 <= 0 or geometric_sigma < 1:
            raise ValueError('Invalid lognormal grain parameters')
        # FLOW approximates a lognormal by piecewise uniform log intervals.
        from scipy.special import ndtri
        p = np.array([0., .01, .03, .06, .10, .16, .24, .36, .64, .76, .84, .90, .94, .97, .99, 1.])
        z = np.r_[-3., np.round(ndtri(p[1:-1]), 4), 3.]
        return cls('lognormal', tuple(d50*geometric_sigma**z), tuple(p))

    @classmethod
    def legacy_d50(cls, d50):
        return cls('legacy_single_d50_piecewise_log_uniform',
                   (d50*.75**1.25, d50, d50*1.5**1.25), (0., .5, 1.))

    @property
    def fingerprint(self): return digest(self.__dict__)

    def moments(self):
        logd = np.log(self.diameter_m); p = np.diff(self.percentiles)
        a, b = logd[:-1], logd[1:]
        meanlog = np.sum(p*(a+b)/2.)
        variance = np.sum(p*(a*a+a*b+b*b)/3.)-meanlog**2
        mean = np.sum(p*np.where(b == a, np.exp(a),
                          (np.exp(b)-np.exp(a))/np.where(b == a, 1., b-a)))
        median = float(np.exp(np.interp(.5, self.percentiles, logd)))
        if self.kind == 'lognormal':
            sigma = (logd[-1]-logd[0])/6.
            mean = median*np.exp(.5*sigma*sigma) # rdsed analytic, not truncated-CDF mean
            variance = sigma*sigma
        return median, float(mean), float(-np.log(median)/np.log(2.)), float(np.sqrt(max(variance, 0.))/np.log(2.))


def prepare_grain_distributions_v101(distributions, nc, *, spatial_scale=None):
    """Spatial scale per class/cell preserves each supplied distribution's shape."""
    ng = len(distributions); points = max(len(d.diameter_m) for d in distributions)
    scale = np.ones((ng, nc)) if spatial_scale is None else np.asarray(spatial_scale, dtype=float)
    if scale.shape != (ng, nc) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid spatial grain distribution scale')
    nodes = np.empty((ng, points, nc)); cdf = np.empty_like(nodes)
    medians = np.empty((ng, nc)); means = np.empty_like(medians)
    phi = np.empty_like(medians); sig = np.empty_like(medians)
    for g, d in enumerate(distributions): # preparation only, no timestep numerical loop
        n = len(d.diameter_m)
        nodes[g, :n] = np.log(np.asarray(d.diameter_m))[:, None]+np.log(scale[g])[None, :]
        nodes[g, n:] = nodes[g, n-1]
        cdf[g, :n] = np.asarray(d.percentiles)[:, None]; cdf[g, n:] = 1.
        median, mean, ph, sigma = d.moments()
        medians[g] = median*scale[g]; means[g] = mean*scale[g]
        phi[g] = ph-np.log2(scale[g]); sig[g] = sigma
    receipt = {'distributions': [d.fingerprint for d in distributions],
               'resolved_arrays': array_sha(nodes, cdf, medians, means, phi, sig)}
    return dict(nodes=nodes, cdf=cdf, medians=medians, means=means, phi=phi, sigma=sig, receipt=receipt)


def characteristics_v101(runtime, fractions, prepared, quantiles=(.1, .5, .9)):
    q = np.ascontiguousarray(quantiles, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError('Quantiles must be between zero and one')
    nc = runtime.nc; ng = len(runtime.methods.fraction_ids)
    if fractions.shape != (ng, nc) or not np.isfinite(host(fractions)).all() or np.any(host(fractions) < 0):
        raise ValueError('Invalid mixture fractions')
    arrays = [np.ascontiguousarray(prepared[k]) for k in ('medians', 'means', 'nodes', 'cdf')]
    if any(not np.isfinite(a).all() for a in arrays) or arrays[0].shape != (ng, nc) or arrays[1].shape != (ng, nc) or arrays[2].shape != arrays[3].shape or arrays[2].shape[::2] != (ng, nc):
        raise ValueError('Prepared distribution dimensions differ')
    npnt = arrays[2].shape[1]
    if hasattr(runtime, 'cp'):
        cp = runtime.cp
        if not isinstance(fractions, cp.ndarray) or fractions.dtype != cp.float64 or not fractions.flags.c_contiguous:
            raise ValueError('CUDA characteristics require resident fractions')
        out = cp.empty((3+len(q), nc), cp.float64)
        runtime.module.get_function('braids_bed_characteristics_v101')(((nc+127)//128,), (128,),
            (np.int32(nc), np.int32(ng), np.int32(npnt), np.int32(len(q)), fractions,
             *(cp.asarray(a) for a in arrays), cp.asarray(q), out))
    else:
        out = np.empty((3+len(q), nc))
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f = native_runtime_v101()[1].braids_bed_characteristics_v101
        f.argtypes = [ctypes.c_int]*5+[real]*7; f.restype = None
        f(nc, ng, npnt, len(q), runtime.threads, np.ascontiguousarray(fractions), *arrays, q, out)
    return out


def fixfac_v101(owner, depth, *, mode=1, threshold_m=.05):
    if mode not in (1, 2) or not np.isfinite(threshold_m) or threshold_m < 0:
        raise ValueError('Invalid FLOW IThresh/Thresh')
    r = owner.runtime; ng = len(r.methods.fraction_ids)
    target = np.ascontiguousarray(r.methods.targets(host(depth), r.nl)[0])
    if hasattr(r, 'cp'):
        out = r.cp.empty((ng, r.nc), r.cp.float64)
        r.module.get_function('braids_bed_fixfac_v101')(((r.nc+127)//128,), (128,),
            (np.int32(r.nc), np.int32(ng), np.int32(r.nl), np.int32(mode), np.float64(threshold_m),
             owner.mass, owner.sv, r.device_rho, r.cp.asarray(target), out))
    else:
        out = np.empty((ng, r.nc)); real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f = native_runtime_v101()[1].braids_bed_fixfac_v101
        f.argtypes = [ctypes.c_int]*5+[ctypes.c_double]+[real]*5; f.restype = None
        f(r.nc, ng, r.nl, r.threads, mode, threshold_m, owner.mass, owner.sv, r.rho, target, out)
    return out


def availability_v101(owner, depth, *, mode=1, threshold_m=.05,
                      tracer_calibration=(), spatial_diameter_m=None):
    """One accepted-column fraction/fixed-stock owner, with FLOW tracer rules.

    FLOW applies tracer calibration only to a mixed bed with Thresh<=0;
    otherwise the saved calibration is inactive, not silently reinterpreted.
    """
    owner.validate(); r = owner.runtime; ng = len(r.methods.fraction_ids)
    calibration = np.ascontiguousarray(tracer_calibration or (0.,)*ng, dtype=float)
    if calibration.shape != (ng,) or not np.isfinite(calibration).all() or np.any(calibration < 0):
        raise ValueError('Invalid per-class tracer calibration')
    fractions = owner.surface_fraction
    fixed = fixfac_v101(owner, depth, mode=mode, threshold_m=threshold_m)
    active = r.methods.bed == 'mixed' and threshold_m <= 0 and np.any(calibration > 0)
    receipt = {'column_state': owner.state_sha256, 'runtime': r.identity,
               'calibration': calibration.tolist(), 'active': bool(active),
               'reason': 'FLOW_mixed_zero_threshold_tracer' if active else
                   'FLOW_layered_or_positive_threshold_no_tracer' if np.any(calibration > 0) else 'no_tracer'}
    if not active: return fractions, fixed, receipt
    diameter = np.ascontiguousarray(np.broadcast_to(
        np.asarray(r.methods.diameter_m)[:, None] if spatial_diameter_m is None else spatial_diameter_m,
        (ng, r.nc)))
    if not np.isfinite(diameter).all() or np.any(diameter <= 0): raise ValueError('Invalid tracer diameter')
    rho = np.ascontiguousarray(r.methods.solid_density_kg_m3)
    receipt['diameters'] = array_sha(diameter)
    if hasattr(r, 'cp'):
        cp = r.cp; outfrac = cp.empty_like(fractions); outfix = cp.empty_like(fixed)
        r.module.get_function('braids_bed_tracer_v101')(((r.nc+127)//128,), (128,),
            (np.int32(r.nc), np.int32(ng), owner.mass[0], cp.asarray(rho), cp.asarray(diameter),
             cp.asarray(calibration), fractions, fixed, outfrac, outfix))
    else:
        outfrac = np.empty_like(fractions); outfix = np.empty_like(fixed)
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f = native_runtime_v101()[1].braids_bed_tracer_v101
        f.argtypes = [ctypes.c_int]*3+[real]*8; f.restype = None
        f(r.nc, ng, r.threads, owner.mass[0], rho, diameter, calibration,
          np.ascontiguousarray(fractions), fixed, outfrac, outfix)
    return outfrac, outfix, receipt
