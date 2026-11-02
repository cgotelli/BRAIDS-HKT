"""Authoritative noncohesive column inventories, shared native CPU/CUDA equations.

Mass is kg/m2 per class and layer; solid fraction and fixed rock define bed level.
This is a bounded implementation route, not public admission of every Q3/Q4 profile.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT/'solver/wave/braids_bed_columns_v101.h'
SOURCE = HEADER.with_suffix('.cpp')
SCOPE = 'braids-q5a-column-owner-v101'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def array_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


@dataclass(frozen=True)
class ColumnMethodsV101:
    fraction_ids: tuple[str, ...]
    solid_density_kg_m3: tuple[float, ...]
    dry_density_kg_m3: tuple[float, ...]
    diameter_m: tuple[float, ...]
    phi_std: tuple[float, ...] = ()
    bed: str = 'layered'
    porosity: int = 0
    solid_fraction: float = .6
    sand_solid_fraction: float = .6
    mud_solid_fraction: float = .4
    top_update: int = 1
    base_update: int = 1
    transport_layer: str = 'constant'
    transport_thickness_m: float = .02
    depth_fraction: float = .01
    minimum_thickness_m: float = .001
    exchange_thickness_m: float = 0.
    underlayer_thickness_m: float = .02
    lagrangian_thickness_m: float | None = None
    eulerian_thickness_m: float | None = None
    lagrangian_underlayers: int = 0

    def __post_init__(self):
        ng = len(self.fraction_ids)
        if ng < 1 or len(set(self.fraction_ids)) != ng or any(not x for x in self.fraction_ids):
            raise ValueError('Unique ordered grain identities required')
        for x in (self.solid_density_kg_m3, self.dry_density_kg_m3, self.diameter_m):
            if len(x) != ng or not np.isfinite(x).all() or min(x) <= 0:
                raise ValueError('Positive per-class densities and diameters required')
        if self.phi_std and (len(self.phi_std) != ng or min(self.phi_std) < 0 or not np.isfinite(self.phi_std).all()):
            raise ValueError('Invalid within-class grain distribution')
        if self.bed not in ('mixed', 'layered') or self.porosity not in range(6):
            raise ValueError('Unsupported bed or porosity method')
        if self.bed == 'mixed' and self.porosity != 0:
            raise ValueError('Classic FLOW mixed bed incorporates porosity in dry density')
        if self.top_update not in (1, 2) or self.base_update not in range(1, 5):
            raise ValueError('Invalid top/base update')
        if self.transport_layer not in ('constant', 'depth'):
            raise ValueError('Dune-scaled layers require Q7a/Q5b; never silently substitute')
        for v in (self.solid_fraction, self.sand_solid_fraction, self.mud_solid_fraction):
            if not 0 < v <= 1:
                raise ValueError('Solid fraction must be in (0,1]')
        vals = (self.transport_thickness_m, self.depth_fraction, self.minimum_thickness_m,
                self.underlayer_thickness_m, self.exchange_thickness_m)
        if not np.isfinite(vals).all() or min(vals[:4]) <= 0 or vals[4] < 0:
            raise ValueError('Invalid layer thickness controls')
        if type(self.lagrangian_underlayers) is not int or self.lagrangian_underlayers < 0:
            raise ValueError('Invalid Lagrangian layer count')
        optional = (self.lagrangian_thickness_m, self.eulerian_thickness_m)
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in optional):
            raise ValueError('Lagrangian/Eulerian layer thicknesses must be positive when supplied')

    @property
    def fingerprint(self):
        return digest({'scope': SCOPE, **asdict(self)})

    @property
    def density(self):
        # Locked FLOW rdmorlyr uses solid density for every nonzero option,
        # including its CDRYB flag 5 (whose getporosity falls through to zero).
        return np.asarray(self.dry_density_kg_m3 if self.porosity == 0
                          else self.solid_density_kg_m3)

    def targets(self, depth, nl):
        depth = np.asarray(depth, dtype=float)
        if depth.ndim != 1 or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError('Finite nonnegative hydraulic depth required')
        if (self.bed == 'mixed') != (nl == 1):
            raise ValueError('Mixed bed has one layer; layered bed has at least two')
        if self.lagrangian_underlayers > max(0, nl-2-int(self.exchange_thickness_m > 0)):
            raise ValueError('Lagrangian underlayers leave no independent basal layer')
        eulerian = self.underlayer_thickness_m if self.eulerian_thickness_m is None else self.eulerian_thickness_m
        lagrangian = self.underlayer_thickness_m if self.lagrangian_thickness_m is None else self.lagrangian_thickness_m
        result = np.full((nl, len(depth)), eulerian)
        result[0] = (self.transport_thickness_m if self.transport_layer == 'constant'
                     else np.maximum(self.depth_fraction*depth, self.minimum_thickness_m))
        if self.exchange_thickness_m > 0:
            if nl < 3:
                raise ValueError('Separate exchange layer requires an underlying basal layer')
            result[1] = self.exchange_thickness_m
        start = 1+int(self.exchange_thickness_m > 0)
        result[start:start+self.lagrangian_underlayers] = lagrangian
        return np.ascontiguousarray(result)


def target_controls(methods, nl):
    """Validate topology once and bind explicit backend-neutral coefficients."""
    methods.targets(np.zeros(1), nl)
    return (1 if methods.transport_layer == 'constant' else 2, methods.lagrangian_underlayers,
            methods.transport_thickness_m, methods.depth_fraction, methods.minimum_thickness_m,
            methods.underlayer_thickness_m if methods.eulerian_thickness_m is None else methods.eulerian_thickness_m,
            methods.exchange_thickness_m,
            methods.underlayer_thickness_m if methods.lagrangian_thickness_m is None else methods.lagrangian_thickness_m)


def column_grain_phi(methods):
    """FLOW packing moments from resolved diameters, expressed on the mm scale.

    setbedfracprop uses phi=13.9 on the metre scale for d <= 100 um.
    The common shift to millimetres does not change mixture variance. This
    affects packing only, never the actual transport or settling diameter.
    """
    diameter = np.asarray(methods.diameter_m)
    return np.where(diameter <= .0001, 13.9+np.log2(.001), -np.log2(diameter/.001))


def porosity_reference(methods, masses, phi=None, sigma=None):
    """Independent equation oracle / initialization; stepping is native, not Python."""
    m = np.asarray(masses, dtype=float)
    phi = (column_grain_phi(methods)[:, None]
           if phi is None else np.asarray(phi))
    sigma = (np.asarray(methods.phi_std or (0.,)*len(methods.fraction_ids))[:, None]
             if sigma is None else np.asarray(sigma))
    total = m.sum(axis=0); f = m/np.where(total > 0, total, 1.)
    mean = (f*phi).sum(axis=0)
    spread = np.sqrt((f*((phi-mean)**2+sigma**2)).sum(axis=0))
    if methods.porosity == 1:
        p = np.maximum(0., .36-.06*spread)
    elif methods.porosity == 2:
        x = 3.7632*np.maximum(spread, 1e-200)**(-.7552)
        p = .45*x/(1.+x)
    elif methods.porosity in (3, 4):
        p = np.full(total.shape, 1.-(methods.solid_fraction if methods.porosity == 3
                                   else methods.sand_solid_fraction))
    else:
        p = np.zeros(total.shape)
    return np.where(total > 0., p, 0.)


def initialize_columns_v101(methods, thickness, fractions, *, basis='solid_volume',
                            phi=None, sigma=None):
    thick = np.asarray(thickness, dtype=float)
    f = np.asarray(fractions, dtype=float)
    if thick.ndim != 2 or f.shape != (thick.shape[0], len(methods.fraction_ids), thick.shape[1]):
        raise ValueError('Initial layer/class/cell shapes differ')
    if (not np.isfinite(thick).all() or not np.isfinite(f).all() or np.any(thick < 0)
            or np.any(f < 0) or not np.allclose(f.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Initial thickness and normalized fractions required')
    if basis not in ('mass', 'solid_volume'):
        raise ValueError('Fraction basis must be mass or solid_volume')
    rho = methods.density
    weights = f if basis == 'mass' else f*rho[None, :, None]
    weights = weights/weights.sum(axis=1, keepdims=True)
    sv = np.stack([1.-porosity_reference(methods, w, phi, sigma) for w in weights])
    mass = weights*(thick*sv)[:, None, :]/(weights/rho[None, :, None]).sum(axis=1)[:, None, :]
    return np.ascontiguousarray(mass), np.ascontiguousarray(sv)


def read_inicomp_v101(path, methods, nc, *, spatial_fields=None):
    """FLOW 01.00/02.00 repeated [Layer] payload, uniform or named spatial fields.

    Named fields must be supplied by the prepared-grid reader; no guessing at
    another mesh's ordering. Input bytes and all resolved arrays are hashed.
    """
    text = Path(path).read_text(encoding='utf-8-sig')
    groups = []; group = None
    for row in text.splitlines():
        row = row.split('#', 1)[0].split(';', 1)[0].strip()
        if not row: continue
        if row.startswith('[') and row.endswith(']'):
            group = {}; groups.append((row[1:-1].lower(), group)); continue
        if '=' not in row or group is None: raise ValueError('Malformed IniComp')
        key, val = (v.strip() for v in row.split('=', 1))
        key = key.lower()
        if key in group: raise ValueError('Duplicate IniComp field')
        group[key] = val
    version = next((g.get('fileversion') for name, g in groups
                    if name == 'bedcompositionfileinformation'), None)
    if version not in ('01.00', '02.00'): raise ValueError('IniComp requires FLOW 01.00/02.00')
    def field(raw):
        try: a = np.full(nc, float(raw))
        except ValueError:
            if spatial_fields is None or raw not in spatial_fields:
                raise ValueError('IniComp spatial field must be prepared explicitly: '+raw)
            a = np.asarray(spatial_fields[raw], dtype=float)
        if a.shape != (nc,) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid IniComp field')
        return a
    masses = []; solids = []
    for name, g in groups:
        if name != 'layer': continue
        kind = g.get('type', '').lower()
        sediment = kind in ('sediment mass', 'sediment thickness')
        if kind not in ('mass fraction', 'volume fraction', 'sediment mass', 'sediment thickness'):
            raise ValueError('Unsupported IniComp layer type')
        key = 'sedbed' if sediment else 'fraction'
        forbidden = 'fraction' if sediment else 'sedbed'
        if any(k.startswith(forbidden) for k in g): raise ValueError('Mixed IniComp units')
        keys = ([key+str(i+1) for i in range(len(methods.fraction_ids))] if version == '01.00'
                else [i.lower() for i in methods.fraction_ids])
        f = np.stack([field(g.get(k, '0')) for k in keys])
        if sediment:
            m = f if kind == 'sediment mass' else f*np.asarray(methods.dry_density_kg_m3)[:, None]
            sv = 1.-porosity_reference(methods, m)
        else:
            if 'thick' not in g: raise ValueError('IniComp fraction layer requires Thick')
            t = field(g['thick'])
            m, sv = initialize_columns_v101(methods, t[None, :], f[None, :, :],
                basis='mass' if kind == 'mass fraction' else 'solid_volume')
            m, sv = m[0], sv[0]
        masses.append(m); solids.append(sv)
    if not masses: raise ValueError('IniComp contains no layers')
    m, sv = np.ascontiguousarray(masses), np.ascontiguousarray(solids)
    return m, sv, {'schema': 1, 'file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   'resolved_sha256': array_sha(m, sv), 'methods': methods.fingerprint}


@lru_cache(maxsize=1)
def native_runtime_v101():
    compiler = shutil.which('g++')
    if not compiler: raise RuntimeError('Native column route requires an OpenMP C++ compiler')
    flags = ['-O3', '-fopenmp', '-shared', '-ffp-contract=off', '-std=c++11']
    if os.name == 'nt':
        # PE link timestamps changed the authenticated runtime in every fresh
        # process, rejecting otherwise identical column checkpoints. Keep the
        # binary hash in the identity, but make repeated builds reproducible.
        flags += ['-static-libstdc++', '-static-libgcc', '-Wl,--no-insert-timestamp']
    inputs={'source':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'equations':hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            'compiler':hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),'flags':flags}
    directory=ROOT/'build/bed_columns_v101'/digest(inputs);directory.mkdir(parents=True,exist_ok=True)
    library=directory/('columns.dll' if os.name=='nt' else 'columns.so')
    # Reuse the authenticated binary across processes. MinGW can also embed
    # temporary object names, even when its PE timestamp is disabled.
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix='compile_',dir=directory) as work:
            candidate=Path(work)/library.name
            build=subprocess.run([compiler,*flags,str(SOURCE),'-o',str(candidate)],cwd=work,capture_output=True,text=True,timeout=120)
            if build.returncode:raise RuntimeError(build.stdout+build.stderr)
            try:
                # Exclusive creation: concurrent initializers keep the first
                # complete binary, never overwrite a DLL loaded by another.
                os.link(candidate,library)
            except FileExistsError:pass
    dependency = os.add_dll_directory(str(Path(compiler).parent)) if os.name == 'nt' else None
    dll = ctypes.CDLL(str(library)); fn = dll.braids_bed_columns_v101
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    fn.argtypes = ([ctypes.c_int]*9+[ctypes.c_double]+[real]*7+[integer]+[ctypes.c_double]*3+
                   [ctypes.c_int]+[real]*6+[integer])
    fn.restype = None
    receipt = {'source': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'equations': hashlib.sha256(HEADER.read_bytes()).hexdigest(),
        'compiler': hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        'binary': hashlib.sha256(library.read_bytes()).hexdigest(), 'flags': flags}
    return fn, dll, dependency, receipt


class NativeColumnRuntimeV101:
    backend = 'native_cpu_openmp'

    def __init__(self, methods, nc, nl, *, threads=4, phi=None, sigma=None,
                 diffusion_depths=(), diffusion_coefficients=None):
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError('Explicit valid native thread count required')
        self.methods, self.nc, self.nl, self.threads = methods, nc, nl, threads
        self.target_controls = target_controls(methods, nl)
        self.rho = np.ascontiguousarray(methods.density)
        self.phi = np.ascontiguousarray(np.broadcast_to(
            column_grain_phi(methods)[:, None] if phi is None else phi,
            (len(methods.fraction_ids), nc)))
        self.sigma = np.ascontiguousarray(np.broadcast_to(
            np.asarray(methods.phi_std or (0.,)*len(methods.fraction_ids))[:, None] if sigma is None else sigma,
            self.phi.shape))
        if not np.isfinite(self.phi).all() or not np.isfinite(self.sigma).all() or np.any(self.sigma < 0):
            raise ValueError('Invalid spatial distribution moments')
        self.mud = np.zeros(len(methods.fraction_ids), dtype=np.int32) # noncohesive Q5a
        self.zd = np.ascontiguousarray(diffusion_depths, dtype=float)
        self.kd = np.empty((0, nc)) if diffusion_coefficients is None else np.ascontiguousarray(diffusion_coefficients, dtype=float)
        if (self.zd.ndim != 1 or self.kd.shape != (len(self.zd), nc)
                or not np.isfinite(self.zd).all() or not np.isfinite(self.kd).all()
                or np.any(self.zd < 0) or np.any(np.diff(self.zd) <= 0) or np.any(self.kd < 0)):
            raise ValueError('NDiff/KDiff/ZDiff shapes and nonnegative values required')
        self.fn, _, _, self.receipt = native_runtime_v101()
        self.identity = digest({'backend': self.backend, 'methods': methods.fingerprint,
            'runtime': self.receipt, 'fields': array_sha(self.phi, self.sigma, self.zd, self.kd),
            'nc': nc, 'nl': nl, 'threads': threads})

    def targets(self, depth):
        depth = np.ascontiguousarray(depth, dtype=float)
        if depth.shape != (self.nc,) or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError('Finite nonnegative column depths required')
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        fn = native_runtime_v101()[1].braids_bed_targets_v101
        fn.argtypes = [ctypes.c_int]*5+[ctypes.c_double]*6+[real]*2
        fn.restype = None
        out = np.empty((self.nl, self.nc))
        fn(self.nc, self.nl, self.threads, *self.target_controls, depth, out)
        return out

    def evaluate(self, mass, sv, delta, targets, dt, *, coarse_mobility=None):
        ng = len(self.methods.fraction_ids)
        inputs = [np.ascontiguousarray(x, dtype=float) for x in (mass, sv, delta, targets)]
        shapes = ((self.nl, ng, self.nc), (self.nl, self.nc), (ng, self.nc), (self.nl, self.nc))
        if any(a.shape != s or not np.isfinite(a).all() for a, s in zip(inputs, shapes)):
            raise ValueError('Column shapes or finite values differ')
        if np.any(inputs[0] < 0) or np.any(inputs[1] <= 0) or np.any(inputs[1] > 1) or np.any(inputs[3] <= 0):
            raise ValueError('Column domain differs')
        if not np.isfinite(dt) or dt < 0: raise ValueError('Invalid composition interval')
        output = np.empty(shapes[0]); solid = np.empty(shapes[1]); source = np.empty(shapes[2])
        work = np.empty((max(7*self.nl+2*self.nl*ng, ng), self.nc)); status = np.empty(self.nc, dtype=np.int32)
        m = self.methods
        fn, extra = self.fn, ()
        if coarse_mobility is not None:
            mobile = np.ascontiguousarray(coarse_mobility, dtype=float)
            if (mobile.shape != shapes[2] or not np.isfinite(mobile).all()
                    or np.any(mobile < 0) or np.any(mobile > 1)
                    or self.nl < 3 + int(m.exchange_thickness_m > 0) + m.lagrangian_underlayers):
                raise ValueError('Invalid coarse mobility or independent basal layer')
            fn = native_runtime_v101()[1].braids_bed_columns_coarse_v101
            real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            fn.argtypes = self.fn.argtypes + [real]; fn.restype = None
            extra = (mobile,)
        fn(self.nc, ng, self.nl, self.threads, m.top_update, m.base_update, m.porosity,
            m.lagrangian_underlayers, int(m.exchange_thickness_m > 0),
            dt, *inputs, self.rho, self.phi, self.sigma, self.mud,
            m.solid_fraction, m.sand_solid_fraction, m.mud_solid_fraction,
            len(self.zd), self.zd, self.kd, output, solid, source, work, status, *extra)
        if np.any(status):
            cells = np.flatnonzero(status)[:8]
            details = [dict(cell=int(c), status=int(status[c]),
                            available=inputs[0][0, :, c].tolist(),
                            delta=inputs[2][:, c].tolist()) for c in cells]
            raise ValueError('Column finite stock overdraw or nonphysical state: '+str(details))
        return output, solid, source

    def limits(self, active, water, depth, bedrate, waterrate, *, morfac,
               max_bed_change_m=0., max_concentration_kg_m3=0., entrainment_fraction=0.):
        arrays = [np.ascontiguousarray(x, dtype=float) for x in (active, water, depth, bedrate, waterrate)]
        ng = len(self.methods.fraction_ids)
        shapes = ((ng, self.nc), (ng, self.nc), (self.nc,), (ng, self.nc), (ng, self.nc))
        if any(a.shape != s or not np.isfinite(a).all() for a, s in zip(arrays, shapes)):
            raise ValueError('Limiter inputs differ')
        controls = (morfac, max_bed_change_m, max_concentration_kg_m3, entrainment_fraction)
        if not np.isfinite(controls).all() or min(controls) < 0 or any(np.any(a < 0) for a in arrays[:3]):
            raise ValueError('Limiter domain differs')
        f = native_runtime_v101()[1].braids_bed_limits_v101
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        f.argtypes = [ctypes.c_int]*3+[ctypes.c_double]+[real]*6+[ctypes.c_double]*3+[real]
        f.restype = None
        out = np.empty((4, self.nc))
        f(self.nc, ng, self.threads, morfac, *arrays, self.rho,
          max_bed_change_m, max_concentration_kg_m3, entrainment_fraction, out)
        return out

    def remap(self, mass, sv, rock, offsets, indices, weights, newnc):
        f = native_runtime_v101()[1].braids_bed_remap_v101
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        f.argtypes = [ctypes.c_int]*5+[integer]*2+[real]*8; f.restype = None
        ng = len(self.methods.fraction_ids)
        out = np.empty((self.nl, ng, newnc)); solid = np.empty((self.nl, newnc)); datum = np.empty(newnc)
        f(newnc, self.nc, ng, self.nl, self.threads, offsets, indices, weights,
          mass, sv, rock, self.rho, out, solid, datum)
        return out, solid, datum

    def integrate_rate(self, rate, dt, enabled=()):
        ng = len(self.methods.fraction_ids)
        a = np.ascontiguousarray(rate, dtype=float)
        mask = np.ascontiguousarray(enabled or (True,)*ng, dtype=np.int32)
        if (a.shape != (ng, self.nc) or not np.isfinite(a).all()
                or mask.shape != (ng,) or np.any((mask != 0) & (mask != 1))
                or not np.isfinite(dt) or dt < 0):
            raise ValueError('Invalid native composition rate, interval or fraction switches')
        f = native_runtime_v101()[1].braids_bed_integrate_v101
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        f.argtypes = [ctypes.c_int]*3+[ctypes.c_double, integer, real, real]; f.restype = None
        delta = np.empty_like(a)
        f(self.nc, ng, self.threads, dt, mask, a, delta)
        return delta


def remap_columns_v101(owner, target_runtime, offsets, indices, weights, target_area):
    """Conservative sparse layer-aligned restriction/prolongation, without time advance.

    This transfers a prescribed mapping, not general hierarchy construction.
    Layers with incompatible physical boundaries are not merged by this API.
    """
    owner.validate()
    if owner.pending is not None: raise ValueError('Remap requires committed columns')
    if (owner.runtime.methods.fingerprint != target_runtime.methods.fingerprint
            or owner.runtime.nl != target_runtime.nl or owner.runtime.backend != target_runtime.backend):
        raise ValueError('Regrid cannot change physics or backend')
    offsets = np.ascontiguousarray(offsets, dtype=np.int32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    weights = np.ascontiguousarray(weights, dtype=float)
    area = np.ascontiguousarray(target_area, dtype=float)
    nc = target_runtime.nc
    if (offsets.shape != (nc+1,) or offsets[0] != 0 or offsets[-1] != len(indices)
            or weights.shape != indices.shape or np.any(np.diff(offsets) <= 0)
            or np.any(indices < 0) or np.any(indices >= owner.runtime.nc)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or area.shape != (nc,) or not np.isfinite(area).all() or np.any(area <= 0)):
        raise ValueError('Invalid conservative regrid mapping')
    row = np.repeat(np.arange(nc), np.diff(offsets))
    if (not np.allclose(np.bincount(row, weights=weights), 1., atol=1e-13, rtol=0)
            or not np.allclose(np.bincount(indices, weights=weights*area[row], minlength=owner.runtime.nc),
                               host(owner.area), atol=1e-12, rtol=0)):
        raise ValueError('Regrid mapping does not conserve area')
    mass, sv, rock = owner.runtime.remap(owner.mass, owner.sv, owner.rock,
                                         offsets, indices, weights, nc)
    mapped = ColumnOwnerV101(target_runtime, mass, sv, rock, area, time_s=owner.time_s)
    mapped.updates = owner.updates
    for key in ('reference', 'external', 'basal'): setattr(mapped, key, getattr(owner, key).copy())
    mapped._seal()
    budget = host((mapped.mass*mapped.area[None, None, :]).sum(axis=(0, 2)))-mapped.reference-mapped.external-mapped.basal
    if np.max(np.abs(budget)) > 1e-10*max(1., np.max(mapped.reference)):
        raise ValueError('Remapped class budget differs')
    return mapped


class ColumnOwnerV101:
    """One accepted owner; native/device trials do not publish clocks or inventories."""

    def __init__(self, runtime, mass, sv, rock, area, *, time_s=0.):
        self.runtime = runtime
        self.xp = getattr(runtime, 'cp', np)
        self.mass = self.xp.array(mass, dtype=self.xp.float64, order='C', copy=True)
        self.sv = self.xp.array(sv, dtype=self.xp.float64, order='C', copy=True)
        self.rock = self.xp.array(rock, dtype=self.xp.float64, order='C', copy=True)
        self.area = self.xp.array(area, dtype=self.xp.float64, order='C', copy=True)
        ng = len(runtime.methods.fraction_ids)
        if (self.mass.shape != (runtime.nl, ng, runtime.nc) or self.sv.shape != (runtime.nl, runtime.nc)
                or self.rock.shape != (runtime.nc,) or self.area.shape != (runtime.nc,)):
            raise ValueError('Owner dimensions differ')
        if (not all(np.isfinite(host(x)).all() for x in (self.mass, self.sv, self.rock, self.area))
                or np.any(host(self.mass) < 0) or np.any(host(self.sv) <= 0)
                or np.any(host(self.sv) > 1) or np.any(host(self.area) <= 0)
                or not np.isfinite(time_s) or time_s < 0):
            raise ValueError('Invalid initial accepted column')
        self.time_s = float(time_s); self.updates = 0
        self.reference = host((self.mass*self.area[None, None, :]).sum(axis=(0, 2))).copy()
        self.external = np.zeros(ng); self.basal = np.zeros(ng)
        self.pending = None; self._seal()

    def _token(self):
        return digest({'runtime': self.runtime.identity, 'updates': self.updates, 'time': self.time_s,
            'arrays': array_sha(*(host(a) for a in (self.mass, self.sv, self.rock, self.area,
                                                   self.reference, self.external, self.basal)))})

    def _seal(self):
        if self.xp is np:
            self.mass.setflags(write=False); self.sv.setflags(write=False)
            self.area.setflags(write=False); self.rock.setflags(write=False)
        self.state_sha256 = self._token()

    def validate(self):
        if self._token() != self.state_sha256: raise ValueError('Accepted column was mutated')

    @property
    def total(self): return self.mass.sum(axis=0)

    @property
    def available(self): return self.mass[0]

    @property
    def surface_fraction(self):
        v = self.mass[0]/self.xp.asarray(self.runtime.rho)[:, None]
        total = v.sum(axis=0)
        return self.xp.where(total[None, :] > 0., v/self.xp.maximum(total[None, :], 1e-300), 0.)

    @property
    def bed(self):
        v = (self.mass/self.xp.asarray(self.runtime.rho)[None, :, None]).sum(axis=1)/self.sv
        return self.rock+v.sum(axis=0)

    def begin(self, delta, depth, dt, *, cancelled=False):
        self.validate()
        if self.pending is not None: raise ValueError('An uncommitted column trial already exists')
        if cancelled: raise InterruptedError('Column update cancelled before launch')
        if self.updates >= 20: raise ValueError('Bounded column route permits at most 20 accepted updates')
        targets = self.runtime.methods.targets(host(depth), self.runtime.nl)
        if self.xp is not np: targets = self.xp.asarray(targets)
        out, sv, basal = self.runtime.evaluate(self.mass, self.sv, delta, targets, dt)
        before = host((self.mass*self.area[None, None, :]).sum(axis=(0, 2)))
        after = host((out*self.area[None, None, :]).sum(axis=(0, 2)))
        ext = host((delta*self.area[None, :]).sum(axis=1))
        base = host((basal*self.area[None, :]).sum(axis=1))
        residual = after-before-ext-base
        if np.max(np.abs(residual)) > 2e-11*max(1., np.max(before)):
            raise ValueError('Classwise column budget does not close')
        self.pending = (out, sv, ext, base, float(dt), residual)
        self.seal_pending()
        return self.pending

    def _pending_token(self):
        return digest({'arrays': array_sha(*(host(x) for x in self.pending[:4])),
                       'dt': self.pending[4], 'residual': host(self.pending[5]).tolist()})

    def seal_pending(self):
        self.pending_sha256 = self._pending_token()

    def validate_pending(self):
        if self.pending is None: raise ValueError('No column trial')
        if self.pending_sha256 != self._pending_token(): raise ValueError('Column trial was mutated')

    def commit(self, *, cancelled=False):
        self.validate()
        if self.pending is None: raise ValueError('No column trial to commit')
        if cancelled:
            self.rollback(); raise InterruptedError('Column update cancelled before publication')
        self.validate_pending()
        out, sv, ext, base, dt, residual = self.pending
        self.mass, self.sv = out, sv; self.external += ext; self.basal += base
        self.time_s += dt; self.updates += 1; self.pending = None; self._seal()
        return residual

    def rollback(self):
        self.pending = None; self.pending_sha256 = None

    def checkpoint(self):
        self.validate()
        if self.pending is not None: raise ValueError('Checkpoint requires committed column')
        d = {'schema': 1, 'scope': SCOPE, 'runtime': self.runtime.identity,
            'methods': self.runtime.methods.fingerprint, 'time_s': self.time_s, 'updates': self.updates,
            **{k: host(getattr(self, k)).tolist() for k in
               ('mass', 'sv', 'rock', 'area', 'reference', 'external', 'basal')},
            'state_sha256': self.state_sha256}
        return {**d, 'checkpoint_sha256': digest(d)}

    def restore(self, d):
        if self.pending is not None: raise ValueError('Restore requires no open trial')
        raw = {k: v for k, v in d.items() if k != 'checkpoint_sha256'}
        if (digest(raw) != d['checkpoint_sha256'] or d['runtime'] != self.runtime.identity
                or d['methods'] != self.runtime.methods.fingerprint or d['scope'] != SCOPE or d['schema'] != 1):
            raise ValueError('Column restart identity differs')
        candidate = ColumnOwnerV101(self.runtime, d['mass'], d['sv'], d['rock'], d['area'], time_s=d['time_s'])
        candidate.updates = d['updates']
        if type(candidate.updates) is not int or not 0 <= candidate.updates <= 20:
            raise ValueError('Invalid restart update count')
        for key in ('reference', 'external', 'basal'):
            value = np.asarray(d[key], dtype=float)
            if value.shape != self.reference.shape or not np.isfinite(value).all():
                raise ValueError('Invalid restart ledger')
            setattr(candidate, key, value.copy())
        candidate._seal()
        if candidate.state_sha256 != d['state_sha256']: raise ValueError('Column restart state differs')
        self.__dict__.update(candidate.__dict__)
