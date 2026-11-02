"""Q7a current-only static bedforms, capacity bundles and accepted dune state.

The numerical equations are shared by the native OpenMP and CUDA backends.
This module owns method compatibility and state identity; it never chooses a
backend from mesh size and never silently disables an unsupported process.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "solver/wave/braids_static_bedforms_v102.h"
SOURCE = HEADER.with_suffix(".cpp")
SCOPE = "braids-q7a-static-bedforms-v102"

HEIGHT = {"van_rijn_1984": 1, "fredsoe_mpm": 2, "fredsoe_eh": 3, "power": 4}
LENGTH = {"van_rijn_1984": 1, "power": 2}
ROUGHNESS = {"van_rijn_2007": 0, "van_rijn_1984": 1, "power": 2}
CAPACITY = {"off": 0, "van_rijn_2007": 1, "santoss": 2}
SETTLING = {"grain_size": 0, "dss_2004": 1}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_digest(*arrays) -> str:
    h = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        h.update(array.dtype.str.encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class StaticMethodsV102:
    """Resolved physical choices.  ``backend`` is deliberately not included."""

    capacity: str = "off"
    height: str | None = None
    length: str | None = None
    roughness: str | None = None
    settling: str = "grain_size"
    bedforms: bool | None = None
    secondary_flow: str = "off"
    waves: bool = False
    slope_effects: bool = True
    phase_lag_effects: bool = True

    def __post_init__(self):
        if self.capacity not in CAPACITY or self.settling not in SETTLING:
            raise ValueError("Unknown Q7a capacity or settling method")
        if self.height is not None and self.height not in HEIGHT:
            raise ValueError("Unknown bedform-height method")
        if self.length is not None and self.length not in LENGTH:
            raise ValueError("Unknown bedform-length method")
        if self.roughness is not None and self.roughness not in ROUGHNESS:
            raise ValueError("Unknown bedform-roughness method")
        if self.secondary_flow not in ("off", "external"):
            raise ValueError("Unknown secondary-flow ownership")
        if self.waves:
            raise ValueError("Q7a is current-only; wave routes are not silently approximated")
        if self.capacity == "santoss" and self.secondary_flow != "off":
            raise ValueError("SANTOSS owns its direction effects; external secondary rotation is forbidden")
        if self.settling == "dss_2004" and self.capacity != "van_rijn_2007":
            raise ValueError("DSS_2004 settling is formula-coupled only to Van Rijn 2007")

    @property
    def fingerprint(self) -> str:
        return _digest({"scope": SCOPE, **asdict(self)})


@dataclass(frozen=True)
class ResolvedStaticMethodsV102:
    methods: StaticMethodsV102
    bedforms: bool
    height: str
    length: str
    roughness: str
    settling: str
    automatic: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return _digest({"scope": SCOPE, "methods": asdict(self.methods),
                        "resolved": {"bedforms": self.bedforms, "height": self.height,
                                     "length": self.length, "roughness": self.roughness,
                                     "settling": self.settling},
                        "automatic": self.automatic, "warnings": self.warnings})


def resolve_static_methods_v102(methods: StaticMethodsV102) -> ResolvedStaticMethodsV102:
    """Apply FLOW bundle defaults while preserving every explicit compatible choice."""
    auto, warnings = [], []
    bedforms = bool(methods.bedforms)
    height = methods.height or "van_rijn_1984"
    length = methods.length or "van_rijn_1984"
    roughness = methods.roughness or "van_rijn_2007"
    settling = methods.settling
    if methods.capacity in ("van_rijn_2007", "santoss") and methods.bedforms is False:
        raise ValueError(f"{methods.capacity} requires bedforms; an explicit off choice cannot be overwritten")
    if methods.capacity in ("van_rijn_2007", "santoss") and methods.bedforms is None:
        bedforms = True; auto.append("bedforms enabled by capacity bundle")
    if methods.capacity == "van_rijn_2007":
        if methods.roughness is not None and methods.roughness != "van_rijn_2007":
            raise ValueError("Van Rijn 2007 requires its static roughness predictor")
        roughness = "van_rijn_2007"
        if methods.roughness is None: auto.append("Van Rijn 2007 static roughness")
        if methods.settling == "grain_size":
            settling = "dss_2004"; auto.append("formula-coupled DSS_2004 settling")
    if methods.capacity == "santoss":
        if methods.roughness is None:
            roughness = "van_rijn_2007"; auto.append("SANTOSS default static Van Rijn 2007 roughness")
        elif methods.roughness != "van_rijn_2007":
            warnings.append("explicit non-default SANTOSS hydraulic roughness retained")
        if settling != "grain_size":
            raise ValueError("SANTOSS does not select the Van Rijn DSS_2004 branch")
    if not bedforms and any(x is not None for x in (methods.height, methods.length, methods.roughness)):
        raise ValueError("Bedform submethods require bedforms enabled")
    return ResolvedStaticMethodsV102(methods, bedforms, height, length, roughness,
                                     settling, tuple(auto), tuple(warnings))


@dataclass(frozen=True)
class StaticParametersV102:
    bedform_calibration: float = 1.0
    critical_shields: float = -1.0
    height_coefficient: float = 0.1
    height_exponent: float = 1.0
    length_coefficient: float = 7.3
    length_exponent: float = 1.0
    roughness_coefficient: float = 1.1
    roughness_exponent: float = 1.0
    ripple_calibration: float = 1.0
    mega_ripple_calibration: float = 0.0
    dune_calibration: float = 0.0
    gravity_m_s2: float = 9.81
    relative_density: float = 1.65
    viscosity_m2_s: float = 1.0e-6
    water_density_kg_m3: float = 1000.0
    sediment_density_kg_m3: float = 2650.0
    maximum_reference_concentration: float = 0.65
    critical_stress_cohesion_exponent: float = 1.0
    mud_critical_stress_exponent: float = 1.0
    flocculation_salinity_limit: float = 30.0
    flocculation_calibration: float = 1.0
    hindered_reference_concentration_kg_m3: float = 1600.0
    temperature_c: float = 20.0
    prescribed_critical_stress_pa: float = 0.0
    santoss_period_s: float = 0.01
    transport_critical_shields: float = -1.0
    critical_stress_mode: int = 1
    suspended_size_mode: int = 1

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values())[:-2], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Finite Q7a parameters required")
        positive = (self.bedform_calibration, self.height_coefficient, self.length_coefficient,
                    self.roughness_coefficient, self.gravity_m_s2, self.relative_density,
                    self.viscosity_m2_s, self.water_density_kg_m3,
                    self.sediment_density_kg_m3, self.maximum_reference_concentration,
                    self.flocculation_calibration, self.hindered_reference_concentration_kg_m3,
                    self.santoss_period_s)
        if min(positive) <= 0 or self.sediment_density_kg_m3 <= self.water_density_kg_m3:
            raise ValueError("Positive physical Q7a parameters and denser sediment required")
        if min(self.ripple_calibration, self.mega_ripple_calibration, self.dune_calibration) < 0:
            raise ValueError("Bedform calibrations cannot be negative")
        if self.critical_stress_mode not in (0, 1) or self.suspended_size_mode not in (0, 1):
            raise ValueError("Unsupported critical-stress or suspended-size mode")

    def wire(self) -> np.ndarray:
        return np.ascontiguousarray([
            self.bedform_calibration, self.critical_shields,
            self.height_coefficient, self.height_exponent,
            self.length_coefficient, self.length_exponent,
            self.roughness_coefficient, self.roughness_exponent,
            self.ripple_calibration, self.mega_ripple_calibration,
            self.dune_calibration, self.gravity_m_s2, self.relative_density,
            self.viscosity_m2_s, self.water_density_kg_m3,
            self.sediment_density_kg_m3, self.maximum_reference_concentration,
            self.critical_stress_cohesion_exponent, self.mud_critical_stress_exponent,
            self.flocculation_salinity_limit, self.flocculation_calibration,
            self.hindered_reference_concentration_kg_m3, self.temperature_c,
            self.prescribed_critical_stress_pa, self.santoss_period_s,
            self.transport_critical_shields], dtype=np.float64)


def _modes(resolved: ResolvedStaticMethodsV102, parameters: StaticParametersV102) -> np.ndarray:
    return np.ascontiguousarray([
        HEIGHT[resolved.height], LENGTH[resolved.length], ROUGHNESS[resolved.roughness],
        CAPACITY[resolved.methods.capacity], parameters.suspended_size_mode,
        parameters.critical_stress_mode, int(resolved.methods.slope_effects),
        int(resolved.methods.phase_lag_effects), int(resolved.bedforms)], dtype=np.int32)


def _field(value, nc: int, name: str, *, positive=False, nonnegative=False) -> np.ndarray:
    a = np.broadcast_to(np.asarray(value, dtype=np.float64), (nc,)).copy()
    if not np.isfinite(a).all() or (positive and np.any(a <= 0)) or (nonnegative and np.any(a < 0)):
        raise ValueError(f"Invalid {name}")
    return np.ascontiguousarray(a)


@lru_cache(maxsize=1)
def _native_library_v102():
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("Q7a native CPU route requires an OpenMP C++ compiler")
    directory = Path(tempfile.mkdtemp(prefix="braids_q7a_v102_"))
    library = directory / ("q7a.dll" if os.name == "nt" else "q7a.so")
    flags = ["-O3", "-fopenmp", "-shared", "-ffp-contract=off", "-std=c++11"]
    if os.name == "nt": flags += ["-static-libstdc++", "-static-libgcc"]
    build = subprocess.run([compiler, *flags, str(SOURCE), "-o", str(library)],
                           cwd=directory, capture_output=True, text=True, timeout=120)
    if build.returncode:
        raise RuntimeError(build.stdout + build.stderr)
    dependency = os.add_dll_directory(str(Path(compiler).parent)) if os.name == "nt" else None
    dll = ctypes.CDLL(str(library))
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    bed = dll.braids_static_bedforms_v102
    bed.argtypes = [ctypes.c_int, ctypes.c_int, integer] + [real] * 8 + [integer]
    bed.restype = None
    capacity = dll.braids_static_capacity_v102
    capacity.argtypes = [ctypes.c_int, ctypes.c_int, integer] + [real] * 15 + [integer]
    capacity.restype = None
    receipt = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "equations_sha256": hashlib.sha256(HEADER.read_bytes()).hexdigest(),
               "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
               "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
               "flags": flags}
    return bed, capacity, dll, dependency, receipt


class NativeStaticRuntimeV102:
    backend = "native_cpu_openmp"

    def __init__(self, methods: StaticMethodsV102, parameters=StaticParametersV102(), *, threads=4):
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError("Explicit CPU thread count in [1,256] required")
        self.resolved = resolve_static_methods_v102(methods)
        self.parameters = parameters; self.threads = threads
        self.mode = _modes(self.resolved, parameters); self.params = parameters.wire()
        self._bed, self._capacity, self._dll, self._dependency, receipt = _native_library_v102()
        self.receipt = receipt
        self.identity = _digest({"backend": self.backend, "methods": self.resolved.fingerprint,
                                 "parameters": asdict(parameters), "runtime": receipt})

    def bedforms(self, depth, u, v, chezy, d50, d90) -> Mapping[str, np.ndarray]:
        nc = np.asarray(depth).size
        fields = [_field(x, nc, n, positive=n in ("depth", "chezy", "d50", "d90"))
                  for x, n in ((depth, "depth"), (u, "u"), (v, "v"), (chezy, "chezy"),
                               (d50, "d50"), (d90, "d90"))]
        out = np.empty((9, nc), dtype=np.float64); status = np.empty(nc, dtype=np.int32)
        self._bed(nc, self.threads, self.mode, self.params, *fields, out, status)
        if np.any(status): raise ValueError(f"Nonphysical Q7a bedform cells: {np.flatnonzero(status).tolist()}")
        names = ("height", "length", "ripple_roughness", "mega_ripple_roughness",
                 "dune_roughness", "total_roughness", "critical_shields", "mobility", "dstar")
        return {name: out[i] for i, name in enumerate(names)}

    def capacity(self, depth, u, v, d10, d50, d90, roughness, *, skin_roughness=None, mud_fraction=0.,
                 concentration_kg_m3=0., salinity=0., slope_x=0., slope_y=0.) -> Mapping[str, np.ndarray]:
        nc = np.asarray(depth).size
        skin_roughness = roughness if skin_roughness is None else skin_roughness
        values = ((depth, "depth", True, False), (u, "u", False, False), (v, "v", False, False),
                  (d10, "d10", True, False), (d50, "d50", True, False), (d90, "d90", True, False),
                  (roughness, "roughness", True, False), (skin_roughness, "skin_roughness", True, False),
                  (mud_fraction, "mud", False, True),
                  (concentration_kg_m3, "concentration", False, True), (salinity, "salinity", False, True),
                  (slope_x, "slope_x", False, False), (slope_y, "slope_y", False, False))
        fields = [_field(x, nc, n, positive=pos, nonnegative=nn) for x, n, pos, nn in values]
        out = np.empty((10, nc), dtype=np.float64); status = np.empty(nc, dtype=np.int32)
        self._capacity(nc, self.threads, self.mode, self.params, *fields, out, status)
        if np.any(status): raise ValueError(f"Nonphysical Q7a capacity cells: {np.flatnonzero(status).tolist()}")
        names = ("qx_kg_m_s", "qy_kg_m_s", "reference_concentration", "settling_diameter_m",
                 "settling_velocity_m_s", "stress_ratio", "transport_shear", "reference_height_m",
                 "magnitude_kg_m_s", "critical_shields")
        return {name: out[i] for i, name in enumerate(names)}


@dataclass(frozen=True)
class AcceptedBedformStateV102:
    accepted_update: int
    height_m: np.ndarray
    length_m: np.ndarray
    roughness_m: np.ndarray
    methods_fingerprint: str
    state_sha256: str


def make_accepted_state_v102(result: Mapping[str, np.ndarray], methods_fingerprint: str,
                             *, accepted_update=0) -> AcceptedBedformStateV102:
    arrays = tuple(np.ascontiguousarray(result[key], dtype=np.float64)
                   for key in ("height", "length", "total_roughness"))
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid accepted bedform arrays")
    digest = _array_digest(*arrays)
    for a in arrays: a.setflags(write=False)
    return AcceptedBedformStateV102(int(accepted_update), *arrays, methods_fingerprint, digest)


def accept_bedform_candidate_v102(parent: AcceptedBedformStateV102,
                                  candidate: Mapping[str, np.ndarray],
                                  methods_fingerprint: str) -> AcceptedBedformStateV102:
    if methods_fingerprint != parent.methods_fingerprint:
        raise ValueError("Bedform method identity changed across an accepted update")
    if parent.accepted_update >= 20:
        raise ValueError("Q7a bounded state supports at most 20 accepted updates")
    return make_accepted_state_v102(candidate, methods_fingerprint,
                                    accepted_update=parent.accepted_update + 1)


def bedform_checkpoint_v102(state: AcceptedBedformStateV102) -> dict:
    return {"schema": 102, "accepted_update": state.accepted_update,
            "methods_fingerprint": state.methods_fingerprint, "state_sha256": state.state_sha256,
            "height_m": state.height_m.tolist(), "length_m": state.length_m.tolist(),
            "roughness_m": state.roughness_m.tolist()}


def restore_bedform_checkpoint_v102(document: Mapping) -> AcceptedBedformStateV102:
    if set(document) != {"schema", "accepted_update", "methods_fingerprint", "state_sha256",
                         "height_m", "length_m", "roughness_m"} or document["schema"] != 102:
        raise ValueError("Unknown Q7a bedform checkpoint schema")
    state = make_accepted_state_v102({"height": document["height_m"], "length": document["length_m"],
                                      "total_roughness": document["roughness_m"]},
                                     str(document["methods_fingerprint"]),
                                     accepted_update=int(document["accepted_update"]))
    if state.state_sha256 != document["state_sha256"]:
        raise ValueError("Q7a bedform checkpoint digest mismatch")
    return state


__all__ = ["StaticMethodsV102", "ResolvedStaticMethodsV102", "StaticParametersV102",
           "NativeStaticRuntimeV102", "AcceptedBedformStateV102", "resolve_static_methods_v102",
           "make_accepted_state_v102", "accept_bedform_candidate_v102",
           "bedform_checkpoint_v102", "restore_bedform_checkpoint_v102"]
