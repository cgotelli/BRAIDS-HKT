"""Recursive authoritative Q5 surface feedback for consecutive Q3/Q4 macros.

This module does not add a transport, exchange, Exner, or hydraulic equation.
It authenticates the evolving top-layer composition owned by Q5b, derives the
FLOW mixture characteristics and hiding/exposure fields once, binds that state
to the live Q3/Q4 provider, and requires every enabled consumer to use the same
state before the accepted combined Q3+Q4 increment is projected back into the
columns.  The next macro must bind the newly accepted surface; a stale binding
is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .bed_columns_v101 import array_sha, host
from .coupled_amr_column_projection_v113 import (
    CoupledAMRColumnProjectionV113,
)
from .dependent_layers_v103 import (
    Q3_SURFACE_CONSUMERS,
    Q4_SURFACE_CONSUMERS,
)
from .grain_distributions_v101 import (
    GrainDistributionV101,
    characteristics_v101,
    prepare_grain_distributions_v101,
)


SCOPE = "braids-q3-q4-q5-recursive-surface-feedback-v114"
ALL_SURFACE_CONSUMERS = Q3_SURFACE_CONSUMERS + Q4_SURFACE_CONSUMERS
SUPPORTED_HIDING_MODES = frozenset((0, 2, 3, 4, 5))
ROOT = Path(__file__).resolve().parents[2]
NATIVE_HEADER = ROOT / "solver/wave/braids_recursive_surface_feedback_v114.h"
NATIVE_SOURCE = NATIVE_HEADER.with_suffix(".cpp")


class RecursiveSurfaceFeedbackV114Error(ValueError):
    """The recursive surface identity, clock, or ownership is inconsistent."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _readonly(value):
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _stock_from_provider(provider):
    value = getattr(provider, "authoritative_bed_mass_kg_m2", None)
    if value is None:
        state = getattr(provider, "state", None)
        value = getattr(state, "authoritative_bed_mass_kg_m2", None)
        if value is None and state is not None:
            value = getattr(getattr(state, "core", None),
                            "authoritative_bed_mass_kg_m2", None)
    if value is None:
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 provider has no authoritative bed stock")
    return np.ascontiguousarray(host(value), dtype=np.float64)


def _provider_clock(provider):
    try:
        return int(provider.accepted_updates), float(provider.time_s)
    except (AttributeError, TypeError, ValueError) as error:
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 provider has no accepted bed clock") from error


@dataclass(frozen=True)
class AuthoritativeSurfaceStateV114:
    backend: str
    column_state_sha256: str
    total_bed_mass_sha256: str
    accepted_updates: int
    time_s: float
    total_bed_mass_kg_m2: object
    active_fraction: object
    hiding_fraction: object
    hiding_factor: object
    mean_diameter_m: object
    geometric_mean_diameter_m: object
    geometric_standard_deviation: object
    d10_m: object
    d50_m: object
    d90_m: object
    array_sha256: str
    scope: str
    state_sha256: str


def validate_surface_state_v114(value: AuthoritativeSurfaceStateV114):
    if not isinstance(value, AuthoritativeSurfaceStateV114):
        raise RecursiveSurfaceFeedbackV114Error(
            "Versioned authoritative surface state required")
    total_bed = np.ascontiguousarray(
        host(value.total_bed_mass_kg_m2), dtype=np.float64)
    arrays = (
        value.active_fraction, value.hiding_fraction, value.hiding_factor,
        value.mean_diameter_m, value.geometric_mean_diameter_m,
        value.geometric_standard_deviation, value.d10_m, value.d50_m,
        value.d90_m,
    )
    observed = array_sha(*(host(item) for item in arrays))
    payload = {
        "backend": value.backend,
        "column_state_sha256": value.column_state_sha256,
        "total_bed_mass_sha256": value.total_bed_mass_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "array_sha256": observed,
        "scope": value.scope,
    }
    if (value.scope != SCOPE or value.accepted_updates < 0 or
            not np.isfinite(value.time_s) or value.time_s < 0.0 or
            array_sha(total_bed) != value.total_bed_mass_sha256 or
            observed != value.array_sha256 or
            _digest(payload) != value.state_sha256):
        raise RecursiveSurfaceFeedbackV114Error(
            "Authoritative surface state was mutated or has invalid identity")
    active = np.asarray(host(value.active_fraction), dtype=np.float64)
    hiding = np.asarray(host(value.hiding_fraction), dtype=np.float64)
    factors = np.asarray(host(value.hiding_factor), dtype=np.float64)
    if (active.ndim != 2 or hiding.shape != active.shape or
            factors.shape != active.shape or
            not all(np.isfinite(np.asarray(host(item))).all()
                    for item in arrays) or np.any(active < 0.0) or
            np.any(hiding < 0.0) or np.any(factors <= 0.0)):
        raise RecursiveSurfaceFeedbackV114Error(
            "Authoritative surface arrays are invalid")
    return value


def bind_authoritative_surface_v114(provider, surface):
    """Bind one accepted Q5 surface to a Q3/Q4 provider without fallback."""
    validate_surface_state_v114(surface)
    if any(getattr(provider, name, None) is not None
           for name in ("_open", "_prepared", "_closed", "_armed")):
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 surface binding requires a closed provider transaction")
    updates, time_s = _provider_clock(provider)
    if updates != surface.accepted_updates or time_s != surface.time_s:
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 provider and Q5 surface clocks differ")
    if not np.allclose(
            _stock_from_provider(provider),
            np.asarray(host(surface.total_bed_mass_kg_m2)),
            rtol=0.0, atol=1.0e-12):
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 provider and Q5 authoritative bed stocks differ")
    backend = str(getattr(provider, "backend", ""))
    if (("cuda" in backend or "gpu" in backend) !=
            ("cuda" in surface.backend or "gpu" in surface.backend)):
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3 provider and Q5 surface backends differ")
    provider._braids_authoritative_surface_v114 = surface
    return surface


def provider_surface_state_v114(provider):
    """Return a live binding, rejecting use after either clock advances."""
    value = getattr(provider, "_braids_authoritative_surface_v114", None)
    if value is None:
        return None
    validate_surface_state_v114(value)
    updates, time_s = _provider_clock(provider)
    if updates != value.accepted_updates or time_s != value.time_s:
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3/Q4 attempted to reuse a stale Q5 surface")
    if not np.allclose(
            _stock_from_provider(provider),
            np.asarray(host(value.total_bed_mass_kg_m2)),
            rtol=0.0, atol=1.0e-12):
        raise RecursiveSurfaceFeedbackV114Error(
            "Q3/Q4 surface binding no longer matches the accepted bed")
    return value


def provider_surface_fraction_v114(provider):
    value = provider_surface_state_v114(provider)
    return (value.active_fraction if value is not None
            else provider.state.active_fraction)


def provider_surface_slope_fields_v114(provider, topology, xp=np):
    """Return FLOW cell hiding and class-face mean diameter fields."""
    value = provider_surface_state_v114(provider)
    ng = int(provider.outer_execution.fraction_count)
    nf = int(topology.nface)
    if value is None:
        return (xp.ones((ng, topology.ncell), dtype=xp.float64),
                xp.ones((ng, nf), dtype=xp.float64))
    hiding = value.hiding_factor
    mean = value.mean_diameter_m
    owner = topology.owner
    neighbour = topology.neighbour
    safe = xp.clip(neighbour, 0, topology.ncell - 1)
    internal = neighbour >= 0
    owner_mean = mean[owner]
    neighbour_mean = mean[safe]
    face_mean = xp.where(
        internal, xp.sqrt(xp.maximum(0.0, owner_mean*neighbour_mean)),
        owner_mean)
    return hiding, xp.ascontiguousarray(
        xp.broadcast_to(face_mean[None, :], (ng, nf)))


@lru_cache(maxsize=1)
def native_surface_hiding_runtime_v114():
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("Recursive Q5 surface feedback requires g++/OpenMP")
    directory = Path(tempfile.mkdtemp(prefix="braids_surface_v114_"))
    library = directory / ("surface_v114.dll" if os.name == "nt"
                           else "surface_v114.so")
    flags = ["-O3", "-fopenmp", "-shared", "-ffp-contract=off",
             "-std=c++11"]
    if os.name == "nt":
        flags += ["-static-libstdc++", "-static-libgcc"]
    build = subprocess.run(
        [compiler, *flags, str(NATIVE_SOURCE), "-o", str(library)],
        cwd=directory, capture_output=True, text=True, timeout=120)
    if build.returncode:
        raise RuntimeError(build.stdout + build.stderr)
    dependency = (os.add_dll_directory(str(Path(compiler).parent))
                  if os.name == "nt" else None)
    dll = ctypes.CDLL(str(library))
    function = dll.braids_surface_hiding_v114
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int] * 4 + [real, real] + \
        [ctypes.c_double] * 2 + [real, real, integer]
    function.restype = None
    receipt = {
        "source_sha256": hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest(),
        "equations_sha256": hashlib.sha256(NATIVE_HEADER.read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "flags": flags,
    }
    return function, dll, dependency, receipt


def _cpu_hiding(active_volume, diameter, mode, parker_alpha, wu_exponent,
                threads):
    ng, nc = active_volume.shape
    factors = np.empty((ng, nc), dtype=np.float64)
    mean = np.empty(nc, dtype=np.float64)
    status = np.empty(nc, dtype=np.int32)
    function = native_surface_hiding_runtime_v114()[0]
    function(nc, ng, int(threads), int(mode),
             np.ascontiguousarray(active_volume, dtype=np.float64),
             np.ascontiguousarray(diameter, dtype=np.float64),
             float(parker_alpha), float(wu_exponent), factors, mean, status)
    if np.any(status != 0):
        raise RecursiveSurfaceFeedbackV114Error(
            "Native hiding/exposure rejected an invalid Q5 surface")
    return factors, mean


class SurfaceFeedbackOwnerV114:
    """Derive one immutable surface identity from one committed Q5b owner."""

    def __init__(self, columns, *, hiding_mode=0, parker_alpha=0.6,
                 wu_exponent=-0.6, distributions=None):
        if hiding_mode not in SUPPORTED_HIDING_MODES:
            raise RecursiveSurfaceFeedbackV114Error(
                "Hiding/exposure mode must be 0, 2, 3, 4 or 5")
        if not np.isfinite((parker_alpha, wu_exponent)).all():
            raise RecursiveSurfaceFeedbackV114Error(
                "Hiding/exposure controls must be finite")
        self.columns = columns
        methods = columns.runtime.methods
        diameter = tuple(float(x) for x in methods.q5a.diameter_m)
        if distributions is None:
            distributions = tuple(
                GrainDistributionV101.legacy_d50(value)
                for value in diameter)
        if len(distributions) != len(diameter):
            raise RecursiveSurfaceFeedbackV114Error(
                "One grain distribution per fraction is required")
        self.hiding_mode = int(hiding_mode)
        self.parker_alpha = float(parker_alpha)
        self.wu_exponent = float(wu_exponent)
        self.distributions = tuple(distributions)
        self.prepared = prepare_grain_distributions_v101(
            self.distributions, columns.runtime.nc)
        self._cuda_hiding = None
        self._native_receipt = (native_surface_hiding_runtime_v114()[3]
                                if columns.xp is np else None)

    @property
    def identity(self):
        return _digest({
            "scope": SCOPE,
            "runtime": self.columns.runtime.identity,
            "hiding_mode": self.hiding_mode,
            "parker_alpha": self.parker_alpha,
            "wu_exponent": self.wu_exponent,
            "distributions": [item.fingerprint for item in self.distributions],
            "native_runtime": self._native_receipt,
        })

    def snapshot(self):
        columns = self.columns
        columns.validate()
        if columns.pending is not None:
            raise RecursiveSurfaceFeedbackV114Error(
                "Surface snapshot requires a committed Q5b column")
        xp = columns.xp
        rho = xp.asarray(columns.runtime.rho)[:, None]
        active_volume = xp.ascontiguousarray(columns.mass[0] / rho)
        hiding_volume = active_volume
        methods = columns.runtime.methods
        if methods.hiding_scope == "active_coarse":
            if not methods.coarse_layer or columns.runtime.nl < 2:
                raise RecursiveSurfaceFeedbackV114Error(
                    "Active+coarse hiding requires a coarse layer")
            hiding_volume = xp.ascontiguousarray(
                active_volume + columns.mass[1] / rho)

        def normalized(volume):
            total = volume.sum(axis=0)
            if bool((total <= 0.0).any()):
                raise RecursiveSurfaceFeedbackV114Error(
                    "Authoritative Q5 surface is depleted")
            return xp.ascontiguousarray(volume / total[None, :])

        active_fraction = normalized(active_volume)
        hiding_fraction = normalized(hiding_volume)
        diameter = np.ascontiguousarray(methods.q5a.diameter_m,
                                        dtype=np.float64)
        if xp is np:
            factors, hiding_mean = _cpu_hiding(
                np.asarray(hiding_volume), diameter, self.hiding_mode,
                self.parker_alpha, self.wu_exponent,
                columns.runtime.threads)
            factors = _readonly(factors)
            hiding_mean = _readonly(hiding_mean)
        else:
            if self._cuda_hiding is None:
                from braids_gpu.hiding_exposure_runtime import HidingExposureCUDA
                self._cuda_hiding = HidingExposureCUDA()
            result = self._cuda_hiding.evaluate(
                hiding_volume, columns.runtime.diameter,
                mode=self.hiding_mode, parker_alpha=self.parker_alpha,
                wu_exponent=self.wu_exponent)
            factors = result.factors
            hiding_mean = result.mean_diameter
        characteristic = characteristics_v101(
            columns.runtime.base, active_fraction, self.prepared,
            quantiles=(0.1, 0.5, 0.9))
        values = (
            active_fraction, hiding_fraction, factors, hiding_mean,
            characteristic[0], characteristic[2], characteristic[3],
            characteristic[4], characteristic[5],
        )
        if xp is np:
            values = tuple(_readonly(item) for item in values)
        arrays_sha = array_sha(*(host(item) for item in values))
        total_bed = columns.total
        if xp is np:
            total_bed = _readonly(total_bed)
        total_sha = array_sha(host(total_bed))
        payload = {
            "backend": columns.runtime.backend,
            "column_state_sha256": columns.state_sha256,
            "total_bed_mass_sha256": total_sha,
            "accepted_updates": columns.updates,
            "time_s": columns.time_s,
            "array_sha256": arrays_sha,
            "scope": SCOPE,
        }
        state = AuthoritativeSurfaceStateV114(
            columns.runtime.backend, columns.state_sha256, total_sha,
            columns.updates, columns.time_s, total_bed, *values, arrays_sha,
            SCOPE,
            _digest(payload))
        return validate_surface_state_v114(state)


@dataclass(frozen=True)
class RecursiveSurfaceFeedbackReceiptV114:
    backend: str
    macro_dt_s: float
    active_consumers: tuple[str, ...]
    coarse_before_surface_sha256: str
    fine_before_surface_sha256: str
    coarse_after_surface_sha256: str
    fine_after_surface_sha256: str
    coarse_surface_changed: bool
    fine_surface_changed: bool
    projection_receipt_sha256: str
    maximum_column_projection_error_kg_m2: float
    maximum_bed_feedback_error_m: float
    maximum_class_budget_residual_kg: float
    accepted_surface_owner_count: int
    accepted_projection_owner_count: int
    automatic_cpu_fallback: bool
    scope: str
    receipt_sha256: str


class CoupledAMRRecursiveSurfaceFeedbackV114:
    """Two-level Q5 surface -> Q3/Q4 -> M14 Q5 feedback transaction."""

    def __init__(self, coarse_columns, fine_columns, *, hiding_mode=0,
                 parker_alpha=0.6, wu_exponent=-0.6, distributions=None,
                 q3_providers=None):
        self.levels = {
            "coarse": SurfaceFeedbackOwnerV114(
                coarse_columns, hiding_mode=hiding_mode,
                parker_alpha=parker_alpha, wu_exponent=wu_exponent,
                distributions=distributions),
            "fine": SurfaceFeedbackOwnerV114(
                fine_columns, hiding_mode=hiding_mode,
                parker_alpha=parker_alpha, wu_exponent=wu_exponent,
                distributions=distributions),
        }
        if q3_providers is not None and (
                not isinstance(q3_providers, Mapping) or
                set(q3_providers) != {"coarse", "fine"}):
            raise RecursiveSurfaceFeedbackV114Error(
                "Q3 providers require explicit coarse and fine owners")
        self.q3_providers = (None if q3_providers is None
                             else dict(q3_providers))
        self.pending = None

    @property
    def columns(self):
        return tuple(self.levels[name].columns
                     for name in ("coarse", "fine"))

    @property
    def backend(self):
        values = {owner.columns.runtime.backend
                  for owner in self.levels.values()}
        if len(values) != 1:
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive AMR surface levels use different backends")
        return next(iter(values))

    def begin(self, parent, *, active_consumers=ALL_SURFACE_CONSUMERS,
              cancelled=False):
        if self.pending is not None:
            raise RecursiveSurfaceFeedbackV114Error(
                "A recursive surface macro is already open")
        if bool(cancelled() if callable(cancelled) else cancelled):
            raise InterruptedError("Recursive surface macro cancelled")
        consumers = tuple(active_consumers)
        if (len(consumers) != len(set(consumers)) or
                any(name not in ALL_SURFACE_CONSUMERS for name in consumers)):
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface consumers are unknown or duplicated")
        surfaces = {name: owner.snapshot()
                    for name, owner in self.levels.items()}
        if any(surface.time_s != float(parent.time_s)
               for surface in surfaces.values()):
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface and joined parent clocks differ")
        if self.q3_providers is not None:
            for name in ("coarse", "fine"):
                bind_authoritative_surface_v114(
                    self.q3_providers[name], surfaces[name])
        self.pending = {
            "parent": parent,
            "surfaces": surfaces,
            "active_consumers": consumers,
            "counts": {name: 0 for name in consumers},
            "projection": None,
        }
        return surfaces

    def consume(self, name):
        if self.pending is None:
            raise RecursiveSurfaceFeedbackV114Error(
                "No recursive surface macro is open")
        if name not in self.pending["counts"]:
            raise RecursiveSurfaceFeedbackV114Error(
                f"Surface consumer {name!r} is not active")
        if self.pending["counts"][name] != 0:
            raise RecursiveSurfaceFeedbackV114Error(
                f"Surface consumer {name!r} was applied more than once")
        self.pending["counts"][name] = 1
        return self.pending["surfaces"]

    def prepare(self, candidate, *, resident_deltas=None, column_inputs=None,
                cancelled=False):
        if self.pending is None or self.pending["projection"] is not None:
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface macro cannot be prepared")
        missing = [name for name, count in self.pending["counts"].items()
                   if count != 1]
        if missing:
            raise RecursiveSurfaceFeedbackV114Error(
                "Surface consumers were not applied exactly once: " +
                ", ".join(missing))
        if column_inputs is None:
            column_inputs = {"coarse": {}, "fine": {}}
        if (not isinstance(column_inputs, Mapping) or
                set(column_inputs) != {"coarse", "fine"} or
                any(not isinstance(value, Mapping)
                    for value in column_inputs.values())):
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive column inputs require coarse and fine mappings")
        if any("hiding_factor" in value for value in column_inputs.values()):
            raise RecursiveSurfaceFeedbackV114Error(
                "The authoritative surface owner controls hiding_factor")
        inputs = {
            name: {"hiding_factor": surface.hiding_factor}
            for name, surface in self.pending["surfaces"].items()}
        for name in inputs:
            inputs[name].update(column_inputs[name])
        projection = CoupledAMRColumnProjectionV113(
            self.columns[0], self.columns[1], self.pending["parent"],
            candidate, resident_deltas=resident_deltas,
            column_inputs=inputs)
        projection.begin(cancelled=cancelled)
        self.pending["projection"] = projection
        self.pending["candidate"] = candidate
        return projection.receipt

    def commit(self, *, cancelled=False, after_coarse_publish=None):
        if self.pending is None or self.pending["projection"] is None:
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface macro has no prepared projection")
        pending = self.pending
        projection = pending["projection"]
        try:
            projected = projection.commit(
                cancelled=cancelled,
                after_coarse_publish=after_coarse_publish)
            after = {name: owner.snapshot()
                     for name, owner in self.levels.items()}
            before = pending["surfaces"]
            payload = {
                "backend": self.backend,
                "macro_dt_s": projected.macro_dt_s,
                "active_consumers": pending["active_consumers"],
                "coarse_before_surface_sha256":
                    before["coarse"].state_sha256,
                "fine_before_surface_sha256": before["fine"].state_sha256,
                "coarse_after_surface_sha256": after["coarse"].state_sha256,
                "fine_after_surface_sha256": after["fine"].state_sha256,
                "coarse_surface_changed": (
                    before["coarse"].array_sha256 !=
                    after["coarse"].array_sha256),
                "fine_surface_changed": (
                    before["fine"].array_sha256 !=
                    after["fine"].array_sha256),
                "projection_receipt_sha256": projected.receipt_sha256,
                "maximum_column_projection_error_kg_m2": max(
                    projected.coarse_projection_max_abs_kg_m2,
                    projected.fine_projection_max_abs_kg_m2),
                "maximum_bed_feedback_error_m": max(
                    projected.coarse_bed_feedback_max_abs_m,
                    projected.fine_bed_feedback_max_abs_m),
                "maximum_class_budget_residual_kg": max(
                    *(abs(x) for x in
                      projected.coarse_class_budget_residual_kg),
                    *(abs(x) for x in
                      projected.fine_class_budget_residual_kg)),
                "accepted_surface_owner_count": 1,
                "accepted_projection_owner_count": 1,
                "automatic_cpu_fallback": False,
                "scope": SCOPE,
            }
            receipt = RecursiveSurfaceFeedbackReceiptV114(
                **payload, receipt_sha256=_digest(payload))
            self.pending = None
            return receipt, after
        except BaseException:
            self.pending = None
            raise

    def rollback(self):
        if self.pending is None:
            raise RecursiveSurfaceFeedbackV114Error(
                "No recursive surface macro is open")
        projection = self.pending.get("projection")
        if projection is not None and projection.pending is not None:
            projection.rollback()
        self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface checkpoint requires a committed macro")
        raw = {
            "scope": SCOPE,
            "backend": self.backend,
            "owners": {name: owner.identity
                       for name, owner in self.levels.items()},
            "columns": {name: owner.columns.checkpoint()
                        for name, owner in self.levels.items()},
        }
        return {**raw, "checkpoint_sha256": _digest(raw)}

    def restore(self, value):
        if self.pending is not None:
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface restore requires a committed macro")
        raw = {key: item for key, item in value.items()
               if key != "checkpoint_sha256"}
        expected = {name: owner.identity for name, owner in self.levels.items()}
        if (_digest(raw) != value.get("checkpoint_sha256") or
                value.get("scope") != SCOPE or
                value.get("backend") != self.backend or
                value.get("owners") != expected):
            raise RecursiveSurfaceFeedbackV114Error(
                "Recursive surface checkpoint identity differs")
        old = {name: owner.columns.checkpoint()
               for name, owner in self.levels.items()}
        try:
            for name, owner in self.levels.items():
                owner.columns.restore(value["columns"][name])
        except BaseException:
            for name, owner in self.levels.items():
                owner.columns.restore(old[name])
            raise
        return {name: owner.snapshot() for name, owner in self.levels.items()}


__all__ = [
    "ALL_SURFACE_CONSUMERS", "SCOPE", "SUPPORTED_HIDING_MODES",
    "AuthoritativeSurfaceStateV114",
    "CoupledAMRRecursiveSurfaceFeedbackV114",
    "RecursiveSurfaceFeedbackReceiptV114",
    "RecursiveSurfaceFeedbackV114Error", "SurfaceFeedbackOwnerV114",
    "bind_authoritative_surface_v114", "provider_surface_fraction_v114",
    "native_surface_hiding_runtime_v114",
    "provider_surface_slope_fields_v114", "provider_surface_state_v114",
    "validate_surface_state_v114",
]
