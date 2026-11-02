"""Real Q3/v67 producer binding for the bounded Q4/v81 lifecycle.

V81 deliberately accepted caller-provided raw rows.  This additive owner
removes that seam for the three split current-only methods: its public trial
API accepts hydraulic/roughness inputs, invokes the compiled v67 Fortran
adapters, seals their five raw channels, and immediately starts the v81
lifecycle.  Raw producer rows are never a public input.

The scope is intentionally bounded to the fixed-grid, wave-free Bijker,
TraFrm 19 and TraFrm 20 ``TRA_COMBINE`` lanes, disabled
``MaximumWaterdepth``, Manning roughness and four accepted updates.  The CUDA
peer lives in :mod:`braids_gpu.q3_producer_lifecycle_binding_v84`.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from .direct_vector_family_live_v67 import (
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
    load_direct_vector_family_execution_v67,
    verify_direct_vector_family_sources_v67,
)
from .route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
    RouteLifecycleExecutionV81,
    RouteLifecycleStateV81,
    RouteLifecycleTrialV81,
    RouteLifecycleV81Error,
    begin_route_lifecycle_cpu_trial_v81,
    cancel_route_lifecycle_trial_v81,
    commit_route_lifecycle_trial_v81,
    load_route_lifecycle_checkpoint_v81,
    rollback_route_lifecycle_trial_v81,
    route_lifecycle_checkpoint_v81,
    validate_route_lifecycle_state_v81,
)
from .suspended_route_live_v74 import (
    RouteHandoffV74,
    compose_route_handoff_v74,
    validate_route_handoff_v74,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-real-q3-producer-lifecycle-binding-v84-v1"
MAX_ACCEPTED_UPDATES = 4
PROFILE_METHODS = (
    METHOD_BIJKER_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
)
PUBLIC_METHODS = (
    "capacity.bijker",
    "capacity.van_thiel_van_rijn",
    "capacity.xbeach_soulsby_van_rijn",
)
RAW_ROWS = (
    "bed_x_kg_m_s", "bed_y_kg_m_s", "suspended_x_kg_m_s",
    "suspended_y_kg_m_s", "equilibrium_concentration_kg_m3",
)
_EXPECTED_PARAMETERS = (
    {
        "CalBs": 5.0, "CalBd": 4.0, "CritCs": 0.4, "CritCd": 0.05,
        "RouKs": -999.0, "WSettle": -999.0,
        "Porosity": 0.4, "TWave": 5.0,
        "FacA": 0.0, "FacU": 0.0, "EpsSL": 0,
    },
    {
        "facua": 0.0, "facAs": 0.2, "facSk": 0.15,
        "waveform": 2, "sws": 1, "lws": 1, "dilatancy": 0,
        "rheeA": 0.75, "pormax": 0.5, "bedslpini": 0,
        "smax": -1.0, "reposeangle": 30.0, "cmax": 0.1,
        "alfad50": 0.0,
    },
    {
        "facua": 0.0, "facAs": 0.2, "facSk": 0.15,
        "waveform": 2, "sws": 1, "lws": 1, "dilatancy": 0,
        "rheeA": 0.75, "pormax": 0.5, "bedslpini": 0,
        "smax": -1.0, "reposeangle": 30.0, "cmax": 0.1,
        "z0": 0.006, "alfad50": 0.0,
    },
)


class Q3ProducerLifecycleV84Error(ValueError):
    """The real-producer binding, receipt or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise Q3ProducerLifecycleV84Error(
            "v84 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*arrays: np.ndarray) -> str:
    result = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise Q3ProducerLifecycleV84Error(
            f"v84 {name} must be finite with shape {shape}")
    if positive and np.any(result <= 0.0):
        raise Q3ProducerLifecycleV84Error(f"v84 {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise Q3ProducerLifecycleV84Error(
            f"v84 {name} must be nonnegative")
    return np.ascontiguousarray(result)


def _parameter_values(row: Mapping[str, object]) -> dict[str, object]:
    native = row["parameter_identity"]["native_identity"]
    if native["schema_id"] == "direct_current_parameters_v66":
        values = native["document"]["parameters"]
    else:  # pragma: no cover - exact-method validation rejects this first
        values = native["parameters"]
    return {str(item["keyword"]): item["value"] for item in values
            if item["keyword"] != "[dummy]"}


def validate_binding_executions_v84(
    producer_execution: Mapping[str, object],
    lifecycle_execution: RouteLifecycleExecutionV81, *, backend: str,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Bind exact v67 producer bytes and identity to the exact v81 owner."""
    if backend not in ("cpu", "implicit_fwave_midpoint_gpu"):
        raise Q3ProducerLifecycleV84Error("v84 backend must be explicit")
    try:
        producer = load_direct_vector_family_execution_v67(producer_execution)
        validate_route_lifecycle_state_v81  # keep imported validator explicit
        verified = verify_direct_vector_family_sources_v67(
            producer, repository_root=root)
    except (TypeError, ValueError) as error:
        raise Q3ProducerLifecycleV84Error(
            "v84 v67 execution or source receipt is invalid") from error
    physical = producer["physical_profile"]
    rows = physical["fraction_methods"]
    if (producer["backend"] != backend or
            tuple(row["method_id"] for row in rows) != PROFILE_METHODS or
            tuple(physical["routes"]) != ("tra_combine",)*3 or
            physical["roughness_method_id"] != "roughness.manning" or
            producer["integrator"] !=
            ("implicit_fwave_midpoint" if backend.endswith("_gpu")
             else "native_openmp_reference") or
            producer["automatic_backend_fallback"] is not False):
        raise Q3ProducerLifecycleV84Error(
            "v84 requires the exact three-lane current-only combine profile")
    for index, expected in enumerate(_EXPECTED_PARAMETERS):
        actual = _parameter_values(rows[index])
        # The C ABI wrapper is intentionally bounded, so every expanded
        # source default is execution identity rather than an ignored option.
        if actual != expected:
            raise Q3ProducerLifecycleV84Error(
                f"v84 native parameter profile differs for lane {index}")
    if not isinstance(lifecycle_execution, RouteLifecycleExecutionV81):
        raise Q3ProducerLifecycleV84Error("v84 lifecycle execution differs")
    expected_role = ("isolated_cpu_oracle" if backend == "cpu" else
                     "actual_cuda")
    resolved_routes = tuple(lifecycle_execution.route_execution["routes"])
    if (lifecycle_execution.role != expected_role or
            tuple(lifecycle_execution.route_execution["method_ids"])
            != PUBLIC_METHODS or
            any(route not in (ROUTE_OFF, ROUTE_TOTAL, ROUTE_COMBINE)
                for route in resolved_routes) or
            lifecycle_execution.q3_receipt["producer_execution_sha256"]
            != producer["fingerprint"]):
        raise Q3ProducerLifecycleV84Error(
            "v84 v67 producer and v81 route/lifecycle identities differ")
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "backend": backend,
        "producer_execution_fingerprint": producer["fingerprint"],
        "producer_source_receipt_fingerprint": verified["fingerprint"],
        "lifecycle_execution_fingerprint": lifecycle_execution.fingerprint,
        "methods": list(PUBLIC_METHODS),
        "routes": list(resolved_routes),
        "maximum_waterdepth": False,
        "roughness": "manning",
        "wave_active": False,
        "external_secondary_rotation": False,
        "surface_fraction_owner": "v81_v78_once",
        "post_producer_density_conversion_count": 0,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_backend_fallback": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


@dataclass(frozen=True)
class Q3AcceptedProducerRowsV84:
    execution_fingerprint: str
    backend: str
    raw_rows: np.ndarray
    wet_mask: np.ndarray
    producer_receipt_sha256: str
    source_receipt_sha256: str
    surface_fraction_application_count: int
    post_producer_density_conversion_count: int
    producer_sha256: str


def _producer_token(value: Q3AcceptedProducerRowsV84) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "backend": value.backend,
        "producer_receipt_sha256": value.producer_receipt_sha256,
        "source_receipt_sha256": value.source_receipt_sha256,
        "surface_fraction_application_count":
            value.surface_fraction_application_count,
        "post_producer_density_conversion_count":
            value.post_producer_density_conversion_count,
        "arrays_sha256": _digest(value.raw_rows, value.wet_mask),
    })


def validate_producer_rows_v84(
    value: Q3AcceptedProducerRowsV84, execution_fingerprint: str,
) -> Q3AcceptedProducerRowsV84:
    if (not isinstance(value, Q3AcceptedProducerRowsV84) or
            value.execution_fingerprint != execution_fingerprint or
            value.backend not in ("cpu", "actual_cuda") or
            value.raw_rows.ndim != 3 or value.raw_rows.shape[:2] != (5, 3) or
            value.wet_mask.shape != (value.raw_rows.shape[2],) or
            value.raw_rows.dtype != np.float64 or
            value.wet_mask.dtype != np.int32 or
            value.surface_fraction_application_count != 0 or
            value.post_producer_density_conversion_count != 0 or
            not np.isfinite(value.raw_rows).all() or
            np.any(value.raw_rows[:, :, value.wet_mask == 0] != 0.0) or
            value.producer_sha256 != _producer_token(value)):
        raise Q3ProducerLifecycleV84Error(
            "v84 producer rows are stale, mutated or not accepted v67 output")
    return value


class NativeQ3SplitProducerV84:
    """Load the additive C ABI and execute the real v67 Fortran adapters."""

    def __init__(self, *, producer_execution: Mapping[str, object],
                 lifecycle_execution: RouteLifecycleExecutionV81,
                 library_path: str | Path, root: str | Path | None = None):
        self.root = Path(root or Path(__file__).resolve().parents[2])
        self.execution = validate_binding_executions_v84(
            producer_execution, lifecycle_execution, backend="cpu",
            root=self.root)
        self.producer_execution = load_direct_vector_family_execution_v67(
            producer_execution)
        self.sediment_threshold_m = (
            lifecycle_execution.lifecycle_execution.base_execution.profile
            .sediment_threshold_m)
        path = Path(library_path).resolve()
        if not path.is_file():
            raise Q3ProducerLifecycleV84Error("v84 native producer library is absent")
        self.library_path = path
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        source = self.root/"solver/wave/braids_q3_split_producer_v84_module.f90"
        if not source.is_file():
            raise Q3ProducerLifecycleV84Error("v84 native wrapper source is absent")
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self._dll_directory_handles = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            candidates = [path.parent]
            compiler = shutil.which("gfortran")
            if compiler is not None:
                candidates.append(Path(compiler).resolve().parent)
            for candidate in candidates:
                try:
                    self._dll_directory_handles.append(
                        os.add_dll_directory(str(candidate)))
                except OSError:
                    pass
        try:
            self.library = ctypes.CDLL(str(path))
            self.function = self.library.braids_q3_split_produce_v84
        except (OSError, AttributeError) as error:
            raise Q3ProducerLifecycleV84Error(
                "v84 native producer library cannot be loaded") from error
        pointer_d = np.ctypeslib.ndpointer(dtype=np.float64, flags="ALIGNED")
        pointer_i = np.ctypeslib.ndpointer(dtype=np.int32, flags="ALIGNED")
        self.function.argtypes = [
            ctypes.c_int, pointer_d, pointer_d, pointer_d, pointer_d,
            pointer_d, pointer_d, pointer_d, pointer_d, pointer_d,
            ctypes.c_double, ctypes.c_double, pointer_d, pointer_i,
        ]
        self.function.restype = None

    def produce(self, *, depth_m: object, momentum_m2_s: object,
                roughness_parameter: object, kinematic_viscosity_m2_s: object,
                settling_velocity_m_s: object,
                dry_tolerance_m: float) -> Q3AcceptedProducerRowsV84:
        nc = int(np.asarray(depth_m).size)
        depth = _array(depth_m, (nc,), "producer depth", nonnegative=True)
        momentum = _array(momentum_m2_s, (2, nc), "producer momentum")
        roughness = _array(
            roughness_parameter, (nc,), "Manning roughness", positive=True)
        viscosity = _array(
            kinematic_viscosity_m2_s, (nc,), "kinematic viscosity",
            positive=True)
        settling = _array(
            settling_velocity_m_s, (3, nc), "settling velocity",
            nonnegative=True)
        if (isinstance(dry_tolerance_m, bool) or
                not isinstance(dry_tolerance_m, (int, float)) or
                not math.isfinite(float(dry_tolerance_m)) or
                float(dry_tolerance_m) < 0.0):
            raise Q3ProducerLifecycleV84Error("v84 dry tolerance differs")
        wet = depth > float(dry_tolerance_m)
        active = wet & (depth > self.sediment_threshold_m)
        safe_depth = np.where(active, depth, 1.0)
        hydraulic_u = np.where(active, momentum[0]/safe_depth, 0.0)
        hydraulic_v = np.where(active, momentum[1]/safe_depth, 0.0)
        hydraulic_u[np.abs(hydraulic_u) < 1.0e-12] = 0.0
        hydraulic_v[np.abs(hydraulic_v) < 1.0e-12] = 0.0
        speed = np.hypot(hydraulic_u, hydraulic_v)
        effective_u = speed*hydraulic_u/(speed+1.0e-6)
        effective_v = speed*hydraulic_v/(speed+1.0e-6)
        physical = self.producer_execution["physical_profile"]
        work_depth = np.maximum(
            safe_depth, float(physical["minimum_work_depth_m"]))
        chezy = np.ascontiguousarray(work_depth**(1.0/6.0)/roughness)
        d50 = np.asfortranarray(physical["d50_m"], dtype=np.float64)
        d90 = np.asfortranarray(physical["d90_m"], dtype=np.float64)
        rhos = np.asfortranarray(
            physical["solid_density_kg_m3"], dtype=np.float64)
        raw = np.empty((5, 3, nc), dtype=np.float64, order="F")
        status = np.empty((3, nc), dtype=np.int32, order="F")
        self.function(
            nc, np.asfortranarray(safe_depth),
            np.asfortranarray(effective_u), np.asfortranarray(effective_v),
            np.asfortranarray(chezy), np.asfortranarray(viscosity),
            np.asfortranarray(settling), d50, d90, rhos,
            float(physical["water_density_kg_m3"]),
            float(physical["gravity_m_s2"]), raw, status)
        raw[:, :, ~active] = 0.0
        if np.any(status[:, active] != 0):
            index = np.argwhere(status[:, active] != 0)[0].tolist()
            raise Q3ProducerLifecycleV84Error(
                f"v84 native v67 producer rejected wet lane/cell {index}")
        raw = np.ascontiguousarray(raw)
        wet_i = np.ascontiguousarray(active.astype(np.int32))
        raw.setflags(write=False); wet_i.setflags(write=False)
        receipt = _fingerprint({
            "v67_execution": self.producer_execution["fingerprint"],
            "native_library_sha256": self.library_sha256,
            "native_wrapper_sha256": self.source_sha256,
            "hydraulic_inputs_sha256": _digest(
                depth, momentum, roughness, viscosity, settling),
        })
        placeholder = Q3AcceptedProducerRowsV84(
            self.execution["fingerprint"], "cpu", raw, wet_i, receipt,
            self.execution["producer_source_receipt_fingerprint"], 0, 0, "")
        return Q3AcceptedProducerRowsV84(
            **{**placeholder.__dict__,
               "producer_sha256": _producer_token(placeholder)})


@dataclass(frozen=True)
class Q3BoundRouteLifecycleTrialV84:
    execution_fingerprint: str
    parent_state_sha256: str
    producer: Q3AcceptedProducerRowsV84
    lifecycle_trial: RouteLifecycleTrialV81
    trial_sha256: str

    @property
    def candidate(self) -> RouteLifecycleStateV81:
        return self.lifecycle_trial.candidate


@dataclass(frozen=True)
class Q3RouteHandoffReceiptV84:
    """Authenticated Q3 producer plus route split, before any scalar work."""

    execution_fingerprint: str
    backend: str
    producer: Q3AcceptedProducerRowsV84
    handoff: RouteHandoffV74
    accepted_endpoint_sha256: str
    active_fraction_sha256: str
    scalar_lifecycle_application_count: int
    receipt_sha256: str


def _route_handoff_token(value: Q3RouteHandoffReceiptV84) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":pure-route-handoff",
        "execution_fingerprint": value.execution_fingerprint,
        "backend": value.backend,
        "producer_sha256": value.producer.producer_sha256,
        "handoff_sha256": value.handoff.handoff_sha256,
        "accepted_endpoint_sha256": value.accepted_endpoint_sha256,
        "active_fraction_sha256": value.active_fraction_sha256,
        "scalar_lifecycle_application_count":
            value.scalar_lifecycle_application_count,
    })


def validate_route_handoff_receipt_v84(
    value: Q3RouteHandoffReceiptV84, *, execution_fingerprint: str,
    route_execution: Mapping[str, object], backend: str,
    accepted_endpoint_sha256: str | None = None,
    active_fraction_sha256: str | None = None,
) -> Q3RouteHandoffReceiptV84:
    if (not isinstance(value, Q3RouteHandoffReceiptV84) or
            value.execution_fingerprint != execution_fingerprint or
            value.backend != backend or
            (accepted_endpoint_sha256 is not None and
             value.accepted_endpoint_sha256 != accepted_endpoint_sha256) or
            (active_fraction_sha256 is not None and
             value.active_fraction_sha256 != active_fraction_sha256) or
            value.scalar_lifecycle_application_count != 0 or
            value.receipt_sha256 != _route_handoff_token(value)):
        raise Q3ProducerLifecycleV84Error(
            "v84 pure route handoff is stale or executed scalar lifecycle")
    validate_producer_rows_v84(value.producer, execution_fingerprint)
    validate_route_handoff_v74(value.handoff, execution=route_execution)
    return value


def _trial_token(value: Q3BoundRouteLifecycleTrialV84) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "producer_sha256": value.producer.producer_sha256,
        "lifecycle_trial_sha256": value.lifecycle_trial.trial_sha256,
    })


class NativeQ3BoundRouteLifecycleV84:
    """CPU transaction whose only raw rows come from the native v67 call."""

    def __init__(self, producer: NativeQ3SplitProducerV84,
                 lifecycle_execution: RouteLifecycleExecutionV81):
        self.producer = producer
        self.lifecycle_execution = lifecycle_execution
        self.execution = producer.execution
        if (self.execution["lifecycle_execution_fingerprint"] !=
                lifecycle_execution.fingerprint):
            raise Q3ProducerLifecycleV84Error(
                "v84 native producer and lifecycle differ")

    def begin_trial(self, parent: RouteLifecycleStateV81, *,
                    roughness_parameter, kinematic_viscosity_m2_s,
                    settling_velocity_m_s, cancelled: bool = False,
                    **lifecycle_inputs) -> Q3BoundRouteLifecycleTrialV84:
        validate_route_lifecycle_state_v81(parent, self.lifecycle_execution)
        if cancelled:
            raise Q3ProducerLifecycleV84Error(
                "v84 cancelled before producer execution")
        forbidden = {key for key in lifecycle_inputs if key.startswith("raw_")}
        if forbidden:
            raise Q3ProducerLifecycleV84Error(
                "v84 public trial API rejects caller-supplied raw rows")
        output = self.producer.produce(
            depth_m=lifecycle_inputs["depth_after_m"],
            momentum_m2_s=lifecycle_inputs["endpoint_momentum_m2_s"],
            roughness_parameter=roughness_parameter,
            kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
            settling_velocity_m_s=settling_velocity_m_s,
            dry_tolerance_m=(
                self.lifecycle_execution.lifecycle_execution
                .base_execution.dry_tolerance_m))
        validate_producer_rows_v84(output, self.execution["fingerprint"])
        raw = output.raw_rows
        try:
            inner = begin_route_lifecycle_cpu_trial_v81(
                self.lifecycle_execution, parent,
                raw_bed_x_kg_m_s=raw[0], raw_bed_y_kg_m_s=raw[1],
                raw_suspended_x_kg_m_s=raw[2],
                raw_suspended_y_kg_m_s=raw[3],
                raw_equilibrium_concentration_kg_m3=raw[4],
                **lifecycle_inputs)
        except (TypeError, ValueError) as error:
            raise Q3ProducerLifecycleV84Error(
                "v84 native output was rejected by v81") from error
        placeholder = Q3BoundRouteLifecycleTrialV84(
            self.execution["fingerprint"], parent.state_sha256,
            output, inner, "")
        return Q3BoundRouteLifecycleTrialV84(
            **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})

    def produce_route_handoff(
        self, *, depth_after_m, endpoint_momentum_m2_s,
        roughness_parameter, kinematic_viscosity_m2_s,
        settling_velocity_m_s, active_fraction,
    ) -> Q3RouteHandoffReceiptV84:
        """Run only Q3 constitutive production and v74 route composition.

        This is the Q4/v93 seam.  It deliberately does not create a v81/v78
        trial, so v92 remains the sole scalar/FCT/exchange lifecycle owner.
        """
        output = self.producer.produce(
            depth_m=depth_after_m,
            momentum_m2_s=endpoint_momentum_m2_s,
            roughness_parameter=roughness_parameter,
            kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
            settling_velocity_m_s=settling_velocity_m_s,
            dry_tolerance_m=(
                self.lifecycle_execution.lifecycle_execution
                .base_execution.dry_tolerance_m))
        validate_producer_rows_v84(output, self.execution["fingerprint"])
        depth = _array(
            depth_after_m, (self.lifecycle_execution.cell_count,),
            "pure handoff depth", nonnegative=True)
        momentum = _array(
            endpoint_momentum_m2_s,
            (2, self.lifecycle_execution.cell_count),
            "pure handoff momentum")
        fraction = _array(
            active_fraction,
            (self.lifecycle_execution.fraction_count,
             self.lifecycle_execution.cell_count),
            "pure handoff active fraction", nonnegative=True)
        if np.any(fraction > 1.0):
            raise Q3ProducerLifecycleV84Error(
                "v84 pure handoff active fraction exceeds one")
        dry = (self.lifecycle_execution.lifecycle_execution
               .base_execution.dry_tolerance_m)
        wet = depth > dry
        safe = np.where(wet, depth, 1.0)
        velocity_x = np.where(wet, momentum[0]/safe, 0.0)
        velocity_y = np.where(wet, momentum[1]/safe, 0.0)
        raw = output.raw_rows
        handoff = compose_route_handoff_v74(
            execution=self.lifecycle_execution.route_execution,
            depth_m=depth, velocity_x_m_s=velocity_x,
            velocity_y_m_s=velocity_y,
            raw_bed_x_kg_m_s=raw[0], raw_bed_y_kg_m_s=raw[1],
            raw_suspended_x_kg_m_s=raw[2],
            raw_suspended_y_kg_m_s=raw[3],
            raw_equilibrium_concentration_kg_m3=raw[4],
            active_fraction=fraction)
        validate_route_handoff_v74(
            handoff, execution=self.lifecycle_execution.route_execution)
        placeholder = Q3RouteHandoffReceiptV84(
            self.execution["fingerprint"], "cpu", output, handoff,
            _digest(depth, momentum), _digest(fraction), 0, "")
        return Q3RouteHandoffReceiptV84(
            **{**placeholder.__dict__,
               "receipt_sha256": _route_handoff_token(placeholder)})

    def _validate(self, parent, trial):
        validate_route_lifecycle_state_v81(parent, self.lifecycle_execution)
        if (not isinstance(trial, Q3BoundRouteLifecycleTrialV84) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.trial_sha256 != _trial_token(trial)):
            raise Q3ProducerLifecycleV84Error(
                "v84 trial is stale, mutated or belongs to another parent")
        validate_producer_rows_v84(trial.producer, self.execution["fingerprint"])

    def commit(self, parent, trial):
        self._validate(parent, trial)
        return commit_route_lifecycle_trial_v81(
            self.lifecycle_execution, parent, trial.lifecycle_trial)

    def rollback(self, parent, trial):
        self._validate(parent, trial)
        return rollback_route_lifecycle_trial_v81(
            self.lifecycle_execution, parent, trial.lifecycle_trial)

    def cancel(self, parent, trial):
        self._validate(parent, trial)
        return cancel_route_lifecycle_trial_v81(
            self.lifecycle_execution, parent, trial.lifecycle_trial)

    def checkpoint(self, state):
        return route_lifecycle_checkpoint_v81(self.lifecycle_execution, state)

    def restore_checkpoint(self, document):
        return load_route_lifecycle_checkpoint_v81(
            self.lifecycle_execution, document)


def method_identity_v84() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "methods": list(PUBLIC_METHODS),
        "routes": ["automatic_off_total_or_combine"]*3,
        "real_q3_producer_execution": True,
        "caller_supplied_raw_rows": False,
        "cpu": "compiled_v67_fortran_adapter_through_c_abi",
        "gpu": "resident_v67_cuda_family",
        "integrator": "implicit_fwave_midpoint_on_gpu",
        "maximum_waterdepth": False,
        "fixed_grid_only": True,
        "wave_active": False,
        "external_secondary_rotation": False,
        "surface_fraction_application_count_before_v81": 0,
        "post_producer_density_conversion_count": 0,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_backend_fallback": False,
        "amr_claimed": False,
        "product_admission_claimed": False,
        "delft_trajectory_equivalence_claimed": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "MAX_ACCEPTED_UPDATES", "NativeQ3BoundRouteLifecycleV84",
    "NativeQ3SplitProducerV84", "PROFILE_METHODS", "PUBLIC_METHODS",
    "Q3AcceptedProducerRowsV84", "Q3BoundRouteLifecycleTrialV84",
    "Q3RouteHandoffReceiptV84",
    "Q3ProducerLifecycleV84Error", "RAW_ROWS", "SCHEMA", "SCOPE_ID",
    "method_identity_v84", "validate_binding_executions_v84",
    "validate_producer_rows_v84", "validate_route_handoff_receipt_v84",
]
