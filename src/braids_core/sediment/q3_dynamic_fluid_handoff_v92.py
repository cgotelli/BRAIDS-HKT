"""Bounded v92 parent-fluid handoff to the accepted Q3/v84 producers.

The accepted v84 CPU/GPU producer identities are retained.  This additive
seam changes only one missing input contract: water density is a cell field
from the accepted v92 parent instead of the static v67 execution profile.
Parent molecular viscosity and parent-timed settling velocity are also sealed
in the handoff.  For the admitted current-only TraFrm 19/20 implementation,
viscosity is a finite/positive admission input but is not present in the
numerical expression; the receipt records that limitation explicitly.
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

from .flow2dh_settling import settlefrm2_noncohesive
from .q3_producer_lifecycle_binding_v84 import (
    Q3AcceptedProducerRowsV84,
    Q3ProducerLifecycleV84Error,
    _producer_token,
    validate_binding_executions_v84,
    validate_producer_rows_v84,
)
from .direct_vector_family_live_v67 import (
    load_direct_vector_family_execution_v67,
)
from .route_lifecycle_composition_v81 import RouteLifecycleExecutionV81
from .shared_turbulence_consumers_v91 import HydraulicTurbulenceReceiptV91
from .transported_fluid_settling_v92 import (
    TransportedFluidExecutionV92,
    TransportedFluidStateV92,
    _validate_state as _validate_fluid_state,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-q3-dynamic-fluid-handoff-v92-v1"
MAX_PARENT_ACCEPTED_UPDATES = 1
ROUTE_METHODS = (
    "capacity.bijker",
    "capacity.van_thiel_van_rijn",
    "capacity.xbeach_soulsby_van_rijn",
)
DENSITY_ROLE = ("numerical", "numerical", "numerical")
VISCOSITY_ROLE = ("not_consumed", "admission_only", "admission_only")
SETTLING_ROLE = ("numerical", "not_consumed", "not_consumed")
DENSITY_NUMERICAL_COUNT = (1, 1, 1)
VISCOSITY_ADMISSION_COUNT = (0, 1, 1)
VISCOSITY_NUMERICAL_COUNT = (0, 0, 0)
SETTLING_NUMERICAL_COUNT = (1, 0, 0)


class Q3DynamicFluidV92Error(ValueError):
    """The bounded parent-fluid/Q3 handoff or producer is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise Q3DynamicFluidV92Error(
            "v92 Q3 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise Q3DynamicFluidV92Error(
            f"v92 Q3 {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise Q3DynamicFluidV92Error(f"v92 Q3 {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise Q3DynamicFluidV92Error(f"v92 Q3 {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise Q3DynamicFluidV92Error(
            f"v92 Q3 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _parent_settling(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
) -> tuple[np.ndarray, np.ndarray]:
    base = execution.base_execution.base_execution.base_execution
    depth = np.asarray(parent.q[0])
    wet = depth > base.dry_tolerance_m
    concentration = np.zeros((execution.cell_count, execution.fraction_count))
    concentration[wet] = (
        np.asarray(parent.suspended_state.suspended_mass_kg_m2)[:, wet]/
        depth[wet][None, :]).T
    ctot = np.ascontiguousarray(np.sum(concentration, axis=1))
    settled = settlefrm2_noncohesive(
        concentration, base.profile.sediment_d50_m,
        base.profile.settling_fac_dss,
        base.profile.sediment_density_kg_m3,
        parent.water_density_cell_kg_m3, parent.temperature_cell_c,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2, wet_mask=wet)
    viscosity = np.asarray(settled.molecular_viscosity_m2_s)
    if not np.array_equal(viscosity,
                          parent.molecular_viscosity_cell_m2_s):
        raise Q3DynamicFluidV92Error(
            "v92 Q3 parent settling viscosity differs from transported state")
    settling = np.ascontiguousarray(settled.settling_velocity_m_s.T)
    ctot.setflags(write=False)
    settling.setflags(write=False)
    return ctot, settling


@dataclass(frozen=True)
class Q3DynamicFluidHandoffV92:
    transported_execution_fingerprint: str
    parent_state_sha256: str
    parent_base_state_sha256: str
    hydraulic_receipt_sha256: str
    parent_accepted_updates: int
    parent_time_s: float
    accepted_q: np.ndarray
    roughness_parameter: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    parent_total_suspended_concentration_kg_m3: np.ndarray
    settling_velocity_m_s: np.ndarray
    arrays_sha256: str
    handoff_sha256: str


def _handoff_token(value: Q3DynamicFluidHandoffV92) -> str:
    return _fingerprint({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":parent-handoff",
        "transported_execution_fingerprint":
            value.transported_execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "parent_base_state_sha256": value.parent_base_state_sha256,
        "hydraulic_receipt_sha256": value.hydraulic_receipt_sha256,
        "parent_accepted_updates": value.parent_accepted_updates,
        "parent_time_s": value.parent_time_s,
        "arrays_sha256": value.arrays_sha256,
        "routes": list(ROUTE_METHODS),
    })


def make_q3_dynamic_fluid_handoff_v92(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    hydraulic_receipt: HydraulicTurbulenceReceiptV91, *,
    accepted_q: object,
    roughness_parameter: object,
) -> Q3DynamicFluidHandoffV92:
    """Seal exact parent rho/nu/ws and one accepted hydraulic endpoint."""
    if not isinstance(execution, TransportedFluidExecutionV92):
        raise Q3DynamicFluidV92Error("v92 Q3 transported execution differs")
    _validate_fluid_state(parent, execution)
    if parent.accepted_updates > MAX_PARENT_ACCEPTED_UPDATES:
        raise Q3DynamicFluidV92Error(
            "v92 Q3 bounded handoff exceeds two-update evidence scope")
    if execution.fraction_count != 3:
        raise Q3DynamicFluidV92Error(
            "v92 Q3 dynamic seam requires the accepted three-lane v84 family")
    q = _array(accepted_q, (3, execution.cell_count),
               "accepted hydraulic endpoint")
    roughness = _array(roughness_parameter, (execution.cell_count,),
                       "roughness", positive=True)
    if (not isinstance(hydraulic_receipt, HydraulicTurbulenceReceiptV91) or
            hydraulic_receipt.execution_fingerprint !=
                execution.base_execution.fingerprint or
            hydraulic_receipt.parent_state_sha256 !=
                parent.base_state.state_sha256 or
            hydraulic_receipt.accepted_q_sha256 != _digest(q)):
        raise Q3DynamicFluidV92Error(
            "v92 Q3 hydraulic receipt is stale or belongs to another parent")
    rho = _array(parent.water_density_cell_kg_m3,
                 (execution.cell_count,), "parent water density", positive=True)
    nu = _array(parent.molecular_viscosity_cell_m2_s,
                (execution.cell_count,), "parent molecular viscosity",
                positive=True)
    ctot, settling = _parent_settling(execution, parent)
    arrays_sha = _digest(q, roughness, rho, nu, ctot, settling)
    placeholder = Q3DynamicFluidHandoffV92(
        execution.fingerprint, parent.state_sha256,
        parent.base_state.state_sha256, hydraulic_receipt.receipt_sha256,
        parent.accepted_updates, parent.time_s, q, roughness, rho, nu, ctot,
        settling, arrays_sha, "")
    return Q3DynamicFluidHandoffV92(
        **{**placeholder.__dict__,
           "handoff_sha256": _handoff_token(placeholder)})


def validate_q3_dynamic_fluid_handoff_v92(
    value: Q3DynamicFluidHandoffV92,
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    hydraulic_receipt: HydraulicTurbulenceReceiptV91,
) -> Q3DynamicFluidHandoffV92:
    _validate_fluid_state(parent, execution)
    if not isinstance(value, Q3DynamicFluidHandoffV92):
        raise Q3DynamicFluidV92Error("v92 Q3 handoff type differs")
    expected = make_q3_dynamic_fluid_handoff_v92(
        execution, parent, hydraulic_receipt,
        accepted_q=value.accepted_q,
        roughness_parameter=value.roughness_parameter)
    if (value.handoff_sha256 != _handoff_token(value) or
            value.handoff_sha256 != expected.handoff_sha256 or
            value.arrays_sha256 != expected.arrays_sha256 or
            any(not np.array_equal(getattr(value, name),
                                   getattr(expected, name)) for name in (
                "accepted_q", "roughness_parameter",
                "water_density_cell_kg_m3",
                "molecular_viscosity_cell_m2_s",
                "parent_total_suspended_concentration_kg_m3",
                "settling_velocity_m_s"))):
        raise Q3DynamicFluidV92Error(
            "v92 Q3 handoff is stale, forged or cross-parent")
    return value


@dataclass(frozen=True)
class Q3DynamicFluidConsumptionReceiptV92:
    handoff_sha256: str
    v84_execution_fingerprint: str
    backend: str
    producer_sha256: str
    density_role: tuple[str, str, str]
    viscosity_role: tuple[str, str, str]
    settling_role: tuple[str, str, str]
    density_numerical_count: tuple[int, int, int]
    viscosity_admission_count: tuple[int, int, int]
    viscosity_numerical_count: tuple[int, int, int]
    settling_numerical_count: tuple[int, int, int]
    consumed_arrays_sha256: str
    receipt_sha256: str


def _consumption_token(value: Q3DynamicFluidConsumptionReceiptV92) -> str:
    return _fingerprint({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":consumption",
        "handoff_sha256": value.handoff_sha256,
        "v84_execution_fingerprint": value.v84_execution_fingerprint,
        "backend": value.backend,
        "producer_sha256": value.producer_sha256,
        "density_role": list(value.density_role),
        "viscosity_role": list(value.viscosity_role),
        "settling_role": list(value.settling_role),
        "density_numerical_count": list(value.density_numerical_count),
        "viscosity_admission_count": list(value.viscosity_admission_count),
        "viscosity_numerical_count": list(value.viscosity_numerical_count),
        "settling_numerical_count": list(value.settling_numerical_count),
        "consumed_arrays_sha256": value.consumed_arrays_sha256,
    })


@dataclass(frozen=True)
class Q3DynamicFluidAcceptedRowsV92:
    producer: Q3AcceptedProducerRowsV84
    consumption: Q3DynamicFluidConsumptionReceiptV92
    result_sha256: str


def _result_token(value: Q3DynamicFluidAcceptedRowsV92) -> str:
    return _fingerprint({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":accepted-rows",
        "producer_sha256": value.producer.producer_sha256,
        "consumption_receipt_sha256": value.consumption.receipt_sha256,
    })


def _accepted_result(
    handoff: Q3DynamicFluidHandoffV92,
    v84_execution_fingerprint: str,
    backend: str,
    producer: Q3AcceptedProducerRowsV84,
) -> Q3DynamicFluidAcceptedRowsV92:
    placeholder_receipt = Q3DynamicFluidConsumptionReceiptV92(
        handoff.handoff_sha256, v84_execution_fingerprint, backend,
        producer.producer_sha256, DENSITY_ROLE, VISCOSITY_ROLE, SETTLING_ROLE,
        DENSITY_NUMERICAL_COUNT, VISCOSITY_ADMISSION_COUNT,
        VISCOSITY_NUMERICAL_COUNT, SETTLING_NUMERICAL_COUNT,
        _digest(handoff.water_density_cell_kg_m3,
                handoff.molecular_viscosity_cell_m2_s,
                handoff.settling_velocity_m_s), "")
    receipt = Q3DynamicFluidConsumptionReceiptV92(
        **{**placeholder_receipt.__dict__,
           "receipt_sha256": _consumption_token(placeholder_receipt)})
    placeholder = Q3DynamicFluidAcceptedRowsV92(producer, receipt, "")
    return Q3DynamicFluidAcceptedRowsV92(
        producer, receipt, _result_token(placeholder))


def validate_q3_dynamic_fluid_result_v92(
    value: Q3DynamicFluidAcceptedRowsV92,
    handoff: Q3DynamicFluidHandoffV92,
    v84_execution_fingerprint: str,
) -> Q3DynamicFluidAcceptedRowsV92:
    if not isinstance(value, Q3DynamicFluidAcceptedRowsV92):
        raise Q3DynamicFluidV92Error("v92 Q3 accepted result type differs")
    validate_producer_rows_v84(value.producer, v84_execution_fingerprint)
    receipt = value.consumption
    expected_arrays = _digest(
        handoff.water_density_cell_kg_m3,
        handoff.molecular_viscosity_cell_m2_s,
        handoff.settling_velocity_m_s)
    if (not isinstance(receipt, Q3DynamicFluidConsumptionReceiptV92) or
            receipt.handoff_sha256 != handoff.handoff_sha256 or
            receipt.v84_execution_fingerprint != v84_execution_fingerprint or
            receipt.producer_sha256 != value.producer.producer_sha256 or
            receipt.backend not in ("cpu", "actual_cuda") or
            receipt.density_role != DENSITY_ROLE or
            receipt.viscosity_role != VISCOSITY_ROLE or
            receipt.settling_role != SETTLING_ROLE or
            receipt.density_numerical_count != DENSITY_NUMERICAL_COUNT or
            receipt.viscosity_admission_count != VISCOSITY_ADMISSION_COUNT or
            receipt.viscosity_numerical_count != VISCOSITY_NUMERICAL_COUNT or
            receipt.settling_numerical_count != SETTLING_NUMERICAL_COUNT or
            receipt.consumed_arrays_sha256 != expected_arrays or
            receipt.receipt_sha256 != _consumption_token(receipt) or
            value.result_sha256 != _result_token(value)):
        raise Q3DynamicFluidV92Error(
            "v92 Q3 consumption receipt is stale, forged or incomplete")
    return value


class NativeQ3DynamicFluidProducerV92:
    """Execute the accepted v67 family through the additive cell-rho ABI."""

    def __init__(self, *, producer_execution: Mapping[str, object],
                 lifecycle_execution: RouteLifecycleExecutionV81,
                 fluid_execution: TransportedFluidExecutionV92,
                 library_path: str | Path,
                 root: str | Path | None = None):
        self.root = Path(root or Path(__file__).resolve().parents[2])
        self.fluid_execution = fluid_execution
        self.execution = validate_binding_executions_v84(
            producer_execution, lifecycle_execution, backend="cpu",
            root=self.root)
        self.producer_execution = load_direct_vector_family_execution_v67(
            producer_execution)
        if fluid_execution.fraction_count != 3:
            raise Q3DynamicFluidV92Error(
                "v92 Q3 native producer requires three transported fractions")
        self.sediment_threshold_m = (
            fluid_execution.base_execution.base_execution.base_execution
            .profile.sediment_threshold_m)
        path = Path(library_path).resolve()
        if not path.is_file():
            raise Q3DynamicFluidV92Error(
                "v92 Q3 dynamic native library is absent")
        self.library_path = path
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        source = (self.root/
                  "solver/wave/braids_q3_dynamic_fluid_producer_v92_module.f90")
        if not source.is_file():
            raise Q3DynamicFluidV92Error(
                "v92 Q3 dynamic native wrapper source is absent")
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
            library = ctypes.CDLL(str(path))
            self.function = library.braids_q3_dynamic_fluid_produce_v92
            self.library = library
        except (OSError, AttributeError) as error:
            raise Q3DynamicFluidV92Error(
                "v92 Q3 dynamic native producer cannot be loaded") from error
        pointer_d = np.ctypeslib.ndpointer(dtype=np.float64, flags="ALIGNED")
        pointer_i = np.ctypeslib.ndpointer(dtype=np.int32, flags="ALIGNED")
        self.function.argtypes = [
            ctypes.c_int, pointer_d, pointer_d, pointer_d, pointer_d,
            pointer_d, pointer_d, pointer_d, pointer_d, pointer_d, pointer_d,
            ctypes.c_double, pointer_d, pointer_i,
        ]
        self.function.restype = None

    def produce(
        self, parent: TransportedFluidStateV92,
        hydraulic_receipt: HydraulicTurbulenceReceiptV91,
        handoff: Q3DynamicFluidHandoffV92,
    ) -> Q3DynamicFluidAcceptedRowsV92:
        handoff = validate_q3_dynamic_fluid_handoff_v92(
            handoff, self.fluid_execution, parent, hydraulic_receipt)
        q = np.asarray(handoff.accepted_q)
        depth = np.asarray(q[0])
        active = ((depth > self.fluid_execution.base_execution.base_execution
                   .base_execution.dry_tolerance_m) &
                  (depth > self.sediment_threshold_m))
        safe_depth = np.where(active, depth, 1.0)
        u = np.where(active, q[1]/safe_depth, 0.0)
        v = np.where(active, q[2]/safe_depth, 0.0)
        u[np.abs(u) < 1.0e-12] = 0.0
        v[np.abs(v) < 1.0e-12] = 0.0
        speed = np.hypot(u, v)
        effective_u = np.asfortranarray(speed*u/(speed+1.0e-6))
        effective_v = np.asfortranarray(speed*v/(speed+1.0e-6))
        physical = self.producer_execution["physical_profile"]
        work_depth = np.maximum(
            safe_depth, float(physical["minimum_work_depth_m"]))
        chezy = np.asfortranarray(
            work_depth**(1.0/6.0)/handoff.roughness_parameter)
        d50 = np.asfortranarray(physical["d50_m"], dtype=np.float64)
        d90 = np.asfortranarray(physical["d90_m"], dtype=np.float64)
        rhos = np.asfortranarray(
            physical["solid_density_kg_m3"], dtype=np.float64)
        nc = self.fluid_execution.cell_count
        raw = np.empty((5, 3, nc), dtype=np.float64, order="F")
        status = np.empty((3, nc), dtype=np.int32, order="F")
        self.function(
            nc, np.asfortranarray(safe_depth), effective_u, effective_v,
            chezy,
            np.asfortranarray(handoff.molecular_viscosity_cell_m2_s),
            np.asfortranarray(handoff.settling_velocity_m_s), d50, d90, rhos,
            np.asfortranarray(handoff.water_density_cell_kg_m3),
            float(physical["gravity_m_s2"]), raw, status)
        raw[:, :, ~active] = 0.0
        if np.any(status[:, active] != 0):
            index = np.argwhere(status[:, active] != 0)[0].tolist()
            raise Q3DynamicFluidV92Error(
                f"v92 Q3 native producer rejected lane/cell {index}")
        raw = np.ascontiguousarray(raw)
        wet_i = np.ascontiguousarray(active.astype(np.int32))
        raw.setflags(write=False)
        wet_i.setflags(write=False)
        producer_receipt = _fingerprint({
            "scope_id": SCOPE_ID + ":native-producer",
            "v84_execution": self.execution["fingerprint"],
            "handoff_sha256": handoff.handoff_sha256,
            "native_library_sha256": self.library_sha256,
            "native_wrapper_sha256": self.source_sha256,
        })
        placeholder = Q3AcceptedProducerRowsV84(
            self.execution["fingerprint"], "cpu", raw, wet_i,
            producer_receipt,
            self.execution["producer_source_receipt_fingerprint"], 0, 0, "")
        producer = Q3AcceptedProducerRowsV84(
            **{**placeholder.__dict__,
               "producer_sha256": _producer_token(placeholder)})
        result = _accepted_result(
            handoff, self.execution["fingerprint"], "cpu", producer)
        return validate_q3_dynamic_fluid_result_v92(
            result, handoff, self.execution["fingerprint"])


def method_identity_q3_dynamic_fluid_v92() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "accepted_q3_identity": "v84_v67_three_lane_current_only",
        "dynamic_parent_density": True,
        "dynamic_parent_viscosity": True,
        "dynamic_parent_settling": True,
        "route_consumption": {
            ROUTE_METHODS[index]: {
                "rho": DENSITY_ROLE[index],
                "nu": VISCOSITY_ROLE[index],
                "settling_velocity": SETTLING_ROLE[index],
            } for index in range(3)
        },
        "max_parent_accepted_updates": MAX_PARENT_ACCEPTED_UPDATES,
        "full_v96_transaction": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "DENSITY_NUMERICAL_COUNT", "DENSITY_ROLE", "MAX_PARENT_ACCEPTED_UPDATES",
    "NativeQ3DynamicFluidProducerV92", "Q3DynamicFluidAcceptedRowsV92",
    "Q3DynamicFluidConsumptionReceiptV92", "Q3DynamicFluidHandoffV92",
    "Q3DynamicFluidV92Error", "ROUTE_METHODS", "SCHEMA", "SCOPE_ID",
    "SETTLING_NUMERICAL_COUNT", "SETTLING_ROLE",
    "VISCOSITY_ADMISSION_COUNT", "VISCOSITY_NUMERICAL_COUNT",
    "VISCOSITY_ROLE", "make_q3_dynamic_fluid_handoff_v92",
    "method_identity_q3_dynamic_fluid_v92",
    "validate_q3_dynamic_fluid_handoff_v92",
    "validate_q3_dynamic_fluid_result_v92",
]
