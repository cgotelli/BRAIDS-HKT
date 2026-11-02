"""Persistent accepted physical scalar-boundary image for v96 hydraulics.

The transported-fluid v92 state intentionally stores only hysteresis memory,
not the resolved physical boundary concentration used by the accepted trial.
Open-boundary DensIn needs that accepted image one update later.  This module
persists the exact sediment and T/S arrays from an authenticated v92 trial and
derives a parent-frozen water/mixture-density receipt before the next Picard
solve.  It never reruns a scalar boundary resolver inside the hydraulic solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np

from .coupled_open_boundary_native_cpu_v96 import (
    AcceptedBoundaryDensityReceiptV96,
    make_accepted_boundary_density_receipt_v96,
)
from .densin_fwave_v85 import mixture_density_kg_m3_v85
from .dynamic_fluid_settling_v88 import resolve_water_density_v88
from .hydraulic_boundary_trace_cpu_v96 import (
    HydraulicBoundaryTraceExecutionV96,
    execution_document_hydraulic_boundary_trace_v96,
)
from .transported_fluid_settling_v92 import (
    TransportedFluidExecutionV92,
    TransportedFluidTrialV92,
    execution_document_v92,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-accepted-boundary-scalar-state-cpu-v96-v1"
SOURCE_INITIAL = "authenticated_initial_boundary_resolution"
SOURCE_ACCEPTED_V92 = "accepted_v92_physical_boundary_image"
SOURCE_KINDS = (SOURCE_INITIAL, SOURCE_ACCEPTED_V92)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AcceptedBoundaryScalarStateV96Error(ValueError):
    """The accepted boundary scalar image is invalid or stale."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar value is not finite canonical JSON") from error


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


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AcceptedBoundaryScalarStateV96Error(
            f"v96 boundary scalar {name} must be a lowercase SHA-256")
    return value


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise AcceptedBoundaryScalarStateV96Error(
            f"v96 boundary scalar {name} must have shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise AcceptedBoundaryScalarStateV96Error(
            f"v96 boundary scalar {name} must be finite and nonnegative")
    result.setflags(write=False)
    return result


def _fluid_array(value: object, shape: tuple[int, ...],
                 open_mask: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise AcceptedBoundaryScalarStateV96Error(
            f"v96 boundary scalar fluid image must have shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    # v92 deliberately publishes -20 C on an unresolved/non-boundary face.
    # Preserve that exact accepted image, but require physical values wherever
    # the hydraulic trace execution says an open boundary exists.
    if (not np.isfinite(result).all() or
            np.any(result[0, open_mask] <= -20.0) or
            np.any(result[1, open_mask] < 0.0)):
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 open-face fluid boundary scalar is outside EOS bounds")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AcceptedBoundaryScalarStateV96:
    transported_fluid_execution_fingerprint: str
    boundary_trace_execution_fingerprint: str
    accepted_update: int
    time_s: float
    suspended_concentration_kg_m3: np.ndarray
    fluid_transport_scalar: np.ndarray
    source_kind: str
    source_receipt_sha256: str
    state_sha256: str


def _state_token(value: AcceptedBoundaryScalarStateV96) -> str:
    return _fingerprint({
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "transported_fluid_execution_fingerprint":
            value.transported_fluid_execution_fingerprint,
        "boundary_trace_execution_fingerprint":
            value.boundary_trace_execution_fingerprint,
        "accepted_update": value.accepted_update,
        "time_s": value.time_s,
        "arrays_sha256": _digest(
            value.suspended_concentration_kg_m3,
            value.fluid_transport_scalar),
        "source_kind": value.source_kind,
        "source_receipt_sha256": value.source_receipt_sha256,
        "timing": "accepted-current-scalar-image-consumed-next-hydraulics",
    })


def _make_state(
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96, *,
    accepted_update: int,
    time_s: float,
    suspended_concentration_kg_m3: object,
    fluid_transport_scalar: object,
    source_kind: str,
    source_receipt_sha256: str,
) -> AcceptedBoundaryScalarStateV96:
    execution_document_v92(execution)
    execution_document_hydraulic_boundary_trace_v96(boundary_execution)
    if (execution.cell_count != boundary_execution.cell_count or
            execution.face_count != boundary_execution.face_count or
            source_kind not in SOURCE_KINDS or
            type(accepted_update) is not int or accepted_update < 0):
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar execution, source, or update differs")
    time = float(time_s)
    if not math.isfinite(time) or time < 0.0:
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar time is invalid")
    ng, nf = execution.fraction_count, execution.face_count
    suspended = _array(
        suspended_concentration_kg_m3, (ng, nf),
        "suspended concentration")
    fluid = _fluid_array(
        fluid_transport_scalar, (2, nf), boundary_execution.open_mask)
    placeholder = AcceptedBoundaryScalarStateV96(
        execution.fingerprint, boundary_execution.fingerprint,
        accepted_update, time, suspended, fluid, source_kind,
        _sha(source_receipt_sha256, "source receipt"), "")
    return AcceptedBoundaryScalarStateV96(
        **{**placeholder.__dict__,
           "state_sha256": _state_token(placeholder)})


def resolve_initial_accepted_boundary_scalar_state_v96(
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96, *,
    suspended_concentration_kg_m3: object,
    fluid_transport_scalar: object,
    initial_boundary_resolution_receipt_sha256: str,
) -> AcceptedBoundaryScalarStateV96:
    return _make_state(
        execution, boundary_execution, accepted_update=0, time_s=0.0,
        suspended_concentration_kg_m3=suspended_concentration_kg_m3,
        fluid_transport_scalar=fluid_transport_scalar,
        source_kind=SOURCE_INITIAL,
        source_receipt_sha256=initial_boundary_resolution_receipt_sha256)


def make_accepted_boundary_scalar_state_from_v92_trial_v96(
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96,
    trial: TransportedFluidTrialV92,
) -> AcceptedBoundaryScalarStateV96:
    if (type(trial) is not TransportedFluidTrialV92 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.candidate.accepted_updates !=
                trial.suspended_trial.candidate.accepted_updates):
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar source trial differs")
    return _make_state(
        execution, boundary_execution,
        accepted_update=trial.candidate.accepted_updates,
        time_s=trial.candidate.time_s,
        suspended_concentration_kg_m3=
            trial.suspended_trial.base_trial.boundary_concentration_kg_m3,
        fluid_transport_scalar=trial.fluid_transport.boundary_transport_scalar,
        source_kind=SOURCE_ACCEPTED_V92,
        source_receipt_sha256=trial.trial_sha256)


def validate_accepted_boundary_scalar_state_v96(
    value: AcceptedBoundaryScalarStateV96,
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96, *,
    accepted_update: int | None = None,
    time_s: float | None = None,
) -> None:
    if type(value) is not AcceptedBoundaryScalarStateV96:
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar state type differs")
    expected = _make_state(
        execution, boundary_execution,
        accepted_update=value.accepted_update, time_s=value.time_s,
        suspended_concentration_kg_m3=
            value.suspended_concentration_kg_m3,
        fluid_transport_scalar=value.fluid_transport_scalar,
        source_kind=value.source_kind,
        source_receipt_sha256=value.source_receipt_sha256)
    if (value.state_sha256 != expected.state_sha256 or
            (accepted_update is not None and
             value.accepted_update != accepted_update) or
            (time_s is not None and value.time_s != float(time_s))):
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar state is stale or cross-clock")


def make_boundary_density_from_accepted_scalar_state_v96(
    value: AcceptedBoundaryScalarStateV96,
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96, *,
    parent_state_sha256: str,
) -> AcceptedBoundaryDensityReceiptV96:
    validate_accepted_boundary_scalar_state_v96(
        value, execution, boundary_execution)
    profile = execution.base_execution.base_execution.base_execution.profile
    temperature, salinity = value.fluid_transport_scalar
    mask = boundary_execution.open_mask
    water_open = resolve_water_density_v88(
        temperature[mask], salinity[mask], eos=execution.density_eos,
        constituents_active=execution.constituents_active,
        reference_water_density_kg_m3=profile.water_density_kg_m3)
    mixture_open = mixture_density_kg_m3_v85(
        value.suspended_concentration_kg_m3[:, mask], water_open,
        profile.sediment_density_kg_m3)
    water = np.zeros(execution.face_count, dtype=np.float64)
    mixture = np.zeros(execution.face_count, dtype=np.float64)
    water[mask] = water_open
    mixture[mask] = mixture_open
    return make_accepted_boundary_density_receipt_v96(
        boundary_execution, parent_state_sha256=parent_state_sha256,
        accepted_boundary_scalar_state_sha256=value.state_sha256,
        source_trial_sha256=value.source_receipt_sha256,
        accepted_update=value.accepted_update, time_s=value.time_s,
        water_density_kg_m3=water, mixture_density_kg_m3=mixture)


def accepted_boundary_scalar_checkpoint_v96(
    value: AcceptedBoundaryScalarStateV96,
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96,
) -> dict[str, object]:
    validate_accepted_boundary_scalar_state_v96(
        value, execution, boundary_execution)
    body = {
        "schema": SCHEMA, "scope_id": SCOPE_ID,
        "transported_fluid_execution_fingerprint":
            value.transported_fluid_execution_fingerprint,
        "boundary_trace_execution_fingerprint":
            value.boundary_trace_execution_fingerprint,
        "accepted_update": value.accepted_update, "time_s": value.time_s,
        "suspended_concentration_kg_m3":
            value.suspended_concentration_kg_m3.tolist(),
        "fluid_transport_scalar": value.fluid_transport_scalar.tolist(),
        "source_kind": value.source_kind,
        "source_receipt_sha256": value.source_receipt_sha256,
        "state_sha256": value.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_accepted_boundary_scalar_checkpoint_v96(
    document: object,
    execution: TransportedFluidExecutionV92,
    boundary_execution: HydraulicBoundaryTraceExecutionV96,
) -> AcceptedBoundaryScalarStateV96:
    keys = {
        "schema", "scope_id", "transported_fluid_execution_fingerprint",
        "boundary_trace_execution_fingerprint", "accepted_update", "time_s",
        "suspended_concentration_kg_m3", "fluid_transport_scalar",
        "source_kind", "source_receipt_sha256", "state_sha256",
        "fingerprint",
    }
    if not isinstance(document, dict) or set(document) != keys:
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["schema"] != SCHEMA or
            document["scope_id"] != SCOPE_ID or
            document["fingerprint"] != _fingerprint(body)):
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar checkpoint identity differs")
    value = _make_state(
        execution, boundary_execution,
        accepted_update=document["accepted_update"],
        time_s=document["time_s"],
        suspended_concentration_kg_m3=
            document["suspended_concentration_kg_m3"],
        fluid_transport_scalar=document["fluid_transport_scalar"],
        source_kind=document["source_kind"],
        source_receipt_sha256=document["source_receipt_sha256"])
    if value.state_sha256 != document["state_sha256"]:
        raise AcceptedBoundaryScalarStateV96Error(
            "v96 boundary scalar checkpoint state differs")
    return value


__all__ = [
    "SOURCE_ACCEPTED_V92", "SOURCE_INITIAL",
    "AcceptedBoundaryScalarStateV96",
    "AcceptedBoundaryScalarStateV96Error",
    "accepted_boundary_scalar_checkpoint_v96",
    "load_accepted_boundary_scalar_checkpoint_v96",
    "make_accepted_boundary_scalar_state_from_v92_trial_v96",
    "make_boundary_density_from_accepted_scalar_state_v96",
    "resolve_initial_accepted_boundary_scalar_state_v96",
    "validate_accepted_boundary_scalar_state_v96",
]
