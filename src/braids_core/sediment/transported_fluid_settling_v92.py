"""Bounded Q4/v92 transported fluid scalars and dynamic settling.

This module composes the accepted v91 hydraulic/turbulence transaction with
one scalar owner.  Temperature and salinity are conservative depth-integrated
constituents.  The temperature lane is transported as the non-negative affine
quantity ``h*(T - T_floor)`` with ``T_floor=-20 C``; an authenticated discrete
water-continuity identity makes this exactly equivalent to transporting
``h*T`` while allowing the accepted non-negative FCT/diffusion kernels to be
shared with salinity and suspended sediment.

The locked FLOW ordering matters.  Current-step SettleFrm=2 uses the parent
temperature, density and *parent pre-transport* total suspended concentration.
The transported endpoint T/S fields are then converted through the selected
equation of state and become the properties for the next accepted update.
No caller-supplied endpoint fluid field is accepted.

V92 retains v91's fully-wet, uniform-Cartesian, fixed-grid scope and four-step
bound.  Moving wet/dry hydraulics remains a v97 requirement; the pure scalar
helper nevertheless records a conservative cleanup for a dry endpoint so the
policy is explicit rather than silently invented by a backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np

from .dynamic_fluid_settling_v88 import (
    EOS_CHOICES,
    EOS_ECKART,
    EOS_NACL,
    EOS_UNESCO,
    resolve_water_density_v88,
)
from .flow2dh_boundary_diffusion import (
    Flow2DHBoundaryMemory,
    flow2dh_horizontal_diffusive_face_mass,
    resolve_flow2dh_open_boundary,
)
from .flow2dh_exchange import advance_flow2dh_equilibrium_exchange
from .flow2dh_settling import settlefrm2_noncohesive
from .hles_producer_v90 import (
    BACKEND_CPU,
    BACKEND_GPU,
    CurrentClosureHandoffV90,
    HlesProducerStateV90,
    validate_current_closure_handoff_v90,
    validate_hles_producer_state_v90,
)
from .multifraction_suspended_lifecycle_v78 import (
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedTrialV78,
    build_multifraction_trial_v78,
    commit_multifraction_trial_v78,
    rollback_multifraction_trial_v78,
    validate_implicit_fwave_midpoint_step_receipt_v78,
)
from .shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    ChkvicUsedTotalReceiptV91,
    HydraulicTurbulenceReceiptV91,
    SharedTurbulenceExecutionV91,
    SharedTurbulenceStateV91,
    apply_chkvic_used_total_v91,
    execution_document_v91,
    load_shared_turbulence_checkpoint_v91,
    make_hydraulic_turbulence_receipt_v91,
    momentum_diffusion_rate_v91,
    shared_turbulence_checkpoint_v91,
    validate_chkvic_receipt_v91,
    validate_hydraulic_turbulence_receipt_v91,
    validate_shared_turbulence_state_v91,
    _wrap_state as _wrap_v91_state,
)
from .suspended_lifecycle_live_v76 import (
    _smooth_bound,
    resolve_suspended_lifecycle_state_v76,
)
from .suspended_support_closures_v82 import (
    SuspendedSupportExecutionV82,
    SuspendedSupportTrialV82,
    _wrap_state as _wrap_v82_state,
    _wrap_trial as _wrap_v82_trial,
    commit_suspended_support_trial_v82,
    resolve_suspended_support_execution_v82,
    rollback_suspended_support_trial_v82,
    _validate_trial as _validate_v82_trial,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-transported-fluid-settling-v92-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MAX_UPDATES = 4
MODE_CONSTANT = "equilibrium_constant"
MODE_TRANSPORTED = "transported_2dh"
MODES = (MODE_CONSTANT, MODE_TRANSPORTED)
TEMPERATURE_FLOOR_C = -20.0
FLUID_TEMPERATURE = 0
FLUID_SALINITY = 1
FLUID_SCALAR_COUNT = 2
FLUID_SCALAR_ORDER = ("temperature_plus_20_c", "salinity_ppt")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TransportedFluidSettlingV92Error(ValueError):
    """A v92 execution, state, scalar update or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise TransportedFluidSettlingV92Error(
            "v92 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TransportedFluidSettlingV92Error(
            f"v92 {name} must be a SHA-256 digest")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            lower: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise TransportedFluidSettlingV92Error(f"v92 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise TransportedFluidSettlingV92Error(
            f"v92 {name} must be positive")
    if lower is not None and result < lower:
        raise TransportedFluidSettlingV92Error(
            f"v92 {name} must be at least {lower}")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False, boolean: bool = False,
           integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu b".replace(" ", ""):
        raise TransportedFluidSettlingV92Error(
            f"v92 {name} must be numeric with shape {shape}")
    dtype = np.bool_ if boolean else np.int32 if integer else np.float64
    result = np.array(raw, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise TransportedFluidSettlingV92Error(f"v92 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise TransportedFluidSettlingV92Error(
            f"v92 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _mutable_array(value: object, shape: tuple[int, ...], name: str, *,
                   nonnegative: bool = False,
                   boolean: bool = False) -> np.ndarray:
    result = _array(value, shape, name, nonnegative=nonnegative,
                    boolean=boolean)
    return np.array(result, copy=True)


@dataclass(frozen=True)
class TransportedFluidExecutionV92:
    base_execution: SharedTurbulenceExecutionV91
    temperature_mode: str
    salinity_mode: str
    equilibrium_temperature_c: float
    equilibrium_salinity_ppt: float
    density_eos: str
    fluid_boundary_return_time_s: np.ndarray
    semantic_id: str
    fingerprint: str

    @property
    def backend(self) -> str:
        return self.base_execution.backend

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count

    @property
    def boundary_count(self) -> int:
        return self.base_execution.base_execution.boundary_count

    @property
    def fraction_count(self) -> int:
        return self.base_execution.base_execution.fraction_count

    @property
    def transported_mask(self) -> tuple[bool, bool]:
        return (self.temperature_mode == MODE_TRANSPORTED,
                self.salinity_mode == MODE_TRANSPORTED)

    @property
    def constituents_active(self) -> bool:
        return any(self.transported_mask)


def _execution_payload(value: TransportedFluidExecutionV92) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v91_execution": execution_document_v91(value.base_execution),
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "temperature_mode": value.temperature_mode,
        "salinity_mode": value.salinity_mode,
        "equilibrium_temperature_c": value.equilibrium_temperature_c,
        "equilibrium_salinity_ppt": value.equilibrium_salinity_ppt,
        "density_eos": value.density_eos,
        "temperature_affine_floor_c": TEMPERATURE_FLOOR_C,
        "transported_scalar_order": list(FLUID_SCALAR_ORDER),
        "fluid_boundary_return_time_s":
            value.fluid_boundary_return_time_s.tolist(),
        "boundary_neumann_override": False,
        "current_settling_inputs":
            "parent_T_parent_rho_parent_pretransport_ctot",
        "endpoint_properties_role": "next_accepted_update_only",
        "q3_dynamic_density_consumer": "deferred_to_v96_composition",
        "scalar_owner": "one_combined_sediment_plus_active_fluid_pass",
        "standalone_fluid_helper_admitted": False,
        "scalar_diffusion": "capped_total_dicuv_divided_by_0p7_once",
        "accepted_update_limit": MAX_UPDATES,
        "moving_wet_dry": (
            value.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97),
        "automatic_cpu_fallback": False,
        "product_admission": False,
        "locked_delft3d_source_read_only": True,
    }


def resolve_transported_fluid_execution_v92(
    base_execution: SharedTurbulenceExecutionV91, *,
    temperature_mode: str = MODE_TRANSPORTED,
    salinity_mode: str = MODE_TRANSPORTED,
    equilibrium_temperature_c: float = 15.0,
    equilibrium_salinity_ppt: float = 0.0,
    density_eos: str = EOS_UNESCO,
    fluid_boundary_return_time_s: object | None = None,
) -> TransportedFluidExecutionV92:
    execution_document_v91(base_execution)
    if temperature_mode not in MODES or salinity_mode not in MODES:
        raise TransportedFluidSettlingV92Error(
            "v92 scalar modes must be equilibrium_constant or transported_2dh")
    temperature = _finite(
        equilibrium_temperature_c, "equilibrium temperature")
    if temperature <= TEMPERATURE_FLOOR_C:
        raise TransportedFluidSettlingV92Error(
            "v92 equilibrium temperature must exceed -20 C")
    salinity = _finite(
        equilibrium_salinity_ppt, "equilibrium salinity", lower=0.0)
    if density_eos not in EOS_CHOICES:
        raise TransportedFluidSettlingV92Error(
            "v92 EOS must be eckart, unesco, or nacl")
    returns = _array(
        np.zeros((FLUID_SCALAR_COUNT, base_execution.base_execution.
                  boundary_count)) if fluid_boundary_return_time_s is None
        else fluid_boundary_return_time_s,
        (FLUID_SCALAR_COUNT,
         base_execution.base_execution.boundary_count),
        "fluid boundary return times")
    active = np.asarray((temperature_mode == MODE_TRANSPORTED,
                         salinity_mode == MODE_TRANSPORTED))
    if returns.size and np.any(returns[~active] != 0.0):
        raise TransportedFluidSettlingV92Error(
            "v92 an equilibrium scalar cannot own boundary return memory")
    placeholder = TransportedFluidExecutionV92(
        base_execution, temperature_mode, salinity_mode, temperature,
        salinity, density_eos, returns, SEMANTIC_ID, "")
    return TransportedFluidExecutionV92(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v92(value: TransportedFluidExecutionV92) -> dict:
    if not isinstance(value, TransportedFluidExecutionV92):
        raise TransportedFluidSettlingV92Error("v92 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise TransportedFluidSettlingV92Error(
            "v92 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class AcceptedScalarHydraulicFaceReceiptV92:
    execution_fingerprint: str
    parent_state_sha256: str
    hydraulic_receipt_sha256: str
    parent_q_sha256: str
    endpoint_q_sha256: str
    accepted_water_flux_m2_s: np.ndarray
    accepted_water_flux_sha256: str
    receipt_sha256: str


def _face_receipt_token(value: AcceptedScalarHydraulicFaceReceiptV92) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":accepted-hydraulic-face-flux",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "hydraulic_receipt_sha256": value.hydraulic_receipt_sha256,
        "parent_q_sha256": value.parent_q_sha256,
        "endpoint_q_sha256": value.endpoint_q_sha256,
        "accepted_water_flux_sha256": value.accepted_water_flux_sha256,
        "face_count": int(value.accepted_water_flux_m2_s.size),
        "owner_count": 1,
    })


def make_accepted_scalar_hydraulic_face_receipt_v92(
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92",
    hydraulic_receipt: HydraulicTurbulenceReceiptV91, *,
    accepted_q: object,
    accepted_water_flux_m2_s: object,
) -> AcceptedScalarHydraulicFaceReceiptV92:
    """Bind scalar face fluxes to one accepted v91 hydraulic endpoint."""
    _validate_state(parent, execution)
    if (not isinstance(hydraulic_receipt, HydraulicTurbulenceReceiptV91) or
            hydraulic_receipt.execution_fingerprint !=
                execution.base_execution.fingerprint or
            hydraulic_receipt.parent_state_sha256 !=
                parent.base_state.state_sha256):
        raise TransportedFluidSettlingV92Error(
            "v92 hydraulic face receipt parent differs")
    endpoint = _array(
        accepted_q, (3, execution.cell_count), "accepted hydraulic endpoint")
    parent_q_sha = _array_digest(parent.q)
    endpoint_q_sha = _array_digest(endpoint)
    if (hydraulic_receipt.accepted_q_sha256 != endpoint_q_sha or
            hydraulic_receipt.parent_state_sha256 !=
                parent.base_state.state_sha256):
        raise TransportedFluidSettlingV92Error(
            "v92 hydraulic face receipt endpoint differs")
    flux = _array(
        accepted_water_flux_m2_s, (execution.face_count,),
        "accepted hydraulic face water flux")
    flux_sha = _array_digest(flux)
    placeholder = AcceptedScalarHydraulicFaceReceiptV92(
        execution.fingerprint, parent.state_sha256,
        hydraulic_receipt.receipt_sha256, parent_q_sha, endpoint_q_sha,
        flux, flux_sha, "")
    return AcceptedScalarHydraulicFaceReceiptV92(
        **{**placeholder.__dict__,
           "receipt_sha256": _face_receipt_token(placeholder)})


def validate_accepted_scalar_hydraulic_face_receipt_v92(
    value: AcceptedScalarHydraulicFaceReceiptV92,
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92",
    hydraulic_receipt: HydraulicTurbulenceReceiptV91, *,
    accepted_q: object,
) -> None:
    _validate_state(parent, execution)
    endpoint = _array(
        accepted_q, (3, execution.cell_count), "accepted hydraulic endpoint")
    if (not isinstance(value, AcceptedScalarHydraulicFaceReceiptV92) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.hydraulic_receipt_sha256 !=
                hydraulic_receipt.receipt_sha256 or
            value.parent_q_sha256 != _array_digest(parent.q) or
            value.endpoint_q_sha256 != _array_digest(endpoint) or
            value.accepted_water_flux_m2_s.shape !=
                (execution.face_count,) or
            value.accepted_water_flux_sha256 !=
                _array_digest(value.accepted_water_flux_m2_s) or
            value.receipt_sha256 != _face_receipt_token(value)):
        raise TransportedFluidSettlingV92Error(
            "v92 accepted hydraulic face receipt is stale or mismatched")


# ``fallve`` precedes ``erosed/eqtran`` in locked Delft3D4 FLOW.  The
# suspended diameter accepted at the end of update n therefore drives the
# settling velocity in update n+1.  V92 originally had only the static
# ``D50*FacDSS`` path.  This typed, parent-bound seam keeps that default exact
# while allowing the v94 IopSus owner to provide the previously accepted DSS
# without opening an unauthenticated loose-array input.
SETTLING_DIAMETER_STATIC_PROFILE = "static_profile_d50_times_facdss"
SETTLING_DIAMETER_ACTIVE_IOPSUS_V94 = "active_iopsus_v94_parent_state"
SETTLING_DIAMETER_SOURCE_KINDS = (
    SETTLING_DIAMETER_STATIC_PROFILE,
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
)
SUSPENDED_CAPACITY_LEGACY_INPUT = "legacy_v92_authenticated_input"
SUSPENDED_CAPACITY_ACTIVE_V83_V94 = "active_v83_v94_exact_provider"
SUSPENDED_CAPACITY_SOURCE_KINDS = (
    SUSPENDED_CAPACITY_LEGACY_INPUT,
    SUSPENDED_CAPACITY_ACTIVE_V83_V94,
)


@dataclass(frozen=True)
class SettlingDiameterReceiptV92:
    execution_fingerprint: str
    parent_state_sha256: str
    source_kind: str
    suspended_diameter_m: np.ndarray
    source_state_sha256: str
    receipt_sha256: str


def _settling_diameter_receipt_token(
    value: SettlingDiameterReceiptV92,
) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":settling-diameter-receipt",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "source_kind": value.source_kind,
        "suspended_diameter_sha256": _array_digest(
            value.suspended_diameter_m),
        "source_state_sha256": value.source_state_sha256,
        "lag_policy": "accepted_parent_DSS_drives_current_fallve",
    })


def make_settling_diameter_receipt_v92(
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92", *,
    suspended_diameter_m: object | None = None,
    source_kind: str = SETTLING_DIAMETER_STATIC_PROFILE,
    source_state_sha256: str | None = None,
) -> SettlingDiameterReceiptV92:
    """Bind the current update's settling diameter to its accepted parent.

    The static/default branch is intentionally not configurable: its array is
    reconstructed from the immutable v78 profile.  The active branch accepts
    only a positive array plus the authenticated state digest of its outer
    owner; v94 additionally validates that exact owner identity.
    """
    _validate_state(parent, execution)
    if source_kind not in SETTLING_DIAMETER_SOURCE_KINDS:
        raise TransportedFluidSettlingV92Error(
            "v92 settling-diameter source kind is unsupported")
    profile = execution.base_execution.base_execution.base_execution.profile
    static = np.broadcast_to(
        np.asarray(profile.sediment_d50_m, dtype=np.float64)[:, None] *
        np.asarray(profile.settling_fac_dss, dtype=np.float64)[:, None],
        (execution.fraction_count, execution.cell_count))
    if source_kind == SETTLING_DIAMETER_STATIC_PROFILE:
        if suspended_diameter_m is not None and not np.array_equal(
                np.asarray(suspended_diameter_m), static):
            raise TransportedFluidSettlingV92Error(
                "v92 static settling diameter differs from D50*FacDSS")
        diameter = _array(
            static, (execution.fraction_count, execution.cell_count),
            "static settling diameter", nonnegative=True)
        if np.any(diameter <= 0.0):
            raise TransportedFluidSettlingV92Error(
                "v92 static settling diameter must be positive")
        source_sha = execution.fingerprint
        if source_state_sha256 is not None and \
                source_state_sha256 != source_sha:
            raise TransportedFluidSettlingV92Error(
                "v92 static settling source identity differs")
    else:
        if suspended_diameter_m is None:
            raise TransportedFluidSettlingV92Error(
                "v92 active settling diameter is absent")
        diameter = _array(
            suspended_diameter_m,
            (execution.fraction_count, execution.cell_count),
            "active settling diameter", nonnegative=True)
        if np.any(diameter <= 0.0):
            raise TransportedFluidSettlingV92Error(
                "v92 active settling diameter must be positive")
        source_sha = _sha(source_state_sha256, "settling source state")
    placeholder = SettlingDiameterReceiptV92(
        execution.fingerprint, parent.state_sha256, source_kind, diameter,
        source_sha, "")
    return SettlingDiameterReceiptV92(
        **{**placeholder.__dict__,
           "receipt_sha256": _settling_diameter_receipt_token(placeholder)})


def validate_settling_diameter_receipt_v92(
    value: SettlingDiameterReceiptV92,
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92",
) -> None:
    _validate_state(parent, execution)
    if (not isinstance(value, SettlingDiameterReceiptV92) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.source_kind not in SETTLING_DIAMETER_SOURCE_KINDS or
            value.suspended_diameter_m.shape !=
                (execution.fraction_count, execution.cell_count) or
            not np.isfinite(value.suspended_diameter_m).all() or
            np.any(value.suspended_diameter_m <= 0.0) or
            _SHA256.fullmatch(value.source_state_sha256) is None or
            value.receipt_sha256 !=
                _settling_diameter_receipt_token(value)):
        raise TransportedFluidSettlingV92Error(
            "v92 settling-diameter receipt is stale or mismatched")
    if value.source_kind == SETTLING_DIAMETER_STATIC_PROFILE:
        expected = make_settling_diameter_receipt_v92(execution, parent)
        if value.receipt_sha256 != expected.receipt_sha256:
            raise TransportedFluidSettlingV92Error(
                "v92 static settling-diameter receipt differs")


@dataclass(frozen=True)
class SuspendedCapacityReceiptV92:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_endpoint_sha256: str
    settling_diameter_receipt_sha256: str
    source_kind: str
    source_receipt_sha256: str
    suspended_capacity_kg_m_s: np.ndarray
    receipt_sha256: str


def _suspended_capacity_receipt_token(
    value: SuspendedCapacityReceiptV92,
) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":suspended-capacity-receipt",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_endpoint_sha256": value.accepted_endpoint_sha256,
        "settling_diameter_receipt_sha256":
            value.settling_diameter_receipt_sha256,
        "source_kind": value.source_kind,
        "source_receipt_sha256": value.source_receipt_sha256,
        "capacity_sha256": _array_digest(
            value.suspended_capacity_kg_m_s),
        "consumer_count": 1,
    })


def make_suspended_capacity_receipt_v92(
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92", *,
    accepted_q: object,
    settling_diameter_receipt: SettlingDiameterReceiptV92,
    suspended_capacity_kg_m_s: object,
    source_kind: str,
    source_receipt_sha256: str,
) -> SuspendedCapacityReceiptV92:
    _validate_state(parent, execution)
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    if source_kind not in SUSPENDED_CAPACITY_SOURCE_KINDS:
        raise TransportedFluidSettlingV92Error(
            "v92 suspended-capacity source kind is unsupported")
    endpoint = _array(
        accepted_q, (3, execution.cell_count), "capacity endpoint")
    capacity = _array(
        suspended_capacity_kg_m_s,
        (execution.fraction_count, execution.cell_count),
        "suspended capacity", nonnegative=True)
    source_sha = _sha(source_receipt_sha256, "capacity source receipt")
    placeholder = SuspendedCapacityReceiptV92(
        execution.fingerprint, parent.state_sha256,
        _array_digest(endpoint), settling_diameter_receipt.receipt_sha256,
        source_kind, source_sha, capacity, "")
    return SuspendedCapacityReceiptV92(
        **{**placeholder.__dict__,
           "receipt_sha256": _suspended_capacity_receipt_token(placeholder)})


def validate_suspended_capacity_receipt_v92(
    value: SuspendedCapacityReceiptV92,
    execution: TransportedFluidExecutionV92,
    parent: "TransportedFluidStateV92", *, accepted_q: object,
    settling_diameter_receipt: SettlingDiameterReceiptV92,
) -> None:
    _validate_state(parent, execution)
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    endpoint = _array(
        accepted_q, (3, execution.cell_count), "capacity endpoint")
    if (not isinstance(value, SuspendedCapacityReceiptV92) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_state_sha256 != parent.state_sha256 or
            value.accepted_endpoint_sha256 != _array_digest(endpoint) or
            value.settling_diameter_receipt_sha256 !=
                settling_diameter_receipt.receipt_sha256 or
            value.source_kind not in SUSPENDED_CAPACITY_SOURCE_KINDS or
            _SHA256.fullmatch(value.source_receipt_sha256) is None or
            value.suspended_capacity_kg_m_s.shape !=
                (execution.fraction_count, execution.cell_count) or
            not np.isfinite(value.suspended_capacity_kg_m_s).all() or
            np.any(value.suspended_capacity_kg_m_s < 0.0) or
            value.receipt_sha256 !=
                _suspended_capacity_receipt_token(value)):
        raise TransportedFluidSettlingV92Error(
            "v92 suspended-capacity receipt is stale or mismatched")


def _properties(execution: TransportedFluidExecutionV92,
                temperature: np.ndarray,
                salinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = execution.base_execution.base_execution.base_execution.profile
    density = resolve_water_density_v88(
        temperature, salinity, eos=execution.density_eos,
        constituents_active=execution.constituents_active,
        reference_water_density_kg_m3=profile.water_density_kg_m3)
    viscosity = np.ascontiguousarray(
        4.0e-5/(20.0+temperature), dtype=np.float64)
    if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
        raise TransportedFluidSettlingV92Error(
            "v92 molecular viscosity is invalid")
    density.setflags(write=False)
    viscosity.setflags(write=False)
    return density, viscosity


@dataclass(frozen=True)
class TransportedFluidStateV92:
    execution_fingerprint: str
    base_state: SharedTurbulenceStateV91
    h_temperature_c_m: np.ndarray
    h_salinity_ppt_m: np.ndarray
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    boundary_last_outflow_fluid: np.ndarray
    boundary_return_clock_s: np.ndarray
    reference_scalar_integral: np.ndarray
    advective_boundary_ledger: np.ndarray
    diffusive_boundary_ledger: np.ndarray
    wetdry_scalar_ledger: np.ndarray
    accepted_receipt_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s

    @property
    def q(self) -> np.ndarray:
        return self.base_state.q

    @property
    def suspended_state(self):
        return self.base_state.suspended_state


def _state_token(value: TransportedFluidStateV92) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.h_temperature_c_m, value.h_salinity_ppt_m,
            value.temperature_cell_c, value.salinity_cell_ppt,
            value.water_density_cell_kg_m3,
            value.molecular_viscosity_cell_m2_s,
            value.boundary_last_outflow_fluid,
            value.boundary_return_clock_s,
            value.reference_scalar_integral,
            value.advective_boundary_ledger,
            value.diffusive_boundary_ledger,
            value.wetdry_scalar_ledger,
        )],
        "accepted_receipt_chain_sha256":
            value.accepted_receipt_chain_sha256,
    })


def _derive_concentrations(
    execution: TransportedFluidExecutionV92,
    base_state: SharedTurbulenceStateV91,
    h_temperature: np.ndarray,
    h_salinity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(base_state.q[0])
    dry = execution.base_execution.base_execution.base_execution.dry_tolerance_m
    wet = depth > dry
    moving_wetdry = (
        execution.base_execution.state_domain_mode ==
        STATE_DOMAIN_MOVING_WETDRY_V97)
    if not moving_wetdry and not np.all(wet):
        raise TransportedFluidSettlingV92Error(
            "v92 composed state retains v91's fully-wet scope; moving wet/dry is v97")
    if moving_wetdry and (
            np.any(h_temperature[~wet] != 0.0) or
            np.any(h_salinity[~wet] != 0.0)):
        raise TransportedFluidSettlingV92Error(
            "v97 dry cells require exact-zero conserved fluid scalars")
    temperature = np.full(
        execution.cell_count, execution.equilibrium_temperature_c,
        dtype=np.float64)
    salinity = np.full(
        execution.cell_count, execution.equilibrium_salinity_ppt,
        dtype=np.float64)
    if execution.temperature_mode == MODE_TRANSPORTED:
        temperature[wet] = h_temperature[wet]/depth[wet]
    if execution.salinity_mode == MODE_TRANSPORTED:
        salinity[wet] = h_salinity[wet]/depth[wet]
    if np.any(temperature <= TEMPERATURE_FLOOR_C):
        raise TransportedFluidSettlingV92Error(
            "v92 transported temperature must exceed -20 C")
    tolerance = 4096.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.abs(salinity))
    if np.any(salinity < -tolerance):
        raise TransportedFluidSettlingV92Error(
            "v92 transported salinity must be nonnegative")
    salinity = np.maximum(salinity, 0.0)
    return np.ascontiguousarray(temperature), np.ascontiguousarray(salinity)


def _validate_boundary_memory(
    execution: TransportedFluidExecutionV92,
    last: np.ndarray,
    clock: np.ndarray,
) -> None:
    base = execution.base_execution.base_execution.base_execution
    open_face = base.boundary_index >= 0
    if (np.any(last[:, ~open_face] != 0.0) or
            np.any(clock[:, ~open_face] != 0.0)):
        raise TransportedFluidSettlingV92Error(
            "v92 fluid boundary memory exists on a non-open face")
    expanded = np.zeros((FLUID_SCALAR_COUNT, execution.face_count))
    if execution.boundary_count:
        expanded[:, open_face] = execution.fluid_boundary_return_time_s[
            :, base.boundary_index[open_face]]
    active = np.asarray(execution.transported_mask)[:, None]
    tolerance = 256.0*np.finfo(np.float64).eps*np.maximum(1.0, expanded)
    if (np.any(clock < 0.0) or
            np.any(clock[expanded <= 0.0] != 0.0) or
            np.any((expanded > 0.0) & (clock > expanded+tolerance)) or
            np.any(last[~np.broadcast_to(active, last.shape)] != 0.0) or
            np.any(clock[~np.broadcast_to(active, clock.shape)] != 0.0)):
        raise TransportedFluidSettlingV92Error(
            "v92 fluid boundary memory is outside its authenticated mode")
    # Public/checkpoint memory is physical [temperature_C, salinity_ppt].
    if (np.any(last[FLUID_TEMPERATURE] <= TEMPERATURE_FLOOR_C) or
            np.any(last[FLUID_SALINITY] < 0.0)):
        raise TransportedFluidSettlingV92Error(
            "v92 physical fluid boundary memory is outside T/S bounds")


def _wrap_state(
    execution: TransportedFluidExecutionV92,
    base_state: SharedTurbulenceStateV91, *,
    h_temperature_c_m: object,
    h_salinity_ppt_m: object,
    boundary_last_outflow_fluid: object,
    boundary_return_clock_s: object,
    reference_scalar_integral: object,
    advective_boundary_ledger: object,
    diffusive_boundary_ledger: object,
    wetdry_scalar_ledger: object,
    accepted_receipt_chain_sha256: str,
    temperature_cell_c: object | None = None,
    salinity_cell_ppt: object | None = None,
    water_density_cell_kg_m3: object | None = None,
    molecular_viscosity_cell_m2_s: object | None = None,
) -> TransportedFluidStateV92:
    execution_document_v92(execution)
    validate_shared_turbulence_state_v91(base_state, execution.base_execution)
    nc, nf, nb = (execution.cell_count, execution.face_count,
                  execution.boundary_count)
    ht = _array(h_temperature_c_m, (nc,), "hT")
    hs = _array(h_salinity_ppt_m, (nc,), "hS")
    last = _array(boundary_last_outflow_fluid,
                  (FLUID_SCALAR_COUNT, nf), "fluid boundary last outflow")
    clock = _array(boundary_return_clock_s, (FLUID_SCALAR_COUNT, nf),
                   "fluid boundary return clock", nonnegative=True)
    reference = _array(reference_scalar_integral,
                       (FLUID_SCALAR_COUNT,), "reference scalar integral")
    advective = _array(advective_boundary_ledger,
                       (FLUID_SCALAR_COUNT, nb),
                       "fluid advective boundary ledger")
    diffusive = _array(diffusive_boundary_ledger,
                       (FLUID_SCALAR_COUNT, nb),
                       "fluid diffusive boundary ledger")
    wetdry = _array(wetdry_scalar_ledger, (FLUID_SCALAR_COUNT,),
                    "fluid wet/dry ledger")
    _validate_boundary_memory(execution, last, clock)
    expected_t, expected_s = _derive_concentrations(
        execution, base_state, ht, hs)
    t = expected_t if temperature_cell_c is None else _array(
        temperature_cell_c, (nc,), "accepted temperature")
    s = expected_s if salinity_cell_ppt is None else _array(
        salinity_cell_ppt, (nc,), "accepted salinity", nonnegative=True)
    if not np.array_equal(t, expected_t) or not np.array_equal(s, expected_s):
        raise TransportedFluidSettlingV92Error(
            "v92 accepted concentration fields differ from conserved hT/hS")
    expected_rho, expected_nu = _properties(execution, t, s)
    rho = expected_rho if water_density_cell_kg_m3 is None else _array(
        water_density_cell_kg_m3, (nc,), "water density", nonnegative=True)
    nu = expected_nu if molecular_viscosity_cell_m2_s is None else _array(
        molecular_viscosity_cell_m2_s, (nc,), "molecular viscosity",
        nonnegative=True)
    if not np.array_equal(rho, expected_rho) or not np.array_equal(
            nu, expected_nu):
        raise TransportedFluidSettlingV92Error(
            "v92 derived fluid properties differ from accepted T/S")
    area = execution.base_execution.base_execution.base_execution.cell_area_m2
    current = np.array((np.sum(ht*area, dtype=np.float64),
                        np.sum(hs*area, dtype=np.float64)))
    boundary = np.sum(advective+diffusive, axis=1)
    active = np.asarray(execution.transported_mask)
    scale = np.maximum(1.0, np.abs(reference)+np.abs(boundary)+np.abs(wetdry))
    residual = current-reference-boundary-wetdry
    if np.any(np.abs(residual[active]) >
              131072.0*np.finfo(np.float64).eps*scale[active]):
        raise TransportedFluidSettlingV92Error(
            "v92 transported hT/hS ledger did not close")
    chain = _sha(accepted_receipt_chain_sha256, "accepted receipt chain")
    placeholder = TransportedFluidStateV92(
        execution.fingerprint, base_state, ht, hs, t, s, rho, nu,
        last, clock, reference, advective, diffusive, wetdry, chain, "")
    return TransportedFluidStateV92(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_transported_fluid_state_v92(
    execution: TransportedFluidExecutionV92, *,
    base_state: SharedTurbulenceStateV91,
    temperature_cell_c: object | None = None,
    salinity_cell_ppt: object | None = None,
) -> TransportedFluidStateV92:
    validate_shared_turbulence_state_v91(base_state, execution.base_execution)
    nc, nf, nb = execution.cell_count, execution.face_count, execution.boundary_count
    if execution.temperature_mode == MODE_TRANSPORTED:
        if temperature_cell_c is None:
            raise TransportedFluidSettlingV92Error(
                "v92 transported temperature initial field is required")
        temperature = _array(
            temperature_cell_c, (nc,), "initial temperature")
        if np.any(temperature <= TEMPERATURE_FLOOR_C):
            raise TransportedFluidSettlingV92Error(
                "v92 transported temperature must exceed -20 C")
    else:
        if temperature_cell_c is not None:
            raise TransportedFluidSettlingV92Error(
                "v92 temperature field supplied for equilibrium mode")
        temperature = np.full(nc, execution.equilibrium_temperature_c)
    if execution.salinity_mode == MODE_TRANSPORTED:
        if salinity_cell_ppt is None:
            raise TransportedFluidSettlingV92Error(
                "v92 transported salinity initial field is required")
        salinity = _array(
            salinity_cell_ppt, (nc,), "initial salinity", nonnegative=True)
    else:
        if salinity_cell_ppt is not None:
            raise TransportedFluidSettlingV92Error(
                "v92 salinity field supplied for equilibrium mode")
        salinity = np.full(nc, execution.equilibrium_salinity_ppt)
    depth = np.asarray(base_state.q[0])
    if (execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97):
        wet = depth > execution.base_execution.base_execution.\
            base_execution.dry_tolerance_m
        ht = np.where(wet, depth*temperature, 0.0)
        hs = np.where(wet, depth*salinity, 0.0)
    else:
        ht = depth*temperature
        hs = depth*salinity
    area = execution.base_execution.base_execution.base_execution.cell_area_m2
    reference = np.array((np.sum(ht*area, dtype=np.float64),
                          np.sum(hs*area, dtype=np.float64)))
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "base_state_sha256": base_state.state_sha256,
        "hT_sha256": _array_digest(ht),
        "hS_sha256": _array_digest(hs),
    })
    return _wrap_state(
        execution, base_state, h_temperature_c_m=ht,
        h_salinity_ppt_m=hs,
        boundary_last_outflow_fluid=np.zeros((2, nf)),
        boundary_return_clock_s=np.zeros((2, nf)),
        reference_scalar_integral=reference,
        advective_boundary_ledger=np.zeros((2, nb)),
        diffusive_boundary_ledger=np.zeros((2, nb)),
        wetdry_scalar_ledger=np.zeros(2),
        accepted_receipt_chain_sha256=chain)


def _validate_state(value: TransportedFluidStateV92,
                    execution: TransportedFluidExecutionV92) -> None:
    if (not isinstance(value, TransportedFluidStateV92) or
            value.execution_fingerprint != execution.fingerprint or
            value.state_sha256 != _state_token(value)):
        raise TransportedFluidSettlingV92Error(
            "v92 state is stale, mutated or execution-mismatched")
    _wrap_state(
        execution, value.base_state,
        h_temperature_c_m=value.h_temperature_c_m,
        h_salinity_ppt_m=value.h_salinity_ppt_m,
        boundary_last_outflow_fluid=value.boundary_last_outflow_fluid,
        boundary_return_clock_s=value.boundary_return_clock_s,
        reference_scalar_integral=value.reference_scalar_integral,
        advective_boundary_ledger=value.advective_boundary_ledger,
        diffusive_boundary_ledger=value.diffusive_boundary_ledger,
        wetdry_scalar_ledger=value.wetdry_scalar_ledger,
        accepted_receipt_chain_sha256=
            value.accepted_receipt_chain_sha256,
        temperature_cell_c=value.temperature_cell_c,
        salinity_cell_ppt=value.salinity_cell_ppt,
        water_density_cell_kg_m3=value.water_density_cell_kg_m3,
        molecular_viscosity_cell_m2_s=
            value.molecular_viscosity_cell_m2_s)


def _fluid_fct_active_lanes_v92(
    topology, mass: np.ndarray, depth: np.ndarray, next_depth: np.ndarray,
    water_flux: np.ndarray, left: np.ndarray, right: np.ndarray,
    boundary: np.ndarray, hydro_constant: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lane-count-independent transcription of the accepted v76 FCT.

    ``topology.fraction_count`` is deliberately ignored.  Fluid constituent
    lanes and sediment fractions are independent inventories.
    """
    nl, nc = mass.shape
    nf, nb = topology.face_count, topology.boundary_count
    if left.shape != (nl, nf) or right.shape != (nl, nf) or \
            boundary.shape != (nl, nf):
        raise TransportedFluidSettlingV92Error(
            "v92 compact fluid FCT lane shapes differ")
    concentration = np.where(
        depth[None, :] > topology.dry_tolerance_m,
        mass/np.where(depth[None, :] > topology.dry_tolerance_m,
                      depth[None, :], 1.0), 0.0)
    low_flux = np.zeros((nl, nf), dtype=np.float64)
    anti = np.zeros_like(low_flux)
    for face in range(nf):
        owner = int(topology.owner[face])
        neighbour = int(topology.neighbour[face])
        boundary_id = int(topology.boundary_index[face])
        water = water_flux[face]
        if neighbour < 0:
            if boundary_id < 0:
                if water != 0.0:
                    raise TransportedFluidSettlingV92Error(
                        "v92 closed face has nonzero accepted water flux")
                continue
            low_flux[:, face] = (
                water*boundary[:, face] if water < 0.0
                else water*concentration[:, owner])
        elif water >= 0.0:
            low_flux[:, face] = water*concentration[:, owner]
            anti[:, face] = water*left[:, face]-low_flux[:, face]
        else:
            low_flux[:, face] = water*concentration[:, neighbour]
            anti[:, face] = water*right[:, face]-low_flux[:, face]
    low_state = np.array(mass, copy=True)
    rplus = np.ones_like(mass)
    rminus = np.ones_like(mass)
    for lane in range(nl):
        for cell in range(nc):
            low = mass[lane, cell]
            lower = upper = concentration[lane, cell]
            pp = pm = 0.0
            for slot in range(topology.cell_face_offset[cell],
                              topology.cell_face_offset[cell+1]):
                face = int(topology.cell_faces[slot])
                sign = topology.cell_face_sign[slot]
                adjacent = (int(topology.neighbour[face]) if sign > 0 else
                            int(topology.owner[face]))
                if adjacent >= 0:
                    value = concentration[lane, adjacent]
                    lower, upper = min(lower, value), max(upper, value)
                elif (topology.boundary_index[face] >= 0 and
                      water_flux[face] < 0.0):
                    value = boundary[lane, face]
                    lower, upper = min(lower, value), max(upper, value)
                z = (-dt*sign*topology.face_length_m[face]/
                     topology.cell_area_m2[cell])
                correction = z*anti[lane, face]
                low += z*low_flux[lane, face]
                pp += max(correction, 0.0)
                pm += min(correction, 0.0)
            tolerance = 4096.0*np.finfo(float).eps*max(
                1.0, abs(mass[lane, cell]))
            if low < -tolerance or not math.isfinite(low):
                raise TransportedFluidSettlingV92Error(
                    "v92 accepted water flux overdraws low-order fluid scalar")
            low = max(0.0, low)
            lower, upper = _smooth_bound(
                topology, concentration, hydro_constant, lane, cell,
                lower, upper)
            low_state[lane, cell] = low
            if pp > 0.0:
                rplus[lane, cell] = min(
                    1.0, max(0.0, upper*next_depth[cell]-low-
                             64.0*np.finfo(float).tiny)/pp)
                rplus[lane, cell] *= 1.0-32.0*np.finfo(float).eps
            if pm < 0.0:
                rminus[lane, cell] = min(
                    1.0, max(0.0, low-lower*next_depth[cell]-
                             64.0*np.finfo(float).tiny)/(-pm))
                rminus[lane, cell] *= 1.0-32.0*np.finfo(float).eps
    scaled = np.zeros_like(anti)
    for lane in range(nl):
        for face in range(nf):
            owner = int(topology.owner[face])
            neighbour = int(topology.neighbour[face])
            correction = anti[lane, face]
            if correction > 0.0:
                factor = rminus[lane, owner]
                if neighbour >= 0:
                    factor = min(factor, rplus[lane, neighbour])
            elif correction < 0.0:
                factor = rplus[lane, owner]
                if neighbour >= 0:
                    factor = min(factor, rminus[lane, neighbour])
            else:
                factor = 1.0
            scaled[lane, face] = factor*correction
    flux = low_flux+scaled
    result = np.array(low_state, copy=True)
    for cell in range(nc):
        for slot in range(topology.cell_face_offset[cell],
                          topology.cell_face_offset[cell+1]):
            face = int(topology.cell_faces[slot])
            result[:, cell] -= (
                dt*topology.cell_face_sign[slot]*
                topology.face_length_m[face]*scaled[:, face]/
                topology.cell_area_m2[cell])
    tolerance = 4096.0*np.finfo(float).eps*np.maximum(1.0, np.abs(mass))
    if np.any(result < -tolerance) or not np.isfinite(result).all():
        raise TransportedFluidSettlingV92Error(
            "v92 fluid FCT result is invalid")
    result = np.maximum(result, 0.0)
    ledger = np.zeros((nl, nb))
    for face in range(nf):
        boundary_id = int(topology.boundary_index[face])
        if boundary_id >= 0:
            ledger[:, boundary_id] -= (
                dt*topology.face_length_m[face]*flux[:, face])
    return result, flux, ledger


def _fluid_diffusion_active_lanes_v92(
    topology, dicuv_cell_m2_s: np.ndarray, mass: np.ndarray,
    depth: np.ndarray, boundary: np.ndarray, exchange_area: np.ndarray,
    face_active: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lane-count-independent shared ``dicuv/0.7`` fluid diffusion."""
    nl, nc = mass.shape
    nf, nb = topology.face_count, topology.boundary_count
    face_amount = np.zeros((nl, nf))
    effective = np.zeros(nf)
    for face in range(nf):
        owner = int(topology.owner[face])
        neighbour = int(topology.neighbour[face])
        boundary_id = int(topology.boundary_index[face])
        if neighbour < 0 and boundary_id < 0:
            if face_active[face]:
                raise TransportedFluidSettlingV92Error(
                    "v92 closed face cannot be fluid-diffusion active")
            continue
        left_depth = depth[owner]
        right_depth = depth[neighbour] if neighbour >= 0 else left_depth
        left = (mass[:, owner]/left_depth if left_depth > 0.0
                else np.zeros(nl))
        if neighbour >= 0:
            right = (mass[:, neighbour]/right_depth if right_depth > 0.0
                     else np.zeros(nl))
            right_dicuv = dicuv_cell_m2_s[neighbour]
        else:
            right = boundary[:, face]
            right_dicuv = dicuv_cell_m2_s[owner]
        oracle = flow2dh_horizontal_diffusive_face_mass(
            left, right, dicuv_cell_m2_s[owner], right_dicuv,
            exchange_area[face], topology.centre_distance_m[face],
            left_depth, right_depth, dt_s=dt,
            dry_threshold_m=topology.profile.diffusion_dry_threshold_m,
            transport_face_active=bool(face_active[face]))
        face_amount[:, face] = oracle.mass_left_to_right_kg
        if face_active[face]:
            effective[face] = float(np.asarray(
                oracle.effective_diffusivity_m2_s).reshape(-1)[0])
    theta = np.ones((nl, nc))
    margin = 1.0-32.0*np.finfo(float).eps
    for lane in range(nl):
        for cell in range(nc):
            outgoing = 0.0
            for slot in range(topology.cell_face_offset[cell],
                              topology.cell_face_offset[cell+1]):
                face = int(topology.cell_faces[slot])
                outgoing += max(
                    topology.cell_face_sign[slot]*face_amount[lane, face],
                    0.0)
            inventory = mass[lane, cell]*topology.cell_area_m2[cell]
            if outgoing > 0.0 and outgoing >= inventory:
                theta[lane, cell] = min(1.0, inventory/outgoing)*margin
    for lane in range(nl):
        for face in range(nf):
            value = face_amount[lane, face]
            donor = (int(topology.owner[face]) if value > 0.0 else
                     int(topology.neighbour[face]) if value < 0.0 else -1)
            if donor >= 0:
                face_amount[lane, face] *= theta[lane, donor]
    result = np.array(mass, copy=True)
    for cell in range(nc):
        for slot in range(topology.cell_face_offset[cell],
                          topology.cell_face_offset[cell+1]):
            face = int(topology.cell_faces[slot])
            result[:, cell] -= (
                topology.cell_face_sign[slot]*face_amount[:, face]/
                topology.cell_area_m2[cell])
    tolerance = 8192.0*np.finfo(float).eps*np.maximum(1.0, np.abs(mass))
    if np.any(result < -tolerance) or not np.isfinite(result).all():
        raise TransportedFluidSettlingV92Error(
            "v92 fluid diffusion result is invalid")
    result = np.maximum(result, 0.0)
    ledger = np.zeros((nl, nb))
    for face in range(nf):
        boundary_id = int(topology.boundary_index[face])
        if boundary_id >= 0:
            ledger[:, boundary_id] -= face_amount[:, face]
    return result, face_amount, ledger, effective


@dataclass(frozen=True)
class FluidScalarTransportResultV92:
    h_temperature_c_m: np.ndarray
    h_salinity_ppt_m: np.ndarray
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    boundary_transport_scalar: np.ndarray
    boundary_last_outflow_fluid: np.ndarray
    boundary_return_clock_s: np.ndarray
    advective_face_flux: np.ndarray
    diffusive_face_amount: np.ndarray
    advective_boundary_ledger: np.ndarray
    diffusive_boundary_ledger: np.ndarray
    wetdry_scalar_ledger: np.ndarray
    effective_horizontal_diffusivity_m2_s: np.ndarray
    water_continuity_residual_m: np.ndarray
    lane_application_count: np.ndarray
    result_sha256: str


def _fluid_result_token(value: FluidScalarTransportResultV92) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":fluid-scalar-result",
        "arrays_sha256": [_array_digest(array) for array in (
            value.h_temperature_c_m, value.h_salinity_ppt_m,
            value.temperature_cell_c, value.salinity_cell_ppt,
            value.boundary_transport_scalar,
            value.boundary_last_outflow_fluid,
            value.boundary_return_clock_s, value.advective_face_flux,
            value.diffusive_face_amount, value.advective_boundary_ledger,
            value.diffusive_boundary_ledger, value.wetdry_scalar_ledger,
            value.effective_horizontal_diffusivity_m2_s,
            value.water_continuity_residual_m,
            value.lane_application_count,
        )],
        "temperature_floor_c": TEMPERATURE_FLOOR_C,
        "scalar_order": list(FLUID_SCALAR_ORDER),
    })


def _transport_fluid_scalars_cpu_v92(
    execution: TransportedFluidExecutionV92,
    scalar_execution: SuspendedSupportExecutionV82, *,
    parent_h_temperature_c_m: object,
    parent_h_salinity_ppt_m: object,
    parent_temperature_c: object,
    parent_salinity_ppt: object,
    previous_boundary_last_outflow_fluid: object,
    previous_boundary_return_clock_s: object,
    depth_before_m: object,
    depth_after_m: object,
    accepted_water_flux_m2_s: object,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    accepted_face_exchange_area_m2: object,
    transport_face_active: object,
    hydro_constant: object,
    dt_s: float,
) -> FluidScalarTransportResultV92:
    """Exercise fluid lanes in isolation for bounded diagnostic tests only.

    This private helper is deliberately *not* an admitted transaction path:
    production v92 concatenates sediment and active fluid lanes in
    :func:`_suspended_dynamic_trial_cpu` and invokes exactly one shared
    FCT/diffusion owner.  Keeping the isolated oracle makes dry cleanup and
    affine-ledger identities testable without pretending there is a second
    lifecycle owner.

    ``reconstructed_*_fluid`` and ``prescribed_boundary_fluid`` use physical
    ``[temperature_C, salinity_ppt]`` order.  The returned face flux/amount and
    boundary ledgers use physical ``[hT, hS]`` units.
    """
    execution_document_v92(execution)
    if scalar_execution.base_execution is not execution.base_execution.base_execution:
        # Identity, not equality, is intentional: the one v78 state owner must
        # not be replaced by a look-alike execution object.
        raise TransportedFluidSettlingV92Error(
            "v92 scalar execution does not own the canonical v78 execution")
    base = scalar_execution.base_execution.base_execution
    nc, nf, nb = execution.cell_count, execution.face_count, execution.boundary_count
    dt = _finite(dt_s, "fluid scalar timestep", positive=True)
    h0 = _mutable_array(depth_before_m, (nc,), "depth before",
                        nonnegative=True)
    h1 = _mutable_array(depth_after_m, (nc,), "depth after",
                        nonnegative=True)
    water = _mutable_array(
        accepted_water_flux_m2_s, (nf,), "accepted water flux")
    ht0 = _mutable_array(
        parent_h_temperature_c_m, (nc,), "parent hT")
    hs0 = _mutable_array(
        parent_h_salinity_ppt_m, (nc,), "parent hS")
    t0 = _mutable_array(parent_temperature_c, (nc,), "parent temperature")
    s0 = _mutable_array(parent_salinity_ppt, (nc,), "parent salinity",
                        nonnegative=True)
    left_physical = _mutable_array(
        reconstructed_left_fluid, (2, nf), "left fluid reconstruction")
    right_physical = _mutable_array(
        reconstructed_right_fluid, (2, nf), "right fluid reconstruction")
    prescribed_physical = _mutable_array(
        prescribed_boundary_fluid, (2, nb), "prescribed boundary fluid")
    if (np.any(t0 <= TEMPERATURE_FLOOR_C) or
            np.any(left_physical[0] <= TEMPERATURE_FLOOR_C) or
            np.any(right_physical[0] <= TEMPERATURE_FLOOR_C) or
            (nb and np.any(prescribed_physical[0] <= TEMPERATURE_FLOOR_C)) or
            np.any(left_physical[1] < 0.0) or
            np.any(right_physical[1] < 0.0) or
            (nb and np.any(prescribed_physical[1] < 0.0))):
        raise TransportedFluidSettlingV92Error(
            "v92 reconstructed/prescribed fluid fields are outside bounds")
    face_area = _mutable_array(
        accepted_face_exchange_area_m2, (nf,), "face exchange area",
        nonnegative=True)
    face_active = _mutable_array(
        transport_face_active, (nf,), "transport face active", boolean=True)
    constant = _mutable_array(
        hydro_constant, (nc,), "hydro constant", boolean=True)
    previous_last = _mutable_array(
        previous_boundary_last_outflow_fluid, (2, nf),
        "previous fluid boundary last outflow")
    previous_clock = _mutable_array(
        previous_boundary_return_clock_s, (2, nf),
        "previous fluid boundary clock", nonnegative=True)
    _validate_boundary_memory(execution, previous_last, previous_clock)

    # The affine lane must be the exact image of physical hT at the accepted
    # parent.  This catches stale depth or endpoint fields before transport.
    affine0 = ht0-TEMPERATURE_FLOOR_C*h0
    expected_affine0 = h0*(t0-TEMPERATURE_FLOOR_C)
    tolerance = 8192.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.abs(expected_affine0))
    moving_wetdry = (
        execution.base_execution.state_domain_mode ==
        STATE_DOMAIN_MOVING_WETDRY_V97)
    wet0 = h0 > base.dry_tolerance_m
    relation_cells = wet0 if moving_wetdry else np.ones(nc, dtype=bool)
    if (np.any(np.abs(
            affine0[relation_cells]-expected_affine0[relation_cells]) >
            tolerance[relation_cells]) or
            np.any(affine0 < -tolerance) or
            np.any(np.abs(
                hs0[relation_cells]-h0[relation_cells]*s0[relation_cells]) >
                tolerance[relation_cells]) or
            (moving_wetdry and
             (np.any(ht0[~wet0] != 0.0) or
              np.any(hs0[~wet0] != 0.0)))):
        raise TransportedFluidSettlingV92Error(
            "v92 parent hT/hS and accepted concentrations differ")
    mass0_all = np.vstack((np.maximum(affine0, 0.0), hs0))
    left_all = np.array(left_physical, copy=True)
    right_all = np.array(right_physical, copy=True)
    prescribed_all = np.array(prescribed_physical, copy=True)
    left_all[0] -= TEMPERATURE_FLOOR_C
    right_all[0] -= TEMPERATURE_FLOOR_C
    prescribed_all[0] -= TEMPERATURE_FLOOR_C

    # Exact discrete water continuity is the proof that affine temperature
    # transport and physical hT transport are equivalent.
    water_divergence = np.zeros(nc)
    for cell in range(nc):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            water_divergence[cell] += (
                base.cell_face_sign[slot]*base.face_length_m[face]*water[face]/
                base.cell_area_m2[cell])
    continuity = h1-h0+dt*water_divergence
    continuity_scale = np.maximum(1.0, np.maximum(np.abs(h0), np.abs(h1)))
    if (execution.temperature_mode == MODE_TRANSPORTED and
            np.any(np.abs(continuity) >
                   32768.0*np.finfo(np.float64).eps*continuity_scale)):
        raise TransportedFluidSettlingV92Error(
            "v92 affine temperature requires accepted discrete water continuity")

    active = np.asarray(execution.transported_mask)
    active_lanes = np.flatnonzero(active)
    lane_count = active_lanes.size
    mass0 = mass0_all[active_lanes]
    left = left_all[active_lanes]
    right = right_all[active_lanes]
    prescribed = prescribed_all[active_lanes]
    interior = np.zeros((lane_count, nc))
    wet0 = h0 > base.dry_tolerance_m
    if lane_count:
        interior[:, wet0] = mass0[:, wet0]/h0[wet0]
    boundary = np.zeros((lane_count, nf))
    next_last = np.array(previous_last, copy=True)
    next_clock = np.array(previous_clock, copy=True)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id < 0 or not lane_count:
            continue
        owner = int(base.owner[face])
        inward_q = -water[face]*base.face_length_m[face]
        previous_affine_all = np.array(previous_last[:, face], copy=True)
        previous_affine_all[FLUID_TEMPERATURE] -= TEMPERATURE_FLOOR_C
        resolved = resolve_flow2dh_open_boundary(
            prescribed[:, boundary_id], interior[:, owner],
            np.full(lane_count, inward_q), dt_s=dt,
            neumann=np.zeros(lane_count, dtype=bool),
            return_time_s=execution.fluid_boundary_return_time_s[
                active_lanes, boundary_id],
            previous_memory=Flow2DHBoundaryMemory(
                previous_affine_all[active_lanes],
                previous_clock[active_lanes, face]))
        boundary[:, face] = resolved.boundary_concentration_kg_m3
        next_values = resolved.memory.last_outflow_concentration_kg_m3
        for compact, lane in enumerate(active_lanes):
            next_last[lane, face] = next_values[compact]
            if lane == FLUID_TEMPERATURE:
                next_last[lane, face] += TEMPERATURE_FLOOR_C
            next_clock[lane, face] = resolved.memory.return_clock_s[compact]
    next_last[~active] = 0.0
    next_clock[~active] = 0.0

    if lane_count:
        advected, affine_flux, affine_advective = \
            _fluid_fct_active_lanes_v92(
                base, mass0, h0, h1, water, left, right, boundary,
                constant, dt)
        diffused, affine_face_mass, affine_diffusive, effective = \
            _fluid_diffusion_active_lanes_v92(
                base, scalar_execution.dicuv_cell_m2_s, advected, h1,
                boundary, face_area, face_active, dt)
    else:
        diffused = np.zeros((0, nc))
        affine_flux = np.zeros((0, nf))
        affine_face_mass = np.zeros((0, nf))
        affine_advective = np.zeros((0, nb))
        affine_diffusive = np.zeros((0, nb))
        effective = np.zeros(nf)
    wet1 = h1 > base.dry_tolerance_m
    cleanup = np.where(wet1[None, :], 0.0, diffused)
    final_affine = diffused-cleanup

    ht1 = h1*execution.equilibrium_temperature_c
    hs1 = h1*execution.equilibrium_salinity_ppt
    temperature = np.full(nc, execution.equilibrium_temperature_c)
    salinity = np.full(nc, execution.equilibrium_salinity_ppt)
    lane_position = {int(lane): position
                     for position, lane in enumerate(active_lanes)}
    if active[FLUID_TEMPERATURE]:
        position = lane_position[FLUID_TEMPERATURE]
        ht1 = final_affine[position]+TEMPERATURE_FLOOR_C*h1
        temperature[wet1] = ht1[wet1]/h1[wet1]
    if active[FLUID_SALINITY]:
        position = lane_position[FLUID_SALINITY]
        hs1 = final_affine[position]
        salinity[wet1] = hs1[wet1]/h1[wet1]
    # The accepted moving-wet/dry state owns no depth-integrated fluid
    # constituent in cells at or below the dry tolerance.  The affine
    # temperature correction is already represented exactly in the cleanup
    # ledger below; retaining ``T_floor*h`` here would create a shadow scalar
    # inventory in a hydraulically dry cell.
    ht1 = np.where(wet1, ht1, 0.0)
    hs1 = np.where(wet1, hs1, 0.0)
    if np.any(temperature[wet1] <= TEMPERATURE_FLOOR_C) or \
            np.any(salinity[wet1] < -8192.0*np.finfo(float).eps):
        raise TransportedFluidSettlingV92Error(
            "v92 fluid scalar transport produced an invalid endpoint")
    salinity = np.maximum(salinity, 0.0)

    physical_flux = np.zeros((2, nf))
    physical_face_mass = np.zeros((2, nf))
    physical_advective = np.zeros((2, nb))
    physical_diffusive = np.zeros((2, nb))
    physical_cleanup = np.zeros(2)
    boundary_physical = np.vstack((
        np.full(nf, execution.equilibrium_temperature_c),
        np.full(nf, execution.equilibrium_salinity_ppt),
    ))
    lane_application_count = np.zeros(2, dtype=np.int32)
    water_boundary = np.zeros(nb)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id >= 0:
            water_boundary[boundary_id] -= (
                dt*base.face_length_m[face]*water[face])
    for compact, lane in enumerate(active_lanes):
        physical_flux[lane] = affine_flux[compact]
        physical_face_mass[lane] = affine_face_mass[compact]
        physical_advective[lane] = affine_advective[compact]
        physical_diffusive[lane] = affine_diffusive[compact]
        physical_cleanup[lane] = -np.sum(
            cleanup[compact]*base.cell_area_m2)
        boundary_physical[lane] = boundary[compact]
        lane_application_count[lane] = 1
        if lane == FLUID_TEMPERATURE:
            physical_flux[lane] += TEMPERATURE_FLOOR_C*water
            physical_advective[lane] += \
                TEMPERATURE_FLOOR_C*water_boundary
            boundary_physical[lane] += TEMPERATURE_FLOOR_C
            # On a genuinely dry endpoint h=0; retain the explicit formula
            # for the isolated helper even though composed v92 is fully wet.
            physical_cleanup[lane] -= np.sum(
                TEMPERATURE_FLOOR_C*np.where(wet1, 0.0, h1)*
                base.cell_area_m2)
    arrays = [np.ascontiguousarray(item) for item in (
        ht1, hs1, temperature, salinity, boundary_physical, next_last, next_clock,
        physical_flux, physical_face_mass, physical_advective,
        physical_diffusive, physical_cleanup, effective, continuity,
        lane_application_count)]
    for item in arrays:
        item.setflags(write=False)
    placeholder = FluidScalarTransportResultV92(*arrays, "")
    return FluidScalarTransportResultV92(
        **{**placeholder.__dict__,
           "result_sha256": _fluid_result_token(placeholder)})


@dataclass(frozen=True)
class TransportedFluidTrialV92:
    execution_fingerprint: str
    parent_state_sha256: str
    cap_receipt: ChkvicUsedTotalReceiptV91
    hydraulic_receipt: HydraulicTurbulenceReceiptV91
    hydraulic_face_receipt: AcceptedScalarHydraulicFaceReceiptV92
    settling_diameter_receipt: SettlingDiameterReceiptV92
    suspended_capacity_receipt: SuspendedCapacityReceiptV92
    scalar_execution: SuspendedSupportExecutionV82
    suspended_trial: SuspendedSupportTrialV82
    fluid_transport: FluidScalarTransportResultV92
    midpoint_q: np.ndarray
    momentum_rate_m2_s2: np.ndarray
    momentum_face_flux_m4_s2: np.ndarray
    parent_total_suspended_concentration_kg_m3: np.ndarray
    settling_water_density_kg_m3: np.ndarray
    settling_molecular_viscosity_m2_s: np.ndarray
    candidate: TransportedFluidStateV92
    owner_counts: tuple[int, int, int, int]
    backend: str
    trial_sha256: str
    wetdry_hydraulic_evidence: object | None = None
    native_scalar_evidence: object | None = None


def _trial_token(value: TransportedFluidTrialV92) -> str:
    payload = {
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "cap_receipt_sha256": value.cap_receipt.receipt_sha256,
        "hydraulic_receipt_sha256": value.hydraulic_receipt.receipt_sha256,
        "hydraulic_face_receipt_sha256":
            value.hydraulic_face_receipt.receipt_sha256,
        "settling_diameter_receipt_sha256":
            value.settling_diameter_receipt.receipt_sha256,
        "suspended_capacity_receipt_sha256":
            value.suspended_capacity_receipt.receipt_sha256,
        "scalar_execution_fingerprint": value.scalar_execution.fingerprint,
        "suspended_trial_sha256": value.suspended_trial.trial_sha256,
        "fluid_transport_sha256": value.fluid_transport.result_sha256,
        "arrays_sha256": [_array_digest(array) for array in (
            value.midpoint_q, value.momentum_rate_m2_s2,
            value.momentum_face_flux_m4_s2,
            value.parent_total_suspended_concentration_kg_m3,
            value.settling_water_density_kg_m3,
            value.settling_molecular_viscosity_m2_s,
        )],
        "candidate_state_sha256": value.candidate.state_sha256,
        "owner_counts": list(value.owner_counts),
        "backend": value.backend,
    }
    if value.wetdry_hydraulic_evidence is not None:
        payload["wetdry_hydraulic_evidence_sha256"] = getattr(
            value.wetdry_hydraulic_evidence, "evidence_sha256", None)
    if value.native_scalar_evidence is not None:
        payload["native_scalar_evidence_sha256"] = getattr(
            value.native_scalar_evidence, "evidence_sha256", None)
    return _fingerprint(payload)


def _wrap_trial(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92, *,
    cap_receipt: ChkvicUsedTotalReceiptV91,
    hydraulic_receipt: HydraulicTurbulenceReceiptV91,
    hydraulic_face_receipt: AcceptedScalarHydraulicFaceReceiptV92,
    settling_diameter_receipt: SettlingDiameterReceiptV92 | None = None,
    suspended_capacity_receipt: SuspendedCapacityReceiptV92,
    scalar_execution: SuspendedSupportExecutionV82,
    suspended_trial: SuspendedSupportTrialV82,
    fluid_transport: FluidScalarTransportResultV92,
    midpoint_q: object,
    momentum_rate_m2_s2: object,
    momentum_face_flux_m4_s2: object,
    parent_total_suspended_concentration_kg_m3: object,
    settling_water_density_kg_m3: object,
    settling_molecular_viscosity_m2_s: object,
    candidate: TransportedFluidStateV92,
    owner_counts: tuple[int, int, int, int] = (1, 1, 1, 1),
    backend: str = "cpu_oracle",
    wetdry_hydraulic_evidence: object | None = None,
    native_scalar_evidence: object | None = None,
) -> TransportedFluidTrialV92:
    if settling_diameter_receipt is None:
        settling_diameter_receipt = make_settling_diameter_receipt_v92(
            execution, parent)
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    validate_suspended_capacity_receipt_v92(
        suspended_capacity_receipt, execution, parent,
        accepted_q=candidate.q,
        settling_diameter_receipt=settling_diameter_receipt)
    arrays = (
        _array(midpoint_q, (3, execution.cell_count), "trial midpoint"),
        _array(momentum_rate_m2_s2, (3, execution.cell_count),
               "trial momentum rate"),
        _array(momentum_face_flux_m4_s2, (2, execution.face_count),
               "trial momentum face flux"),
        _array(parent_total_suspended_concentration_kg_m3,
               (execution.cell_count,), "parent total concentration",
               nonnegative=True),
        _array(settling_water_density_kg_m3, (execution.cell_count,),
               "settling water density", nonnegative=True),
        _array(settling_molecular_viscosity_m2_s, (execution.cell_count,),
               "settling molecular viscosity", nonnegative=True),
    )
    if owner_counts != (1, 1, 1, 1):
        raise TransportedFluidSettlingV92Error(
            "v92 owner counts must be hydraulic/suspended/fluid/settling once")
    placeholder = TransportedFluidTrialV92(
        execution.fingerprint, parent.state_sha256, cap_receipt,
        hydraulic_receipt, hydraulic_face_receipt,
        settling_diameter_receipt, suspended_capacity_receipt,
        scalar_execution, suspended_trial,
        fluid_transport, *arrays, candidate, owner_counts, backend, "",
        wetdry_hydraulic_evidence, native_scalar_evidence)
    return TransportedFluidTrialV92(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _hydraulic_cpu_candidate(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90, *,
    augmented_fwave_base_rate_m2_s2: object,
    dt_s: float,
    nonlinear_rtol: float,
    nonlinear_atol: float,
    nonlinear_maxiter: int,
) -> tuple[ChkvicUsedTotalReceiptV91, HydraulicTurbulenceReceiptV91,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_execution = execution.base_execution
    base_parent = parent.base_state
    dt = _finite(dt_s, "CPU timestep", positive=True)
    base_rate = _array(
        augmented_fwave_base_rate_m2_s2, (3, execution.cell_count),
        "frozen augmented-fwave base rate")
    rtol = _finite(nonlinear_rtol, "CPU nonlinear rtol", positive=True)
    atol = _finite(nonlinear_atol, "CPU nonlinear atol", positive=True)
    if (type(nonlinear_maxiter) is not int or nonlinear_maxiter < 1 or
            rtol >= 1.0 or atol >= 1.0):
        raise TransportedFluidSettlingV92Error(
            "v92 CPU nonlinear controls differ")
    cap = apply_chkvic_used_total_v91(
        base_execution, parent_hles, handoff, dt_s=dt)
    midpoint = np.array(base_parent.q+0.5*dt*base_rate, copy=True)
    residual_evaluations = 0
    converged = False
    rate = flux = None
    for _ in range(nonlinear_maxiter):
        rate, flux = momentum_diffusion_rate_v91(
            base_execution, midpoint, cap.capped_total_vicuv_m2_s)
        residual_evaluations += 1
        candidate = base_parent.q+0.5*dt*(base_rate+rate)
        defect = float(np.linalg.norm(candidate-midpoint))
        scale = max(1.0, float(np.linalg.norm(candidate)))
        midpoint = candidate
        if defect <= max(atol, rtol*scale):
            converged = True
            break
    if not converged or rate is None or flux is None:
        raise TransportedFluidSettlingV92Error(
            "v92 CPU implicit midpoint did not converge")
    rate, flux = momentum_diffusion_rate_v91(
        base_execution, midpoint, cap.capped_total_vicuv_m2_s)
    residual_evaluations += 1
    accepted_q = np.ascontiguousarray(base_parent.q+dt*(base_rate+rate))
    hydraulic = make_hydraulic_turbulence_receipt_v91(
        base_execution, base_parent, parent_hles, handoff, cap,
        midpoint_q=midpoint, accepted_q=accepted_q, dt_s=dt,
        residual_evaluation_count=residual_evaluations,
        momentum_rate_evaluation_count=residual_evaluations,
        backend="cpu_oracle", actual_implicit_controller=False)
    return cap, hydraulic, midpoint, rate, flux, accepted_q


def _suspended_dynamic_trial_cpu(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    scalar_execution: SuspendedSupportExecutionV82,
    hydraulic: HydraulicTurbulenceReceiptV91, *,
    hydraulic_face_receipt: AcceptedScalarHydraulicFaceReceiptV92,
    settling_diameter_receipt: SettlingDiameterReceiptV92,
    depth_before_m: np.ndarray,
    depth_after_m: np.ndarray,
    endpoint_momentum_m2_s: np.ndarray,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    dt_s: float,
    scalar_inputs: Mapping[str, object],
    suspended_capacity_resolver: Callable[..., object] | None = None,
    return_capacity_receipt: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple:
    """One combined sediment+active-fluid scalar pass, then one exchange."""
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise InterruptedError(
            "v92 CPU trial cancelled before scalar consumers")
    base_v78 = scalar_execution.base_execution
    base = base_v78.base_execution
    parent_v78 = parent.base_state.suspended_state
    parent_v82 = _wrap_v82_state(scalar_execution, parent_v78)
    validate_implicit_fwave_midpoint_step_receipt_v78(
        hydraulic.base_step_receipt,
        hydraulic_execution_sha256=base_v78.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt_s)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    required = {
        "reconstructed_left_concentration_kg_m3",
        "reconstructed_right_concentration_kg_m3",
        "prescribed_boundary_concentration_kg_m3",
        "accepted_face_exchange_area_m2", "transport_face_active",
        "shear_velocity_m_s",
        "surface_fraction", "hydro_constant",
    }
    if suspended_capacity_resolver is None:
        required.add("suspended_transport_capacity_kg_m_s")
    missing = required-set(scalar_inputs)
    if missing:
        raise TransportedFluidSettlingV92Error(
            "v92 suspended inputs are absent: " + ", ".join(sorted(missing)))
    validate_accepted_scalar_hydraulic_face_receipt_v92(
        hydraulic_face_receipt, execution, parent, hydraulic,
        accepted_q=np.vstack((depth_after_m, endpoint_momentum_m2_s)))
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    water = np.array(
        hydraulic_face_receipt.accepted_water_flux_m2_s, copy=True)
    left = _mutable_array(
        scalar_inputs["reconstructed_left_concentration_kg_m3"], (ng, nf),
        "left suspended reconstruction", nonnegative=True)
    right = _mutable_array(
        scalar_inputs["reconstructed_right_concentration_kg_m3"], (ng, nf),
        "right suspended reconstruction", nonnegative=True)
    prescribed = _mutable_array(
        scalar_inputs["prescribed_boundary_concentration_kg_m3"], (ng, nb),
        "prescribed suspended boundary", nonnegative=True)
    face_area = _mutable_array(
        scalar_inputs["accepted_face_exchange_area_m2"], (nf,),
        "face exchange area", nonnegative=True)
    face_active = _mutable_array(
        scalar_inputs["transport_face_active"], (nf,),
        "transport face active", boolean=True)
    shear = _mutable_array(
        scalar_inputs["shear_velocity_m_s"], (nc,), "shear velocity",
        nonnegative=True)
    capacity = (None if suspended_capacity_resolver is not None else
                _mutable_array(
                    scalar_inputs["suspended_transport_capacity_kg_m_s"],
                    (ng, nc), "suspended capacity", nonnegative=True))
    surface = _mutable_array(
        scalar_inputs["surface_fraction"], (ng, nc), "surface fraction",
        nonnegative=True)
    if np.any(surface > 1.0):
        raise TransportedFluidSettlingV92Error(
            "v92 surface fraction exceeds one")
    constant = _mutable_array(
        scalar_inputs["hydro_constant"], (nc,), "hydro constant",
        boolean=True)
    h0 = np.asarray(depth_before_m)
    h1 = np.asarray(depth_after_m)
    dt = float(dt_s)
    old = parent_v78.lifecycle

    # Prepare compact active fluid lanes.  Temperature uses an internal
    # affine representation; all public state and ledgers remain physical.
    fluid_left_all = _mutable_array(
        reconstructed_left_fluid, (2, nf), "left fluid reconstruction")
    fluid_right_all = _mutable_array(
        reconstructed_right_fluid, (2, nf), "right fluid reconstruction")
    fluid_prescribed_all = _mutable_array(
        prescribed_boundary_fluid, (2, nb), "prescribed boundary fluid")
    if (np.any(fluid_left_all[0] <= TEMPERATURE_FLOOR_C) or
            np.any(fluid_right_all[0] <= TEMPERATURE_FLOOR_C) or
            (nb and np.any(
                fluid_prescribed_all[0] <= TEMPERATURE_FLOOR_C)) or
            np.any(fluid_left_all[1] < 0.0) or
            np.any(fluid_right_all[1] < 0.0) or
            (nb and np.any(fluid_prescribed_all[1] < 0.0))):
        raise TransportedFluidSettlingV92Error(
            "v92 reconstructed/prescribed fluid fields are outside bounds")
    active = np.asarray(execution.transported_mask)
    active_lanes = np.flatnonzero(active)
    lane_position = {int(lane): position
                     for position, lane in enumerate(active_lanes)}
    ht0 = np.asarray(parent.h_temperature_c_m)
    hs0 = np.asarray(parent.h_salinity_ppt_m)
    affine_temperature = ht0-TEMPERATURE_FLOOR_C*h0
    expected_affine = h0*(parent.temperature_cell_c-TEMPERATURE_FLOOR_C)
    relation_tolerance = 8192.0*np.finfo(float).eps*np.maximum(
        1.0, np.abs(expected_affine))
    moving_wetdry = (
        execution.base_execution.state_domain_mode ==
        STATE_DOMAIN_MOVING_WETDRY_V97)
    wet0 = h0 > base.dry_tolerance_m
    relation_cells = wet0 if moving_wetdry else np.ones(nc, dtype=bool)
    if (np.any(np.abs(
            affine_temperature[relation_cells]-
            expected_affine[relation_cells]) >
            relation_tolerance[relation_cells]) or
            np.any(affine_temperature < -relation_tolerance) or
            np.any(np.abs(
                hs0[relation_cells]-
                h0[relation_cells]*parent.salinity_cell_ppt[relation_cells]) >
                relation_tolerance[relation_cells]) or
            (moving_wetdry and
             (np.any(ht0[~wet0] != 0.0) or
              np.any(hs0[~wet0] != 0.0)))):
        raise TransportedFluidSettlingV92Error(
            "v92 parent hT/hS and accepted concentrations differ")
    fluid_mass_all = np.vstack((
        np.maximum(affine_temperature, 0.0), hs0))
    fluid_left_all[0] -= TEMPERATURE_FLOOR_C
    fluid_right_all[0] -= TEMPERATURE_FLOOR_C
    fluid_prescribed_all[0] -= TEMPERATURE_FLOOR_C

    water_divergence = np.zeros(nc)
    for cell in range(nc):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = int(base.cell_faces[slot])
            water_divergence[cell] += (
                base.cell_face_sign[slot]*base.face_length_m[face]*water[face]/
                base.cell_area_m2[cell])
    continuity = h1-h0+dt*water_divergence
    continuity_scale = np.maximum(1.0, np.maximum(np.abs(h0), np.abs(h1)))
    if active[FLUID_TEMPERATURE] and np.any(
            np.abs(continuity) >
            32768.0*np.finfo(float).eps*continuity_scale):
        raise TransportedFluidSettlingV92Error(
            "v92 affine temperature requires accepted discrete water continuity")

    # Resolve each boundary family with its own source-faithful controls, then
    # concatenate before the single numerical scalar pass.
    sediment_interior = np.zeros((ng, nc))
    wet0 = h0 > base.dry_tolerance_m
    sediment_interior[:, wet0] = (
        old.suspended_mass_kg_m2[:, wet0]/h0[wet0])
    sediment_boundary = np.zeros((ng, nf))
    next_sediment_last = np.array(
        old.boundary_last_outflow_concentration_kg_m3, copy=True)
    next_sediment_clock = np.array(old.boundary_return_clock_s, copy=True)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id < 0:
            continue
        owner = int(base.owner[face])
        inward_q = -water[face]*base.face_length_m[face]
        resolved = resolve_flow2dh_open_boundary(
            prescribed[:, boundary_id], sediment_interior[:, owner],
            np.full(ng, inward_q), dt_s=dt,
            neumann=np.asarray(base.profile.neumann_boundary_by_fraction,
                               dtype=bool),
            return_time_s=base.boundary_return_time_s[:, boundary_id],
            previous_memory=Flow2DHBoundaryMemory(
                old.boundary_last_outflow_concentration_kg_m3[:, face],
                old.boundary_return_clock_s[:, face]))
        sediment_boundary[:, face] = \
            resolved.boundary_concentration_kg_m3
        next_sediment_last[:, face] = \
            resolved.memory.last_outflow_concentration_kg_m3
        next_sediment_clock[:, face] = resolved.memory.return_clock_s

    fluid_count = active_lanes.size
    fluid_mass = fluid_mass_all[active_lanes]
    fluid_left = fluid_left_all[active_lanes]
    fluid_right = fluid_right_all[active_lanes]
    fluid_prescribed = fluid_prescribed_all[active_lanes]
    fluid_interior = np.zeros((fluid_count, nc))
    if fluid_count:
        fluid_interior[:, wet0] = fluid_mass[:, wet0]/h0[wet0]
    fluid_boundary = np.zeros((fluid_count, nf))
    next_fluid_last = np.array(parent.boundary_last_outflow_fluid, copy=True)
    next_fluid_clock = np.array(parent.boundary_return_clock_s, copy=True)
    _validate_boundary_memory(execution, next_fluid_last, next_fluid_clock)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id < 0 or not fluid_count:
            continue
        owner = int(base.owner[face])
        inward_q = -water[face]*base.face_length_m[face]
        previous_affine_all = np.array(
            parent.boundary_last_outflow_fluid[:, face], copy=True)
        previous_affine_all[FLUID_TEMPERATURE] -= TEMPERATURE_FLOOR_C
        resolved = resolve_flow2dh_open_boundary(
            fluid_prescribed[:, boundary_id], fluid_interior[:, owner],
            np.full(fluid_count, inward_q), dt_s=dt,
            neumann=np.zeros(fluid_count, dtype=bool),
            return_time_s=execution.fluid_boundary_return_time_s[
                active_lanes, boundary_id],
            previous_memory=Flow2DHBoundaryMemory(
                previous_affine_all[active_lanes],
                parent.boundary_return_clock_s[active_lanes, face]))
        fluid_boundary[:, face] = resolved.boundary_concentration_kg_m3
        for compact, lane in enumerate(active_lanes):
            next_fluid_last[lane, face] = \
                resolved.memory.last_outflow_concentration_kg_m3[compact]
            if lane == FLUID_TEMPERATURE:
                next_fluid_last[lane, face] += TEMPERATURE_FLOOR_C
            next_fluid_clock[lane, face] = \
                resolved.memory.return_clock_s[compact]
    next_fluid_last[~active] = 0.0
    next_fluid_clock[~active] = 0.0

    # Locked FLOW operator order: fallve consumes the accepted parent r0
    # concentration and parent T/rho before any constituent transport starts.
    parent_concentration = np.zeros((nc, ng))
    parent_concentration[wet0] = (
        old.suspended_mass_kg_m2[:, wet0]/h0[wet0]).T
    parent_ctot = np.ascontiguousarray(
        np.sum(parent_concentration, axis=1))
    # FLOW evaluates fallve after the hydraulic update.  The concentration
    # and T/rho inputs remain the accepted parent image, while the additive
    # moving-wet/dry route gates settling with the accepted endpoint wet set.
    # Historical fixed-domain v92 keeps its parent-wet identity unchanged.
    settling_wet = (h1 > base.dry_tolerance_m if
        execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97 else wet0)
    settling = settlefrm2_noncohesive(
        parent_concentration, base.profile.sediment_d50_m,
        base.profile.settling_fac_dss,
        base.profile.sediment_density_kg_m3,
        parent.water_density_cell_kg_m3,
        parent.temperature_cell_c,
        settling_diameter_m=
            settling_diameter_receipt.suspended_diameter_m.T,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2, wet_mask=settling_wet)
    accepted_q = np.ascontiguousarray(np.vstack((h1, endpoint_momentum_m2_s)))
    if suspended_capacity_resolver is not None:
        if not callable(suspended_capacity_resolver):
            raise TransportedFluidSettlingV92Error(
                "v92 suspended-capacity resolver must be callable")
        if cancelled():
            raise InterruptedError(
                "v92 CPU trial cancelled before capacity production")
        try:
            resolved_capacity = suspended_capacity_resolver(
                execution=execution, parent=parent,
                accepted_q=accepted_q,
                settling_velocity_m_s=np.ascontiguousarray(
                    settling.settling_velocity_m_s.T),
                settling_diameter_receipt=settling_diameter_receipt,
                surface_fraction=np.ascontiguousarray(surface),
                shear_velocity_m_s=np.ascontiguousarray(shear))
        except InterruptedError:
            raise
        except Exception as error:
            raise TransportedFluidSettlingV92Error(
                "v92 suspended-capacity resolver failed") from error
        if not isinstance(resolved_capacity, SuspendedCapacityReceiptV92):
            raise TransportedFluidSettlingV92Error(
                "v92 suspended-capacity resolver did not return its typed receipt")
        validate_suspended_capacity_receipt_v92(
            resolved_capacity, execution, parent, accepted_q=accepted_q,
            settling_diameter_receipt=settling_diameter_receipt)
        capacity_receipt = resolved_capacity
        capacity = _mutable_array(
            resolved_capacity.suspended_capacity_kg_m_s, (ng, nc),
            "resolved suspended capacity", nonnegative=True)
        if cancelled():
            raise InterruptedError(
                "v92 CPU trial cancelled after capacity production")
    else:
        assert capacity is not None
        capacity_receipt = make_suspended_capacity_receipt_v92(
            execution, parent, accepted_q=accepted_q,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_kg_m_s=capacity,
            source_kind=SUSPENDED_CAPACITY_LEGACY_INPUT,
            source_receipt_sha256=_array_digest(capacity))
    assert capacity is not None
    if not np.array_equal(settling.molecular_viscosity_m2_s,
                          parent.molecular_viscosity_cell_m2_s):
        raise TransportedFluidSettlingV92Error(
            "v92 settling viscosity differs from parent transported fluid state")

    combined_mass = np.concatenate((old.suspended_mass_kg_m2, fluid_mass))
    combined_left = np.concatenate((left, fluid_left))
    combined_right = np.concatenate((right, fluid_right))
    combined_boundary = np.concatenate((sediment_boundary, fluid_boundary))
    combined_advected, combined_flux, combined_advective_ledger = \
        _fluid_fct_active_lanes_v92(
            base, combined_mass, h0, h1, water, combined_left,
            combined_right, combined_boundary, constant, dt)
    combined_diffused, combined_face_mass, combined_diffusive_ledger, \
        effective = _fluid_diffusion_active_lanes_v92(
            base, scalar_execution.dicuv_cell_m2_s, combined_advected, h1,
            combined_boundary, face_area, face_active, dt)
    diffused = combined_diffused[:ng]
    advective_flux = combined_flux[:ng]
    face_mass = combined_face_mass[:ng]
    advective_ledger = combined_advective_ledger[:ng]
    diffusive_ledger = combined_diffusive_ledger[:ng]
    fluid_diffused = combined_diffused[ng:]
    fluid_flux = combined_flux[ng:]
    fluid_face_mass = combined_face_mass[ng:]
    fluid_advective = combined_advective_ledger[ng:]
    fluid_diffusive = combined_diffusive_ledger[ng:]

    wet1 = h1 > base.dry_tolerance_m
    speed = np.zeros(nc)
    speed[wet1] = np.hypot(
        endpoint_momentum_m2_s[0, wet1],
        endpoint_momentum_m2_s[1, wet1])/h1[wet1]
    exchange = advance_flow2dh_equilibrium_exchange(
        diffused, old.bed_mass_kg_m2, capacity, surface, speed, shear,
        h0, h1, settling.settling_velocity_m_s.T, base.cell_area_m2,
        dt_s=dt, fac_tsd=base.profile.fac_tsd,
        sediment_threshold_m=base.profile.sediment_threshold_m,
        minimum_reference_depth_m=base.profile.reference_depth_floor_m,
        flow_wet_mask=wet1)
    cleanup = np.where(wet1[None, :], 0.0,
                       exchange.suspended_mass_kg_m2)
    final_suspended = exchange.suspended_mass_kg_m2-cleanup
    final_bed = exchange.bed_mass_kg_m2+cleanup
    cleanup_water = -np.sum(cleanup*base.cell_area_m2[None, :], axis=1)
    cleanup_bed = -cleanup_water
    lifecycle = resolve_suspended_lifecycle_state_v76(
        base, suspended_mass_kg_m2=final_suspended,
        bed_mass_kg_m2=final_bed,
        boundary_last_outflow_concentration_kg_m3=next_sediment_last,
        boundary_return_clock_s=next_sediment_clock,
        advective_boundary_ledger_kg=
            old.advective_boundary_ledger_kg+advective_ledger,
        diffusive_boundary_ledger_kg=
            old.diffusive_boundary_ledger_kg+diffusive_ledger,
        water_exchange_ledger_kg=
            old.water_exchange_ledger_kg+
            exchange.suspended_exchange_ledger_kg,
        bed_exchange_ledger_kg=
            old.bed_exchange_ledger_kg+exchange.bed_exchange_ledger_kg,
        wetdry_water_ledger_kg=old.wetdry_water_ledger_kg+cleanup_water,
        wetdry_bed_ledger_kg=old.wetdry_bed_ledger_kg+cleanup_bed,
        reference_water_mass_kg=old.reference_water_mass_kg,
        reference_bed_mass_kg=old.reference_bed_mass_kg,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt)
    v78_trial: MultifractionSuspendedTrialV78 = build_multifraction_trial_v78(
        base_v78, parent_v78, hydraulic.base_step_receipt, lifecycle,
        boundary_concentration_kg_m3=sediment_boundary,
        advective_face_flux_kg_m_s=advective_flux,
        diffusive_face_mass_owner_out_kg=face_mass,
        settling_velocity_m_s=settling.settling_velocity_m_s.T,
        equilibrium_concentration_kg_m3=
            exchange.equilibrium_concentration_kg_m3,
        transferred_to_suspension_kg_m2=
            exchange.transferred_to_suspension_kg_m2,
        wetdry_transfer_to_bed_kg_m2=cleanup,
        observed_parallel_workers=1, backend=ROLE_CPU_ORACLE)
    suspended_trial = _wrap_v82_trial(
        scalar_execution, parent_v82, v78_trial, effective)

    # Split the fluid lanes after the one shared numerical pass and recover
    # physical hT/hS plus physical ledgers.
    fluid_cleanup = np.where(wet1[None, :], 0.0, fluid_diffused)
    final_fluid = fluid_diffused-fluid_cleanup
    ht1 = h1*execution.equilibrium_temperature_c
    hs1 = h1*execution.equilibrium_salinity_ppt
    temperature = np.full(nc, execution.equilibrium_temperature_c)
    salinity = np.full(nc, execution.equilibrium_salinity_ppt)
    if active[FLUID_TEMPERATURE]:
        position = lane_position[FLUID_TEMPERATURE]
        ht1 = final_fluid[position]+TEMPERATURE_FLOOR_C*h1
        temperature[wet1] = ht1[wet1]/h1[wet1]
    if active[FLUID_SALINITY]:
        position = lane_position[FLUID_SALINITY]
        hs1 = final_fluid[position]
        salinity[wet1] = hs1[wet1]/h1[wet1]
    if (np.any(temperature[wet1] <= TEMPERATURE_FLOOR_C) or
            np.any(salinity[wet1] < -8192.0*np.finfo(float).eps)):
        raise TransportedFluidSettlingV92Error(
            "v92 combined scalar pass produced invalid endpoint T/S")
    salinity = np.maximum(salinity, 0.0)
    physical_flux = np.zeros((2, nf))
    physical_face_mass = np.zeros((2, nf))
    physical_advective = np.zeros((2, nb))
    physical_diffusive = np.zeros((2, nb))
    physical_cleanup = np.zeros(2)
    boundary_physical = np.vstack((
        np.full(nf, execution.equilibrium_temperature_c),
        np.full(nf, execution.equilibrium_salinity_ppt)))
    lane_application_count = np.zeros(2, dtype=np.int32)
    water_boundary = np.zeros(nb)
    for face in range(nf):
        boundary_id = int(base.boundary_index[face])
        if boundary_id >= 0:
            water_boundary[boundary_id] -= (
                dt*base.face_length_m[face]*water[face])
    for compact, lane in enumerate(active_lanes):
        physical_flux[lane] = fluid_flux[compact]
        physical_face_mass[lane] = fluid_face_mass[compact]
        physical_advective[lane] = fluid_advective[compact]
        physical_diffusive[lane] = fluid_diffusive[compact]
        physical_cleanup[lane] = -np.sum(
            fluid_cleanup[compact]*base.cell_area_m2)
        boundary_physical[lane] = fluid_boundary[compact]
        lane_application_count[lane] = 1
        if lane == FLUID_TEMPERATURE:
            physical_flux[lane] += TEMPERATURE_FLOOR_C*water
            physical_advective[lane] += \
                TEMPERATURE_FLOOR_C*water_boundary
            boundary_physical[lane] += TEMPERATURE_FLOOR_C
            physical_cleanup[lane] -= np.sum(
                TEMPERATURE_FLOOR_C*np.where(wet1, 0.0, h1)*
                base.cell_area_m2)
    fluid_arrays = [np.ascontiguousarray(item) for item in (
        ht1, hs1, temperature, salinity, boundary_physical,
        next_fluid_last, next_fluid_clock, physical_flux,
        physical_face_mass, physical_advective, physical_diffusive,
        physical_cleanup, effective, continuity, lane_application_count)]
    for item in fluid_arrays:
        item.setflags(write=False)
    fluid_placeholder = FluidScalarTransportResultV92(*fluid_arrays, "")
    fluid_result = FluidScalarTransportResultV92(
        **{**fluid_placeholder.__dict__,
           "result_sha256": _fluid_result_token(fluid_placeholder)})
    if cancelled():
        raise InterruptedError(
            "v92 CPU trial cancelled before scalar publication")
    result = (suspended_trial, fluid_result,
              np.ascontiguousarray(parent_ctot),
              np.ascontiguousarray(parent.water_density_cell_kg_m3),
              np.ascontiguousarray(parent.molecular_viscosity_cell_m2_s))
    if type(return_capacity_receipt) is not bool:
        raise TransportedFluidSettlingV92Error(
            "v92 capacity-receipt return selector must be Boolean")
    return result+(capacity_receipt,) if return_capacity_receipt else result


def begin_transported_fluid_cpu_trial_v92(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90, *,
    augmented_fwave_base_rate_m2_s2: object,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    dt_s: float,
    settling_diameter_receipt: SettlingDiameterReceiptV92 | None = None,
    suspended_capacity_resolver: Callable[..., object] | None = None,
    cancelled: bool | Callable[[], bool] = False,
    nonlinear_rtol: float = 1.0e-13,
    nonlinear_atol: float = 1.0e-14,
    nonlinear_maxiter: int = 256,
    **scalar_inputs,
) -> TransportedFluidTrialV92:
    """Evaluate one immutable CPU v92 candidate without duplicate exchange."""
    if execution.backend != BACKEND_CPU:
        raise TransportedFluidSettlingV92Error(
            "v92 CPU trial requires the explicit CPU execution")
    if (execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97):
        raise TransportedFluidSettlingV92Error(
            "v97 moving-wet/dry execution requires the preconverged native "
            "active-set owner")
    _validate_state(parent, execution)
    # A locally terminal bounded state must fail before consulting caller-
    # supplied external receipts; otherwise a stale handoff can mask the
    # authoritative accepted-update limit.
    if parent.accepted_updates >= MAX_UPDATES:
        raise TransportedFluidSettlingV92Error(
            "v92 accepted update limit was reached")
    validate_hles_producer_state_v90(parent_hles,
                                      execution.base_execution.hles_execution)
    validate_current_closure_handoff_v90(
        handoff, execution.base_execution.hles_execution, parent_hles)
    if type(cancelled) is bool:
        cancel_check = lambda: cancelled
    elif callable(cancelled):
        cancel_check = cancelled
    else:
        raise TypeError("cancelled must be Boolean or callable")
    if cancel_check():
        raise InterruptedError(
            "v92 CPU trial cancelled before any consumer")
    if (parent.accepted_updates != parent_hles.accepted_updates or
            parent.time_s != parent_hles.time_s):
        raise TransportedFluidSettlingV92Error("v92 parent clocks differ")
    dt = _finite(dt_s, "CPU timestep", positive=True)
    cap, hydraulic, midpoint, rate, momentum_flux, accepted_q = \
        _hydraulic_cpu_candidate(
            execution, parent, parent_hles, handoff,
            augmented_fwave_base_rate_m2_s2=
                augmented_fwave_base_rate_m2_s2,
            dt_s=dt, nonlinear_rtol=nonlinear_rtol,
            nonlinear_atol=nonlinear_atol,
            nonlinear_maxiter=nonlinear_maxiter)
    if cancel_check():
        raise InterruptedError(
            "v92 CPU trial cancelled after hydraulic production")
    # Authenticate the accepted hydraulic face field once, then remove the
    # loose caller array before the combined scalar owner is entered.  Every
    # downstream consumer receives only the typed, q-bound receipt.
    scalar_inputs = dict(scalar_inputs)
    if "accepted_water_flux_m2_s" not in scalar_inputs:
        raise TransportedFluidSettlingV92Error(
            "v92 accepted hydraulic face water flux is absent")
    accepted_water_flux = scalar_inputs.pop("accepted_water_flux_m2_s")
    hydraulic_face = make_accepted_scalar_hydraulic_face_receipt_v92(
        execution, parent, hydraulic, accepted_q=accepted_q,
        accepted_water_flux_m2_s=accepted_water_flux)
    if settling_diameter_receipt is None:
        settling_diameter_receipt = make_settling_diameter_receipt_v92(
            execution, parent)
    validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    scalar_execution = resolve_suspended_support_execution_v82(
        execution.base_execution.base_execution,
        dicuv_cell_m2_s=cap.capped_total_dicuv_m2_s)
    suspended_trial, fluid, parent_ctot, settling_rho, settling_nu, \
        capacity_receipt = \
        _suspended_dynamic_trial_cpu(
            execution, parent, scalar_execution, hydraulic,
            hydraulic_face_receipt=hydraulic_face,
            settling_diameter_receipt=settling_diameter_receipt,
            depth_before_m=np.asarray(parent.q[0]),
            depth_after_m=np.asarray(accepted_q[0]),
            endpoint_momentum_m2_s=np.asarray(accepted_q[1:]),
            reconstructed_left_fluid=reconstructed_left_fluid,
            reconstructed_right_fluid=reconstructed_right_fluid,
            prescribed_boundary_fluid=prescribed_boundary_fluid,
            dt_s=dt, scalar_inputs=scalar_inputs,
            suspended_capacity_resolver=suspended_capacity_resolver,
            return_capacity_receipt=True, cancelled=cancel_check)
    chain_v91 = _fingerprint({
        "previous": parent.base_state.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "scalar": suspended_trial.trial_sha256,
    })
    candidate_base = _wrap_v91_state(
        execution.base_execution, q=accepted_q,
        suspended_state=suspended_trial.candidate.base_state,
        receipt_chain=chain_v91)
    chain = _fingerprint({
        "previous": parent.accepted_receipt_chain_sha256,
        "hydraulic": hydraulic.receipt_sha256,
        "hydraulic_face": hydraulic_face.receipt_sha256,
        "suspended": suspended_trial.trial_sha256,
        "fluid": fluid.result_sha256,
    })
    candidate = _wrap_state(
        execution, candidate_base,
        h_temperature_c_m=fluid.h_temperature_c_m,
        h_salinity_ppt_m=fluid.h_salinity_ppt_m,
        boundary_last_outflow_fluid=fluid.boundary_last_outflow_fluid,
        boundary_return_clock_s=fluid.boundary_return_clock_s,
        reference_scalar_integral=parent.reference_scalar_integral,
        advective_boundary_ledger=(
            parent.advective_boundary_ledger+
            fluid.advective_boundary_ledger),
        diffusive_boundary_ledger=(
            parent.diffusive_boundary_ledger+
            fluid.diffusive_boundary_ledger),
        wetdry_scalar_ledger=(
            parent.wetdry_scalar_ledger+fluid.wetdry_scalar_ledger),
        accepted_receipt_chain_sha256=chain)
    return _wrap_trial(
        execution, parent, cap_receipt=cap,
        hydraulic_receipt=hydraulic,
        hydraulic_face_receipt=hydraulic_face,
        settling_diameter_receipt=settling_diameter_receipt,
        suspended_capacity_receipt=capacity_receipt,
        scalar_execution=scalar_execution,
        suspended_trial=suspended_trial, fluid_transport=fluid,
        midpoint_q=midpoint, momentum_rate_m2_s2=rate,
        momentum_face_flux_m4_s2=momentum_flux,
        parent_total_suspended_concentration_kg_m3=parent_ctot,
        settling_water_density_kg_m3=settling_rho,
        settling_molecular_viscosity_m2_s=settling_nu,
        candidate=candidate)


def _ctot_cross_field_equal(
    backend: str, actual: object, expected: object,
) -> bool:
    """Compare only the host/CUDA nonnegative concentration reduction."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if (left.shape != right.shape or not np.isfinite(left).all() or
            not np.isfinite(right).all()):
        return False
    if backend == "cpu_oracle":
        return bool(np.array_equal(left, right))
    if backend != "actual_cuda":
        return False
    return bool(np.allclose(
        left, right, rtol=32.0*np.finfo(np.float64).eps, atol=0.0))


def _validate_trial(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: TransportedFluidTrialV92,
) -> None:
    _validate_state(parent, execution)
    if (not isinstance(trial, TransportedFluidTrialV92) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.owner_counts != (1, 1, 1, 1) or
            trial.backend not in ("cpu_oracle", "actual_cuda") or
            trial.backend != ("cpu_oracle" if execution.backend == BACKEND_CPU
                              else "actual_cuda") or
            trial.trial_sha256 != _trial_token(trial) or
            trial.fluid_transport.result_sha256 !=
                _fluid_result_token(trial.fluid_transport)):
        raise TransportedFluidSettlingV92Error(
            "v92 trial is stale, mutated or belongs to another parent")
    validate_chkvic_receipt_v91(
        trial.cap_receipt, execution.base_execution, parent_hles, handoff)
    validate_hydraulic_turbulence_receipt_v91(
        trial.hydraulic_receipt, execution.base_execution, parent.base_state,
        parent_hles, handoff, trial.cap_receipt)
    validate_accepted_scalar_hydraulic_face_receipt_v92(
        trial.hydraulic_face_receipt, execution, parent,
        trial.hydraulic_receipt, accepted_q=trial.candidate.q)
    validate_settling_diameter_receipt_v92(
        trial.settling_diameter_receipt, execution, parent)
    validate_suspended_capacity_receipt_v92(
        trial.suspended_capacity_receipt, execution, parent,
        accepted_q=trial.candidate.q,
        settling_diameter_receipt=trial.settling_diameter_receipt)
    scalar_parent = _wrap_v82_state(
        trial.scalar_execution, parent.suspended_state)
    _validate_v82_trial(
        trial.scalar_execution, scalar_parent, trial.suspended_trial)
    moving_wetdry = (
        execution.base_execution.state_domain_mode ==
        STATE_DOMAIN_MOVING_WETDRY_V97)
    native_cpu_settling = False
    if moving_wetdry:
        try:
            if execution.backend == BACKEND_CPU:
                from .transported_fluid_wetdry_cpu_v97 import (
                    WetDryHydraulicEvidenceV97,
                    validate_wetdry_hydraulic_evidence_v97,
                )
                evidence_type = WetDryHydraulicEvidenceV97
                validator = validate_wetdry_hydraulic_evidence_v97
            else:
                from braids_gpu.transported_fluid_wetdry_v97 import (
                    WetDryHydraulicEvidenceGPUV97,
                    validate_wetdry_hydraulic_evidence_gpu_v97,
                )
                evidence_type = WetDryHydraulicEvidenceGPUV97
                validator = validate_wetdry_hydraulic_evidence_gpu_v97
            if type(trial.wetdry_hydraulic_evidence) is not evidence_type:
                raise TypeError(
                    "v97 wet/dry hydraulic evidence capability differs")
            validator(
                trial.wetdry_hydraulic_evidence,
                execution.base_execution, parent,
                trial.cap_receipt, midpoint_q=trial.midpoint_q,
                accepted_q=trial.candidate.q,
                momentum_rate_m2_s2=trial.momentum_rate_m2_s2,
                momentum_face_flux_m4_s2=
                    trial.momentum_face_flux_m4_s2)
        except (TypeError, ValueError) as error:
            raise TransportedFluidSettlingV92Error(
                "v97 wet/dry hydraulic evidence differs") from error
        if execution.backend == BACKEND_CPU:
            try:
                from .transported_fluid_native_scalar_cpu_v96 import (
                    NativeCombinedScalarEvidenceV96,
                    _evidence_token as _native_scalar_evidence_token,
                )
                scalar_evidence = trial.native_scalar_evidence
                if (type(scalar_evidence) is not
                        NativeCombinedScalarEvidenceV96 or
                        scalar_evidence.evidence_sha256 !=
                            _native_scalar_evidence_token(scalar_evidence) or
                        scalar_evidence.suspended_trial_sha256 !=
                            trial.suspended_trial.trial_sha256 or
                        scalar_evidence.fluid_result_sha256 !=
                            trial.fluid_transport.result_sha256):
                    raise ValueError(
                        "native scalar evidence is stale or cross-trial")
                native_cpu_settling = (
                    scalar_evidence.settling_exchange_runtime_sha256
                    is not None)
            except (TypeError, ValueError) as error:
                raise TransportedFluidSettlingV92Error(
                    "v97 native scalar evidence differs") from error
        expected_rate = trial.momentum_rate_m2_s2
        expected_flux = trial.momentum_face_flux_m4_s2
    else:
        if trial.wetdry_hydraulic_evidence is not None:
            raise TransportedFluidSettlingV92Error(
                "historical v92 trial acquired v97 hydraulic evidence")
        if trial.native_scalar_evidence is not None:
            raise TransportedFluidSettlingV92Error(
                "historical v92 trial acquired v97 native scalar evidence")
        expected_rate, expected_flux = momentum_diffusion_rate_v91(
            execution.base_execution, trial.midpoint_q,
            trial.cap_receipt.capped_total_vicuv_m2_s)
    expected_hydraulic = make_hydraulic_turbulence_receipt_v91(
        execution.base_execution, parent.base_state, parent_hles, handoff,
        trial.cap_receipt, midpoint_q=trial.midpoint_q,
        accepted_q=trial.candidate.q,
        dt_s=trial.hydraulic_receipt.dt_s,
        residual_evaluation_count=
            trial.hydraulic_receipt.residual_evaluation_count,
        momentum_rate_evaluation_count=
            trial.hydraulic_receipt.momentum_rate_evaluation_count,
        backend=trial.backend,
        actual_implicit_controller=(trial.backend == "actual_cuda"))
    if trial.backend == "cpu_oracle":
        momentum_equal = (
            np.array_equal(expected_rate, trial.momentum_rate_m2_s2) and
            np.array_equal(expected_flux, trial.momentum_face_flux_m4_s2))
    else:
        momentum_equal = (
            np.allclose(expected_rate, trial.momentum_rate_m2_s2,
                        rtol=2.e-15, atol=2.e-16) and
            np.allclose(expected_flux, trial.momentum_face_flux_m4_s2,
                        rtol=2.e-15, atol=2.e-16))

    base = execution.base_execution.base_execution.base_execution
    wet = parent.q[0] > base.dry_tolerance_m
    parent_concentration = np.zeros(
        (execution.cell_count, execution.fraction_count))
    parent_concentration[wet] = (
        parent.suspended_state.suspended_mass_kg_m2[:, wet] /
        parent.q[0, wet]).T
    expected_ctot = np.sum(parent_concentration, axis=1)
    settling_wet = (
        trial.candidate.q[0] > base.dry_tolerance_m if
        execution.base_execution.state_domain_mode ==
            STATE_DOMAIN_MOVING_WETDRY_V97 else wet)
    expected_settling = settlefrm2_noncohesive(
        parent_concentration, base.profile.sediment_d50_m,
        base.profile.settling_fac_dss,
        base.profile.sediment_density_kg_m3,
        parent.water_density_cell_kg_m3, parent.temperature_cell_c,
        settling_diameter_m=
            trial.settling_diameter_receipt.suspended_diameter_m.T,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2,
        wet_mask=settling_wet).settling_velocity_m_s.T
    settling_equal = (
        np.allclose(
            trial.suspended_trial.base_trial.settling_velocity_m_s,
            expected_settling, rtol=2.e-14, atol=2.e-16)
        if native_cpu_settling else
        (np.array_equal(
            trial.suspended_trial.base_trial.settling_velocity_m_s,
            expected_settling) if trial.backend == "cpu_oracle" else
        np.allclose(
            trial.suspended_trial.base_trial.settling_velocity_m_s,
            expected_settling, rtol=2.e-14, atol=2.e-16)))
    # The actual-CUDA reduction of nonnegative per-fraction concentration may
    # differ from NumPy's host reduction by a few final mantissa bits.  Keep
    # the CPU oracle exact and relax only this recomputed cross-field; all
    # resident/host arrays, receipt tokens, owner counts and state identities
    # remain exact.
    ctot_equal = _ctot_cross_field_equal(
        trial.backend,
        trial.parent_total_suspended_concentration_kg_m3,
        expected_ctot)
    if (not momentum_equal or
            expected_hydraulic.receipt_sha256 !=
                trial.hydraulic_receipt.receipt_sha256 or
            not ctot_equal or
            not settling_equal or
            not np.array_equal(trial.settling_water_density_kg_m3,
                               parent.water_density_cell_kg_m3) or
            not np.array_equal(trial.settling_molecular_viscosity_m2_s,
                               parent.molecular_viscosity_cell_m2_s)):
        raise TransportedFluidSettlingV92Error(
            "v92 hydraulic/settling cross-fields differ")
    _validate_state(trial.candidate, execution)


def commit_transported_fluid_trial_v92(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: TransportedFluidTrialV92,
) -> TransportedFluidStateV92:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    scalar_parent = _wrap_v82_state(
        trial.scalar_execution, parent.suspended_state)
    committed = commit_suspended_support_trial_v82(
        trial.scalar_execution, scalar_parent, trial.suspended_trial)
    if committed.base_state.state_sha256 != \
            trial.candidate.suspended_state.state_sha256:
        raise TransportedFluidSettlingV92Error(
            "v92 committed suspended state differs")
    return trial.candidate


def rollback_transported_fluid_trial_v92(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    parent_hles: HlesProducerStateV90,
    handoff: CurrentClosureHandoffV90,
    trial: TransportedFluidTrialV92,
) -> TransportedFluidStateV92:
    _validate_trial(execution, parent, parent_hles, handoff, trial)
    scalar_parent = _wrap_v82_state(
        trial.scalar_execution, parent.suspended_state)
    rolled = rollback_suspended_support_trial_v82(
        trial.scalar_execution, scalar_parent, trial.suspended_trial)
    if rolled is not scalar_parent:
        raise TransportedFluidSettlingV92Error(
            "v92 suspended rollback differs")
    return parent


def cancel_transported_fluid_trial_v92(*args, **kwargs):
    return rollback_transported_fluid_trial_v92(*args, **kwargs)


def transported_fluid_checkpoint_v92(
    execution: TransportedFluidExecutionV92,
    state: TransportedFluidStateV92,
) -> dict:
    _validate_state(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "base_v91_checkpoint": shared_turbulence_checkpoint_v91(
            execution.base_execution, state.base_state),
        "conserved_fluid_state": {
            "h_temperature_c_m": state.h_temperature_c_m.tolist(),
            "h_salinity_ppt_m": state.h_salinity_ppt_m.tolist(),
            "temperature_cell_c": state.temperature_cell_c.tolist(),
            "salinity_cell_ppt": state.salinity_cell_ppt.tolist(),
            "water_density_cell_kg_m3":
                state.water_density_cell_kg_m3.tolist(),
            "molecular_viscosity_cell_m2_s":
                state.molecular_viscosity_cell_m2_s.tolist(),
            "boundary_last_outflow_fluid":
                state.boundary_last_outflow_fluid.tolist(),
            "boundary_return_clock_s":
                state.boundary_return_clock_s.tolist(),
            "reference_scalar_integral":
                state.reference_scalar_integral.tolist(),
            "advective_boundary_ledger":
                state.advective_boundary_ledger.tolist(),
            "diffusive_boundary_ledger":
                state.diffusive_boundary_ledger.tolist(),
            "wetdry_scalar_ledger": state.wetdry_scalar_ledger.tolist(),
            "accepted_receipt_chain_sha256":
                state.accepted_receipt_chain_sha256,
        },
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_transported_fluid_checkpoint_v92(
    execution: TransportedFluidExecutionV92,
    document: Mapping[str, object],
) -> TransportedFluidStateV92:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "base_v91_checkpoint", "conserved_fluid_state",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise TransportedFluidSettlingV92Error(
            "v92 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR):
        raise TransportedFluidSettlingV92Error(
            "v92 checkpoint identity differs")
    fields = document["conserved_fluid_state"]
    expected_fields = {
        "h_temperature_c_m", "h_salinity_ppt_m", "temperature_cell_c",
        "salinity_cell_ppt", "water_density_cell_kg_m3",
        "molecular_viscosity_cell_m2_s",
        "boundary_last_outflow_fluid",
        "boundary_return_clock_s", "reference_scalar_integral",
        "advective_boundary_ledger", "diffusive_boundary_ledger",
        "wetdry_scalar_ledger", "accepted_receipt_chain_sha256",
    }
    if not isinstance(fields, Mapping) or set(fields) != expected_fields:
        raise TransportedFluidSettlingV92Error(
            "v92 checkpoint fluid inventory differs")
    try:
        base = load_shared_turbulence_checkpoint_v91(
            execution.base_execution, document["base_v91_checkpoint"])
        state = _wrap_state(
            execution, base,
            h_temperature_c_m=fields["h_temperature_c_m"],
            h_salinity_ppt_m=fields["h_salinity_ppt_m"],
            boundary_last_outflow_fluid=fields["boundary_last_outflow_fluid"],
            boundary_return_clock_s=fields["boundary_return_clock_s"],
            reference_scalar_integral=fields["reference_scalar_integral"],
            advective_boundary_ledger=fields["advective_boundary_ledger"],
            diffusive_boundary_ledger=fields["diffusive_boundary_ledger"],
            wetdry_scalar_ledger=fields["wetdry_scalar_ledger"],
            accepted_receipt_chain_sha256=
                fields["accepted_receipt_chain_sha256"],
            temperature_cell_c=fields["temperature_cell_c"],
            salinity_cell_ppt=fields["salinity_cell_ppt"],
            water_density_cell_kg_m3=
                fields["water_density_cell_kg_m3"],
            molecular_viscosity_cell_m2_s=
                fields["molecular_viscosity_cell_m2_s"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, TransportedFluidSettlingV92Error):
            raise
        raise TransportedFluidSettlingV92Error(
            "v92 checkpoint state is invalid") from error
    if document["state_sha256"] != state.state_sha256:
        raise TransportedFluidSettlingV92Error(
            "v92 checkpoint state digest differs")
    return state


LOCKED_SOURCES = {
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute/difu.f90":
        "e8e11d438a6e3aa8e912ce1133efdec093b5bad60605aefa6c4b037ea6aa5c17",
}


def verify_locked_sources_v92(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in LOCKED_SOURCES.items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise TransportedFluidSettlingV92Error(
                "v92 locked source hash differs: " + relative)
        result[relative] = actual
    return result


def method_identity_v92() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "backend_roles": [ROLE_CPU_ORACLE, ROLE_CUDA],
        "transported_state": ["h_temperature_c_m", "h_salinity_ppt_m"],
        "temperature_affine_floor_c": TEMPERATURE_FLOOR_C,
        "affine_equivalence_gate": "accepted_discrete_water_continuity",
        "boundary": "Dirichlet_plus_Thatcher_Harleman_no_sediment_NeuBc",
        "horizontal_diffusion": "shared_capped_dicuv_divided_by_0p7_once",
        "settling_call_order": (
            "parent_T_rho_nu_and_parent_pretransport_ctot_then_"
            "FCT_diffusion_exchange_then_endpoint_EOS_for_next_update"),
        "scalar_owner": "one_combined_sediment_plus_active_fluid_pass",
        "standalone_fluid_helper_admitted": False,
        "derived_not_transported": ["rho_water", "nu_water"],
        "owner_counts": {
            "hydraulic": 1, "suspended": 1,
            "fluid_scalar": 1, "settling_exchange": 1,
        },
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "product_admission": False,
        "not_claimed": [
            "moving_wet_dry_composed_hydraulics", "cut_cells", "general_AMR",
            "DensIn_nonlinear_feedback", "Q3_dynamic_density_consumption_v96",
            "long_run_equivalence",
        ],
        "delft3d_source_modified": False,
    }


__all__ = [
    "AcceptedScalarHydraulicFaceReceiptV92", "CHECKPOINT_SCOPE_ID",
    "EOS_CHOICES", "EOS_ECKART", "EOS_NACL",
    "EOS_UNESCO", "FLUID_SCALAR_COUNT", "FLUID_SCALAR_ORDER",
    "FluidScalarTransportResultV92", "INTEGRATOR", "MAX_UPDATES",
    "MODE_CONSTANT", "MODE_TRANSPORTED", "MODES", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "TEMPERATURE_FLOOR_C", "TransportedFluidExecutionV92",
    "SETTLING_DIAMETER_ACTIVE_IOPSUS_V94",
    "SETTLING_DIAMETER_SOURCE_KINDS",
    "SETTLING_DIAMETER_STATIC_PROFILE", "SettlingDiameterReceiptV92",
    "SUSPENDED_CAPACITY_ACTIVE_V83_V94",
    "SUSPENDED_CAPACITY_LEGACY_INPUT", "SUSPENDED_CAPACITY_SOURCE_KINDS",
    "SuspendedCapacityReceiptV92",
    "TransportedFluidSettlingV92Error", "TransportedFluidStateV92",
    "TransportedFluidTrialV92", "begin_transported_fluid_cpu_trial_v92",
    "cancel_transported_fluid_trial_v92",
    "commit_transported_fluid_trial_v92", "execution_document_v92",
    "load_transported_fluid_checkpoint_v92", "method_identity_v92",
    "make_accepted_scalar_hydraulic_face_receipt_v92",
    "make_settling_diameter_receipt_v92",
    "make_suspended_capacity_receipt_v92",
    "resolve_transported_fluid_execution_v92",
    "resolve_transported_fluid_state_v92",
    "rollback_transported_fluid_trial_v92",
    "transported_fluid_checkpoint_v92", "verify_locked_sources_v92",
    "validate_accepted_scalar_hydraulic_face_receipt_v92",
    "validate_settling_diameter_receipt_v92",
    "validate_suspended_capacity_receipt_v92",
]
