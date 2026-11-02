"""Canonical additive identity for the Q3-H3 nonzero operator transaction.

The accepted Q3-H3/v57 document remains the identity of the zero-Coriolis,
zero-diffusion and zero-Betac foundation.  This module adds an outer v58
identity for the three nonzero operators and, critically, records their
ordering.  Backend/integrator selection remains a separate saved choice.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from .prognostic_secondary_execution import (
    validate_prognostic_secondary_execution_document,
)


SCHEMA = 1
ENVELOPE_VERSION = 58
PARENT_ENVELOPE_VERSION = 57
CHECKPOINT_SCHEMA = 9
EXECUTION_KEY = "prognostic_secondary_nonzero_execution_identity"
ORDERED_ROUTE_KEY = "prognostic_secondary_nonzero_ordered_route_identity"
SCOPE_ID = "braids-q3h3-flow2dh-prognostic-secondary-nonzero-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()

TRANSACTION_ORDER = (
    "accepted_start_snapshot_and_identity_validation",
    "accepted_start_betac_tensor_and_face_flux_freeze",
    "hydraulic_trial_with_frozen_betac_feedback",
    "post_hydraulic_friction_and_exact_constant_coriolis",
    "post_hydraulic_curvature_source_sink_and_open_equilibrium_trace",
    "signed_hI_conservative_advection_with_hydraulics_held_fixed",
    "signed_hI_new_depth_backward_euler_scalar_diffusion",
    "old_depth_explicit_source_new_depth_implicit_sink",
    "resolved_wet_dry_cleanup_after_transport_and_reaction",
    "transported_hI_bedload_rotation_once",
    "channel_composition_face_reconstruction_limiting_and_exner",
    "atomic_state_and_ledger_commit_or_full_rollback",
)


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Q3-H3/v58 {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3-H3/v58 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Q3-H3/v58 {name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"Q3-H3/v58 {name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"Q3-H3/v58 {name} must be nonnegative")
    return 0.0 if result == 0.0 else result


def _build_body(*, parent_fingerprint: str, capacity_execution_fingerprint: str,
                capacity_profile_fingerprint: str,
                hiding_execution_fingerprint: str | None,
                coriolis_disposition: str,
                coriolis_constant_s_inverse: float, coriolis_source: str,
                rmincf: float, chzmin: float, espir: float, betac: float,
                scalar_diffusivity_m2_s: float,
                von_karman_constant: float) -> dict[str, object]:
    parent = _sha256(parent_fingerprint, "parent execution fingerprint")
    capacity_execution = _sha256(
        capacity_execution_fingerprint, "capacity execution fingerprint")
    capacity_profile = _sha256(
        capacity_profile_fingerprint, "capacity profile fingerprint")
    hiding = None if hiding_execution_fingerprint is None else _sha256(
        hiding_execution_fingerprint, "hiding execution fingerprint")
    coriolis = _finite(coriolis_constant_s_inverse, "Coriolis constant")
    if coriolis_disposition not in {"zero", "nonzero"}:
        raise ValueError("Q3-H3/v58 Coriolis disposition must be zero or nonzero")
    if ((coriolis == 0.0 and coriolis_disposition != "zero") or
            (coriolis != 0.0 and coriolis_disposition != "nonzero")):
        raise ValueError("Q3-H3/v58 Coriolis disposition and value disagree")
    expected_source = "constant_zero_planar" if coriolis == 0.0 else "constant_planar"
    if coriolis_source != expected_source:
        raise ValueError("Q3-H3/v58 Coriolis field identity differs")

    resolved_von_karman = _finite(
        von_karman_constant, "von Karman constant", positive=True)
    if resolved_von_karman != 0.41:
        raise ValueError("Q3-H3/v58 von Karman constant must be 0.41")

    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_envelope_version": PARENT_ENVELOPE_VERSION,
        "parent_execution_fingerprint": parent,
        "capacity_execution_fingerprint": capacity_execution,
        "capacity_profile_fingerprint": capacity_profile,
        "hiding_execution_fingerprint": hiding,
        "grid_scope": "fixed_cartesian_fully_wet",
        "backend_selection": "explicit_user_choice",
        "backend_integrator_binding": "separate_saved_runtime_identity",
        "automatic_cpu_fallback": False,
        "coriolis_disposition": coriolis_disposition,
        "coriolis_constant_s_inverse": coriolis,
        "coriolis_source": coriolis_source,
        "hydraulic_coriolis_policy": "exact_endpoint_rotation_after_hydraulic_flux_trial",
        "rmincf": _finite(rmincf, "Rmincf", positive=True),
        "chzmin": _finite(chzmin, "Chzmin", positive=True),
        "espir": _finite(espir, "Espir", nonnegative=True),
        "betac": _finite(betac, "Betac"),
        "betac_activation_threshold": 1.0e-4,
        "betac_speed_floor_m_s": 1.0e-8,
        "betac_time_lag": "accepted_start_once_per_macro_step",
        "betac_face_flux": "negative_tensor_dot_outward_normal",
        "scalar_diffusivity_m2_s": _finite(
            scalar_diffusivity_m2_s, "scalar diffusivity", nonnegative=True),
        "scalar_schmidt_number": 0.7,
        "scalar_diffusion_scheme": "new_depth_symmetric_graph_backward_euler",
        "scalar_diffusion_linear_solver": "jacobi_preconditioned_conjugate_gradient",
        "scalar_diffusion_relative_tolerance": 1.0e-13,
        "scalar_diffusion_absolute_tolerance": 1.0e-14,
        "scalar_diffusion_max_iterations": 256,
        "von_karman_constant": resolved_von_karman,
        "open_boundary_trace_policy": "adjacent_interior_equilibrium_all_flow_directions",
        "wall_policy": "zero_scalar_flux_and_zero_secondary_stress_traction",
        "diffusion_boundary_ledger": "per_boundary_signed_inward_integrated_hI_m4_s",
        "stress_boundary_ledger": "per_boundary_signed_integrated_momentum_m4_s",
        "coriolis_ledger": "signed_interior_integrated_momentum_increment_m4_s",
        "transaction_order": list(TRANSACTION_ORDER),
        "failure_policy": "reject_and_restore_state_and_every_ledger",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }


def prognostic_secondary_nonzero_execution_document(
        parent: Mapping[str, object], *,
        coriolis_disposition: str,
        coriolis_constant_s_inverse: float,
        coriolis_source: str,
        betac: float,
        scalar_diffusivity_m2_s: float,
) -> dict[str, object]:
    """Build v58 from one authenticated frozen v57 foundation document.

    The parent must be the accepted neutral v57 identity: Coriolis, Betac and
    scalar diffusion are all exactly zero there.  V58 alone owns the required
    additive controls.  This prevents a non-admissible v57 document from being
    used as nominal parent authority.
    """
    validated = validate_prognostic_secondary_execution_document(parent)
    if (
        validated["coriolis_disposition"] != "zero" or
        validated["coriolis_constant_s_inverse"] != 0.0 or
        validated["coriolis_source"] != "constant_zero_planar" or
        validated["betac"] != 0.0 or
        validated["scalar_diffusivity_m2_s"] != 0.0
    ):
        raise ValueError("Q3-H3/v58 parent must be the neutral admitted v57 branch")
    body = _build_body(
        parent_fingerprint=validated["fingerprint"],
        capacity_execution_fingerprint=validated["capacity_execution_fingerprint"],
        capacity_profile_fingerprint=validated["capacity_profile_fingerprint"],
        hiding_execution_fingerprint=validated["hiding_execution_fingerprint"],
        coriolis_disposition=coriolis_disposition,
        coriolis_constant_s_inverse=coriolis_constant_s_inverse,
        coriolis_source=coriolis_source,
        rmincf=validated["rmincf"], chzmin=validated["chzmin"],
        espir=validated["espir"],
        betac=betac,
        scalar_diffusivity_m2_s=scalar_diffusivity_m2_s,
        von_karman_constant=validated["von_karman_constant"],
    )
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_prognostic_secondary_nonzero_execution_document(
        value: Mapping[str, object], *,
        parent: Mapping[str, object] | None = None) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted additive v58 identity."""
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H3/v58 execution identity must be an object")
    try:
        body = _build_body(
            parent_fingerprint=value["parent_execution_fingerprint"],
            capacity_execution_fingerprint=value["capacity_execution_fingerprint"],
            capacity_profile_fingerprint=value["capacity_profile_fingerprint"],
            hiding_execution_fingerprint=value["hiding_execution_fingerprint"],
            coriolis_disposition=value["coriolis_disposition"],
            coriolis_constant_s_inverse=value["coriolis_constant_s_inverse"],
            coriolis_source=value["coriolis_source"],
            rmincf=value["rmincf"], chzmin=value["chzmin"],
            espir=value["espir"], betac=value["betac"],
            scalar_diffusivity_m2_s=value["scalar_diffusivity_m2_s"],
            von_karman_constant=value["von_karman_constant"],
        )
        rebuilt = {
            **body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}
        if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
            raise ValueError("Q3-H3/v58 execution identity is stale or forged")
        if parent is not None:
            validated_parent = validate_prognostic_secondary_execution_document(parent)
            if (
                validated_parent["coriolis_disposition"] != "zero" or
                validated_parent["coriolis_constant_s_inverse"] != 0.0 or
                validated_parent["coriolis_source"] != "constant_zero_planar" or
                validated_parent["betac"] != 0.0 or
                validated_parent["scalar_diffusivity_m2_s"] != 0.0
            ):
                raise ValueError(
                    "Q3-H3/v58 parent must be the neutral admitted v57 branch")
            if (
                rebuilt["parent_execution_fingerprint"] != validated_parent["fingerprint"] or
                rebuilt["capacity_execution_fingerprint"] !=
                    validated_parent["capacity_execution_fingerprint"] or
                rebuilt["capacity_profile_fingerprint"] !=
                    validated_parent["capacity_profile_fingerprint"] or
                rebuilt["hiding_execution_fingerprint"] !=
                    validated_parent["hiding_execution_fingerprint"]
            ):
                raise ValueError("Q3-H3/v58 parent identity binding differs")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Q3-H3/v58"):
            raise
        raise ValueError("Invalid Q3-H3/v58 execution identity") from error
    return rebuilt


def prognostic_secondary_nonzero_ordered_route(
        document: Mapping[str, object]) -> dict[str, object]:
    """Bind the additive operator identity to its ordered v57 foundations."""
    identity = validate_prognostic_secondary_nonzero_execution_document(document)
    return {
        "ordered_parts": [
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
            "prognostic_secondary_execution_fingerprint",
            "prognostic_secondary_nonzero_execution_fingerprint",
        ],
        "capacity_execution_fingerprint":
            identity["capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":
            identity["capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":
            identity["hiding_execution_fingerprint"],
        "prognostic_secondary_execution_fingerprint":
            identity["parent_execution_fingerprint"],
        "prognostic_secondary_nonzero_execution_fingerprint":
            identity["fingerprint"],
    }


def validate_prognostic_secondary_nonzero_ordered_route(
        value: Mapping[str, object], document: Mapping[str, object],
) -> dict[str, object]:
    """Require the exact five-part ordered route for one v58 document."""
    expected = prognostic_secondary_nonzero_ordered_route(document)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("Q3-H3/v58 ordered route identity differs")
    return expected


__all__ = [
    "CHECKPOINT_SCHEMA", "ENVELOPE_VERSION", "EXECUTION_KEY",
    "ORDERED_ROUTE_KEY", "PARENT_ENVELOPE_VERSION", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "TRANSACTION_ORDER",
    "prognostic_secondary_nonzero_ordered_route",
    "prognostic_secondary_nonzero_execution_document",
    "validate_prognostic_secondary_nonzero_ordered_route",
    "validate_prognostic_secondary_nonzero_execution_document",
]
