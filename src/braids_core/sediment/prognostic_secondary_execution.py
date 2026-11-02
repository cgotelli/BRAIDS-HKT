"""Additive Q3-H3 classic-FLOW prognostic-secondary identity.

The v57 envelope is a sibling of Q3-H2/v56.  It wraps the same immutable
Q3-C/v54 or Q3-H1/v55 scalar-capacity foundation directly and reuses the
authenticated H2 curvature, equilibrium-target and typed-roughness
primitives.  It does *not* execute the v56 equilibrium bed rotation.  One
transported signed spiral-intensity state owns the sole bed-direction
rotation instead.

This module authenticates semantics and runtime layout; it does not grant a
route admission.  In particular, finite nonzero classic ``Betac`` and scalar
diffusivity values are representable so their identities cannot collide, but
the first live Q3-H3 gate rejects them until their separate operator gates
pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .secondary_foundation_execution import (
    CURVATURE_CAP_POLICY,
    CURVATURE_DISCRETIZATION,
    DEPTH_PAIR_POLICY as H2_REFERENCE_DEPTH_PAIR_POLICY,
    METRIC_POLICY,
    ROUGHNESS_METHOD_IDS,
    ROUGHNESS_POLICY,
    SCOPE_ID as H2_REFERENCE_SCOPE_ID,
    SEMANTIC_ID as H2_REFERENCE_SEMANTIC_ID,
)


SCHEMA = 1
ENVELOPE_VERSION = 57
SCOPE_ID = "braids-q3h3-flow2dh-prognostic-secondary-composition-v1"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
EXECUTION_KEY = "prognostic_secondary_execution_identity"
ORDERED_ROUTE_KEY = "prognostic_secondary_ordered_route_identity"

METHOD_ID = "secondary.prognostic"
MODEL = 2
FIXED_GRID_SCOPE = "fixed_grid"
BACKEND_SELECTION = "explicit_user_choice"
CAPACITY_METHOD_BY_LAW = MappingProxyType({
    2: "capacity.mpm",
    8: "capacity.engelund_hansen",
    11: "capacity.general",
})
CORIOLIS_DISPOSITIONS = frozenset(("zero", "nonzero"))

# The runtime/checkpoint slot is the conservative transported quantity h*I,
# not the primitive spiral intensity I used by the direction operator.
STATE_SEMANTICS = "signed_depth_times_spiral_intensity_m2_s"
INTEGRATED_LEDGER_UNITS = "m4_s"
SOURCE_SINK_POLICY = "old_depth_explicit_source_new_depth_implicit_sink"
DEPTH_PAIR_POLICY = (
    "accepted_old_depth_post_hydraulic_new_depth_"
    "new_velocity_and_curvature_v1"
)
TARGET_FORCING_POLICY = (
    "new_trial_velocity_curvature_and_typed_chezy_"
    "accepted_old_depth_explicit_source_new_depth_implicit_sink_v1"
)
OPEN_BOUNDARY_POLICY = "adjacent_interior_equilibrium_all_directions"
BED_BOUNDARY_ROTATION_POLICY = (
    "roles12_trace_owner_equilibrium_role3_rotated_owner_v1"
)
DRY_POLICY = "zero"
TRANSPORT_POLICY = "conservative_reconstructed_advection"
DIRECTION_POLICY = "transported_state_once"
EQUILIBRIUM_ROTATION_POLICY = "not_executed"
INITIAL_STATE_POLICY = "zero_signed_hI"
PUBLIC_FRAME_POLICY = "physical_state_only_auxiliary_hI_private"
CHECKPOINT_POLICY = "full_signed_hI_ledgers_source_and_identity"
GPU_CHECKPOINT_SCHEMA = 8
INFLOW_MODE = 1

H2_REUSED_PRIMITIVES = (
    "typed_raw_chezy",
    "classic_flow2dh_streamline_curvature",
    "classic_flow2dh_rmincf_cap",
    "classic_flow2dh_old_new_depth_equilibrium_target",
    "classic_flow2dh_chzmin_source_floor",
    "classic_flow2dh_open_boundary_equilibrium",
)

COMPOSITION_ORDER = (
    "accepted_step_old_depth_and_post_hydraulic_new_trial_state",
    "new_trial_velocity_curvature_and_typed_chezy_equilibrium_target",
    "adjacent_interior_open_boundary_equilibrium_trace",
    "signed_hI_conservative_reconstructed_advection",
    "classic_flow2dh_old_depth_explicit_source_new_depth_implicit_sink",
    "dry_cell_signed_hI_cleanup",
    "transactional_trial_surface_active_volume",
    "optional_hiding_exposure_factors_once_per_trial_stage",
    "raw_scalar_cell_and_owner_indexed_boundary_capacity",
    "transported_hI_raw_bed_rotation_once",
    "channel_composition_once",
    "face_reconstruction",
    "bed_slope_off",
    "donor_and_finite_availability_limiting",
    "conservative_flux_and_exner",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(body: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def _sha256(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Q3-H3 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Q3-H3 {name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"Q3-H3 {name} must be a finite number") from error
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        relation = "positive" if positive else "nonnegative" if nonnegative else "finite"
        raise ValueError(f"Q3-H3 {name} must be {relation}")
    return 0.0 if result == 0.0 else result


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"Q3-H3 {name} must be a positive integer")
    return value


def _fraction_ids(value: object, count: int) -> list[str]:
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or
            len(value) != count):
        raise ValueError("Q3-H3 ordered fraction identities differ from fraction count")
    result = list(value)
    if (len(set(result)) != count or
            any(not isinstance(item, str) or _SOURCE_ID.fullmatch(item) is None
                for item in result)):
        raise ValueError("Q3-H3 fraction identities must be ordered, unique identifiers")
    return result


def prognostic_secondary_execution_document(
    *,
    method_id: str,
    model: int,
    capacity_law: int,
    capacity_execution_fingerprint: str,
    capacity_profile_fingerprint: str,
    roughness_method_id: str,
    roughness_fingerprint: str,
    fraction_ids: Sequence[str],
    bed_layer_count: int,
    transport_direction: bool,
    inflow_mode: int,
    coriolis_disposition: str,
    coriolis_constant_s_inverse: float,
    coriolis_source: str,
    rmincf: float,
    chzmin: float,
    espir: float,
    betac: float,
    scalar_diffusivity_m2_s: float,
    von_karman_constant: float = 0.41,
    gpu_checkpoint_schema: int = GPU_CHECKPOINT_SCHEMA,
    hiding_execution_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build one canonical fixed-grid v57 prognostic composition."""
    if method_id != METHOD_ID or type(model) is not int or model != MODEL:
        raise ValueError("Q3-H3 requires classic FLOW secondary.prognostic model 2")
    if type(capacity_law) is not int or capacity_law not in CAPACITY_METHOD_BY_LAW:
        raise ValueError("Q3-H3 capacity law must be 2, 8 or 11")
    if roughness_method_id not in ROUGHNESS_METHOD_IDS:
        raise ValueError("Q3-H3 requires a known typed C/n/ks roughness method")
    capacity_execution = _sha256(
        capacity_execution_fingerprint, "Q3-C execution fingerprint")
    capacity_profile = _sha256(
        capacity_profile_fingerprint, "Q3-C profile fingerprint")
    hiding_execution = _sha256(
        hiding_execution_fingerprint, "Q3-H1 execution fingerprint", optional=True)
    roughness = _sha256(roughness_fingerprint, "typed roughness fingerprint")
    if hiding_execution is not None and capacity_law not in (2, 11):
        raise ValueError("Q3-H3 cannot compose Q3-H1 hiding with capacity law 8")

    layers = _positive_integer(bed_layer_count, "bed-layer count")
    if (not isinstance(fraction_ids, Sequence) or
            isinstance(fraction_ids, (str, bytes))):
        raise ValueError("Q3-H3 fraction identities must be a sequence")
    count = _positive_integer(len(fraction_ids), "fraction count")
    fractions = _fraction_ids(fraction_ids, count)
    if type(transport_direction) is not bool:
        raise ValueError("Q3-H3 transport-direction switch must be true or false")
    if type(inflow_mode) is not int or inflow_mode != INFLOW_MODE:
        raise ValueError("Q3-H3 requires adjacent-interior equilibrium inflow mode 1")
    if type(gpu_checkpoint_schema) is not int or gpu_checkpoint_schema != GPU_CHECKPOINT_SCHEMA:
        raise ValueError(f"Q3-H3 requires GPU checkpoint schema {GPU_CHECKPOINT_SCHEMA}")

    if coriolis_disposition not in CORIOLIS_DISPOSITIONS:
        raise ValueError("Q3-H3 Coriolis disposition must be zero or nonzero")
    coriolis = _finite(coriolis_constant_s_inverse, "Coriolis constant")
    if ((coriolis_disposition == "zero" and coriolis != 0.0) or
            (coriolis_disposition == "nonzero" and coriolis == 0.0)):
        raise ValueError("Q3-H3 Coriolis disposition and explicit value disagree")
    if (not isinstance(coriolis_source, str) or
            _SOURCE_ID.fullmatch(coriolis_source) is None):
        raise ValueError("Q3-H3 Coriolis source must be a canonical identifier")
    if coriolis_source == "constant_zero_planar" and coriolis != 0.0:
        raise ValueError("Q3-H3 constant_zero_planar Coriolis source requires zero")

    resolved_rmincf = _finite(rmincf, "Rmincf", positive=True)
    resolved_chzmin = _finite(chzmin, "Chzmin", positive=True)
    resolved_espir = _finite(espir, "Espir", nonnegative=True)
    resolved_betac = _finite(betac, "classic Betac")
    diffusivity = _finite(
        scalar_diffusivity_m2_s, "spiral-intensity scalar diffusivity", nonnegative=True)
    von_karman = _finite(von_karman_constant, "von Karman constant", positive=True)

    physical_state_count = 3 + (1 + layers) * count
    physical_primitive_count = 4 + count
    physical_ledger_count = 1 + 2 * count
    layout = {
        "index_base": 0,
        "fraction_count": count,
        "bed_layer_count": layers,
        "physical_state_count": physical_state_count,
        "runtime_state_count": physical_state_count + 1,
        "secondary_state_index": physical_state_count,
        "physical_primitive_count": physical_primitive_count,
        "runtime_primitive_count": physical_primitive_count + 1,
        "secondary_primitive_index": 3 + count,
        "physical_ledger_count": physical_ledger_count,
        "runtime_ledger_count": physical_ledger_count + 1,
        "secondary_ledger_index": physical_ledger_count,
    }
    has_hiding = hiding_execution is not None
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "grid_scope": FIXED_GRID_SCOPE,
        "method_id": method_id,
        "model": model,
        "selector_provenance": "Sub1=I;Equili=N",
        "capacity_law": capacity_law,
        "capacity_method_id": CAPACITY_METHOD_BY_LAW[capacity_law],
        "base_envelope_version": 55 if has_hiding else 54,
        "base_reference_policy": "opaque_frozen_foundation_secondary.off_not_reinterpreted",
        "capacity_foundation_envelope_version": 54,
        "capacity_execution_fingerprint": capacity_execution,
        "capacity_profile_fingerprint": capacity_profile,
        "hiding_foundation_envelope_version": 55 if has_hiding else None,
        "hiding_execution_fingerprint": hiding_execution,
        "ordered_fraction_ids": fractions,
        "roughness_method_id": roughness_method_id,
        "roughness_fingerprint": roughness,
        "roughness_policy": ROUGHNESS_POLICY,
        "h2_reference_scope_id": H2_REFERENCE_SCOPE_ID,
        "h2_reference_semantic_id": H2_REFERENCE_SEMANTIC_ID,
        "h2_reused_primitives": list(H2_REUSED_PRIMITIVES),
        "h2_equilibrium_rotation_executed": False,
        "h2_reference_depth_pair_policy": H2_REFERENCE_DEPTH_PAIR_POLICY,
        "curvature_discretization": CURVATURE_DISCRETIZATION,
        "curvature_cap_policy": CURVATURE_CAP_POLICY,
        "metric_policy": METRIC_POLICY,
        "depth_pair_policy": DEPTH_PAIR_POLICY,
        "target_forcing_policy": TARGET_FORCING_POLICY,
        "coriolis_disposition": coriolis_disposition,
        "coriolis_constant_s_inverse": coriolis,
        "coriolis_source": coriolis_source,
        "rmincf": resolved_rmincf,
        "chzmin": resolved_chzmin,
        "espir": resolved_espir,
        "betac": resolved_betac,
        "scalar_diffusivity_m2_s": diffusivity,
        "von_karman_constant": von_karman,
        "transport_direction": transport_direction,
        "inflow_mode": inflow_mode,
        "state_semantics": STATE_SEMANTICS,
        "integrated_ledger_units": INTEGRATED_LEDGER_UNITS,
        "source_sink_policy": SOURCE_SINK_POLICY,
        "open_boundary_policy": OPEN_BOUNDARY_POLICY,
        "bed_boundary_rotation_policy": BED_BOUNDARY_ROTATION_POLICY,
        "dry_policy": DRY_POLICY,
        "transport_policy": TRANSPORT_POLICY,
        "direction_policy": DIRECTION_POLICY,
        "equilibrium_rotation_policy": EQUILIBRIUM_ROTATION_POLICY,
        "initial_state_policy": INITIAL_STATE_POLICY,
        "public_frame_policy": PUBLIC_FRAME_POLICY,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "gpu_checkpoint_schema": gpu_checkpoint_schema,
        "runtime_layout": layout,
        "composition_order": list(COMPOSITION_ORDER),
        "raw_bed_rotation_scope": "bedload_only_before_channel_composition",
        "suspended_channel_rotation": False,
        "downstream_numerics": (
            "existing_channel_composer_then_unchanged_face_reconstruction_"
            "availability_and_exner"
        ),
        "backend_selection": BACKEND_SELECTION,
        "automatic_cpu_fallback": False,
        "runtime_admission_authority": "separate_exact_route_gate",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def validate_prognostic_secondary_execution_document(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted Q3-H3 identity."""
    if not isinstance(value, Mapping):
        raise ValueError("Q3-H3 prognostic-secondary execution identity must be an object")
    try:
        rebuilt = prognostic_secondary_execution_document(
            method_id=value["method_id"],
            model=value["model"],
            capacity_law=value["capacity_law"],
            capacity_execution_fingerprint=value["capacity_execution_fingerprint"],
            capacity_profile_fingerprint=value["capacity_profile_fingerprint"],
            hiding_execution_fingerprint=value["hiding_execution_fingerprint"],
            roughness_method_id=value["roughness_method_id"],
            roughness_fingerprint=value["roughness_fingerprint"],
            fraction_ids=value["ordered_fraction_ids"],
            bed_layer_count=value["runtime_layout"]["bed_layer_count"],
            transport_direction=value["transport_direction"],
            inflow_mode=value["inflow_mode"],
            coriolis_disposition=value["coriolis_disposition"],
            coriolis_constant_s_inverse=value["coriolis_constant_s_inverse"],
            coriolis_source=value["coriolis_source"],
            rmincf=value["rmincf"],
            chzmin=value["chzmin"],
            espir=value["espir"],
            betac=value["betac"],
            scalar_diffusivity_m2_s=value["scalar_diffusivity_m2_s"],
            von_karman_constant=value["von_karman_constant"],
            gpu_checkpoint_schema=value["gpu_checkpoint_schema"],
        )
        supplied_bytes = _canonical(value)
        rebuilt_bytes = _canonical(rebuilt)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid Q3-H3 prognostic-secondary execution identity") from error
    if dict(value) != rebuilt or supplied_bytes != rebuilt_bytes:
        raise ValueError("Q3-H3 prognostic-secondary execution identity is stale or forged")
    return rebuilt


__all__ = [
    "BACKEND_SELECTION", "BED_BOUNDARY_ROTATION_POLICY",
    "CAPACITY_METHOD_BY_LAW", "CHECKPOINT_POLICY",
    "COMPOSITION_ORDER", "CORIOLIS_DISPOSITIONS", "DIRECTION_POLICY",
    "DEPTH_PAIR_POLICY", "DRY_POLICY", "ENVELOPE_VERSION", "EQUILIBRIUM_ROTATION_POLICY",
    "EXECUTION_KEY", "FIXED_GRID_SCOPE", "GPU_CHECKPOINT_SCHEMA",
    "H2_REFERENCE_SCOPE_ID", "H2_REFERENCE_SEMANTIC_ID",
    "H2_REUSED_PRIMITIVES", "INFLOW_MODE", "INITIAL_STATE_POLICY",
    "INTEGRATED_LEDGER_UNITS",
    "METHOD_ID", "MODEL", "OPEN_BOUNDARY_POLICY", "ORDERED_ROUTE_KEY",
    "PUBLIC_FRAME_POLICY", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "SOURCE_SINK_POLICY", "STATE_SEMANTICS", "TARGET_FORCING_POLICY",
    "TRANSPORT_POLICY",
    "prognostic_secondary_execution_document",
    "validate_prognostic_secondary_execution_document",
]
