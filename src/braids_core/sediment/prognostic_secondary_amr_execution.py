"""Portable Q3-H3/v59 AMR identity, transfer, and scalar-face reference.

This module is the shared numerical contract for the first synchronous AMR
slice described by ``h3_amr/DESIGN.md``.  It is intentionally independent of
the native and CUDA implementations so both backends can be checked against
the same arithmetic and identities.

Only the signed prognostic secondary state ``M = h I`` is implemented here.
The routines do not constitute a live solver route and do not change the
accepted v57/v58 implementations.  Recursive time subcycling and wet/dry AMR
remain outside this contract.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import TypeAlias

import numpy as np

from .prognostic_secondary_nonzero_execution import (
    prognostic_secondary_nonzero_ordered_route,
    validate_prognostic_secondary_nonzero_execution_document,
)


SCHEMA = 1
ENVELOPE_VERSION = 59
PARENT_ENVELOPE_VERSION = 58
CHECKPOINT_SCHEMA = 10
EXECUTION_KEY = "prognostic_secondary_amr_execution_identity"
ORDERED_ROUTE_KEY = "prognostic_secondary_amr_ordered_route_identity"
ROUTE_FINGERPRINT_KEY = "prognostic_secondary_amr_execution_fingerprint"
SCOPE_ID = "braids-q3h3-flow2dh-prognostic-secondary-amr-v1"
TIME_COUPLING = "synchronous_selected_mesh_one_clock_q3h_v1"
CPU_INTEGRATOR = "ssp43"
GPU_INTEGRATOR = "implicit_fwave_midpoint"
SCHMIDT_NUMBER = 0.7
TOPOLOGY_LOCAL_INVALIDATIONS = (
    "hydraulic_predictor",
    "scalar_diffusion_matrix",
    "frozen_betac_segment_field",
)

SEMANTIC_DOCUMENT = {
    "schema": SCHEMA,
    "scope_id": SCOPE_ID,
    "conserved_secondary_state": "signed_M_equals_h_times_I",
    "integrated_inventory": "H_equals_area_times_M",
    "physical_owner": "one_nonoverlapping_active_leaf_representation",
    "restriction": "exact_area_average_without_sign_or_magnitude_limiter",
    "prolongation": (
        "limited_signed_primitive_I_then_directional_capacity_redistribution_"
        "and_exact_parent_AhI_closure"
    ),
    "canonical_face_orientation": "positive_owner_to_neighbour",
    "face_catalog_admission": (
        "hash_bound_selected_leaf_geometry_rejects_duplicate_or_collinear_"
        "overlap"
    ),
    "internal_face_application": "minus_owner_plus_neighbour_once",
    "boundary_ledger_sign": "inward_positive_is_negative_outward_register",
    "advection": "signed_upwind_primitive_intensity_per_physical_segment",
    "advection_trace_gate": (
        "accepted_primitive_inside_stencil_bounds_and_raw_trace_finite_"
        "before_clamp"
    ),
    "diffusion": (
        "one_composite_new_depth_backward_Euler_system_on_physical_segments_"
        "with_reflux_by_construction_and_no_postsolve_patch"
    ),
    "diffusion_geometry": (
        "G_equals_length_times_arithmetic_new_depth_times_D_over_0p7_times_"
        "positive_normal_projected_distance"
    ),
    "pcg_target": "max_absolute_or_relative_times_rhs_norm_without_unit_floor",
    "diffusion_acceptance": (
        "global_equation_norm_then_separate_identity_bound_inventory_and_"
        "register_checks"
    ),
    "betac_momentum_mapping": (
        "delft_velocity_source_h_times_divergence_with_accepted_depth_v1"
    ),
    "time_coupling": TIME_COUPLING,
    "wetness": "fully_wet_only",
    "refinement_ratio": 2,
    "recursive_implicit_subcycling": False,
    "backend_selection": "explicit_user_choice_no_fallback",
    "topology_change": "complete_family_at_accepted_macro_boundary_only",
    "topology_cache_action": list(TOPOLOGY_LOCAL_INVALIDATIONS),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_amr_payload_sha256(value: object, *, name: str) -> str:
    """Hash one JSON-compatible AMR identity payload canonically."""
    if not isinstance(name, str) or not name.strip():
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 canonical payload name must be nonempty"
        )
    try:
        return hashlib.sha256(_canonical(value)).hexdigest()
    except (TypeError, ValueError, OverflowError) as error:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} is not a finite JSON identity payload"
        ) from error


def canonical_amr_instance_fingerprints(
    *,
    topology_document: object,
    face_catalog_document: object,
    boundary_catalog_document: object,
) -> dict[str, str]:
    """Return the one shared CPU/GPU hashing convention for an AMR instance."""
    return {
        "topology_fingerprint": canonical_amr_payload_sha256(
            topology_document, name="topology document"
        ),
        "face_catalog_fingerprint": canonical_amr_payload_sha256(
            face_catalog_document, name="face catalog document"
        ),
        "boundary_catalog_fingerprint": canonical_amr_payload_sha256(
            boundary_catalog_document, name="boundary catalog document"
        ),
    }


SEMANTIC_ID = hashlib.sha256(_canonical(SEMANTIC_DOCUMENT)).hexdigest()


class PrognosticSecondaryAMRError(ValueError):
    """Fail-closed error for the isolated Q3-H3/v59 AMR contract."""


SourceID: TypeAlias = str | int


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be a lowercase SHA-256 digest"
        )
    return value


def _finite(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be finite"
        )
    if positive and result <= 0.0:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be positive"
        )
    if nonnegative and result < 0.0:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be nonnegative"
        )
    return 0.0 if result == 0.0 else result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be a positive integer"
        )
    return value


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be Boolean"
        )
    return value


def _build_execution_body(
    *,
    parent_execution_fingerprint: str,
    source_contract_sha256: str,
    canonical_fixture_sha256: str,
    betac_operator_amendment_sha256: str,
    topology_fingerprint: str,
    face_catalog_fingerprint: str,
    boundary_catalog_fingerprint: str,
    implementation_source_sha256: str,
    backend: str,
    integrator: str,
    cuda_device_id: int | None,
    maximum_level: int,
    refinement_ratio: int,
    dry_tolerance_m: float,
    conservation_tolerance: float,
    gravity_m_s2: float,
    von_karman_constant: float,
    chzmin: float,
    relative_linear_tolerance: float,
    absolute_linear_tolerance: float,
    maximum_linear_iterations: int,
    deterministic_reduction_id: str,
) -> dict[str, object]:
    parent = _sha256(parent_execution_fingerprint, "parent execution fingerprint")
    source_contract = _sha256(source_contract_sha256, "source-contract fingerprint")
    fixture = _sha256(canonical_fixture_sha256, "canonical fixture fingerprint")
    betac_amendment = _sha256(
        betac_operator_amendment_sha256,
        "Betac operator amendment fingerprint",
    )
    topology = _sha256(topology_fingerprint, "topology fingerprint")
    faces = _sha256(face_catalog_fingerprint, "face-catalog fingerprint")
    boundaries = _sha256(
        boundary_catalog_fingerprint, "boundary-catalog fingerprint"
    )
    implementation_source = _sha256(
        implementation_source_sha256, "implementation source fingerprint"
    )
    if backend not in {"cpu", "gpu"}:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 backend must be the explicit CPU or GPU choice"
        )
    required_integrator = CPU_INTEGRATOR if backend == "cpu" else GPU_INTEGRATOR
    if integrator != required_integrator:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {backend} integrator must be {required_integrator}"
        )
    if backend == "gpu":
        if (
            isinstance(cuda_device_id, bool)
            or not isinstance(cuda_device_id, int)
            or cuda_device_id < 0
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 GPU execution requires an explicit CUDA device ID"
            )
    elif cuda_device_id is not None:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 CPU execution may not carry a CUDA device ID"
        )
    levels = _positive_integer(maximum_level, "maximum level")
    ratio = _positive_integer(refinement_ratio, "refinement ratio")
    if ratio != 2:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 refinement ratio must be exactly 2:1"
        )
    if levels < 1:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 AMR requires at least two spatial levels"
        )
    if (
        not isinstance(deterministic_reduction_id, str)
        or not deterministic_reduction_id.strip()
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 deterministic reduction identity must be nonempty"
        )
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_envelope_version": PARENT_ENVELOPE_VERSION,
        "parent_execution_fingerprint": parent,
        "source_contract_sha256": source_contract,
        "canonical_fixture_sha256": fixture,
        "betac_operator_amendment_sha256": betac_amendment,
        "topology_fingerprint": topology,
        "face_catalog_fingerprint": faces,
        "boundary_catalog_fingerprint": boundaries,
        "implementation_source_sha256": implementation_source,
        "grid_scope": "balanced_mixed_level_fixed_cartesian_fully_wet",
        "time_coupling": TIME_COUPLING,
        "backend_selection": "explicit_user_choice",
        "backend": backend,
        "integrator": integrator,
        "cuda_device_id": cuda_device_id,
        "automatic_cpu_fallback": False,
        "refinement_ratio": ratio,
        "maximum_level": levels,
        "dry_tolerance_m": _finite(
            dry_tolerance_m, "dry tolerance", positive=True
        ),
        "conservation_tolerance": _finite(
            conservation_tolerance, "conservation tolerance", positive=True
        ),
        "gravity_m_s2": _finite(gravity_m_s2, "gravity", positive=True),
        "von_karman_constant": _finite(
            von_karman_constant, "von Karman constant", positive=True
        ),
        "chzmin": _finite(chzmin, "minimum Chezy", positive=True),
        "relative_linear_tolerance": _finite(
            relative_linear_tolerance,
            "relative linear tolerance",
            nonnegative=True,
        ),
        "absolute_linear_tolerance": _finite(
            absolute_linear_tolerance,
            "absolute linear tolerance",
            nonnegative=True,
        ),
        "maximum_linear_iterations": _positive_integer(
            maximum_linear_iterations, "maximum linear iterations"
        ),
        "deterministic_reduction_id": deterministic_reduction_id,
        "state_ownership": SEMANTIC_DOCUMENT["physical_owner"],
        "face_register_components": [
            "J_adv_hI_m4_per_s",
            "J_diff_hI_m4_per_s",
            "K_betac_velocity_x_m3_per_s",
            "K_betac_velocity_y_m3_per_s",
        ],
        "betac_momentum_mapping": (
            "cell_delta_Ahu_equals_accepted_depth_times_signed_K_betac"
        ),
        "recursive_implicit_subcycling": False,
        "wet_dry_amr": False,
        "failure_policy": "reject_before_mutation_or_restore_every_accepted_byte",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }


def prognostic_secondary_amr_execution_document(
    parent: Mapping[str, object],
    *,
    source_contract_sha256: str,
    canonical_fixture_sha256: str,
    betac_operator_amendment_sha256: str,
    topology_fingerprint: str,
    face_catalog_fingerprint: str,
    boundary_catalog_fingerprint: str,
    implementation_source_sha256: str,
    backend: str,
    integrator: str,
    cuda_device_id: int | None,
    maximum_level: int,
    refinement_ratio: int = 2,
    dry_tolerance_m: float,
    conservation_tolerance: float,
    gravity_m_s2: float,
    von_karman_constant: float,
    chzmin: float,
    relative_linear_tolerance: float,
    absolute_linear_tolerance: float,
    maximum_linear_iterations: int,
    deterministic_reduction_id: str,
) -> dict[str, object]:
    """Build a content-addressed v59 identity from an authenticated v58."""
    validated = validate_prognostic_secondary_nonzero_execution_document(parent)
    body = _build_execution_body(
        parent_execution_fingerprint=validated["fingerprint"],
        source_contract_sha256=source_contract_sha256,
        canonical_fixture_sha256=canonical_fixture_sha256,
        betac_operator_amendment_sha256=betac_operator_amendment_sha256,
        topology_fingerprint=topology_fingerprint,
        face_catalog_fingerprint=face_catalog_fingerprint,
        boundary_catalog_fingerprint=boundary_catalog_fingerprint,
        implementation_source_sha256=implementation_source_sha256,
        backend=backend,
        integrator=integrator,
        cuda_device_id=cuda_device_id,
        maximum_level=maximum_level,
        refinement_ratio=refinement_ratio,
        dry_tolerance_m=dry_tolerance_m,
        conservation_tolerance=conservation_tolerance,
        gravity_m_s2=gravity_m_s2,
        von_karman_constant=von_karman_constant,
        chzmin=chzmin,
        relative_linear_tolerance=relative_linear_tolerance,
        absolute_linear_tolerance=absolute_linear_tolerance,
        maximum_linear_iterations=maximum_linear_iterations,
        deterministic_reduction_id=deterministic_reduction_id,
    )
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_prognostic_secondary_amr_execution_document(
    value: Mapping[str, object],
    *,
    parent: Mapping[str, object] | None = None,
    source_contract_sha256: str | None = None,
    canonical_fixture_sha256: str | None = None,
    betac_operator_amendment_sha256: str | None = None,
    topology_fingerprint: str | None = None,
    face_catalog_fingerprint: str | None = None,
    boundary_catalog_fingerprint: str | None = None,
    implementation_source_sha256: str | None = None,
) -> dict[str, object]:
    """Rebuild and byte-compare an untrusted v59 identity before mutation."""
    if not isinstance(value, Mapping):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 execution identity must be an object"
        )
    try:
        body = _build_execution_body(
            parent_execution_fingerprint=value["parent_execution_fingerprint"],
            source_contract_sha256=value["source_contract_sha256"],
            canonical_fixture_sha256=value["canonical_fixture_sha256"],
            betac_operator_amendment_sha256=value[
                "betac_operator_amendment_sha256"
            ],
            topology_fingerprint=value["topology_fingerprint"],
            face_catalog_fingerprint=value["face_catalog_fingerprint"],
            boundary_catalog_fingerprint=value["boundary_catalog_fingerprint"],
            implementation_source_sha256=value["implementation_source_sha256"],
            backend=value["backend"],
            integrator=value["integrator"],
            cuda_device_id=value["cuda_device_id"],
            maximum_level=value["maximum_level"],
            refinement_ratio=value["refinement_ratio"],
            dry_tolerance_m=value["dry_tolerance_m"],
            conservation_tolerance=value["conservation_tolerance"],
            gravity_m_s2=value["gravity_m_s2"],
            von_karman_constant=value["von_karman_constant"],
            chzmin=value["chzmin"],
            relative_linear_tolerance=value["relative_linear_tolerance"],
            absolute_linear_tolerance=value["absolute_linear_tolerance"],
            maximum_linear_iterations=value["maximum_linear_iterations"],
            deterministic_reduction_id=value["deterministic_reduction_id"],
        )
        rebuilt = {
            **body,
            "fingerprint": hashlib.sha256(_canonical(body)).hexdigest(),
        }
        if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 execution identity is stale or forged"
            )
        expected = {
            "source_contract_sha256": source_contract_sha256,
            "canonical_fixture_sha256": canonical_fixture_sha256,
            "betac_operator_amendment_sha256": (
                betac_operator_amendment_sha256
            ),
            "topology_fingerprint": topology_fingerprint,
            "face_catalog_fingerprint": face_catalog_fingerprint,
            "boundary_catalog_fingerprint": boundary_catalog_fingerprint,
            "implementation_source_sha256": implementation_source_sha256,
        }
        for key, expected_value in expected.items():
            if expected_value is not None and rebuilt[key] != _sha256(
                expected_value, key.replace("_", " ")
            ):
                raise PrognosticSecondaryAMRError(
                    f"Q3-H3/v59 {key.replace('_', ' ')} differs"
                )
        if parent is not None:
            validated_parent = (
                validate_prognostic_secondary_nonzero_execution_document(parent)
            )
            if rebuilt["parent_execution_fingerprint"] != validated_parent[
                "fingerprint"
            ]:
                raise PrognosticSecondaryAMRError(
                    "Q3-H3/v59 parent execution identity differs"
                )
    except PrognosticSecondaryAMRError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise PrognosticSecondaryAMRError(
            "Invalid Q3-H3/v59 execution identity"
        ) from error
    return rebuilt


def prognostic_secondary_amr_ordered_route(
    document: Mapping[str, object], parent: Mapping[str, object]
) -> dict[str, object]:
    """Append v59 to the exact authenticated five-part v58 route."""
    identity = validate_prognostic_secondary_amr_execution_document(
        document, parent=parent
    )
    parent_route = prognostic_secondary_nonzero_ordered_route(parent)
    ordered_parts = [*parent_route["ordered_parts"], ROUTE_FINGERPRINT_KEY]
    return {
        "ordered_parts": ordered_parts,
        **{
            key: value
            for key, value in parent_route.items()
            if key != "ordered_parts"
        },
        ROUTE_FINGERPRINT_KEY: identity["fingerprint"],
    }


def validate_prognostic_secondary_amr_ordered_route(
    value: Mapping[str, object],
    document: Mapping[str, object],
    parent: Mapping[str, object],
) -> dict[str, object]:
    expected = prognostic_secondary_amr_ordered_route(document, parent)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 ordered route identity differs"
        )
    return expected


def _float_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be a {ndim}-D numeric array"
        )
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be finite"
        )
    return result


def _scaled_close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def active_signed_hi_inventory(
    areas_m2: object, signed_hi_m2_s: object, active: object
) -> float:
    """Sum only active, non-overlapping ``A*hI`` owners."""
    areas = _float_array(areas_m2, "areas", ndim=1)
    signed_hi = _float_array(signed_hi_m2_s, "signed hI", ndim=1)
    mask_raw = np.asarray(active)
    if mask_raw.dtype.kind != "b" or mask_raw.ndim != 1:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 active ownership mask must be one-dimensional Boolean"
        )
    mask = np.array(mask_raw, dtype=bool, copy=False)
    if areas.shape != signed_hi.shape or mask.shape != areas.shape:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 active inventory arrays have incompatible shapes"
        )
    if np.any(areas <= 0.0):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 active inventory areas must be positive"
        )
    return math.fsum(
        float(area * value) for area, value in zip(areas[mask], signed_hi[mask])
    )


def selected_family_signed_hi_inventory(
    *,
    parent_area_m2: float,
    parent_signed_hi_m2_s: float,
    parent_active: bool,
    child_areas_m2: object,
    child_signed_hi_m2_s: object,
    child_active: object,
    refinement_ratio: int = 2,
) -> float:
    """Return one family inventory while rejecting overlapping ownership."""
    parent_is_active = _strict_bool(parent_active, "parent active flag")
    parent_area = _finite(parent_area_m2, "parent area", positive=True)
    parent_hi = _finite(parent_signed_hi_m2_s, "parent signed hI")
    ratio = _positive_integer(refinement_ratio, "refinement ratio")
    if ratio != 2:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 refinement ratio must be exactly 2:1"
        )
    areas = _float_array(child_areas_m2, "child areas", ndim=1)
    values = _float_array(child_signed_hi_m2_s, "child signed hI", ndim=1)
    raw_mask = np.asarray(child_active)
    if (
        areas.size != ratio**2
        or values.shape != areas.shape
        or raw_mask.dtype.kind != "b"
        or raw_mask.shape != areas.shape
        or np.any(areas <= 0.0)
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 selected family arrays are incomplete or incompatible"
        )
    mask = np.asarray(raw_mask, dtype=bool)
    if parent_is_active:
        if np.any(mask):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 parent and descendant representations overlap"
            )
        return parent_area * parent_hi
    if not np.all(mask):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 refined parent requires its complete active child family"
        )
    return active_signed_hi_inventory(areas, values, mask)


def restrict_signed_hi_family(
    child_areas_m2: object,
    child_signed_hi_m2_s: object,
    *,
    parent_area_m2: float,
    refinement_ratio: int = 2,
    complete_family: bool = True,
    pending_register: bool = False,
    overlaps_selected_leaf: bool = False,
    changes_finest_boundary_owner: bool = False,
    closure_tolerance: float = 1.0e-12,
) -> float:
    """Area-restrict one complete signed ``hI`` family without limiting."""
    ratio = _positive_integer(refinement_ratio, "refinement ratio")
    if ratio != 2:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 refinement ratio must be exactly 2:1"
        )
    guards = {
        "complete family": complete_family,
        "pending register": pending_register,
        "overlapping selected leaf": overlaps_selected_leaf,
        "changed finest boundary owner": changes_finest_boundary_owner,
    }
    for name, value in guards.items():
        _strict_bool(value, name)
    if not complete_family:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 coarsening requires a complete sibling family"
        )
    if pending_register:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 coarsening is forbidden with a pending face register"
        )
    if overlaps_selected_leaf:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 selected AMR owners may not overlap"
        )
    if changes_finest_boundary_owner:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 coarsening may not change a finest boundary owner"
        )
    areas = _float_array(child_areas_m2, "child areas", ndim=1)
    values = _float_array(child_signed_hi_m2_s, "child signed hI", ndim=1)
    if areas.shape != values.shape or areas.size != ratio**2:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 coarsening requires one complete 2-D sibling family"
        )
    if np.any(areas <= 0.0):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 child areas must be positive"
        )
    parent_area = _finite(parent_area_m2, "parent area", positive=True)
    tolerance = _finite(
        closure_tolerance, "closure tolerance", positive=True
    )
    area_sum = math.fsum(float(value) for value in areas)
    if not _scaled_close(area_sum, parent_area, tolerance):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 child areas do not close to the parent area"
        )
    inventory = math.fsum(
        float(area * value) for area, value in zip(areas, values)
    )
    parent_value = inventory / parent_area
    if not _scaled_close(parent_area * parent_value, inventory, tolerance):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 signed hI restriction failed family closure"
        )
    return parent_value


@dataclass(frozen=True)
class SignedHIProlongation:
    child_signed_hi_m2_s: np.ndarray
    child_intensity: np.ndarray
    target_inventory_m4_s: float
    child_inventory_m4_s: float
    residual_before_redistribution_m4_s: float
    residual_after_redistribution_m4_s: float
    invalidated_caches: tuple[str, ...] = TOPOLOGY_LOCAL_INVALIDATIONS


def prolong_signed_hi_family(
    *,
    parent_area_m2: float,
    parent_depth_m: float,
    parent_signed_hi_m2_s: float,
    child_areas_m2: object,
    child_depth_m: object,
    child_offsets_m: object,
    primitive_gradient_per_m: object,
    primitive_bounds: Sequence[float],
    dry_tolerance_m: float,
    refinement_ratio: int = 2,
    closure_tolerance: float = 1.0e-12,
) -> SignedHIProlongation:
    """Prolong signed ``hI`` with bounded primitive directional capacity.

    Inputs are copied before any work.  Consequently a failed transfer cannot
    mutate accepted parent or child/cache bytes.
    """
    ratio = _positive_integer(refinement_ratio, "refinement ratio")
    if ratio != 2:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 refinement ratio must be exactly 2:1"
        )
    parent_area = _finite(parent_area_m2, "parent area", positive=True)
    parent_depth = _finite(parent_depth_m, "parent depth", positive=True)
    parent_hi = _finite(parent_signed_hi_m2_s, "parent signed hI")
    dry = _finite(dry_tolerance_m, "dry tolerance", positive=True)
    tolerance = _finite(
        closure_tolerance, "closure tolerance", positive=True
    )
    if parent_depth <= dry:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 parent is outside the fully wet AMR scope"
        )
    areas = _float_array(child_areas_m2, "child areas", ndim=1)
    depths = _float_array(child_depth_m, "child depths", ndim=1)
    offsets = _float_array(child_offsets_m, "child offsets", ndim=2)
    gradient = _float_array(
        primitive_gradient_per_m, "primitive intensity gradient", ndim=1
    )
    if (
        areas.size != ratio**2
        or depths.shape != areas.shape
        or offsets.shape != (areas.size, 2)
        or gradient.shape != (2,)
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 prolongation requires one complete 2-D sibling family"
        )
    if np.any(areas <= 0.0) or np.any(depths <= dry):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 prolongation requires positive areas and fully wet children"
        )
    area_sum = math.fsum(float(value) for value in areas)
    if not _scaled_close(area_sum, parent_area, tolerance):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 child areas do not close to the parent area"
        )
    parent_water = parent_area * parent_depth
    child_water = math.fsum(
        float(area * depth) for area, depth in zip(areas, depths)
    )
    if not _scaled_close(child_water, parent_water, tolerance):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 staged child water does not close to the parent volume"
        )
    bounds = _float_array(
        primitive_bounds, "primitive stencil bounds", ndim=1
    )
    if bounds.shape != (2,):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 primitive bounds must contain exactly two values"
        )
    lower = float(bounds[0])
    upper = float(bounds[1])
    if lower > upper:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 primitive bounds are reversed"
        )
    parent_intensity = parent_hi / parent_depth
    bound_scale = max(1.0, abs(lower), abs(upper), abs(parent_intensity))
    if (
        parent_intensity < lower - tolerance * bound_scale
        or parent_intensity > upper + tolerance * bound_scale
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 parent intensity lies outside the signed stencil bounds"
        )
    intensity = np.clip(
        parent_intensity + offsets @ gradient, lower, upper
    )
    weights = areas * depths
    target = parent_area * parent_hi

    def inventory() -> float:
        return math.fsum(
            float(weight * value) for weight, value in zip(weights, intensity)
        )

    residual_before = target - inventory()
    residual = residual_before
    if residual != 0.0:
        if residual > 0.0:
            capacities = weights * np.maximum(0.0, upper - intensity)
        else:
            capacities = weights * np.maximum(0.0, intensity - lower)
        capacity = math.fsum(float(value) for value in capacities)
        if abs(residual) > capacity + tolerance * max(1.0, abs(target)):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 directional capacity cannot close parent signed hI"
            )
        if capacity > 0.0:
            fraction = min(1.0, abs(residual) / capacity)
            if residual > 0.0:
                intensity += fraction * np.maximum(0.0, upper - intensity)
            else:
                intensity -= fraction * np.maximum(0.0, intensity - lower)

    # Put floating-point residue in the child with most capacity in the
    # required direction.  Iterate only over this four-child family.
    for _ in range(8):
        residual = target - inventory()
        if residual == 0.0:
            break
        if residual > 0.0:
            remaining = weights * np.maximum(0.0, upper - intensity)
            direction = 1.0
        else:
            remaining = weights * np.maximum(0.0, intensity - lower)
            direction = -1.0
        index = int(np.argmax(remaining))
        if remaining[index] <= 0.0:
            if _scaled_close(target, inventory(), tolerance):
                break
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 no directional capacity remains for roundoff closure"
            )
        correction = min(abs(residual), float(remaining[index]))
        intensity[index] += direction * correction / weights[index]

    child_hi = depths * intensity
    child_inventory = math.fsum(
        float(area * value) for area, value in zip(areas, child_hi)
    )
    residual_after = target - child_inventory
    check_scale = max(1.0, abs(lower), abs(upper))
    if (
        np.any(intensity < lower - tolerance * check_scale)
        or np.any(intensity > upper + tolerance * check_scale)
        or not _scaled_close(child_inventory, target, tolerance)
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 signed hI prolongation failed bound or closure checks"
        )
    intensity.setflags(write=False)
    child_hi.setflags(write=False)
    return SignedHIProlongation(
        child_signed_hi_m2_s=child_hi,
        child_intensity=intensity,
        target_inventory_m4_s=target,
        child_inventory_m4_s=child_inventory,
        residual_before_redistribution_m4_s=residual_before,
        residual_after_redistribution_m4_s=residual_after,
    )


@dataclass(frozen=True)
class PhysicalFaceSegment:
    source_id: SourceID
    owner: int
    neighbour: int | None
    normal: tuple[float, float]
    midpoint_m: tuple[float, float]
    length_m: float
    level: int
    owner_level: int
    neighbour_level: int | None
    owner_center_m: tuple[float, float]
    neighbour_center_m: tuple[float, float] | None = None
    logical_boundary_id: SourceID | None = None
    boundary_kind: str = "internal"


def _valid_source_id(value: object, name: str) -> SourceID:
    if isinstance(value, bool):
        raise PrognosticSecondaryAMRError(f"Q3-H3/v59 {name} is invalid")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value:
        return value
    raise PrognosticSecondaryAMRError(f"Q3-H3/v59 {name} is invalid")


def _pair(value: object, name: str) -> tuple[float, float]:
    result = _float_array(value, name, ndim=1)
    if result.shape != (2,):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must contain two coordinates"
        )
    return float(result[0]), float(result[1])


def validate_physical_face_segments(
    segments: Sequence[PhysicalFaceSegment],
    *,
    cell_count: int,
    maximum_level: int,
) -> tuple[PhysicalFaceSegment, ...]:
    """Validate one ordered, oriented, non-overlapping physical-face catalog.

    This is still a selected-leaf contract: the caller must derive the catalog
    from the active AMR representation.  The geometry checks below reject
    duplicate and collinearly overlapping parent/child representatives, but do
    not infer AMR family membership from coordinates.
    """
    count = _positive_integer(cell_count, "cell count")
    max_level = _positive_integer(maximum_level, "maximum level")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 face segments must be an ordered sequence"
        )
    result: list[PhysicalFaceSegment] = []
    seen: set[tuple[type, SourceID]] = set()
    previous_source_id: SourceID | None = None
    for raw in segments:
        if not isinstance(raw, PhysicalFaceSegment):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face catalog contains a non-segment value"
            )
        source_id = _valid_source_id(raw.source_id, "face source ID")
        typed_id = (type(source_id), source_id)
        if typed_id in seen:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 canonical face source IDs must be unique"
            )
        seen.add(typed_id)
        if previous_source_id is not None and (
            type(source_id) is not type(previous_source_id)
            or not previous_source_id < source_id
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 canonical face source IDs must be strictly ordered"
            )
        previous_source_id = source_id
        if (
            isinstance(raw.owner, bool)
            or not isinstance(raw.owner, int)
            or not 0 <= raw.owner < count
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face owner is outside the active leaf catalog"
            )
        if raw.boundary_kind not in {"internal", "open", "wall"}:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face boundary kind is unsupported"
            )
        if raw.boundary_kind == "internal":
            if (
                isinstance(raw.neighbour, bool)
                or not isinstance(raw.neighbour, int)
                or not 0 <= raw.neighbour < count
                or raw.neighbour == raw.owner
                or raw.neighbour_center_m is None
                or raw.logical_boundary_id is not None
            ):
                raise PrognosticSecondaryAMRError(
                    "Q3-H3/v59 internal face ownership is invalid"
                )
        else:
            if raw.neighbour is not None or raw.neighbour_center_m is not None:
                raise PrognosticSecondaryAMRError(
                    "Q3-H3/v59 exterior face may not have a neighbour"
                )
            if raw.boundary_kind == "open":
                _valid_source_id(raw.logical_boundary_id, "logical boundary ID")
                if raw.level != max_level:
                    raise PrognosticSecondaryAMRError(
                        "Q3-H3/v59 open boundary owner must be on the finest level"
                    )
            elif raw.logical_boundary_id is not None:
                raise PrognosticSecondaryAMRError(
                    "Q3-H3/v59 wall face may not own an open-boundary ID"
                )
        if (
            isinstance(raw.level, bool)
            or not isinstance(raw.level, int)
            or raw.level < 0
            or raw.level > max_level
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face level is invalid"
            )
        if (
            isinstance(raw.owner_level, bool)
            or not isinstance(raw.owner_level, int)
            or not 0 <= raw.owner_level <= max_level
            or (
                raw.neighbour is None
                and raw.neighbour_level is not None
            )
            or (
                raw.neighbour is not None
                and (
                    isinstance(raw.neighbour_level, bool)
                    or not isinstance(raw.neighbour_level, int)
                    or not 0 <= raw.neighbour_level <= max_level
                )
            )
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face endpoint levels are invalid"
            )
        touching_level = (
            raw.owner_level
            if raw.neighbour_level is None
            else max(raw.owner_level, raw.neighbour_level)
        )
        if raw.level != touching_level:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face level must equal the finest touching cell level"
            )
        if (
            raw.neighbour_level is not None
            and abs(raw.owner_level - raw.neighbour_level) > 1
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 internal face endpoints violate 2:1 balance"
            )
        normal = _pair(raw.normal, "face normal")
        if not math.isclose(math.hypot(*normal), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face normal must be unit length"
            )
        midpoint = _pair(raw.midpoint_m, "face midpoint")
        owner_center = _pair(raw.owner_center_m, "owner centre")
        neighbour_center = (
            None
            if raw.neighbour_center_m is None
            else _pair(raw.neighbour_center_m, "neighbour centre")
        )
        length = _finite(raw.length_m, "face length", positive=True)
        if neighbour_center is None:
            direction = (
                midpoint[0] - owner_center[0],
                midpoint[1] - owner_center[1],
            )
        else:
            direction = (
                neighbour_center[0] - owner_center[0],
                neighbour_center[1] - owner_center[1],
            )
        projected_distance = direction[0] * normal[0] + direction[1] * normal[1]
        orientation_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0, math.hypot(*direction)
        )
        if projected_distance <= orientation_tolerance:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face normal is not oriented from owner toward "
                "neighbour or exterior"
            )
        result.append(
            PhysicalFaceSegment(
                source_id=source_id,
                owner=raw.owner,
                neighbour=raw.neighbour,
                normal=normal,
                midpoint_m=midpoint,
                length_m=length,
                level=raw.level,
                owner_level=raw.owner_level,
                neighbour_level=raw.neighbour_level,
                owner_center_m=owner_center,
                neighbour_center_m=neighbour_center,
                logical_boundary_id=raw.logical_boundary_id,
                boundary_kind=raw.boundary_kind,
            )
        )

    # Reject the cached-parent-plus-fine-representatives defect without an
    # O(N^2) all-pairs scan.  Mesh-derived collinear faces share a canonical
    # tangent and line offset; intervals may touch at an endpoint but may not
    # overlap by a positive physical length.
    if result:
        geometry_scale = max(
            1.0,
            *(abs(value) for segment in result for value in segment.midpoint_m),
            *(segment.length_m for segment in result),
        )
        coordinate_tolerance = 64.0 * np.finfo(np.float64).eps * geometry_scale
        angle_tolerance = 64.0 * np.finfo(np.float64).eps
        line_groups: dict[tuple[int, int, int], list[tuple[float, float, SourceID]]] = {}
        for segment in result:
            nx, ny = segment.normal
            tx, ty = -ny, nx
            if tx < 0.0 or (tx == 0.0 and ty < 0.0):
                tx, ty = -tx, -ty
            line_normal_x, line_normal_y = -ty, tx
            line_offset = (
                line_normal_x * segment.midpoint_m[0]
                + line_normal_y * segment.midpoint_m[1]
            )
            centre_coordinate = (
                tx * segment.midpoint_m[0] + ty * segment.midpoint_m[1]
            )
            key = (
                round(tx / angle_tolerance),
                round(ty / angle_tolerance),
                round(line_offset / coordinate_tolerance),
            )
            line_groups.setdefault(key, []).append(
                (
                    centre_coordinate - 0.5 * segment.length_m,
                    centre_coordinate + 0.5 * segment.length_m,
                    segment.source_id,
                )
            )
        for intervals in line_groups.values():
            intervals.sort(key=lambda item: (item[0], item[1]))
            previous_end = -math.inf
            previous_source: SourceID | None = None
            for start, end, source_id in intervals:
                if start < previous_end - coordinate_tolerance:
                    raise PrognosticSecondaryAMRError(
                        "Q3-H3/v59 physical face geometry overlaps between "
                        f"source IDs {previous_source!r} and {source_id!r}"
                    )
                if end > previous_end:
                    previous_end = end
                    previous_source = source_id
    return tuple(result)


def signed_hi_upwind_transfer(
    *,
    dt_s: float,
    length_m: float,
    normal_discharge_per_width_m2_s: float,
    owner_intensity: float,
    neighbour_intensity: float,
) -> float:
    """Return outward-positive, time-integrated signed-``hI`` advection."""
    dt = _finite(dt_s, "time step", nonnegative=True)
    length = _finite(length_m, "face length", positive=True)
    discharge = _finite(
        normal_discharge_per_width_m2_s, "normal discharge per width"
    )
    owner = _finite(owner_intensity, "owner intensity")
    neighbour = _finite(neighbour_intensity, "neighbour intensity")
    trace = owner if discharge >= 0.0 else neighbour
    return dt * length * discharge * trace


def reconstructed_signed_intensity_trace(
    *,
    cell_intensity: float,
    limited_gradient_per_m: object,
    cell_center_m: object,
    segment_midpoint_m: object,
    primitive_bounds: object,
) -> float:
    """Reconstruct one bounded signed primitive trace at one segment.

    A coarse cell sharing a face with several fine cells must call this once
    for every fine segment midpoint.  Reusing one parent-face-centre value is
    deliberately impossible through this API because the midpoint is an
    explicit required input.
    """
    centre_value = _finite(cell_intensity, "cell primitive intensity")
    gradient = _float_array(
        limited_gradient_per_m, "limited primitive gradient", ndim=1
    )
    centre = _float_array(cell_center_m, "cell centre", ndim=1)
    midpoint = _float_array(segment_midpoint_m, "segment midpoint", ndim=1)
    bounds = _float_array(primitive_bounds, "primitive stencil bounds", ndim=1)
    if gradient.shape != (2,) or centre.shape != (2,) or midpoint.shape != (2,):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 trace reconstruction requires two-dimensional geometry"
        )
    if bounds.shape != (2,) or bounds[0] > bounds[1]:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 primitive trace bounds are invalid"
        )
    bound_tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, abs(float(bounds[0])), abs(float(bounds[1])), abs(centre_value)
    )
    if (centre_value < bounds[0] - bound_tolerance or
            centre_value > bounds[1] + bound_tolerance):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 accepted cell primitive is outside its stencil bounds"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        raw = centre_value + float(np.dot(gradient, midpoint - centre))
    if not math.isfinite(raw):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 reconstructed primitive trace is nonfinite"
        )
    return min(float(bounds[1]), max(float(bounds[0]), raw))


def _mapping_values(
    value: Mapping[SourceID, object],
    ids: tuple[SourceID, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} must be keyed by stable face source ID"
        )
    expected = {(type(source_id), source_id) for source_id in ids}
    actual = {
        (type(key), _valid_source_id(key, f"{name} face source ID"))
        for key in value
    }
    if actual != expected:
        raise PrognosticSecondaryAMRError(
            f"Q3-H3/v59 {name} face IDs differ from the canonical catalog"
        )
    return np.array(
        [_finite(value[source_id], name) for source_id in ids],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CanonicalHIRegister:
    segments: tuple[PhysicalFaceSegment, ...]
    advective_m4_s: np.ndarray
    diffusive_m4_s: np.ndarray
    betac_velocity_integral_m3_s: np.ndarray
    digest: str

    def cell_increment(self, cell_count: int, component: str) -> np.ndarray:
        count = _positive_integer(cell_count, "cell count")
        if component == "advective":
            values = self.advective_m4_s
        elif component == "diffusive":
            values = self.diffusive_m4_s
        elif component == "combined":
            values = self.advective_m4_s + self.diffusive_m4_s
        else:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 register component must be advective, diffusive, or combined"
            )
        change = np.zeros(count, dtype=np.float64)
        for segment, transfer in zip(self.segments, values):
            change[segment.owner] -= transfer
            if segment.neighbour is not None:
                change[segment.neighbour] += transfer
        return change

    def boundary_ledger(self, component: str) -> dict[SourceID, float]:
        if component == "advective":
            values = self.advective_m4_s
        elif component == "diffusive":
            values = self.diffusive_m4_s
        elif component == "combined":
            values = self.advective_m4_s + self.diffusive_m4_s
        else:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 boundary component must be advective, diffusive, or combined"
            )
        result: dict[SourceID, float] = {}
        for segment, transfer in zip(self.segments, values):
            if segment.boundary_kind == "open":
                boundary = segment.logical_boundary_id
                assert boundary is not None
                result[boundary] = result.get(boundary, 0.0) - float(transfer)
        return result

    def betac_momentum_increment(
        self, cell_count: int, accepted_depth_m: object
    ) -> np.ndarray:
        """Map Delft's velocity source to integrated ``A*h*u`` increments."""
        count = _positive_integer(cell_count, "cell count")
        depth = _float_array(
            accepted_depth_m, "accepted Betac depth", ndim=1
        )
        if depth.shape != (count,) or np.any(depth <= 0.0):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 accepted Betac depth must be positive per cell"
            )
        change = np.zeros((2, count), dtype=np.float64)
        for segment, velocity_integral in zip(
            self.segments, self.betac_velocity_integral_m3_s
        ):
            change[:, segment.owner] -= (
                depth[segment.owner] * velocity_integral
            )
            if segment.neighbour is not None:
                change[:, segment.neighbour] += (
                    depth[segment.neighbour] * velocity_integral
                )
        return change

    def betac_boundary_momentum_ledger(
        self, accepted_depth_m: object
    ) -> dict[SourceID, tuple[float, float]]:
        """Return inward-positive boundary ``A*h*u`` source increments."""
        if not self.segments:
            return {}
        count = max(
            max(
                segment.owner,
                segment.owner if segment.neighbour is None else segment.neighbour,
            )
            for segment in self.segments
        ) + 1
        depth = _float_array(
            accepted_depth_m, "accepted Betac depth", ndim=1
        )
        if depth.shape != (count,) or np.any(depth <= 0.0):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 accepted Betac depth must be positive per cell"
            )
        result: dict[SourceID, np.ndarray] = {}
        for segment, velocity_integral in zip(
            self.segments, self.betac_velocity_integral_m3_s
        ):
            if segment.boundary_kind == "open":
                boundary = segment.logical_boundary_id
                assert boundary is not None
                row = result.setdefault(boundary, np.zeros(2, dtype=np.float64))
                row -= depth[segment.owner] * velocity_integral
        return {
            key: (float(value[0]), float(value[1]))
            for key, value in result.items()
        }


def canonical_hi_register(
    segments: Sequence[PhysicalFaceSegment],
    *,
    cell_count: int,
    maximum_level: int,
    advective_by_source_id: Mapping[SourceID, object],
    diffusive_by_source_id: Mapping[SourceID, object],
    betac_velocity_integral_by_source_id: (
        Mapping[SourceID, Sequence[float]] | None
    ) = None,
) -> CanonicalHIRegister:
    """Freeze one once-per-segment register and its content digest."""
    canonical_segments = validate_physical_face_segments(
        segments, cell_count=cell_count, maximum_level=maximum_level
    )
    ids = tuple(segment.source_id for segment in canonical_segments)
    advective = _mapping_values(
        advective_by_source_id, ids, "advective register"
    )
    diffusive = _mapping_values(
        diffusive_by_source_id, ids, "diffusive register"
    )
    if betac_velocity_integral_by_source_id is None:
        betac = np.zeros((len(ids), 2), dtype=np.float64)
    else:
        expected = {(type(source_id), source_id) for source_id in ids}
        if not isinstance(betac_velocity_integral_by_source_id, Mapping):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 Betac register must be keyed by stable face source ID"
            )
        actual = {
            (type(key), _valid_source_id(key, "Betac register face source ID"))
            for key in betac_velocity_integral_by_source_id
        }
        if actual != expected:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 Betac register face IDs differ from the canonical catalog"
            )
        rows = [
            _pair(
                betac_velocity_integral_by_source_id[source_id],
                "Betac velocity-integral register row",
            )
            for source_id in ids
        ]
        betac = np.asarray(rows, dtype=np.float64).reshape((-1, 2))
    for index, segment in enumerate(canonical_segments):
        if segment.boundary_kind == "wall" and (
            advective[index] != 0.0
            or diffusive[index] != 0.0
            or np.any(betac[index] != 0.0)
        ):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 wall segment register must be exactly zero"
            )
    payload = {
        "segments": [
            {
                "source_id": segment.source_id,
                "owner": segment.owner,
                "neighbour": segment.neighbour,
                "normal": list(segment.normal),
                "midpoint_m": list(segment.midpoint_m),
                "length_m": segment.length_m,
                "level": segment.level,
                "owner_level": segment.owner_level,
                "neighbour_level": segment.neighbour_level,
                "owner_center_m": list(segment.owner_center_m),
                "neighbour_center_m": (
                    None
                    if segment.neighbour_center_m is None
                    else list(segment.neighbour_center_m)
                ),
                "logical_boundary_id": segment.logical_boundary_id,
                "boundary_kind": segment.boundary_kind,
            }
            for segment in canonical_segments
        ],
        "advective_m4_s": advective.tolist(),
        "diffusive_m4_s": diffusive.tolist(),
        "betac_velocity_integral_m3_s": betac.tolist(),
    }
    for array in (advective, diffusive, betac):
        array.setflags(write=False)
    return CanonicalHIRegister(
        segments=canonical_segments,
        advective_m4_s=advective,
        diffusive_m4_s=diffusive,
        betac_velocity_integral_m3_s=betac,
        digest=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


@dataclass(frozen=True)
class CompositeHIDiffusionResult:
    signed_hi_after_advection_m2_s: np.ndarray
    intensity_after_diffusion: np.ndarray
    signed_hi_after_diffusion_m2_s: np.ndarray
    conductance_by_source_id: dict[SourceID, float]
    register: CanonicalHIRegister
    inventory_before_m4_s: float
    inventory_after_advection_m4_s: float
    inventory_after_diffusion_m4_s: float
    weighted_energy_before: float
    weighted_energy_after: float
    weighted_variance_before: float
    weighted_variance_after: float
    construction: str = "single_composite_backward_euler_no_postsolve_patch"


def _weighted_moments(
    intensity: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    total_weight = math.fsum(float(value) for value in weights)
    mean = math.fsum(
        float(weight * value) for weight, value in zip(weights, intensity)
    ) / total_weight
    energy = 0.5 * math.fsum(
        float(weight * value * value)
        for weight, value in zip(weights, intensity)
    )
    variance = math.fsum(
        float(weight * (value - mean) ** 2)
        for weight, value in zip(weights, intensity)
    )
    return energy, variance


def composite_hi_backward_euler(
    *,
    areas_m2: object,
    new_depth_m: object,
    signed_hi_initial_m2_s: object,
    segments: Sequence[PhysicalFaceSegment],
    advective_by_source_id: Mapping[SourceID, object],
    dt_s: float,
    scalar_diffusivity_m2_s: float,
    maximum_level: int,
    dry_tolerance_m: float,
    open_intensity_by_source_id: Mapping[SourceID, object] | None = None,
    open_depth_by_source_id: Mapping[SourceID, object] | None = None,
    schmidt_number: float = SCHMIDT_NUMBER,
    closure_tolerance: float = 1.0e-12,
) -> CompositeHIDiffusionResult:
    """Apply advection then one composite new-depth BE diffusion solve.

    Coarse/fine physical segments enter the matrix exactly once.  The returned
    diffusive register is reconstructed from the converged solution; it is a
    diagnostic of the matrix update, never a separate post-solve correction.
    """
    areas = _float_array(areas_m2, "cell areas", ndim=1)
    depths = _float_array(new_depth_m, "new depths", ndim=1)
    initial_hi = _float_array(
        signed_hi_initial_m2_s, "initial signed hI", ndim=1
    )
    if areas.shape != depths.shape or areas.shape != initial_hi.shape:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 composite scalar state arrays have incompatible shapes"
        )
    if np.any(areas <= 0.0):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 composite cell areas must be positive"
        )
    dry = _finite(dry_tolerance_m, "dry tolerance", positive=True)
    if np.any(depths <= dry):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 composite scalar solve is fully wet only"
        )
    dt = _finite(dt_s, "time step", nonnegative=True)
    diffusivity = _finite(
        scalar_diffusivity_m2_s, "scalar diffusivity", nonnegative=True
    )
    schmidt = _finite(schmidt_number, "Schmidt number", positive=True)
    if schmidt != SCHMIDT_NUMBER:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 scalar Schmidt number must be exactly 0.7"
        )
    tolerance = _finite(
        closure_tolerance, "closure tolerance", positive=True
    )
    canonical_segments = validate_physical_face_segments(
        segments, cell_count=areas.size, maximum_level=maximum_level
    )
    ids = tuple(segment.source_id for segment in canonical_segments)
    advective = _mapping_values(
        advective_by_source_id, ids, "advective register"
    )
    zeros = {source_id: 0.0 for source_id in ids}
    advective_register = canonical_hi_register(
        canonical_segments,
        cell_count=areas.size,
        maximum_level=maximum_level,
        advective_by_source_id={
            source_id: float(value) for source_id, value in zip(ids, advective)
        },
        diffusive_by_source_id=zeros,
    )
    integrated_initial = areas * initial_hi
    integrated_advective = (
        integrated_initial
        + advective_register.cell_increment(areas.size, "advective")
    )
    hi_advective = integrated_advective / areas
    weights = areas * depths
    intensity_advective = integrated_advective / weights
    matrix = np.diag(weights)
    rhs = np.array(integrated_advective, copy=True)
    conductance: dict[SourceID, float] = {}
    open_segments = tuple(
        segment
        for segment in canonical_segments
        if segment.boundary_kind == "open"
    )
    open_ids = tuple(segment.source_id for segment in open_segments)
    if open_ids:
        if open_intensity_by_source_id is None or open_depth_by_source_id is None:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 open diffusion requires authenticated trace intensity and depth"
            )
        open_intensity = _mapping_values(
            open_intensity_by_source_id, open_ids, "open trace intensity"
        )
        open_depth = _mapping_values(
            open_depth_by_source_id, open_ids, "open trace depth"
        )
        open_i = dict(zip(open_ids, open_intensity))
        open_h = dict(zip(open_ids, open_depth))
        if any(value <= dry for value in open_h.values()):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 open endpoint trace is outside the fully wet scope"
            )
    else:
        if open_intensity_by_source_id not in (None, {}) or open_depth_by_source_id not in (None, {}):
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 open trace data supplied to a closed composite mesh"
            )
        open_i = {}
        open_h = {}

    for segment in canonical_segments:
        if segment.boundary_kind == "wall" or diffusivity == 0.0 or dt == 0.0:
            conductance[segment.source_id] = 0.0
            continue
        nx, ny = segment.normal
        ox, oy = segment.owner_center_m
        if segment.neighbour is not None:
            assert segment.neighbour_center_m is not None
            dx = segment.neighbour_center_m[0] - ox
            dy = segment.neighbour_center_m[1] - oy
            distance = abs(dx * nx + dy * ny)
            face_depth = 0.5 * (
                depths[segment.owner] + depths[segment.neighbour]
            )
        else:
            dx = segment.midpoint_m[0] - ox
            dy = segment.midpoint_m[1] - oy
            distance = dx * nx + dy * ny
            face_depth = open_h[segment.source_id]
        if not math.isfinite(distance) or distance <= 0.0:
            raise PrognosticSecondaryAMRError(
                "Q3-H3/v59 face normal-projected distance must be positive"
            )
        value = segment.length_m * face_depth * diffusivity / (
            schmidt * distance
        )
        conductance[segment.source_id] = float(value)
        coefficient = dt * value
        owner = segment.owner
        matrix[owner, owner] += coefficient
        if segment.neighbour is not None:
            neighbour = segment.neighbour
            matrix[neighbour, neighbour] += coefficient
            matrix[owner, neighbour] -= coefficient
            matrix[neighbour, owner] -= coefficient
        else:
            rhs[owner] += coefficient * open_i[segment.source_id]

    try:
        intensity_after = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 composite backward-Euler solve failed"
        ) from error
    if not np.isfinite(intensity_after).all():
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 composite backward-Euler solution is nonfinite"
        )
    diffusive_values: dict[SourceID, float] = {}
    for segment in canonical_segments:
        value = conductance[segment.source_id]
        if segment.boundary_kind == "wall" or value == 0.0 or dt == 0.0:
            transfer = 0.0
        elif segment.neighbour is not None:
            transfer = dt * value * (
                intensity_after[segment.owner]
                - intensity_after[segment.neighbour]
            )
        else:
            transfer = dt * value * (
                intensity_after[segment.owner]
                - open_i[segment.source_id]
            )
        diffusive_values[segment.source_id] = float(transfer)
    register = canonical_hi_register(
        canonical_segments,
        cell_count=areas.size,
        maximum_level=maximum_level,
        advective_by_source_id={
            source_id: float(value) for source_id, value in zip(ids, advective)
        },
        diffusive_by_source_id=diffusive_values,
    )
    integrated_after = weights * intensity_after
    reconstructed_after = integrated_advective + register.cell_increment(
        areas.size, "diffusive"
    )
    if not np.allclose(
        integrated_after,
        reconstructed_after,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise PrognosticSecondaryAMRError(
            "Q3-H3/v59 BE state does not close to its diffusive face register"
        )
    hi_after = depths * intensity_after
    inventory_initial = math.fsum(float(value) for value in integrated_initial)
    inventory_advective = math.fsum(float(value) for value in integrated_advective)
    inventory_after = math.fsum(float(value) for value in integrated_after)
    energy_before, variance_before = _weighted_moments(
        intensity_advective, weights
    )
    energy_after, variance_after = _weighted_moments(intensity_after, weights)
    for array in (hi_advective, intensity_after, hi_after):
        array.setflags(write=False)
    return CompositeHIDiffusionResult(
        signed_hi_after_advection_m2_s=hi_advective,
        intensity_after_diffusion=intensity_after,
        signed_hi_after_diffusion_m2_s=hi_after,
        conductance_by_source_id=conductance,
        register=register,
        inventory_before_m4_s=inventory_initial,
        inventory_after_advection_m4_s=inventory_advective,
        inventory_after_diffusion_m4_s=inventory_after,
        weighted_energy_before=energy_before,
        weighted_energy_after=energy_after,
        weighted_variance_before=variance_before,
        weighted_variance_after=variance_after,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CPU_INTEGRATOR",
    "CanonicalHIRegister",
    "CompositeHIDiffusionResult",
    "ENVELOPE_VERSION",
    "EXECUTION_KEY",
    "GPU_INTEGRATOR",
    "ORDERED_ROUTE_KEY",
    "PARENT_ENVELOPE_VERSION",
    "PhysicalFaceSegment",
    "PrognosticSecondaryAMRError",
    "ROUTE_FINGERPRINT_KEY",
    "SCHEMA",
    "SCHMIDT_NUMBER",
    "SEMANTIC_DOCUMENT",
    "SEMANTIC_ID",
    "SCOPE_ID",
    "SignedHIProlongation",
    "TIME_COUPLING",
    "TOPOLOGY_LOCAL_INVALIDATIONS",
    "active_signed_hi_inventory",
    "canonical_amr_instance_fingerprints",
    "canonical_amr_payload_sha256",
    "canonical_hi_register",
    "composite_hi_backward_euler",
    "prognostic_secondary_amr_execution_document",
    "prognostic_secondary_amr_ordered_route",
    "prolong_signed_hi_family",
    "reconstructed_signed_intensity_trace",
    "restrict_signed_hi_family",
    "selected_family_signed_hi_inventory",
    "signed_hi_upwind_transfer",
    "validate_physical_face_segments",
    "validate_prognostic_secondary_amr_execution_document",
    "validate_prognostic_secondary_amr_ordered_route",
]
