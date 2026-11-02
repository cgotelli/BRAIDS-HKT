"""Shared wet/dry and AMR ownership contract for suspended sediment.

M2-A/M2-B deliberately froze the fixed-grid FLOW-2DH suspension profile at
schema v3.  This module composes the additional M2-C numerical ownership
rules without reinterpreting that physical profile.  CPU and GPU routes bind
both the semantic identity below and an instance fingerprint to their input,
checkpoint, restart, and result metadata.

The contract concerns *where accepted conserved mass lives*.  It does not
change a transport-capacity formula, settling law, or the distinct FLOW
``DryFlc`` diffusion gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping

import numpy as np


SUSPENDED_AMR_CONTRACT_SCHEMA = 1
SUSPENDED_AMR_CONTRACT_SCOPE_ID = (
    "braids-suspended-wetdry-amr-ownership-v1"
)
SUSPENDED_AMR_METHOD_ID = "braids.suspended_wetdry_amr_ownership_v1"

SUSPENDED_AMR_SEMANTIC_DOCUMENT = {
    "schema": SUSPENDED_AMR_CONTRACT_SCHEMA,
    "scope_id": SUSPENDED_AMR_CONTRACT_SCOPE_ID,
    "method_id": SUSPENDED_AMR_METHOD_ID,
    "dry_state": {
        "predicate": "accepted_h_le_hydraulic_dry_tolerance",
        "momentum": "zero",
        "suspended_inventory": "zero",
        "secondary_inventory": "zero_when_present",
        "diffusion_DryFlc": "distinct_constitutive_face_gate_not_dry_state",
    },
    "wet_to_dry": {
        "operation": (
            "atomically_transfer_each_suspended_class_to_its_active_bed_"
            "reservoir_before_acceptance"
        ),
        "class_total": "exactly_conserved",
        "phase_ledger": "record_suspended_loss_and_equal_active_bed_gain",
        "resuspension": "only_by_the_selected_exchange_operator_on_a_later_wet_state",
    },
    "restriction": {
        "operation": "area_weight_every_conserved_state_row",
        "dry_result": "apply_wet_to_dry_phase_transfer_before_leaf_install",
        "class_total": "exactly_conserved",
    },
    "prolongation": {
        "water": "stage_fill_over_child_bed_with_exact_parent_volume",
        "suspension": (
            "bounded_concentration_reconstruction_on_wet_children_then_"
            "classwise_exact_mass_closure"
        ),
        "dry_children": "zero_suspended_inventory",
        "class_total": "exactly_conserved",
    },
    "reflux": {
        "synchronous_shared_face": (
            "one_accepted_time_integral_of_advection_plus_horizontal_diffusion_"
            "is_applied_with_opposite_signs_to_both_coarse_fine_neighbors"
        ),
        "subcycled_register": (
            "accepted_fine_time_integrated_shared_face_advection_plus_"
            "horizontal_diffusion_replaces_the_coarse_contribution_for_every_"
            "suspended_class"
        ),
        "phase_repair": "within_the_same_grain_class_only",
        "boundary_ledgers": "root_parent_owned_exactly_once",
        "exchange_ledgers": "sum_every_accepted_level_substep_exactly_once",
    },
    "open_boundary_memory": {
        "topology_rule": "every_open_boundary_segment_remains_at_finest_level",
        "identity": "stable_authenticated_candidate_face_id",
        "regrid": "exact_one_to_one_remap_or_reject_before_commit",
        "coarsening_average": "forbidden",
    },
    "transaction": {
        "candidate": (
            "state_topology_flux_register_boundary_memory_ledgers_and_clocks"
        ),
        "commit": "atomic_after_all_finite_positivity_and_conservation_checks",
        "rejection_or_cancellation": "byte_exact_accepted_state_rollback",
        "restart": "authenticate_semantics_instance_topology_and_all_accepted_state",
    },
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


SUSPENDED_AMR_SEMANTIC_ID = hashlib.sha256(
    _canonical(SUSPENDED_AMR_SEMANTIC_DOCUMENT)
).hexdigest()


class SuspendedAMRContractError(ValueError):
    """Fail-closed M2-C contract error."""


@dataclass(frozen=True)
class SuspendedAMRContract:
    schema: int
    scope_id: str
    semantic_id: str
    suspension_profile_fingerprint: str
    fraction_count: int
    hydraulic_dry_tolerance_m: float
    diffusion_dry_threshold_m: float
    amr_enabled: bool
    amr_levels: int
    amr_refinement_ratio: int
    amr_wet_depth_m: float
    amr_regrid_interval_s: float
    fingerprint: str

    def as_dict(self) -> dict:
        return suspended_amr_contract_document(self)


def suspended_amr_contract_document(contract: SuspendedAMRContract) -> dict:
    """Return the canonical serializable instance document."""
    return {
        "schema": contract.schema,
        "scope_id": contract.scope_id,
        "semantic_id": contract.semantic_id,
        "suspension_profile_fingerprint": contract.suspension_profile_fingerprint,
        "fraction_count": contract.fraction_count,
        "hydraulic_dry_tolerance_m": contract.hydraulic_dry_tolerance_m,
        "diffusion_dry_threshold_m": contract.diffusion_dry_threshold_m,
        "amr_enabled": contract.amr_enabled,
        "amr_levels": contract.amr_levels,
        "amr_refinement_ratio": contract.amr_refinement_ratio,
        "amr_wet_depth_m": contract.amr_wet_depth_m,
        "amr_regrid_interval_s": contract.amr_regrid_interval_s,
        "fingerprint": contract.fingerprint,
    }


def _instance_payload(**values) -> dict:
    return {
        "schema": SUSPENDED_AMR_CONTRACT_SCHEMA,
        "scope_id": SUSPENDED_AMR_CONTRACT_SCOPE_ID,
        "semantic_id": SUSPENDED_AMR_SEMANTIC_ID,
        **values,
    }


def resolve_suspended_amr_contract(
    suspension_profile_fingerprint: str,
    *,
    fraction_count: int,
    hydraulic_dry_tolerance_m: float,
    diffusion_dry_threshold_m: float,
    amr_enabled: bool,
    amr_levels: int = 1,
    amr_refinement_ratio: int = 2,
    amr_wet_depth_m: float = 0.0,
    amr_regrid_interval_s: float = 0.0,
) -> SuspendedAMRContract:
    """Resolve and content-address one CPU/GPU M2-C instance."""
    if (not isinstance(suspension_profile_fingerprint, str) or
            re.fullmatch(r"[0-9a-f]{64}", suspension_profile_fingerprint) is None):
        raise SuspendedAMRContractError(
            "Suspension profile fingerprint must be a lowercase SHA-256 digest"
        )
    if isinstance(fraction_count, bool) or not isinstance(fraction_count, int) or fraction_count < 1:
        raise SuspendedAMRContractError("Fraction count must be a positive integer")
    if not isinstance(amr_enabled, bool):
        raise SuspendedAMRContractError("AMR enabled must be Boolean")
    if (isinstance(amr_levels, bool) or not isinstance(amr_levels, int) or
            amr_levels < 1):
        raise SuspendedAMRContractError("AMR levels must be a positive integer")
    if (isinstance(amr_refinement_ratio, bool) or
            not isinstance(amr_refinement_ratio, int) or
            amr_refinement_ratio != 2):
        raise SuspendedAMRContractError(
            "The supported AMR refinement ratio is exactly 2:1"
        )
    numeric = {
        "hydraulic_dry_tolerance_m": hydraulic_dry_tolerance_m,
        "diffusion_dry_threshold_m": diffusion_dry_threshold_m,
        "amr_wet_depth_m": amr_wet_depth_m,
        "amr_regrid_interval_s": amr_regrid_interval_s,
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(float(value)) or float(value) < 0.0
           for value in numeric.values()):
        raise SuspendedAMRContractError(
            "Wet/dry and AMR thresholds/intervals must be finite and nonnegative"
        )
    if float(hydraulic_dry_tolerance_m) <= 0.0:
        raise SuspendedAMRContractError(
            "Hydraulic dry tolerance must be strictly positive"
        )
    if amr_enabled:
        if amr_levels < 2 or float(amr_regrid_interval_s) <= 0.0:
            raise SuspendedAMRContractError(
                "Enabled AMR requires at least two levels and a positive regrid interval"
            )
    elif (amr_levels != 1 or float(amr_wet_depth_m) != 0.0 or
          float(amr_regrid_interval_s) != 0.0):
        raise SuspendedAMRContractError(
            "Fixed-grid instances must use one level and zero AMR thresholds/clocks"
        )
    values = dict(
        suspension_profile_fingerprint=suspension_profile_fingerprint,
        fraction_count=fraction_count,
        hydraulic_dry_tolerance_m=float(hydraulic_dry_tolerance_m),
        diffusion_dry_threshold_m=float(diffusion_dry_threshold_m),
        amr_enabled=amr_enabled,
        amr_levels=amr_levels,
        amr_refinement_ratio=amr_refinement_ratio,
        amr_wet_depth_m=float(amr_wet_depth_m),
        amr_regrid_interval_s=float(amr_regrid_interval_s),
    )
    fingerprint = hashlib.sha256(_canonical(_instance_payload(**values))).hexdigest()
    return SuspendedAMRContract(
        schema=SUSPENDED_AMR_CONTRACT_SCHEMA,
        scope_id=SUSPENDED_AMR_CONTRACT_SCOPE_ID,
        semantic_id=SUSPENDED_AMR_SEMANTIC_ID,
        fingerprint=fingerprint,
        **values,
    )


def load_suspended_amr_contract(raw: Mapping[str, object]) -> SuspendedAMRContract:
    """Validate a serialized contract by resolving it again."""
    if not isinstance(raw, Mapping):
        raise SuspendedAMRContractError("Suspended AMR contract must be a mapping")
    expected = {
        "schema", "scope_id", "semantic_id", "suspension_profile_fingerprint",
        "fraction_count", "hydraulic_dry_tolerance_m",
        "diffusion_dry_threshold_m", "amr_enabled", "amr_levels",
        "amr_refinement_ratio", "amr_wet_depth_m", "amr_regrid_interval_s",
        "fingerprint",
    }
    if set(raw) != expected:
        raise SuspendedAMRContractError("Suspended AMR contract fields differ")
    result = resolve_suspended_amr_contract(
        str(raw["suspension_profile_fingerprint"]),
        fraction_count=raw["fraction_count"],
        hydraulic_dry_tolerance_m=raw["hydraulic_dry_tolerance_m"],
        diffusion_dry_threshold_m=raw["diffusion_dry_threshold_m"],
        amr_enabled=raw["amr_enabled"],
        amr_levels=raw["amr_levels"],
        amr_refinement_ratio=raw["amr_refinement_ratio"],
        amr_wet_depth_m=raw["amr_wet_depth_m"],
        amr_regrid_interval_s=raw["amr_regrid_interval_s"],
    )
    if (raw["schema"] != result.schema or raw["scope_id"] != result.scope_id or
            raw["semantic_id"] != result.semantic_id or
            raw["fingerprint"] != result.fingerprint):
        raise SuspendedAMRContractError(
            "Suspended AMR contract identity/fingerprint differs"
        )
    return result


def transfer_suspension_from_dry_cells(
    depth_m, suspended_kg_m2, active_bed_kg_m2, *, dry_tolerance_m: float
):
    """Reference wet-to-dry phase transfer used by CPU/GPU microtests.

    Arrays are returned as independent copies together with the signed
    suspended-phase change.  Class totals are preserved exactly by the same
    floating-point add/subtract pair; no mass is clipped or distributed to a
    different cell or class.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    suspended = np.asarray(suspended_kg_m2, dtype=np.float64)
    bed = np.asarray(active_bed_kg_m2, dtype=np.float64)
    if (depth.ndim != 1 or suspended.ndim != 2 or bed.shape != suspended.shape or
            suspended.shape[1] != depth.size):
        raise SuspendedAMRContractError(
            "Depth and class-by-cell phase arrays have incompatible shapes"
        )
    if (not np.isfinite(depth).all() or not np.isfinite(suspended).all() or
            not np.isfinite(bed).all() or np.any(depth < 0.0) or
            np.any(suspended < 0.0) or np.any(bed < 0.0)):
        raise SuspendedAMRContractError(
            "Wet/dry phase-transfer state must be finite and nonnegative"
        )
    if (isinstance(dry_tolerance_m, bool) or
            not isinstance(dry_tolerance_m, (int, float)) or
            not math.isfinite(float(dry_tolerance_m)) or
            float(dry_tolerance_m) <= 0.0):
        raise SuspendedAMRContractError(
            "Hydraulic dry tolerance must be finite and positive"
        )
    dry = depth <= float(dry_tolerance_m)
    moved = np.zeros_like(suspended)
    moved[:, dry] = suspended[:, dry]
    next_suspended = suspended.copy()
    next_bed = bed.copy()
    next_suspended[:, dry] = 0.0
    next_bed[:, dry] += moved[:, dry]
    return next_suspended, next_bed, -moved


__all__ = [
    "SUSPENDED_AMR_CONTRACT_SCHEMA",
    "SUSPENDED_AMR_CONTRACT_SCOPE_ID",
    "SUSPENDED_AMR_METHOD_ID",
    "SUSPENDED_AMR_SEMANTIC_DOCUMENT",
    "SUSPENDED_AMR_SEMANTIC_ID",
    "SuspendedAMRContract",
    "SuspendedAMRContractError",
    "load_suspended_amr_contract",
    "resolve_suspended_amr_contract",
    "suspended_amr_contract_document",
    "transfer_suspension_from_dry_cells",
]
