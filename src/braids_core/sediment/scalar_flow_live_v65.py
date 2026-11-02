"""Bounded fixed-grid live composition for the six v64 scalar FLOW laws.

Version 65 is an additive, backend-neutral transaction contract.  It starts
with authenticated v64 *pre-active-fraction* mass channels and composes them
through the accepted Q3 channel, face-role, FLOW-slope and conservative-Exner
seams.  The implementation in this file is NumPy-portable so that the CPU
reference and a resident GPU implementation can share shapes and identities.

The admitted live transaction is intentionally narrow: current only, fully
wet, fixed grid, first-order face reconstruction and ``TRA_BEDLOAD``.  The
channel composer may expose a diagnostic ``TRA_COMBINE`` retained channel, but
v65 will not commit it without the Q4 water-column mass/exchange owner.
``TRA_ADVDIFF`` bypasses ``eqtran`` and is rejected here.

No function mutates an accepted state.  ``begin_bed_exner_trial_v65`` builds a
complete candidate, ``commit_trial_v65`` returns it only after checking the
parent digest, and rollback/cancellation return the byte-equivalent parent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .channel_composition import (
    TRA_ADVDIFF,
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionProfile,
    channel_composition_profile_document,
    compose_mass_channels,
    load_channel_composition_profile,
)
from .scalar_flow_cpu_producer_v64 import (
    load_cpu_producer_execution,
    load_mixture_profile,
)
from .scalar_flow_execution_v63 import load_scalar_flow_execution
from .classic_flow_taurat_v69 import (
    TAURAT_POSITIVE_ZERO,
    source_contract_document_v69,
)


SCHEMA = 1
ENVELOPE_VERSION = 65
SCOPE_ID = "braids-q3h5-scalar-flow-fixed-grid-live-v65-v1"
SUPPORTED_SELECTORS = frozenset((7, 13, 14, 16, 17, 18))
MAX_ACCEPTED_STEPS = 20
LOCKED_UPWBED_PATH = (
    "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
    "flow2d3d_kernel/src/compute_sediment/upwbed.f90"
)
LOCKED_UPWBED_SHA256 = (
    "06be422d88821f3f983bc136d511fdaa80ab04c8529640474ab39b478d918158"
)

FACE_INTERNAL = 0
FACE_TRACE_INLET = 1
FACE_TRACE_OUTLET = 2
FACE_OWNER_OUTLET = 3
FACE_CLOSED = 4

DONOR_NONE = 0
DONOR_OWNER = 1
DONOR_NEIGHBOUR = 2
DONOR_TRACE = 3

SLOPE_OFF = "bed_slope.off"
SLOPE_KOCH_FLOKSTRA = "bed_slope.delft_koch_flokstra"
SLOPE_PARKER_ANDREWS = "bed_slope.delft_parker_andrews"
SLOPE_BAGNOLD_IKEDA = "bed_slope.delft_bagnold_ikeda"
LIVE_SLOPE_METHODS = frozenset((
    SLOPE_OFF, SLOPE_BAGNOLD_IKEDA, SLOPE_KOCH_FLOKSTRA,
    SLOPE_PARKER_ANDREWS,
))

FACE_ROLE_POLICY = {
    "internal": "one_shared_flux_owner_to_neighbour_positive",
    "closed": "exact_zero",
    "trace_inlet": "negative_trace_capacity_with_owner_availability",
    "trace_outlet": "positive_trace_capacity_with_owner_availability",
    "owner_outlet": "positive_owner_capacity_with_owner_availability",
}

ORDERED_STAGES = (
    "authenticated_v64_pre_fraction_mass_channels",
    "external_secondary_rotation_of_bed_only_already_applied",
    "global_channel_factors_and_class_fold_once",
    "optional_explicit_transverse_component_stencil",
    "first_order_face_reconstruction",
    "upwbed_total_transport_sign_selection",
    "optional_classic_flow_islope2_islope3_or_islope4_direction",
    "corrected_direction_final_donor",
    "active_fraction_once",
    "finite_inventory_fill_once",
    "role_signed_shared_face_flux",
    "conservative_atomic_exner_candidate",
)

UNSUPPORTED_GATES = (
    "wave_or_scour_forcing",
    "wet_dry_or_newly_wet_faces",
    "amr_or_reflux",
    "trafrm7_external_or_function_settling_and_dynamic_reference_height",
    "trafrm3_water_column_commit_without_q4",
    "trafrm2_advdiff_bypasses_eqtran",
)

DEFAULT_SLOPE_CONTROLS = {
    "alfabs": 1.0,
    "alfabn": 1.5,
    "ashld": 0.85,
    "bshld": 0.5,
    "cshld": 0.0,
    "dshld": 0.0,
    "alfpa": 0.4,
    "thcrpa": 0.047,
}
TAURAT_SOURCE_CONTRACT_FINGERPRINT = source_contract_document_v69()[
    "fingerprint"]

_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ScalarFlowLiveV65Error(ValueError):
    """Fail-closed v65 profile, state or transaction error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalarFlowLiveV65Error(f"Q3-H5/v65 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           dtype=np.float64, finite: bool = True, copy: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} must have shape {shape}, got {result.shape}")
    if finite and not np.all(np.isfinite(result)):
        raise ScalarFlowLiveV65Error(f"Q3-H5/v65 {name} must be finite")
    if copy:
        result = np.array(result, dtype=dtype, copy=True, order="C")
        result.setflags(write=False)
    return result


def _integer_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} must be an integer array of shape {shape}")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def scalar_flow_live_semantic_document_v65() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "selectors": sorted(SUPPORTED_SELECTORS),
        "backend_neutral": True,
        "array_layout": "C_contiguous_float64_or_int64",
        "cell_vector_shape": "(2,nfraction,ncell)",
        "face_vector_shape": "(2,nfraction,nface)",
        "cell_scalar_shape": "(nfraction,ncell)",
        "face_scalar_shape": "(nfraction,nface)",
        "topology_shape": "(nface,)",
        "face_roles": {
            "internal": FACE_INTERNAL,
            "trace_inlet": FACE_TRACE_INLET,
            "trace_outlet": FACE_TRACE_OUTLET,
            "owner_outlet": FACE_OWNER_OUTLET,
            "closed": FACE_CLOSED,
        },
        "donors": {
            "none": DONOR_NONE,
            "owner": DONOR_OWNER,
            "neighbour": DONOR_NEIGHBOUR,
            "trace": DONOR_TRACE,
        },
        "transport_class_scope": {
            str(TRA_BEDLOAD): "fold_and_live_bed_exner",
            str(TRA_COMBINE): "diagnostic_retained_channel_only_requires_q4_to_commit",
            str(TRA_ADVDIFF): "rejected_bypasses_eqtran_requires_q4",
        },
        "slope_scope": {
            SLOPE_OFF: "live",
            SLOPE_KOCH_FLOKSTRA: "live_islope3",
            SLOPE_PARKER_ANDREWS: "live_islope4",
            SLOPE_BAGNOLD_IKEDA: "live_islope2_exact_erosed_zero_taurat",
        },
        "positive_trafrm_taurat_source_contract_fingerprint": (
            TAURAT_SOURCE_CONTRACT_FINGERPRINT),
        "upwind_source": (
            "locked_upwbed_sutot_svtot_post_secondary_composed_total_"
            "transport_direction_per_fraction"
        ),
        "locked_upwbed": {
            "path": LOCKED_UPWBED_PATH,
            "sha256": LOCKED_UPWBED_SHA256,
            "source_lines": "113-140_and_154-181",
        },
        "lateral_source": (
            "locked_upwbed_per_component_quarter_neighbour_center_"
            "correction_CSR_applied_before_upwind_selection"
        ),
        "availability": {
            "authoritative_fields": ["active_fraction", "accepted_bed_mass"],
            "materialization": "active_fraction_times_cell_global_inventory_fill",
            "projection_application_count": 1,
            "exner_application_count": 0,
        },
        "ordered_stages": list(ORDERED_STAGES),
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "transaction": (
            "immutable_parent_complete_candidate_parent_digest_commit_or_exact_rollback"
        ),
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "unsupported_gates": list(UNSUPPORTED_GATES),
        "normal_product_admission": False,
    }


SEMANTIC_ID = hashlib.sha256(
    _canonical(scalar_flow_live_semantic_document_v65())
).hexdigest()


def selector7_settling_reference_document_v65(
    *, scalar_execution: Mapping[str, object],
    producer_execution: Mapping[str, object],
    mixture_profile: Mapping[str, object],
    settling_reference_proof_fingerprint: str | None = None,
) -> dict[str, object]:
    """Bind only the explicit v64 TraFrm-7 Aks/WSettle route.

    External settling functions, dynamic reference heights and any inferred
    defaults remain outside v65.  The v64 settling dependency fingerprint is
    part of this document, so a caller cannot provide an arbitrary live value.
    """
    scalar = load_scalar_flow_execution(scalar_execution)
    mixture = load_mixture_profile(mixture_profile)
    producer, backend = _load_backend_producer_v65(
        producer_execution, scalar=scalar, mixture=mixture)
    if scalar["reference_selector"] != 7:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 settling/reference identity is only for TraFrm 7")
    parameters = {row["keyword"]: row["value"] for row in scalar["parameters"]}
    if backend == "cpu":
        dependency = next(
            row for row in producer["dependencies"]
            if row["stage"] == "settling_velocity")
        if dependency["status"] != "consumed" or dependency["fingerprint"] is None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 TraFrm 7 requires its v64 settling dependency")
        settling_proof = dependency["fingerprint"]
        if settling_reference_proof_fingerprint is not None and \
                settling_reference_proof_fingerprint != settling_proof:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 CPU TraFrm 7 settling proof differs from v64")
    else:
        settling_proof = _sha(
            settling_reference_proof_fingerprint,
            "GPU TraFrm 7 settling/reference proof")
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "reference_selector": 7,
        "backend": backend,
        "scalar_execution_fingerprint": scalar["fingerprint"],
        "producer_execution_fingerprint": producer["fingerprint"],
        "settling_dependency_fingerprint": settling_proof,
        "reference_height_m": _finite(
            parameters["Aks"], "TraFrm 7 Aks", positive=True),
        "settling_velocity_m_s": _finite(
            parameters["WSettle"], "TraFrm 7 WSettle", nonnegative=True),
        "route": "explicit_v64_parameters_only",
        "external_function": False,
        "dynamic_reference_height": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_selector7_document(
    value: Mapping[str, object], *, scalar_execution: Mapping[str, object],
    producer_execution: Mapping[str, object], mixture_profile: Mapping[str, object],
) -> dict[str, object]:
    expected = selector7_settling_reference_document_v65(
        scalar_execution=scalar_execution,
        producer_execution=producer_execution,
        mixture_profile=mixture_profile,
        settling_reference_proof_fingerprint=(
            value.get("settling_dependency_fingerprint")
            if isinstance(value, Mapping) else None),
    )
    if not isinstance(value, Mapping) or dict(value) != expected or \
            _canonical(value) != _canonical(expected):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 TraFrm 7 settling/reference identity differs")
    return expected


def _reject_dirac_mixture(mixture: Mapping[str, object]) -> None:
    max_mud = int(mixture["max_mud_sediment_type"])
    for row in mixture["ordered_fractions"]:
        if int(row["sediment_type"]) > max_mud and \
                len(row["percentile_table"]) < 2:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 Dirac/single-point within-class CDF is not admitted")


def _slope_controls(values: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(DEFAULT_SLOPE_CONTROLS):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 requires all eight FLOW slope controls")
    return {
        key: _finite(values[key], key, nonnegative=(key == "thcrpa"))
        for key in DEFAULT_SLOPE_CONTROLS
    }


def _load_backend_producer_v65(
    producer_execution: Mapping[str, object], *, scalar: Mapping[str, object],
    mixture: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Validate the selected v64 producer without permitting fallback."""
    if not isinstance(producer_execution, Mapping):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 backend producer identity must be an object")
    backend = producer_execution.get("backend")
    if backend == "cpu":
        producer = load_cpu_producer_execution(
            producer_execution, scalar_execution=scalar,
            mixture_profile=mixture)
        return producer, "cpu"
    if backend == "implicit_fwave_midpoint_gpu":
        # Lazy import keeps the portable CPU executor free of a CUDA runtime
        # dependency while still rebuilding the canonical GPU producer bytes.
        from .scalar_flow_gpu_producer_v64 import (  # noqa: PLC0415
            load_scalar_flow_gpu_producer_document,
        )
        producer = load_scalar_flow_gpu_producer_document(producer_execution)
        dependencies = producer["dependencies"]
        fraction_ids = [row["fraction_id"] for row in mixture["ordered_fractions"]]
        if (producer["integrator"] != "implicit_fwave_midpoint" or
                producer["automatic_cpu_fallback"] is not False or
                producer["reference_selector"] != scalar["reference_selector"] or
                dependencies["scalar_execution_fingerprint"] != scalar["fingerprint"] or
                producer["fraction_ids"] != fraction_ids):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 selected GPU producer dependencies differ")
        return producer, "implicit_fwave_midpoint_gpu"
    raise ScalarFlowLiveV65Error(
        "Q3-H5/v65 producer backend must be cpu or implicit_fwave_midpoint_gpu")


def scalar_flow_live_execution_document_v65(
    *, scalar_execution: Mapping[str, object],
    producer_execution: Mapping[str, object],
    mixture_profile: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    backend: str,
    bed_slope_method_id: str,
    slope_controls: Mapping[str, object],
    secondary_execution_fingerprint: str,
    face_reconstruction_fingerprint: str,
    boundary_roles_fingerprint: str,
    exner_fingerprint: str,
    bed_slope_execution_fingerprint: str | None = None,
    hiding_execution_fingerprint: str | None = None,
    selector7_settling_reference: Mapping[str, object] | None = None,
    upwind_bedload: bool = True,
    laterally_averaged_bedload: bool = False,
    lateral_stencil_fingerprint: str | None = None,
    bulk_density_kg_m3: Sequence[float] = (),
) -> dict[str, object]:
    """Build the CPU/GPU-common v65 physical and transaction identity."""
    scalar = load_scalar_flow_execution(scalar_execution)
    mixture = load_mixture_profile(mixture_profile)
    producer, producer_backend = _load_backend_producer_v65(
        producer_execution, scalar=scalar, mixture=mixture)
    selector = int(scalar["reference_selector"])
    if selector not in SUPPORTED_SELECTORS:
        raise ScalarFlowLiveV65Error("Q3-H5/v65 selector is outside the six-law set")
    # ``compdiam`` has an explicit exact-diameter/Dirac branch.  A one-point
    # non-mud table is therefore a complete distribution, not a missing broad
    # CDF.  The v64 producer owns its exact percentile scan and the live layer
    # only binds that authenticated table to accepted state.
    if isinstance(channel_profile, ChannelCompositionProfile):
        channel_document = channel_composition_profile_document(channel_profile)
    else:
        channel_document = channel_composition_profile_document(
            load_channel_composition_profile(channel_profile))
    fraction_ids = [row["fraction_id"] for row in channel_document["fractions"]]
    mixture_ids = [row["fraction_id"] for row in mixture["ordered_fractions"]]
    if fraction_ids != mixture_ids:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 channel and mixture fraction order differs")
    if (not isinstance(bulk_density_kg_m3, Sequence) or
            isinstance(bulk_density_kg_m3, (str, bytes)) or
            len(bulk_density_kg_m3) != len(fraction_ids)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 requires one bulk density per ordered fraction")
    bulk_density = [
        _finite(value, f"bulk density for {fraction_ids[index]}", positive=True)
        for index, value in enumerate(bulk_density_kg_m3)
    ]
    transport_classes = [int(row["transport_class"])
                         for row in channel_document["fractions"]]
    if any(value == TRA_ADVDIFF for value in transport_classes):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 TRA_ADVDIFF bypasses eqtran and belongs to Q4")
    if backend not in ("cpu", "implicit_fwave_midpoint_gpu"):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 backend must be cpu or implicit_fwave_midpoint_gpu")
    if backend != producer_backend:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 backend and v64 producer backend differ")
    if type(upwind_bedload) is not bool or type(laterally_averaged_bedload) is not bool:
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face controls must be Boolean")
    if laterally_averaged_bedload:
        lateral_fp = _sha(lateral_stencil_fingerprint, "lateral stencil")
    else:
        if lateral_stencil_fingerprint is not None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 lateral stencil supplied while control is off")
        lateral_fp = None
    if bed_slope_method_id not in LIVE_SLOPE_METHODS:
        raise ScalarFlowLiveV65Error("Q3-H5/v65 bed-slope method is unsupported")
    if producer.get("bed_slope_method_id", producer.get("slope_method_id")) != \
            bed_slope_method_id:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 v64 producer and live bed-slope methods differ")
    controls = _slope_controls(slope_controls)
    if bed_slope_method_id == SLOPE_OFF:
        if bed_slope_execution_fingerprint is not None or \
                hiding_execution_fingerprint is not None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 slope-off may not carry slope/hiding execution identities")
        slope_fp = None
        hiding_fp = None
    else:
        slope_fp = _sha(bed_slope_execution_fingerprint, "bed-slope execution")
        if bed_slope_method_id == SLOPE_PARKER_ANDREWS:
            hiding_fp = _sha(hiding_execution_fingerprint, "slope hiding execution")
        else:
            if hiding_execution_fingerprint is not None:
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 selected slope does not consume generic hiding")
            hiding_fp = None
    if selector == 7:
        settling = _validate_selector7_document(
            selector7_settling_reference,
            scalar_execution=scalar, producer_execution=producer,
            mixture_profile=mixture)
        settling_fp = settling["fingerprint"]
    else:
        if selector7_settling_reference is not None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 non-TraFrm-7 route may not carry settling/reference identity")
        settling = None
        settling_fp = None
    dependencies = {
        "scalar_execution": scalar["fingerprint"],
        "backend_producer_execution": producer["fingerprint"],
        "backend_producer_scope_id": producer["scope_id"],
        "backend_distribution_fingerprint": (
            producer["dependencies"]["distribution_fingerprint"]
            if producer_backend == "implicit_fwave_midpoint_gpu" else
            producer["mixture_profile_fingerprint"]),
        "mixture_profile": mixture["fingerprint"],
        "channel_profile": channel_document["fingerprint"],
        "secondary_execution": _sha(
            secondary_execution_fingerprint, "secondary execution"),
        "face_reconstruction": _sha(
            face_reconstruction_fingerprint, "face reconstruction"),
        "boundary_roles": _sha(boundary_roles_fingerprint, "boundary roles"),
        "exner": _sha(exner_fingerprint, "Exner execution"),
        "bed_slope_execution": slope_fp,
        "positive_trafrm_taurat_source_contract": (
            TAURAT_SOURCE_CONTRACT_FINGERPRINT
            if bed_slope_method_id == SLOPE_BAGNOLD_IKEDA else None),
        "slope_hiding_execution": hiding_fp,
        "selector7_settling_reference": settling_fp,
        "lateral_stencil": lateral_fp,
    }
    if producer_backend == "cpu":
        producer_regime = producer["effective_velocity_regime"]
        effective_velocity_regime = {
            "wave_enabled": producer_regime["wave"],
            "scour_enabled": producer_regime["scour"],
            "maximum_waterdepth": False,
            "working_depth_floor_m": 0.01,
        }
    else:
        effective_velocity_regime = {
            "wave_enabled": producer["wave_enabled"],
            "scour_enabled": producer["scour_enabled"],
            "maximum_waterdepth": producer["maximum_waterdepth"],
            "working_depth_floor_m": producer["working_depth_floor_m"],
        }
    # This hash is deliberately free of backend/source/receipt identities.
    # It answers only whether CPU and GPU selected the same physical methods,
    # values and controls; the implementation receipts remain in
    # ``dependencies`` and in the outer execution fingerprint.
    physical_profile = {
        "profile_kind": "scalar_flow_v65_physical_methods",
        "reference_selector": selector,
        "capacity": {
            "method_id": scalar["method_id"],
            "formula_code": scalar["formula_code"],
            "parameters": scalar["parameters"],
            "roughness_method_id": scalar["roughness_method_id"],
            "sediment_shear_method_id": scalar["sediment_shear_method_id"],
            "hiding_method_id": scalar["hiding_method_id"],
            "mixture_contract": scalar["mixture_contract"],
            "shear_contract": scalar["shear_contract"],
        },
        "mixture": {
            "ordered_fractions": mixture["ordered_fractions"],
            "max_mud_sediment_type": mixture["max_mud_sediment_type"],
            "min_dxx_sediment_type": mixture["min_dxx_sediment_type"],
            "spatial_d50": mixture["spatial_d50"],
            "spatial_log_sigma": mixture["spatial_log_sigma"],
            "composition_basis": mixture["composition_basis"],
            "broad_distribution_policy": mixture["broad_distribution_policy"],
            "surface_fraction_sum_policy": mixture[
                "surface_fraction_sum_policy"],
            "depleted_inventory_policy": mixture[
                "depleted_inventory_policy"],
        },
        "channel_composition": {
            "bed_factor": channel_document["bed_factor"],
            "suspended_factor": channel_document["suspended_factor"],
            "epsilon": channel_document["epsilon"],
            "fractions": channel_document["fractions"],
        },
        "bulk_density_kg_m3": bulk_density,
        "transport_classes": transport_classes,
        "bed_slope_method_id": bed_slope_method_id,
        "slope_stress_semantics": (
            TAURAT_POSITIVE_ZERO
            if bed_slope_method_id == SLOPE_BAGNOLD_IKEDA else
            "absolute_shields_from_total_current_RP_USTAR_squared"
            if bed_slope_method_id != SLOPE_OFF else "not_consumed"
        ),
        "slope_controls": controls,
        "upwind_bedload": upwind_bedload,
        "laterally_averaged_bedload": laterally_averaged_bedload,
        "lateral_stencil_method": (
            "locked_upwbed_component_quarter_neighbour_center_correction"
            if laterally_averaged_bedload else "off"),
        "effective_velocity_roughness_method_id": producer.get(
            "effective_velocity_roughness_method_id",
            producer.get("roughness_method_id")),
        "effective_velocity_regime": effective_velocity_regime,
        "sediment_shear_method_id": scalar["sediment_shear_method_id"],
        "capacity_hiding_method_id": scalar["hiding_method_id"],
        "slope_hiding_method_id": (
            producer.get("slope_hiding_method_id")
            if bed_slope_method_id == SLOPE_PARKER_ANDREWS else
            "hiding.not_consumed_by_slope"),
        "secondary_method_id": producer["secondary_method_id"],
        "selector7_settling": (
            None if settling is None else {
                "reference_height_m": settling["reference_height_m"],
                "settling_velocity_m_s": settling["settling_velocity_m_s"],
                "route": settling["route"],
                "external_function": settling["external_function"],
                "dynamic_reference_height": settling[
                    "dynamic_reference_height"],
            }),
        "face_reconstruction_method": "first_order_fixed_grid",
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "exner_method": "shared_face_conservative_mass_update",
    }
    if producer_backend == "cpu":
        producer_stages = {
            row["stage"]: row["fingerprint"]
            for row in producer["dependencies"]
        }
        expected_shared = {
            "channel_composition": channel_document["fingerprint"],
            "secondary_steering": dependencies["secondary_execution"],
            "face_reconstruction": dependencies["face_reconstruction"],
            "boundary_roles": dependencies["boundary_roles"],
            "exner": dependencies["exner"],
        }
        for stage, expected in expected_shared.items():
            if producer_stages[stage] != expected:
                raise ScalarFlowLiveV65Error(
                    f"Q3-H5/v65 v64 CPU {stage} dependency differs")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "qualification": "bounded_fixed_grid_live_composition_not_product_dispatch",
        "backend": backend,
        "automatic_backend_fallback": False,
        "physical_profile_fingerprint": _fingerprint(physical_profile),
        "reference_selector": selector,
        "method_id": scalar["method_id"],
        "ordered_fraction_ids": fraction_ids,
        "bulk_density_kg_m3": bulk_density,
        "active_fraction_basis": "accepted_bed_mass_divided_by_bulk_density",
        "transport_classes": transport_classes,
        "transport_class_scope": (
            "class1_live_class3_component_only_class2_rejected"),
        "bed_slope_method_id": bed_slope_method_id,
        "slope_controls": controls,
        "upwind_bedload": upwind_bedload,
        "laterally_averaged_bedload": laterally_averaged_bedload,
        "face_reconstruction": "first_order_fixed_grid",
        "mesh_scope": "fixed_grid",
        "wet_scope": "fully_wet_positive_depth",
        "forcing_scope": "current_only_wave_false_scour_false",
        "mixture_scope": (
            "broad_or_single_point_dirac_cdf_nonnegative_class_stock_"
            "retained_identity_at_zero_cell_total"
        ),
        "producer_mixture_binding": (
            "backend_distribution_and_common_v65_mixture_bound_separately_"
            "ordered_fraction_ids_checked_no_cross_schema_hash_equivalence_inferred"
        ),
        "locked_upwbed": {
            "path": LOCKED_UPWBED_PATH,
            "sha256": LOCKED_UPWBED_SHA256,
        },
        "dependencies": dependencies,
        "ordered_stages": list(ORDERED_STAGES),
        "face_role_policy": dict(FACE_ROLE_POLICY),
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
        "slope_stress_provenance": (
            "locked_erosed_exact_zero_never_capacity_or_shear_diagnostic"
            if bed_slope_method_id == SLOPE_BAGNOLD_IKEDA else
            "face_total_current_RP_USTAR_squared_after_secondary_divided_by_"
            "gravity_relative_density_D50_never_v63_capacity_diagnostic"
            if bed_slope_method_id != SLOPE_OFF else "not_consumed"
        ),
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "unsupported_gates": list(UNSUPPORTED_GATES),
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_scalar_flow_live_execution_v65(
    value: Mapping[str, object], *, scalar_execution: Mapping[str, object],
    producer_execution: Mapping[str, object], mixture_profile: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    selector7_settling_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 execution identity must be an object")
    try:
        dependencies = value["dependencies"]
        rebuilt = scalar_flow_live_execution_document_v65(
            scalar_execution=scalar_execution,
            producer_execution=producer_execution,
            mixture_profile=mixture_profile,
            channel_profile=channel_profile,
            backend=value["backend"],
            bed_slope_method_id=value["bed_slope_method_id"],
            slope_controls=value["slope_controls"],
            secondary_execution_fingerprint=dependencies["secondary_execution"],
            face_reconstruction_fingerprint=dependencies["face_reconstruction"],
            boundary_roles_fingerprint=dependencies["boundary_roles"],
            exner_fingerprint=dependencies["exner"],
            bed_slope_execution_fingerprint=dependencies["bed_slope_execution"],
            hiding_execution_fingerprint=dependencies["slope_hiding_execution"],
            selector7_settling_reference=selector7_settling_reference,
            upwind_bedload=value["upwind_bedload"],
            laterally_averaged_bedload=value["laterally_averaged_bedload"],
            lateral_stencil_fingerprint=dependencies["lateral_stencil"],
            bulk_density_kg_m3=value["bulk_density_kg_m3"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ScalarFlowLiveV65Error(
            "Invalid Q3-H5/v65 execution identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 execution identity is stale or forged")
    return rebuilt


@dataclass(frozen=True)
class ScalarFlowV65LateralStencil:
    """Component-specific locked ``upwbed`` transverse averaging.

    ``x_*`` averages the x transport through the transverse y neighbours
    (locked ``upwbed`` +/- ``icy``); ``y_*`` averages the y transport through
    +/- ``icx``.  Each row has one self entry and zero, one, or two unique
    transverse entries.  Every transverse entry has weight 0.25 and the self
    weight is ``1 - 0.25*ntransverse``.  The two operators must not be
    conflated.
    """

    ncell: int
    x_offsets: np.ndarray
    x_indices: np.ndarray
    x_weights: np.ndarray
    y_offsets: np.ndarray
    y_indices: np.ndarray
    y_weights: np.ndarray
    geometry_fingerprint: str
    fingerprint: str


def _csr_component(*, ncell: int, offsets: object, indices: object,
                   weights: object, name: str) -> tuple[np.ndarray, ...]:
    raw_offsets = np.asarray(offsets)
    raw_indices = np.asarray(indices)
    if raw_offsets.shape != (ncell + 1,) or raw_offsets.dtype.kind not in "iu":
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} offsets must have integer shape {(ncell + 1,)}")
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} indices must be a one-dimensional integer array")
    raw_weights = np.asarray(weights, dtype=np.float64)
    if raw_weights.shape != raw_indices.shape or not np.all(np.isfinite(raw_weights)):
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} CSR weights are incomplete or nonfinite")
    off = np.array(raw_offsets, dtype=np.int64, copy=True)
    ind = np.array(raw_indices, dtype=np.int64, copy=True)
    weight = np.array(raw_weights, dtype=np.float64, copy=True)
    if (off[0] != 0 or off[-1] != ind.size or np.any(off[1:] <= off[:-1]) or
            np.any(ind < 0) or np.any(ind >= ncell) or np.any(weight < 0.0)):
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 {name} CSR topology or weights are invalid")
    for cell in range(ncell):
        row_indices = ind[off[cell]:off[cell + 1]]
        row = weight[off[cell]:off[cell + 1]]
        if (row.size < 1 or row.size > 3 or
                np.unique(row_indices).size != row_indices.size or
                np.count_nonzero(row_indices == cell) != 1):
            raise ScalarFlowLiveV65Error(
                f"Q3-H5/v65 {name} row {cell} is not the locked one-self "
                "plus at-most-two-unique-neighbours stencil")
        self_position = int(np.flatnonzero(row_indices == cell)[0])
        transverse = np.delete(row, self_position)
        required_self = 1.0 - 0.25*transverse.size
        if (row[self_position] != required_self or
                np.any(transverse != 0.25)):
            raise ScalarFlowLiveV65Error(
                f"Q3-H5/v65 {name} row {cell} is not the locked upwbed "
                "quarter-neighbour/center-correction stencil")
    for value in (off, ind, weight):
        value.setflags(write=False)
    return off, ind, weight


def resolve_lateral_stencil_v65(
    *, ncell: int,
    x_offsets: object, x_indices: object, x_weights: object,
    y_offsets: object, y_indices: object, y_weights: object,
    geometry_fingerprint: str,
) -> ScalarFlowV65LateralStencil:
    if isinstance(ncell, bool) or not isinstance(ncell, int) or ncell <= 0:
        raise ScalarFlowLiveV65Error("Q3-H5/v65 lateral ncell must be positive")
    x = _csr_component(
        ncell=ncell, offsets=x_offsets, indices=x_indices, weights=x_weights,
        name="x-component")
    y = _csr_component(
        ncell=ncell, offsets=y_offsets, indices=y_indices, weights=y_weights,
        name="y-component")
    geometry = _sha(geometry_fingerprint, "lateral geometry")
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "ncell": ncell,
        "geometry_fingerprint": geometry,
        "x": {
            "offsets": x[0].tolist(), "indices": x[1].tolist(),
            "weights": x[2].tolist(),
            "locked_transverse_direction": "plus_minus_icy_for_su",
        },
        "y": {
            "offsets": y[0].tolist(), "indices": y[1].tolist(),
            "weights": y[2].tolist(),
            "locked_transverse_direction": "plus_minus_icx_for_sv",
        },
        "normalization": "locked_upwbed_quarter_neighbours_center_correction",
        "source_anchor": "locked_upwbed_f90_lines_113_126_and_154_167",
    }
    return ScalarFlowV65LateralStencil(
        ncell, *x, *y, geometry, _fingerprint(body))


def lateral_stencil_document_v65(
    stencil: ScalarFlowV65LateralStencil,
) -> dict[str, object]:
    if not isinstance(stencil, ScalarFlowV65LateralStencil):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 expected a lateral stencil")
    rebuilt = resolve_lateral_stencil_v65(
        ncell=stencil.ncell,
        x_offsets=stencil.x_offsets, x_indices=stencil.x_indices,
        x_weights=stencil.x_weights,
        y_offsets=stencil.y_offsets, y_indices=stencil.y_indices,
        y_weights=stencil.y_weights,
        geometry_fingerprint=stencil.geometry_fingerprint,
    )
    if rebuilt.fingerprint != stencil.fingerprint:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 lateral stencil identity is stale or forged")
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "ncell": rebuilt.ncell,
        "geometry_fingerprint": rebuilt.geometry_fingerprint,
        "x": {
            "offsets": rebuilt.x_offsets.tolist(),
            "indices": rebuilt.x_indices.tolist(),
            "weights": rebuilt.x_weights.tolist(),
            "locked_transverse_direction": "plus_minus_icy_for_su",
        },
        "y": {
            "offsets": rebuilt.y_offsets.tolist(),
            "indices": rebuilt.y_indices.tolist(),
            "weights": rebuilt.y_weights.tolist(),
            "locked_transverse_direction": "plus_minus_icx_for_sv",
        },
        "normalization": "locked_upwbed_quarter_neighbours_center_correction",
        "source_anchor": "locked_upwbed_f90_lines_113_126_and_154_167",
        "fingerprint": rebuilt.fingerprint,
    }


def _apply_csr_component(
    values: np.ndarray, offsets: np.ndarray, indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # values: (nfraction,ncell); scalar loop is the portable CPU oracle.
    result = np.empty_like(values)
    for cell in range(values.shape[1]):
        start, stop = int(offsets[cell]), int(offsets[cell + 1])
        result[:, cell] = np.sum(
            values[:, indices[start:stop]]*weights[start:stop][None, :], axis=1)
    return result


def apply_lateral_stencil_v65(
    bed_mass: object, stencil: ScalarFlowV65LateralStencil,
) -> np.ndarray:
    if not isinstance(stencil, ScalarFlowV65LateralStencil):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 lateral stencil is missing")
    ncell = stencil.ncell
    raw = np.asarray(bed_mass, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[0] != 2 or raw.shape[2] != ncell or \
            not np.all(np.isfinite(raw)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 lateral bed field must have shape (2,nfraction,ncell)")
    output = np.empty_like(raw)
    output[0] = _apply_csr_component(
        raw[0], stencil.x_offsets, stencil.x_indices, stencil.x_weights)
    output[1] = _apply_csr_component(
        raw[1], stencil.y_offsets, stencil.y_indices, stencil.y_weights)
    if not np.all(np.isfinite(output)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 lateral stencil produced nonfinite transport")
    return output


@dataclass(frozen=True)
class ScalarFlowV65State:
    """Accepted fixed-grid state; arrays are immutable C-order float64."""

    execution_fingerprint: str
    bed_mass_kg_m2: np.ndarray       # (nfraction,ncell), nonnegative; cell total > 0
    bulk_density_kg_m3: np.ndarray   # (nfraction,), execution-bound
    active_fraction: np.ndarray      # (nfraction,ncell), column normalized
    boundary_signed_kg: np.ndarray   # (nfraction,), positive out of domain
    boundary_absolute_kg: np.ndarray # (nfraction,)
    accepted_step: int
    time_s: float
    state_sha256: str


def resolve_state_v65(
    *, execution: Mapping[str, object], bed_mass_kg_m2: object,
    active_fraction: object | None = None,
    boundary_signed_kg: object | None = None,
    boundary_absolute_kg: object | None = None,
    accepted_step: int = 0, time_s: float = 0.0,
) -> ScalarFlowV65State:
    resolved_execution = _runtime_execution(execution)
    execution_fingerprint = str(resolved_execution["fingerprint"])
    raw_mass = np.asarray(bed_mass_kg_m2, dtype=np.float64)
    if raw_mass.ndim != 2 or min(raw_mass.shape) <= 0:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 bed mass must have shape (nfraction,ncell)")
    nfrac, ncell = raw_mass.shape
    mass = _array(raw_mass, (nfrac, ncell), "bed mass")
    if np.any(mass < 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 bed mass may not be negative")
    density = _array(
        resolved_execution["bulk_density_kg_m3"], (nfrac,), "bulk density")
    if np.any(density <= 0.0):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 bulk density must be positive")
    volume = np.asarray(mass)/np.asarray(density)[:, None]
    total_volume = np.sum(volume, axis=0, keepdims=True)
    positive_stock = total_volume[0] > 0.0
    derived = np.zeros_like(volume)
    derived[:, positive_stock] = (
        volume[:, positive_stock]/total_volume[:, positive_stock])
    if active_fraction is None:
        if np.any(~positive_stock):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 depleted cells require an explicit retained "
                "surface identity")
        fraction = _array(derived, (nfrac, ncell), "active fraction")
    else:
        fraction = _array(
            active_fraction, (nfrac, ncell), "active fraction")
        tolerance = 512.0*np.finfo(np.float64).eps
        if (np.any(fraction < 0.0) or
                not np.allclose(np.sum(fraction, axis=0), 1.0,
                                rtol=0.0, atol=tolerance*nfrac) or
                not np.allclose(fraction[:, positive_stock],
                                derived[:, positive_stock],
                                rtol=tolerance, atol=tolerance)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 active fractions differ from positive accepted "
                "solid-volume stock or are not normalized retained identities")
    signed = _array(
        np.zeros(nfrac) if boundary_signed_kg is None else boundary_signed_kg,
        (nfrac,), "signed boundary ledger")
    absolute = _array(
        np.zeros(nfrac) if boundary_absolute_kg is None else boundary_absolute_kg,
        (nfrac,), "absolute boundary ledger")
    if np.any(absolute < 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 absolute boundary ledger must be nonnegative")
    if (isinstance(accepted_step, bool) or not isinstance(accepted_step, int) or
            accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 accepted step is invalid")
    accepted_time = _finite(time_s, "accepted time", nonnegative=True)
    digest = _array_digest(mass, density, fraction, signed, absolute)
    body = {
        "execution_fingerprint": execution_fingerprint,
        "accepted_step": accepted_step,
        "time_s": accepted_time,
        "array_sha256": digest,
    }
    return ScalarFlowV65State(
        execution_fingerprint, mass, density, fraction, signed, absolute, accepted_step,
        accepted_time, _fingerprint(body))


@dataclass(frozen=True)
class ScalarFlowV65CellChannels:
    """Post-channel cell fields bound to one execution and v64 producer."""

    execution_fingerprint: str
    producer_execution_fingerprint: str
    composed_bed_mass: np.ndarray           # (2,nfraction,ncell)
    retained_suspended_mass: np.ndarray     # (2,nfraction,ncell)
    signed_suspended_mass_rate: np.ndarray  # (nfraction,ncell)
    equilibrium_concentration: np.ndarray   # (nfraction,ncell)
    total_transport_direction: np.ndarray   # (2,nfraction,ncell)
    digest: str


@dataclass(frozen=True)
class ScalarFlowV65Production:
    """Authenticated CPU v64-output receipt for the public live entry point.

    The portable component adapter remains useful for operator oracles, but a
    live proposal must carry this execution/producer-bound receipt.  The
    receipt also owns the total-current USTAR-squared field used to derive
    active-slope face forcing; a matching fingerprint string alone is not
    accepted as provenance.
    """

    execution_fingerprint: str
    producer_execution_fingerprint: str
    producer_output_sha256: str
    slope_total_current_ust2: np.ndarray  # (ncell,), after secondary
    channels: ScalarFlowV65CellChannels
    production_sha256: str


@dataclass(frozen=True)
class ScalarFlowV65SlopeFields:
    """Inputs consumed only by a live classic-FLOW face correction."""

    downhill: np.ndarray       # (2,nface), positive downhill
    total_current_ust2: np.ndarray  # (nface,), after secondary, face-averaged
    ust2_provenance_fingerprint: str # selected v64 producer execution
    stress_semantics: str
    gravity_m_s2: float
    solid_density_kg_m3: np.ndarray # (nfraction,)
    water_density_kg_m3: np.ndarray # (nface,)
    stress: np.ndarray         # taurat for ISlope2; absolute Shields for 3/4
    depth: np.ndarray          # (nfraction,nface)
    d50: np.ndarray            # (nfraction,nface)
    mean_diameter: np.ndarray  # (nfraction,nface)
    hiding: np.ndarray         # (nfraction,ncell), ISlope4 only
    source_production_sha256: str | None
    digest: str


@dataclass(frozen=True)
class ScalarFlowV65FaceBatch:
    """One canonical face catalog and optional raw trace capacities.

    Normals point from ``owner`` toward ``neighbour`` or out of the domain.
    Trace raw channels are consumed only for trace inlet/outlet roles and must
    be exact zero elsewhere; this prevents stale or duplicate trace capacity.
    """

    owner: np.ndarray                    # (nface,) int64
    neighbour: np.ndarray                # (nface,) int64, -1 on boundary
    role: np.ndarray                     # (nface,) int64 FACE_*
    normal: np.ndarray                   # (2,nface)
    length_m: np.ndarray                 # (nface,)
    trace_steered_bed_mass: np.ndarray   # (2,nfraction,nface)
    trace_raw_suspended_mass: np.ndarray # (2,nfraction,nface)
    trace_base_velocity: np.ndarray      # (2,nface)
    trace_depth_m: np.ndarray            # (nface,)
    slope: ScalarFlowV65SlopeFields | None = None


@dataclass(frozen=True)
class ScalarFlowV65Trial:
    """Complete immutable candidate produced from one accepted parent."""

    execution_fingerprint: str
    parent_state_sha256: str
    dt_s: float
    morphological_factor: float
    face_raw_candidate: np.ndarray       # (2,nfraction,nface)
    face_corrected_candidate: np.ndarray # (2,nfraction,nface)
    raw_donor: np.ndarray                # (nfraction,nface)
    donor: np.ndarray                    # (nfraction,nface)
    active_fraction_factor: np.ndarray   # (nfraction,ncell)
    inventory_fill: np.ndarray           # (nfraction,ncell)
    availability: np.ndarray             # product, (nfraction,ncell)
    face_flux_kg_m_s: np.ndarray         # (nfraction,nface)
    cell_delta_kg_m2: np.ndarray         # (nfraction,ncell)
    candidate: ScalarFlowV65State
    boundary_transfer_kg: np.ndarray     # (nfraction,), positive outward
    depletion_roundoff_transfer_kg: np.ndarray # recorded positive outward
    conservation_residual_kg: np.ndarray # (nfraction,)
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str


def _runtime_execution(value: Mapping[str, object], *, cpu: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 runtime execution is missing")
    if (value.get("schema") != SCHEMA or value.get("scope_id") != SCOPE_ID or
            value.get("semantic_id") != SEMANTIC_ID or
            value.get("envelope_version") != ENVELOPE_VERSION):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 runtime execution header differs")
    supplied = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if supplied != _fingerprint(body):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 runtime execution fingerprint differs")
    if cpu and value.get("backend") != "cpu":
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 portable CPU transaction rejects non-CPU execution")
    return dict(value)


def _runtime_channel_profile(
    execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
) -> ChannelCompositionProfile:
    profile = (channel_profile if isinstance(channel_profile, ChannelCompositionProfile)
               else load_channel_composition_profile(channel_profile))
    document = channel_composition_profile_document(profile)
    dependency = execution["dependencies"]["channel_profile"]
    if document["fingerprint"] != dependency:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 runtime channel profile differs from execution")
    ids = [value.fraction_id for value in profile.fractions]
    classes = [value.transport_class for value in profile.fractions]
    if ids != execution["ordered_fraction_ids"] or \
            classes != execution["transport_classes"]:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 runtime channel layout differs from execution")
    return profile


def compose_cell_channels_v65(
    *, execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    steered_bed_mass: object,
    raw_suspended_mass: object,
    base_velocity: object,
    depth_m: object,
) -> ScalarFlowV65CellChannels:
    """Apply channel factors/folding for an explicit component/oracle input.

    ``steered_bed_mass`` is already post-secondary.  ``raw_suspended_mass``
    remains aligned with the unrotated ``base_velocity`` and is never rotated.
    The returned channel receipt is bound to the selected execution and v64
    producer identity, but only :func:`compose_cell_production_v65` creates the
    production receipt required by the public live transaction.
    """
    resolved = _runtime_execution(execution)
    profile = _runtime_channel_profile(resolved, channel_profile)
    nfrac = len(profile.fractions)
    raw_bed = np.asarray(steered_bed_mass, dtype=np.float64)
    if raw_bed.ndim != 3 or raw_bed.shape[0] != 2 or raw_bed.shape[1] != nfrac:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 steered bed must have shape (2,nfraction,ncell)")
    ncell = raw_bed.shape[2]
    raw_bed = _array(raw_bed, (2, nfrac, ncell), "steered bed", copy=False)
    raw_suspended = _array(
        raw_suspended_mass, (2, nfrac, ncell), "raw suspended", copy=False)
    velocity = _array(base_velocity, (2, ncell), "base velocity", copy=False)
    depth = _array(depth_m, (ncell,), "cell depth", copy=False)
    if np.any(depth <= 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 is fully wet only; every cell depth must be positive")
    bed = np.empty_like(raw_bed)
    suspended = np.empty_like(raw_suspended)
    signed = np.empty((nfrac, ncell), dtype=np.float64)
    concentration = np.empty((nfrac, ncell), dtype=np.float64)
    for fraction, row in enumerate(profile.fractions):
        for cell in range(ncell):
            result = compose_mass_channels(
                depth=float(depth[cell]),
                velocity_x=float(velocity[0, cell]),
                velocity_y=float(velocity[1, cell]),
                raw_bed_x=float(raw_bed[0, fraction, cell]),
                raw_bed_y=float(raw_bed[1, fraction, cell]),
                raw_suspended_x=float(raw_suspended[0, fraction, cell]),
                raw_suspended_y=float(raw_suspended[1, fraction, cell]),
                transport_class=row.transport_class,
                bed_factor=profile.bed_factor,
                suspended_factor=profile.suspended_factor,
                epsilon=profile.epsilon,
            )
            bed[:, fraction, cell] = (
                result.composed_bed_x, result.composed_bed_y)
            suspended[:, fraction, cell] = (
                result.retained_suspended_x, result.retained_suspended_y)
            signed[fraction, cell] = result.signed_suspended_rate
            concentration[fraction, cell] = result.equilibrium_concentration
    # sutot/svtot is a separate sign-control field in locked upwbed.  In this
    # admitted current-only scalar route it is the composed current bed channel
    # plus the retained formula-suspended channel where applicable.
    total = bed + suspended
    if not all(np.all(np.isfinite(value)) for value in (
            bed, suspended, signed, concentration, total)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 channel composition produced nonfinite output")
    frozen = []
    for value in (bed, suspended, signed, concentration, total):
        item = np.array(value, dtype=np.float64, copy=True, order="C")
        item.setflags(write=False)
        frozen.append(item)
    execution_fingerprint = str(resolved["fingerprint"])
    producer_fingerprint = str(
        resolved["dependencies"]["backend_producer_execution"])
    body = {
        "execution_fingerprint": execution_fingerprint,
        "producer_execution_fingerprint": producer_fingerprint,
        "arrays_sha256": _array_digest(*frozen),
    }
    return ScalarFlowV65CellChannels(
        execution_fingerprint, producer_fingerprint, *frozen,
        _fingerprint(body))


def _validate_cell_channels_v65(
    channels: ScalarFlowV65CellChannels, *, execution: Mapping[str, object],
    nfrac: int, ncell: int,
) -> None:
    expected_shape = (2, nfrac, ncell)
    producer_fingerprint = execution["dependencies"]["backend_producer_execution"]
    if (not isinstance(channels, ScalarFlowV65CellChannels) or
            channels.execution_fingerprint != execution["fingerprint"] or
            channels.producer_execution_fingerprint != producer_fingerprint or
            channels.composed_bed_mass.shape != expected_shape or
            channels.retained_suspended_mass.shape != expected_shape or
            channels.signed_suspended_mass_rate.shape != (nfrac, ncell) or
            channels.equilibrium_concentration.shape != (nfrac, ncell) or
            channels.total_transport_direction.shape != expected_shape):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 cell-channel execution, producer, or shape differs")
    arrays_sha256 = _array_digest(
        channels.composed_bed_mass, channels.retained_suspended_mass,
        channels.signed_suspended_mass_rate,
        channels.equilibrium_concentration,
        channels.total_transport_direction)
    expected = _fingerprint({
        "execution_fingerprint": channels.execution_fingerprint,
        "producer_execution_fingerprint": channels.producer_execution_fingerprint,
        "arrays_sha256": arrays_sha256,
    })
    if channels.digest != expected:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 cell channels are stale, mutated, or unbound")


def compose_cell_production_v65(
    *, execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    steered_bed_mass: object, raw_suspended_mass: object,
    base_velocity: object, depth_m: object,
    slope_total_current_ust2: object,
) -> ScalarFlowV65Production:
    """Create the execution-bound receipt required by the CPU live proposal.

    This is the narrow hand-off from the selected v64 producer.  It hashes the
    exact upstream arrays before channel composition and retains the producer's
    post-secondary total-current USTAR-squared field for slope provenance.
    Synthetic operator fixtures should use ``compose_cell_channels_v65`` and
    ``begin_component_bed_exner_trial_v65`` instead.
    """
    resolved = _runtime_execution(execution, cpu=True)
    channels = compose_cell_channels_v65(
        execution=resolved, channel_profile=channel_profile,
        steered_bed_mass=steered_bed_mass,
        raw_suspended_mass=raw_suspended_mass,
        base_velocity=base_velocity, depth_m=depth_m)
    ncell = channels.composed_bed_mass.shape[2]
    ust2 = _array(
        slope_total_current_ust2, (ncell,),
        "production total-current RP_USTAR squared")
    if np.any(ust2 < 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 production RP_USTAR squared must be nonnegative")
    raw_bed = _array(
        steered_bed_mass, channels.composed_bed_mass.shape,
        "production steered bed", copy=False)
    raw_suspended = _array(
        raw_suspended_mass, channels.composed_bed_mass.shape,
        "production raw suspended", copy=False)
    velocity = _array(base_velocity, (2, ncell), "production base velocity",
                      copy=False)
    depth = _array(depth_m, (ncell,), "production depth", copy=False)
    producer_output_sha256 = _array_digest(
        raw_bed, raw_suspended, velocity, depth, ust2)
    body = {
        "execution_fingerprint": resolved["fingerprint"],
        "producer_execution_fingerprint": channels.producer_execution_fingerprint,
        "producer_output_sha256": producer_output_sha256,
        "slope_total_current_ust2_sha256": _array_digest(ust2),
        "channel_receipt_sha256": channels.digest,
    }
    return ScalarFlowV65Production(
        str(resolved["fingerprint"]), channels.producer_execution_fingerprint,
        producer_output_sha256, ust2, channels, _fingerprint(body))


def _validate_production_v65(
    production: ScalarFlowV65Production, *, execution: Mapping[str, object],
    nfrac: int, ncell: int,
) -> None:
    if (not isinstance(production, ScalarFlowV65Production) or
            production.execution_fingerprint != execution["fingerprint"] or
            production.producer_execution_fingerprint !=
            execution["dependencies"]["backend_producer_execution"] or
            not isinstance(production.producer_output_sha256, str) or
            _SHA.fullmatch(production.producer_output_sha256) is None):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 live proposal requires its authenticated v64 production")
    _validate_cell_channels_v65(
        production.channels, execution=execution, nfrac=nfrac, ncell=ncell)
    ust2 = _array(
        production.slope_total_current_ust2, (ncell,),
        "production total-current RP_USTAR squared", copy=False)
    if np.any(ust2 < 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 production RP_USTAR squared must be nonnegative")
    body = {
        "execution_fingerprint": production.execution_fingerprint,
        "producer_execution_fingerprint": production.producer_execution_fingerprint,
        "producer_output_sha256": production.producer_output_sha256,
        "slope_total_current_ust2_sha256": _array_digest(ust2),
        "channel_receipt_sha256": production.channels.digest,
    }
    if production.production_sha256 != _fingerprint(body):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 v64 production receipt is stale or mutated")


def _resolve_slope_fields_v65_impl(
    *, nfrac: int, ncell: int, nface: int,
    downhill: object, total_current_ust2: object,
    total_current_ust2_provenance_fingerprint: str,
    gravity_m_s2: float, solid_density_kg_m3: object,
    water_density_kg_m3: object, depth: object, d50: object,
    mean_diameter: object, hiding: object,
    source_production_fingerprint: str | None = None,
    bed_slope_method_id: str = SLOPE_KOCH_FLOKSTRA,
) -> ScalarFlowV65SlopeFields:
    down = _array(downhill, (2, nface), "slope downhill")
    ust2 = _array(
        total_current_ust2, (nface,), "face total-current RP_USTAR squared")
    provenance = _sha(
        total_current_ust2_provenance_fingerprint,
        "total-current RP_USTAR squared provenance")
    source_production = (
        None if source_production_fingerprint is None else
        _sha(source_production_fingerprint, "source production receipt"))
    gravity = _finite(gravity_m_s2, "slope gravity", positive=True)
    solid = _array(
        solid_density_kg_m3, (nfrac,), "slope solid density")
    water = _array(
        water_density_kg_m3, (nface,), "slope water density")
    water_depth = _array(depth, (nfrac, nface), "slope depth")
    fraction_d50 = _array(d50, (nfrac, nface), "slope D50")
    mean = _array(
        mean_diameter, (nfrac, nface), "slope mean diameter")
    hide = _array(hiding, (nfrac, ncell), "slope hiding")
    if (np.any(ust2 < 0.0) or np.any(solid <= 0.0) or np.any(water <= 0.0) or
            np.any(solid[:, None] <= water[None, :]) or
            np.any(water_depth <= 0.0) or
            np.any(fraction_d50 <= 0.0) or np.any(mean <= 0.0) or
            np.any(hide < 0.0)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 slope fields are outside their physical domain")
    if bed_slope_method_id == SLOPE_BAGNOLD_IKEDA:
        stress_semantics = TAURAT_POSITIVE_ZERO
        theta = _array(
            np.zeros((nfrac, nface), dtype=np.float64),
            (nfrac, nface), "locked positive-TraFrm taurat")
    elif bed_slope_method_id in (SLOPE_KOCH_FLOKSTRA, SLOPE_PARKER_ANDREWS):
        stress_semantics = "absolute_shields_from_total_current_RP_USTAR_squared"
        relative = (np.asarray(solid)[:, None] - np.asarray(water)[None, :])/\
            np.asarray(water)[None, :]
        theta_value = np.asarray(ust2)[None, :]/gravity/relative/np.asarray(
            fraction_d50)
        theta = _array(theta_value, (nfrac, nface), "derived slope Shields")
    else:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 slope fields require live ISlope 2, 3 or 4")
    body = {
        "provenance": provenance,
        "source_production_sha256": source_production,
        "stress_semantics": stress_semantics,
        "gravity_m_s2": gravity,
        "arrays_sha256": _array_digest(
            down, ust2, solid, water, theta, water_depth, fraction_d50,
            mean, hide),
        "stress_derivation": (
            "exact_zero_BRAIDS_binding_to_FLOW_erosed_caller_intent_"
            "with_gfortran_byte_preservation_evidence"
            if bed_slope_method_id == SLOPE_BAGNOLD_IKEDA else
            "total_current_RP_USTAR_squared_after_secondary_divided_by_"
            "gravity_relative_density_D50"),
    }
    return ScalarFlowV65SlopeFields(
        down, ust2, provenance, stress_semantics, gravity, solid, water,
        theta, water_depth,
        fraction_d50, mean, hide, source_production, _fingerprint(body))


def resolve_slope_fields_v65(
    *, nfrac: int, ncell: int, nface: int,
    downhill: object, total_current_ust2: object,
    total_current_ust2_provenance_fingerprint: str,
    gravity_m_s2: float, solid_density_kg_m3: object,
    water_density_kg_m3: object, depth: object, d50: object,
    mean_diameter: object, hiding: object,
    bed_slope_method_id: str = SLOPE_KOCH_FLOKSTRA,
) -> ScalarFlowV65SlopeFields:
    """Build component-only slope fields; no production provenance is minted."""
    return _resolve_slope_fields_v65_impl(
        nfrac=nfrac, ncell=ncell, nface=nface, downhill=downhill,
        total_current_ust2=total_current_ust2,
        total_current_ust2_provenance_fingerprint=(
            total_current_ust2_provenance_fingerprint),
        gravity_m_s2=gravity_m_s2,
        solid_density_kg_m3=solid_density_kg_m3,
        water_density_kg_m3=water_density_kg_m3, depth=depth, d50=d50,
        mean_diameter=mean_diameter, hiding=hiding,
        source_production_fingerprint=None,
        bed_slope_method_id=bed_slope_method_id)


def resolve_slope_fields_from_production_v65(
    *, production: ScalarFlowV65Production,
    execution: Mapping[str, object], nfrac: int, ncell: int,
    owner: object, neighbour: object, role: object,
    downhill: object, gravity_m_s2: float, solid_density_kg_m3: object,
    water_density_kg_m3: object, depth: object, d50: object,
    mean_diameter: object, hiding: object,
) -> ScalarFlowV65SlopeFields:
    """Derive face USTAR-squared only from the bound CPU v64 production."""
    resolved = _runtime_execution(execution, cpu=True)
    if resolved["bed_slope_method_id"] == SLOPE_OFF:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 slope-off production may not create slope fields")
    _validate_production_v65(
        production, execution=resolved, nfrac=nfrac, ncell=ncell)
    own_raw = np.asarray(owner)
    if own_raw.ndim != 1:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 slope owner must have shape (nface,)")
    nface = own_raw.size
    own = _integer_array(owner, (nface,), "slope owner")
    other = _integer_array(neighbour, (nface,), "slope neighbour")
    roles = _integer_array(role, (nface,), "slope role")
    valid = np.isin(roles, [
        FACE_INTERNAL, FACE_TRACE_INLET, FACE_TRACE_OUTLET,
        FACE_OWNER_OUTLET, FACE_CLOSED])
    internal = roles == FACE_INTERNAL
    if (np.any(~valid) or np.any(own < 0) or np.any(own >= ncell) or
            np.any(other[internal] < 0) or np.any(other[internal] >= ncell) or
            np.any(other[internal] == own[internal]) or
            np.any(other[~internal] != -1)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 production slope topology differs")
    cell_ust2 = np.asarray(production.slope_total_current_ust2)
    face_ust2 = np.asarray(cell_ust2[own], dtype=np.float64).copy()
    face_ust2[internal] = 0.5*(
        cell_ust2[own[internal]]+cell_ust2[other[internal]])
    return _resolve_slope_fields_v65_impl(
        nfrac=nfrac, ncell=ncell, nface=nface, downhill=downhill,
        total_current_ust2=face_ust2,
        total_current_ust2_provenance_fingerprint=(
            production.producer_execution_fingerprint),
        gravity_m_s2=gravity_m_s2,
        solid_density_kg_m3=solid_density_kg_m3,
        water_density_kg_m3=water_density_kg_m3, depth=depth, d50=d50,
        mean_diameter=mean_diameter, hiding=hiding,
        source_production_fingerprint=production.production_sha256,
        bed_slope_method_id=str(resolved["bed_slope_method_id"]))


def resolve_face_batch_v65(
    *, nfrac: int, ncell: int,
    owner: object, neighbour: object, role: object,
    normal: object, length_m: object,
    trace_steered_bed_mass: object | None = None,
    trace_raw_suspended_mass: object | None = None,
    trace_base_velocity: object | None = None,
    trace_depth_m: object | None = None,
    slope: ScalarFlowV65SlopeFields | None = None,
) -> ScalarFlowV65FaceBatch:
    raw_owner = np.asarray(owner)
    if raw_owner.ndim != 1:
        raise ScalarFlowLiveV65Error("Q3-H5/v65 owner must have shape (nface,)")
    nface = raw_owner.size
    own = _integer_array(owner, (nface,), "owner")
    other = _integer_array(neighbour, (nface,), "neighbour")
    roles = _integer_array(role, (nface,), "face role")
    normals = _array(normal, (2, nface), "face normal")
    length = _array(length_m, (nface,), "face length")
    if np.any(own < 0) or np.any(own >= ncell):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face owner is outside the mesh")
    if np.any(length <= 0.0):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face lengths must be positive")
    tolerance = 1024.0*np.finfo(np.float64).eps
    if not np.allclose(np.sum(normals*normals, axis=0), 1.0,
                       rtol=0.0, atol=tolerance):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face normals must be unit length")
    valid_roles = {FACE_INTERNAL, FACE_TRACE_INLET, FACE_TRACE_OUTLET,
                   FACE_OWNER_OUTLET, FACE_CLOSED}
    if any(int(value) not in valid_roles for value in roles):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face role is invalid")
    internal = roles == FACE_INTERNAL
    boundary = ~internal
    if (np.any(other[internal] < 0) or np.any(other[internal] >= ncell) or
            np.any(other[internal] == own[internal]) or
            np.any(other[boundary] != -1)):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face topology and roles differ")
    zeros_vector = np.zeros((2, nfrac, nface), dtype=np.float64)
    zeros_velocity = np.zeros((2, nface), dtype=np.float64)
    zeros_depth = np.zeros(nface, dtype=np.float64)
    trace_bed = _array(
        zeros_vector if trace_steered_bed_mass is None else
        trace_steered_bed_mass,
        (2, nfrac, nface), "trace steered bed")
    trace_suspended = _array(
        zeros_vector if trace_raw_suspended_mass is None else
        trace_raw_suspended_mass,
        (2, nfrac, nface), "trace suspended")
    trace_velocity = _array(
        zeros_velocity if trace_base_velocity is None else trace_base_velocity,
        (2, nface), "trace base velocity")
    trace_depth = _array(
        zeros_depth if trace_depth_m is None else trace_depth_m,
        (nface,), "trace depth")
    trace_roles = (roles == FACE_TRACE_INLET) | (roles == FACE_TRACE_OUTLET)
    nontrace = ~trace_roles
    if (np.any(trace_bed[:, :, nontrace] != 0.0) or
            np.any(trace_suspended[:, :, nontrace] != 0.0) or
            np.any(trace_velocity[:, nontrace] != 0.0) or
            np.any(trace_depth[nontrace] != 0.0)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 stale trace fields supplied to a non-trace face")
    if np.any(trace_depth[trace_roles] <= 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 trace faces must be fully wet")
    resolved_slope = slope
    if slope is not None:
        if not isinstance(slope, ScalarFlowV65SlopeFields):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 slope field shapes differ from face batch")
        rebuilt_slope = _resolve_slope_fields_v65_impl(
            nfrac=nfrac, ncell=ncell, nface=nface,
            downhill=slope.downhill,
            total_current_ust2=slope.total_current_ust2,
            total_current_ust2_provenance_fingerprint=(
                slope.ust2_provenance_fingerprint),
            gravity_m_s2=slope.gravity_m_s2,
            solid_density_kg_m3=slope.solid_density_kg_m3,
            water_density_kg_m3=slope.water_density_kg_m3,
            depth=slope.depth, d50=slope.d50,
            mean_diameter=slope.mean_diameter, hiding=slope.hiding,
            source_production_fingerprint=slope.source_production_sha256,
            bed_slope_method_id=(
                SLOPE_BAGNOLD_IKEDA
                if slope.stress_semantics == TAURAT_POSITIVE_ZERO else
                SLOPE_KOCH_FLOKSTRA))
        if (rebuilt_slope.digest != slope.digest or
                rebuilt_slope.stress_semantics != slope.stress_semantics or
                not np.array_equal(rebuilt_slope.stress, slope.stress)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 slope fields do not match total-current "
                "RP_USTAR-squared derivation")
        resolved_slope = rebuilt_slope
    return ScalarFlowV65FaceBatch(
        own, other, roles, normals, length, trace_bed, trace_suspended,
        trace_velocity, trace_depth, resolved_slope)


def _compose_trace_channels(
    batch: ScalarFlowV65FaceBatch, profile: ChannelCompositionProfile,
) -> tuple[np.ndarray, np.ndarray]:
    nfrac = len(profile.fractions)
    nface = batch.owner.size
    trace_bed = np.zeros((2, nfrac, nface), dtype=np.float64)
    trace_suspended = np.zeros_like(trace_bed)
    for face in range(nface):
        if batch.role[face] not in (FACE_TRACE_INLET, FACE_TRACE_OUTLET):
            continue
        for fraction, row in enumerate(profile.fractions):
            result = compose_mass_channels(
                depth=float(batch.trace_depth_m[face]),
                velocity_x=float(batch.trace_base_velocity[0, face]),
                velocity_y=float(batch.trace_base_velocity[1, face]),
                raw_bed_x=float(batch.trace_steered_bed_mass[0, fraction, face]),
                raw_bed_y=float(batch.trace_steered_bed_mass[1, fraction, face]),
                raw_suspended_x=float(
                    batch.trace_raw_suspended_mass[0, fraction, face]),
                raw_suspended_y=float(
                    batch.trace_raw_suspended_mass[1, fraction, face]),
                transport_class=row.transport_class,
                bed_factor=profile.bed_factor,
                suspended_factor=profile.suspended_factor,
                epsilon=profile.epsilon,
            )
            trace_bed[:, fraction, face] = (
                result.composed_bed_x, result.composed_bed_y)
            trace_suspended[:, fraction, face] = (
                result.retained_suspended_x, result.retained_suspended_y)
    return trace_bed, trace_suspended


def _apply_flow_slope_v65(
    q: np.ndarray, downhill: np.ndarray, stress: float, depth: float,
    d50: float, mean_diameter: float, hiding_left: float,
    hiding_right: float, method_id: str, controls: Mapping[str, float],
) -> np.ndarray:
    """Portable transcription of accepted H4 ISlope 2/3/4 arithmetic."""
    result = np.array(q, dtype=np.float64, copy=True)
    magnitude = math.hypot(float(q[0]), float(q[1]))
    if not math.isfinite(magnitude):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 nonfinite face transport before slope")
    if magnitude <= 1.0e-6 or method_id == SLOPE_OFF:
        return result
    if (stress < 0.0 or depth <= 0.0 or d50 <= 0.0 or mean_diameter <= 0.0 or
            min(hiding_left, hiding_right) < 0.0):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 active slope input is outside its physical domain")
    dzds = (float(downhill[0])*float(q[0]) +
            float(downhill[1])*float(q[1]))/magnitude
    default_ratio = float(np.float32(30.0)/np.float32(180.0))
    tangent = math.tan(default_ratio*math.acos(-1.0))
    dzds = min(float(np.float32(0.9))*tangent, dzds)
    if method_id == SLOPE_BAGNOLD_IKEDA:
        if stress != 0.0:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 positive-TraFrm ISlope 2 requires exact-zero taurat")
        dzdn = (-float(downhill[0])*float(q[1]) +
                float(downhill[1])*float(q[0]))/magnitude
        denominator = math.cos(math.atan(dzds))*(tangent-dzds)
        if denominator == 0.0 or not math.isfinite(denominator):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 Bagnold longitudinal denominator is singular")
        bagnold = tangent/denominator
        along = max(0.0, 1.0+controls["alfabs"]*(bagnold-1.0))
        normal = controls["alfabn"]*dzdn
        result[:] = (
            along*float(q[0])-along*float(q[1])*normal,
            along*float(q[1])+along*float(q[0])*normal,
        )
        if not np.all(np.isfinite(result)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 FLOW slope produced nonfinite transport")
        return result
    if method_id == SLOPE_KOCH_FLOKSTRA:
        ftheta = 0.0
        if stress != 0.0:
            ftheta = (controls["ashld"]*(stress**controls["bshld"])*
                      ((d50/depth)**controls["cshld"])*
                      ((d50/mean_diameter)**controls["dshld"]))
    elif method_id == SLOPE_PARKER_ANDREWS:
        ftheta = 0.0
        if stress != 0.0:
            hiding = 0.5*(hiding_left+hiding_right)
            denominator = max(stress*0.1, hiding*controls["thcrpa"])
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 Parker slope denominator is singular")
            ftheta = controls["alfpa"]*math.sqrt(stress/denominator)
    else:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 positive-TraFrm slope method is not admitted")
    if float(downhill[0]) != 0.0 or float(downhill[1]) != 0.0:
        sina = ftheta*float(q[0])/magnitude + float(downhill[0])
        cosa = ftheta*float(q[1])/magnitude + float(downhill[1])
    else:
        sina = float(q[0])/magnitude
        cosa = float(q[1])/magnitude
    norm = math.hypot(sina, cosa)
    if norm <= np.finfo(np.float64).tiny or not math.isfinite(norm):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 FLOW slope direction is singular")
    corrected_magnitude = magnitude*(1.0+controls["alfabs"]*dzds)
    result[:] = (corrected_magnitude*sina/norm,
                 corrected_magnitude*cosa/norm)
    if not np.all(np.isfinite(result)):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 FLOW slope produced nonfinite transport")
    return result


def _validate_state_for_execution(
    state: ScalarFlowV65State, execution: Mapping[str, object],
) -> None:
    if not isinstance(state, ScalarFlowV65State):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 accepted state is missing")
    if state.execution_fingerprint != execution["fingerprint"]:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 state and execution fingerprints differ")
    rebuilt = resolve_state_v65(
        execution=execution, bed_mass_kg_m2=state.bed_mass_kg_m2,
        active_fraction=state.active_fraction,
        boundary_signed_kg=state.boundary_signed_kg,
        boundary_absolute_kg=state.boundary_absolute_kg,
        accepted_step=state.accepted_step, time_s=state.time_s)
    if rebuilt.state_sha256 != state.state_sha256:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 accepted state is stale or forged")


def _face_candidates_v65(
    *, execution: Mapping[str, object], channels: ScalarFlowV65CellChannels,
    batch: ScalarFlowV65FaceBatch, profile: ChannelCompositionProfile,
    lateral_stencil: ScalarFlowV65LateralStencil | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nfrac, ncell = channels.composed_bed_mass.shape[1:]
    nface = batch.owner.size
    if execution["laterally_averaged_bedload"]:
        if lateral_stencil is None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 lateral averaging requires its explicit stencil")
        if (lateral_stencil.ncell != ncell or lateral_stencil.fingerprint !=
                execution["dependencies"]["lateral_stencil"]):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 lateral stencil differs from execution identity")
        cell_bed = apply_lateral_stencil_v65(
            channels.composed_bed_mass, lateral_stencil)
    else:
        if lateral_stencil is not None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 lateral stencil supplied while control is off")
        cell_bed = np.asarray(channels.composed_bed_mass)
    trace_bed, trace_suspended = _compose_trace_channels(batch, profile)
    raw = np.zeros((2, nfrac, nface), dtype=np.float64)
    corrected = np.zeros_like(raw)
    raw_donor = np.zeros((nfrac, nface), dtype=np.int64)
    donor = np.zeros((nfrac, nface), dtype=np.int64)
    donor_cell = np.full((nfrac, nface), -1, dtype=np.int64)
    slope_method = execution["bed_slope_method_id"]
    controls = execution["slope_controls"]
    if slope_method == SLOPE_OFF:
        if batch.slope is not None:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 slope-off face batch may not carry slope fields")
    elif batch.slope is None:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 active slope requires authenticated face fields")
    elif batch.slope.ust2_provenance_fingerprint != \
            execution["dependencies"]["backend_producer_execution"]:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 slope RP_USTAR-squared provenance differs from v64 producer")
    elif (slope_method == SLOPE_BAGNOLD_IKEDA and
          (batch.slope.stress_semantics != TAURAT_POSITIVE_ZERO or
           np.any(batch.slope.stress != 0.0))):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 positive-TraFrm ISlope 2 taurat receipt differs")
    elif (slope_method != SLOPE_BAGNOLD_IKEDA and
          batch.slope.stress_semantics !=
          "absolute_shields_from_total_current_RP_USTAR_squared"):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 ISlope 3/4 absolute-Shields receipt differs")
    for face in range(nface):
        owner = int(batch.owner[face])
        neighbour = int(batch.neighbour[face])
        role = int(batch.role[face])
        normal = batch.normal[:, face]
        for fraction in range(nfrac):
            if role == FACE_INTERNAL:
                left = cell_bed[:, fraction, owner]
                right = cell_bed[:, fraction, neighbour]
                if execution["upwind_bedload"]:
                    # Locked upwbed uses sutot/svtot sign fields, not hydraulic
                    # velocity.  Test signs on the un-smoothed total transport.
                    left_sign = float(np.dot(
                        channels.total_transport_direction[:, fraction, owner],
                        normal))
                    right_sign = float(np.dot(
                        channels.total_transport_direction[:, fraction, neighbour],
                        normal))
                    if left_sign > 0.0 and right_sign > 0.0:
                        candidate = left
                        raw_donor[fraction, face] = DONOR_OWNER
                    elif left_sign < 0.0 and right_sign < 0.0:
                        candidate = right
                        raw_donor[fraction, face] = DONOR_NEIGHBOUR
                    else:
                        candidate = 0.5*(left+right)
                        raw_donor[fraction, face] = (
                            DONOR_NEIGHBOUR if np.dot(candidate, normal) < 0.0
                            else DONOR_OWNER)
                else:
                    candidate = 0.5*(left+right)
                    raw_donor[fraction, face] = (
                        DONOR_NEIGHBOUR if np.dot(candidate, normal) < 0.0
                        else DONOR_OWNER)
            elif role in (FACE_TRACE_INLET, FACE_TRACE_OUTLET):
                candidate = trace_bed[:, fraction, face]
                raw_donor[fraction, face] = DONOR_TRACE
            elif role == FACE_OWNER_OUTLET:
                candidate = cell_bed[:, fraction, owner]
                raw_donor[fraction, face] = DONOR_OWNER
            elif role == FACE_CLOSED:
                candidate = np.zeros(2, dtype=np.float64)
                raw_donor[fraction, face] = DONOR_NONE
            else:  # protected by resolve_face_batch_v65
                raise ScalarFlowLiveV65Error("Q3-H5/v65 unknown face role")
            raw[:, fraction, face] = candidate
            if role != FACE_CLOSED and slope_method != SLOPE_OFF:
                fields = batch.slope
                assert fields is not None
                hiding_left = float(fields.hiding[fraction, owner])
                hiding_right = (float(fields.hiding[fraction, neighbour])
                                if role == FACE_INTERNAL else hiding_left)
                corrected[:, fraction, face] = _apply_flow_slope_v65(
                    candidate, fields.downhill[:, face],
                    float(fields.stress[fraction, face]),
                    float(fields.depth[fraction, face]),
                    float(fields.d50[fraction, face]),
                    float(fields.mean_diameter[fraction, face]),
                    hiding_left, hiding_right, slope_method, controls)
            else:
                corrected[:, fraction, face] = candidate
            normal_value = float(np.dot(corrected[:, fraction, face], normal))
            if role == FACE_INTERNAL:
                if normal_value < 0.0:
                    donor[fraction, face] = DONOR_NEIGHBOUR
                    donor_cell[fraction, face] = neighbour
                else:
                    donor[fraction, face] = DONOR_OWNER
                    donor_cell[fraction, face] = owner
            elif role in (FACE_TRACE_INLET, FACE_TRACE_OUTLET):
                donor[fraction, face] = DONOR_TRACE
                # Accepted boundary policy uses owner active fraction/fixfac.
                donor_cell[fraction, face] = owner
            elif role == FACE_OWNER_OUTLET:
                donor[fraction, face] = DONOR_OWNER
                donor_cell[fraction, face] = owner
    if np.any(trace_suspended != 0.0):
        # Diagnostic class-3 trace data is valid at composition level, but a
        # committable v65 bed transaction is guarded before this helper.
        pass
    return raw, corrected, raw_donor, donor, donor_cell


def _begin_bed_exner_trial_v65_impl(
    *, execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    state: ScalarFlowV65State,
    channels: ScalarFlowV65CellChannels,
    faces: ScalarFlowV65FaceBatch,
    cell_area_m2: object,
    dt_s: float,
    morphological_factor: float = 1.0,
    lateral_stencil: ScalarFlowV65LateralStencil | None = None,
    source_production_sha256: str | None = None,
) -> ScalarFlowV65Trial:
    """Shared implementation behind the live and component-only entries."""
    resolved = _runtime_execution(execution, cpu=True)
    profile = _runtime_channel_profile(resolved, channel_profile)
    _validate_state_for_execution(state, resolved)
    if state.accepted_step >= int(resolved["accepted_step_limit"]):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 bounded accepted-step limit has been reached")
    if any(row.transport_class == TRA_COMBINE for row in profile.fractions):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 TRA_COMBINE is component-only until Q4 conserves "
            "retained water-column mass and exchange")
    if any(row.transport_class != TRA_BEDLOAD for row in profile.fractions):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 live bed transaction requires TRA_BEDLOAD fractions")
    nfrac, ncell = state.bed_mass_kg_m2.shape
    _validate_cell_channels_v65(
        channels, execution=resolved, nfrac=nfrac, ncell=ncell)
    if not isinstance(faces, ScalarFlowV65FaceBatch):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 face batch is missing")
    # Dataclass construction and post-resolution array mutation must not bypass
    # the topology, trace, geometry, or derived-slope checks.
    faces = resolve_face_batch_v65(
        nfrac=nfrac, ncell=ncell, owner=faces.owner,
        neighbour=faces.neighbour, role=faces.role, normal=faces.normal,
        length_m=faces.length_m,
        trace_steered_bed_mass=faces.trace_steered_bed_mass,
        trace_raw_suspended_mass=faces.trace_raw_suspended_mass,
        trace_base_velocity=faces.trace_base_velocity,
        trace_depth_m=faces.trace_depth_m, slope=faces.slope)
    nface = faces.owner.size
    if faces.trace_steered_bed_mass.shape != (2, nfrac, nface):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 face batch fraction shape differs")
    if source_production_sha256 is not None and \
            resolved["bed_slope_method_id"] != SLOPE_OFF:
        if (faces.slope is None or
                faces.slope.source_production_sha256 !=
                source_production_sha256):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 live slope is not derived from this v64 production")
    area = _array(cell_area_m2, (ncell,), "cell area", copy=False)
    if np.any(area <= 0.0):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 cell areas must be positive")
    dt = _finite(dt_s, "trial dt", positive=True)
    morfac = _finite(
        morphological_factor, "morphological factor", nonnegative=True)
    raw, corrected, raw_donor, donor, donor_cell = _face_candidates_v65(
        execution=resolved, channels=channels, batch=faces, profile=profile,
        lateral_stencil=lateral_stencil)
    potential = np.zeros((nfrac, nface), dtype=np.float64)
    preliminary = np.zeros_like(potential)
    outgoing = np.zeros((nfrac, ncell), dtype=np.float64)
    for face in range(nface):
        role = int(faces.role[face])
        for fraction in range(nfrac):
            value = float(np.dot(corrected[:, fraction, face],
                                 faces.normal[:, face]))
            if role == FACE_TRACE_INLET:
                value = min(0.0, value)
            elif role in (FACE_TRACE_OUTLET, FACE_OWNER_OUTLET):
                value = max(0.0, value)
            elif role == FACE_CLOSED:
                value = 0.0
            potential[fraction, face] = value
            cell = int(donor_cell[fraction, face])
            if cell >= 0:
                preliminary[fraction, face] = (
                    value*state.active_fraction[fraction, cell])
                is_outgoing = (
                    (role == FACE_INTERNAL and preliminary[fraction, face] != 0.0) or
                    (role in (FACE_TRACE_OUTLET, FACE_OWNER_OUTLET) and
                     preliminary[fraction, face] > 0.0))
                if is_outgoing:
                    outgoing[fraction, cell] += (
                        abs(preliminary[fraction, face])*faces.length_m[face]*
                        dt*morfac/area[cell])
    fill = np.ones((nfrac, ncell), dtype=np.float64)
    fill[state.bed_mass_kg_m2 == 0.0] = 0.0
    demand = outgoing > 0.0
    fill[demand] = np.minimum(
        1.0, state.bed_mass_kg_m2[demand]/outgoing[demand])
    availability = state.active_fraction*fill
    flux = np.zeros((nfrac, nface), dtype=np.float64)
    delta = np.zeros((nfrac, ncell), dtype=np.float64)
    boundary = np.zeros(nfrac, dtype=np.float64)
    boundary_absolute = np.zeros(nfrac, dtype=np.float64)
    for face in range(nface):
        owner = int(faces.owner[face])
        neighbour = int(faces.neighbour[face])
        role = int(faces.role[face])
        scale = dt*morfac*faces.length_m[face]
        for fraction in range(nfrac):
            cell = int(donor_cell[fraction, face])
            if cell >= 0:
                flux[fraction, face] = (
                    potential[fraction, face]*availability[fraction, cell])
            transfer = scale*flux[fraction, face]
            delta[fraction, owner] -= transfer/area[owner]
            if role == FACE_INTERNAL:
                delta[fraction, neighbour] += transfer/area[neighbour]
            else:
                boundary[fraction] += transfer
                boundary_absolute[fraction] += abs(transfer)
    candidate_raw = np.asarray(state.bed_mass_kg_m2)+delta
    scale_mass = np.maximum(1.0, np.abs(state.bed_mass_kg_m2))
    tolerance = 2048.0*np.finfo(np.float64).eps*scale_mass
    if np.any(candidate_raw < -tolerance):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 Exner candidate exceeds finite class inventory")
    clamped_mass = np.maximum(candidate_raw, 0.0)
    # An inventory-limited export is analytically exact depletion.  Floating
    # multiply/divide roundoff can otherwise leave a bounded tolerance-scale
    # residue, which
    # would spuriously replace the retained surface identity by a pure class.
    numerically_depleted = (
        np.sum(clamped_mass, axis=0) <= np.sum(tolerance, axis=0))
    candidate_mass = np.array(clamped_mass, copy=True, order="C")
    candidate_mass[:, numerically_depleted] = 0.0
    depletion_roundoff = np.sum(
        (clamped_mass-candidate_mass)*area[None, :], axis=1)
    # ``delta`` is the accepted state change, not merely the raw face result.
    # The positive roundoff transfer explicitly closes the ledger for the
    # roundoff-scale mass removed by the all-class depletion snap.
    if np.any(numerically_depleted):
        delta = np.array(delta, copy=True, order="C")
        delta[:, numerically_depleted] = (
            candidate_mass[:, numerically_depleted]-
            np.asarray(state.bed_mass_kg_m2)[:, numerically_depleted])
    volume = candidate_mass/np.asarray(state.bulk_density_kg_m3)[:, None]
    total_volume = np.sum(volume, axis=0)
    positive_stock = total_volume > 0.0
    candidate_fraction = np.array(state.active_fraction, copy=True, order="C")
    candidate_fraction[:, positive_stock] = (
        volume[:, positive_stock]/total_volume[positive_stock][None, :])
    residual = (
        np.sum(delta*area[None, :], axis=1)+boundary+depletion_roundoff)
    residual_tolerance = 4096.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.sum(np.abs(delta*area[None, :]), axis=1)+np.abs(boundary))
    if np.any(np.abs(residual) > residual_tolerance):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 conservative Exner ledger does not close")
    candidate_state = resolve_state_v65(
        execution=resolved,
        bed_mass_kg_m2=candidate_mass,
        active_fraction=candidate_fraction,
        boundary_signed_kg=(np.asarray(state.boundary_signed_kg)+boundary+
                            depletion_roundoff),
        boundary_absolute_kg=(np.asarray(state.boundary_absolute_kg)+
                              boundary_absolute+depletion_roundoff),
        accepted_step=state.accepted_step+1,
        time_s=state.time_s+dt,
    )
    arrays = []
    for value in (raw, corrected, raw_donor, donor,
                  state.active_fraction, fill, availability, flux, delta,
                  boundary, depletion_roundoff, residual):
        item = np.array(value, copy=True, order="C")
        item.setflags(write=False)
        arrays.append(item)
    trial_body = {
        "execution_fingerprint": resolved["fingerprint"],
        "parent_state_sha256": state.state_sha256,
        "candidate_state_sha256": candidate_state.state_sha256,
        "dt_s": dt,
        "morphological_factor": morfac,
        "arrays_sha256": _array_digest(*arrays),
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
    }
    return ScalarFlowV65Trial(
        resolved["fingerprint"], state.state_sha256, dt, morfac,
        *arrays[:9], candidate_state, arrays[9], arrays[10], arrays[11], 1, 1,
        _fingerprint(trial_body))


def begin_component_bed_exner_trial_v65(
    *, execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    state: ScalarFlowV65State, channels: ScalarFlowV65CellChannels,
    faces: ScalarFlowV65FaceBatch, cell_area_m2: object, dt_s: float,
    morphological_factor: float = 1.0,
    lateral_stencil: ScalarFlowV65LateralStencil | None = None,
) -> ScalarFlowV65Trial:
    """Propose from synthetic/raw channels for bounded operator evidence only."""
    return _begin_bed_exner_trial_v65_impl(
        execution=execution, channel_profile=channel_profile, state=state,
        channels=channels, faces=faces, cell_area_m2=cell_area_m2, dt_s=dt_s,
        morphological_factor=morphological_factor,
        lateral_stencil=lateral_stencil, source_production_sha256=None)


def begin_bed_exner_trial_v65(
    *, execution: Mapping[str, object],
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    state: ScalarFlowV65State, production: ScalarFlowV65Production,
    faces: ScalarFlowV65FaceBatch, cell_area_m2: object, dt_s: float,
    morphological_factor: float = 1.0,
    lateral_stencil: ScalarFlowV65LateralStencil | None = None,
) -> ScalarFlowV65Trial:
    """Build one live CPU candidate from an authenticated v64 production."""
    resolved = _runtime_execution(execution, cpu=True)
    _validate_state_for_execution(state, resolved)
    nfrac, ncell = state.bed_mass_kg_m2.shape
    _validate_production_v65(
        production, execution=resolved, nfrac=nfrac, ncell=ncell)
    return _begin_bed_exner_trial_v65_impl(
        execution=resolved, channel_profile=channel_profile, state=state,
        channels=production.channels, faces=faces,
        cell_area_m2=cell_area_m2, dt_s=dt_s,
        morphological_factor=morphological_factor,
        lateral_stencil=lateral_stencil,
        source_production_sha256=production.production_sha256)


def _validate_trial_parent(
    state: ScalarFlowV65State, trial: ScalarFlowV65Trial,
) -> None:
    if not isinstance(trial, ScalarFlowV65Trial):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 trial is missing")
    if (not isinstance(state, ScalarFlowV65State) or
            state.state_sha256 != trial.parent_state_sha256 or
            state.execution_fingerprint != trial.execution_fingerprint):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 trial parent state differs")
    arrays = (
        trial.face_raw_candidate, trial.face_corrected_candidate,
        trial.raw_donor, trial.donor, trial.active_fraction_factor,
        trial.inventory_fill, trial.availability, trial.face_flux_kg_m_s,
        trial.cell_delta_kg_m2, trial.boundary_transfer_kg,
        trial.depletion_roundoff_transfer_kg,
        trial.conservation_residual_kg,
    )
    body = {
        "execution_fingerprint": trial.execution_fingerprint,
        "parent_state_sha256": trial.parent_state_sha256,
        "candidate_state_sha256": trial.candidate.state_sha256,
        "dt_s": trial.dt_s,
        "morphological_factor": trial.morphological_factor,
        "arrays_sha256": _array_digest(*arrays),
        "active_fraction_application_count": (
            trial.active_fraction_application_count),
        "inventory_fill_application_count": (
            trial.inventory_fill_application_count),
    }
    if (_fingerprint(body) != trial.trial_sha256 or
            trial.active_fraction_application_count != 1 or
            trial.inventory_fill_application_count != 1):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 trial identity or once-only counters differ")


def commit_trial_v65(
    state: ScalarFlowV65State, trial: ScalarFlowV65Trial,
) -> ScalarFlowV65State:
    _validate_trial_parent(state, trial)
    return trial.candidate


def rollback_trial_v65(
    state: ScalarFlowV65State, trial: ScalarFlowV65Trial,
) -> ScalarFlowV65State:
    _validate_trial_parent(state, trial)
    return state


def cancel_trial_v65(
    state: ScalarFlowV65State, trial: ScalarFlowV65Trial,
) -> ScalarFlowV65State:
    _validate_trial_parent(state, trial)
    return state


def state_checkpoint_document_v65(
    state: ScalarFlowV65State,
) -> dict[str, object]:
    if not isinstance(state, ScalarFlowV65State):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 checkpoint state is missing")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "checkpoint",
        "execution_fingerprint": state.execution_fingerprint,
        "accepted_step": state.accepted_step,
        "time_s": state.time_s,
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "bulk_density_kg_m3": state.bulk_density_kg_m3.tolist(),
        "active_fraction": state.active_fraction.tolist(),
        "boundary_signed_kg": state.boundary_signed_kg.tolist(),
        "boundary_absolute_kg": state.boundary_absolute_kg.tolist(),
        "state_sha256": state.state_sha256,
        "transaction_schema": (
            "complete_accepted_state_no_trial_arrays_exact_profile_restart"),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_state_checkpoint_v65(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> ScalarFlowV65State:
    resolved = _runtime_execution(execution)
    if not isinstance(value, Mapping):
        raise ScalarFlowLiveV65Error("Q3-H5/v65 checkpoint must be an object")
    expected_keys = {
        "schema", "scope_id", "semantic_id", "envelope_version", "stage",
        "execution_fingerprint", "accepted_step", "time_s",
        "bed_mass_kg_m2", "bulk_density_kg_m3", "active_fraction",
        "boundary_signed_kg", "boundary_absolute_kg", "state_sha256",
        "transaction_schema", "fingerprint",
    }
    if set(value) != expected_keys:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 checkpoint inventory differs")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or value["scope_id"] != SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["envelope_version"] != ENVELOPE_VERSION or
            value["stage"] != "checkpoint" or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["bulk_density_kg_m3"] != resolved["bulk_density_kg_m3"]):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 checkpoint header/profile differs")
    state = resolve_state_v65(
        execution=resolved,
        bed_mass_kg_m2=value["bed_mass_kg_m2"],
        active_fraction=value["active_fraction"],
        boundary_signed_kg=value["boundary_signed_kg"],
        boundary_absolute_kg=value["boundary_absolute_kg"],
        accepted_step=value["accepted_step"], time_s=value["time_s"],
    )
    if state.state_sha256 != value["state_sha256"]:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 checkpoint state digest differs")
    return state


__all__ = [
    "DEFAULT_SLOPE_CONTROLS", "DONOR_NEIGHBOUR", "DONOR_NONE",
    "DONOR_OWNER", "DONOR_TRACE", "ENVELOPE_VERSION", "FACE_CLOSED",
    "FACE_INTERNAL", "FACE_OWNER_OUTLET", "FACE_ROLE_POLICY",
    "FACE_TRACE_INLET", "FACE_TRACE_OUTLET", "LIVE_SLOPE_METHODS",
    "LOCKED_UPWBED_PATH", "LOCKED_UPWBED_SHA256", "MAX_ACCEPTED_STEPS",
    "ORDERED_STAGES", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "SLOPE_BAGNOLD_IKEDA", "SLOPE_KOCH_FLOKSTRA",
    "SLOPE_OFF", "SLOPE_PARKER_ANDREWS", "SUPPORTED_SELECTORS",
    "UNSUPPORTED_GATES", "ScalarFlowLiveV65Error", "ScalarFlowV65CellChannels",
    "ScalarFlowV65FaceBatch", "ScalarFlowV65LateralStencil",
    "ScalarFlowV65Production", "ScalarFlowV65SlopeFields", "ScalarFlowV65State",
    "ScalarFlowV65Trial",
    "apply_lateral_stencil_v65", "begin_bed_exner_trial_v65",
    "begin_component_bed_exner_trial_v65", "cancel_trial_v65",
    "commit_trial_v65", "compose_cell_channels_v65",
    "compose_cell_production_v65",
    "lateral_stencil_document_v65", "load_scalar_flow_live_execution_v65",
    "load_state_checkpoint_v65", "resolve_face_batch_v65",
    "resolve_lateral_stencil_v65", "resolve_slope_fields_from_production_v65",
    "resolve_slope_fields_v65",
    "resolve_state_v65", "rollback_trial_v65",
    "scalar_flow_live_execution_document_v65",
    "scalar_flow_live_semantic_document_v65",
    "selector7_settling_reference_document_v65",
    "state_checkpoint_document_v65",
]
