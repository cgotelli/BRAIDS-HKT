"""Production controller for the explicitly selected GPU f-wave midpoint.

The accepted hydraulic update is owned by the existing augmented-f-wave face
pipeline evaluated at a converged implicit midpoint.  A frozen acoustic
operator is used only as a simplified-Newton Jacobian.  Open-boundary traces
remain authoritative in the nonlinear residual and water ledger; their trace
derivative is deliberately omitted from the approximate Jacobian.

This module never selects a backend and never falls back to an explicit method.
Its initial production envelope is fixed-grid, safely wet, low-Froude flow.
Bedload is advanced once from the hydraulic midpoint by the resident
sediment/Exner operator, making that route explicitly weakly coupled.  The
version-2 component composition adds private secondary-flow source/advection
and momentum diffusion without making sediment part of the Newton solve.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from braids_core.gpu_capabilities import gpu_integrator_label
from braids_core.sediment_method_catalog import admitted_legacy_codes
from .law11_soursin_exchange import (
    EXCHANGE_MODE as FLOW2DH_EXCHANGE_MODE,
    method_identity as flow2dh_method_identity,
    validate_scope as validate_flow2dh_scope,
)
from .flow2dh_boundary_diffusion import (
    method_identity as flow2dh_boundary_diffusion_identity,
)
from .forcing_profile import BINDING_FIELD, forcing_identity


HERE = Path(__file__).resolve().parent
TOKEN = "implicit_fwave_midpoint"
LABEL = gpu_integrator_label(TOKEN)
METHOD_SCHEMA = 1
CHECKPOINT_SCHEMA = 2
CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")
KERNEL_NAMES = (
    "implicit_linear_acoustic_faces", "implicit_linear_acoustic_finish",
    "implicit_full_rate_from_faces", "implicit_water_divergence_faces",
    "implicit_depth_from_flux", "implicit_advective_cfl_faces",
    "implicit_bicg_direction", "implicit_bicg_alpha",
    "implicit_bicg_residual_stage", "implicit_bicg_omega",
    "implicit_bicg_update", "implicit_bicg_convergence",
)
ADMITTED_BEDLOAD_LAWS = admitted_legacy_codes("capacity", "implicit_gpu")
ADMITTED_BED_SLOPE_MODES = admitted_legacy_codes("bed_slope", "implicit_gpu")
ADMITTED_SECONDARY_MODELS = admitted_legacy_codes("secondary_flow", "implicit_gpu")
Q3H2_COUPLING = (
    "weak_midpoint_q3h2_classic_flow_equilibrium_secondary_composition_v1")
Q3H3_COUPLING = (
    "weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1")


class ImplicitStepCancelled(RuntimeError):
    """Internal cooperative-cancellation signal; never a numerical fallback."""


class ImplicitStepRejected(RuntimeError):
    """A retryable implicit trial failure with no accepted-state mutation."""


def checkpoint_schema(data: dict) -> int:
    """Version restart state only when resident boundary memory is present."""
    coupling = coupling_identity(data)
    if coupling == Q3H3_COUPLING:
        return 8
    if coupling == Q3H2_COUPLING:
        return 7
    if coupling == "weak_midpoint_q3h1_hiding_capacity_composition_v1":
        return 6
    if coupling == "weak_midpoint_q3c_scalar_capacity_composition_v1":
        return 5
    if coupling in (
            "weak_midpoint_m1_general_channel_composition_v2",
            "weak_midpoint_m2_law11_flow2dh_channel_composition_v2"):
        return 4
    return (3 if coupling == "weak_midpoint_m2_law11_flow2dh_v1"
            else CHECKPOINT_SCHEMA)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def coupling_identity(data: dict) -> str:
    """Keep the original admitted methods distinct from component composition."""
    if int(data["bedlaw"]) == 0:
        return "hydraulic_only_v1"
    m1_schema = int(data.get("m1_profile_runtime_schema", 1))
    marker = data.get("m1_profile_public_admission")
    if ("prognostic_secondary_execution_identity" in data and
            isinstance(marker, dict) and
            marker.get("admission_id") in (
                "q3c.scalar_capacity_total_v1",
                "q3h1.scalar_capacity_hiding_v1") and
            int(data["bedlaw"]) in (2, 8, 11)):
        return Q3H3_COUPLING
    if ("secondary_foundation_execution_identity" in data and
            isinstance(marker, dict) and
            marker.get("admission_id") in (
                "q3c.scalar_capacity_total_v1",
                "q3h1.scalar_capacity_hiding_v1") and
            int(data["bedlaw"]) in (2, 8, 11)):
        return Q3H2_COUPLING
    if (isinstance(marker, dict) and
            marker.get("admission_id") ==
                "q3h1.scalar_capacity_hiding_v1" and
            int(data["bedlaw"]) in (2, 11)):
        return "weak_midpoint_q3h1_hiding_capacity_composition_v1"
    if (isinstance(marker, dict) and
            marker.get("admission_id") == "q3c.scalar_capacity_total_v1" and
            int(data["bedlaw"]) in (2, 8, 11)):
        return "weak_midpoint_q3c_scalar_capacity_composition_v1"
    if (int(data["bedlaw"]) == 11 and
            int(data["exchange_mode"]) == FLOW2DH_EXCHANGE_MODE):
        return ("weak_midpoint_m2_law11_flow2dh_channel_composition_v2"
                if m1_schema == 2 else
                "weak_midpoint_m2_law11_flow2dh_v1")
    if int(data["bedlaw"]) == 11:
        return ("weak_midpoint_m1_general_channel_composition_v2"
                if m1_schema == 2 else "weak_midpoint_m1_general_v1")
    if int(data["bedlaw"]) in (5, 6, 7):
        return "weak_midpoint_m3_rp_taub_v1"
    if (int(data["bedlaw"]) == 2 and int(data["bed_slope"]) == 0 and
            int(data["secondary_flow_model"]) == 0 and
            float(data.get("horizontal_viscosity", 0.0)) == 0.0):
        return "weak_midpoint_bedload_v1"
    return "weak_midpoint_delft_components_v2"


def validate_scope(data: dict) -> None:
    """Admit explicit fixed-grid Delft3D component composition, not fallback."""
    q = np.asarray(data["q"])
    nc = int(data["nc"])
    dry = float(data["dry"])
    _require(q.shape == (3, nc) and q.dtype == np.float64 and
             np.isfinite(q).all(), "Implicit f-wave state must be finite FP64")
    _require(np.all(q[0] > 20.0 * dry),
             "Implicit f-wave requires a safely fully wet initial state")
    active = np.asarray(data["active"])
    _require(active.shape == (nc,) and np.all(active == 1),
             "Implicit f-wave excludes inactive/cut-out cells")
    _require(not bool(np.any(data.get("amr_enabled", 0))),
             "Implicit f-wave AMR is not yet qualified")
    exchange_mode = int(data["exchange_mode"])
    suspended_requested = bool(
        np.any(data.get("suspended_transport_active", False)) or
        np.any(data.get("suspended_mass", 0)) or
        np.any(data.get("fraction_concentration", 0)) or
        np.any(data.get("fraction_series_concentration", 0)))
    if exchange_mode == 0:
        _require(not suspended_requested,
                 "Implicit f-wave passive suspension is not yet admitted")
    else:
        _require(exchange_mode == FLOW2DH_EXCHANGE_MODE and
                 int(data["bedlaw"]) == 11,
                 "Implicit f-wave suspended transport requires law 11 exchange mode 4")
        validate_flow2dh_scope(data)
    secondary = int(data["secondary_flow_model"])
    _require(secondary in ADMITTED_SECONDARY_MODELS,
             "Implicit f-wave requires Delft3D OFF/equilibrium/prognostic secondary flow")
    mass = np.asarray(data.get("secondary_mass", np.zeros(nc)))
    _require(mass.shape == (nc,) and np.isfinite(mass).all() and
             (secondary == 2 or not bool(np.any(mass))),
             "Implicit f-wave secondary inventory differs from its selected model")
    for key in ("bank_failure", "bank_front_tracking", "bank_toe_erosion_mode",
                "bank_avalanche", "bank_erosion_fraction", "general_avalanche",
                "bedform_roughness", "bedload_adaptation",
                "morphological_acceleration", "nmorfac"):
        _require(not bool(np.any(data.get(key, 0))),
                 "Implicit f-wave unsupported control: " + key)
    viscosity = float(data.get("horizontal_viscosity", 0.0))
    _require(np.isfinite(viscosity) and viscosity >= 0.0,
             "Implicit f-wave viscosity must be finite and nonnegative")
    if viscosity > 0.0:
        _require(int(data.get("horizontal_viscosity_mode", 0)) == 1 and
                 int(data.get("horizontal_viscosity_geometry", 0)) == 1,
                 "Implicit f-wave viscosity requires the resident velocity-Laplacian route")
    _require(float(data.get("morfac", 1.0)) == 1.0,
             "Implicit f-wave weak coupling currently requires MorFac 1")
    _require(not bool(np.any(data.get("nhydro_series", 0))) and
             not bool(np.any(data.get("nfraction_series", 0))) and
             not bool(np.any(data.get("nseries", 0))),
             "Implicit f-wave currently requires constant boundary forcing")
    bedlaw = int(data["bedlaw"])
    public_marker = data.get("m1_profile_public_admission")
    scalar_foundation = (
        isinstance(public_marker, dict) and
        public_marker.get("admission_id") in (
            "q3c.scalar_capacity_total_v1",
            "q3h1.scalar_capacity_hiding_v1") and
        bedlaw in (2, 8, 11)
    )
    q3h_hiding = (
        isinstance(public_marker, dict) and
        public_marker.get("admission_id") ==
            "q3h1.scalar_capacity_hiding_v1")
    q3h2_secondary = False
    q3h4_slope = None
    if scalar_foundation:
        # The corrected Delft3D capacity foundations have a distinct,
        # authenticated prepared-data contract.  They must never enter the
        # historical integer-law 2/8 routes merely because the numeric code is
        # the same.
        from .m1_profile_transport import (
            validate_bed_slope_composition,
            validate_prognostic_secondary_composition,
            validate_scalar_capacity_prepared,
            validate_secondary_composition,
        )
        validate_scalar_capacity_prepared(data)
        q3h2_secondary = validate_secondary_composition(data) is not None
        q3h3_secondary = (
            validate_prognostic_secondary_composition(data) is not None)
        q3h4_slope = validate_bed_slope_composition(data)
    elif bedlaw == 11:
        # Research admission is authenticated independently of the public
        # catalogue.  Keep this check host-only and before RawModule creation.
        from .m1_profile_transport import validate_law11_prepared
        validate_law11_prepared(data)
        q3h3_secondary = False
    elif bedlaw in (5, 6, 7):
        # This is an authenticated fixed-grid bridge, not a catalogue alias
        # and not MPM fallback.  Its validator proves that the selected
        # RP_TAUB consumer is live and that open sediment boundaries are off.
        from .m1_rp_taub_transport import validate_rp_taub_prepared
        validate_rp_taub_prepared(data)
        q3h3_secondary = False
    else:
        q3h3_secondary = False
        _require(bedlaw in ADMITTED_BEDLOAD_LAWS,
                  "Implicit f-wave supports static bed, MPM2, Engelund-Hansen8, or the authenticated law-11 prototype")
    if bedlaw in (2, 5, 6, 7, 8, 11):
        _require(1 <= int(data["nfrac"]) <= 32 and
                  int(data["bed_slope"]) in (
                      (q3h4_slope["persistence_code"],)
                      if q3h4_slope is not None else
                      ADMITTED_BED_SLOPE_MODES) and
                  int(data["hiding_exposure"]) in (
                      (2, 3, 4, 5) if q3h_hiding else (0,)) and
                  int(data["nbed_layers"]) >=
                      (1 if scalar_foundation or bedlaw == 11 else 2),
                  "Implicit f-wave bedload has an unsupported class/slope/hiding layout")
        if q3h_hiding:
            _require(bedlaw in (2, 11) and int(data["nfrac"]) == 2 and
                     int(data["bed_slope"]) == (
                         0 if q3h4_slope is None else
                         q3h4_slope["persistence_code"]) and
                     secondary == (2 if q3h3_secondary else
                                   1 if q3h2_secondary else 0) and
                     exchange_mode == 0,
                     "Q3-H1 requires two fractions, law 2/11, and its "
                     "authenticated slope/secondary/exchange scope")
        if q3h2_secondary:
            _require(secondary == 1 and
                     int(data["bed_slope"]) == (
                         0 if q3h4_slope is None else
                         q3h4_slope["persistence_code"]) and
                     exchange_mode == 0 and viscosity == 0.0,
                     "Q3-H2 requires equilibrium secondary, authenticated "
                     "optional H4 slope, and excludes exchange/viscosity")
        if q3h3_secondary:
            _require(secondary == 2 and
                     int(data["bed_slope"]) == (
                         0 if q3h4_slope is None else
                         q3h4_slope["persistence_code"]) and
                     exchange_mode == 0 and viscosity == 0.0,
                     "Q3-H3 requires prognostic secondary, authenticated "
                     "optional H4 slope, and excludes exchange/viscosity")
    else:
        _require(secondary == 0 and int(data["bed_slope"]) == 0 and viscosity == 0.0,
                 "Implicit static-bed identity excludes sediment component composition")
    for key in ("bed", "area", "length", "normal", "owner", "neighbour",
                "distance", "manning_cell"):
        value = np.asarray(data[key])
        _require(np.isfinite(value).all(),
                 "Nonfinite implicit f-wave input: " + key)
    _require(np.all(np.asarray(data["area"]) > 0.0) and
             np.all(np.asarray(data["length"]) > 0.0) and
             np.all(np.asarray(data["manning_cell"]) >= 0.0),
             "Implicit f-wave geometry/friction is invalid")
    if BINDING_FIELD in data:
        forcing_identity(data)
    internal = np.asarray(data["neighbour"]) >= 0
    distance = np.asarray(data["distance"])[0] + np.asarray(data["distance"])[1]
    # A single rectangular column is a valid degenerate operator test: there
    # is no internal acoustic coupling to precondition. Do not require a
    # fictitious neighbour merely to exercise an accepted coarsened parent.
    # Multi-cell domains retain the original positive-internal-distance guard.
    one_column = (int(data['nc']) == 1 and int(data['nf']) == 4 and
        np.all(np.asarray(data['owner']) == 0) and not np.any(internal) and
        set(map(tuple, np.asarray(data['normal']).T)) ==
            {(-1., 0.), (1., 0.), (0., -1.), (0., 1.)})
    _require((np.any(internal) or one_column) and np.all(distance[internal] > 0.0),
             "Implicit f-wave needs positive internal face distances")


def method_identity(*, coupling: str, data=None, nonlinear_rtol=5e-2,
                    nonlinear_atol=1e-14, nonlinear_maxiter=5,
                    inner_tolerance_fraction=.2, linear_maxiter=400,
                    advective_cfl_limit=.8, froude_limit=.65) -> dict:
    result = dict(
        implicit_midpoint_schema=METHOD_SCHEMA,
        nonlinear_rtol=float(nonlinear_rtol),
        nonlinear_atol=float(nonlinear_atol),
        nonlinear_maxiter=int(nonlinear_maxiter),
        inner_tolerance_fraction=float(inner_tolerance_fraction),
        linear_maxiter=int(linear_maxiter),
        advective_cfl_limit=float(advective_cfl_limit),
        froude_limit=float(froude_limit),
        coupling=coupling,
        full_residual="existing_augmented_fwave_faces_v1",
        approximate_jacobian="frozen_acoustic_upwind_v1",
        open_boundary_jacobian="frozen_trace_full_residual_v1",
        timestep_controller="advective_froude_envelope_v1",
        automatic_explicit_fallback=False,
        automatic_CPU_fallback=False,
    )
    if coupling in (
            "weak_midpoint_m1_general_channel_composition_v2",
            "weak_midpoint_m2_law11_flow2dh_channel_composition_v2"):
        # CPU-v51 uses midpoint sampled mode-9 face wetness and a uniform
        # section speed.  This identity is part of restart/coupling state;
        # legacy integrated subface closure remains a separate policy.
        from .integrated_normal_trace import NATIVE_MODE9_FACE_POLICY
        result["normal_face_policy"] = NATIVE_MODE9_FACE_POLICY
    if data is not None and BINDING_FIELD in data:
        forcing = forcing_identity(data)
        result.update(
            hydraulic_friction_split=(
                "symmetric_half_step_legacy_manning_v1" if
                forcing["hydraulic_roughness_method_id"] ==
                "roughness.manning" else
                "symmetric_half_step_typed_resident_v1"),
            **forcing,
        )
    else:
        result["manning_split"] = "symmetric_half_step_resident_v1"
    if coupling in (
            "weak_midpoint_q3c_scalar_capacity_composition_v1",
            "weak_midpoint_q3h1_hiding_capacity_composition_v1",
            Q3H2_COUPLING, Q3H3_COUPLING):
        _require(data is not None,
                 "Q3-C scalar-capacity identity requires prepared data")
        from .m1_profile_transport import profile_identity
        hiding = (data is not None and isinstance(
            data.get("m1_profile_public_admission"), dict) and
            data["m1_profile_public_admission"].get("admission_id") ==
                "q3h1.scalar_capacity_hiding_v1")
        classic_secondary = coupling == Q3H2_COUPLING
        prognostic_secondary = coupling == Q3H3_COUPLING
        result.update(
            component_coupling_schema=(11 if prognostic_secondary else
                                       10 if classic_secondary else
                                       9 if hiding else 8),
            capacity_dispatch="resident_q3c_scalar_capacity_laws_2_8_11_v1",
            capacity_raw_channels="resident_bed_and_suspended_diagnostics_v1",
            capacity_composition="resident_delft_channel_composer_before_faces_v1",
            bed_capacity_channel="resident_reconstructed_face_exner_v1",
            suspended_capacity_channel="composed_once_into_total_bed_route_v1",
            exner_transaction="whole_candidate_accept_or_rollback_v1",
        )
        if hiding:
            result.update(
                hiding_dispatch="resident_q3h1_hiding_exposure_modes_2_3_4_5_v1",
                hiding_stage_owner=(
                    "transactional_trial_surface_once_per_stage_commit_on_accept_v1"),
                hiding_boundary_owner="owner_indexed_factors_for_trace_capacity_v1",
            )
        if classic_secondary:
            result.update(
                secondary_dispatch=(
                    "resident_q3h2_classic_flow_equilibrium_secondary_v1"),
                secondary_stage_owner=(
                    "every_transactional_trial_stage_before_hiding_capacity_v1"),
                secondary_cell_rotation=(
                    "raw_bed_after_capacity_hiding_before_composer_v1"),
                secondary_boundary_rotation=(
                    "roles12_trace_owner_equilibrium_role3_rotated_owner_v1"),
                secondary_suspended_rotation=False,
                secondary_legacy_rotation_disabled=True,
            )
        if prognostic_secondary:
            result.update(
                secondary_dispatch=(
                    "resident_q3h3_classic_flow_prognostic_secondary_v1"),
                secondary_state_semantics=(
                    data["prognostic_secondary_execution_identity"]
                    ["state_semantics"]),
                secondary_integrated_ledger_units=(
                    data["prognostic_secondary_execution_identity"]
                    ["integrated_ledger_units"]),
                secondary_transaction=(
                    "accepted_qold_then_post_hydraulic_qnew_then_fct_"
                    "advection_then_full_classic_BE_reaction_v1"),
                secondary_depth_pair=(
                    "accepted_old_depth_post_hydraulic_new_depth_v1"),
                secondary_source_forcing=(
                    "new_velocity_raw_curvature_typed_chezy_v1"),
                secondary_boundary=(
                    "adjacent_interior_classic_equilibrium_all_open_"
                    "flow_directions_v1"),
                secondary_stage_owner=(
                    "one_transaction_after_hydraulics_before_capacity_v1"),
                secondary_cell_rotation=(
                    "final_transported_hI_raw_bed_after_capacity_hiding_"
                    "before_composer_once_v1"),
                secondary_cell_rotation_forcing=(
                    "weak_midpoint_depth_velocity_and_capacity_typed_chezy_v1"),
                secondary_boundary_rotation=(
                    data["prognostic_secondary_execution_identity"]
                    ["bed_boundary_rotation_policy"]),
                secondary_source_split="single_full_step_BE_after_advection_v1",
                secondary_generic_model2_prepost_source_disabled=True,
                secondary_generic_model2_filter_disabled=True,
                secondary_suspended_rotation=False,
                secondary_legacy_rotation_disabled=True,
            )
        result.update(profile_identity(data))
    elif coupling == "weak_midpoint_delft_components_v2":
        result.update(component_coupling_schema=2,
            secondary_source_split="symmetric_private_half_steps_v2",
            secondary_advection="converged_midpoint_water_faces_v2",
            momentum_diffusion="resident_post_source_velocity_laplacian_v2")
    elif coupling in ("weak_midpoint_m1_general_v1",
                      "weak_midpoint_m1_general_channel_composition_v2"):
        composed = coupling.endswith("channel_composition_v2")
        result.update(
            component_coupling_schema=6 if composed else 3,
            capacity_dispatch=("resident_m1_profile_law11_v2" if composed
                               else "resident_m1_profile_law11_v1"),
            bed_capacity_channel="resident_face_reconstruction_exner_v1",
            suspended_capacity_channel="resident_diagnostic_only_m2_pending",
            suspended_capacity_discarded=False,
        )
        if composed:
            _require(data is not None,
                     "Channel-composed law-11 identity requires prepared data")
            from .m1_profile_transport import profile_identity
            result.update(profile_identity(data))
    elif coupling == "weak_midpoint_m3_rp_taub_v1":
        _require(data is not None,
                 "RP_TAUB implicit identity requires prepared forcing")
        from .m1_rp_taub_transport import method_identity as rp_taub_identity
        result.update(
            component_coupling_schema=5,
            capacity_dispatch="resident_m1_rp_taub_laws567_v1",
            bed_capacity_channel="resident_face_reconstruction_exner_v1",
            suspended_capacity_channel="disabled",
        )
        result.update(rp_taub_identity(data))
    elif coupling in ("weak_midpoint_m2_law11_flow2dh_v1",
                      "weak_midpoint_m2_law11_flow2dh_channel_composition_v2"):
        _require(data is not None,
                 "FLOW 2DH implicit identity requires the prepared suspension profile")
        composed = coupling.endswith("channel_composition_v2")
        result.update(
            component_coupling_schema=7 if composed else 4,
            capacity_dispatch=("resident_m1_profile_law11_v2" if composed
                               else "resident_m1_profile_law11_v1"),
            bed_capacity_channel="resident_face_reconstruction_exner_v1",
            suspended_capacity_channel=(
                "resident_signed_composed_endpoint_flow2dh_exchange_v2"
                if composed else "resident_endpoint_flow2dh_exchange_v1"),
            suspended_advection="accepted_midpoint_water_flux_bounded_fct_v1",
            suspended_source_order="advection_then_endpoint_capacity_then_exchange_v1",
            capacity_endpoint=(
                "accepted_post_typed_friction_hydraulics_and_trial_bed_v1"
                if data is not None and BINDING_FIELD in data else
                "accepted_post_manning_hydraulics_and_trial_bed_v1"),
            exchange_depth_pair="accepted_pre_step_h0_and_endpoint_h1_v1",
            suspended_boundary_ledger="signed_positive_into_domain_kg",
            suspended_exchange_ledger="signed_suspended_gain_and_bed_gain_kg",
            wet_dry_transport_policy="existing_bounded_fct_separate_from_SedThr_exchange",
        )
        result.update(flow2dh_method_identity(data))
        result.update(flow2dh_boundary_diffusion_identity(data))
        if composed:
            from .m1_profile_transport import profile_identity
            result.update(profile_identity(data))
    return result


class ImplicitMidpointController:
    """Transactional hydraulic midpoint composed with one resident bedload leg."""

    def __init__(self, solver, *, source_path=None, nonlinear_rtol=5e-2,
                 nonlinear_atol=1e-14, nonlinear_maxiter=5,
                 linear_rtol=5e-2, linear_atol=1e-10, linear_maxiter=400,
                 linear_check_interval=1, inner_tolerance_fraction=.2,
                 advective_cfl_limit=.8, froude_limit=.65):
        import cupy as cp

        validate_scope(solver.data)
        controls = np.array((nonlinear_rtol, nonlinear_atol, linear_rtol,
                             linear_atol, inner_tolerance_fraction,
                             advective_cfl_limit, froude_limit), dtype=float)
        _require(np.isfinite(controls).all() and
                 0.0 < nonlinear_rtol < 1.0 and
                 0.0 < nonlinear_atol < 1.0 and
                 0.0 < linear_rtol < 1.0 and 0.0 < linear_atol < 1.0 and
                 0.0 < inner_tolerance_fraction <= .25 and
                 0.0 < advective_cfl_limit <= 1.0 and
                 0.0 < froude_limit < 1.0 and
                 isinstance(nonlinear_maxiter, int) and nonlinear_maxiter > 0 and
                 isinstance(linear_maxiter, int) and linear_maxiter > 0 and
                 isinstance(linear_check_interval, int) and linear_check_interval > 0,
                 "Invalid implicit f-wave solver controls")
        self.cp, self.solver = cp, solver
        self.nc, self.nf, self.nb = solver.nc, solver.nf, solver.nb
        self.grav = np.float64(solver.data["grav"])
        self.dry = np.float64(solver.data["dry"])
        self.wet_floor = np.float64(20.0 * float(self.dry))
        self.nonlinear_rtol = float(nonlinear_rtol)
        self.nonlinear_atol = float(nonlinear_atol)
        self.nonlinear_maxiter = int(nonlinear_maxiter)
        self.linear_rtol = float(linear_rtol)
        self.linear_atol = float(linear_atol)
        self.linear_maxiter = int(linear_maxiter)
        self.linear_check_interval = int(linear_check_interval)
        self.inner_tolerance_fraction = float(inner_tolerance_fraction)
        self.advective_cfl_limit = float(advective_cfl_limit)
        self.froude_limit = float(froude_limit)
        self.cancelled = lambda: False
        requested_path = Path(source_path or HERE / "implicit_midpoint_fwave.cu")
        _require(requested_path.is_file() and not requested_path.is_symlink(),
                 "Implicit CUDA source must be a regular non-symlink file")
        path = requested_path.resolve()
        source = path.read_text(encoding="utf-8")
        self.source_path = path
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.module = cp.RawModule(code=source, options=CUDA_OPTIONS,
                                   name_expressions=KERNEL_NAMES)
        self.kernels = {name: self.module.get_function(name)
                        for name in KERNEL_NAMES}
        self.forcing = (forcing_identity(solver.data)
                        if BINDING_FIELD in solver.data else None)
        self.typed_roughness = None
        self.roughness_mode = "manning"
        self.roughness_waqua = False
        self.roughness_work_depth = float(
            solver.data.get("manning_friction_chezy_minimum_depth", 0.0))
        if self.forcing is not None:
            binding = solver.data[BINDING_FIELD]
            self.roughness_mode = binding["hydraulic_roughness_mode"]
            self.roughness_waqua = bool(binding["white_colebrook_chzk2d"])
            self.roughness_work_depth = float(
                binding["effective_minimum_work_depth_m"])
            if self.roughness_mode != "manning":
                from .hydraulic_roughness import HydraulicRoughnessCUDA
                roughness_source = path.with_name("hydraulic_roughness.cu")
                _require(roughness_source.is_file() and
                         not roughness_source.is_symlink(),
                         "Archived typed-roughness CUDA source is missing")
                self.typed_roughness = HydraulicRoughnessCUDA(
                    source_path=roughness_source)
        g = solver.reconstruction.geometry
        self.owner, self.neighbour = g["owner"], g["neighbour"]
        self.normal, self.active = g["normal"], g["active"]
        self.boundary_mode = g["boundary_mode"]
        self.offset, self.faces, self.signs = g["offset"], g["faces"], g["signs"]
        self.area = cp.asarray(np.ascontiguousarray(solver.data["area"], dtype=np.float64))
        self.length = cp.asarray(np.ascontiguousarray(solver.data["length"], dtype=np.float64))
        self.base = cp.empty_like(solver.state["q"])
        # Boundary wrappers bind exactly this resident allocation.
        self.midpoint = solver.stepper.stage["q"]
        self.qnew = solver.stepper.next["q"]
        self.rate = cp.empty_like(self.base)
        self.residual = cp.empty_like(self.base)
        self.rhs = cp.empty_like(self.base)
        self.delta = cp.empty_like(self.base)
        self.predictor_rate = cp.zeros_like(self.base)
        self.have_predictor = False
        self.midpoint_water = cp.empty(self.nf, dtype=cp.float64)
        # Read-only handoff from the sole live boundary/flux owner to the
        # outer v96 receipt and exterior-density consumer.  The trace is a
        # physical boundary state, never a right state for a second Riemann
        # solve; the flux is the already pressure-corrected physical flux.
        self.midpoint_boundary_trace = cp.zeros(
            (3, self.nf), dtype=cp.float64)
        self.midpoint_physical_flux = cp.zeros(
            (3, self.nf), dtype=cp.float64)
        self.water_ledger = cp.zeros(self.nb, dtype=cp.float64)
        selected_coupling = coupling_identity(solver.data)
        self.component_composition = (
            selected_coupling == "weak_midpoint_delft_components_v2")
        self.prognostic_secondary_composition = (
            selected_coupling == Q3H3_COUPLING)
        if self.prognostic_secondary_composition:
            _require(
                getattr(solver, "prognostic_secondary", None) is not None and
                getattr(solver.sediment, "prognostic_secondary", None) is
                    solver.prognostic_secondary,
                "Q3-H3 implicit midpoint resident owner is missing")
        self.suspended_composition = (
            selected_coupling in (
                "weak_midpoint_m2_law11_flow2dh_v1",
                "weak_midpoint_m2_law11_flow2dh_channel_composition_v2"))
        self.secondary_base = cp.zeros_like(solver.state["secondary"])
        self.secondary_trial = cp.zeros_like(solver.state["secondary"])
        self.source_first = cp.zeros((), dtype=cp.float64)
        self.source_second = cp.zeros((), dtype=cp.float64)
        self.source_ledger = cp.zeros((), dtype=cp.float64)
        self.secondary_ledger = cp.zeros_like(solver.ledger["secondary"])
        self.suspended_trial = (cp.empty_like(solver.state["suspended"])
                                if self.suspended_composition else None)
        self.suspended_ledger = (cp.zeros_like(solver.ledger["suspended"])
                                 if self.suspended_composition else None)
        self.exchange_ledger = (cp.zeros_like(solver.ledger["exchange"])
                                if self.suspended_composition else None)
        self.advective_suspended_ledger = (
            cp.zeros_like(solver.ledger["suspended"])
            if self.suspended_composition else None)
        self.diffusive_suspended_ledger = (
            cp.zeros_like(solver.ledger["suspended"])
            if self.suspended_composition else None)
        self.divergence = cp.empty(self.nc, dtype=cp.float64)
        self.gradx = cp.empty(self.nc, dtype=cp.float64)
        self.grady = cp.empty(self.nc, dtype=cp.float64)
        self.diffusion = cp.empty_like(self.base)
        self.linear_rate = cp.empty_like(self.base)
        self.cell_metric = cp.empty(self.nc, dtype=cp.float64)
        self.cell_status = cp.empty(self.nc, dtype=cp.int32)
        self.friction_speed = cp.empty(self.nc, dtype=cp.float64)
        # The common fixed-grid transaction wrapper detaches these canonical
        # outputs before restoring a rejected trial.  Alias them deliberately:
        # otherwise the wrapper would replace the implicit result with stale
        # explicit-stepper status/CFL arrays and could mask a rejection.
        _require(solver.stepper.status.shape == () and
                 solver.stepper.status.dtype == cp.int32 and
                 solver.stepper.cfl.shape == () and
                 solver.stepper.cfl.dtype == cp.float64,
                 "Implicit f-wave canonical status/CFL storage differs")
        self.status = solver.stepper.status
        self.metric = solver.stepper.cfl
        self.bi_x = cp.empty_like(self.base)
        self.bi_r = cp.empty_like(self.base)
        self.bi_rhat = cp.empty_like(self.base)
        self.bi_p = cp.empty_like(self.base)
        self.bi_v = cp.empty_like(self.base)
        self.bi_s = cp.empty_like(self.base)
        self.bi_t = cp.empty_like(self.base)
        self.bi_true = cp.empty_like(self.base)
        self.bi_product = cp.empty_like(self.base)
        self.bi_scalars = cp.empty(10, dtype=cp.float64)
        self.bi_control = cp.empty(3, dtype=cp.int32)
        self.last_diagnostics = {}
        self.accepted_steps = self.rejected_steps = 0
        self.fwave_evaluations = self.linear_matvecs = 0
        # Q4/v91 is an explicitly bound additive consumer.  It is absent for
        # every historical route and can never be selected from mesh size or
        # inferred from a positive legacy viscosity scalar.
        self.shared_turbulence_v91 = None
        # Q4/v95 is an explicitly selected, validation-only residual owner.
        # It is absent from every historical path and does not change backend
        # selection.  V96 will own the eventual joint hydraulic/scalar commit.
        self.densin_residual_v95 = None
        # Explicit experimental bend bundle. Never inferred from old deck flags.
        self.bend_physics_v1 = None
        # V97 may provisionally evaluate an accepted outer-owner bed without
        # publishing it into the solver layers.  The resident v97 owner sets
        # and clears this exact array around one implicit solve; every
        # historical route leaves it absent.
        self._v97_provisional_bed = None
        self.equilibrium_eta = cp.empty(self.nc, dtype=cp.float64)
        self.equilibrium_candidate = False
        self._refresh_equilibrium_scales()

    def bind_cancelled(self, callback):
        _require(callable(callback), "Implicit cancellation callback must be callable")
        self.cancelled = callback

    def bind_bend_physics_v1(self, owner):
        from .bend_methods_v1 import ProductionBendOwner
        _require(isinstance(owner, ProductionBendOwner) and owner.solver is self.solver,
                 'Invalid explicit bend owner')
        _require(self.bend_physics_v1 is None and self.shared_turbulence_v91 is None and
                 self.densin_residual_v95 is None and not self.component_composition and
                 not self.prognostic_secondary_composition and not self.suspended_composition,
                 'Bend bundle cannot duplicate another physical owner')
        self.bend_physics_v1 = owner
        self.equilibrium_candidate = False

    def bind_shared_turbulence_v91(self, consumer):
        """Bind the selected v91 owner to every authoritative residual call.

        The resident owner validates exact geometry and rejects a concurrently
        configured legacy post-source diffusion object.  Binding also disables
        the analytical lake shortcut so evidence always traverses ``_full_rate``.
        """
        _require(self.bend_physics_v1 is None, 'Bend bundle already owns momentum mixing')
        _require(self.shared_turbulence_v91 is None or
                 self.shared_turbulence_v91 is consumer,
                 "Another shared-turbulence consumer is already bound")
        _require(self.densin_residual_v95 is None,
                 "v91 and validation-only v95 residual owners require v96 "
                 "composition")
        _require(getattr(consumer, "is_v91_shared_turbulence_consumer", False),
                 "Invalid shared-turbulence consumer binding")
        consumer.bind_controller(self)
        self.shared_turbulence_v91 = consumer
        self.equilibrium_candidate = False

    def bind_densin_residual_v95(self, consumer):
        """Bind the bounded v95 density owner to every true residual call."""
        _require(self.bend_physics_v1 is None, 'Bend bundle cannot duplicate a density residual owner')
        _require(self.densin_residual_v95 is None or
                 self.densin_residual_v95 is consumer,
                 "Another v95 density-residual consumer is already bound")
        _require(self.shared_turbulence_v91 is None,
                 "validation-only v95 and v91 residual owners require v96 "
                 "composition")
        _require(getattr(consumer, "is_v95_densin_residual_consumer", False),
                 "Invalid v95 density-residual consumer binding")
        consumer.bind_controller(self)
        self.densin_residual_v95 = consumer
        self.equilibrium_candidate = False

    def _current_bed(self, layers=None):
        """Return the one authenticated bed visible to this hydraulic step.

        Historical routes continue to derive bed elevation from the solver's
        resident layers.  The v96 joint owner instead freezes its accepted
        parent bed for every nonlinear residual and publishes the Exner bed
        only for the next update.
        """
        if self._v97_provisional_bed is not None:
            return self._v97_provisional_bed
        consumer = self.densin_residual_v95
        if (consumer is not None and
                getattr(consumer, "is_v96_coupled_residual_consumer", False)):
            return consumer.accepted_bed_for_controller()
        return self.solver.bed(
            self.solver.state["layers"] if layers is None else layers)

    def _refresh_equilibrium_scales(self):
        """Refresh the machine-scale analytical closed-lake admission."""
        cp = self.cp
        bed = self._current_bed()
        q = self.solver.state["q"]
        self.equilibrium_eta[...] = q[0] + bed
        state_scale = max(1.0, float(cp.max(cp.abs(q[0])).item()),
                          float(cp.max(cp.abs(bed)).item()),
                          float(cp.max(cp.abs(self.equilibrium_eta)).item()))
        momentum_scale = max(1.0, state_scale * math.sqrt(float(self.grav) * state_scale))
        eps = np.finfo(np.float64).eps
        self.equilibrium_state_scale = state_scale
        self.equilibrium_eta_tolerance = 64.0 * eps * state_scale
        self.equilibrium_momentum_tolerance = 64.0 * eps * momentum_scale
        self.equilibrium_midpoint_defect_bound = (
            128.0 * eps * math.sqrt(3.0 * self.nc) * state_scale)
        self.equilibrium_face_flux_bound = 128.0 * eps * momentum_scale
        self.equilibrium_candidate = bool(
            self.bend_physics_v1 is None and
            not self.component_composition and
            not self.prognostic_secondary_composition and
            not self.suspended_composition and
            not self.nb and self._closed_lake_at_rest())

    def _closed_lake_at_rest(self):
        cp = self.cp
        bed = self._current_bed()
        q = self.solver.state["q"]
        self.equilibrium_eta[...] = q[0] + bed
        eta_range = float((cp.max(self.equilibrium_eta) -
                           cp.min(self.equilibrium_eta)).item())
        momentum = float(cp.max(cp.abs(q[1:])).item())
        return bool(eta_range <= self.equilibrium_eta_tolerance and
                    momentum <= self.equilibrium_momentum_tolerance)

    def _exact_lake_step(self, dt, time, commit):
        """Verify the full residual before preserving the analytical invariant."""
        cp, s = self.cp, self.solver
        full_cfl = self._full_rate(self.base, dt, time + .5 * dt)
        raw_defect = float(cp.linalg.norm(.5 * dt * self.rate).item())
        raw_face_flux = float(cp.max(cp.abs(self.midpoint_water)).item())
        if (raw_defect > self.equilibrium_midpoint_defect_bound or
                raw_face_flux > self.equilibrium_face_flux_bound):
            self.equilibrium_candidate = False
            return None
        cp.copyto(self.midpoint, self.base); cp.copyto(self.qnew, self.base)
        self.water_ledger.fill(0)
        if commit:
            cp.copyto(s.state["q"], self.qnew)
            self.predictor_rate.fill(0); self.have_predictor = False
            self.accepted_steps += 1
        self.metric.fill(0)
        self.last_diagnostics = dict(
            analytical_closed_lake=True, full_acoustic_cfl=full_cfl,
            verified_raw_midpoint_defect=raw_defect,
            verified_midpoint_defect_bound=self.equilibrium_midpoint_defect_bound,
            verified_raw_face_flux=raw_face_flux,
            verified_face_flux_bound=self.equilibrium_face_flux_bound,
            effective_timestep_metric=0.0, water_mass_residual_m3=0.0,
            fwave_evaluations=1, linear_matvecs=0)
        return dict(status=self.status, cfl=self.metric,
                    diagnostics=self.last_diagnostics)

    def _check_cancelled(self):
        if self.cancelled():
            raise ImplicitStepCancelled("Implicit f-wave step cancelled")

    def _friction_half_step(self, q, half_dt, phase):
        """Apply the selected C/M/W split without changing the f-wave solve."""
        cp, s = self.cp, self.solver
        if self.roughness_mode == "manning":
            # Retain the qualified kernel, including its n=0 no-friction
            # identity. The standalone typed operator correctly rejects zero
            # C/ks and therefore must not replace this legacy Manning branch.
            s.stepper.hydro._launch("ti_manning", self.nc,
                (q, s.boundary.arrays["manning"], np.int64(self.nc),
                 np.float64(half_dt), self.grav, self.dry,
                 s.stepper.hydro.chezy_min, self.cell_status))
            if bool(cp.any(self.cell_status != 0).item()):
                raise ImplicitStepRejected(
                    phase + " Manning friction half-step failed")
            return
        depth = q[0]
        wet = depth > self.dry
        denominator = cp.where(wet, depth, 1.0)
        self.friction_speed[...] = cp.where(
            wet, cp.sqrt(q[1]*q[1] + q[2]*q[2])/denominator, 0.0)
        try:
            rate, status = self.typed_roughness.friction_rate(
                depth, self.friction_speed,
                s.boundary.arrays["manning"], mode=self.roughness_mode,
                gravity=float(self.grav),
                minimum_work_depth=self.roughness_work_depth,
                waqua_white_colebrook=self.roughness_waqua)
        except ValueError as error:
            raise ImplicitStepRejected(
                phase + " typed friction half-step failed") from error
        if bool(cp.any(status != 0).item()):
            raise ImplicitStepRejected(
                phase + " typed friction half-step failed")
        scale = 1.0 + np.float64(half_dt)*rate
        q[1] /= scale
        q[2] /= scale

    def _launch(self, name, size, args):
        if size:
            self.kernels[name](((size + 127) // 128,), (128,), args)

    def _advective_metric(self, q, dt):
        cp = self.cp
        self.cell_metric.fill(0)
        self._launch("implicit_advective_cfl_faces", self.nf,
            (q, self.area, self.owner, self.neighbour, self.normal,
             self.length, np.int64(self.nc), np.int64(self.nf),
             np.float64(dt), self.cell_metric))
        advective = float(cp.max(self.cell_metric).item())
        h = q[0]
        velocity = cp.sqrt(q[1] * q[1] + q[2] * q[2]) / h
        froude = float(cp.max(velocity / cp.sqrt(self.grav * h)).item())
        return advective, froude

    def _all_status_clear(self, statuses):
        cp = self.cp
        return not any(bool(cp.any(value != 0).item())
                       for value in statuses if value is not None and value.size)

    def _full_rate(self, q, dt, time):
        """Evaluate authoritative production boundary and augmented-f-wave faces."""
        self._check_cancelled()
        s, cp = self.solver, self.cp
        # A nonlinear residual evaluation is not another time-integrator
        # stage.  Reset the live boundary's per-stage latch/record cursor so
        # repeated Newton residuals all evaluate the same midpoint closure.
        s.boundary.begin_trial()
        bed = self._current_bed()
        reconstructed = s.reconstruction.reconstruct(q)
        s.boundary.apply(q, reconstructed["trace"])
        cp.copyto(self.midpoint_boundary_trace,
                  reconstructed["trace"][:3])
        secondary = s.secondary.reconstruct(q, bed,
            self.secondary_base if self.component_composition else s.state["secondary"])
        s.flux.inputs["q"] = q
        old_target = s.flux.controls["cfl_target"]
        s.flux.controls["dt"] = np.float64(dt)
        s.flux.controls["cfl_target"] = np.float64(1.e100)

        def postassemble(buffers):
            s.boundary.apply_pressure(q, bed, reconstructed["trace"],
                                      buffers["fl"], buffers["fr"],
                                      buffers["face_status"])
        try:
            flux, stress = s.flux.step(s.secondary, postassemble=postassemble)
        finally:
            s.flux.controls["cfl_target"] = old_target
        statuses = (reconstructed["primitive_status"],
                    reconstructed["reconstruction_status"],
                    s.boundary.status, s.boundary.face_status,
                    flux["augmented_status"], flux["face_status"],
                    flux["cell_status"], secondary["status"], stress)
        if not self._all_status_clear(statuses):
            raise ImplicitStepRejected("Augmented f-wave residual status is nonzero")
        if float(cp.min(flux["theta"]).item()) < 1.0:
            raise ImplicitStepRejected(
                "Implicit residual activated the explicit draining limiter")
        self._launch("implicit_full_rate_from_faces", self.nc,
            (q, self.area, self.length, self.normal, flux["fl"], flux["fr"],
             self.offset, self.faces, self.signs, self.active,
             np.int64(self.nc), np.int64(self.nf), self.grav, self.rate))
        cp.copyto(self.midpoint_water, flux["fl"][0])
        cp.copyto(self.midpoint_physical_flux, flux["fl"][:3])
        if self.shared_turbulence_v91 is not None:
            # This is deliberately inside the true nonlinear residual.  The
            # resident hook reads the one pre-hydraulic handoff, adds one
            # variable-vicuv face rate, and counts this physical evaluation.
            self.shared_turbulence_v91.add_momentum_rate(q, self.rate)
        if self.densin_residual_v95 is not None:
            # The accepted concentration is frozen before the hydraulic trial,
            # while the current nonlinear midpoint depth is consumed here.
            # This is the single physical DensIn application for this residual.
            self.densin_residual_v95.add_momentum_rate(q, self.rate)
        if self.bend_physics_v1 is not None:
            self.bend_physics_v1.add_momentum_rate(q, self.midpoint_water, self.rate)
        self.fwave_evaluations += 1
        return float(cp.max(flux["cfl_ratio"]).item())

    def _linear_acoustic_rate(self, increment, href):
        self.divergence.fill(0); self.gradx.fill(0); self.grady.fill(0)
        self.diffusion.fill(0)
        self._launch("implicit_linear_acoustic_faces", self.nf,
            (increment, href, self.owner, self.neighbour, self.boundary_mode,
             self.normal, self.length, np.int64(self.nc), np.int64(self.nf),
             self.grav, self.divergence, self.gradx, self.grady,
             self.diffusion))
        self._launch("implicit_linear_acoustic_finish", self.nc,
            (href, self.area, self.divergence, self.gradx, self.grady,
             self.diffusion, np.int64(self.nc), self.grav, self.linear_rate))
        return self.linear_rate

    def _linear_matrix(self, vector, href, tau, out):
        self.linear_matvecs += 1
        self.cp.multiply(self._linear_acoustic_rate(vector, href), -tau, out=out)
        out += vector
        return out

    def _bicgstab(self, rhs, href, tau, outer_tolerance):
        cp = self.cp
        x, r, rhat = self.bi_x, self.bi_r, self.bi_rhat
        p, v, work, t = self.bi_p, self.bi_v, self.bi_s, self.bi_t
        x.fill(0); p.fill(0); v.fill(0)
        cp.copyto(r, rhs); cp.copyto(rhat, rhs)
        rhs_norm = float(cp.linalg.norm(rhs).item())
        target = self.inner_tolerance_fraction * outer_tolerance
        effective_atol = min(self.linear_atol, .95 * target)
        effective_rtol = min(self.linear_rtol,
            .95 * target / max(rhs_norm, np.finfo(np.float64).tiny))
        tolerance = max(effective_atol, effective_rtol * rhs_norm)
        residual = rhs_norm
        if residual <= tolerance:
            return x, 0, residual, residual, tolerance, target
        self.bi_scalars.fill(0); self.bi_scalars[1:4] = 1.0
        self.bi_scalars[9] = tolerance; self.bi_control.fill(0)
        total = np.int64(3 * self.nc)

        def dot(left, right, slot):
            cp.multiply(left, right, out=self.bi_product)
            cp.sum(self.bi_product, out=self.bi_scalars[slot])

        launched = 0; control = np.zeros(3, dtype=np.int32)
        while launched < self.linear_maxiter and not control[0] and not control[1]:
            self._check_cancelled()
            batch = min(self.linear_check_interval, self.linear_maxiter - launched)
            for _ in range(batch):
                dot(rhat, r, 0)
                self._launch("implicit_bicg_direction", 3 * self.nc,
                    (total, p, r, v, self.bi_scalars, self.bi_control))
                self._linear_matrix(p, href, tau, v)
                dot(rhat, v, 4)
                self.kernels["implicit_bicg_alpha"]((1,), (1,),
                    (self.bi_scalars, self.bi_control))
                self._launch("implicit_bicg_residual_stage", 3 * self.nc,
                    (total, r, v, self.bi_scalars, self.bi_control, work))
                dot(work, work, 7)
                self._linear_matrix(work, href, tau, t)
                dot(t, t, 5); dot(t, work, 6)
                self.kernels["implicit_bicg_omega"]((1,), (1,),
                    (self.bi_scalars, self.bi_control))
                self._launch("implicit_bicg_update", 3 * self.nc,
                    (total, x, r, p, work, t, self.bi_scalars,
                     self.bi_control))
                dot(r, r, 8)
                self.kernels["implicit_bicg_convergence"]((1,), (1,),
                    (self.bi_scalars, self.bi_control))
            launched += batch
            control = cp.asnumpy(self.bi_control)
        scalar = cp.asnumpy(self.bi_scalars)
        iteration = int(control[2])
        recurrence = math.sqrt(max(float(scalar[8]), 0.0))
        if control[1] or not control[0] or recurrence > tolerance:
            raise ImplicitStepRejected(
                "Implicit BiCGSTAB failed or did not converge")
        self._linear_matrix(x, href, tau, self.bi_true)
        self.bi_true -= rhs
        true_residual = float(cp.linalg.norm(self.bi_true).item())
        if not math.isfinite(true_residual) or true_residual > target:
            raise ImplicitStepRejected(
                "Implicit linear true residual exceeds the outer tolerance")
        return x, iteration, recurrence, true_residual, tolerance, target

    def _nonlinear_tolerance(self):
        cp = self.cp
        update = float(cp.linalg.norm(self.midpoint - self.base).item())
        forcing = float(cp.linalg.norm(.5 * self.current_dt * self.rate).item())
        scale = max(update, forcing)
        tolerance = max(self.nonlinear_atol * math.sqrt(3.0 * self.nc),
                        self.nonlinear_rtol * scale)
        return tolerance, scale

    def checkpoint_fields(self):
        _require(self.bend_physics_v1 is None,
                 'Use the bend owner checkpoint; hydraulic-only restart would omit physics')
        result = dict(
            implicit_predictor_rate=self.cp.asnumpy(self.predictor_rate),
            implicit_have_predictor=np.int32(self.have_predictor),
        )
        if self.suspended_composition:
            result.update(
                self.solver.flow2dh_boundary_diffusion.checkpoint_fields())
        return result

    def restore_checkpoint(self, fields):
        _require(self.bend_physics_v1 is None,
                 'Use the bend owner restart; hydraulic-only restart would omit physics')
        rate = np.asarray(fields["implicit_predictor_rate"])
        flag = np.asarray(fields["implicit_have_predictor"])
        _require(rate.shape == (3, self.nc) and rate.dtype == np.float64 and
                 np.isfinite(rate).all() and flag.shape == () and
                 flag.dtype == np.int32 and int(flag) in (0, 1),
                 "Implicit predictor checkpoint differs")
        self.predictor_rate.set(rate)
        self.have_predictor = bool(int(flag))
        if self.suspended_composition:
            self.solver.flow2dh_boundary_diffusion.restore_checkpoint(fields)
        self._refresh_equilibrium_scales()

    def step(self, dt, time, *, sediment_callback=None, commit=True):
        cp, s = self.cp, self.solver
        if self.shared_turbulence_v91 is not None and bool(commit):
            # V91 authenticates a provisional hydraulic endpoint which must
            # still pass through the one v82 scalar consumer before the outer
            # transaction can commit.  Letting this controller commit its
            # hydraulic state directly would publish q/layers before that
            # scalar transaction and falsely label a partial update accepted.
            # The eventual v96 owner may compose that atomic commit; bounded
            # v91 therefore admits the live controller only with commit=False.
            self.shared_turbulence_v91.abort_hydraulic_trial()
            raise ValueError(
                "v91 shared turbulence requires commit=False until the "
                "outer scalar transaction owns the atomic commit")
        if self.densin_residual_v95 is not None and bool(commit):
            self.densin_residual_v95.abort_hydraulic_trial()
            raise ValueError(
                "v95 density residual requires commit=False until the "
                "v96 joint transaction owns the atomic commit")
        if not np.isfinite([dt, time]).all() or dt <= 0.0:
            raise ValueError("Finite positive implicit timestep required")
        self.status.fill(0); self.metric.fill(0)
        self.current_dt = float(dt)
        initial_fwave = self.fwave_evaluations
        initial_matvec = self.linear_matvecs
        cfl_target = float(s.data["cfl_target"])
        max_ratio = 0.0
        prognostic_trial_open = False
        prognostic_conservation_residual = 0.0
        prognostic_source_integral = 0.0
        prognostic_boundary_integral = 0.0
        try:
            self._check_cancelled()
            if self.shared_turbulence_v91 is not None:
                self.shared_turbulence_v91.require_open_for_controller(
                    dt, time, s.state["q"])
            if self.densin_residual_v95 is not None:
                self.densin_residual_v95.require_open_for_controller(
                    dt, time, s.state["q"])
            cp.copyto(self.base, s.state["q"])
            self._friction_half_step(self.base, .5 * dt, "Pre-midpoint")
            if self.bend_physics_v1 is not None:
                _require(sediment_callback is None, 'Bend bundle owns the only sediment callback')
                self.bend_physics_v1.begin(self.base, self._current_bed(), dt, time)
            if self.component_composition:
                # Prognostic hI is never written into the accepted state during
                # Newton iterations. The first source half-step also owns the
                # curvature retained for inflow intensities and momentum stress.
                cp.copyto(self.secondary_base, s.state["secondary"])
                self.source_first.fill(0); self.source_second.fill(0)
                self.source_ledger.fill(0); self.secondary_ledger.fill(0)
                source = s.secondary.relax(self.base, self.secondary_base, .5 * dt)
                if int(source["status"].item()) != 0:
                    raise ImplicitStepRejected("Pre-midpoint secondary source half-step failed")
                cp.copyto(self.source_first, source["source_integral"])
            if self.equilibrium_candidate:
                if self._closed_lake_at_rest():
                    lake = self._exact_lake_step(dt, time, commit)
                    if lake is not None:
                        return lake
                self.equilibrium_candidate = False
            advective, froude = self._advective_metric(self.base, dt)
            maximum_advective, maximum_froude = advective, froude
            max_ratio = max(advective / self.advective_cfl_limit,
                            froude / self.froude_limit)
            if max_ratio > 1.0:
                raise ImplicitStepRejected("Implicit low-Froude envelope exceeded")
            tau = .5 * float(dt)
            if self.have_predictor:
                self.midpoint[...] = self.base + tau * self.predictor_rate
            else:
                cp.copyto(self.midpoint, self.base)
            rows = []; linear_rows = []; converged = False; full_cfl = 0.0
            tolerance = math.nan; scaled = math.inf
            for iteration in range(self.nonlinear_maxiter + 1):
                full_cfl = self._full_rate(self.midpoint, dt, time + tau)
                self.residual[...] = self.midpoint - self.base - tau * self.rate
                residual_norm = float(cp.linalg.norm(self.residual).item())
                tolerance, scale = self._nonlinear_tolerance()
                scaled = residual_norm / tolerance
                rows.append(dict(iteration=iteration, norm=residual_norm,
                                 tolerance=tolerance, scaled=scaled, scale=scale))
                if not math.isfinite(residual_norm):
                    raise ImplicitStepRejected("Nonfinite midpoint residual")
                if residual_norm <= tolerance:
                    converged = True; break
                if iteration == self.nonlinear_maxiter:
                    break
                cp.negative(self.residual, out=self.rhs)
                values = self._bicgstab(self.rhs, self.base[0], tau, tolerance)
                solution, iterations, recurrence, true, linear_tol, target = values
                cp.copyto(self.delta, solution); self.midpoint += self.delta
                linear_rows.append(dict(iterations=iterations,
                    recurrence_residual=recurrence, true_residual=true,
                    tolerance=linear_tol, outer_fraction_target=target))
                if (not bool(cp.isfinite(self.midpoint).all().item()) or
                        float(cp.min(self.midpoint[0]).item()) <= self.wet_floor):
                    raise ImplicitStepRejected("Newton left the fully wet envelope")
                stage_advective, stage_froude = self._advective_metric(self.midpoint, dt)
                maximum_advective = max(maximum_advective, stage_advective)
                maximum_froude = max(maximum_froude, stage_froude)
                max_ratio = max(max_ratio,
                    stage_advective / self.advective_cfl_limit,
                    stage_froude / self.froude_limit)
                if max_ratio > 1.0:
                    raise ImplicitStepRejected("Newton left the low-Froude envelope")
            if not converged:
                raise ImplicitStepRejected("Implicit midpoint did not converge")

            self.qnew[...] = self.base + dt * self.rate
            residual_depth = self.qnew[0].copy()
            self.divergence.fill(0)
            self._launch("implicit_water_divergence_faces", self.nf,
                (self.midpoint_water, self.owner, self.neighbour, self.length,
                 np.int64(self.nf), self.divergence))
            self._launch("implicit_depth_from_flux", self.nc,
                (self.base[0], self.area, self.divergence, np.int64(self.nc),
                 np.float64(dt), self.wet_floor, self.qnew[0], self.cell_status))
            if (bool(cp.any(self.cell_status != 0).item()) or
                    not bool(cp.isfinite(self.qnew).all().item())):
                raise ImplicitStepRejected("Accepted full face flux left wet envelope")
            depth_defect = float(cp.max(cp.abs(self.qnew[0] - residual_depth)).item())
            s.boundary.integrate_ledger(self.midpoint_water, dt)
            cp.copyto(self.water_ledger, s.boundary.ledger)
            mass_delta = float(cp.sum(self.area * (self.qnew[0] - self.base[0])).item())
            boundary_delta = float(cp.sum(self.water_ledger).item())
            mass_scale = max(1.0, abs(float(cp.sum(self.area * self.base[0]).item())))
            mass_residual = mass_delta - boundary_delta
            if abs(mass_residual) > max(1.e-12, 1.e-11 * mass_scale):
                raise ImplicitStepRejected("Implicit water ledger is not conservative")
            endpoint = self.qnew - (2.0 * self.midpoint - self.base) + 2.0 * self.residual
            endpoint_defect = float(cp.max(cp.abs(endpoint)).item())
            endpoint_scale = max(1.0, float(cp.max(cp.abs(self.base)).item()),
                                 float(cp.max(cp.abs(self.midpoint)).item()),
                                 float(cp.max(cp.abs(self.qnew)).item()))
            endpoint_bound = 128.0 * np.finfo(np.float64).eps * endpoint_scale
            if endpoint_defect > endpoint_bound:
                raise ImplicitStepRejected("Implicit endpoint identity failed")
            if self.densin_residual_v95 is not None:
                # Capture the exact split base and conservative implicit
                # endpoint before the second Manning friction half-step.
                self.densin_residual_v95.capture_converged_residual_endpoint(
                    self.base, self.qnew)

            if self.prognostic_secondary_composition:
                # The v57 FLOW target owns the accepted old depth and the
                # complete post-hydraulic endpoint velocity/curvature.  Apply
                # the second friction half-step before opening the signed-hI
                # transaction; the generic model-2 source/filter path is inert.
                self._friction_half_step(
                    self.qnew, .5 * dt, "Post-midpoint")
                if float(cp.min(self.qnew[0]).item()) <= self.wet_floor:
                    raise ImplicitStepRejected(
                        "Post-midpoint friction wet-floor failed")
                prognostic = s.prognostic_secondary
                prognostic.begin_trial(s.state["secondary"])
                prognostic_trial_open = True
                accepted_bed = self._current_bed()
                base_reconstruction = s.reconstruction.reconstruct(self.base)
                advanced = prognostic.advance(
                    self.base, self.qnew, accepted_bed,
                    self.midpoint_water, dt,
                    base_reconstruction["active"],
                    hydraulic_buffers=base_reconstruction)
                if int(advanced["status"].item()) != 0:
                    raise ImplicitStepRejected(
                        "Q3-H3 prognostic secondary update failed")
                cp.copyto(self.secondary_trial, prognostic.trial)
                cp.copyto(self.secondary_ledger,
                          prognostic.boundary_ledger)
                cp.sum(prognostic.source_ledger_cells,
                       out=self.source_ledger)
                prognostic_source_integral = float(
                    self.source_ledger.item())
                prognostic_boundary_integral = float(
                    cp.sum(self.secondary_ledger).item())
                inventory_delta = float(cp.sum(
                    self.area * (prognostic.trial -
                                 s.state["secondary"])).item())
                prognostic_conservation_residual = (
                    inventory_delta - prognostic_boundary_integral -
                    prognostic_source_integral)
                inventory_scale = max(
                    1.0,
                    abs(float(cp.sum(
                        self.area * s.state["secondary"]).item())),
                    abs(inventory_delta),
                    abs(prognostic_boundary_integral),
                    abs(prognostic_source_integral))
                if abs(prognostic_conservation_residual) > max(
                        1.e-12, 1.e-11 * inventory_scale):
                    raise ImplicitStepRejected(
                        "Q3-H3 signed-hI ledger is not conservative")

            if sediment_callback is not None:
                # The unchanged Euler sediment leg performs one final boundary
                # evaluation at the same midpoint and owns the single accepted
                # diagnostic stage record.  Start that record transaction from
                # a clean cursor after the nonlinear residual evaluations.
                s.boundary.begin_trial()
                if self.prognostic_secondary_composition:
                    sediment = sediment_callback(
                        self.midpoint, dt, time + tau,
                        secondary=s.prognostic_secondary.trial)
                elif self.component_composition:
                    sediment = sediment_callback(self.midpoint, dt, time + tau,
                                                  secondary=self.secondary_base)
                else:
                    sediment = sediment_callback(self.midpoint, dt, time + tau)
            else:
                sediment = None
            if sediment is not None and int(sediment["status"].item()) != 0:
                raise ImplicitStepRejected("Weak midpoint sediment leg failed")

            if not self.prognostic_secondary_composition:
                self._friction_half_step(
                    self.qnew, .5 * dt, "Post-midpoint")
                if float(cp.min(self.qnew[0]).item()) <= self.wet_floor:
                    raise ImplicitStepRejected(
                        "Post-midpoint friction wet-floor failed")

            trial_layers = (sediment["state"]["layers"] if sediment is not None
                            else s.state["layers"])
            trial_secondary = (sediment["state"]["secondary"] if sediment is not None
                               else s.state["secondary"])
            bed_ledger = (sediment["ledger"]["bed"] if sediment is not None
                          else s.zero_bed_ledger)
            secondary_ledger = (sediment["ledger"]["secondary"] if sediment is not None
                                else cp.zeros_like(s.ledger["secondary"]))
            if self.bend_physics_v1 is not None:
                self.bend_physics_v1.finish(self.base, self.qnew, self.midpoint_water)
                trial_layers = self.bend_physics_v1.layers_trial
                bed_ledger = self.bend_physics_v1.bed_ledger
            if self.prognostic_secondary_composition:
                trial_secondary = s.prognostic_secondary.trial
                secondary_ledger = self.secondary_ledger
            suspended_diagnostics = None
            if self.suspended_composition:
                # The Euler callback owns only bedload/Exner. Suspended mass is
                # reconstructed from the accepted split state and transported
                # once with the converged midpoint water flux.  Resolve the
                # source-exact FLOW boundary per face before reconstruction;
                # this preserves mixed flow directions under one boundary id.
                flow2dh = s.flow2dh_boundary_diffusion
                endpoint_hydraulic = s.reconstruction.reconstruct(self.qnew)
                accepted_face_area, transport_face_active, area_status = (
                    flow2dh.face_area_from_trace(endpoint_hydraulic["trace"]))
                if bool(cp.any(area_status != 0).item()):
                    raise ImplicitStepRejected(
                        "Accepted hydraulic diffusion face area failed")
                boundary = flow2dh.resolve_boundary(
                    self.base[0], s.state["suspended"], s.bc_concentration,
                    self.midpoint_water, dt)
                if any(bool(cp.any(value != 0).item())
                       for value in boundary["status"]):
                    raise ImplicitStepRejected(
                        "FLOW 2DH suspended boundary resolution failed")
                base_reconstruction = s.reconstruction.reconstruct(self.base)
                suspended_faces = flow2dh.reconstruct(
                    self.base[0], s.state["suspended"], base_reconstruction)
                advected = flow2dh.advect(
                    self.base[0], s.state["suspended"], self.qnew[0],
                    self.midpoint_water, suspended_faces["left"],
                    suspended_faces["right"],
                    base_reconstruction["hydro_constant"], dt)
                suspended_status = (suspended_faces["status"],
                                    *advected["status"])
                if any(bool(cp.any(value != 0).item())
                       for value in suspended_status):
                    raise ImplicitStepRejected(
                        "Converged midpoint suspended advection failed")
                cp.copyto(self.advective_suspended_ledger, advected["ledger"])
                advective_ledger_residual = (
                    self.advective_suspended_ledger -
                    boundary["expected_advective_ledger"])
                advective_ledger_scale = cp.maximum(
                    1.0, cp.maximum(cp.abs(self.advective_suspended_ledger),
                                    cp.abs(boundary[
                                        "expected_advective_ledger"])))
                if bool(cp.any(cp.abs(advective_ledger_residual) >
                               1.e-12*advective_ledger_scale).item()):
                    raise ImplicitStepRejected(
                        "Accepted FCT suspended boundary ledger differs")

                # FLOW's physical face law is retained while BRAIDS records a
                # distinct explicit donor-limited post-FCT update.  The face
                # area comes from the authenticated accepted-trace arithmetic
                # depth-times-width adapter above, never a cell-depth mean
                # reconstructed inside the diffusion operator.
                diffused = flow2dh.diffuse(
                    self.qnew[0], advected["next_mass"], accepted_face_area,
                    transport_face_active, dt)
                if any(bool(cp.any(value != 0).item())
                       for value in diffused["status"]):
                    raise ImplicitStepRejected(
                        "FLOW 2DH horizontal suspended diffusion failed")
                cp.copyto(self.suspended_trial, diffused["next_mass"])
                cp.copyto(self.diffusive_suspended_ledger,
                           diffused["ledger"])
                cp.add(self.advective_suspended_ledger,
                       self.diffusive_suspended_ledger,
                       out=self.suspended_ledger)

                # Recompute law-11 ssus and total shear at one co-located,
                # post-friction endpoint. FLOW's assembled scalar mass form
                # retains the accepted pre-step depth h0 and endpoint h1 even
                # though capacity, speed and shear all use the endpoint.
                endpoint_bed = self._current_bed(trial_layers)
                endpoint_reconstruction = s.reconstruction.reconstruct(self.qnew)
                s.sediment.prepare(
                    self.qnew, endpoint_bed, trial_layers, trial_secondary,
                    endpoint_reconstruction["active"])
                capacity = s.sediment.last_capacity_result
                if (capacity is None or
                        bool(cp.any(capacity.status != 0).item()) or
                        bool(cp.any(s.sediment.buffers["cell_status"] != 0).item())):
                    raise ImplicitStepRejected(
                        "Endpoint law-11 suspended capacity failed")
                exchanged = s.law11_soursin_exchange.step(
                    self.base[0], self.qnew[0], self.qnew[1:3],
                    self.suspended_trial, trial_layers[0],
                    s.sediment.buffers["equilibrium_suspended_capacity"],
                    s.flow2dh_surface_fraction, capacity.total_shear,
                    endpoint_reconstruction["active"], dt)
                reduced = s.exchange_reduction.step(
                    self.suspended_trial, trial_layers[0],
                    exchanged["suspended"], exchanged["active"])
                if (bool(cp.any(exchanged["status"] != 0).item()) or
                        bool(cp.any(reduced["status"] != 0).item())):
                    raise ImplicitStepRejected(
                        "Endpoint FLOW 2DH suspended exchange failed")
                cp.copyto(self.suspended_trial, exchanged["suspended"])
                cp.copyto(trial_layers[0], exchanged["active"])
                cp.copyto(self.exchange_ledger, reduced["ledger"])
                rebalance_status = s.sediment.rebalance(trial_layers)
                if (rebalance_status is not None and
                        bool(cp.any(rebalance_status != 0).item())):
                    raise ImplicitStepRejected(
                        "Post-exchange active-layer rebalance failed")

                before_stock = cp.sum(
                    (cp.sum(s.state["layers"], axis=0) +
                     s.state["suspended"]) * self.area[None, :], axis=1)
                after_stock = cp.sum(
                    (cp.sum(trial_layers, axis=0) + self.suspended_trial) *
                    self.area[None, :], axis=1)
                boundary_stock = (cp.sum(bed_ledger, axis=1) +
                                  cp.sum(self.suspended_ledger, axis=1))
                stock_residual = after_stock - before_stock - boundary_stock
                exchange_residual = cp.sum(self.exchange_ledger, axis=1)
                stock_scale = cp.maximum(
                    1.0, cp.maximum(cp.abs(before_stock), cp.abs(after_stock)))
                stock_scale = cp.maximum(stock_scale, cp.abs(boundary_stock))
                conservation_tolerance = 1.e-12 * stock_scale
                if (bool(cp.any(cp.abs(stock_residual) >
                               conservation_tolerance).item()) or
                        bool(cp.any(cp.abs(exchange_residual) >
                                   conservation_tolerance).item())):
                    raise ImplicitStepRejected(
                        "Suspended plus bed inventory ledger is not conservative")
                limiter_kg = cp.sum(
                    exchanged["limiter_correction"] * self.area[None, :],
                    axis=1)
                suspended_diagnostics = dict(
                    suspended_advective_boundary_ledger_kg=cp.asnumpy(cp.sum(
                        self.advective_suspended_ledger, axis=1)).tolist(),
                    suspended_advective_ledger_residual_kg=cp.asnumpy(cp.sum(
                        advective_ledger_residual, axis=1)).tolist(),
                    suspended_diffusive_boundary_ledger_kg=cp.asnumpy(cp.sum(
                        self.diffusive_suspended_ledger, axis=1)).tolist(),
                    suspended_boundary_ledger_kg=cp.asnumpy(cp.sum(
                        self.suspended_ledger, axis=1)).tolist(),
                    exchange_ledger_kg=cp.asnumpy(self.exchange_ledger).tolist(),
                    combined_inventory_residual_kg=cp.asnumpy(
                        stock_residual).tolist(),
                    exchange_internal_residual_kg=cp.asnumpy(
                        exchange_residual).tolist(),
                    safety_limiter_correction_kg=cp.asnumpy(limiter_kg).tolist(),
                    safety_limiter_applied=bool(cp.any(limiter_kg > 0.0).item()),
                    sediment_active_cells=int(cp.sum(
                        exchanged["sediment_active"]).item()),
                )
            if self.component_composition:
                # The Euler callback's hydraulic endpoint is deliberately
                # discarded. Its scalar advection must therefore also be
                # discarded: use the same converged face-water flux and final
                # hydraulic depth as the authoritative midpoint water ledger.
                bed = self._current_bed()
                # FCT transports an inventory carried by the old depth. Using
                # midpoint depth here would break its local water/scalar bounds
                # even though the converged face flux is conservative.
                reconstructed = s.secondary.reconstruct(
                    self.base, bed, self.secondary_base)
                advected = s.secondary.advect(self.base, self.qnew,
                    self.secondary_base, self.midpoint_water, dt)
                if (int(reconstructed["status"].item()) != 0 or
                        int(advected["status"].item()) != 0):
                    raise ImplicitStepRejected("Converged midpoint secondary advection failed")
                cp.copyto(self.secondary_trial, advected["next"])
                cp.copyto(self.secondary_ledger, advected["ledger"])
                source = s.secondary.relax(self.qnew, self.secondary_trial, .5 * dt)
                if int(source["status"].item()) != 0:
                    raise ImplicitStepRejected("Post-midpoint secondary source half-step failed")
                cp.copyto(self.source_second, source["source_integral"])
                cp.add(self.source_first, self.source_second, out=self.source_ledger)
                if self.shared_turbulence_v91 is None:
                    diffusion_status = s.post_source(dict(q=self.qnew,
                        layers=trial_layers, secondary=self.secondary_trial), dt)
                    if (diffusion_status is not None and
                            bool(cp.any(diffusion_status != 0).item())):
                        raise ImplicitStepRejected("Post-source momentum diffusion failed")
                else:
                    # V91 already owns viscosity in every nonlinear residual.
                    # Calling the old split path here would apply it twice.
                    diffusion_status = None
                if (not bool(cp.isfinite(self.qnew).all().item()) or
                        not bool(cp.isfinite(self.secondary_trial).all().item()) or
                        not bool(cp.isfinite(self.source_ledger).item())):
                    raise ImplicitStepRejected("Nonfinite composed midpoint state")
                trial_secondary = self.secondary_trial
                secondary_ledger = self.secondary_ledger
            self._check_cancelled()
            v91_receipt = None
            if self.shared_turbulence_v91 is not None:
                v91_receipt = self.shared_turbulence_v91.finish_hydraulic_trial(
                    self.midpoint, self.qnew, self.midpoint_water,
                    dt_s=dt, time_s=time)
            densin_v95_receipt = None
            if self.densin_residual_v95 is not None:
                densin_v95_receipt = (
                    self.densin_residual_v95.finish_hydraulic_trial(
                        self.midpoint, self.qnew, self.midpoint_water,
                        dt_s=dt, time_s=time))
            if commit:
                cp.copyto(s.state["q"], self.qnew)
                cp.copyto(s.state["layers"], trial_layers)
                if self.prognostic_secondary_composition:
                    s.prognostic_secondary.finish_trial(
                        s.state["secondary"], commit=True)
                    prognostic_trial_open = False
                else:
                    cp.copyto(s.state["secondary"], trial_secondary)
                if self.suspended_composition:
                    cp.copyto(s.state["suspended"], self.suspended_trial)
                    s.flow2dh_boundary_diffusion.commit_boundary_memory()
                s.ledger["water"] += self.water_ledger
                s.ledger["bed"] += bed_ledger
                s.ledger["secondary"] += secondary_ledger
                if self.suspended_composition:
                    s.ledger["suspended"] += self.suspended_ledger
                    s.ledger["exchange"] += self.exchange_ledger
                if (self.component_composition or
                        self.prognostic_secondary_composition):
                    s.ledger["source"] += self.source_ledger
            elif self.prognostic_secondary_composition:
                s.prognostic_secondary.finish_trial(
                    s.state["secondary"], commit=False)
                prognostic_trial_open = False
            if commit:
                cp.copyto(self.predictor_rate, self.rate)
                self.have_predictor = True
                self.accepted_steps += 1
                if self.bend_physics_v1 is not None:
                    self.bend_physics_v1.physics.commit()
            effective = cfl_target * max_ratio
            self.metric.fill(effective)
            self.last_diagnostics = dict(
                nonlinear_rows=rows, linear_rows=linear_rows,
                nonlinear_corrections=len(linear_rows),
                full_acoustic_cfl=full_cfl,
                effective_timestep_metric=effective,
                advective_cfl=maximum_advective, froude=maximum_froude,
                depth_flux_defect_m=depth_defect,
                water_mass_residual_m3=mass_residual,
                endpoint_identity_defect=endpoint_defect,
                endpoint_identity_bound=endpoint_bound,
                fwave_evaluations=self.fwave_evaluations - initial_fwave,
                linear_matvecs=self.linear_matvecs - initial_matvec,
            )
            if self.component_composition:
                self.last_diagnostics.update(component_coupling_schema=2,
                    secondary_source_integral=float(self.source_ledger.item()),
                    secondary_boundary_integral=float(cp.sum(self.secondary_ledger).item()),
                    momentum_diffusion_applied=(
                        s.diffusion is not None or
                        self.shared_turbulence_v91 is not None))
            if self.bend_physics_v1 is not None:
                self.last_diagnostics['bend_bundle_v1'] = dict(self.bend_physics_v1.physics.last)
            if self.prognostic_secondary_composition:
                self.last_diagnostics.update(
                    component_coupling_schema=11,
                    prognostic_secondary_checkpoint_schema=8,
                    prognostic_secondary_update_order=(
                        "accepted_qold_to_post_hydraulic_qnew_then_"
                        "advection_then_full_BE_then_capacity_rotation_v1"),
                    secondary_source_integral=
                        prognostic_source_integral,
                    secondary_boundary_integral=
                        prognostic_boundary_integral,
                    secondary_inventory_residual=
                        prognostic_conservation_residual,
                    secondary_state_units="m2_s",
                    secondary_integrated_budget_units=(
                        s.prognostic_secondary.identity[
                            "integrated_ledger_units"]),
                    momentum_diffusion_applied=False)
            if self.suspended_composition:
                self.last_diagnostics.update(
                    suspended_coupling_schema=1, **suspended_diagnostics)
            if v91_receipt is not None:
                receipt_key = ("shared_turbulence_v91" if commit else
                               "shared_turbulence_v91_trial")
                self.last_diagnostics.update({
                    receipt_key: v91_receipt,
                    "shared_turbulence_true_residual": True,
                    "shared_turbulence_outer_transaction_committed":
                        bool(commit),
                    "shared_turbulence_duplicate_post_source_count": 0,
                })
            if densin_v95_receipt is not None:
                self.last_diagnostics.update({
                    "densin_residual_v95_trial": densin_v95_receipt,
                    "densin_true_nonlinear_residual": True,
                    "densin_outer_transaction_committed": False,
                    "densin_post_convergence_application_count": 0,
                })
            return dict(status=self.status, cfl=self.metric,
                        diagnostics=self.last_diagnostics)
        except ImplicitStepCancelled as exc:
            if self.shared_turbulence_v91 is not None:
                self.shared_turbulence_v91.abort_hydraulic_trial()
            if self.densin_residual_v95 is not None:
                self.densin_residual_v95.abort_hydraulic_trial()
            if prognostic_trial_open:
                s.prognostic_secondary.abort_trial()
                prognostic_trial_open = False
            self.status.fill(1); self.metric.fill(max(cfl_target, 1.0))
            self.rejected_steps += 1
            self.last_diagnostics = dict(cancelled=True, error=str(exc))
            return dict(status=self.status, cfl=self.metric,
                        diagnostics=self.last_diagnostics)
        except ImplicitStepRejected as exc:
            if self.shared_turbulence_v91 is not None:
                self.shared_turbulence_v91.abort_hydraulic_trial()
            if self.densin_residual_v95 is not None:
                self.densin_residual_v95.abort_hydraulic_trial()
            if prognostic_trial_open:
                s.prognostic_secondary.abort_trial()
                prognostic_trial_open = False
            self.status.fill(1)
            self.metric.fill(cfl_target * max(2.0, max_ratio))
            self.rejected_steps += 1
            self.last_diagnostics = dict(cancelled=False, error=str(exc),
                fwave_evaluations=self.fwave_evaluations - initial_fwave,
                linear_matvecs=self.linear_matvecs - initial_matvec)
            return dict(status=self.status, cfl=self.metric,
                        diagnostics=self.last_diagnostics)
        except Exception:
            if self.shared_turbulence_v91 is not None:
                self.shared_turbulence_v91.abort_hydraulic_trial()
            if self.densin_residual_v95 is not None:
                self.densin_residual_v95.abort_hydraulic_trial()
            if prognostic_trial_open:
                s.prognostic_secondary.abort_trial()
            raise
        finally:
            if self.bend_physics_v1 is not None:
                self.bend_physics_v1.physics.abort()
