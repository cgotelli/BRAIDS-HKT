"""Canonical physical/runtime contract for FLOW 2DH suspended transport.

The capacity profile answers *how much equilibrium transport is available*.
This profile answers *how that suspended channel is routed*: advected
concentration, Galappatti adaptation, settling controls and fraction basis.
Keeping the identities separate preserves M1 restart meaning while M2 evolves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from braids_core.sediment_method_catalog import method, semantic_catalog_id

from .capacity_profile import (
    CapacityProfileError,
    CapacityRuntimeProfile,
    capacity_profile_document,
    capacity_profile_semantic_id,
    load_capacity_profile,
)
from .flow2dh_exchange import (
    FLOW2DH_EXCHANGE_ID,
    FLOW2DH_EXCHANGE_SCHEMA,
    FLOW2DH_EXCHANGE_SEMANTIC_ID,
)
from .flow2dh_boundary_diffusion import (
    FLOW2DH_HORIZONTAL_DIFFUSION_ID,
    FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA,
    FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
    FLOW2DH_HORIZONTAL_SCHMIDT,
    FLOW2DH_OPEN_BOUNDARY_ID,
    FLOW2DH_OPEN_BOUNDARY_SCHEMA,
    FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
)
from .flow2dh_settling import (
    FLOW2DH_SETTLING_FORM,
    FLOW2DH_SETTLING_ID,
    FLOW2DH_SETTLING_SCHEMA,
    FLOW2DH_SETTLING_SEMANTIC_ID,
)


SUSPENSION_PROFILE_SCHEMA = 3
SUSPENSION_PROFILE_SCOPE_ID = "delft3d4-flow-2dh-current-suspension-runtime-v3"
TRANSPORT_ROUTE_METHOD_ID = "transport_route.combine"
# This historical stable ID is deliberately retained; the catalogue label and
# source contract now identify the generic FLOW 2DH Galappatti route.
SUSPENSION_METHOD_ID = "suspension.van_rijn_exchange"
SETTLING_METHOD_ID = "settling.grain_size"
ADVECTION_METHOD_ID = "braids.accepted_water_flux_fct_v1"
REFERENCE_DEPTH_FLOOR_M = 0.01
FRACTION_BASIS_CODES = {"dry_mass": 1, "solid_volume": 2}
# Schema v3 admits only the reproducible constant-temperature slice.  A future
# spatial-field schema must bind a complete field/state digest, not merely the
# equilibrium fallback scalar.
TEMPERATURE_MODE_CODES = {"equilibrium_constant": 1}
SEDIMENT_ACTIVITY_CONTRACT = "h1_times_boolean_wet_mask_gt_sedthr"
WET_MASK_PROVIDER = "braids_h1_gt_dry_tolerance_maps_to_flow_kfs_v1"

# M2-B boundary/diffusion identities.  The boundary values, schedules and
# return times are per-boundary case data and are authenticated by the case
# deck fingerprint; this profile binds how those values are interpreted.
BOUNDARY_PRESCRIPTION_METHOD_ID = (
    "d3d4_flow2dh_incbcc_prescribed_constituent_concentration_v1"
)
BOUNDARY_CONCENTRATION_PROVIDER = (
    "authenticated_case_logical_boundary_fraction_series_uniform_face_"
    "expansion_v1"
)
BOUNDARY_RETURN_METHOD_ID = "d3d4_flow2dh_thatcher_harleman_return_time_v1"
BOUNDARY_RETURN_TIME_PROVIDER = (
    "authenticated_case_logical_boundary_fraction_return_time_uniform_face_"
    "expansion_v1"
)
BOUNDARY_SCHEDULE_IDENTITY_CONTRACT = (
    "values_times_and_cursors_bound_by_case_deck_and_checkpoint_fingerprints"
)
THATCHER_HARLEMAN_MEMORY_CONTRACT = (
    "checkpoint_per_face_per_fraction_last_outflow_concentration_kg_m3_and_"
    "remaining_return_clock_s"
)
THATCHER_HARLEMAN_CLOCK_CONTRACT = (
    "physical_transport_interval_not_morphological_clock"
)
SEDIMENT_CLASS_CODES = {"sand": 1, "mud": 2}
NEUBC_RESOLUTION_CONTRACT = (
    "NeuBcSand_for_sand_NeuBcMud_for_mud_after_Thatcher_Harleman"
)
BOUNDARY_CONCENTRATION_UNIT = "kg_dry_sediment_per_m3_mixture"
BOUNDARY_ADVECTIVE_LEDGER_SIGN = "positive_into_domain"

HORIZONTAL_DIFFUSIVITY_MODE_CODES = {"uniform_constant": 1}
HORIZONTAL_DIFFUSIVITY_PROVIDER = (
    "profile_uniform_dicuv_expanded_to_runtime_cell_state_v1"
)
HORIZONTAL_SCHMIDT_BRANCH_ID = (
    "d3d4_flow2dh_sigma_kmax1_hardcoded_schmidt_0p7_v1"
)
DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER = (
    "braids_accepted_reconstructed_hydraulic_face_trace_arithmetic_depth_"
    "times_width_with_porosity_resolved_in_face_geometry_v1"
)
DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER = (
    "adjacent_owner_h1_for_open_boundary_v1"
)
DIFFUSION_FACE_ACTIVITY_CONTRACT = (
    "owned_transport_face_and_strict_min_h1_endpoints_gt_DryFlc"
)
DIFFUSION_FACE_FLUX_SIGN = "positive_left_to_right"
BOUNDARY_DIFFUSIVE_LEDGER_SIGN = (
    "boundary_exterior_is_left_positive_into_domain"
)
BOUNDARY_TOTAL_LEDGER_SIGN = (
    "advective_plus_diffusive_positive_into_domain"
)


class SuspensionProfileError(ValueError):
    """Fail-closed suspension-profile error with a stable category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class Flow2DHSuspensionProfile:
    schema: int
    scope_id: str
    semantic_id: str
    sediment_catalog_id: str
    capacity_profile: CapacityRuntimeProfile
    transport_route_method_id: str
    suspension_method_id: str
    settling_method_id: str
    exchange_method_id: str
    exchange_semantic_id: str
    settling_operator_id: str
    settling_semantic_id: str
    advection_method_id: str
    fraction_basis: str
    fraction_basis_code: int
    fac_tsd: float
    sediment_threshold_m: float
    reference_depth_floor_m: float
    settling_form: int
    settling_fac_dss: tuple[float, ...]
    sediment_d50_m: tuple[float, ...]
    sediment_density_kg_m3: tuple[float, ...]
    settling_reference_concentration_kg_m3: float
    water_density_kg_m3: float
    temperature_mode: str
    temperature_mode_code: int
    temperature_value_c: float
    suspended_calibration_factor: float
    gravity_m_s2: float
    sediment_activity_contract: str
    wet_mask_provider: str
    open_boundary_operator_id: str
    open_boundary_semantic_id: str
    boundary_prescription_method_id: str
    boundary_concentration_provider: str
    boundary_return_method_id: str
    boundary_return_time_provider: str
    boundary_schedule_identity_contract: str
    thatcher_harleman_memory_contract: str
    thatcher_harleman_clock_contract: str
    sediment_fraction_classes: tuple[str, ...]
    sediment_fraction_class_codes: tuple[int, ...]
    neubc_sand: bool
    neubc_mud: bool
    neumann_boundary_by_fraction: tuple[bool, ...]
    neubc_resolution_contract: str
    boundary_concentration_unit: str
    boundary_advective_ledger_sign: str
    horizontal_diffusion_operator_id: str
    horizontal_diffusion_semantic_id: str
    horizontal_diffusivity_provider: str
    horizontal_diffusivity_mode: str
    horizontal_diffusivity_mode_code: int
    horizontal_diffusivity_m2_s: float
    horizontal_schmidt_branch_id: str
    horizontal_schmidt_value: float
    diffusion_dry_threshold_m: float
    diffusion_face_exchange_area_provider: str
    diffusion_boundary_exterior_depth_provider: str
    diffusion_face_activity_contract: str
    diffusion_face_flux_sign: str
    boundary_diffusive_ledger_sign: str
    boundary_total_ledger_sign: str
    fingerprint: str

    def as_dict(self) -> dict:
        return suspension_profile_document(self)


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SuspensionProfileError(
            "Suspension profile is not canonical JSON", category="invalid_configuration"
        ) from error


def _method_semantics(method_id: str) -> dict:
    row = method(method_id)
    return {
        "method_id": row.method_id,
        "component": row.component,
        "reference_selector": row.reference_selector,
        "legacy_code": row.legacy_code,
        "source_id": row.source_id,
        "output_contract": row.output_contract,
    }


def suspension_profile_semantic_document() -> dict:
    return {
        "schema": SUSPENSION_PROFILE_SCHEMA,
        "scope_id": SUSPENSION_PROFILE_SCOPE_ID,
        "sediment_catalog_id": semantic_catalog_id(),
        "capacity_profile_semantic_id": capacity_profile_semantic_id(),
        "physical_methods": [
            _method_semantics(TRANSPORT_ROUTE_METHOD_ID),
            _method_semantics(SUSPENSION_METHOD_ID),
            _method_semantics(SETTLING_METHOD_ID),
        ],
        "exchange_method_id": FLOW2DH_EXCHANGE_ID,
        "exchange_semantic_id": FLOW2DH_EXCHANGE_SEMANTIC_ID,
        "exchange_schema": FLOW2DH_EXCHANGE_SCHEMA,
        "settling_operator_id": FLOW2DH_SETTLING_ID,
        "settling_semantic_id": FLOW2DH_SETTLING_SEMANTIC_ID,
        "settling_schema": FLOW2DH_SETTLING_SCHEMA,
        "advection_method_id": ADVECTION_METHOD_ID,
        "open_boundary_operator_id": FLOW2DH_OPEN_BOUNDARY_ID,
        "open_boundary_semantic_id": FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
        "open_boundary_schema": FLOW2DH_OPEN_BOUNDARY_SCHEMA,
        "boundary_prescription_method_id": BOUNDARY_PRESCRIPTION_METHOD_ID,
        "boundary_concentration_provider": BOUNDARY_CONCENTRATION_PROVIDER,
        "boundary_return_method_id": BOUNDARY_RETURN_METHOD_ID,
        "boundary_return_time_provider": BOUNDARY_RETURN_TIME_PROVIDER,
        "boundary_schedule_identity_contract": BOUNDARY_SCHEDULE_IDENTITY_CONTRACT,
        "thatcher_harleman_memory_contract": THATCHER_HARLEMAN_MEMORY_CONTRACT,
        "thatcher_harleman_clock_contract": THATCHER_HARLEMAN_CLOCK_CONTRACT,
        "sediment_class_codes": SEDIMENT_CLASS_CODES,
        "neubc_resolution_contract": NEUBC_RESOLUTION_CONTRACT,
        "boundary_concentration_unit": BOUNDARY_CONCENTRATION_UNIT,
        "boundary_advective_ledger_sign": BOUNDARY_ADVECTIVE_LEDGER_SIGN,
        "horizontal_diffusion_operator_id": FLOW2DH_HORIZONTAL_DIFFUSION_ID,
        "horizontal_diffusion_semantic_id": (
            FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID
        ),
        "horizontal_diffusion_schema": FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA,
        "horizontal_diffusivity_mode_codes": HORIZONTAL_DIFFUSIVITY_MODE_CODES,
        "horizontal_diffusivity_provider": HORIZONTAL_DIFFUSIVITY_PROVIDER,
        "horizontal_schmidt_branch_id": HORIZONTAL_SCHMIDT_BRANCH_ID,
        "horizontal_schmidt_value": FLOW2DH_HORIZONTAL_SCHMIDT,
        "diffusion_face_exchange_area_provider": (
            DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER
        ),
        "diffusion_boundary_exterior_depth_provider": (
            DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER
        ),
        "diffusion_face_activity_contract": DIFFUSION_FACE_ACTIVITY_CONTRACT,
        "diffusion_face_flux_sign": DIFFUSION_FACE_FLUX_SIGN,
        "boundary_diffusive_ledger_sign": BOUNDARY_DIFFUSIVE_LEDGER_SIGN,
        "boundary_total_ledger_sign": BOUNDARY_TOTAL_LEDGER_SIGN,
        "fraction_basis_codes": FRACTION_BASIS_CODES,
        "temperature_mode_codes": TEMPERATURE_MODE_CODES,
        "reference_depth_floor_m": REFERENCE_DEPTH_FLOOR_M,
        "sediment_activity_contract": SEDIMENT_ACTIVITY_CONTRACT,
        "wet_mask_provider": WET_MASK_PROVIDER,
        "controls": [
            "FacTsd", "SedThr", "IFractions", "SettleFrm", "FacDSS",
            "Cref", "SedD50", "RhoSol", "RhoWater", "Temperature",
            "Sus", "Gravity", "NeuBcSand", "NeuBcMud", "dicuv", "DryFlc",
        ],
    }


def suspension_profile_semantic_id() -> str:
    return hashlib.sha256(
        _canonical(suspension_profile_semantic_document()).encode("ascii")
    ).hexdigest()


def _payload(profile: Flow2DHSuspensionProfile) -> dict:
    return {
        "schema": profile.schema,
        "scope_id": profile.scope_id,
        "semantic_id": profile.semantic_id,
        "sediment_catalog_id": profile.sediment_catalog_id,
        "capacity_profile": capacity_profile_document(profile.capacity_profile),
        "transport_route_method_id": profile.transport_route_method_id,
        "suspension_method_id": profile.suspension_method_id,
        "settling_method_id": profile.settling_method_id,
        "exchange_method_id": profile.exchange_method_id,
        "exchange_semantic_id": profile.exchange_semantic_id,
        "settling_operator_id": profile.settling_operator_id,
        "settling_semantic_id": profile.settling_semantic_id,
        "advection_method_id": profile.advection_method_id,
        "fraction_basis": profile.fraction_basis,
        "fraction_basis_code": profile.fraction_basis_code,
        "fac_tsd": profile.fac_tsd,
        "sediment_threshold_m": profile.sediment_threshold_m,
        "reference_depth_floor_m": profile.reference_depth_floor_m,
        "settling_form": profile.settling_form,
        "settling_fac_dss": list(profile.settling_fac_dss),
        "sediment_d50_m": list(profile.sediment_d50_m),
        "sediment_density_kg_m3": list(profile.sediment_density_kg_m3),
        "settling_reference_concentration_kg_m3":
            profile.settling_reference_concentration_kg_m3,
        "water_density_kg_m3": profile.water_density_kg_m3,
        "temperature_mode": profile.temperature_mode,
        "temperature_mode_code": profile.temperature_mode_code,
        "temperature_value_c": profile.temperature_value_c,
        "suspended_calibration_factor": profile.suspended_calibration_factor,
        "gravity_m_s2": profile.gravity_m_s2,
        "sediment_activity_contract": profile.sediment_activity_contract,
        "wet_mask_provider": profile.wet_mask_provider,
        "open_boundary_operator_id": profile.open_boundary_operator_id,
        "open_boundary_semantic_id": profile.open_boundary_semantic_id,
        "boundary_prescription_method_id": profile.boundary_prescription_method_id,
        "boundary_concentration_provider": profile.boundary_concentration_provider,
        "boundary_return_method_id": profile.boundary_return_method_id,
        "boundary_return_time_provider": profile.boundary_return_time_provider,
        "boundary_schedule_identity_contract": (
            profile.boundary_schedule_identity_contract
        ),
        "thatcher_harleman_memory_contract": (
            profile.thatcher_harleman_memory_contract
        ),
        "thatcher_harleman_clock_contract": profile.thatcher_harleman_clock_contract,
        "sediment_fraction_classes": list(profile.sediment_fraction_classes),
        "sediment_fraction_class_codes": list(profile.sediment_fraction_class_codes),
        "neubc_sand": profile.neubc_sand,
        "neubc_mud": profile.neubc_mud,
        "neumann_boundary_by_fraction": list(profile.neumann_boundary_by_fraction),
        "neubc_resolution_contract": profile.neubc_resolution_contract,
        "boundary_concentration_unit": profile.boundary_concentration_unit,
        "boundary_advective_ledger_sign": profile.boundary_advective_ledger_sign,
        "horizontal_diffusion_operator_id": profile.horizontal_diffusion_operator_id,
        "horizontal_diffusion_semantic_id": profile.horizontal_diffusion_semantic_id,
        "horizontal_diffusivity_provider": profile.horizontal_diffusivity_provider,
        "horizontal_diffusivity_mode": profile.horizontal_diffusivity_mode,
        "horizontal_diffusivity_mode_code": (
            profile.horizontal_diffusivity_mode_code
        ),
        "horizontal_diffusivity_m2_s": profile.horizontal_diffusivity_m2_s,
        "horizontal_schmidt_branch_id": profile.horizontal_schmidt_branch_id,
        "horizontal_schmidt_value": profile.horizontal_schmidt_value,
        "diffusion_dry_threshold_m": profile.diffusion_dry_threshold_m,
        "diffusion_face_exchange_area_provider": (
            profile.diffusion_face_exchange_area_provider
        ),
        "diffusion_boundary_exterior_depth_provider": (
            profile.diffusion_boundary_exterior_depth_provider
        ),
        "diffusion_face_activity_contract": profile.diffusion_face_activity_contract,
        "diffusion_face_flux_sign": profile.diffusion_face_flux_sign,
        "boundary_diffusive_ledger_sign": profile.boundary_diffusive_ledger_sign,
        "boundary_total_ledger_sign": profile.boundary_total_ledger_sign,
    }


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest()


def resolve_suspension_profile(
    capacity_profile: CapacityRuntimeProfile,
    *,
    fraction_basis: str,
    settling_fac_dss,
    sediment_d50_m,
    sediment_density_kg_m3,
    fac_tsd: float = 1.0,
    sediment_threshold_m: float = 0.5,
    settling_reference_concentration_kg_m3: float = 1.0e4,
    water_density_kg_m3: float = 1000.0,
    temperature_mode: str = "equilibrium_constant",
    temperature_value_c: float = 20.0,
    suspended_calibration_factor: float = 1.0,
    gravity_m_s2: float = 9.81,
    neubc_sand: bool = True,
    neubc_mud: bool = False,
    horizontal_diffusivity_mode: str = "uniform_constant",
    horizontal_diffusivity_m2_s: float = 0.0,
    diffusion_dry_threshold_m: float = 0.0,
) -> Flow2DHSuspensionProfile:
    """Resolve the generic FLOW 2DH ``TRA_COMBINE`` physical contract.

    Backend admission is intentionally not decided here.  CPU/GPU runtime
    selectors must separately prove that this authenticated profile is
    connected to their conservative state and reject it otherwise.
    """

    try:
        capacity_profile_document(capacity_profile)
    except CapacityProfileError as error:
        raise SuspensionProfileError(
            str(error), category="capacity_identity_mismatch"
        ) from error
    if fraction_basis not in FRACTION_BASIS_CODES:
        raise SuspensionProfileError(
            "IFractions basis must be 'dry_mass' or 'solid_volume'",
            category="invalid_configuration",
        )
    if (isinstance(fac_tsd, bool) or not isinstance(fac_tsd, (int, float)) or
            not math.isfinite(float(fac_tsd)) or float(fac_tsd) <= 0.0):
        raise SuspensionProfileError(
            "FacTsd must be finite and positive", category="invalid_configuration"
        )
    if (isinstance(sediment_threshold_m, bool) or
            not isinstance(sediment_threshold_m, (int, float)) or
            not math.isfinite(float(sediment_threshold_m)) or
            float(sediment_threshold_m) < 0.0):
        raise SuspensionProfileError(
            "SedThr must be finite and nonnegative", category="invalid_configuration"
        )
    if not isinstance(neubc_sand, bool) or not isinstance(neubc_mud, bool):
        raise SuspensionProfileError(
            "NeuBcSand and NeuBcMud must be Boolean",
            category="invalid_configuration",
        )
    if horizontal_diffusivity_mode not in HORIZONTAL_DIFFUSIVITY_MODE_CODES:
        if horizontal_diffusivity_mode in ("runtime_field", "file_field"):
            raise SuspensionProfileError(
                "Spatial dicuv fields require an authenticated field/state identity "
                "and are not admitted by suspension schema v3",
                category="unsupported_configuration",
            )
        raise SuspensionProfileError(
            "Horizontal diffusivity mode must be 'uniform_constant'",
            category="invalid_configuration",
        )
    for name, value in (
        ("dicuv", horizontal_diffusivity_m2_s),
        ("DryFlc", diffusion_dry_threshold_m),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or float(value) < 0.0):
            raise SuspensionProfileError(
                f"{name} must be finite and nonnegative",
                category="invalid_configuration",
            )
    fraction_count = len(capacity_profile.fractions)
    # This profile is the audited noncohesive route.  Every current capacity
    # fraction therefore resolves through FLOW's sand flag.  NeuBcMud is still
    # authenticated because its source default and future branch must never be
    # silently inferred from the sand setting.
    sediment_fraction_classes = tuple("sand" for _ in range(fraction_count))
    sediment_fraction_class_codes = tuple(
        SEDIMENT_CLASS_CODES[value] for value in sediment_fraction_classes
    )
    neumann_boundary_by_fraction = tuple(
        neubc_sand if value == "sand" else neubc_mud
        for value in sediment_fraction_classes
    )
    resolved_fraction_values = {}
    for name, raw in (
        ("FacDSS", settling_fac_dss),
        ("SedD50", sediment_d50_m),
        ("RhoSol", sediment_density_kg_m3),
    ):
        if isinstance(raw, (str, bytes)):
            values = ()
        else:
            try:
                values = tuple(raw)
            except TypeError:
                values = ()
        if len(values) != fraction_count:
            raise SuspensionProfileError(
                f"{name} must provide exactly one value per capacity fraction",
                category="invalid_configuration",
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in values
        ):
            raise SuspensionProfileError(
                f"{name} values must be finite and positive",
                category="invalid_configuration",
            )
        resolved_fraction_values[name] = tuple(float(value) for value in values)
    if temperature_mode not in TEMPERATURE_MODE_CODES:
        if temperature_mode == "spatial_field":
            raise SuspensionProfileError(
                "Spatial temperature fields require an authenticated field "
                "identity and are not admitted by suspension schema v3",
                category="unsupported_configuration",
            )
        raise SuspensionProfileError(
            "Temperature mode must be 'equilibrium_constant'",
            category="invalid_configuration",
        )
    scalar_rules = (
        ("Cref", settling_reference_concentration_kg_m3, True),
        ("water density", water_density_kg_m3, True),
        ("temperature", temperature_value_c, False),
        ("Sus", suspended_calibration_factor, False),
        ("gravity", gravity_m_s2, True),
    )
    for name, value, positive in scalar_rules:
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or
                (positive and float(value) <= 0.0) or
                (not positive and name == "Sus" and float(value) < 0.0)):
            qualifier = "positive" if positive else "within its physical domain"
            raise SuspensionProfileError(
                f"{name} must be finite and {qualifier}",
                category="invalid_configuration",
            )
    if float(temperature_value_c) <= -20.0:
        raise SuspensionProfileError(
            "Temperature must exceed -20 C for SettleFrm=2",
            category="invalid_configuration",
        )
    if any(
        density <= float(water_density_kg_m3)
        for density in resolved_fraction_values["RhoSol"]
    ):
        raise SuspensionProfileError(
            "Each RhoSol must exceed the resolved water density",
            category="invalid_configuration",
        )
    placeholder = Flow2DHSuspensionProfile(
        schema=SUSPENSION_PROFILE_SCHEMA,
        scope_id=SUSPENSION_PROFILE_SCOPE_ID,
        semantic_id=suspension_profile_semantic_id(),
        sediment_catalog_id=semantic_catalog_id(),
        capacity_profile=capacity_profile,
        transport_route_method_id=TRANSPORT_ROUTE_METHOD_ID,
        suspension_method_id=SUSPENSION_METHOD_ID,
        settling_method_id=SETTLING_METHOD_ID,
        exchange_method_id=FLOW2DH_EXCHANGE_ID,
        exchange_semantic_id=FLOW2DH_EXCHANGE_SEMANTIC_ID,
        settling_operator_id=FLOW2DH_SETTLING_ID,
        settling_semantic_id=FLOW2DH_SETTLING_SEMANTIC_ID,
        advection_method_id=ADVECTION_METHOD_ID,
        fraction_basis=fraction_basis,
        fraction_basis_code=FRACTION_BASIS_CODES[fraction_basis],
        fac_tsd=float(fac_tsd),
        sediment_threshold_m=float(sediment_threshold_m),
        reference_depth_floor_m=REFERENCE_DEPTH_FLOOR_M,
        settling_form=FLOW2DH_SETTLING_FORM,
        settling_fac_dss=resolved_fraction_values["FacDSS"],
        sediment_d50_m=resolved_fraction_values["SedD50"],
        sediment_density_kg_m3=resolved_fraction_values["RhoSol"],
        settling_reference_concentration_kg_m3=float(
            settling_reference_concentration_kg_m3
        ),
        water_density_kg_m3=float(water_density_kg_m3),
        temperature_mode=temperature_mode,
        temperature_mode_code=TEMPERATURE_MODE_CODES[temperature_mode],
        temperature_value_c=float(temperature_value_c),
        suspended_calibration_factor=float(suspended_calibration_factor),
        gravity_m_s2=float(gravity_m_s2),
        sediment_activity_contract=SEDIMENT_ACTIVITY_CONTRACT,
        wet_mask_provider=WET_MASK_PROVIDER,
        open_boundary_operator_id=FLOW2DH_OPEN_BOUNDARY_ID,
        open_boundary_semantic_id=FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
        boundary_prescription_method_id=BOUNDARY_PRESCRIPTION_METHOD_ID,
        boundary_concentration_provider=BOUNDARY_CONCENTRATION_PROVIDER,
        boundary_return_method_id=BOUNDARY_RETURN_METHOD_ID,
        boundary_return_time_provider=BOUNDARY_RETURN_TIME_PROVIDER,
        boundary_schedule_identity_contract=BOUNDARY_SCHEDULE_IDENTITY_CONTRACT,
        thatcher_harleman_memory_contract=THATCHER_HARLEMAN_MEMORY_CONTRACT,
        thatcher_harleman_clock_contract=THATCHER_HARLEMAN_CLOCK_CONTRACT,
        sediment_fraction_classes=sediment_fraction_classes,
        sediment_fraction_class_codes=sediment_fraction_class_codes,
        neubc_sand=neubc_sand,
        neubc_mud=neubc_mud,
        neumann_boundary_by_fraction=neumann_boundary_by_fraction,
        neubc_resolution_contract=NEUBC_RESOLUTION_CONTRACT,
        boundary_concentration_unit=BOUNDARY_CONCENTRATION_UNIT,
        boundary_advective_ledger_sign=BOUNDARY_ADVECTIVE_LEDGER_SIGN,
        horizontal_diffusion_operator_id=FLOW2DH_HORIZONTAL_DIFFUSION_ID,
        horizontal_diffusion_semantic_id=(
            FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID
        ),
        horizontal_diffusivity_provider=HORIZONTAL_DIFFUSIVITY_PROVIDER,
        horizontal_diffusivity_mode=horizontal_diffusivity_mode,
        horizontal_diffusivity_mode_code=(
            HORIZONTAL_DIFFUSIVITY_MODE_CODES[horizontal_diffusivity_mode]
        ),
        horizontal_diffusivity_m2_s=float(horizontal_diffusivity_m2_s),
        horizontal_schmidt_branch_id=HORIZONTAL_SCHMIDT_BRANCH_ID,
        horizontal_schmidt_value=FLOW2DH_HORIZONTAL_SCHMIDT,
        diffusion_dry_threshold_m=float(diffusion_dry_threshold_m),
        diffusion_face_exchange_area_provider=(
            DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER
        ),
        diffusion_boundary_exterior_depth_provider=(
            DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER
        ),
        diffusion_face_activity_contract=DIFFUSION_FACE_ACTIVITY_CONTRACT,
        diffusion_face_flux_sign=DIFFUSION_FACE_FLUX_SIGN,
        boundary_diffusive_ledger_sign=BOUNDARY_DIFFUSIVE_LEDGER_SIGN,
        boundary_total_ledger_sign=BOUNDARY_TOTAL_LEDGER_SIGN,
        fingerprint="",
    )
    return Flow2DHSuspensionProfile(
        **{**placeholder.__dict__, "fingerprint": _fingerprint(_payload(placeholder))}
    )


def suspension_profile_document(profile: Flow2DHSuspensionProfile) -> dict:
    if not isinstance(profile, Flow2DHSuspensionProfile):
        raise SuspensionProfileError(
            "Expected Flow2DHSuspensionProfile", category="invalid_configuration"
        )
    if (profile.schema != SUSPENSION_PROFILE_SCHEMA or
            profile.scope_id != SUSPENSION_PROFILE_SCOPE_ID or
            profile.semantic_id != suspension_profile_semantic_id() or
            profile.sediment_catalog_id != semantic_catalog_id()):
        raise SuspensionProfileError(
            "Suspension profile semantic identity is stale",
            category="identity_mismatch",
        )
    rebuilt = resolve_suspension_profile(
        profile.capacity_profile,
        fraction_basis=profile.fraction_basis,
        settling_fac_dss=profile.settling_fac_dss,
        sediment_d50_m=profile.sediment_d50_m,
        sediment_density_kg_m3=profile.sediment_density_kg_m3,
        fac_tsd=profile.fac_tsd,
        sediment_threshold_m=profile.sediment_threshold_m,
        settling_reference_concentration_kg_m3=
            profile.settling_reference_concentration_kg_m3,
        water_density_kg_m3=profile.water_density_kg_m3,
        temperature_mode=profile.temperature_mode,
        temperature_value_c=profile.temperature_value_c,
        suspended_calibration_factor=profile.suspended_calibration_factor,
        gravity_m_s2=profile.gravity_m_s2,
        neubc_sand=profile.neubc_sand,
        neubc_mud=profile.neubc_mud,
        horizontal_diffusivity_mode=profile.horizontal_diffusivity_mode,
        horizontal_diffusivity_m2_s=profile.horizontal_diffusivity_m2_s,
        diffusion_dry_threshold_m=profile.diffusion_dry_threshold_m,
    )
    payload = _payload(profile)
    if _canonical(payload) != _canonical(_payload(rebuilt)):
        raise SuspensionProfileError(
            "Suspension profile contents do not match the resolved contract",
            category="identity_mismatch",
        )
    expected = _fingerprint(payload)
    if profile.fingerprint != expected:
        raise SuspensionProfileError(
            "Suspension profile fingerprint does not match its contents",
            category="identity_mismatch",
        )
    result = dict(payload)
    result["fingerprint"] = expected
    return json.loads(_canonical(result))


def suspension_profile_json(
    profile: Flow2DHSuspensionProfile, *, indent: int | None = 2
) -> str:
    document = suspension_profile_document(profile)
    return json.dumps(document, sort_keys=True, indent=indent,
                      ensure_ascii=True, allow_nan=False) + ("\n" if indent is not None else "")


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SuspensionProfileError(
                f"Duplicate JSON key: {key!r}", category="invalid_serialization"
            )
        result[key] = value
    return result


def load_suspension_profile(
    document: str | bytes | Mapping[str, object],
) -> Flow2DHSuspensionProfile:
    if isinstance(document, (str, bytes)):
        try:
            raw = json.loads(document, object_pairs_hook=_strict_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 SuspensionProfileError(
                                     f"Non-finite JSON number: {value}",
                                     category="invalid_serialization")))
        except SuspensionProfileError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise SuspensionProfileError(
                "Invalid suspension-profile JSON", category="invalid_serialization"
            ) from error
    elif isinstance(document, Mapping):
        try:
            raw = json.loads(_canonical(document))
        except SuspensionProfileError as error:
            raise SuspensionProfileError(
                str(error), category="invalid_serialization"
            ) from error
    else:
        raise SuspensionProfileError(
            "Suspension profile must be JSON text or a mapping",
            category="invalid_serialization",
        )
    if not isinstance(raw, dict):
        raise SuspensionProfileError(
            "Suspension profile must be a JSON object", category="invalid_serialization"
        )
    try:
        capacity = load_capacity_profile(raw["capacity_profile"])
        rebuilt = resolve_suspension_profile(
            capacity,
            fraction_basis=raw["fraction_basis"],
            settling_fac_dss=raw["settling_fac_dss"],
            sediment_d50_m=raw["sediment_d50_m"],
            sediment_density_kg_m3=raw["sediment_density_kg_m3"],
            fac_tsd=raw["fac_tsd"],
            sediment_threshold_m=raw["sediment_threshold_m"],
            settling_reference_concentration_kg_m3=
                raw["settling_reference_concentration_kg_m3"],
            water_density_kg_m3=raw["water_density_kg_m3"],
            temperature_mode=raw["temperature_mode"],
            temperature_value_c=raw["temperature_value_c"],
            suspended_calibration_factor=raw["suspended_calibration_factor"],
            gravity_m_s2=raw["gravity_m_s2"],
            neubc_sand=raw["neubc_sand"],
            neubc_mud=raw["neubc_mud"],
            horizontal_diffusivity_mode=raw["horizontal_diffusivity_mode"],
            horizontal_diffusivity_m2_s=raw["horizontal_diffusivity_m2_s"],
            diffusion_dry_threshold_m=raw["diffusion_dry_threshold_m"],
        )
    except (KeyError, TypeError, CapacityProfileError) as error:
        raise SuspensionProfileError(
            "Malformed suspension-profile document", category="invalid_serialization"
        ) from error
    expected = suspension_profile_document(rebuilt)
    if _canonical(raw) != _canonical(expected):
        category = "identity_mismatch" if (
            raw.get("fingerprint") != expected["fingerprint"] or
            raw.get("semantic_id") != expected["semantic_id"] or
            raw.get("sediment_catalog_id") != expected["sediment_catalog_id"]
        ) else "invalid_serialization"
        raise SuspensionProfileError(
            "Suspension profile is stale, forged, or non-canonical", category=category
        )
    return rebuilt


__all__ = [
    "ADVECTION_METHOD_ID",
    "BOUNDARY_ADVECTIVE_LEDGER_SIGN",
    "BOUNDARY_CONCENTRATION_PROVIDER",
    "BOUNDARY_CONCENTRATION_UNIT",
    "BOUNDARY_DIFFUSIVE_LEDGER_SIGN",
    "BOUNDARY_PRESCRIPTION_METHOD_ID",
    "BOUNDARY_RETURN_METHOD_ID",
    "BOUNDARY_RETURN_TIME_PROVIDER",
    "BOUNDARY_SCHEDULE_IDENTITY_CONTRACT",
    "BOUNDARY_TOTAL_LEDGER_SIGN",
    "DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER",
    "DIFFUSION_FACE_ACTIVITY_CONTRACT",
    "DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER",
    "DIFFUSION_FACE_FLUX_SIGN",
    "FRACTION_BASIS_CODES",
    "HORIZONTAL_DIFFUSIVITY_MODE_CODES",
    "HORIZONTAL_DIFFUSIVITY_PROVIDER",
    "HORIZONTAL_SCHMIDT_BRANCH_ID",
    "NEUBC_RESOLUTION_CONTRACT",
    "SEDIMENT_CLASS_CODES",
    "SEDIMENT_ACTIVITY_CONTRACT",
    "THATCHER_HARLEMAN_CLOCK_CONTRACT",
    "THATCHER_HARLEMAN_MEMORY_CONTRACT",
    "Flow2DHSuspensionProfile",
    "REFERENCE_DEPTH_FLOOR_M",
    "SETTLING_METHOD_ID",
    "SUSPENSION_METHOD_ID",
    "SUSPENSION_PROFILE_SCHEMA",
    "SUSPENSION_PROFILE_SCOPE_ID",
    "TEMPERATURE_MODE_CODES",
    "SuspensionProfileError",
    "TRANSPORT_ROUTE_METHOD_ID",
    "WET_MASK_PROVIDER",
    "load_suspension_profile",
    "resolve_suspension_profile",
    "suspension_profile_document",
    "suspension_profile_json",
    "suspension_profile_semantic_document",
    "suspension_profile_semantic_id",
]
