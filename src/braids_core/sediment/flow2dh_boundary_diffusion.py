"""Delft3D4 FLOW 2DH suspended-boundary and diffusion oracles.

The routines in this module record two source-closed physical contracts from
the locked, unmodified FLOW reference:

* open-boundary concentration selection, including ``NeuBcSand`` and the
  stateful Thatcher--Harleman reversal rule; and
* the horizontal scalar-diffusion face law used by the 2DH ADI solver.

They deliberately do not reproduce FLOW's ADI constituent solver.  BRAIDS
uses accepted hydraulic face fluxes and a conservative FCT advection route;
the boundary result therefore reports the explicit FCT mapping separately
from the source-exact concentration selection.  Likewise, the diffusion
oracle returns one shared face mass, suitable for equation-level CPU/GPU
comparisons without pretending that an explicit update is FLOW's ADI method.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .passive import TransportError


FLOW2DH_OPEN_BOUNDARY_ID = "d3d4_flow2dh_suspended_open_boundary_v1"
FLOW2DH_HORIZONTAL_DIFFUSION_ID = "d3d4_flow2dh_horizontal_scalar_diffusion_v1"
FLOW2DH_HORIZONTAL_SCHMIDT = 0.7

FLOW2DH_OPEN_BOUNDARY_SCHEMA = {
    "method": FLOW2DH_OPEN_BOUNDARY_ID,
    "prescribed_value": "rbnd_from_incbcc_time_space_vertical_interpolation",
    "thatcher_harleman": {
        "positive_return_time": (
            "outflow_copies_adjacent_interior_and_stores_it; returning_inflow_"
            "cosine_blends_stored_outflow_to_current_prescription"
        ),
        "zero_return_time": "prescribed_rbnd",
        "negative_return_time": "adjacent_interior_for_all_flow_directions",
        "zero_normal_velocity": "reset_return_clock_then_use_current_prescription",
    },
    "neubc_override": (
        "after_Thatcher_Harleman_replace_boundary_with_max(0,adjacent_interior)"
    ),
    "neubc_meaning": "zero_normal_concentration_gradient_not_capacity_equilibrium",
    "adjacent_interior_state": "pre_transport_r0",
    "direction_input": (
        "caller_resolved_positive_inward; FLOW_axis_boundaries_use_directed_component_"
        "and_oblique_boundaries_choose_the_larger_absolute_directed_u_or_v_component;_"
        "BRAIDS_maps_accepted_face_normal_discharge"
    ),
    "defaults": {"sand": True, "mud": False},
    "braids_advection_mapping": (
        "accepted_inward_water_volume_times_resolved_boundary_concentration_on_"
        "inflow; interior_upwind_concentration_on_outflow"
    ),
    "concentration_unit": "kg_dry_sediment_per_m3_mixture",
    "ledger_sign": "positive_into_domain",
    "source_order": [
        "incbcc:prescribed_rbnd",
        "thahbc:flow_reversal_and_return_time",
        "difu:Dirichlet_boundary_rows",
        "difu:NeuBcSand_or_NeuBcMud_adjacent_interior_override",
        "BRAIDS:accepted_flux_FCT_boundary_mass",
    ],
}
FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA = {
    "method": FLOW2DH_HORIZONTAL_DIFFUSION_ID,
    "cell_diffusivity": "dicuv_m2_s",
    "face_diffusivity": "0.5*(dicuv_left+dicuv_right)/0.7",
    "face_mass_rate": (
        "exchange_area_m2*face_diffusivity_m2_s*"
        "(concentration_left-concentration_right)/centre_distance_m"
    ),
    "face_exchange_area_2dh": "hydraulic_face_depth_m*face_width_m",
    "activity": "transport_face_active_and_min(endpoint_h1)>DryFlc",
    "rate_sign": "positive_left_to_right",
    "accepted_flux_concentration_state": (
        "r1_in_the_implicit_sweep_direction_and_r0_in_the_explicit_transverse_direction"
    ),
    "ownership_mask": (
        "DIFU_applies_max(0,2-abs(kcs))_per_matrix_endpoint;_the_oracle_requires_"
        "caller_resolved_transport_face_active_and_owner"
    ),
    "source_time_discretisation": (
        "ADI_half_steps; implicit_in_sweep_direction_and_explicit_transverse;_"
        "directions_reverse_between_stage1_and_stage2"
    ),
    "oracle_scope": "shared_face_constitutive_mass_only_not_ADI_time_update",
    "source_order": [
        "difu:old_volume_mass_initialisation",
        "difu:horizontal_advection_assembly",
        "difu:horizontal_diffusion_assembly",
        "difu:settling_and_open_boundary_rows",
        "difu:explicit_sources_and_implicit_sinks",
        "difu:ADI_solve",
    ],
}


def _semantic_id(schema: dict) -> str:
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID = _semantic_id(FLOW2DH_OPEN_BOUNDARY_SCHEMA)
FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID = _semantic_id(
    FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA
)


@dataclass(frozen=True)
class Flow2DHBoundaryMemory:
    """Per-face Thatcher--Harleman state required for exact restart."""

    last_outflow_concentration_kg_m3: np.ndarray
    return_clock_s: np.ndarray


@dataclass(frozen=True)
class Flow2DHBoundaryResult:
    """Resolved boundary concentration and BRAIDS accepted-flux ledger."""

    boundary_concentration_kg_m3: np.ndarray
    advective_concentration_kg_m3: np.ndarray
    advective_mass_rate_in_kg_s: np.ndarray
    advective_mass_in_kg: np.ndarray
    boundary_mass_in_kg: float
    inflow_mask: np.ndarray
    outflow_mask: np.ndarray
    stagnant_mask: np.ndarray
    neumann_mask: np.ndarray
    memory: Flow2DHBoundaryMemory
    method_id: str = FLOW2DH_OPEN_BOUNDARY_ID
    semantic_id: str = FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID


@dataclass(frozen=True)
class Flow2DHDiffusiveFaceResult:
    """One shared horizontal diffusive face flux and its activity diagnostics."""

    effective_diffusivity_m2_s: np.ndarray
    mass_rate_left_to_right_kg_s: np.ndarray
    mass_left_to_right_kg: np.ndarray
    active_mask: np.ndarray
    method_id: str = FLOW2DH_HORIZONTAL_DIFFUSION_ID
    semantic_id: str = FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID


def _finite(name: str, value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise TransportError(f"{name} must be finite")
    return result


def _boolean(name: str, value) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "b":
        raise TransportError(f"{name} must be Boolean")
    return result


def _broadcast(*named_values: tuple[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    try:
        return tuple(np.broadcast_arrays(*(value for _, value in named_values)))
    except ValueError as error:
        names = ", ".join(name for name, _ in named_values)
        raise TransportError(f"{names} must be broadcast-compatible") from error


def resolve_flow2dh_open_boundary(
    prescribed_concentration_kg_m3,
    adjacent_interior_concentration_kg_m3,
    inward_water_discharge_m3_s,
    *,
    dt_s: float,
    neumann=True,
    return_time_s=0.0,
    previous_memory: Flow2DHBoundaryMemory | None = None,
) -> Flow2DHBoundaryResult:
    """Resolve FLOW's open-boundary concentration and a BRAIDS mass ledger.

    Positive water discharge is **into** the model.  FLOW's ``thahbc`` uses
    the opposite velocity convention, so negative values here are outflow.
    ``dt_s`` is both the accepted BRAIDS transport interval and the physical
    decrement corresponding to FLOW's ``2*hdt`` Thatcher--Harleman clock.

    ``NeuBcSand`` is represented by ``neumann=True``.  Despite the historical
    internal name ``eqmbcsand``, this does not evaluate a transport-capacity
    formula: it copies the nonnegative adjacent interior concentration after
    the prescribed/return-time rule.  The returned advective mass is the
    explicit BRAIDS FCT mapping.  It is not a reproduction of FLOW's ADI
    advection stencil.
    """

    if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
            not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0):
        raise TransportError("dt_s must be finite and positive")
    prescribed = _finite(
        "prescribed_concentration_kg_m3", prescribed_concentration_kg_m3
    )
    interior = _finite(
        "adjacent_interior_concentration_kg_m3",
        adjacent_interior_concentration_kg_m3,
    )
    inward_q = _finite("inward_water_discharge_m3_s", inward_water_discharge_m3_s)
    return_time = _finite("return_time_s", return_time_s)
    neumann_mask = _boolean("neumann", neumann)
    prescribed, interior, inward_q, return_time, neumann_mask = _broadcast(
        ("prescribed_concentration_kg_m3", prescribed),
        ("adjacent_interior_concentration_kg_m3", interior),
        ("inward_water_discharge_m3_s", inward_q),
        ("return_time_s", return_time),
        ("neumann", neumann_mask),
    )
    shape = prescribed.shape
    if np.any(prescribed < 0.0) or np.any(interior < 0.0):
        raise TransportError("sediment boundary concentrations must be nonnegative")

    if previous_memory is None:
        last_outflow = np.zeros(shape, dtype=float)
        return_clock = np.zeros(shape, dtype=float)
    else:
        last_outflow = _finite(
            "previous last_outflow_concentration_kg_m3",
            previous_memory.last_outflow_concentration_kg_m3,
        )
        return_clock = _finite(
            "previous return_clock_s", previous_memory.return_clock_s
        )
        last_outflow, return_clock = _broadcast(
            ("previous last_outflow_concentration_kg_m3", last_outflow),
            ("previous return_clock_s", return_clock),
        )
        try:
            last_outflow = np.broadcast_to(last_outflow, shape)
            return_clock = np.broadcast_to(return_clock, shape)
        except ValueError as error:
            raise TransportError("previous boundary memory must broadcast to faces") from error
        if np.any(last_outflow < 0.0) or np.any(return_clock < 0.0):
            raise TransportError("previous boundary memory must be nonnegative")

    inflow = inward_q > 0.0
    outflow = inward_q < 0.0
    stagnant = ~(inflow | outflow)
    positive_return = return_time > 0.0
    negative_return = return_time < 0.0
    clock_tolerance = 256.0 * np.finfo(float).eps * np.maximum(1.0, return_time)
    if np.any(positive_return & (return_clock > return_time + clock_tolerance)):
        raise TransportError(
            "previous return_clock_s exceeds the configured positive return_time_s"
        )
    # With an authenticated, fixed return-time profile, a nonpositive mode
    # can never create live Thatcher--Harleman clock state: T=0 stays on the
    # prescribed branch and T<0 clears the clock.  A nonzero restored value is
    # therefore corrupt/unreachable checkpoint state, not a state to repair.
    if np.any(~positive_return & (return_clock != 0.0)):
        raise TransportError(
            "previous return_clock_s must be zero for nonpositive return_time_s"
        )

    next_last = np.array(last_outflow, dtype=float, copy=True)
    next_clock = np.array(return_clock, dtype=float, copy=True)
    boundary = np.array(prescribed, dtype=float, copy=True)

    # A negative return time is FLOW's unconditional reflected/zero-gradient
    # boundary option, independent of the instantaneous flow direction.
    boundary[negative_return] = interior[negative_return]
    next_clock[negative_return] = 0.0

    th_outflow = positive_return & outflow
    next_last[th_outflow] = interior[th_outflow]
    next_clock[th_outflow] = return_time[th_outflow]
    boundary[th_outflow] = interior[th_outflow]

    th_inflow = positive_return & inflow
    next_clock[th_inflow] = np.maximum(
        return_clock[th_inflow] - float(dt_s), 0.0
    )
    ratio = np.zeros(shape, dtype=float)
    ratio[th_inflow] = next_clock[th_inflow] / return_time[th_inflow]
    cosine_weight = 0.5 * (np.cos(np.pi * ratio) + 1.0)
    boundary[th_inflow] = (
        last_outflow[th_inflow]
        + cosine_weight[th_inflow]
        * (prescribed[th_inflow] - last_outflow[th_inflow])
    )

    # FLOW resets the return clock when the normal boundary velocity is zero;
    # INCBCC's current prescribed value remains in rbnd for that invocation.
    th_stagnant = positive_return & stagnant
    next_clock[th_stagnant] = 0.0

    # DIFU applies NeuBcSand/NeuBcMud *after* rbnd and Thatcher--Harleman.
    boundary = np.where(neumann_mask, np.maximum(0.0, interior), boundary)

    # This is the conservative accepted-flux mapping used by BRAIDS.  On
    # outflow the interior state is upwind even when FLOW's Dirichlet rbnd is
    # prescribed.  Diffusive boundary exchange is intentionally separate.
    advective_concentration = np.where(inflow, boundary, interior)
    mass_rate_in = inward_q * advective_concentration
    mass_in = mass_rate_in * float(dt_s)
    return Flow2DHBoundaryResult(
        boundary_concentration_kg_m3=np.array(boundary, copy=True),
        advective_concentration_kg_m3=np.array(advective_concentration, copy=True),
        advective_mass_rate_in_kg_s=np.array(mass_rate_in, copy=True),
        advective_mass_in_kg=np.array(mass_in, copy=True),
        boundary_mass_in_kg=float(np.sum(mass_in)),
        inflow_mask=np.array(inflow, dtype=bool, copy=True),
        outflow_mask=np.array(outflow, dtype=bool, copy=True),
        stagnant_mask=np.array(stagnant, dtype=bool, copy=True),
        neumann_mask=np.array(neumann_mask, dtype=bool, copy=True),
        memory=Flow2DHBoundaryMemory(
            last_outflow_concentration_kg_m3=np.array(next_last, copy=True),
            return_clock_s=np.array(next_clock, copy=True),
        ),
    )


def flow2dh_horizontal_diffusive_face_mass(
    concentration_left_kg_m3,
    concentration_right_kg_m3,
    diffusivity_left_m2_s,
    diffusivity_right_m2_s,
    face_exchange_area_m2,
    centre_distance_m,
    depth_left_m,
    depth_right_m,
    *,
    dt_s: float,
    dry_threshold_m: float,
    transport_face_active=True,
) -> Flow2DHDiffusiveFaceResult:
    """Evaluate the locked FLOW 2DH horizontal diffusion law on shared faces.

    The returned sign is positive from ``left`` to ``right``.  For an exterior
    boundary face, place the exterior boundary state on the left and the
    interior cell on the right to make positive mass an inward-domain ledger.
    A ``NeuBcSand`` boundary supplies equal left/right concentrations and thus
    exactly zero diffusive mass.

    ``face_exchange_area_m2`` is FLOW's ``areau``/``areav`` quantity.  In 2DH
    Cartesian geometry it is hydraulic face depth times face width.  Requiring
    it explicitly avoids silently replacing FLOW's face depth with an
    arithmetic or harmonic reconstruction from cell depths.
    """

    for name, value, allow_zero in (
        ("dt_s", dt_s, False),
        ("dry_threshold_m", dry_threshold_m, True),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or
                (float(value) < 0.0 if allow_zero else float(value) <= 0.0)):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise TransportError(f"{name} must be finite and {qualifier}")

    left_c = _finite("concentration_left_kg_m3", concentration_left_kg_m3)
    right_c = _finite("concentration_right_kg_m3", concentration_right_kg_m3)
    left_d = _finite("diffusivity_left_m2_s", diffusivity_left_m2_s)
    right_d = _finite("diffusivity_right_m2_s", diffusivity_right_m2_s)
    area = _finite("face_exchange_area_m2", face_exchange_area_m2)
    distance = _finite("centre_distance_m", centre_distance_m)
    left_h = _finite("depth_left_m", depth_left_m)
    right_h = _finite("depth_right_m", depth_right_m)
    face_active = _boolean("transport_face_active", transport_face_active)
    (
        left_c, right_c, left_d, right_d, area, distance,
        left_h, right_h, face_active,
    ) = _broadcast(
        ("concentration_left_kg_m3", left_c),
        ("concentration_right_kg_m3", right_c),
        ("diffusivity_left_m2_s", left_d),
        ("diffusivity_right_m2_s", right_d),
        ("face_exchange_area_m2", area),
        ("centre_distance_m", distance),
        ("depth_left_m", left_h),
        ("depth_right_m", right_h),
        ("transport_face_active", face_active),
    )
    if np.any(left_d < 0.0) or np.any(right_d < 0.0):
        raise TransportError("horizontal diffusivity must be nonnegative")
    if np.any(area < 0.0):
        raise TransportError("face_exchange_area_m2 must be nonnegative")
    if np.any(distance <= 0.0):
        raise TransportError("centre_distance_m must be positive")
    if np.any(left_h < 0.0) or np.any(right_h < 0.0):
        raise TransportError("endpoint depths must be nonnegative")

    active = (
        face_active
        & (np.minimum(left_h, right_h) > float(dry_threshold_m))
    )
    effective = 0.5 * (left_d + right_d) / FLOW2DH_HORIZONTAL_SCHMIDT
    rate = np.where(
        active,
        area * effective * (left_c - right_c) / distance,
        0.0,
    )
    return Flow2DHDiffusiveFaceResult(
        effective_diffusivity_m2_s=np.array(effective, copy=True),
        mass_rate_left_to_right_kg_s=np.array(rate, copy=True),
        mass_left_to_right_kg=np.array(rate * float(dt_s), copy=True),
        active_mask=np.array(active, dtype=bool, copy=True),
    )


__all__ = [
    "FLOW2DH_HORIZONTAL_DIFFUSION_ID",
    "FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA",
    "FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID",
    "FLOW2DH_HORIZONTAL_SCHMIDT",
    "FLOW2DH_OPEN_BOUNDARY_ID",
    "FLOW2DH_OPEN_BOUNDARY_SCHEMA",
    "FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID",
    "Flow2DHBoundaryMemory",
    "Flow2DHBoundaryResult",
    "Flow2DHDiffusiveFaceResult",
    "flow2dh_horizontal_diffusive_face_mass",
    "resolve_flow2dh_open_boundary",
]
