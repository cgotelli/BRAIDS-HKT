"""Delft3D FLOW 2DH equilibrium-suspension exchange reference operator.

This module is the backend-neutral oracle for the current-only ``TRA_COMBINE``
route.  A transport formula supplies an equilibrium suspended transport rate;
FLOW converts that rate to a depth-averaged equilibrium concentration and the
Galappatti closure supplies explicit and implicit source terms.  The numerical
transport of suspended mass remains a separate conservative operation.

The source anchors are the locked, read-only Delft3D routines ``eqtran.f90``
and ``soursin_2d.f90``.  No Delft3D source is copied into or modified by this
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .passive import TransportError


FLOW2DH_EXCHANGE_ID = "d3d4_flow2dh_galappatti_v1"
FLOW2DH_EXCHANGE_SCHEMA = {
    "method": FLOW2DH_EXCHANGE_ID,
    "capacity_to_concentration": "ssus/(speed+1e-6)/h*surface_fraction",
    "adaptation": "FLOW_soursin_2d_Galappatti",
    "adaptation_factor_keyword": "FacTsd",
    "source_treatment": "mass_form_explicit_source_implicit_sink_backward_euler",
    "inventory": "dry_sediment_mass_per_bed_area",
    "exchange_ledger_sign": "positive_bed_to_suspended",
    "fraction_basis": "configured_FLOW_IFractions_basis",
    "sediment_activity": "new_depth_times_flow_wet_mask_gt_SedThr",
    "reference_depth_floor_m": 0.01,
    "finite_inventory_limiter": "braids_nonnegative_available_mass_v1_not_FLOW_red_soursin",
}
FLOW2DH_EXCHANGE_SEMANTIC_ID = hashlib.sha256(
    json.dumps(FLOW2DH_EXCHANGE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

_EPS = 1.0e-6


def _f32(value: float) -> float:
    """Promote a locked-source default-real literal to binary64."""

    return float(np.float32(value))


@dataclass(frozen=True)
class GalappattiTerms:
    """FLOW 2DH source/sink coefficients for a suspended fraction."""

    explicit_source_kg_m3_s: np.ndarray
    implicit_source_s_1: np.ndarray
    settling_sink_s_1: np.ndarray
    total_implicit_sink_s_1: np.ndarray
    adaptation_velocity_m_s: np.ndarray
    dimensionless_timescale: np.ndarray


@dataclass(frozen=True)
class Flow2DHExchangeResult:
    """Conservative water/bed exchange result for all fractions and cells."""

    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    equilibrium_concentration_kg_m3: np.ndarray
    terms: GalappattiTerms
    transferred_to_suspension_kg_m2: np.ndarray
    suspended_exchange_ledger_kg: np.ndarray
    bed_exchange_ledger_kg: np.ndarray
    mass_before_kg: np.ndarray
    mass_after_kg: np.ndarray
    balance_residual_kg: np.ndarray
    safety_limiter_correction_kg: np.ndarray
    safety_limiter_applied: bool
    sediment_active: np.ndarray
    method_id: str = FLOW2DH_EXCHANGE_ID
    semantic_id: str = FLOW2DH_EXCHANGE_SEMANTIC_ID


def _finite_array(name: str, value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if shape is not None and result.shape != shape:
        raise TransportError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise TransportError(f"{name} must be finite")
    return result


def equilibrium_concentration_from_capacity(
    suspended_transport_kg_m_s,
    speed_m_s,
    water_depth_m,
    surface_fraction,
) -> np.ndarray:
    """Convert FLOW's equilibrium suspended transport to concentration.

    ``suspended_transport_kg_m_s`` is the non-negative transport magnitude per
    unit width after the formula's calibration.  The result is dry sediment
    mass per water volume.  ``surface_fraction`` is FLOW's active surface
    fraction multiplier in the configured ``IFractions`` basis.  The first
    M2 slice uses one fraction (and therefore exactly one) so mass/volume basis
    conversion cannot be hidden.
    """

    transport = _finite_array("suspended_transport_kg_m_s", suspended_transport_kg_m_s)
    speed = _finite_array("speed_m_s", speed_m_s)
    depth = _finite_array("water_depth_m", water_depth_m)
    fraction = _finite_array("surface_fraction", surface_fraction)
    try:
        transport, speed, depth, fraction = np.broadcast_arrays(
            transport, speed, depth, fraction
        )
    except ValueError as error:
        raise TransportError("capacity-to-concentration inputs are not broadcast-compatible") from error
    if (np.any(transport < 0.0) or np.any(speed < 0.0) or
            np.any(depth <= 0.0) or np.any((fraction < 0.0) | (fraction > 1.0))):
        raise TransportError(
            "capacity, speed and surface fraction must be non-negative; depth must be positive"
        )
    return transport / (speed + _EPS) / depth * fraction


def galappatti_source_sink(
    equilibrium_concentration_kg_m3,
    speed_m_s,
    shear_velocity_m_s,
    depth_before_m,
    depth_after_m,
    settling_velocity_m_s,
    *,
    fac_tsd: float = 1.0,
    dimensionless_timescale=0.0,
) -> GalappattiTerms:
    """Evaluate the locked FLOW ``soursin_2d`` algebra in source order.

    A positive ``dimensionless_timescale`` is a formula-provided value.  Zero
    requests the Galappatti expression used by FLOW.  The bare coefficients in
    the reference routine are default-real literals; promoting their binary32
    values here prevents a misleading precision advantage in the oracle.
    """

    arrays = [
        _finite_array("equilibrium_concentration_kg_m3", equilibrium_concentration_kg_m3),
        _finite_array("speed_m_s", speed_m_s),
        _finite_array("shear_velocity_m_s", shear_velocity_m_s),
        _finite_array("depth_before_m", depth_before_m),
        _finite_array("depth_after_m", depth_after_m),
        _finite_array("settling_velocity_m_s", settling_velocity_m_s),
        _finite_array("dimensionless_timescale", dimensionless_timescale),
    ]
    try:
        equilibrium, speed, shear, h0, h1, settling, supplied = np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise TransportError("Galappatti inputs are not broadcast-compatible") from error
    if (not math.isfinite(fac_tsd) or fac_tsd <= 0.0 or
            np.any(equilibrium < 0.0) or np.any(speed < 0.0) or
            np.any(shear < 0.0) or np.any(settling < 0.0) or
            np.any(h0 <= 0.0) or np.any(h1 <= 0.0) or np.any(supplied < 0.0)):
        raise TransportError("Galappatti inputs are outside their physical domain")

    wsl = np.maximum(1.0e-3, settling)
    active = (speed > _EPS) & (shear > _EPS)
    computed = np.zeros_like(equilibrium)
    if np.any(active):
        u = np.minimum(shear[active] / speed[active], 0.15)
        w = np.where(shear[active] > wsl[active], wsl[active] / shear[active], 1.0)
        x = w
        x2 = x * x
        x3 = x2 * x
        exponent = (
            (_f32(1.547) - _f32(20.12) * u) * x3
            + (_f32(326.832) * u ** _f32(2.2047) - _f32(0.2)) * x2
            + (_f32(0.1385) * np.log(u) - _f32(6.4061)) * x
            + (_f32(0.5467) * u + _f32(2.1963))
        )
        computed[active] = x * np.exp(exponent)

    timescale = np.where(supplied > 0.0, supplied, computed)
    if np.any(active & (timescale <= 0.0)) or not np.isfinite(timescale).all():
        raise TransportError("Galappatti adaptation timescale is non-positive or non-finite")
    adaptation_velocity = np.zeros_like(equilibrium)
    adaptation_velocity[active] = (
        wsl[active] / (timescale[active] * fac_tsd)
    )
    explicit = np.zeros_like(equilibrium)
    implicit = np.zeros_like(equilibrium)
    explicit[active] = equilibrium[active] * adaptation_velocity[active] / h0[active]
    implicit[active] = (adaptation_velocity[active] - wsl[active]) / h1[active]
    settling_sink = wsl / h1
    total_sink = settling_sink + implicit
    if np.any(total_sink < -1024.0 * np.finfo(float).eps):
        raise TransportError("Galappatti total implicit sink became negative")
    total_sink = np.maximum(total_sink, 0.0)
    return GalappattiTerms(
        explicit_source_kg_m3_s=explicit,
        implicit_source_s_1=implicit,
        settling_sink_s_1=settling_sink,
        total_implicit_sink_s_1=total_sink,
        adaptation_velocity_m_s=adaptation_velocity,
        dimensionless_timescale=timescale,
    )


def advance_flow2dh_equilibrium_exchange(
    suspended_mass_kg_m2,
    bed_mass_kg_m2,
    suspended_transport_kg_m_s,
    surface_fraction,
    speed_m_s,
    shear_velocity_m_s,
    depth_before_m,
    depth_after_m,
    settling_velocity_m_s,
    cell_area_m2,
    *,
    dt_s: float,
    fac_tsd: float = 1.0,
    dimensionless_timescale=0.0,
    sediment_threshold_m: float = 0.5,
    minimum_reference_depth_m: float = 0.01,
    flow_wet_mask=True,
) -> Flow2DHExchangeResult:
    """Apply one conservative FLOW-style implicit exchange update.

    Fraction/cell inventories have shape ``(n_fraction, n_cell)``.  Hydraulic
    and area inputs have shape ``(n_cell,)``; fraction controls may be either
    ``(n_fraction,)`` or full fraction/cell arrays.  Exchange is disabled
    unless the new water depth exceeds FLOW's ``SedThr`` *and* the endpoint
    FLOW wet mask is true.  ``flow_wet_mask`` must be a Boolean scalar or one
    Boolean value per cell; integer masks are rejected so the provider
    contract cannot be weakened accidentally.  Inactive columns are untouched
    here; wet/dry mass transfer is a separate transport policy.
    """

    suspended = _finite_array("suspended_mass_kg_m2", suspended_mass_kg_m2)
    bed = _finite_array("bed_mass_kg_m2", bed_mass_kg_m2, suspended.shape)
    capacity = _finite_array("suspended_transport_kg_m_s", suspended_transport_kg_m_s,
                             suspended.shape)
    fraction = _finite_array("surface_fraction", surface_fraction)
    if suspended.ndim != 2:
        raise TransportError("suspended and bed inventories must have shape (fraction, cell)")
    ng, nc = suspended.shape
    h0 = _finite_array("depth_before_m", depth_before_m, (nc,))
    h1 = _finite_array("depth_after_m", depth_after_m, (nc,))
    speed = _finite_array("speed_m_s", speed_m_s, (nc,))
    shear = _finite_array("shear_velocity_m_s", shear_velocity_m_s, (nc,))
    area = _finite_array("cell_area_m2", cell_area_m2, (nc,))
    wet_mask = np.asarray(flow_wet_mask)
    if wet_mask.dtype.kind != "b":
        raise TransportError("flow_wet_mask must contain Boolean values")
    if wet_mask.ndim == 0:
        wet_mask = np.full(nc, bool(wet_mask), dtype=bool)
    elif wet_mask.shape != (nc,):
        raise TransportError(f"flow_wet_mask must be scalar or have shape {(nc,)}")

    def fraction_field(name: str, value) -> np.ndarray:
        array = _finite_array(name, value)
        if array.shape == (ng,):
            array = np.broadcast_to(array[:, None], (ng, nc))
        elif array.shape != (ng, nc):
            raise TransportError(f"{name} must have shape {(ng,)} or {(ng, nc)}")
        return array

    fraction = fraction_field("surface_fraction", fraction)
    settling = fraction_field("settling_velocity_m_s", settling_velocity_m_s)
    timescale = fraction_field("dimensionless_timescale", dimensionless_timescale) \
        if np.asarray(dimensionless_timescale).ndim else float(dimensionless_timescale)
    if (not math.isfinite(dt_s) or dt_s <= 0.0 or
            not math.isfinite(sediment_threshold_m) or sediment_threshold_m < 0.0 or
            not math.isfinite(minimum_reference_depth_m) or
            minimum_reference_depth_m <= 0.0 or
            np.any(suspended < 0.0) or np.any(bed < 0.0) or
            np.any(capacity < 0.0) or np.any((fraction < 0.0) | (fraction > 1.0)) or
            np.any(h0 < 0.0) or np.any(h1 < 0.0) or np.any(speed < 0.0) or
            np.any(shear < 0.0) or np.any(settling < 0.0) or np.any(area <= 0.0)):
        raise TransportError("FLOW 2DH exchange inputs are outside their physical domain")

    old_total = suspended + bed
    sediment_active = wet_mask & (h1 > sediment_threshold_m)
    safe_h0 = np.where(
        sediment_active, np.maximum(minimum_reference_depth_m, h0), 1.0
    )
    safe_h1 = np.where(
        sediment_active, np.maximum(minimum_reference_depth_m, h1), 1.0
    )
    equilibrium = equilibrium_concentration_from_capacity(
        capacity, speed[None, :], safe_h1[None, :], fraction
    )
    raw_terms = galappatti_source_sink(
        equilibrium,
        speed[None, :],
        shear[None, :],
        safe_h0[None, :],
        safe_h1[None, :],
        settling,
        fac_tsd=fac_tsd,
        dimensionless_timescale=timescale,
    )
    active2d = sediment_active[None, :]
    terms = GalappattiTerms(
        explicit_source_kg_m3_s=np.where(
            active2d, raw_terms.explicit_source_kg_m3_s, 0.0),
        implicit_source_s_1=np.where(
            active2d, raw_terms.implicit_source_s_1, 0.0),
        settling_sink_s_1=np.where(
            active2d, raw_terms.settling_sink_s_1, 0.0),
        total_implicit_sink_s_1=np.where(
            active2d, raw_terms.total_implicit_sink_s_1, 0.0),
        adaptation_velocity_m_s=np.where(
            active2d, raw_terms.adaptation_velocity_m_s, 0.0),
        dimensionless_timescale=np.where(
            active2d, raw_terms.dimensionless_timescale, 0.0),
    )
    # FLOW assembles its scalar equation in volume/mass form:
    #   (h1/dt + K*h1) C1 = h0*C0/dt + S*h0.
    # Evolving h*C directly prevents a changing water depth from spuriously
    # diluting or concentrating suspended inventory.
    candidate = (
        suspended + dt_s * h0[None, :] * terms.explicit_source_kg_m3_s
    ) / (1.0 + dt_s * terms.total_implicit_sink_s_1)
    candidate[:, ~sediment_active] = suspended[:, ~sediment_active]
    suspended_after = np.minimum(np.maximum(candidate, 0.0), old_total)
    limiter_correction = np.sum(
        np.abs(suspended_after - candidate) * area[None, :], axis=1
    )
    bed_after = old_total - suspended_after
    transfer = suspended_after - suspended
    water_ledger = np.sum(transfer * area[None, :], axis=1)
    bed_ledger = -water_ledger
    before = np.sum(old_total * area[None, :], axis=1)
    after = np.sum((suspended_after + bed_after) * area[None, :], axis=1)
    residual = after - before
    scale = np.maximum(1.0, np.maximum(np.abs(before), np.abs(after)))
    if np.any(np.abs(residual) > 4096.0 * np.finfo(float).eps * scale):
        raise TransportError("FLOW 2DH water/bed exchange failed its mass ledger")
    return Flow2DHExchangeResult(
        suspended_mass_kg_m2=suspended_after,
        bed_mass_kg_m2=bed_after,
        equilibrium_concentration_kg_m3=equilibrium,
        terms=terms,
        transferred_to_suspension_kg_m2=transfer,
        suspended_exchange_ledger_kg=water_ledger,
        bed_exchange_ledger_kg=bed_ledger,
        mass_before_kg=before,
        mass_after_kg=after,
        balance_residual_kg=residual,
        safety_limiter_correction_kg=limiter_correction,
        safety_limiter_applied=bool(np.any(limiter_correction > 0.0)),
        sediment_active=sediment_active,
    )
