"""Source-ordered Delft3D FLOW 2DH ``SettleFrm=2`` settling oracle.

The locked FLOW chain computes a per-fraction representative settling
diameter, evaluates one of three clear-water diameter branches, and finally
applies Richardson--Zaki hindering using the *total* suspended concentration
of all sediment fractions.  This module records that physical operator; it
does not reproduce FLOW's surrounding ADI scalar discretisation.

The source anchors are the read-only ``rdsed.f90``, ``fallve.f90``, and
``eqsettle.f90`` routines.  No Delft3D source is copied or modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .passive import TransportError


FLOW2DH_SETTLING_ID = "d3d4_flow2dh_settlefrm2_noncohesive_v1"
FLOW2DH_SETTLING_FORM = 2
SAND_DIAMETER_THRESHOLD_M = 64.0e-6
GRAVEL_DIAMETER_THRESHOLD_M = 2.0e-3
HINDERED_SETTLING_EXPONENT = 5.0

# This schema is deliberately explicit about the inactive-cell convention.
# FLOW skips ``fallve`` where ``kfs==0``; the oracle returns a zero sentinel
# there and exposes ``active_mask`` so callers cannot mistake it for an
# evaluated physical settling velocity.
FLOW2DH_SETTLING_SCHEMA = {
    "method": FLOW2DH_SETTLING_ID,
    "reference_selector": "SettleFrm=2",
    "settling_diameter": "DSS=SedD50*FacDSS",
    "molecular_viscosity_m2_s": "4e-5/(20+temperature_c)",
    "diameter_branches": [
        "DSS<1.5*Dsand:Stokes",
        "1.5*Dsand<=DSS<0.5*Dgravel:transitional",
        "DSS>=0.5*Dgravel:turbulent",
    ],
    "thresholds_m": {
        "Dsand": SAND_DIAMETER_THRESHOLD_M,
        "Dgravel": GRAVEL_DIAMETER_THRESHOLD_M,
    },
    "hindering": "max(0,1-max(0,total_concentration)/Cref)^5",
    "total_concentration": "sum_all_suspended_sediment_fractions_kg_m3",
    "wet_mask_input": "boolean_provider_mapping_to_FLOW_kfs_nonzero",
    "inactive_output": "zero_sentinel_validity_exposed_by_active_mask",
    "source_order": [
        "rdsed:DSS",
        "fallve:total_concentration",
        "eqsettle:clear_water_velocity",
        "eqsettle:hindering",
    ],
}
FLOW2DH_SETTLING_SEMANTIC_ID = hashlib.sha256(
    json.dumps(
        FLOW2DH_SETTLING_SCHEMA, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()
FLOW2DH_SETTLING_DYNAMIC_DSS_ID = (
    "d3d4_flow2dh_settlefrm2_noncohesive_parent_dss_v2")
FLOW2DH_SETTLING_DYNAMIC_DSS_SCHEMA = {
    **FLOW2DH_SETTLING_SCHEMA,
    "method": FLOW2DH_SETTLING_DYNAMIC_DSS_ID,
    "settling_diameter": (
        "accepted_parent_cellwise_DSS; FLOW fallve precedes erosed/eqtran"),
    "source_order": [
        "fallve:accepted_parent_DSS",
        "fallve:parent_total_concentration",
        "eqsettle:clear_water_velocity",
        "eqsettle:hindering",
        "erosed/eqtran:candidate_DSS_for_next_update",
    ],
}
FLOW2DH_SETTLING_DYNAMIC_DSS_SEMANTIC_ID = hashlib.sha256(
    json.dumps(
        FLOW2DH_SETTLING_DYNAMIC_DSS_SCHEMA, sort_keys=True,
        separators=(",", ":")
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class Flow2DHSettlingResult:
    """Resolved settling diagnostics, with the fraction axis last."""

    settling_velocity_m_s: np.ndarray
    clear_water_settling_velocity_m_s: np.ndarray
    settling_diameter_m: np.ndarray
    total_concentration_kg_m3: np.ndarray
    hindered_factor: np.ndarray
    molecular_viscosity_m2_s: np.ndarray
    active_mask: np.ndarray
    method_id: str = FLOW2DH_SETTLING_ID
    semantic_id: str = FLOW2DH_SETTLING_SEMANTIC_ID


def _finite_array(name: str, value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise TransportError(f"{name} must be finite")
    return result


def settlefrm2_noncohesive(
    suspended_concentration_kg_m3,
    sediment_d50_m,
    settling_fac_dss,
    sediment_density_kg_m3,
    water_density_kg_m3,
    temperature_c,
    *,
    settling_diameter_m=None,
    settling_reference_concentration_kg_m3: float = 1.0e4,
    gravity_m_s2: float = 9.81,
    wet_mask=True,
) -> Flow2DHSettlingResult:
    """Evaluate locked FLOW's ordinary noncohesive ``SettleFrm=2`` chain.

    ``suspended_concentration_kg_m3`` has a mandatory trailing axis containing
    *all* suspended sediment fractions in FLOW's ``lsed`` loop, including a
    fraction whose own settling law is not form 2.  The one-dimensional
    property arrays identify only the noncohesive target fractions for which
    form 2 is evaluated, and may therefore have a different length.  Water
    density, temperature, and the Boolean wet mask broadcast over the leading
    cell dimensions.  FLOW applies the resulting common total-concentration
    hindering factor to every target fraction in a cell.

    ``wet_mask`` is the explicit BRAIDS provider mapping to FLOW's
    ``kfs != 0`` source gate.  An inactive output is a zero sentinel, not a
    claim that FLOW overwrites its stored settling field with zero.
    """

    concentration = _finite_array(
        "suspended_concentration_kg_m3", suspended_concentration_kg_m3
    )
    if concentration.ndim < 1 or concentration.shape[-1] == 0:
        raise TransportError(
            "suspended_concentration_kg_m3 must have a nonempty trailing fraction axis"
        )
    if np.any(concentration < 0.0):
        raise TransportError("Suspended concentrations must be nonnegative")
    cell_shape = concentration.shape[:-1]

    d50 = _finite_array("sediment_d50_m", sediment_d50_m)
    fac_dss = _finite_array("settling_fac_dss", settling_fac_dss)
    rho_s = _finite_array("sediment_density_kg_m3", sediment_density_kg_m3)
    target_fraction_count = d50.size if d50.ndim == 1 else 0
    if target_fraction_count == 0:
        raise TransportError("At least one SettleFrm=2 target fraction is required")
    for name, value in (
        ("sediment_d50_m", d50),
        ("settling_fac_dss", fac_dss),
        ("sediment_density_kg_m3", rho_s),
    ):
        if value.shape != (target_fraction_count,):
            raise TransportError(
                f"{name} must have shape ({target_fraction_count},), got {value.shape}"
            )
    if np.any(d50 <= 0.0) or np.any(fac_dss <= 0.0) or np.any(rho_s <= 0.0):
        raise TransportError("SedD50, FacDSS, and RhoSol must be positive")

    rho_w = _finite_array("water_density_kg_m3", water_density_kg_m3)
    temperature = _finite_array("temperature_c", temperature_c)
    raw_mask = np.asarray(wet_mask)
    if raw_mask.dtype.kind != "b":
        raise TransportError("wet_mask must be Boolean")
    try:
        rho_w = np.broadcast_to(rho_w, cell_shape)
        temperature = np.broadcast_to(temperature, cell_shape)
        active = np.broadcast_to(raw_mask, cell_shape)
    except ValueError as error:
        raise TransportError(
            "Water density, temperature, and wet mask must broadcast over cells"
        ) from error
    if np.any(rho_w <= 0.0) or np.any(temperature <= -20.0):
        raise TransportError(
            "Water density must be positive and temperature must exceed -20 C"
        )
    if (isinstance(settling_reference_concentration_kg_m3, bool) or
            not isinstance(settling_reference_concentration_kg_m3, (int, float)) or
            not math.isfinite(float(settling_reference_concentration_kg_m3)) or
            float(settling_reference_concentration_kg_m3) <= 0.0):
        raise TransportError("Cref must be finite and positive")
    if (isinstance(gravity_m_s2, bool) or
            not isinstance(gravity_m_s2, (int, float)) or
            not math.isfinite(float(gravity_m_s2)) or float(gravity_m_s2) <= 0.0):
        raise TransportError("Gravity must be finite and positive")

    # Append the fraction axis only after validating the broadcast cell state.
    rho_w_f = rho_w[..., np.newaxis]
    temperature_f = temperature[..., np.newaxis]
    rho_s_f = rho_s.reshape((1,) * len(cell_shape) + (target_fraction_count,))
    default_dss = d50 * fac_dss
    if settling_diameter_m is None:
        dss = default_dss
        dss_f = dss.reshape(
            (1,) * len(cell_shape) + (target_fraction_count,))
    else:
        supplied_dss = _finite_array(
            "settling_diameter_m", settling_diameter_m)
        expected_shape = cell_shape + (target_fraction_count,)
        if supplied_dss.shape == (target_fraction_count,):
            dss = supplied_dss
            dss_f = dss.reshape(
                (1,) * len(cell_shape) + (target_fraction_count,))
        elif supplied_dss.shape == expected_shape:
            dss = supplied_dss
            dss_f = dss
        else:
            raise TransportError(
                "settling_diameter_m must have shape "
                f"({target_fraction_count},) or {expected_shape}, got "
                f"{supplied_dss.shape}")
        if np.any(dss <= 0.0):
            raise TransportError("settling_diameter_m must be positive")

    if np.any(rho_s_f <= rho_w_f):
        raise TransportError(
            "RhoSol must exceed local water density for noncohesive settling"
        )

    viscosity = 4.0e-5 / (20.0 + temperature)
    viscosity_f = viscosity[..., np.newaxis]
    relative_density = rho_s_f / rho_w_f
    output_shape = cell_shape + (target_fraction_count,)
    clear = np.empty(output_shape, dtype=float)

    dss_grid = np.broadcast_to(dss_f, output_shape)
    relative_density_grid = np.broadcast_to(relative_density, output_shape)
    viscosity_grid = np.broadcast_to(viscosity_f, output_shape)
    stokes = dss_grid < 1.5 * SAND_DIAMETER_THRESHOLD_M
    transitional = (~stokes) & (dss_f < 0.5 * GRAVEL_DIAMETER_THRESHOLD_M)
    turbulent = ~(stokes | transitional)

    clear[...] = 0.0
    clear[stokes] = (
        (relative_density_grid[stokes] - 1.0) * float(gravity_m_s2) *
        dss_grid[stokes] ** 2 / (18.0 * viscosity_grid[stokes])
    )
    transitional_dss = dss_grid[transitional]
    transitional_coefficient = np.where(
        transitional_dss < 2.0 * SAND_DIAMETER_THRESHOLD_M,
        (-2.9912 / SAND_DIAMETER_THRESHOLD_M) * transitional_dss + 15.9824,
        10.0,
    )
    clear[transitional] = (
        transitional_coefficient * viscosity_grid[transitional] /
        transitional_dss * (
            np.sqrt(
                1.0 + (relative_density_grid[transitional] - 1.0) *
                float(gravity_m_s2) * transitional_dss ** 3 /
                (100.0 * viscosity_grid[transitional] ** 2)
            ) - 1.0
        )
    )
    clear[turbulent] = 1.1 * np.sqrt(
        (relative_density_grid[turbulent] - 1.0) * float(gravity_m_s2) *
        dss_grid[turbulent]
    )

    total = np.sum(concentration, axis=-1)
    hindering = np.maximum(
        0.0,
        1.0 - np.maximum(0.0, total) /
        float(settling_reference_concentration_kg_m3),
    ) ** HINDERED_SETTLING_EXPONENT
    velocity = clear * hindering[..., np.newaxis]
    velocity = np.where(active[..., np.newaxis], velocity, 0.0)

    if not np.isfinite(clear).all() or not np.isfinite(velocity).all():
        raise TransportError("SettleFrm=2 produced a non-finite velocity")
    dynamic_dss = settling_diameter_m is not None
    return Flow2DHSettlingResult(
        settling_velocity_m_s=velocity,
        clear_water_settling_velocity_m_s=clear,
        settling_diameter_m=dss.copy(),
        total_concentration_kg_m3=total,
        hindered_factor=hindering,
        molecular_viscosity_m2_s=viscosity,
        active_mask=np.asarray(active, dtype=bool).copy(),
        method_id=(FLOW2DH_SETTLING_DYNAMIC_DSS_ID if dynamic_dss else
                   FLOW2DH_SETTLING_ID),
        semantic_id=(FLOW2DH_SETTLING_DYNAMIC_DSS_SEMANTIC_ID
                     if dynamic_dss else FLOW2DH_SETTLING_SEMANTIC_ID),
    )


__all__ = [
    "FLOW2DH_SETTLING_FORM",
    "FLOW2DH_SETTLING_DYNAMIC_DSS_ID",
    "FLOW2DH_SETTLING_DYNAMIC_DSS_SCHEMA",
    "FLOW2DH_SETTLING_DYNAMIC_DSS_SEMANTIC_ID",
    "FLOW2DH_SETTLING_ID",
    "FLOW2DH_SETTLING_SCHEMA",
    "FLOW2DH_SETTLING_SEMANTIC_ID",
    "Flow2DHSettlingResult",
    "GRAVEL_DIAMETER_THRESHOLD_M",
    "HINDERED_SETTLING_EXPONENT",
    "SAND_DIAMETER_THRESHOLD_M",
    "settlefrm2_noncohesive",
]
