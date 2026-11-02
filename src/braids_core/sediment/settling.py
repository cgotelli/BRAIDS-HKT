"""Conservative settling closures and one-fraction Exner bed update."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .passive import TransportError


HINDERED_SETTLING_EXPONENT = 5.0
DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3 = 1.0e4


def hindered_settling_factor(
    total_concentration_kg_m3,
    reference_concentration_kg_m3: float = DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3,
):
    """Return the bounded Richardson-Zaki/Mehta settling multiplier.

    ``total_concentration_kg_m3`` is the sum of all suspended dry-sediment
    concentrations.  The result is in [0, 1], is exactly one at the dilute
    limit, and is exactly zero at and above the reference concentration.
    """

    concentration = np.asarray(total_concentration_kg_m3, dtype=float)
    if (isinstance(reference_concentration_kg_m3, bool) or
            not math.isfinite(reference_concentration_kg_m3) or
            reference_concentration_kg_m3 <= 0 or
            not np.isfinite(concentration).all()):
        raise TransportError("Hindered-settling concentrations must be finite and the reference must be positive")
    if np.any(concentration < 0):
        raise TransportError("Total suspended concentration must be nonnegative")
    factor = np.maximum(0.0, 1.0 - concentration / reference_concentration_kg_m3) ** HINDERED_SETTLING_EXPONENT
    return float(factor) if factor.ndim == 0 else factor


@dataclass(frozen=True)
class SettlingExnerResult:
    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    bed_elevation_m: np.ndarray
    non_erodible_base_m: np.ndarray
    transferred_mass_kg: float
    total_mass_before_kg: float
    total_mass_after_kg: float
    balance_residual_kg: float


def advance_settling_exner(
    water_depth_m,
    suspended_mass_kg_m2,
    bed_mass_kg_m2,
    bed_elevation_m,
    *,
    dt_s: float,
    settling_velocity_m_s: float,
    grain_density_kg_m3: float,
    porosity: float,
    cell_area_m2: float,
    dry_tolerance_m: float = 1.0e-10,
    erosion_rate_kg_m2_s: float = 0.0,
) -> SettlingExnerResult:
    """Apply exact settling/erosion exchange with finite bed availability."""

    h = np.asarray(water_depth_m, dtype=float)
    suspended = np.asarray(suspended_mass_kg_m2, dtype=float)
    bed_mass = np.asarray(bed_mass_kg_m2, dtype=float)
    bed = np.asarray(bed_elevation_m, dtype=float)
    if h.ndim != 2 or suspended.shape != h.shape or bed_mass.shape != h.shape or bed.shape != h.shape:
        raise TransportError("settling states must be same-shaped 2D arrays")
    scalars = (dt_s, settling_velocity_m_s, grain_density_kg_m3, porosity,
               cell_area_m2, dry_tolerance_m, erosion_rate_kg_m2_s)
    if not all(math.isfinite(value) for value in scalars) or not all(
            np.all(np.isfinite(value)) for value in (h, suspended, bed_mass, bed)):
        raise TransportError("settling inputs must be finite")
    if (dt_s <= 0 or settling_velocity_m_s < 0 or erosion_rate_kg_m2_s < 0
            or grain_density_kg_m3 <= 0
            or not 0 <= porosity < 1 or cell_area_m2 <= 0 or dry_tolerance_m < 0):
        raise TransportError("settling parameters are outside their valid range")
    tolerance = 1024 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(suspended), initial=0.0)),
        float(np.max(np.abs(bed_mass), initial=0.0)),
    )
    if np.any(h < 0) or np.any(suspended < -tolerance) or np.any(bed_mass < -tolerance):
        raise TransportError("settling inventories and water depth must be non-negative")
    suspended = np.maximum(suspended, 0.0)
    bed_mass = np.maximum(bed_mass, 0.0)
    bulk_density = (1.0 - porosity) * grain_density_kg_m3
    base = bed - bed_mass / bulk_density
    suspended_after = np.zeros_like(suspended)
    wet = h > dry_tolerance_m
    if settling_velocity_m_s == 0.0:
        suspended_after[wet] = suspended[wet] + erosion_rate_kg_m2_s * dt_s
    else:
        rate = settling_velocity_m_s / h[wet]
        equilibrium = erosion_rate_kg_m2_s / rate
        suspended_after[wet] = (
            equilibrium + (suspended[wet] - equilibrium) * np.exp(-rate * dt_s)
        )
    # Numerically dry columns deposit their residual suspension and do not erode.
    available = suspended + bed_mass
    suspended_after = np.minimum(np.maximum(suspended_after, 0.0), available)
    bed_mass_after = available - suspended_after
    transfer = bed_mass_after - bed_mass
    bed_after = bed + transfer / bulk_density
    total_before = float(np.sum(suspended + bed_mass) * cell_area_m2)
    total_after = float(np.sum(suspended_after + bed_mass_after) * cell_area_m2)
    return SettlingExnerResult(
        suspended_after,
        bed_mass_after,
        bed_after,
        base,
        float(np.sum(transfer) * cell_area_m2),
        total_before,
        total_after,
        total_after - total_before,
    )
