"""Conservative physical diffusion of suspended sediment.

The evolved state is dry suspended mass per bed area, ``S = h*C``.  This
operator discretizes

    d(h C)/dt = div(h K grad(C))

with a two-point face flux and harmonic face transmissivity.  It is kept
separate from advective transport so numerical and prescribed physical
diffusion can be measured independently.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .passive import TransportError, _concentration, _validate_state


@dataclass(frozen=True)
class SuspendedDiffusionResult:
    """Updated state and diagnostics from one explicit diffusion step."""

    suspended_mass_kg_m2: np.ndarray
    concentration_kg_m3: np.ndarray
    sediment_face_mass_x_kg: np.ndarray
    sediment_face_mass_y_kg: np.ndarray
    mass_before_kg: float
    mass_after_kg: float
    balance_residual_kg: float
    minimum_concentration_kg_m3: float
    maximum_concentration_kg_m3: float
    maximum_diffusive_cfl: float


def _harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    total = left + right
    return np.divide(2.0 * left * right, total, out=np.zeros_like(total), where=total > 0.0)


def advance_suspended_diffusion(
    h_m,
    suspended_mass_before_kg_m2,
    diffusivity_m2_s,
    *,
    dt_s: float,
    dx_m: float,
    dy_m: float,
    periodic_x: bool = False,
    periodic_y: bool = False,
    dry_tolerance_m: float = 1.0e-3,
) -> SuspendedDiffusionResult:
    """Advance physical diffusion by one conservative explicit step.

    ``diffusivity_m2_s`` may be a non-negative scalar or a cell array matching
    ``h_m``.  Closed boundaries have zero normal diffusive flux.  Periodic
    seams use the same shared face flux on both sides.
    """

    if not all(math.isfinite(v) and v > 0.0 for v in (dt_s, dx_m, dy_m)):
        raise TransportError("dt_s, dx_m, and dy_m must be finite and positive")
    area = dx_m * dy_m
    h, suspended = _validate_state(
        h_m, suspended_mass_before_kg_m2, area, dry_tolerance_m
    )
    diffusivity = np.asarray(diffusivity_m2_s, dtype=float)
    if diffusivity.ndim == 0:
        diffusivity = np.full_like(h, float(diffusivity))
    if diffusivity.shape != h.shape:
        raise TransportError(
            f"diffusivity_m2_s must be scalar or have shape {h.shape}, got {diffusivity.shape}"
        )
    if not np.all(np.isfinite(diffusivity)) or np.any(diffusivity < 0.0):
        raise TransportError("diffusivity_m2_s must be finite and non-negative")

    ny, nx = h.shape
    wet = h > dry_tolerance_m
    concentration = _concentration(h, suspended, dry_tolerance_m)
    transmissivity = np.where(wet, h * diffusivity, 0.0)

    tx = np.zeros((ny, nx + 1), dtype=float)
    ty = np.zeros((ny + 1, nx), dtype=float)
    if nx > 1:
        tx[:, 1:nx] = _harmonic(transmissivity[:, :-1], transmissivity[:, 1:])
    if ny > 1:
        ty[1:ny, :] = _harmonic(transmissivity[:-1, :], transmissivity[1:, :])
    if periodic_x:
        seam = _harmonic(transmissivity[:, -1], transmissivity[:, 0])
        tx[:, 0] = seam
        tx[:, -1] = seam
    if periodic_y:
        seam = _harmonic(transmissivity[-1, :], transmissivity[0, :])
        ty[0, :] = seam
        ty[-1, :] = seam

    face_mass_x = np.zeros_like(tx)
    face_mass_y = np.zeros_like(ty)
    if nx > 1:
        face_mass_x[:, 1:nx] = (
            -tx[:, 1:nx] * (concentration[:, 1:] - concentration[:, :-1])
            / dx_m * dt_s * dy_m
        )
    if ny > 1:
        face_mass_y[1:ny, :] = (
            -ty[1:ny, :] * (concentration[1:, :] - concentration[:-1, :])
            / dy_m * dt_s * dx_m
        )
    if periodic_x:
        seam = (
            -tx[:, 0] * (concentration[:, 0] - concentration[:, -1])
            / dx_m * dt_s * dy_m
        )
        face_mass_x[:, 0] = seam
        face_mass_x[:, -1] = seam
    if periodic_y:
        seam = (
            -ty[0, :] * (concentration[0, :] - concentration[-1, :])
            / dy_m * dt_s * dx_m
        )
        face_mass_y[0, :] = seam
        face_mass_y[-1, :] = seam

    coefficient_sum = (
        (tx[:, :-1] + tx[:, 1:]) / dx_m**2
        + (ty[:-1, :] + ty[1:, :]) / dy_m**2
    )
    local_cfl = np.zeros_like(h)
    local_cfl[wet] = dt_s * coefficient_sum[wet] / h[wet]
    maximum_cfl = float(np.max(local_cfl, initial=0.0))
    if maximum_cfl > 1.0 + 1024.0 * np.finfo(float).eps:
        raise TransportError(
            f"explicit diffusion stability limit exceeded: maximum diffusive CFL={maximum_cfl:.9g} > 1"
        )

    cell_mass_before = suspended * area
    cell_mass_after = (
        cell_mass_before
        - (face_mass_x[:, 1:] - face_mass_x[:, :-1])
        - (face_mass_y[1:, :] - face_mass_y[:-1, :])
    )
    scale = max(1.0, float(np.max(np.abs(cell_mass_before), initial=0.0)))
    tolerance = 2048.0 * np.finfo(float).eps * scale
    if float(np.min(cell_mass_after, initial=0.0)) < -tolerance:
        raise TransportError("diffusion produced negative suspended mass")
    suspended_after = np.maximum(cell_mass_after, 0.0) / area
    concentration_after = _concentration(h, suspended_after, dry_tolerance_m)
    mass_before = float(cell_mass_before.sum())
    mass_after = float(cell_mass_after.sum())

    return SuspendedDiffusionResult(
        suspended_mass_kg_m2=suspended_after,
        concentration_kg_m3=concentration_after,
        sediment_face_mass_x_kg=face_mass_x,
        sediment_face_mass_y_kg=face_mass_y,
        mass_before_kg=mass_before,
        mass_after_kg=mass_after,
        balance_residual_kg=mass_after - mass_before,
        minimum_concentration_kg_m3=float(np.min(concentration_after, initial=0.0)),
        maximum_concentration_kg_m3=float(np.max(concentration_after, initial=0.0)),
        maximum_diffusive_cfl=maximum_cfl,
    )
