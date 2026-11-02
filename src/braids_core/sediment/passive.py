"""Source-free passive transport using accepted finite-volume water fluxes.

The transported state is suspended dry mass per bed area, ``S = h*C`` in
kg/m2.  The flux arrays contain water *volume integrated over one accepted
hydraulic step*, so multiplying them by an upwind concentration gives the dry
mass crossing the same face during that step.

Array conventions for a cell array shaped ``(ny, nx)`` are:

* ``x_m3[j, i]`` has shape ``(ny, nx + 1)`` and is positive from cell
  ``i-1`` to cell ``i`` (eastward).
* ``y_m3[j, i]`` has shape ``(ny + 1, nx)`` and is positive from cell
  ``j-1`` to cell ``j`` (northward).

The first-order upwind mode remains the conservative, bounded fallback and
diffusion reference.  MUSCL supplies a midpoint, MC-limited second-order
candidate.  FCT limits the MUSCL antidiffusive correction against local cell
bounds after the low-order update, preserving conservation in multiple
dimensions.  The WENO option replaces the MUSCL candidate with fifth-order
WENO-Z reconstruction and applies the same conservative FCT bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class TransportError(ValueError):
    """Raised when an accepted-flux transport contract is inconsistent."""


@dataclass(frozen=True)
class AcceptedWaterFlux:
    """Water volumes crossing all Cartesian faces during one accepted step."""

    x_m3: np.ndarray
    y_m3: np.ndarray
    periodic_x: bool = False
    periodic_y: bool = False


@dataclass(frozen=True)
class PassiveTransportResult:
    """Updated suspended state and conservative step diagnostics."""

    suspended_mass_kg_m2: np.ndarray
    concentration_kg_m3: np.ndarray
    sediment_face_mass_x_kg: np.ndarray
    sediment_face_mass_y_kg: np.ndarray
    mass_before_kg: float
    mass_after_kg: float
    boundary_mass_in_kg: float
    balance_residual_kg: float
    minimum_concentration_kg_m3: float
    maximum_concentration_kg_m3: float
    scheme: str
    redistributed_dry_mass_kg: float = 0.0
    inactive_dry_mass_kg: float = 0.0


def _finite_array(name: str, values, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise TransportError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise TransportError(f"{name} contains non-finite values")
    return array


def _validate_state(
    h, suspended_mass, cell_area_m2: float, dry_tolerance_m: float,
    *, allow_dry_inventory: bool = False,
):
    h = np.asarray(h, dtype=float)
    suspended_mass = np.asarray(suspended_mass, dtype=float)
    if h.ndim != 2 or h.shape != suspended_mass.shape:
        raise TransportError("h and suspended mass must be same-shaped 2D arrays")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(suspended_mass)):
        raise TransportError("state contains non-finite values")
    if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        raise TransportError("cell_area_m2 must be finite and positive")
    if not math.isfinite(dry_tolerance_m) or dry_tolerance_m < 0:
        raise TransportError("dry_tolerance_m must be finite and non-negative")
    scale = max(1.0, float(np.max(np.abs(suspended_mass), initial=0.0)))
    tolerance = 128 * np.finfo(float).eps * scale
    if np.min(h, initial=0.0) < 0:
        raise TransportError("water depth cannot be negative")
    if np.min(suspended_mass, initial=0.0) < -tolerance:
        raise TransportError("suspended mass cannot be negative")
    if (not allow_dry_inventory
            and np.any(np.abs(suspended_mass[h <= dry_tolerance_m]) > tolerance)):
        raise TransportError("dry cells cannot contain suspended mass")
    return h, np.maximum(suspended_mass, 0.0)


def _validate_flux(flux: AcceptedWaterFlux, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = _finite_array("x_m3", flux.x_m3, (ny, nx + 1))
    y = _finite_array("y_m3", flux.y_m3, (ny + 1, nx))
    scale = max(1.0, float(np.max(np.abs(x), initial=0.0)), float(np.max(np.abs(y), initial=0.0)))
    tolerance = 128 * np.finfo(float).eps * scale
    if flux.periodic_x:
        if not np.allclose(x[:, 0], x[:, -1], rtol=0, atol=tolerance):
            raise TransportError("periodic x boundary volumes must match")
    elif np.any(np.abs(x[:, (0, -1)]) > tolerance):
        raise TransportError("closed x boundary volumes must be zero")
    if flux.periodic_y:
        if not np.allclose(y[0, :], y[-1, :], rtol=0, atol=tolerance):
            raise TransportError("periodic y boundary volumes must match")
    elif np.any(np.abs(y[(0, -1), :]) > tolerance):
        raise TransportError("closed y boundary volumes must be zero")
    return x, y


def _redistribute_numerically_dry_mass(
    cell_mass: np.ndarray,
    h_after: np.ndarray,
    sediment_x: np.ndarray,
    sediment_y: np.ndarray,
    x_volume: np.ndarray,
    y_volume: np.ndarray,
    flux: AcceptedWaterFlux,
    dry_tolerance_m: float,
) -> float:
    """Move threshold-dry residual mass along accepted outgoing water faces."""

    ny, nx = cell_mass.shape
    dry = h_after <= dry_tolerance_m
    redistributed = 0.0

    def add_x(j: int, face: int, amount: float) -> None:
        sediment_x[j, face] += amount
        if flux.periodic_x and face in (0, nx):
            sediment_x[j, 0] += amount if face == nx else 0.0
            sediment_x[j, nx] += amount if face == 0 else 0.0

    def add_y(face: int, i: int, amount: float) -> None:
        sediment_y[face, i] += amount
        if flux.periodic_y and face in (0, ny):
            sediment_y[0, i] += amount if face == ny else 0.0
            sediment_y[ny, i] += amount if face == 0 else 0.0

    for j, i in np.argwhere(dry):
        residual = float(cell_mass[j, i])
        if residual <= 0.0:
            cell_mass[j, i] = 0.0
            continue
        candidates: list[tuple[str, int, int, int | None, int | None, float, float]] = []
        # (axis, face_j, face_i, recipient_j, recipient_i, oriented_sign, weight)
        west = float(x_volume[j, i])
        if west < 0.0:
            recipient = (j, i - 1) if i > 0 else ((j, nx - 1) if flux.periodic_x else None)
            if recipient is None or not dry[recipient]:
                candidates.append(("x", j, i, *(recipient or (None, None)), -1.0, -west))
        east = float(x_volume[j, i + 1])
        if east > 0.0:
            recipient = (j, i + 1) if i < nx - 1 else ((j, 0) if flux.periodic_x else None)
            if recipient is None or not dry[recipient]:
                candidates.append(("x", j, i + 1, *(recipient or (None, None)), 1.0, east))
        south = float(y_volume[j, i])
        if south < 0.0:
            recipient = (j - 1, i) if j > 0 else ((ny - 1, i) if flux.periodic_y else None)
            if recipient is None or not dry[recipient]:
                candidates.append(("y", j, i, *(recipient or (None, None)), -1.0, -south))
        north = float(y_volume[j + 1, i])
        if north > 0.0:
            recipient = (j + 1, i) if j < ny - 1 else ((0, i) if flux.periodic_y else None)
            if recipient is None or not dry[recipient]:
                candidates.append(("y", j + 1, i, *(recipient or (None, None)), 1.0, north))
        total_weight = math.fsum(item[-1] for item in candidates)
        if total_weight <= 0.0:
            # At an advancing shoreline, a sub-threshold cell may have only
            # received water. Return its unresolved scalar mass to the wet
            # donor by reducing that incoming scalar face flux.
            fallback = []
            if i > 0 and not dry[j, i - 1] and x_volume[j, i] != 0.0:
                fallback.append(("x", j, i, j, i - 1, -1.0, abs(float(x_volume[j, i]))))
            if i < nx - 1 and not dry[j, i + 1] and x_volume[j, i + 1] != 0.0:
                fallback.append(("x", j, i + 1, j, i + 1, 1.0, abs(float(x_volume[j, i + 1]))))
            if j > 0 and not dry[j - 1, i] and y_volume[j, i] != 0.0:
                fallback.append(("y", j, i, j - 1, i, -1.0, abs(float(y_volume[j, i]))))
            if j < ny - 1 and not dry[j + 1, i] and y_volume[j + 1, i] != 0.0:
                fallback.append(("y", j + 1, i, j + 1, i, 1.0, abs(float(y_volume[j + 1, i]))))
            candidates = fallback
            total_weight = math.fsum(item[-1] for item in candidates)
        if total_weight <= 0.0:
            raise TransportError(
                "a numerically dry cell retains suspended mass without an outgoing wet/boundary face"
            )
        for axis, face_j, face_i, recipient_j, recipient_i, sign, weight in candidates:
            amount = residual * weight / total_weight
            if axis == "x":
                add_x(face_j, face_i, sign * amount)
            else:
                add_y(face_j, face_i, sign * amount)
            if recipient_j is not None:
                cell_mass[recipient_j, recipient_i] += amount
        cell_mass[j, i] = 0.0
        redistributed += residual
    return redistributed


def water_depth_from_accepted_flux(
    h_before_m: np.ndarray,
    flux: AcceptedWaterFlux,
    *,
    cell_area_m2: float,
) -> np.ndarray:
    """Apply the source-free water continuity update represented by ``flux``."""

    h = np.asarray(h_before_m, dtype=float)
    if h.ndim != 2 or not np.all(np.isfinite(h)) or np.any(h < 0):
        raise TransportError("h_before_m must be a finite, non-negative 2D array")
    if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        raise TransportError("cell_area_m2 must be finite and positive")
    x, y = _validate_flux(flux, h.shape)
    volume_after = h * cell_area_m2 - (x[:, 1:] - x[:, :-1]) - (y[1:, :] - y[:-1, :])
    tolerance = 256 * np.finfo(float).eps * max(1.0, float(np.max(h * cell_area_m2, initial=0.0)))
    if np.min(volume_after, initial=0.0) < -tolerance:
        raise TransportError("accepted water flux drains a cell below zero volume")
    return np.maximum(volume_after, 0.0) / cell_area_m2


def _concentration(h: np.ndarray, suspended_mass: np.ndarray, dry_tolerance_m: float) -> np.ndarray:
    concentration = np.zeros_like(h)
    wet = h > dry_tolerance_m
    concentration[wet] = suspended_mass[wet] / h[wet]
    return concentration


def _upwind_face_concentrations(
    concentration: np.ndarray,
    x_volume: np.ndarray,
    y_volume: np.ndarray,
    flux: AcceptedWaterFlux,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = concentration.shape
    cx = np.zeros((ny, nx + 1), dtype=float)
    cy = np.zeros((ny + 1, nx), dtype=float)

    cx[:, 1:nx] = np.where(
        x_volume[:, 1:nx] >= 0, concentration[:, :-1], concentration[:, 1:]
    )
    if flux.periodic_x:
        seam = np.where(x_volume[:, 0] >= 0, concentration[:, -1], concentration[:, 0])
        cx[:, 0] = seam
        cx[:, -1] = seam

    cy[1:ny, :] = np.where(
        y_volume[1:ny, :] >= 0, concentration[:-1, :], concentration[1:, :]
    )
    if flux.periodic_y:
        seam = np.where(y_volume[0, :] >= 0, concentration[-1, :], concentration[0, :])
        cy[0, :] = seam
        cy[-1, :] = seam
    return cx, cy


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = (np.signbit(a) == np.signbit(b)) & (np.signbit(b) == np.signbit(c))
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.copysign(magnitude, a), 0.0)


def _limited_slopes(concentration: np.ndarray, *, axis: int, periodic: bool) -> np.ndarray:
    if axis == 1:
        if periodic:
            previous, following = np.roll(concentration, 1, axis=1), np.roll(concentration, -1, axis=1)
        else:
            previous = np.concatenate([concentration[:, :1], concentration[:, :-1]], axis=1)
            following = np.concatenate([concentration[:, 1:], concentration[:, -1:]], axis=1)
    else:
        if periodic:
            previous, following = np.roll(concentration, 1, axis=0), np.roll(concentration, -1, axis=0)
        else:
            previous = np.concatenate([concentration[:1, :], concentration[:-1, :]], axis=0)
            following = np.concatenate([concentration[1:, :], concentration[-1:, :]], axis=0)
    backward = concentration - previous
    forward = following - concentration
    return _minmod3(2.0 * backward, 0.5 * (following - previous), 2.0 * forward)


def _muscl_face_concentrations(
    concentration: np.ndarray,
    x_volume: np.ndarray,
    y_volume: np.ndarray,
    flux: AcceptedWaterFlux,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded MC reconstruction of half-step cell concentrations."""

    ny, nx = concentration.shape
    sx = _limited_slopes(concentration, axis=1, periodic=flux.periodic_x)
    sy = _limited_slopes(concentration, axis=0, periodic=flux.periodic_y)
    cx = np.zeros((ny, nx + 1), dtype=float)
    cy = np.zeros((ny + 1, nx), dtype=float)

    left = concentration[:, :-1] + 0.5 * sx[:, :-1]
    right = concentration[:, 1:] - 0.5 * sx[:, 1:]
    interior = np.where(x_volume[:, 1:nx] >= 0, left, right)
    cx[:, 1:nx] = np.clip(interior, np.minimum(concentration[:, :-1], concentration[:, 1:]),
                           np.maximum(concentration[:, :-1], concentration[:, 1:]))
    if flux.periodic_x:
        left = concentration[:, -1] + 0.5 * sx[:, -1]
        right = concentration[:, 0] - 0.5 * sx[:, 0]
        seam = np.where(x_volume[:, 0] >= 0, left, right)
        seam = np.clip(seam, np.minimum(concentration[:, -1], concentration[:, 0]),
                        np.maximum(concentration[:, -1], concentration[:, 0]))
        cx[:, 0] = seam
        cx[:, -1] = seam

    lower = concentration[:-1, :] + 0.5 * sy[:-1, :]
    upper = concentration[1:, :] - 0.5 * sy[1:, :]
    interior = np.where(y_volume[1:ny, :] >= 0, lower, upper)
    cy[1:ny, :] = np.clip(interior, np.minimum(concentration[:-1, :], concentration[1:, :]),
                           np.maximum(concentration[:-1, :], concentration[1:, :]))
    if flux.periodic_y:
        lower = concentration[-1, :] + 0.5 * sy[-1, :]
        upper = concentration[0, :] - 0.5 * sy[0, :]
        seam = np.where(y_volume[0, :] >= 0, lower, upper)
        seam = np.clip(seam, np.minimum(concentration[-1, :], concentration[0, :]),
                        np.maximum(concentration[-1, :], concentration[0, :]))
        cy[0, :] = seam
        cy[-1, :] = seam
    return cx, cy


def _shift(values: np.ndarray, offset: int, *, axis: int, periodic: bool) -> np.ndarray:
    """Return ``values[i + offset]`` with periodic or constant extension."""

    if periodic:
        return np.roll(values, -offset, axis=axis)
    indices = np.clip(np.arange(values.shape[axis]) + offset, 0, values.shape[axis] - 1)
    return np.take(values, indices, axis=axis)


def _weno_z_left(
    vmm: np.ndarray,
    vm: np.ndarray,
    v0: np.ndarray,
    vp: np.ndarray,
    vpp: np.ndarray,
) -> np.ndarray:
    """Fifth-order WENO-Z value at the right face of the central cell."""

    p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0
    p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0
    p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0
    beta0 = (13.0 / 12.0) * (vmm - 2.0 * vm + v0) ** 2 + 0.25 * (
        vmm - 4.0 * vm + 3.0 * v0
    ) ** 2
    beta1 = (13.0 / 12.0) * (vm - 2.0 * v0 + vp) ** 2 + 0.25 * (vm - vp) ** 2
    beta2 = (13.0 / 12.0) * (v0 - 2.0 * vp + vpp) ** 2 + 0.25 * (
        3.0 * v0 - 4.0 * vp + vpp
    ) ** 2
    tau5 = np.abs(beta0 - beta2)
    scale = np.maximum.reduce([vmm * vmm, vm * vm, v0 * v0, vp * vp, vpp * vpp])
    epsilon = 1e-40 + 1e-14 * scale
    alpha0 = 0.1 * (1.0 + (tau5 / (beta0 + epsilon)) ** 2)
    alpha1 = 0.6 * (1.0 + (tau5 / (beta1 + epsilon)) ** 2)
    alpha2 = 0.3 * (1.0 + (tau5 / (beta2 + epsilon)) ** 2)
    total = alpha0 + alpha1 + alpha2
    return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / total


def _weno_face_concentrations(
    concentration: np.ndarray,
    x_volume: np.ndarray,
    y_volume: np.ndarray,
    flux: AcceptedWaterFlux,
) -> tuple[np.ndarray, np.ndarray]:
    """Upwind fifth-order WENO-Z face values from midpoint concentrations."""

    ny, nx = concentration.shape

    def reconstructions(axis: int, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
        shifts = [
            _shift(concentration, offset, axis=axis, periodic=periodic)
            for offset in (-2, -1, 0, 1, 2)
        ]
        right = _weno_z_left(*shifts)
        left = _weno_z_left(*shifts[::-1])
        return right, left

    right_x, left_x = reconstructions(1, flux.periodic_x)
    cx = np.zeros((ny, nx + 1), dtype=float)
    cx[:, 1:nx] = np.where(
        x_volume[:, 1:nx] >= 0, right_x[:, :-1], left_x[:, 1:]
    )
    if flux.periodic_x:
        seam = np.where(x_volume[:, 0] >= 0, right_x[:, -1], left_x[:, 0])
        cx[:, 0] = seam
        cx[:, -1] = seam

    right_y, left_y = reconstructions(0, flux.periodic_y)
    cy = np.zeros((ny + 1, nx), dtype=float)
    cy[1:ny, :] = np.where(
        y_volume[1:ny, :] >= 0, right_y[:-1, :], left_y[1:, :]
    )
    if flux.periodic_y:
        seam = np.where(y_volume[0, :] >= 0, right_y[-1, :], left_y[0, :])
        cy[0, :] = seam
        cy[-1, :] = seam
    return cx, cy


def _cell_mass_after_flux(cell_mass_before, sediment_x, sediment_y):
    return (
        cell_mass_before
        - (sediment_x[:, 1:] - sediment_x[:, :-1])
        - (sediment_y[1:, :] - sediment_y[:-1, :])
    )


def _neighbor_bounds(
    concentration: np.ndarray, flux: AcceptedWaterFlux
) -> tuple[np.ndarray, np.ndarray]:
    """Return extrema over each cell and its four face neighbors."""

    if flux.periodic_x:
        west = np.roll(concentration, 1, axis=1)
        east = np.roll(concentration, -1, axis=1)
    else:
        west = np.concatenate([concentration[:, :1], concentration[:, :-1]], axis=1)
        east = np.concatenate([concentration[:, 1:], concentration[:, -1:]], axis=1)
    if flux.periodic_y:
        south = np.roll(concentration, 1, axis=0)
        north = np.roll(concentration, -1, axis=0)
    else:
        south = np.concatenate([concentration[:1, :], concentration[:-1, :]], axis=0)
        north = np.concatenate([concentration[1:, :], concentration[-1:, :]], axis=0)
    return (
        np.minimum.reduce([concentration, west, east, south, north]),
        np.maximum.reduce([concentration, west, east, south, north]),
    )


def _fct_face_masses(
    concentration_before: np.ndarray,
    h_after: np.ndarray,
    cell_mass_before: np.ndarray,
    low_x: np.ndarray,
    low_y: np.ndarray,
    high_x: np.ndarray,
    high_y: np.ndarray,
    flux: AcceptedWaterFlux,
    cell_area_m2: float,
    dry_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a conservative Zalesak-type limiter to MUSCL corrections."""

    low_mass = _cell_mass_after_flux(cell_mass_before, low_x, low_y)
    cmin, cmax = _neighbor_bounds(concentration_before, flux)
    volume_after = h_after * cell_area_m2
    lower_mass = cmin * volume_after
    upper_mass = cmax * volume_after
    scale = max(1.0, float(np.max(np.abs(cell_mass_before), initial=0.0)))
    tolerance = 4096 * np.finfo(float).eps * scale
    active_cell = h_after > dry_tolerance_m
    if (np.any(low_mass[active_cell] < lower_mass[active_cell] - tolerance)
            or np.any(low_mass[active_cell] > upper_mass[active_cell] + tolerance)):
        raise TransportError("FCT low-order state violates local bounds; reduce the transport CFL")

    anti_x = high_x - low_x
    anti_y = high_y - low_y
    contributions = (
        anti_x[:, :-1],
        -anti_x[:, 1:],
        anti_y[:-1, :],
        -anti_y[1:, :],
    )
    positive = sum(np.maximum(item, 0.0) for item in contributions)
    negative = sum(np.maximum(-item, 0.0) for item in contributions)
    allowance_positive = np.maximum(upper_mass - low_mass, 0.0)
    allowance_negative = np.maximum(low_mass - lower_mass, 0.0)
    r_positive = np.ones_like(low_mass)
    r_negative = np.ones_like(low_mass)
    active = positive > 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.divide(
            allowance_positive,
            positive,
            out=np.ones_like(positive),
            where=active,
        )
    r_positive[active] = np.minimum(1.0, ratio[active])
    active = negative > 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.divide(
            allowance_negative,
            negative,
            out=np.ones_like(negative),
            where=active,
        )
    r_negative[active] = np.minimum(1.0, ratio[active])
    r_positive[~active_cell] = 0.0
    r_negative[~active_cell] = 0.0

    ny, nx = concentration_before.shape
    alpha_x = np.zeros_like(anti_x)
    interior = anti_x[:, 1:nx]
    alpha_x[:, 1:nx] = np.where(
        interior >= 0,
        np.minimum(r_negative[:, :-1], r_positive[:, 1:]),
        np.minimum(r_positive[:, :-1], r_negative[:, 1:]),
    )
    if flux.periodic_x:
        seam = anti_x[:, 0]
        alpha = np.where(
            seam >= 0,
            np.minimum(r_negative[:, -1], r_positive[:, 0]),
            np.minimum(r_positive[:, -1], r_negative[:, 0]),
        )
        alpha_x[:, 0] = alpha
        alpha_x[:, -1] = alpha

    alpha_y = np.zeros_like(anti_y)
    interior = anti_y[1:ny, :]
    alpha_y[1:ny, :] = np.where(
        interior >= 0,
        np.minimum(r_negative[:-1, :], r_positive[1:, :]),
        np.minimum(r_positive[:-1, :], r_negative[1:, :]),
    )
    if flux.periodic_y:
        seam = anti_y[0, :]
        alpha = np.where(
            seam >= 0,
            np.minimum(r_negative[-1, :], r_positive[0, :]),
            np.minimum(r_positive[-1, :], r_negative[0, :]),
        )
        alpha_y[0, :] = alpha
        alpha_y[-1, :] = alpha
    # Keep roundoff from crossing an active bound and triggering a later
    # nonconservative zero clip. The safety factor is far below truncation error.
    alpha_x *= 1.0 - 1e-12
    alpha_y *= 1.0 - 1e-12
    return low_x + alpha_x * anti_x, low_y + alpha_y * anti_y


def advance_suspended_mass(
    h_before_m: np.ndarray,
    h_after_m: np.ndarray,
    suspended_mass_before_kg_m2: np.ndarray,
    flux: AcceptedWaterFlux,
    *,
    cell_area_m2: float,
    dry_tolerance_m: float = 1e-10,
    scheme: str = "upwind",
    wet_dry_policy: str = "error",
) -> PassiveTransportResult:
    """Advance ``S=h*C`` with the exact accepted water face volumes.

    This source-free operation rejects a hydraulic state that is inconsistent
    with the supplied flux.  That check is what makes uniform concentration an
    exact discrete invariant.  Hydraulic or sediment sources will later enter
    as separate conservative operators with their own ledgers.
    """

    if wet_dry_policy not in {"error", "redistribute", "retain"}:
        raise TransportError("wet_dry_policy must be 'error', 'redistribute', or 'retain'")
    h_before, suspended_before = _validate_state(
        h_before_m, suspended_mass_before_kg_m2, cell_area_m2, dry_tolerance_m,
        allow_dry_inventory=wet_dry_policy == "retain",
    )
    h_after = _finite_array("h_after_m", h_after_m, h_before.shape)
    if np.any(h_after < 0):
        raise TransportError("h_after_m cannot be negative")
    x_volume, y_volume = _validate_flux(flux, h_before.shape)
    expected_h_after = water_depth_from_accepted_flux(
        h_before, flux, cell_area_m2=cell_area_m2
    )
    water_scale = max(1.0, float(np.max(np.abs(expected_h_after), initial=0.0)))
    water_tolerance = 512 * np.finfo(float).eps * water_scale
    mismatch = float(np.max(np.abs(h_after - expected_h_after), initial=0.0))
    if mismatch > water_tolerance:
        raise TransportError(
            f"h_after_m is inconsistent with accepted water flux (max mismatch {mismatch:.6g} m)"
        )

    if scheme not in {"upwind", "muscl", "fct", "weno"}:
        raise TransportError("scheme must be 'upwind', 'muscl', 'fct', or 'weno'")
    concentration_before = _concentration(h_before, suspended_before, dry_tolerance_m)
    upwind_cx, upwind_cy = _upwind_face_concentrations(
        concentration_before, x_volume, y_volume, flux
    )
    low_x = x_volume * upwind_cx
    low_y = y_volume * upwind_cy
    if scheme == "upwind":
        sediment_x, sediment_y = low_x, low_y
    else:
        half_flux = AcceptedWaterFlux(
            0.5 * x_volume, 0.5 * y_volume, flux.periodic_x, flux.periodic_y
        )
        h_half = water_depth_from_accepted_flux(h_before, half_flux, cell_area_m2=cell_area_m2)
        half_cell_mass = _cell_mass_after_flux(
            suspended_before * cell_area_m2,
            0.5 * low_x,
            0.5 * low_y,
        )
        half_tolerance = 1024 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(suspended_before * cell_area_m2), initial=0.0))
        )
        if np.min(half_cell_mass, initial=0.0) < -half_tolerance:
            raise TransportError("MUSCL midpoint produces negative suspended mass; reduce the transport CFL")
        suspended_half = np.maximum(half_cell_mass, 0.0) / cell_area_m2
        concentration_half = _concentration(h_half, suspended_half, dry_tolerance_m)
        if scheme == "weno":
            cx, cy = _weno_face_concentrations(
                concentration_half, x_volume, y_volume, flux
            )
        else:
            cx, cy = _muscl_face_concentrations(
                concentration_half, x_volume, y_volume, flux
            )
        high_x = x_volume * cx
        high_y = y_volume * cy
        if scheme == "muscl":
            sediment_x, sediment_y = high_x, high_y
        else:
            sediment_x, sediment_y = _fct_face_masses(
                concentration_before,
                h_after,
                suspended_before * cell_area_m2,
                low_x,
                low_y,
                high_x,
                high_y,
                flux,
                cell_area_m2,
                dry_tolerance_m,
            )
    cell_mass_before = suspended_before * cell_area_m2
    cell_mass_after = _cell_mass_after_flux(cell_mass_before, sediment_x, sediment_y)

    mass_scale = max(1.0, float(np.max(np.abs(cell_mass_before), initial=0.0)))
    mass_tolerance = 1024 * np.finfo(float).eps * mass_scale
    if np.min(cell_mass_after, initial=0.0) < -mass_tolerance:
        raise TransportError("accepted step produces negative suspended mass; reduce the transport CFL")
    cell_mass_after = np.maximum(cell_mass_after, 0.0)
    redistributed_dry_mass = 0.0
    if wet_dry_policy == "redistribute":
        redistributed_dry_mass = _redistribute_numerically_dry_mass(
            cell_mass_after, h_after, sediment_x, sediment_y,
            x_volume, y_volume, flux, dry_tolerance_m,
        )
    suspended_after = cell_mass_after / cell_area_m2
    if (wet_dry_policy != "retain"
            and np.any((h_after <= dry_tolerance_m)
                       & (suspended_after > mass_tolerance / cell_area_m2))):
        raise TransportError("accepted step leaves suspended mass in a dry cell")
    concentration_after = _concentration(h_after, suspended_after, dry_tolerance_m)
    inactive_dry_mass = float(np.sum(cell_mass_after[h_after <= dry_tolerance_m]))

    boundary_out = math.fsum(
        [float(np.sum(sediment_x[:, -1] - sediment_x[:, 0])),
         float(np.sum(sediment_y[-1, :] - sediment_y[0, :]))]
    )
    boundary_in = -boundary_out
    mass_before = float(np.sum(cell_mass_before))
    mass_after = float(np.sum(cell_mass_after))
    residual = mass_after - mass_before - boundary_in
    return PassiveTransportResult(
        suspended_mass_kg_m2=suspended_after,
        concentration_kg_m3=concentration_after,
        sediment_face_mass_x_kg=sediment_x,
        sediment_face_mass_y_kg=sediment_y,
        mass_before_kg=mass_before,
        mass_after_kg=mass_after,
        boundary_mass_in_kg=boundary_in,
        balance_residual_kg=residual,
        minimum_concentration_kg_m3=float(np.min(concentration_after, initial=0.0)),
        maximum_concentration_kg_m3=float(np.max(concentration_after, initial=0.0)),
        scheme=scheme,
        redistributed_dry_mass_kg=redistributed_dry_mass,
        inactive_dry_mass_kg=inactive_dry_mass,
    )
