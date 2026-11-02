"""Generic FLOW 2DH ownership composer for direct-vector formula outputs.

The constitutive formula has already produced solid-mass quantities.  This
operator deliberately performs neither a density conversion nor an external
direction rotation.  It only normalizes the three output ownership patterns
used by the current-only direct Delft3D4 formulas and applies the FLOW route
ordering:

``tra_bedload``
    Scale the bed and non-bed channels, fold the non-bed channel into the bed
    vector, and apply the active-bed fraction once to the resulting total.

``tra_combine``
    Apply the active-bed fraction once to the scaled bed vector and retain a
    pre-exchange equilibrium mass concentration for Q4.  An explicit
    suspended vector is converted with ``|q_s|/(|u| + eps)/h``; a formula
    concentration is retained directly.  The active fraction and ``Sus``
    factor are each applied once to that Q4 value.

This module is a small backend-neutral oracle.  Formula selection uses stable
method strings; the only integer codes are a dedicated output-kind enum, not
the overlapping historical Delft3D ``TraFrm``/internal-law numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from types import MappingProxyType


class DirectVectorRouteError(ValueError):
    """Raised when a route cannot be interpreted without changing physics."""


class DirectVectorOutputKind(IntEnum):
    """Formula-output ownership, independent of historical numeric law IDs."""

    ZERO_NONBED = 0
    EXPLICIT_SUSPENDED_VECTOR = 1
    EQUILIBRIUM_CONCENTRATION = 2


ROUTE_TRA_BEDLOAD = "tra_bedload"
ROUTE_TRA_COMBINE = "tra_combine"
ROUTES = frozenset((ROUTE_TRA_BEDLOAD, ROUTE_TRA_COMBINE))

METHOD_SOULSBY = "soulsby_current"
METHOD_SOULSBY_VAN_RIJN = "soulsby_van_rijn_current"
METHOD_BIJKER = "bijker_current"
METHOD_VAN_RIJN_1993 = "van_rijn_1993_current"
METHOD_VAN_THIEL_VAN_RIJN_2008 = "van_thiel_van_rijn_2008_current"
METHOD_XBEACH_SOULSBY_VAN_RIJN = "xbeach_soulsby_van_rijn_current"

METHOD_OUTPUT_KINDS = MappingProxyType({
    METHOD_SOULSBY: DirectVectorOutputKind.ZERO_NONBED,
    METHOD_SOULSBY_VAN_RIJN: DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR,
    METHOD_BIJKER: DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR,
    METHOD_VAN_RIJN_1993: DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
    METHOD_VAN_THIEL_VAN_RIJN_2008:
        DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
    METHOD_XBEACH_SOULSBY_VAN_RIJN:
        DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION,
})


@dataclass(frozen=True)
class DirectVectorRouteResult:
    """One normalized, active-fraction-weighted route result."""

    method: str
    output_kind: DirectVectorOutputKind
    route: str
    bed_mass_x: float
    bed_mass_y: float
    q4_equilibrium_mass_concentration: float


def resolve_direct_vector_output_kind(
        method: str) -> DirectVectorOutputKind:
    """Resolve an exact method string to its non-bed output ownership."""
    if type(method) is not str:
        raise DirectVectorRouteError(
            "direct-vector method must be an exact string, not a numeric law ID"
        )
    try:
        return METHOD_OUTPUT_KINDS[method]
    except KeyError as error:
        raise DirectVectorRouteError(
            f"unsupported direct-vector method: {method!r}"
        ) from error


def _finite(name: str, value: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise DirectVectorRouteError(f"{name} must be finite numeric data")
    return float(value)


def compose_direct_vector_route_v66(
    *,
    method: str,
    route: str,
    depth: float,
    velocity_x: float,
    velocity_y: float,
    raw_bed_mass_x: float,
    raw_bed_mass_y: float,
    raw_suspended_mass_x: float = 0.0,
    raw_suspended_mass_y: float = 0.0,
    raw_equilibrium_mass_concentration: float = 0.0,
    bed_factor: float = 1.0,
    suspended_factor: float = 1.0,
    active_fraction: float = 1.0,
    speed_epsilon: float = 1.0e-6,
) -> DirectVectorRouteResult:
    """Compose one direct-vector formula output with exact FLOW ownership.

    All raw quantities are already solid-mass-valued.  A dry cell is accepted
    only when every constitutive output is exactly zero; this makes an
    impossible dry transport source fail closed instead of disappearing in a
    later division or wet/dry mask.
    """
    output_kind = resolve_direct_vector_output_kind(method)
    if type(route) is not str or route not in ROUTES:
        raise DirectVectorRouteError(f"unsupported direct-vector route: {route!r}")

    values = {
        "depth": depth,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "raw_bed_mass_x": raw_bed_mass_x,
        "raw_bed_mass_y": raw_bed_mass_y,
        "raw_suspended_mass_x": raw_suspended_mass_x,
        "raw_suspended_mass_y": raw_suspended_mass_y,
        "raw_equilibrium_mass_concentration":
            raw_equilibrium_mass_concentration,
        "bed_factor": bed_factor,
        "suspended_factor": suspended_factor,
        "active_fraction": active_fraction,
        "speed_epsilon": speed_epsilon,
    }
    numeric = {name: _finite(name, value) for name, value in values.items()}
    depth = numeric["depth"]
    velocity_x = numeric["velocity_x"]
    velocity_y = numeric["velocity_y"]
    raw_bed_mass_x = numeric["raw_bed_mass_x"]
    raw_bed_mass_y = numeric["raw_bed_mass_y"]
    raw_suspended_mass_x = numeric["raw_suspended_mass_x"]
    raw_suspended_mass_y = numeric["raw_suspended_mass_y"]
    raw_equilibrium_mass_concentration = numeric[
        "raw_equilibrium_mass_concentration"
    ]
    bed_factor = numeric["bed_factor"]
    suspended_factor = numeric["suspended_factor"]
    active_fraction = numeric["active_fraction"]
    speed_epsilon = numeric["speed_epsilon"]

    if depth < 0.0:
        raise DirectVectorRouteError("depth must be nonnegative")
    if not 0.0 <= active_fraction <= 1.0:
        raise DirectVectorRouteError("active_fraction must be in [0, 1]")
    if speed_epsilon <= 0.0:
        raise DirectVectorRouteError("speed_epsilon must be positive")
    if raw_equilibrium_mass_concentration < 0.0:
        raise DirectVectorRouteError(
            "raw equilibrium mass concentration must be nonnegative"
        )

    vector_nonzero = (
        raw_suspended_mass_x != 0.0 or raw_suspended_mass_y != 0.0
    )
    concentration_nonzero = raw_equilibrium_mass_concentration != 0.0
    if output_kind is DirectVectorOutputKind.ZERO_NONBED:
        if vector_nonzero or concentration_nonzero:
            raise DirectVectorRouteError(
                f"{method} owns an exactly zero non-bed channel"
            )
    elif output_kind is DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR:
        if concentration_nonzero:
            raise DirectVectorRouteError(
                f"{method} owns an explicit suspended vector, not a formula "
                "equilibrium concentration"
            )
    elif vector_nonzero:
        raise DirectVectorRouteError(
            f"{method} owns a formula equilibrium concentration, not an "
            "explicit suspended vector"
        )

    all_source_zero = not any((
        raw_bed_mass_x != 0.0,
        raw_bed_mass_y != 0.0,
        vector_nonzero,
        concentration_nonzero,
    ))
    if depth == 0.0:
        if not all_source_zero:
            raise DirectVectorRouteError(
                "dry cells must not carry a constitutive transport source"
            )
        return DirectVectorRouteResult(
            method, output_kind, route, 0.0, 0.0, 0.0,
        )

    scaled_bed_x = bed_factor * raw_bed_mass_x
    scaled_bed_y = bed_factor * raw_bed_mass_y
    if output_kind is DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR:
        nonbed_x = suspended_factor * raw_suspended_mass_x
        nonbed_y = suspended_factor * raw_suspended_mass_y
    elif output_kind is DirectVectorOutputKind.EQUILIBRIUM_CONCENTRATION:
        scaled_concentration = (
            suspended_factor * raw_equilibrium_mass_concentration
        )
        nonbed_x = scaled_concentration * depth * velocity_x
        nonbed_y = scaled_concentration * depth * velocity_y
    else:
        scaled_concentration = 0.0
        nonbed_x = 0.0
        nonbed_y = 0.0

    if route == ROUTE_TRA_BEDLOAD:
        bed_x = active_fraction * (scaled_bed_x + nonbed_x)
        bed_y = active_fraction * (scaled_bed_y + nonbed_y)
        q4_concentration = 0.0
    else:
        bed_x = active_fraction * scaled_bed_x
        bed_y = active_fraction * scaled_bed_y
        if output_kind is DirectVectorOutputKind.ZERO_NONBED:
            q4_concentration = 0.0
        elif output_kind is DirectVectorOutputKind.EXPLICIT_SUSPENDED_VECTOR:
            suspended_magnitude = math.hypot(
                raw_suspended_mass_x, raw_suspended_mass_y,
            )
            speed = math.hypot(velocity_x, velocity_y)
            q4_concentration = active_fraction * suspended_factor * (
                suspended_magnitude / (speed + speed_epsilon) / depth
            )
        else:
            q4_concentration = active_fraction * scaled_concentration

    result_values = (bed_x, bed_y, q4_concentration)
    if not all(math.isfinite(value) for value in result_values):
        raise DirectVectorRouteError(
            "direct-vector route composition overflowed"
        )
    return DirectVectorRouteResult(
        method, output_kind, route, bed_x, bed_y, q4_concentration,
    )


__all__ = [
    "DirectVectorOutputKind",
    "DirectVectorRouteError",
    "DirectVectorRouteResult",
    "METHOD_BIJKER",
    "METHOD_OUTPUT_KINDS",
    "METHOD_SOULSBY",
    "METHOD_SOULSBY_VAN_RIJN",
    "METHOD_VAN_RIJN_1993",
    "METHOD_VAN_THIEL_VAN_RIJN_2008",
    "METHOD_XBEACH_SOULSBY_VAN_RIJN",
    "ROUTES",
    "ROUTE_TRA_BEDLOAD",
    "ROUTE_TRA_COMBINE",
    "compose_direct_vector_route_v66",
    "resolve_direct_vector_output_kind",
]
