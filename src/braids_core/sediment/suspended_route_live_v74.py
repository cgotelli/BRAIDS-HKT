"""Bounded Q4 ownership gate for split Delft3D FLOW 2DH transport routes.

This module consumes the already mass-valued Q3 direct-vector hand-off.  It
does not reimplement the constitutive formulae.  Its only numerical work is
the locked ``eqtran`` route split followed, for ``TRA_COMBINE``, by the 2DH
``soursin_2d`` mass-form exchange update.

The v74 envelope is intentionally small: current-only Bijker (an explicit
suspended vector) and current-only TraFrm 19/20 (an equilibrium suspended
concentration), fully wet fixed cells, no horizontal fluxes, and at most four
accepted updates.  Settling-law evaluation, diffusion, open boundaries,
wet/dry transfer, AMR, mixtures and product admission remain outside scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping, Sequence

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids-q4-split-suspended-route-owner-v74-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
MAX_ACCEPTED_UPDATES = 4

ROUTE_OFF = 0
ROUTE_TOTAL = 1
ROUTE_COMBINE = 3
ROUTES = frozenset((ROUTE_OFF, ROUTE_TOTAL, ROUTE_COMBINE))

OUTPUT_SUSPENDED_VECTOR = 1
OUTPUT_EQUILIBRIUM_CONCENTRATION = 2

METHOD_BIJKER_CURRENT = "capacity.bijker"
METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT = "capacity.van_thiel_van_rijn"
METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT = (
    "capacity.xbeach_soulsby_van_rijn"
)

METHOD_OUTPUT_KIND = {
    METHOD_BIJKER_CURRENT: OUTPUT_SUSPENDED_VECTOR,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT:
        OUTPUT_EQUILIBRIUM_CONCENTRATION,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT:
        OUTPUT_EQUILIBRIUM_CONCENTRATION,
}

BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EPS = 1.0e-6

# Locked, read-only source anchors.  The module verifies these bytes before it
# resolves an execution identity; it never writes below resources/Delft3D-main.
LOCKED_SOURCES = {
    "sediment_basics": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_data/src/sediment_basics_module.f90",
        "73d9e3889dbdbcaa401517abb4a6fb9c893d9ad56ec14892ec4297d7d348ba64",
    ),
    "rdsed": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdsed.f90",
        "b3de2a67842935bee704eecee329e0e0a71fdd0bb95704a15f9d2ef567514fd2",
    ),
    "rdstm": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdstm.f90",
        "e2e54cf21ceef4e2b1b2f9e3cc26d8c3eb4df425c324b0b9b47828e7af7858b0",
    ),
    "eqtran": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "tranb5": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tranb5.f90",
        "ca5933ccda6592174c627c81d2d10e977064dd47f96560d349b1a3fca50a2070",
    ),
    "trab19": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab19.f90",
        "bfc530b573b95c4467a9de4bd34e25757f3990d489426db995b28d4b7ad34c63",
    ),
    "trab20": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/trab20.f90",
        "a7cbf17f232fc6edae5a1cf5f7c9b2ca1e1e0626aee6dbb5270efc1c245e4d19",
    ),
    "soursin_2d": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/soursin_2d.f90",
        "a1cb9ea0b9dde77251ba2751dc5ec285e7e74171569e7b65a75d5ced1ada51ea",
    ),
    "erosed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/erosed.f90",
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    ),
}


class SuspendedRouteV74Error(ValueError):
    """Raised before an ambiguous or unsupported route can be executed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locked_sources_v74(root: str | Path | None = None) -> tuple[dict, ...]:
    """Authenticate every locked source file used by the bounded v74 gate."""
    repository = (Path(root).resolve() if root is not None
                  else Path(__file__).resolve().parents[2])
    rows: list[dict] = []
    for symbol, (relative, expected) in LOCKED_SOURCES.items():
        path = repository / relative
        if not path.is_file():
            raise SuspendedRouteV74Error(
                f"locked Delft3D source is missing: {relative}"
            )
        actual = _file_sha256(path)
        if actual != expected:
            raise SuspendedRouteV74Error(
                f"locked Delft3D source hash differs: {relative}"
            )
        rows.append({
            "symbol": symbol, "path": relative, "sha256": actual,
            "read_only_reference": True,
        })
    return tuple(rows)


def route_semantic_document_v74() -> dict[str, object]:
    """Static source-audited ownership contract hashed into each execution."""
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "locked_transport_bits": {
            "TRA_NONE": ROUTE_OFF,
            "TRA_BEDLOAD_total": ROUTE_TOTAL,
            "TRA_COMBINE": ROUTE_COMBINE,
        },
        "off_caveat": (
            "TRA_NONE is a locked parser sentinel for non-sediment blocks; "
            "v74 exposes an explicit zero-output BRAIDS route but does not "
            "claim a runtime Delft3D sediment fraction with tratyp=0"
        ),
        "total": (
            "TotalLoad=true selects TRA_BEDLOAD; eqtran folds the formula "
            "suspended half into the algebraic total-load vector"
        ),
        "combine": (
            "TRA_COMBINE keeps the bed vector separate and gives the formula "
            "suspended half to the 2DH scalar/exchange owner"
        ),
        "split_outputs": {
            METHOD_BIJKER_CURRENT: (
                "sscu/sscv vector; combine converts its magnitude to conc2d"
            ),
            METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT:
                "equi_conc cesus; combine retains cesus as conc2d",
            METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT:
                "equi_conc cesus; combine retains cesus as conc2d",
        },
        "factor_order": [
            "Q3 formula output already in solid-mass units",
            "Bed and Sus factors exactly once",
            "surface fraction exactly once",
            "total folds or combine retains, never both",
            "combine-only soursin_2d mass-form exchange",
        ],
        "post_q3_density_conversion_count": 0,
        "external_secondary_rotation": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "scope": {
            "forcing": "current_only_no_wave",
            "mesh": "fixed_fully_wet_cells_without_horizontal_flux",
            "settling": "supplied_velocity_not_settling_law_qualification",
            "exchange": "locked_soursin_2d_algebra_limiter_inactive_or_recorded",
            "production_admission": False,
        },
        "unsupported": [
            "TRA_ADVDIFF cohesive erosilt-only route",
            "wave-enabled Bijker or TraFrm 19/20",
            "settling-law evaluation and hindering",
            "horizontal advection/diffusion and open boundaries",
            "wet/dry transfer, AMR, mixtures and bed layers",
            "normal runner, plugin and QGIS admission",
        ],
    }


SEMANTIC_ID = _fingerprint(route_semantic_document_v74())


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise SuspendedRouteV74Error(f"{name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise SuspendedRouteV74Error(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise SuspendedRouteV74Error(f"{name} must be nonnegative")
    return result


def _exact_sequence(value: object, name: str) -> tuple:
    if (not isinstance(value, Sequence) or
            isinstance(value, (str, bytes, bytearray))):
        raise SuspendedRouteV74Error(f"{name} must be an ordered sequence")
    return tuple(value)


def resolve_route_execution_v74(
    *, method_ids: Sequence[str], routes: Sequence[int], backend: str,
    q3_handoff_fingerprint: str, bed_factor: float = 1.0,
    suspended_factor: float = 1.0, speed_epsilon: float = _EPS,
    fac_tsd: float = 1.0, sediment_threshold_m: float = 0.5,
    minimum_reference_depth_m: float = 0.01,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Resolve the immutable v74 route identity and verify locked sources."""
    methods = _exact_sequence(method_ids, "method_ids")
    route_values = _exact_sequence(routes, "routes")
    if not methods or len(methods) != len(route_values):
        raise SuspendedRouteV74Error(
            "method_ids and routes must have the same nonzero length"
        )
    if any(type(value) is not str or value not in METHOD_OUTPUT_KIND
           for value in methods):
        raise SuspendedRouteV74Error(
            "v74 supports only current-only Bijker and TraFrm 19/20"
        )
    if any(type(value) is not int or value not in ROUTES
           for value in route_values):
        raise SuspendedRouteV74Error(
            "v74 routes must be exact off/total/combine integer identities"
        )
    if type(backend) is not str or backend not in BACKENDS:
        raise SuspendedRouteV74Error(
            "backend must be explicit CPU or implicit_fwave_midpoint_gpu"
        )
    if (type(q3_handoff_fingerprint) is not str or
            not _SHA.fullmatch(q3_handoff_fingerprint)):
        raise SuspendedRouteV74Error(
            "Q3 handoff fingerprint must be an exact SHA-256 identity"
        )
    numeric = {
        "bed_factor": _finite(bed_factor, "bed_factor", nonnegative=True),
        "suspended_factor": _finite(
            suspended_factor, "suspended_factor", nonnegative=True),
        "speed_epsilon": _finite(
            speed_epsilon, "speed_epsilon", positive=True),
        "fac_tsd": _finite(fac_tsd, "fac_tsd", positive=True),
        "sediment_threshold_m": _finite(
            sediment_threshold_m, "sediment_threshold_m", nonnegative=True),
        "minimum_reference_depth_m": _finite(
            minimum_reference_depth_m, "minimum_reference_depth_m",
            positive=True),
    }
    source_rows = verify_locked_sources_v74(root)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": backend,
        "automatic_backend_fallback": False,
        "gpu_integrator": (
            "implicit_fwave_midpoint" if backend.endswith("_gpu") else None
        ),
        "method_ids": list(methods),
        "output_kinds": [METHOD_OUTPUT_KIND[value] for value in methods],
        "routes": list(route_values),
        "q3_handoff_fingerprint": q3_handoff_fingerprint,
        **numeric,
        "locked_sources": list(source_rows),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_route_execution_v74(value: Mapping[str, object], *,
                             root: str | Path | None = None) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SuspendedRouteV74Error("v74 execution must be an object")
    expected = {
        "schema", "scope_id", "semantic_id", "backend",
        "automatic_backend_fallback", "gpu_integrator", "method_ids",
        "output_kinds", "routes", "q3_handoff_fingerprint", "bed_factor",
        "suspended_factor", "speed_epsilon", "fac_tsd",
        "sediment_threshold_m", "minimum_reference_depth_m",
        "locked_sources", "accepted_update_limit", "normal_product_admission",
        "fingerprint",
    }
    if set(value) != expected:
        raise SuspendedRouteV74Error("v74 execution field inventory differs")
    rebuilt = resolve_route_execution_v74(
        method_ids=value["method_ids"], routes=value["routes"],
        backend=value["backend"],
        q3_handoff_fingerprint=value["q3_handoff_fingerprint"],
        bed_factor=value["bed_factor"],
        suspended_factor=value["suspended_factor"],
        speed_epsilon=value["speed_epsilon"], fac_tsd=value["fac_tsd"],
        sediment_threshold_m=value["sediment_threshold_m"],
        minimum_reference_depth_m=value["minimum_reference_depth_m"],
        root=root,
    )
    if rebuilt != dict(value):
        raise SuspendedRouteV74Error(
            "v74 execution identity or locked source receipt differs"
        )
    return rebuilt


def _array(value: object, shape: tuple[int, ...], name: str,
           *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise SuspendedRouteV74Error(
            f"{name} must be finite float64 data with shape {shape}"
        )
    if nonnegative and np.any(result < 0.0):
        raise SuspendedRouteV74Error(f"{name} must be nonnegative")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteHandoffV74:
    execution_fingerprint: str
    q3_handoff_fingerprint: str
    bed_capacity_x: np.ndarray
    bed_capacity_y: np.ndarray
    equilibrium_concentration: np.ndarray
    suspended_equivalent_x: np.ndarray
    suspended_equivalent_y: np.ndarray
    bed_suspended_owner_count: np.ndarray
    water_suspended_owner_count: np.ndarray
    suspended_factor_application_count: np.ndarray
    active_fraction_application_count: np.ndarray
    post_q3_density_conversion_count: int
    handoff_sha256: str


def _handoff_token(value: RouteHandoffV74) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "q3_handoff_fingerprint": value.q3_handoff_fingerprint,
        "post_q3_density_conversion_count":
            value.post_q3_density_conversion_count,
        "arrays_sha256": _array_digest(
            value.bed_capacity_x, value.bed_capacity_y,
            value.equilibrium_concentration,
            value.suspended_equivalent_x, value.suspended_equivalent_y,
            value.bed_suspended_owner_count,
            value.water_suspended_owner_count,
            value.suspended_factor_application_count,
            value.active_fraction_application_count,
        ),
    })


def compose_route_handoff_v74(
    *, execution: Mapping[str, object], depth_m: object,
    velocity_x_m_s: object, velocity_y_m_s: object,
    raw_bed_x_kg_m_s: object, raw_bed_y_kg_m_s: object,
    raw_suspended_x_kg_m_s: object, raw_suspended_y_kg_m_s: object,
    raw_equilibrium_concentration_kg_m3: object,
    active_fraction: object,
) -> RouteHandoffV74:
    """Consume one authenticated Q3 split without duplicating its non-bed half."""
    resolved = load_route_execution_v74(execution)
    methods = tuple(resolved["method_ids"])
    routes = tuple(resolved["routes"])
    nlane = len(methods)
    depth_raw = np.asarray(depth_m, dtype=np.float64)
    if depth_raw.ndim != 1 or depth_raw.size < 1:
        raise SuspendedRouteV74Error("depth must be a nonempty cell vector")
    ncell = int(depth_raw.size)
    depth = _array(depth_raw, (ncell,), "depth", nonnegative=True)
    u = _array(velocity_x_m_s, (ncell,), "velocity_x")
    v = _array(velocity_y_m_s, (ncell,), "velocity_y")
    shape = (nlane, ncell)
    bed_x = _array(raw_bed_x_kg_m_s, shape, "raw_bed_x")
    bed_y = _array(raw_bed_y_kg_m_s, shape, "raw_bed_y")
    suspended_x = _array(
        raw_suspended_x_kg_m_s, shape, "raw_suspended_x")
    suspended_y = _array(
        raw_suspended_y_kg_m_s, shape, "raw_suspended_y")
    concentration = _array(
        raw_equilibrium_concentration_kg_m3, shape,
        "raw_equilibrium_concentration", nonnegative=True,
    )
    fraction = _array(active_fraction, shape, "active_fraction")
    if np.any((fraction < 0.0) | (fraction > 1.0)):
        raise SuspendedRouteV74Error("active_fraction must be in [0,1]")
    speed = np.hypot(u, v)
    if np.any(depth <= 0.0):
        raise SuspendedRouteV74Error(
            "v74 is a fully wet gate and requires positive depth"
        )

    out_bed_x = np.zeros(shape)
    out_bed_y = np.zeros(shape)
    out_eq = np.zeros(shape)
    equivalent_x = np.zeros(shape)
    equivalent_y = np.zeros(shape)
    bed_owner = np.zeros(nlane, dtype=np.int32)
    water_owner = np.zeros(nlane, dtype=np.int32)
    sus_factor_count = np.zeros(nlane, dtype=np.int32)
    fraction_count = np.zeros(nlane, dtype=np.int32)
    bed_factor = float(resolved["bed_factor"])
    sus_factor = float(resolved["suspended_factor"])
    epsilon = float(resolved["speed_epsilon"])

    for lane, (method, route) in enumerate(zip(methods, routes, strict=True)):
        output_kind = METHOD_OUTPUT_KIND[method]
        vector_nonzero = np.any(suspended_x[lane] != 0.0) or np.any(
            suspended_y[lane] != 0.0)
        concentration_nonzero = np.any(concentration[lane] != 0.0)
        if output_kind == OUTPUT_SUSPENDED_VECTOR:
            if concentration_nonzero:
                raise SuspendedRouteV74Error(
                    "Bijker owns a suspended vector, not equilibrium concentration"
                )
            scaled_nonbed_x = sus_factor * suspended_x[lane]
            scaled_nonbed_y = sus_factor * suspended_y[lane]
            derived_eq = (sus_factor * np.hypot(
                suspended_x[lane], suspended_y[lane]
            ) / (speed + epsilon) / depth)
        else:
            if vector_nonzero:
                raise SuspendedRouteV74Error(
                    "TraFrm 19/20 own equilibrium concentration, not a "
                    "suspended vector"
                )
            scaled_eq = sus_factor * concentration[lane]
            scaled_nonbed_x = scaled_eq * depth * u
            scaled_nonbed_y = scaled_eq * depth * v
            derived_eq = scaled_eq
        if route == ROUTE_OFF:
            continue
        scaled_bed_x = bed_factor * bed_x[lane]
        scaled_bed_y = bed_factor * bed_y[lane]
        sus_factor_count[lane] = 1
        fraction_count[lane] = 1
        equivalent_x[lane] = fraction[lane] * scaled_nonbed_x
        equivalent_y[lane] = fraction[lane] * scaled_nonbed_y
        if route == ROUTE_TOTAL:
            out_bed_x[lane] = fraction[lane] * (
                scaled_bed_x + scaled_nonbed_x)
            out_bed_y[lane] = fraction[lane] * (
                scaled_bed_y + scaled_nonbed_y)
            bed_owner[lane] = 1
        else:
            out_bed_x[lane] = fraction[lane] * scaled_bed_x
            out_bed_y[lane] = fraction[lane] * scaled_bed_y
            out_eq[lane] = fraction[lane] * derived_eq
            water_owner[lane] = 1

    arrays = [
        np.ascontiguousarray(value) for value in (
            out_bed_x, out_bed_y, out_eq, equivalent_x, equivalent_y,
            bed_owner, water_owner, sus_factor_count, fraction_count,
        )
    ]
    for value in arrays:
        value.setflags(write=False)
    placeholder = RouteHandoffV74(
        resolved["fingerprint"], resolved["q3_handoff_fingerprint"],
        *arrays[:5], *arrays[5:], 0, "",
    )
    return RouteHandoffV74(
        placeholder.execution_fingerprint,
        placeholder.q3_handoff_fingerprint,
        placeholder.bed_capacity_x, placeholder.bed_capacity_y,
        placeholder.equilibrium_concentration,
        placeholder.suspended_equivalent_x,
        placeholder.suspended_equivalent_y,
        placeholder.bed_suspended_owner_count,
        placeholder.water_suspended_owner_count,
        placeholder.suspended_factor_application_count,
        placeholder.active_fraction_application_count,
        placeholder.post_q3_density_conversion_count,
        _handoff_token(placeholder),
    )


def validate_route_handoff_v74(
    value: RouteHandoffV74, *, execution: Mapping[str, object],
) -> RouteHandoffV74:
    resolved = load_route_execution_v74(execution)
    if (not isinstance(value, RouteHandoffV74) or
            value.execution_fingerprint != resolved["fingerprint"] or
            value.q3_handoff_fingerprint !=
            resolved["q3_handoff_fingerprint"] or
            value.post_q3_density_conversion_count != 0 or
            value.handoff_sha256 != _handoff_token(value)):
        raise SuspendedRouteV74Error(
            "v74 handoff is stale, mutated, or execution-mismatched"
        )
    if np.any(
        value.bed_suspended_owner_count *
        value.water_suspended_owner_count != 0
    ):
        raise SuspendedRouteV74Error(
            "formula suspended contribution has duplicate owners"
        )
    return value


@dataclass(frozen=True)
class SuspendedRouteStateV74:
    execution_fingerprint: str
    bed_mass_kg_m2: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    suspended_exchange_ledger_kg: np.ndarray
    bed_exchange_ledger_kg: np.ndarray
    accepted_updates: int
    time_s: float
    state_sha256: str


def _state_token(value: SuspendedRouteStateV74) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "arrays_sha256": _array_digest(
            value.bed_mass_kg_m2, value.suspended_mass_kg_m2,
            value.suspended_exchange_ledger_kg,
            value.bed_exchange_ledger_kg,
        ),
    })


def resolve_route_state_v74(
    *, execution: Mapping[str, object], bed_mass_kg_m2: object,
    suspended_mass_kg_m2: object,
    suspended_exchange_ledger_kg: object | None = None,
    bed_exchange_ledger_kg: object | None = None,
    accepted_updates: int = 0, time_s: float = 0.0,
) -> SuspendedRouteStateV74:
    resolved = load_route_execution_v74(execution)
    nlane = len(resolved["method_ids"])
    raw = np.asarray(bed_mass_kg_m2, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != nlane or raw.shape[1] < 1:
        raise SuspendedRouteV74Error(
            "v74 phase inventories must have shape (route_lane,cell)"
        )
    shape = raw.shape
    bed = _array(raw, shape, "bed mass", nonnegative=True)
    suspended = _array(
        suspended_mass_kg_m2, shape, "suspended mass", nonnegative=True)
    water_ledger = _array(
        np.zeros(nlane) if suspended_exchange_ledger_kg is None
        else suspended_exchange_ledger_kg,
        (nlane,), "suspended exchange ledger",
    )
    bed_ledger = _array(
        np.zeros(nlane) if bed_exchange_ledger_kg is None
        else bed_exchange_ledger_kg,
        (nlane,), "bed exchange ledger",
    )
    if (isinstance(accepted_updates, bool) or type(accepted_updates) is not int or
            not 0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise SuspendedRouteV74Error("v74 accepted update count is invalid")
    accepted_time = _finite(time_s, "accepted time", nonnegative=True)
    placeholder = SuspendedRouteStateV74(
        resolved["fingerprint"], bed, suspended, water_ledger, bed_ledger,
        accepted_updates, accepted_time, "",
    )
    return SuspendedRouteStateV74(
        placeholder.execution_fingerprint, placeholder.bed_mass_kg_m2,
        placeholder.suspended_mass_kg_m2,
        placeholder.suspended_exchange_ledger_kg,
        placeholder.bed_exchange_ledger_kg, placeholder.accepted_updates,
        placeholder.time_s, _state_token(placeholder),
    )


def validate_route_state_v74(
    value: SuspendedRouteStateV74, *, execution: Mapping[str, object],
) -> SuspendedRouteStateV74:
    resolved = load_route_execution_v74(execution)
    if (not isinstance(value, SuspendedRouteStateV74) or
            value.execution_fingerprint != resolved["fingerprint"] or
            value.state_sha256 != _state_token(value)):
        raise SuspendedRouteV74Error(
            "v74 accepted state is stale, mutated, or execution-mismatched"
        )
    return value


def _f32(value: float) -> float:
    return float(np.float32(value))


@dataclass(frozen=True)
class SuspendedRouteTrialV74:
    execution_fingerprint: str
    parent_state_sha256: str
    handoff_sha256: str
    candidate: SuspendedRouteStateV74
    equilibrium_concentration_kg_m3: np.ndarray
    dimensionless_timescale: np.ndarray
    adaptation_velocity_m_s: np.ndarray
    transferred_to_suspension_kg_m2: np.ndarray
    limiter_correction_kg: np.ndarray
    mass_residual_kg: np.ndarray
    exchange_application_count: np.ndarray
    duplicate_suspended_owner_count: np.ndarray
    observed_parallel_workers: int
    trial_sha256: str


def _trial_token(value: SuspendedRouteTrialV74) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "handoff_sha256": value.handoff_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "observed_parallel_workers": value.observed_parallel_workers,
        "arrays_sha256": _array_digest(
            value.equilibrium_concentration_kg_m3,
            value.dimensionless_timescale,
            value.adaptation_velocity_m_s,
            value.transferred_to_suspension_kg_m2,
            value.limiter_correction_kg, value.mass_residual_kg,
            value.exchange_application_count,
            value.duplicate_suspended_owner_count,
        ),
    })


def begin_route_trial_v74(
    *, execution: Mapping[str, object], parent: SuspendedRouteStateV74,
    handoff: RouteHandoffV74, depth_before_m: object,
    depth_after_m: object, speed_m_s: object, shear_velocity_m_s: object,
    settling_velocity_m_s: object, cell_area_m2: object, dt_s: float,
    cancelled: bool = False,
) -> SuspendedRouteTrialV74:
    """Build one immutable candidate; the accepted parent is never mutated."""
    resolved = load_route_execution_v74(execution)
    validate_route_state_v74(parent, execution=resolved)
    validate_route_handoff_v74(handoff, execution=resolved)
    if cancelled:
        raise SuspendedRouteV74Error(
            "v74 update cancelled before candidate publication"
        )
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        raise SuspendedRouteV74Error("v74 accepted update limit was reached")
    dt = _finite(dt_s, "dt_s", positive=True)
    nlane, ncell = parent.bed_mass_kg_m2.shape
    if handoff.equilibrium_concentration.shape != (nlane, ncell):
        raise SuspendedRouteV74Error("v74 handoff/state topology differs")
    h0 = _array(depth_before_m, (ncell,), "depth_before", nonnegative=True)
    h1 = _array(depth_after_m, (ncell,), "depth_after", nonnegative=True)
    if np.any(h0 <= 0.0) or np.any(h1 <= 0.0):
        raise SuspendedRouteV74Error(
            "v74 is a fully wet gate and requires positive endpoint depths"
        )
    speed = _array(speed_m_s, (ncell,), "speed", nonnegative=True)
    shear = _array(
        shear_velocity_m_s, (ncell,), "shear velocity", nonnegative=True)
    area = _array(cell_area_m2, (ncell,), "cell area")
    if np.any(area <= 0.0):
        raise SuspendedRouteV74Error("cell area must be positive")
    settling_raw = np.asarray(settling_velocity_m_s, dtype=np.float64)
    if settling_raw.shape == (nlane,):
        settling_raw = np.broadcast_to(settling_raw[:, None], (nlane, ncell))
    settling = _array(
        settling_raw, (nlane, ncell), "settling velocity", nonnegative=True)

    routes = np.asarray(resolved["routes"], dtype=np.int32)
    combine = routes == ROUTE_COMBINE
    active_cell = h1 > float(resolved["sediment_threshold_m"])
    active = combine[:, None] & active_cell[None, :]
    safe_h0 = np.maximum(float(resolved["minimum_reference_depth_m"]), h0)
    safe_h1 = np.maximum(float(resolved["minimum_reference_depth_m"]), h1)
    wsl = np.maximum(1.0e-3, settling)
    moving = active & (speed[None, :] > _EPS) & (shear[None, :] > _EPS)
    timescale = np.zeros((nlane, ncell))
    if np.any(moving):
        speed2 = np.broadcast_to(speed[None, :], (nlane, ncell))
        shear2 = np.broadcast_to(shear[None, :], (nlane, ncell))
        ratio = np.minimum(shear2[moving] / speed2[moving], 0.15)
        w = np.where(
            shear2[moving] > wsl[moving],
            wsl[moving] / shear2[moving], 1.0,
        )
        x2 = w*w
        x3 = x2*w
        exponent = (
            (_f32(1.547) - _f32(20.12)*ratio)*x3
            + (_f32(326.832)*ratio**_f32(2.2047)-_f32(0.2))*x2
            + (_f32(0.1385)*np.log(ratio)-_f32(6.4061))*w
            + (_f32(0.5467)*ratio+_f32(2.1963))
        )
        timescale[moving] = w*np.exp(exponent)
    if np.any(moving & ((timescale <= 0.0) | ~np.isfinite(timescale))):
        raise SuspendedRouteV74Error("Galappatti timescale is invalid")
    hots = np.zeros_like(timescale)
    hots[moving] = (
        wsl[moving] / timescale[moving] / float(resolved["fac_tsd"])
    )
    explicit = np.zeros_like(timescale)
    explicit[moving] = (
        handoff.equilibrium_concentration[moving] * hots[moving] /
        np.broadcast_to(safe_h0[None, :], (nlane, ncell))[moving]
    )
    total_sink = np.zeros_like(timescale)
    safe_h1_2 = np.broadcast_to(safe_h1[None, :], (nlane, ncell))
    total_sink[moving] = hots[moving] / safe_h1_2[moving]
    settling_only = active & ~moving
    total_sink[settling_only] = wsl[settling_only] / safe_h1_2[settling_only]

    old_suspended = parent.suspended_mass_kg_m2
    old_bed = parent.bed_mass_kg_m2
    total = old_suspended + old_bed
    candidate_raw = (
        old_suspended + dt*h0[None, :]*explicit
    ) / (1.0+dt*total_sink)
    candidate_suspended = np.where(
        active, np.minimum(np.maximum(candidate_raw, 0.0), total),
        old_suspended,
    )
    candidate_bed = total-candidate_suspended
    transfer = candidate_suspended-old_suspended
    water_delta = np.sum(transfer*area[None, :], axis=1)
    bed_delta = -water_delta
    before = np.sum(total*area[None, :], axis=1)
    after = np.sum((candidate_suspended+candidate_bed)*area[None, :], axis=1)
    residual = after-before
    scale = np.maximum(1.0, np.maximum(np.abs(before), np.abs(after)))
    if np.any(np.abs(residual) > 4096.0*np.finfo(float).eps*scale):
        raise SuspendedRouteV74Error("v74 phase mass ledger did not close")
    limiter = np.sum(
        np.abs(candidate_suspended-candidate_raw)*area[None, :], axis=1)
    exchange_count = combine.astype(np.int32)
    duplicate = (
        handoff.bed_suspended_owner_count *
        handoff.water_suspended_owner_count
    ).astype(np.int32)
    if np.any(duplicate != 0):
        raise SuspendedRouteV74Error(
            "formula suspended contribution was applied to both phase routes"
        )
    candidate = resolve_route_state_v74(
        execution=resolved, bed_mass_kg_m2=candidate_bed,
        suspended_mass_kg_m2=candidate_suspended,
        suspended_exchange_ledger_kg=(
            parent.suspended_exchange_ledger_kg+water_delta),
        bed_exchange_ledger_kg=parent.bed_exchange_ledger_kg+bed_delta,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt,
    )
    trial_arrays = [
        np.ascontiguousarray(value) for value in (
            handoff.equilibrium_concentration, timescale, hots, transfer,
            limiter, residual, exchange_count, duplicate,
        )
    ]
    for value in trial_arrays:
        value.setflags(write=False)
    placeholder = SuspendedRouteTrialV74(
        resolved["fingerprint"], parent.state_sha256, handoff.handoff_sha256,
        candidate, *trial_arrays, 1, "",
    )
    return SuspendedRouteTrialV74(
        placeholder.execution_fingerprint, placeholder.parent_state_sha256,
        placeholder.handoff_sha256, placeholder.candidate,
        placeholder.equilibrium_concentration_kg_m3,
        placeholder.dimensionless_timescale,
        placeholder.adaptation_velocity_m_s,
        placeholder.transferred_to_suspension_kg_m2,
        placeholder.limiter_correction_kg, placeholder.mass_residual_kg,
        placeholder.exchange_application_count,
        placeholder.duplicate_suspended_owner_count,
        placeholder.observed_parallel_workers, _trial_token(placeholder),
    )


def _validate_trial(parent: SuspendedRouteStateV74,
                    trial: SuspendedRouteTrialV74,
                    execution: Mapping[str, object]) -> None:
    resolved = load_route_execution_v74(execution)
    validate_route_state_v74(parent, execution=resolved)
    if (not isinstance(trial, SuspendedRouteTrialV74) or
            trial.execution_fingerprint != resolved["fingerprint"] or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise SuspendedRouteV74Error(
            "v74 trial is stale, mutated, or has a different parent"
        )
    validate_route_state_v74(trial.candidate, execution=resolved)


def commit_route_trial_v74(
    parent: SuspendedRouteStateV74, trial: SuspendedRouteTrialV74, *,
    execution: Mapping[str, object],
) -> SuspendedRouteStateV74:
    _validate_trial(parent, trial, execution)
    return trial.candidate


def rollback_route_trial_v74(
    parent: SuspendedRouteStateV74, trial: SuspendedRouteTrialV74, *,
    execution: Mapping[str, object],
) -> SuspendedRouteStateV74:
    _validate_trial(parent, trial, execution)
    return parent


def cancel_route_trial_v74(
    parent: SuspendedRouteStateV74, trial: SuspendedRouteTrialV74, *,
    execution: Mapping[str, object],
) -> SuspendedRouteStateV74:
    return rollback_route_trial_v74(parent, trial, execution=execution)


def route_checkpoint_document_v74(
    state: SuspendedRouteStateV74, *, execution: Mapping[str, object],
) -> dict[str, object]:
    resolved = load_route_execution_v74(execution)
    validate_route_state_v74(state, execution=resolved)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "q3_handoff_fingerprint": resolved["q3_handoff_fingerprint"],
        "accepted_updates": state.accepted_updates,
        "time_s": state.time_s,
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "suspended_mass_kg_m2": state.suspended_mass_kg_m2.tolist(),
        "suspended_exchange_ledger_kg":
            state.suspended_exchange_ledger_kg.tolist(),
        "bed_exchange_ledger_kg": state.bed_exchange_ledger_kg.tolist(),
        "state_sha256": state.state_sha256,
        "transaction_schema": "complete_accepted_state_no_trial_arrays",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_route_checkpoint_v74(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> SuspendedRouteStateV74:
    resolved = load_route_execution_v74(execution)
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "q3_handoff_fingerprint", "accepted_updates", "time_s",
        "bed_mass_kg_m2", "suspended_mass_kg_m2",
        "suspended_exchange_ledger_kg", "bed_exchange_ledger_kg",
        "state_sha256", "transaction_schema", "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SuspendedRouteV74Error("v74 checkpoint inventory differs")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or
            value["scope_id"] != CHECKPOINT_SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["q3_handoff_fingerprint"] !=
            resolved["q3_handoff_fingerprint"] or
            value["transaction_schema"] !=
            "complete_accepted_state_no_trial_arrays"):
        raise SuspendedRouteV74Error(
            "v74 checkpoint identity or transaction schema differs"
        )
    state = resolve_route_state_v74(
        execution=resolved, bed_mass_kg_m2=value["bed_mass_kg_m2"],
        suspended_mass_kg_m2=value["suspended_mass_kg_m2"],
        suspended_exchange_ledger_kg=
            value["suspended_exchange_ledger_kg"],
        bed_exchange_ledger_kg=value["bed_exchange_ledger_kg"],
        accepted_updates=value["accepted_updates"], time_s=value["time_s"],
    )
    if state.state_sha256 != value["state_sha256"]:
        raise SuspendedRouteV74Error("v74 checkpoint state digest differs")
    return state


__all__ = [
    "BACKENDS", "CHECKPOINT_SCOPE_ID", "LOCKED_SOURCES",
    "MAX_ACCEPTED_UPDATES", "METHOD_BIJKER_CURRENT", "METHOD_OUTPUT_KIND",
    "METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT",
    "METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT",
    "OUTPUT_EQUILIBRIUM_CONCENTRATION", "OUTPUT_SUSPENDED_VECTOR",
    "ROUTE_COMBINE", "ROUTE_OFF", "ROUTE_TOTAL", "SCHEMA", "SCOPE_ID",
    "SEMANTIC_ID", "RouteHandoffV74", "SuspendedRouteStateV74",
    "SuspendedRouteTrialV74", "SuspendedRouteV74Error",
    "begin_route_trial_v74", "cancel_route_trial_v74",
    "commit_route_trial_v74", "compose_route_handoff_v74",
    "load_route_checkpoint_v74", "load_route_execution_v74",
    "resolve_route_execution_v74", "resolve_route_state_v74",
    "rollback_route_trial_v74", "route_checkpoint_document_v74",
    "route_semantic_document_v74", "validate_route_handoff_v74",
    "validate_route_state_v74", "verify_locked_sources_v74",
]
