"""Authenticated Q3 Dirac-CDF and retained depleted-surface semantics.

This module closes two narrow surface-composition gaps shared by the scalar
and direct-vector live bedload routes.  It deliberately does *not* implement
layers, substrate exposure, burial, or sorting; those remain owned by Q5.

Classic FLOW ``compdiam`` treats a zero-width within-class distribution as a
Dirac mass at its exact diameter.  At an exact cumulative edge its strict
comparison leaves Dxx unassigned; BRAIDS records and corrects that source
defect with the standard right-continuous inverse CDF.  ``getmfrac``/``getvfrac`` instead return a
uniform diagnostic fraction when the represented stock is exactly zero.  A
Q3 BRAIDS state has no authoritative substrate from which to reconstruct that
choice, so it retains its last accepted normalized surface identity while
keeping every class mass and every inventory availability exactly zero.  The
choice is explicit in the execution/restart identity and never creates mass.
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

from .scalar_flow_cpu_producer_v64 import load_mixture_profile


SCHEMA = 1
ENVELOPE_VERSION = 69
SCOPE_ID = "braids-q3-surface-identity-live-v69-v1"
MAX_ACCEPTED_STEPS = 4
ROUTE_SCALAR = "scalar_flow_v65"
ROUTE_DIRECT = "direct_vector_family_v67"
BACKEND_CPU = "cpu"
BACKEND_GPU = "implicit_fwave_midpoint_gpu"
INTEGRATOR_GPU = "implicit_fwave_midpoint"

LOCKED_SOURCES = {
    "compdiam": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/compdiam.f90"
        ),
        "sha256": "1beb16c5bcc10c24704dbe443e2f5ddd709106adf9561f3a89dd1813327f517a",
        "anchors": ["181-200", "246-342"],
    },
    "compsandfrac": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/compsandfrac.f90"
        ),
        "sha256": "42ec4e73de1191242a7dc43ce205c1c865008c6bde54799be2576ac5db33bfdd",
        "anchors": ["82-119"],
    },
    "bedcomposition": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/bedcomposition_module.f90"
        ),
        "sha256": "ee22800f75d37979317d821579120e6337ab0b6f140b362a966a94712d9a3bed",
        "anchors": ["2482-2565", "2819-2879", "2935-3003"],
    },
    "erosed": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/erosed.f90"
        ),
        "sha256": "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
        "anchors": ["655", "723-760", "997-1015"],
    },
}

IMPLEMENTATION_SOURCES = (
    "braids_core/sediment/surface_identity_live_v69.py",
    "solver/wave/braids_surface_identity_live_v69_module.f90",
    "braids_gpu/surface_identity_live_v69.cu",
    "braids_gpu/surface_identity_live_v69.py",
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")


class SurfaceIdentityV69Error(ValueError):
    """Fail-closed surface-semantics error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_hashes(paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = _root()/relative
        if not path.is_file():
            raise SurfaceIdentityV69Error(
                f"Q3/v69 implementation source is missing: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_locked_sources_v69() -> dict[str, str]:
    actual = _source_hashes([row["path"] for row in LOCKED_SOURCES.values()])
    for row in LOCKED_SOURCES.values():
        if actual[row["path"]] != row["sha256"]:
            raise SurfaceIdentityV69Error(
                f"locked Delft3D source changed: {row['path']}")
    return actual


def _verify_parent(parent: Mapping[str, object]) -> tuple[str, str, str, list[str]]:
    if not isinstance(parent, Mapping) or not isinstance(parent.get("fingerprint"), str):
        raise SurfaceIdentityV69Error("Q3/v69 parent execution is malformed")
    body = {key: value for key, value in parent.items() if key != "fingerprint"}
    if _fingerprint(body) != parent["fingerprint"]:
        raise SurfaceIdentityV69Error("Q3/v69 parent execution fingerprint differs")
    scope = parent.get("scope_id")
    backend = parent.get("backend")
    if scope == "braids-q3h5-scalar-flow-fixed-grid-live-v65-v1":
        route = ROUTE_SCALAR
        ids = parent.get("ordered_fraction_ids")
    elif scope == "braids-q3-h6-direct-vector-family-live-v67-v1":
        route = ROUTE_DIRECT
        profile = parent.get("physical_profile")
        ids = profile.get("fraction_ids") if isinstance(profile, Mapping) else None
    else:
        raise SurfaceIdentityV69Error("Q3/v69 parent lifecycle scope is unsupported")
    if backend not in (BACKEND_CPU, BACKEND_GPU):
        raise SurfaceIdentityV69Error("Q3/v69 parent backend is unsupported")
    if not isinstance(ids, list) or not ids or any(
            not isinstance(value, str) or not value for value in ids):
        raise SurfaceIdentityV69Error("Q3/v69 parent fraction order is malformed")
    return route, str(scope), str(backend), list(ids)


def surface_identity_execution_document_v69(
    *, parent_execution: Mapping[str, object],
    mixture_profile: Mapping[str, object],
) -> dict[str, object]:
    """Bind one live parent to exact-CDF and retained-identity semantics."""
    route, parent_scope, backend, fraction_ids = _verify_parent(parent_execution)
    mixture = load_mixture_profile(mixture_profile)
    mixture_ids = [row["fraction_id"] for row in mixture["ordered_fractions"]]
    if mixture_ids != fraction_ids:
        raise SurfaceIdentityV69Error(
            "Q3/v69 mixture and parent fraction order differs")
    verify_locked_sources_v69()
    implementation = _source_hashes(IMPLEMENTATION_SOURCES)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "route": route,
        "backend": backend,
        "integrator": INTEGRATOR_GPU if backend == BACKEND_GPU else "parent_cpu_openmp",
        "automatic_backend_fallback": False,
        "parent_scope_id": parent_scope,
        "parent_execution_fingerprint": parent_execution["fingerprint"],
        "mixture_profile_fingerprint": mixture["fingerprint"],
        "ordered_fraction_ids": fraction_ids,
        "dirac_within_class_cdf": (
            "one_point_exact_diameter_follows_locked_compdiam_stage5_with_"
            "right_continuous_cumulative_edge_correction"
        ),
        "delft_dirac_edge_distinction": (
            "locked_strict_greater_than_can_leave_exact_edge_Dxx_unassigned;_"
            "BRAIDS_uses_right_continuous_inverse_CDF"
        ),
        "characteristic_diameters": ["D10", "D50", "D90", "Dg"],
        "depleted_surface_identity": (
            "retain_last_accepted_normalized_identity_with_exact_zero_mass_"
            "and_zero_inventory_availability"
        ),
        "delft_zero_stock_distinction": (
            "locked_getmfrac_getvfrac_return_uniform_diagnostic_identity;_"
            "BRAIDS_Q3_retention_is_explicit_provisional_not_Q5_layers"
        ),
        "synthetic_fraction_mass": False,
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "locked_sources": {key: dict(value) for key, value in LOCKED_SOURCES.items()},
        "implementation_sources": implementation,
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_surface_identity_execution_v69(
    value: Mapping[str, object], *, parent_execution: Mapping[str, object],
    mixture_profile: Mapping[str, object],
) -> dict[str, object]:
    expected = surface_identity_execution_document_v69(
        parent_execution=parent_execution, mixture_profile=mixture_profile)
    if not isinstance(value, Mapping) or dict(value) != expected or \
            _canonical(value) != _canonical(expected):
        raise SurfaceIdentityV69Error("Q3/v69 execution is stale or forged")
    return expected


@dataclass(frozen=True)
class SurfaceStatisticsV69:
    dm_m: np.ndarray
    dg_m: np.ndarray
    d10_m: np.ndarray
    d50_m: np.ndarray
    d90_m: np.ndarray
    dgsd: np.ndarray
    sand_fraction: np.ndarray
    mud_fraction: np.ndarray
    has_exposed_nonmud: np.ndarray


def _percentiles_delft_dirac(
    fraction: np.ndarray, sediment_type: np.ndarray,
    minimum_type: int, tables: Sequence[Sequence[Mapping[str, float]]],
    requested: Sequence[float],
) -> np.ndarray:
    """Delft ``compdiam`` merge scan with the documented Dirac-edge fix."""
    nfrac = fraction.size
    stage = np.zeros(nfrac, dtype=np.int64)
    counts = np.asarray([len(row) for row in tables], dtype=np.int64)
    log_table = [np.asarray([math.log(float(point["diameter_m"]))
                             for point in row], dtype=np.float64)
                 for row in tables]
    cdf_table = [np.asarray([float(point["probability"])
                             for point in row], dtype=np.float64)
                 for row in tables]
    fracdiam = float(np.sum(fraction[sediment_type >= minimum_type]))
    output = np.zeros(len(requested), dtype=np.float64)
    log_previous = 0.0
    accumulated = 0.0
    index = 0
    required = float(requested[index])
    max_iterations = int(np.sum(counts)+nfrac+len(requested)+8)
    iterations = 0
    while required > accumulated:
        iterations += 1
        if iterations > max_iterations:
            raise SurfaceIdentityV69Error("Q3/v69 percentile scan did not converge")
        log_diameter = 999.999
        trigger = -1
        density = 0.0
        for grain in range(nfrac):
            current = int(stage[grain])
            count = int(counts[grain])
            if current < count:
                if current > 0:
                    factor = float(fraction[grain]/fracdiam) if fracdiam > 0.0 else 1.0
                    density += factor*(
                        cdf_table[grain][current]-cdf_table[grain][current-1]) / (
                        log_table[grain][current]-log_table[grain][current-1])
                candidate = float(log_table[grain][current])
                if candidate < log_diameter:
                    log_diameter = candidate
                    trigger = grain
        if trigger < 0:
            output[index:] = math.exp(log_previous)
            break
        interval = density*(log_diameter-log_previous)
        while interval >= required-accumulated:
            if interval > 0.0:
                log_previous += ((log_diameter-log_previous)*
                                 (required-accumulated)/interval)
            accumulated = required
            output[index] = math.exp(log_previous)
            index += 1
            if index >= len(requested):
                return output
            required = float(requested[index])
            interval = density*(log_diameter-log_previous)
        accumulated += interval
        log_previous = log_diameter
        if stage[trigger] == 0:
            row = log_table[trigger]
            if row[-1] <= row[0]:
                stage[trigger] = max(5, int(counts[trigger]))
                # ``compdiam`` uses a strict ``>`` here.  At the exact edge
                # of a Dirac point mass that leaves Dxx unassigned,
                # which is not a valid quantile.  BRAIDS deliberately uses
                # the right-continuous inverse CDF: the point owns its upper
                # cumulative edge as well.
                while float(fraction[trigger]) >= required-accumulated:
                    output[index] = math.exp(log_diameter)
                    index += 1
                    if index >= len(requested):
                        return output
                    required = float(requested[index])
                accumulated += float(fraction[trigger])
            else:
                stage[trigger] = 1
        else:
            stage[trigger] += 1
    return output


def surface_statistics_v69(
    surface_identity: object, *, mixture_profile: Mapping[str, object],
) -> SurfaceStatisticsV69:
    """Evaluate Dm/Dg/D10/D50/D90/DgSD and sand/mud fractions.

    ``surface_identity`` is metadata and must already be normalized.  It may
    describe a depleted cell, but this routine never receives or changes mass.
    """
    mixture = load_mixture_profile(mixture_profile)
    fraction = np.asarray(surface_identity, dtype=np.float64)
    rows = mixture["ordered_fractions"]
    nfrac = len(rows)
    if fraction.ndim != 2 or fraction.shape[0] != nfrac or fraction.shape[1] < 1:
        raise SurfaceIdentityV69Error(
            "Q3/v69 surface identity must have shape (fraction, cell)")
    tolerance = 512.0*np.finfo(np.float64).eps*nfrac
    if (not np.all(np.isfinite(fraction)) or np.any(fraction < 0.0) or
            not np.allclose(np.sum(fraction, axis=0), 1.0,
                            rtol=0.0, atol=tolerance)):
        raise SurfaceIdentityV69Error(
            "Q3/v69 surface identity must be finite, nonnegative and normalized")
    sediment_type = np.asarray([row["sediment_type"] for row in rows], dtype=np.int64)
    mean = np.asarray([row["mean_diameter_m"] for row in rows], dtype=np.float64)
    median = np.asarray([row["median_diameter_m"] for row in rows], dtype=np.float64)
    tables = [row["percentile_table"] for row in rows]
    minimum = int(mixture["min_dxx_sediment_type"])
    max_mud = int(mixture["max_mud_sediment_type"])
    ncell = fraction.shape[1]
    values = np.zeros((8, ncell), dtype=np.float64)
    exposed = np.zeros(ncell, dtype=bool)
    eligible = sediment_type >= minimum
    nonmud = sediment_type > max_mud
    for cell in range(ncell):
        frac = fraction[:, cell]
        fracdiam = float(np.sum(frac[eligible]))
        fracnonmud = float(np.sum(frac[nonmud]))
        values[7, cell] = float(np.sum(frac[~nonmud]))
        exposed[cell] = fracnonmud > 0.0
        if fracdiam > 0.0:
            weight = frac[eligible]/fracdiam
            values[0, cell] = float(np.sum(weight*mean[eligible]))
            logdg = float(np.sum(weight*np.log(median[eligible])))
            values[1, cell] = math.exp(logdg)
            values[5, cell] = math.exp(math.sqrt(float(np.sum(
                weight*(np.log(median[eligible])-logdg)**2))))
        if fracnonmud > 0.0:
            values[6, cell] = float(np.sum(
                frac[nonmud & (mean < 2.0e-3)])/fracnonmud)
        values[2:5, cell] = _percentiles_delft_dirac(
            frac, sediment_type, minimum, tables, (0.10, 0.50, 0.90))
    frozen = []
    for row in values:
        item = np.array(row, copy=True, order="C")
        item.setflags(write=False)
        frozen.append(item)
    exposed.setflags(write=False)
    return SurfaceStatisticsV69(
        frozen[0], frozen[1], frozen[2], frozen[3], frozen[4], frozen[5],
        frozen[6], frozen[7], exposed,
    )


def retained_surface_identity_v69(
    *, candidate_bed_mass_kg_m2: object, bulk_density_kg_m3: object,
    parent_surface_identity: object,
) -> np.ndarray:
    """Derive positive-stock fractions and retain identity at exact depletion."""
    mass = np.asarray(candidate_bed_mass_kg_m2, dtype=np.float64)
    if mass.ndim != 2 or mass.shape[1] < 1 or not np.all(np.isfinite(mass)) or \
            np.any(mass < 0.0):
        raise SurfaceIdentityV69Error("Q3/v69 candidate mass is invalid")
    density = np.asarray(bulk_density_kg_m3, dtype=np.float64)
    parent = np.asarray(parent_surface_identity, dtype=np.float64)
    if density.shape != (mass.shape[0],) or np.any(~np.isfinite(density)) or \
            np.any(density <= 0.0) or parent.shape != mass.shape:
        raise SurfaceIdentityV69Error("Q3/v69 density or parent identity differs")
    tolerance = 512.0*np.finfo(np.float64).eps*mass.shape[0]
    if (np.any(~np.isfinite(parent)) or np.any(parent < 0.0) or
            not np.allclose(np.sum(parent, axis=0), 1.0,
                            rtol=0.0, atol=tolerance)):
        raise SurfaceIdentityV69Error("Q3/v69 parent identity is invalid")
    volume = mass/density[:, None]
    total = np.sum(volume, axis=0)
    positive = total > 0.0
    result = np.array(parent, copy=True, order="C")
    result[:, positive] = volume[:, positive]/total[positive][None, :]
    result.setflags(write=False)
    return result


__all__ = [
    "BACKEND_CPU", "BACKEND_GPU", "ENVELOPE_VERSION", "IMPLEMENTATION_SOURCES",
    "INTEGRATOR_GPU", "LOCKED_SOURCES", "MAX_ACCEPTED_STEPS", "ROUTE_DIRECT",
    "ROUTE_SCALAR", "SCHEMA", "SCOPE_ID", "SurfaceIdentityV69Error",
    "SurfaceStatisticsV69", "load_surface_identity_execution_v69",
    "retained_surface_identity_v69", "surface_identity_execution_document_v69",
    "surface_statistics_v69", "verify_locked_sources_v69",
]
