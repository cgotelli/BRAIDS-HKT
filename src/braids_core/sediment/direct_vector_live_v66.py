"""Authenticated Q3-H6a lifecycle for the first direct-vector FLOW laws.

This additive v66 envelope binds the current-only Soulsby (FLOW ``TraFrm=12``)
and Soulsby/Van Rijn (``TraFrm=11``) implementations to the common face and
Exner consumer.  It is deliberately fail-closed: direct vectors cannot receive
external secondary-flow rotation, positive-TraFrm ``ISlope=2`` remains outside
scope, and a ``TRA_COMBINE`` fraction may be inspected but cannot be committed
until Q4 owns its water-column mass and exchange.

The module owns canonical execution and portable accepted-state identities.
The CUDA implementation keeps trial arrays resident and crosses the host
boundary only for an explicit checkpoint.
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

from .capacity_parameters import canonical_parameter_items, formula_contract
from .channel_composition import (
    TRA_ADVDIFF,
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionProfile,
    channel_composition_profile_document,
    load_channel_composition_profile,
)


SCHEMA = 1
ENVELOPE_VERSION = 66
SCOPE_ID = "braids-q3h6a-direct-vector-fixed-grid-live-v66-v1"
MAX_ACCEPTED_STEPS = 20
BACKENDS = frozenset(("cpu", "implicit_fwave_midpoint_gpu"))
SUPPORTED_FORMULA_CODES = frozenset((4, 13))
SUPPORTED_METHODS = frozenset(("capacity.soulsby", "capacity.soulsby_van_rijn"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")

_COMMON_REFERENCE_SOURCES = {
    "eqtran": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "rdtrafrm": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdtrafrm.f90",
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    ),
    "upwbed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/upwbed.f90",
        "06be422d88821f3f983bc136d511fdaa80ab04c8529640474ab39b478d918158",
    ),
}

_LAW = {
    4: {
        "method_id": "capacity.soulsby",
        "reference_selector": 12,
        "source_routine": "trab12",
        "reference_source": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/trab12.f90",
            "887ed1a644afaee8764f4590cefee67affc1f5a44876ce8bc5892a4a013fd417",
        ),
        "cpu_source": (
            "solver/wave/braids_bedload_current_module.f90",
            "ca59bf18e07c48200ab8a5ec43326349f19dc97f044c71c846dad7ec21a8e402",
        ),
        "suspended_output": "source_exact_zero",
    },
    13: {
        "method_id": "capacity.soulsby_van_rijn",
        "reference_selector": 11,
        "source_routine": "trab11",
        "reference_source": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/trab11.f90",
            "f1bfce663f2851bd8fc1ec96783394562d016e44a8d3d5cbc5138e326782b1ab",
        ),
        "cpu_source": (
            "solver/wave/braids_soulsby_van_rijn_module.f90",
            "125e72f37a074df7a95e158651fd7365f69c4400457e388e8b5ceaa2ba1364b9",
        ),
        "suspended_output": "formula_direct_vector",
    },
}

_CPU_SHARED_SOURCES = {
    "face_exner": (
        "solver/wave/braids_scalar_flow_live_v65_module.f90",
        "6956e5a88494d6b5dba5c4a2d46e671c61cbb6eee6bf7ca304fa5d8922002499",
    ),
}

_GPU_SOURCES = {
    "authenticated_lifecycle_python": (
        "braids_gpu/direct_vector_live_v66.py",
        "fd24f567687bb4df08f691e66a2becbdf3fcccb5daf381e064f58928a4c4e6c6",
    ),
    "authenticated_lifecycle_cuda": (
        "braids_gpu/direct_vector_live_v66.cu",
        "da18dc63a59d2a5daf1748de724a9cfe9494dd625aa33a5b5b38ad11922eec77",
    ),
    "capacity_dispatch_python": (
        "braids_gpu/m1_capacity_dispatch.py",
        "df6f4c0f4eceba8c398173a5d608c1493f76dfaad3e4bfc6639aa5ea0e6cf678",
    ),
    "capacity_dispatch_cuda": (
        "braids_gpu/m1_capacity_dispatch.cu",
        "34938043207edb14e66cf05dc10ba84832f7c7641bdd851e35b83349436fee3b",
    ),
    "face_exner_python": (
        "braids_gpu/scalar_capacity_runtime.py",
        "566c1baff6bdf9462ba7bdf006aab80033324e6012bd4fd839f30d4f6d3c65ab",
    ),
    "face_exner_cuda": (
        "braids_gpu/scalar_capacity_runtime.cu",
        "7144971b9736f43b484f4edd50039b9dc3a2ab7389357a30c15fed4a078935fc",
    ),
}


class DirectVectorLiveV66Error(ValueError):
    """Fail-closed direct-vector identity or lifecycle error."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectVectorLiveV66Error(f"Q3-H6a/v66 {name} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise DirectVectorLiveV66Error(
            f"Q3-H6a/v66 {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise DirectVectorLiveV66Error(
            f"Q3-H6a/v66 {name} must be a lowercase SHA-256 fingerprint")
    return value


def _sequence(value: object, name: str, *, length: int | None = None,
              positive: bool = False) -> list[float]:
    if (not isinstance(value, Sequence) or
            isinstance(value, (str, bytes, bytearray))):
        raise DirectVectorLiveV66Error(f"Q3-H6a/v66 {name} must be a sequence")
    result = [_finite(item, f"{name}[{index}]", positive=positive)
              for index, item in enumerate(value)]
    if not result or (length is not None and len(result) != length):
        raise DirectVectorLiveV66Error(
            f"Q3-H6a/v66 {name} has the wrong number of fractions")
    return result


def _source_rows(formula_code: int, backend: str) -> tuple[list[dict], list[dict]]:
    law = _LAW[formula_code]
    reference = [
        {"name": name, "path": path, "sha256": digest}
        for name, (path, digest) in sorted(_COMMON_REFERENCE_SOURCES.items())
    ]
    path, digest = law["reference_source"]
    reference.append({
        "name": law["source_routine"], "path": path, "sha256": digest,
    })
    if backend == "cpu":
        path, digest = law["cpu_source"]
        implementations = [{
            "name": "capacity_formula", "path": path, "sha256": digest,
        }]
        implementations.extend(
            {"name": name, "path": path, "sha256": digest}
            for name, (path, digest) in sorted(_CPU_SHARED_SOURCES.items())
        )
    else:
        implementations = [
            {"name": name, "path": path, "sha256": digest}
            for name, (path, digest) in sorted(_GPU_SOURCES.items())
        ]
    return reference, implementations


def direct_vector_live_semantic_document_v66() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "methods": {
            str(code): {
                "method_id": row["method_id"],
                "reference_selector": row["reference_selector"],
                "direction": "formula_direct_vector",
                "external_secondary_rotation": False,
                "suspended_output": row["suspended_output"],
            }
            for code, row in sorted(_LAW.items())
        },
        "density_conversion": "formula_output_is_mass_after_exactly_one_conversion",
        "channel_order": [
            "formula_bed_and_nonbed_vectors",
            "no_external_secondary_rotation",
            "global_BedFac_and_SusFac",
            "TRA_BEDLOAD_fold_nonbed_into_bed_or_TRA_COMBINE_retain_for_Q4",
            "active_fraction_once",
            "finite_inventory_fill_once",
            "shared_face_flux",
            "conservative_exner_candidate",
        ],
        "transport_class_ownership": {
            str(TRA_BEDLOAD): (
                "Q3_owns_BedFac_bed_plus_SusFac_nonbed_as_one_Exner_vector"
            ),
            str(TRA_COMBINE): (
                "Q3_exposes_BedFac_bed_and_Q4_must_own_SusFac_nonbed_"
                "water_column_before_any_public_commit"
            ),
            str(TRA_ADVDIFF): "bypasses_eqtran_and_is_owned_by_Q4",
        },
        "transaction": (
            "immutable_parent_complete_resident_candidate_commit_or_exact_rollback"
        ),
        "checkpoint": "portable_complete_accepted_state_no_trial_arrays",
        "backend_choice": "explicit_user_choice_no_fallback",
        "integrators": {
            "cpu": "native_openmp_reference",
            "implicit_fwave_midpoint_gpu": "implicit_fwave_midpoint",
        },
        "scope": {
            "forcing": "current_only",
            "mesh": "fixed_grid",
            "wet_dry": "fully_wet",
            "reconstruction": "first_order",
            "upwind": True,
            "lateral_averaging": False,
            "bed_slope": "off",
            "maximum_waterdepth": False,
        },
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "normal_product_admission": False,
    }


SEMANTIC_ID = _fingerprint(direct_vector_live_semantic_document_v66())


def _channel_document(value: ChannelCompositionProfile | Mapping[str, object]) -> dict:
    if isinstance(value, ChannelCompositionProfile):
        return channel_composition_profile_document(value)
    return channel_composition_profile_document(load_channel_composition_profile(value))


def _parameter_rows(method_id: str, supplied: Mapping[str, object],
                    nfrac: int) -> list[dict[str, object]]:
    if not isinstance(supplied, Mapping):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 formula parameters must be a mapping")
    contract = formula_contract(method_id)
    expected = {row.keyword for row in contract.parameters}
    if set(supplied) != expected:
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 formula parameter inventory differs")
    columns: dict[str, list[float | int]] = {}
    for keyword, values in supplied.items():
        if (not isinstance(values, Sequence) or
                isinstance(values, (str, bytes, bytearray)) or
                len(values) != nfrac):
            raise DirectVectorLiveV66Error(
                f"Q3-H6a/v66 {keyword} must contain one value per fraction")
        columns[keyword] = list(values)
    rows = []
    for fraction in range(nfrac):
        one = {key: values[fraction] for key, values in columns.items()}
        try:
            resolved = canonical_parameter_items(method_id, one)
        except ValueError as error:
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 formula parameter value differs") from error
        rows.append({"fraction_index": fraction,
                     "values": [{"keyword": key, "value": value}
                                for key, value in resolved]})
    return rows


def direct_vector_live_execution_document_v66(
    *, formula_code: int, backend: str,
    channel_profile: ChannelCompositionProfile | Mapping[str, object],
    parameters: Mapping[str, object], d50_m: Sequence[float],
    d90_m: Sequence[float], solid_density_kg_m3: Sequence[float],
    bulk_density_kg_m3: Sequence[float], roughness_method_id: str,
    water_density_kg_m3: float = 1000.0, gravity_m_s2: float = 9.81,
    minimum_work_depth_m: float = 0.01,
    capacity_profile_fingerprint: str = "",
    forcing_fingerprint: str = "", roughness_fingerprint: str = "",
    face_reconstruction_fingerprint: str = "",
    boundary_roles_fingerprint: str = "", exner_fingerprint: str = "",
) -> dict[str, object]:
    """Build the strict CPU/GPU v66 execution identity.

    This first live slice admits only the already-implemented fixed-grid,
    current-only face path.  Unsupported controls are not optional arguments,
    which prevents a caller from attaching inert method labels.
    """
    if type(formula_code) is not int or formula_code not in _LAW:
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 formula code must be exactly 4 or 13")
    if backend not in BACKENDS:
        raise DirectVectorLiveV66Error("Q3-H6a/v66 backend is unsupported")
    law = _LAW[formula_code]
    channel = _channel_document(channel_profile)
    fractions = channel["fractions"]
    nfrac = len(fractions)
    fraction_ids = [row["fraction_id"] for row in fractions]
    classes = [row["transport_class"] for row in fractions]
    if any(value == TRA_ADVDIFF for value in classes):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 TRA_ADVDIFF bypasses this capacity path")
    if any(value not in (TRA_BEDLOAD, TRA_COMBINE) for value in classes):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 transport class is unsupported")
    d50 = _sequence(d50_m, "D50", length=nfrac, positive=True)
    d90 = _sequence(d90_m, "D90", length=nfrac, positive=True)
    solid_density = _sequence(
        solid_density_kg_m3, "solid density", length=nfrac, positive=True)
    bulk_density = _sequence(
        bulk_density_kg_m3, "bulk density", length=nfrac, positive=True)
    if any(d90[index] < d50[index] for index in range(nfrac)):
        raise DirectVectorLiveV66Error("Q3-H6a/v66 D90 may not be below D50")
    if (not isinstance(roughness_method_id, str) or roughness_method_id not in {
            "roughness.chezy", "roughness.manning",
            "roughness.white_colebrook", "roughness.white_colebrook_waqua"}):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 requires a typed hydraulic roughness method")
    parameter_rows = _parameter_rows(law["method_id"], parameters, nfrac)
    dependencies = {
        "capacity_profile": _sha(capacity_profile_fingerprint, "capacity profile"),
        "forcing": _sha(forcing_fingerprint, "forcing"),
        "roughness": _sha(roughness_fingerprint, "roughness"),
        "channel_profile": channel["fingerprint"],
        "face_reconstruction": _sha(
            face_reconstruction_fingerprint, "face reconstruction"),
        "boundary_roles": _sha(boundary_roles_fingerprint, "boundary roles"),
        "exner": _sha(exner_fingerprint, "Exner"),
    }
    reference_sources, implementation_sources = _source_rows(formula_code, backend)
    physical = {
        "method_id": law["method_id"],
        "formula_code": formula_code,
        "reference_selector": law["reference_selector"],
        "parameters": parameter_rows,
        "fraction_ids": fraction_ids,
        "transport_classes": classes,
        "d50_m": d50,
        "d90_m": d90,
        "solid_density_kg_m3": solid_density,
        "bulk_density_kg_m3": bulk_density,
        "water_density_kg_m3": _finite(
            water_density_kg_m3, "water density", positive=True),
        "gravity_m_s2": _finite(gravity_m_s2, "gravity", positive=True),
        "minimum_work_depth_m": _finite(
            minimum_work_depth_m, "minimum work depth", nonnegative=True),
        "roughness_method_id": roughness_method_id,
        "bed_factor": channel["bed_factor"],
        "suspended_factor": channel["suspended_factor"],
        "epsilon": channel["epsilon"],
        "external_secondary_rotation": False,
        "upwind_bedload": True,
        "laterally_averaged_bedload": False,
        "bed_slope_method_id": "bed_slope.off",
        "maximum_waterdepth": False,
        "forcing_scope": "current_only",
    }
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "qualification": "q3h6a_bounded_live_pair_not_normal_product_dispatch",
        "backend": backend,
        "integrator": ("native_openmp_reference" if backend == "cpu"
                       else "implicit_fwave_midpoint"),
        "automatic_backend_fallback": False,
        "method_id": law["method_id"],
        "formula_code": formula_code,
        "reference_selector": law["reference_selector"],
        "source_routine": law["source_routine"],
        "formula_direction": "direct_vector",
        "suspended_output": law["suspended_output"],
        "physical_profile": physical,
        "physical_profile_fingerprint": _fingerprint(physical),
        "dependencies": dependencies,
        "reference_sources": reference_sources,
        "implementation_sources": implementation_sources,
        "transport_class_scope": (
            "class1_live_class3_component_only_requires_Q4_class2_rejected"),
        "active_fraction_application_count": 1,
        "inventory_fill_application_count": 1,
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_live_execution_v66(value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 execution identity must be an object")
    try:
        from .channel_composition import resolve_channel_composition_profile
        physical = value["physical_profile"]
        dependencies = value["dependencies"]
        parameter_rows = physical["parameters"]
        keywords = [row["keyword"] for row in parameter_rows[0]["values"]]
        parameters = {
            keyword: [
                next(item["value"] for item in row["values"]
                     if item["keyword"] == keyword)
                for row in parameter_rows
            ] for keyword in keywords
        }
        channel_profile = resolve_channel_composition_profile(
            physical["fraction_ids"], physical["transport_classes"],
            bed_factor=physical["bed_factor"],
            suspended_factor=physical["suspended_factor"],
            epsilon=physical["epsilon"],
        )
        rebuilt = direct_vector_live_execution_document_v66(
            formula_code=value["formula_code"], backend=value["backend"],
            channel_profile=channel_profile, parameters=parameters,
            d50_m=physical["d50_m"], d90_m=physical["d90_m"],
            solid_density_kg_m3=physical["solid_density_kg_m3"],
            bulk_density_kg_m3=physical["bulk_density_kg_m3"],
            roughness_method_id=physical["roughness_method_id"],
            water_density_kg_m3=physical["water_density_kg_m3"],
            gravity_m_s2=physical["gravity_m_s2"],
            minimum_work_depth_m=physical["minimum_work_depth_m"],
            capacity_profile_fingerprint=dependencies["capacity_profile"],
            forcing_fingerprint=dependencies["forcing"],
            roughness_fingerprint=dependencies["roughness"],
            face_reconstruction_fingerprint=dependencies["face_reconstruction"],
            boundary_roles_fingerprint=dependencies["boundary_roles"],
            exner_fingerprint=dependencies["exner"],
        )
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as error:
        raise DirectVectorLiveV66Error(
            "Invalid Q3-H6a/v66 execution identity") from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 execution identity is stale or forged")
    return rebuilt


def verify_direct_vector_sources_v66(
    execution: Mapping[str, object], *, repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify implementation bytes and locked read-only Delft3D provenance."""
    resolved = load_direct_vector_live_execution_v66(execution)
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    rows = []
    for category in ("reference_sources", "implementation_sources"):
        for source in resolved[category]:
            path = root/source["path"]
            if not path.is_file():
                raise DirectVectorLiveV66Error(
                    f"Q3-H6a/v66 authenticated source is missing: {source['path']}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise DirectVectorLiveV66Error(
                    f"Q3-H6a/v66 authenticated source differs: {source['path']}")
            rows.append({**source, "category": category, "verified": True})
    body = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "sources": rows,
        "delft3d_source_access": "read_only_hash_verification",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _immutable_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise DirectVectorLiveV66Error(
            f"Q3-H6a/v66 {name} must be finite with shape {shape}")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode("ascii"))
        digest.update(_canonical(list(data.shape)))
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DirectVectorV66State:
    execution_fingerprint: str
    bed_mass_kg_m2: np.ndarray
    bulk_density_kg_m3: np.ndarray
    active_fraction: np.ndarray
    boundary_signed_kg: np.ndarray
    boundary_absolute_kg: np.ndarray
    accepted_step: int
    time_s: float
    state_sha256: str


def resolve_state_v66(
    *, execution: Mapping[str, object], bed_mass_kg_m2: object,
    active_fraction: object | None = None,
    boundary_signed_kg: object | None = None,
    boundary_absolute_kg: object | None = None,
    accepted_step: int = 0, time_s: float = 0.0,
) -> DirectVectorV66State:
    resolved = load_direct_vector_live_execution_v66(execution)
    raw = np.asarray(bed_mass_kg_m2, dtype=np.float64)
    nfrac = len(resolved["physical_profile"]["fraction_ids"])
    if raw.ndim != 2 or raw.shape[0] != nfrac or raw.shape[1] <= 0:
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 bed mass must be (fraction,cell)")
    mass = _immutable_array(raw, raw.shape, "bed mass")
    if np.any(mass < 0.0):
        raise DirectVectorLiveV66Error("Q3-H6a/v66 bed mass may not be negative")
    density = _immutable_array(
        resolved["physical_profile"]["bulk_density_kg_m3"], (nfrac,),
        "bulk density")
    volume = mass/density[:, None]
    total = np.sum(volume, axis=0, keepdims=True)
    if np.any(total <= 0.0):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 all-class depleted cells are outside H6a")
    derived = volume/total
    if active_fraction is None:
        fraction = _immutable_array(derived, raw.shape, "active fraction")
    else:
        fraction = _immutable_array(active_fraction, raw.shape, "active fraction")
        tolerance = 512.0*np.finfo(np.float64).eps
        if (np.any(fraction < 0.0) or
                not np.allclose(fraction, derived, rtol=tolerance, atol=tolerance)):
            raise DirectVectorLiveV66Error(
                "Q3-H6a/v66 active fraction differs from accepted stock")
    signed = _immutable_array(
        np.zeros(nfrac) if boundary_signed_kg is None else boundary_signed_kg,
        (nfrac,), "signed boundary ledger")
    absolute = _immutable_array(
        np.zeros(nfrac) if boundary_absolute_kg is None else boundary_absolute_kg,
        (nfrac,), "absolute boundary ledger")
    if np.any(absolute < 0.0):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 absolute boundary ledger may not be negative")
    if (isinstance(accepted_step, bool) or type(accepted_step) is not int or
            accepted_step < 0 or accepted_step > MAX_ACCEPTED_STEPS):
        raise DirectVectorLiveV66Error("Q3-H6a/v66 accepted step is invalid")
    accepted_time = _finite(time_s, "accepted time", nonnegative=True)
    token = _fingerprint({
        "execution_fingerprint": resolved["fingerprint"],
        "accepted_step": accepted_step,
        "time_s": accepted_time,
        "arrays_sha256": _array_digest(mass, density, fraction, signed, absolute),
    })
    return DirectVectorV66State(
        resolved["fingerprint"], mass, density, fraction, signed, absolute,
        accepted_step, accepted_time, token)


def state_checkpoint_document_v66(state: DirectVectorV66State) -> dict[str, object]:
    if not isinstance(state, DirectVectorV66State):
        raise DirectVectorLiveV66Error("Q3-H6a/v66 checkpoint state is missing")
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "envelope_version": ENVELOPE_VERSION,
        "stage": "checkpoint",
        "execution_fingerprint": state.execution_fingerprint,
        "accepted_step": state.accepted_step,
        "time_s": state.time_s,
        "bed_mass_kg_m2": state.bed_mass_kg_m2.tolist(),
        "bulk_density_kg_m3": state.bulk_density_kg_m3.tolist(),
        "active_fraction": state.active_fraction.tolist(),
        "boundary_signed_kg": state.boundary_signed_kg.tolist(),
        "boundary_absolute_kg": state.boundary_absolute_kg.tolist(),
        "state_sha256": state.state_sha256,
        "transaction_schema": "complete_accepted_state_no_trial_arrays",
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_state_checkpoint_v66(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> DirectVectorV66State:
    resolved = load_direct_vector_live_execution_v66(execution)
    expected = {
        "schema", "scope_id", "semantic_id", "envelope_version", "stage",
        "execution_fingerprint", "accepted_step", "time_s", "bed_mass_kg_m2",
        "bulk_density_kg_m3", "active_fraction", "boundary_signed_kg",
        "boundary_absolute_kg", "state_sha256", "transaction_schema",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 checkpoint inventory differs")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if (value["fingerprint"] != _fingerprint(body) or
            value["schema"] != SCHEMA or value["scope_id"] != SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["envelope_version"] != ENVELOPE_VERSION or
            value["stage"] != "checkpoint" or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["bulk_density_kg_m3"] !=
            resolved["physical_profile"]["bulk_density_kg_m3"]):
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 checkpoint header or execution differs")
    state = resolve_state_v66(
        execution=resolved, bed_mass_kg_m2=value["bed_mass_kg_m2"],
        active_fraction=value["active_fraction"],
        boundary_signed_kg=value["boundary_signed_kg"],
        boundary_absolute_kg=value["boundary_absolute_kg"],
        accepted_step=value["accepted_step"], time_s=value["time_s"])
    if state.state_sha256 != value["state_sha256"]:
        raise DirectVectorLiveV66Error(
            "Q3-H6a/v66 checkpoint state digest differs")
    return state


__all__ = [
    "BACKENDS", "ENVELOPE_VERSION", "MAX_ACCEPTED_STEPS", "SCHEMA",
    "SCOPE_ID", "SEMANTIC_ID", "SUPPORTED_FORMULA_CODES", "SUPPORTED_METHODS",
    "DirectVectorLiveV66Error", "DirectVectorV66State",
    "direct_vector_live_execution_document_v66",
    "direct_vector_live_semantic_document_v66",
    "load_direct_vector_live_execution_v66", "load_state_checkpoint_v66",
    "resolve_state_v66", "state_checkpoint_document_v66",
    "verify_direct_vector_sources_v66",
]
