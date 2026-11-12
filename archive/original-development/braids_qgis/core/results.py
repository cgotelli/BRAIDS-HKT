"""Convert native cut-cell frames to QGIS multiband raster products."""
from __future__ import annotations

import csv
import html
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
from osgeo import gdal, ogr

from braids_core.boundaries.geometry_io import (geometry_artifact, load_geometry,
                                                load_hierarchy_index)
from braids_core.sediment.capacity_execution_identity import (
    EH_DELFT_METHOD_ID, EH_LEGACY_METHOD_ID, MPM_DELFT_METHOD_ID,
    MPM_LEGACY_METHOD_ID, capacity_execution_semantic_id,
)
from braids_core.sediment.capacity_foundation_execution import (
    SEMANTIC_ID as CAPACITY_FOUNDATION_EXECUTION_SEMANTIC_ID,
)
from braids_core.sediment.capacity_profile import (
    CapacityProfileError,
    capacity_profile_document,
    capacity_profile_semantic_id,
    load_capacity_profile,
)
from braids_core.sediment.channel_composition import (
    ChannelCompositionError,
    channel_composition_native_wire_document,
    load_channel_composition_profile,
)
from braids_core.sediment.execution_profile_v2 import (
    ExecutionProfileError,
    bind_component_profile,
    execution_profile_document,
    execution_profile_payload_json,
    execution_profile_semantic_id,
    is_frozen_q3sp_execution_profile,
    load_execution_profile,
    resolve_execution_profile,
)
from braids_core.sediment.hiding_foundation_execution import (
    EXECUTION_KEY as HIDING_FOUNDATION_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as HIDING_FOUNDATION_ORDERED_ROUTE_KEY,
    validate_hiding_foundation_execution_document,
)
from braids_core.sediment.secondary_foundation_execution import (
    EXECUTION_KEY as SECONDARY_FOUNDATION_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY,
    validate_secondary_foundation_execution_document,
)
from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)
from braids_core.windows_paths import filesystem_path
from .mesh_visualization import write_result_mesh
from .project import (
    HIDING_FOUNDATION_BINDING_SCHEMA,
    HIDING_FOUNDATION_BINDING_SCOPE_ID,
    HIDING_FOUNDATION_BINDING_SEMANTIC_ID,
    SECONDARY_FOUNDATION_BINDING_KEY,
    _validate_secondary_foundation_binding_document,
    PROGNOSTIC_SECONDARY_BINDING_KEY,
    PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY,
    PROGNOSTIC_SECONDARY_STATE_UNITS_KEY,
    PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY,
    _validate_prognostic_secondary_binding_document,
    RUN_MARKER,
)


NODATA = np.float32(-3.4028235e38)
WET_DEPTH_M = 1e-8
CUT_CELL_DISPLAY_SUBDIVISIONS = 4
# Keep results in QGIS's native raster format but favor interactivity over
# storage density.  DEFLATE's per-band compression noticeably delays writing
# and opening temporal stacks; tiled, uncompressed GeoTIFFs avoid that CPU
# cost and are supported by every QGIS/GDAL installation that BRAIDS targets.
FAST_GEOTIFF_OPTIONS = ("TILED=YES", "COMPRESS=NONE", "BIGTIFF=IF_SAFER",
                        "INTERLEAVE=BAND")
LIVE_PREVIEW_DIRECTORY = "live_preview"
LIVE_PREVIEW_VARIABLES = {"depth_m": "m", "velocity_m_s": "m/s", "stage_m": "m",
                          "discharge_x_m2_s": "m2/s", "discharge_y_m2_s": "m2/s"}
GPU_LIVE_PREVIEW_VARIABLES = {
    "depth_m": "m", "velocity_m_s": "m/s", "stage_m": "m",
    "bed_elevation_m": "m", "bed_change_m": "m",
    "concentration_kg_m3": "kg/m3",
}
# These are per-cell bank-history diagnostics, not additional conserved mass.
# The cumulative quantities are removed volume divided by donor-cell area,
# hence metres of equivalent depth rather than cubic metres.
GPU_BANK_RESULT_FIELDS = (
    ("bank_history_undercut", "bank_undercut_m", "m"),
    ("bank_history_toe_cumulative", "bank_toe_removed_depth_m", "m"),
    ("bank_history_collapse_cumulative", "bank_collapse_depth_m", "m"),
    ("bank_history_collapse_count", "bank_collapse_count", "count"),
)
GPU_BEDFORM_RESULT_FIELDS = (
    ('bedform_effective_manning', 'bedform_effective_manning_n', 's/m^(1/3)'),
    ('bedform_dune_height', 'bedform_dune_height_m', 'm'),
    ('bedform_dune_length', 'bedform_dune_length_m', 'm'),
    ('bedform_total_ks', 'bedform_total_ks_m', 'm'),
    ('bedform_iterations', 'bedform_iterations', 'count'),
)
ROUGHNESS_RESULT_CONTRACTS = {
    "chezy": {"variable":"chezy_C", "units":"m^0.5/s", "label":"Chezy C"},
    "manning": {"variable":"manning_n", "units":"s/m^(1/3)", "label":"Manning n"},
    "white_colebrook": {"variable":"nikuradse_ks", "units":"m", "label":"Nikuradse ks"},
}
DYNAMIC_MESH_CACHE = "adaptive_mesh_cache.json"
WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset((5, 32))
WINDOWS_REPLACE_ATTEMPTS = 8


def _replace_with_windows_retry(source: Path, target: Path) -> None:
    """Publish a closed result file despite a brief Windows sharing lock."""
    delay = .02
    for attempt in range(WINDOWS_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if (getattr(exc, "winerror", None) not in
                    WINDOWS_TRANSIENT_REPLACE_ERRORS or
                    attempt + 1 == WINDOWS_REPLACE_ATTEMPTS):
                raise
            time.sleep(delay)
            delay = min(1., 2. * delay)


def _write_text_atomic(path: Path, payload: str) -> None:
    """Write a small result sidecar without exposing partial JSON/XML."""
    temporary = path.with_name(f".{uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        _replace_with_windows_retry(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _roughness_result_contract(case=None, project=None, source=None):
    """Name raw resistance bytes by physical type, never by storage alias."""
    case = case if isinstance(case, dict) else {}
    project = project if isinstance(project, dict) else {}
    source = source if isinstance(source, dict) else {}
    binding = source.get("delft_forcing_binding")
    if not isinstance(binding, dict):
        binding = source.get("typed_forcing_binding")
    declared = case.get("hydraulic_roughness")
    if not isinstance(declared, dict):
        declared = project.get("roughness")
    if not isinstance(declared, dict):
        declared = {}
    mode = binding.get("hydraulic_roughness_mode") if isinstance(binding, dict) else None
    if mode is None:mode = declared.get("mode")
    if mode is None:
        native = source.get("native_hydraulic_roughness_mode")
        try:native = int(np.asarray(native).item())
        except (TypeError,ValueError):native = 2
        mode = {1:"chezy",2:"manning",3:"white_colebrook"}.get(native,"manning")
    if mode not in ROUGHNESS_RESULT_CONTRACTS:
        raise ValueError("Result hydraulic roughness mode is unknown")
    contract={"mode":mode,**ROUGHNESS_RESULT_CONTRACTS[mode],
              "storage_field":"manning_cell",
              "storage_field_is_physical_manning":mode=="manning"}
    if isinstance(binding,dict):
        if (binding.get("roughness_parameter_name")!=contract["variable"] or
                binding.get("roughness_parameter_units")!=contract["units"]):
            raise ValueError("Authenticated roughness result name/units differ")
        contract["forcing_fingerprint"]=binding.get("fingerprint")
    if declared:
        declared_name=declared.get("parameter_name")
        declared_units=declared.get("parameter_units")
        if declared_name is not None and declared_name not in (
                contract["variable"],contract["label"]):
            raise ValueError("Saved roughness result name differs from its mode")
        if declared_units is not None and declared_units!=contract["units"]:
            raise ValueError("Saved roughness result units differ from its mode")
    return contract


def _gpu_q3c_result_identity(
    worker: Mapping[str, Any],source: Mapping[str, Any],
    *,secondary_receipt: Mapping[str, Any] | None=None,
    prognostic_receipt: Mapping[str, Any] | None=None,
    prepared_receipt: Mapping[str, Any] | None=None,
) -> dict | None:
    """Retain the already-verified Q3-C method/profile identity in results.

    ``read_gpu_run`` authenticates the worker method against every frame and
    ``read_gpu_prepared`` authenticates these prepared documents against the
    native v54 initializer.  This compact contract cross-checks both views so
    the QGIS result manifest does not discard which capacity law, typed
    roughness and execution profile produced the visualized fields.
    """
    admission=source.get("m1_profile_public_admission")
    admission_id=(admission.get("admission_id")
                  if isinstance(admission,Mapping) else None)
    secondary_raw=source.get(SECONDARY_FOUNDATION_EXECUTION_KEY)
    q3h2=secondary_raw is not None
    prognostic_raw=source.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)
    q3h3=prognostic_raw is not None
    if q3h2 and q3h3:
        raise ValueError(
            "GPU result cannot compose equilibrium and prognostic secondary")
    if not ((admission_id=="q3c.scalar_capacity_total_v1") or
            ((q3h2 or q3h3) and
             admission_id=="q3h1.scalar_capacity_hiding_v1")):
        return None
    method=worker.get("method")
    execution=source.get("capacity_foundation_execution_identity")
    ordered=source.get("capacity_foundation_ordered_route_identity")
    profile=source.get("_execution_profile_v2_contract")
    if (not isinstance(method,Mapping) or not isinstance(execution,Mapping) or
            not isinstance(ordered,Mapping) or not isinstance(profile,Mapping)):
        raise ValueError("GPU Q3-C result identity is incomplete")
    identities=profile.get("identities")
    components=profile.get("components")
    if (not isinstance(identities,Mapping) or not isinstance(components,list) or
            profile.get("sidecar_schema")!=1 or
            execution.get("method_id")!=method.get("capacity_formula") or
            execution.get("formula_code")!=method.get("capacity_formula_code") or
            execution.get("forcing_fingerprint")!=
                method.get("delft_forcing_fingerprint") or
            execution.get("capacity_profile_fingerprint")!=
                method.get("capacity_profile_fingerprint") or
            ordered.get("ordered_parts")!=[
                "capacity_execution_fingerprint","capacity_profile_fingerprint"] or
            ordered.get("capacity_execution_fingerprint")!=
                execution.get("fingerprint") or
            ordered.get("capacity_profile_fingerprint")!=
                execution.get("capacity_profile_fingerprint") or
            method.get("m1_profile_admission")!=admission_id or
            method.get("coupling")!=(
                "weak_midpoint_q3h3_classic_flow_prognostic_"
                "secondary_composition_v1" if q3h3 else
                "weak_midpoint_q3h2_classic_flow_equilibrium_"
                "secondary_composition_v1" if q3h2 else
                "weak_midpoint_q3c_scalar_capacity_composition_v1") or
            method.get("transport_checkpoint_schema")!=(
                8 if q3h3 else 7 if q3h2 else 5) or
            method.get("automatic_CPU_fallback") is not False or
            identities.get("execution_profile_v2_fingerprint")!=
                method.get("execution_profile_v2_fingerprint") or
            identities.get("execution_profile_v2_semantic_id")!=
                method.get("execution_profile_v2_semantic_id") or
            identities.get("execution_profile_v2_backend")!=
                "implicit_fwave_midpoint_gpu" or
            identities.get("execution_profile_v2_admission")!="runtime" or
            components!=method.get("execution_profile_v2_components")):
        raise ValueError(
            "GPU Q3-C method, capacity route, and execution profile differ")
    secondary_result={}
    if q3h2:
        identity=validate_secondary_foundation_execution_document(secondary_raw)
        route=source.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)
        expected_route={
            "ordered_parts":[
                "capacity_execution_fingerprint",
                "capacity_profile_fingerprint",
                "hiding_execution_fingerprint",
                "secondary_execution_fingerprint"],
            "capacity_execution_fingerprint":identity[
                "capacity_execution_fingerprint"],
            "capacity_profile_fingerprint":identity[
                "capacity_profile_fingerprint"],
            "hiding_execution_fingerprint":identity[
                "hiding_execution_fingerprint"],
            "secondary_execution_fingerprint":identity["fingerprint"],
        }
        expected_method={
            "secondary_execution_schema":identity["schema"],
            "secondary_execution_scope_id":identity["scope_id"],
            "secondary_execution_semantic_id":identity["semantic_id"],
            "secondary_execution_fingerprint":identity["fingerprint"],
            "secondary_method_id":identity["method_id"],
            "secondary_model":identity["model"],
            "secondary_base_envelope_version":identity[
                "base_envelope_version"],
            "secondary_capacity_law":identity["capacity_law"],
            "secondary_rmincf":identity["rmincf"],
            "secondary_chzmin":identity["chzmin"],
            "secondary_espir":identity["espir"],
            "secondary_betac":identity["betac"],
            "secondary_curvature_discretization":identity[
                "curvature_discretization"],
            "secondary_curvature_cap_policy":identity[
                "curvature_cap_policy"],
            "secondary_metric_policy":identity["metric_policy"],
            "secondary_roughness_policy":identity["roughness_policy"],
            "secondary_refresh_timing":identity["refresh_timing"],
            "secondary_boundary_policy":identity["boundary_policy"],
            "secondary_composition_order":identity["composition_order"],
            "secondary_raw_bed_rotation_scope":identity[
                "raw_bed_rotation_scope"],
            "suspended_channel_rotation":identity[
                "suspended_channel_rotation"],
            "secondary_backend_selection":identity["backend_selection"],
        }
        preimage=json.dumps(identity,sort_keys=True,separators=(",",":"),
                            ensure_ascii=True,allow_nan=False)
        if (route!=expected_route or
                not isinstance(secondary_receipt,Mapping) or
                set(secondary_receipt)!={
                    "envelope_version","canonical_preimage_ascii",
                    "canonical_preimage_sha256",
                    SECONDARY_FOUNDATION_EXECUTION_KEY,
                    SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY} or
                secondary_receipt.get("envelope_version")!=56 or
                secondary_receipt.get("canonical_preimage_ascii")!=preimage or
                secondary_receipt.get("canonical_preimage_sha256")!=
                    hashlib.sha256(preimage.encode("ascii")).hexdigest() or
                secondary_receipt.get(SECONDARY_FOUNDATION_EXECUTION_KEY)!=
                    identity or
                secondary_receipt.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=
                    expected_route or
                any(method.get(key)!=value
                    for key,value in expected_method.items())):
            raise ValueError(
                "GPU Q3-H2 result identity differs from preparation or method")
        secondary_result={
            "secondary_foundation_identity":dict(secondary_receipt),
            "secondary_foundation_prepared_receipt_verified":True,
        }
        if prognostic_receipt is not None:
            raise ValueError(
                "GPU Q3-H2 result unexpectedly carries a Q3-H3 receipt")
    elif q3h3:
        identity=validate_prognostic_secondary_execution_document(
            prognostic_raw)
        route=source.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)
        expected_route={
            "capacity_execution_fingerprint":identity[
                "capacity_execution_fingerprint"],
            "capacity_profile_fingerprint":identity[
                "capacity_profile_fingerprint"],
            "hiding_execution_fingerprint":identity[
                "hiding_execution_fingerprint"],
            "prognostic_secondary_execution_fingerprint":identity[
                "fingerprint"],
        }
        expected_method={
            "prognostic_secondary_execution_schema":identity["schema"],
            "prognostic_secondary_execution_scope_id":identity["scope_id"],
            "prognostic_secondary_execution_semantic_id":identity[
                "semantic_id"],
            "prognostic_secondary_execution_fingerprint":identity[
                "fingerprint"],
            "prognostic_secondary_method_id":identity["method_id"],
            "prognostic_secondary_model":identity["model"],
            "prognostic_secondary_state_semantics":identity[
                "state_semantics"],
            "prognostic_secondary_integrated_ledger_units":identity[
                "integrated_ledger_units"],
            "prognostic_secondary_base_envelope_version":identity[
                "base_envelope_version"],
            "prognostic_secondary_capacity_law":identity["capacity_law"],
            "prognostic_secondary_rmincf":identity["rmincf"],
            "prognostic_secondary_chzmin":identity["chzmin"],
            "prognostic_secondary_espir":identity["espir"],
            "prognostic_secondary_betac":identity["betac"],
            "prognostic_secondary_scalar_diffusivity_m2_s":identity[
                "scalar_diffusivity_m2_s"],
            "prognostic_secondary_source_sink_policy":identity[
                "source_sink_policy"],
            "prognostic_secondary_open_boundary_policy":identity[
                "open_boundary_policy"],
            "prognostic_secondary_bed_boundary_rotation_policy":identity[
                "bed_boundary_rotation_policy"],
            "prognostic_secondary_depth_pair_policy":identity[
                "depth_pair_policy"],
            "prognostic_secondary_target_forcing_policy":identity[
                "target_forcing_policy"],
            "prognostic_secondary_composition_order":identity[
                "composition_order"],
            "prognostic_secondary_checkpoint_schema":identity[
                "gpu_checkpoint_schema"],
            "prognostic_secondary_backend_selection":identity[
                "backend_selection"],
            "component_coupling_schema":11,
            "secondary_state_semantics":identity["state_semantics"],
            "secondary_integrated_ledger_units":identity[
                "integrated_ledger_units"],
            "secondary_boundary_rotation":identity[
                "bed_boundary_rotation_policy"],
            "secondary_legacy_rotation_disabled":True,
        }
        preimage=json.dumps(identity,sort_keys=True,separators=(",",":"),
                             ensure_ascii=True,allow_nan=False)
        visible=prepared_receipt if isinstance(prepared_receipt,Mapping) else {}
        expected_outer_keys={
            "envelope_version","canonical_preimage_ascii",
            "canonical_preimage_sha256",
            "prognostic_secondary_execution_fingerprint",
            "prognostic_secondary_bed_boundary_rotation_policy",
            "prognostic_secondary_state_units",
            "prognostic_secondary_integrated_ledger_units",
            PROGNOSTIC_SECONDARY_EXECUTION_KEY,
            PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
        }
        if (route!=expected_route or secondary_receipt is not None or
                not isinstance(prognostic_receipt,Mapping) or
                set(prognostic_receipt)!=expected_outer_keys or
                prognostic_receipt.get("envelope_version")!=57 or
                prognostic_receipt.get("canonical_preimage_ascii")!=preimage or
                prognostic_receipt.get("canonical_preimage_sha256")!=
                    hashlib.sha256(preimage.encode("ascii")).hexdigest() or
                prognostic_receipt.get(
                    "prognostic_secondary_execution_fingerprint")!=
                    identity["fingerprint"] or
                prognostic_receipt.get(
                    "prognostic_secondary_bed_boundary_rotation_policy")!=
                    identity["bed_boundary_rotation_policy"] or
                prognostic_receipt.get("prognostic_secondary_state_units")!=
                    "m2_s" or
                prognostic_receipt.get(
                    "prognostic_secondary_integrated_ledger_units")!=
                    identity["integrated_ledger_units"] or
                prognostic_receipt.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)!=
                    identity or
                prognostic_receipt.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=
                    expected_route or
                visible.get("prognostic_secondary_execution_fingerprint")!=
                    identity["fingerprint"] or
                visible.get("prognostic_secondary_state_semantics")!=
                    identity["state_semantics"] or
                visible.get("prognostic_secondary_integrated_ledger_units")!=
                    identity["integrated_ledger_units"] or
                visible.get(
                    "prognostic_secondary_bed_boundary_rotation_policy")!=
                    identity["bed_boundary_rotation_policy"] or
                visible.get("prognostic_secondary_state_units")!="m2_s" or
                identity["integrated_ledger_units"]!="m4_s" or
                any(method.get(key)!=value
                    for key,value in expected_method.items())):
            raise ValueError(
                "GPU Q3-H3 result identity differs from preparation or method")
        secondary_result={
            "prognostic_secondary_identity":dict(prognostic_receipt),
            "prognostic_secondary_prepared_receipt_verified":True,
            "prognostic_secondary_state_units":"m2_s",
            "prognostic_secondary_integrated_ledger_units":
                identity["integrated_ledger_units"],
        }
    elif secondary_receipt is not None or prognostic_receipt is not None:
        raise ValueError(
            "GPU Q3-C result unexpectedly carries a secondary receipt")
    # Force a JSON round-trip here so no dataclass/NumPy implementation object
    # can leak into the durable QGIS manifest.
    return json.loads(json.dumps({
        "gpu_method_identity":{
            "sha256":worker.get("method_identity"),
            "method":dict(method),
        },
        "execution_profile_v2":{
            "sidecar_schema":profile["sidecar_schema"],
            **dict(identities),
            "components":components,
        },
        "capacity_route_identity":{
            "envelope_version":54,
            "execution_identity":dict(execution),
            "ordered_route_identity":dict(ordered),
        },
        **secondary_result,
    },sort_keys=True,separators=(",",":"),allow_nan=False))


def _capacity_result_contract(run_root: Path, case: dict, marker: dict) -> dict | None:
    """Cross-check historical v50 or forcing-bound Q3-C v54 identity."""
    route=case.get("capacity_route_identity")
    if route is None:
        return None
    expected_keys={
        "envelope_version","identity_schema","ordered_parts","method_id",
        "capacity_execution_semantic_id","capacity_profile_semantic_id",
        "capacity_execution_fingerprint","capacity_profile_fingerprint",
        "capacity_profile_acal",
        "corrected_mpm_acal","corrected_mpm_critical_shields",
    }
    methods={EH_LEGACY_METHOD_ID,EH_DELFT_METHOD_ID,
             MPM_LEGACY_METHOD_ID,MPM_DELFT_METHOD_ID,"capacity.general"}
    sha=re.compile(r"[0-9a-f]{64}\Z")
    envelope=route.get("envelope_version") if isinstance(route,dict) else None
    foundation=envelope==54
    expected_execution_semantic=(
        CAPACITY_FOUNDATION_EXECUTION_SEMANTIC_ID if foundation else
        capacity_execution_semantic_id())
    if (not isinstance(route,dict) or set(route)!=expected_keys or
            envelope not in (50,54) or route.get("identity_schema")!=1 or
            route.get("ordered_parts")!=[
                "capacity_execution_fingerprint","capacity_profile_fingerprint"] or
            route.get("method_id") not in methods or
            isinstance(route.get("capacity_profile_acal"),bool) or
            not isinstance(route.get("capacity_profile_acal"),(int,float)) or
            not math.isfinite(float(route["capacity_profile_acal"])) or
            float(route["capacity_profile_acal"])<0. or
            route.get("capacity_execution_semantic_id")!=
                expected_execution_semantic or
            route.get("capacity_profile_semantic_id")!=
                capacity_profile_semantic_id() or
            not isinstance(route.get("capacity_execution_fingerprint"),str) or
            sha.fullmatch(route["capacity_execution_fingerprint"]) is None or
            not isinstance(route.get("capacity_profile_fingerprint"),str) or
            sha.fullmatch(route["capacity_profile_fingerprint"]) is None):
        raise ValueError("Native capacity result identity is invalid")
    if ((foundation and route["method_id"] not in {
            EH_DELFT_METHOD_ID, MPM_DELFT_METHOD_ID, "capacity.general"}) or
            (not foundation and route["method_id"] == "capacity.general")):
        raise ValueError("Native capacity result method/envelope identity differs")
    if route["method_id"]==MPM_DELFT_METHOD_ID and (
            route["corrected_mpm_acal"]!=route["capacity_profile_acal"] or
            route["corrected_mpm_critical_shields"]!=.047 or
            (not foundation and route["capacity_profile_acal"]!=1.)):
        raise ValueError("Corrected MPM result identity differs from its admitted slice")
    sidecar_name=marker.get("capacity_execution_sidecar")
    sidecar_sha=marker.get("capacity_execution_sha256")
    sidecar_path=run_root/sidecar_name if isinstance(sidecar_name,str) else None
    if (sidecar_name!="capacity_execution.json" or
            not isinstance(sidecar_sha,str) or sha.fullmatch(sidecar_sha) is None or
            sidecar_path is None or not sidecar_path.is_file() or
            sidecar_path.is_symlink() or _digest(sidecar_path)!=sidecar_sha):
        raise ValueError("Capacity execution sidecar changed after preparation")
    sidecar=json.loads(sidecar_path.read_text(encoding="utf-8"))
    execution=sidecar.get(
        "capacity_foundation_execution_identity" if foundation else
        "execution_identity")
    ordered=sidecar.get(
        "capacity_foundation_ordered_route_identity" if foundation else
        "ordered_route_identity")
    if (sidecar.get("native_deck_identity_serialized") is not True or
            sidecar.get("native_deck_envelope_version")!=envelope or
            not isinstance(execution,dict) or not isinstance(ordered,dict) or
            execution.get("method_id")!=route["method_id"] or
            execution.get("semantic_id")!=route["capacity_execution_semantic_id"] or
            execution.get("fingerprint")!=route["capacity_execution_fingerprint"] or
            ordered.get("ordered_parts")!=route["ordered_parts"] or
            ordered.get("capacity_execution_fingerprint")!=
                route["capacity_execution_fingerprint"] or
            ordered.get("capacity_profile_fingerprint")!=
                route["capacity_profile_fingerprint"] or
            (foundation and marker.get(
                "capacity_foundation_execution_identity")!=execution) or
            (foundation and marker.get(
                "capacity_foundation_ordered_route_identity")!=ordered) or
            marker.get("capacity_profile_fingerprint")!=
                route["capacity_profile_fingerprint"] or
            marker.get("capacity_profile_semantic_id")!=
                route["capacity_profile_semantic_id"]):
        raise ValueError(
            "Native deck, capacity sidecar, and prepared marker identities differ")
    return {**route,"capacity_execution_sidecar_sha256":sidecar_sha,
            "native_deck_fingerprint":case.get("fingerprint")}


def _hiding_foundation_result_contract(
    run_root: Path, case: dict, marker: dict,
) -> dict | None:
    """Authenticate the complete additive Q3-H1/v55 result route.

    Q3-H1 is an outer envelope around the retained Q3-C/v54 capacity route.
    The saved project therefore carries only its deferred declaration, whereas
    the prepared sidecar, resolved inputs, run marker and native case all carry
    the forcing-bound execution document.  A result is accepted only when all
    of those independently durable views agree with the fingerprints printed
    by the native executable before its first update.
    """
    case_format=case.get("format")
    receipt=case.get("hiding_foundation_identity")
    if case_format not in (
            "xbraids_cutcell-v55","xbraids_cutcell-v56",
            "xbraids_cutcell-v57"):
        if receipt is not None:
            raise ValueError(
                "Historical native result unexpectedly claims a v55 hiding identity")
        return None
    if case_format in ("xbraids_cutcell-v56","xbraids_cutcell-v57") and \
            receipt is None:
        # A v56/v57 document may wrap v54 directly (law 2/8/11 without hiding).
        return None
    if not isinstance(receipt,dict):
        raise ValueError("Native v55 result lacks its hiding execution identity")
    receipt_keys={
        "envelope_version","canonical_preimage_ascii",
        "canonical_preimage_sha256",HIDING_FOUNDATION_EXECUTION_KEY,
        HIDING_FOUNDATION_ORDERED_ROUTE_KEY,
    }
    if set(receipt)!=receipt_keys or receipt.get("envelope_version")!=55:
        raise ValueError("Native v55 hiding identity receipt is invalid")
    try:
        identity=validate_hiding_foundation_execution_document(
            receipt[HIDING_FOUNDATION_EXECUTION_KEY])
        preimage=json.dumps(
            identity,sort_keys=True,separators=(",",":"),ensure_ascii=True,
            allow_nan=False)
        encoded=preimage.encode("ascii")
    except (KeyError,TypeError,ValueError,UnicodeEncodeError) as exc:
        raise ValueError(
            f"Native v55 hiding execution identity is invalid: {exc}") from exc
    preimage_sha=hashlib.sha256(encoded).hexdigest()
    if (receipt.get("canonical_preimage_ascii")!=preimage or
            receipt.get("canonical_preimage_sha256")!=preimage_sha):
        raise ValueError("Native v55 hiding canonical preimage differs")
    route={
        "ordered_parts":[
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
        ],
        "capacity_execution_fingerprint":
            identity["capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":
            identity["capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":identity["fingerprint"],
    }
    if receipt.get(HIDING_FOUNDATION_ORDERED_ROUTE_KEY)!=route:
        raise ValueError("Native v55 hiding ordered route differs")

    capacity_route=case.get("capacity_route_identity")
    if (not isinstance(capacity_route,dict) or
            capacity_route.get("envelope_version")!=54 or
            capacity_route.get("capacity_execution_fingerprint")!=
                identity["capacity_execution_fingerprint"] or
            capacity_route.get("capacity_profile_fingerprint")!=
                identity["capacity_profile_fingerprint"]):
        raise ValueError("Native v55 hiding route differs from nested Q3-C")
    if (marker.get(HIDING_FOUNDATION_EXECUTION_KEY)!=identity or
            marker.get(HIDING_FOUNDATION_ORDERED_ROUTE_KEY)!=route or
            marker.get("capacity_profile_fingerprint")!=
                identity["capacity_profile_fingerprint"] or
            marker.get("fingerprint")!=case.get("fingerprint")):
        raise ValueError(
            "Native v55 case and prepared marker hiding identities differ")
    base_execution=marker.get("capacity_foundation_execution_identity")
    base_route=marker.get("capacity_foundation_ordered_route_identity")
    if (not isinstance(base_execution,dict) or
            base_execution.get("fingerprint")!=
                identity["capacity_execution_fingerprint"] or
            not isinstance(base_route,dict) or
            base_route.get("capacity_execution_fingerprint")!=
                identity["capacity_execution_fingerprint"] or
            base_route.get("capacity_profile_fingerprint")!=
                identity["capacity_profile_fingerprint"]):
        raise ValueError("Prepared v55 hiding identity differs from nested Q3-C")

    binding_key="hiding_foundation_binding"
    binding=marker.get(binding_key)
    binding_fingerprint=marker.get("hiding_foundation_binding_fingerprint")
    binding_keys={
        "schema","scope_id","semantic_id","state","execution_scope_id",
        "execution_semantic_id","mode","method_id","capacity_law",
        "fraction_ids","parker_alpha","wu_exponent",
        "capacity_profile_fingerprint","base_hiding_method_id",
        "composition_order","preparation_requirement","backend_selection",
        "automatic_cpu_fallback","fingerprint",
    }
    if not isinstance(binding,dict) or set(binding)!=binding_keys:
        raise ValueError("Prepared v55 hiding declaration is invalid")
    binding_body={key:value for key,value in binding.items()
                  if key!="fingerprint"}
    try:
        calculated_binding_fingerprint=hashlib.sha256(json.dumps(
            binding_body,sort_keys=True,separators=(",",":"),
            ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
    except (TypeError,ValueError,UnicodeEncodeError) as exc:
        raise ValueError("Prepared v55 hiding declaration is invalid") from exc
    expected_binding={
        "execution_scope_id":identity["scope_id"],
        "execution_semantic_id":identity["semantic_id"],
        "mode":identity["mode"],
        "method_id":identity["method_id"],
        "capacity_law":identity["capacity_law"],
        "fraction_ids":identity["fraction_ids"],
        "parker_alpha":identity["parker_alpha"],
        "wu_exponent":identity["wu_exponent"],
        "capacity_profile_fingerprint":
            identity["capacity_profile_fingerprint"],
        "base_hiding_method_id":"hiding.none",
        "composition_order":identity["composition_order"],
        "backend_selection":identity["backend_selection"],
        "automatic_cpu_fallback":identity["automatic_cpu_fallback"],
    }
    if (binding.get("schema")!=HIDING_FOUNDATION_BINDING_SCHEMA or
            binding.get("scope_id")!=HIDING_FOUNDATION_BINDING_SCOPE_ID or
            binding.get("semantic_id")!=
                HIDING_FOUNDATION_BINDING_SEMANTIC_ID or
            binding.get("state")!="deferred_prepared_capacity_execution" or
            binding.get("fingerprint")!=calculated_binding_fingerprint or
            binding_fingerprint!=calculated_binding_fingerprint or
            any(binding.get(key)!=value
                for key,value in expected_binding.items())):
        raise ValueError(
            "Prepared v55 hiding declaration differs from its execution identity")

    sha=re.compile(r"[0-9a-f]{64}\Z")
    project_path=run_root/"project.json"
    project_sha=marker.get("project_sha256")
    if (not isinstance(project_sha,str) or sha.fullmatch(project_sha) is None or
            not project_path.is_file() or project_path.is_symlink() or
            _digest(project_path)!=project_sha):
        raise ValueError("Prepared v55 project changed after preparation")
    try:
        project=json.loads(project_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Prepared v55 project is invalid") from exc
    if project.get(binding_key)!=binding:
        raise ValueError(
            "Saved v55 project declaration differs from its prepared marker")

    sidecar_name=marker.get("capacity_execution_sidecar")
    sidecar_sha=marker.get("capacity_execution_sha256")
    sidecar_path=(run_root/sidecar_name
                  if isinstance(sidecar_name,str) else None)
    if (sidecar_name!="capacity_execution.json" or
            not isinstance(sidecar_sha,str) or sha.fullmatch(sidecar_sha) is None or
            sidecar_path is None or not sidecar_path.is_file() or
            sidecar_path.is_symlink() or _digest(sidecar_path)!=sidecar_sha):
        raise ValueError("Prepared v55 capacity sidecar changed after preparation")
    try:
        sidecar=json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Prepared v55 capacity sidecar is invalid") from exc
    if (sidecar.get(binding_key)!=binding or
            sidecar.get(HIDING_FOUNDATION_EXECUTION_KEY)!=identity or
            sidecar.get(HIDING_FOUNDATION_ORDERED_ROUTE_KEY)!=route):
        raise ValueError(
            "Prepared v55 capacity sidecar hiding identity differs")

    resolved_path=run_root/"resolved_inputs.json"
    resolved_sha=marker.get("resolved_inputs_sha256")
    if (not isinstance(resolved_sha,str) or sha.fullmatch(resolved_sha) is None or
            not resolved_path.is_file() or resolved_path.is_symlink() or
            _digest(resolved_path)!=resolved_sha):
        raise ValueError("Prepared v55 resolved inputs changed after preparation")
    try:
        resolved=json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Prepared v55 resolved inputs are invalid") from exc
    sediment=resolved.get("sediment")
    if (not isinstance(sediment,dict) or sediment.get(binding_key)!=binding or
            sediment.get(HIDING_FOUNDATION_EXECUTION_KEY)!=identity or
            sediment.get(HIDING_FOUNDATION_ORDERED_ROUTE_KEY)!=route):
        raise ValueError("Prepared v55 resolved-input hiding identity differs")

    diagnostic_names=(
        "channel_composition.txt","channel_composition_boundary.txt",
        "channel_composition_face.txt",
    )
    expected_headers={
        "hiding_foundation_identity_envelope":"55",
        "hiding_foundation_execution_fingerprint":identity["fingerprint"],
    }
    for name in diagnostic_names:
        path=run_root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Native v55 hiding result receipt is missing: {name}")
        headers={}
        try:
            lines=path.read_text(encoding="ascii").splitlines()
        except (OSError,UnicodeError) as exc:
            raise ValueError(
                f"Native v55 hiding result receipt is invalid: {name}") from exc
        for raw in lines:
            if not raw.startswith("# "):
                continue
            parts=raw[2:].split()
            if len(parts)==2 and parts[0] in expected_headers:
                if parts[0] in headers:
                    raise ValueError(
                        f"Native v55 result repeats a hiding identity header: {name}")
                headers[parts[0]]=parts[1]
        if headers!=expected_headers:
            raise ValueError(
                f"Native v55 hiding headers disagree with prepared identity: {name}")
    return {
        "envelope_version":55,
        HIDING_FOUNDATION_EXECUTION_KEY:identity,
        HIDING_FOUNDATION_ORDERED_ROUTE_KEY:route,
        "hiding_foundation_binding_fingerprint":binding_fingerprint,
        "canonical_preimage_sha256":preimage_sha,
        "capacity_execution_sidecar_sha256":sidecar_sha,
        "resolved_inputs_sha256":resolved_sha,
        "native_result_headers_verified":True,
        "native_result_diagnostics":list(diagnostic_names),
        "case_metadata_fingerprint":case.get("fingerprint"),
    }


def _secondary_foundation_result_contract(
    run_root: Path,case: dict,marker: dict,
) -> dict | None:
    """Authenticate all durable views and native headers for Q3-H2/v56."""
    receipt=case.get("secondary_foundation_identity")
    if case.get("format")!="xbraids_cutcell-v56":
        if receipt is not None:
            raise ValueError(
                "Historical native result unexpectedly claims a v56 secondary identity")
        return None
    if not isinstance(receipt,dict) or set(receipt)!={
            "envelope_version","canonical_preimage_ascii",
            "canonical_preimage_sha256",SECONDARY_FOUNDATION_EXECUTION_KEY,
            SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY} or \
            receipt.get("envelope_version")!=56:
        raise ValueError("Native v56 result lacks its secondary execution identity")
    try:
        identity=validate_secondary_foundation_execution_document(
            receipt[SECONDARY_FOUNDATION_EXECUTION_KEY])
        preimage=json.dumps(identity,sort_keys=True,separators=(",",":"),
                            ensure_ascii=True,allow_nan=False)
        preimage_sha=hashlib.sha256(preimage.encode("ascii")).hexdigest()
    except (KeyError,TypeError,ValueError,UnicodeEncodeError) as exc:
        raise ValueError(
            f"Native v56 secondary execution identity is invalid: {exc}") from exc
    if (receipt.get("canonical_preimage_ascii")!=preimage or
            receipt.get("canonical_preimage_sha256")!=preimage_sha):
        raise ValueError("Native v56 secondary canonical preimage differs")
    route={
        "ordered_parts":[
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
            "secondary_execution_fingerprint"],
        "capacity_execution_fingerprint":identity[
            "capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":identity[
            "capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":identity[
            "hiding_execution_fingerprint"],
        "secondary_execution_fingerprint":identity["fingerprint"],
    }
    if receipt.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=route:
        raise ValueError("Native v56 secondary ordered route differs")
    base=case.get("capacity_route_identity")
    hiding=case.get("hiding_foundation_identity")
    if (not isinstance(base,dict) or base.get("envelope_version")!=54 or
            base.get("capacity_execution_fingerprint")!=
                identity["capacity_execution_fingerprint"] or
            base.get("capacity_profile_fingerprint")!=
                identity["capacity_profile_fingerprint"] or
            (identity["base_envelope_version"]==55 and (
                not isinstance(hiding,dict) or hiding.get("envelope_version")!=55 or
                hiding.get(HIDING_FOUNDATION_EXECUTION_KEY,{}).get("fingerprint")!=
                    identity["hiding_execution_fingerprint"])) or
            (identity["base_envelope_version"]==54 and hiding is not None)):
        raise ValueError("Native v56 secondary route differs from its nested foundation")
    binding=marker.get(SECONDARY_FOUNDATION_BINDING_KEY)
    try:
        binding=_validate_secondary_foundation_binding_document(binding)
    except ValueError as exc:
        raise ValueError("Prepared v56 secondary declaration is invalid") from exc
    binding_fingerprint=marker.get("secondary_foundation_binding_fingerprint")
    if (binding_fingerprint!=binding["fingerprint"] or
            binding["execution_scope_id"]!=identity["scope_id"] or
            binding["execution_semantic_id"]!=identity["semantic_id"] or
            binding["method_id"]!=identity["method_id"] or
            binding["model"]!=identity["model"] or
            binding["capacity_law"]!=identity["capacity_law"] or
            binding["capacity_profile_fingerprint"]!=
                identity["capacity_profile_fingerprint"] or
            binding["base_envelope_version"]!=identity["base_envelope_version"] or
            binding["roughness_method_id"]!=identity["roughness_method_id"] or
            any(binding[key]!=identity[key] for key in (
                "coriolis_disposition","coriolis_constant_s_inverse",
                "coriolis_source","rmincf","chzmin","espir","betac")) or
            marker.get(SECONDARY_FOUNDATION_EXECUTION_KEY)!=identity or
            marker.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=route or
            marker.get("fingerprint")!=case.get("fingerprint")):
        raise ValueError(
            "Prepared v56 secondary declaration differs from its execution identity")
    sha=re.compile(r"[0-9a-f]{64}\Z")
    project_path=run_root/"project.json"
    sidecar_path=run_root/"capacity_execution.json"
    resolved_path=run_root/"resolved_inputs.json"
    required=(
        (project_path,marker.get("project_sha256"),"project"),
        (sidecar_path,marker.get("capacity_execution_sha256"),"capacity sidecar"),
        (resolved_path,marker.get("resolved_inputs_sha256"),"resolved inputs"),
    )
    for path,expected,label in required:
        if (not isinstance(expected,str) or sha.fullmatch(expected) is None or
                not path.is_file() or path.is_symlink() or _digest(path)!=expected):
            raise ValueError(f"Prepared v56 {label} changed after preparation")
    try:
        project=json.loads(project_path.read_text(encoding="utf-8"))
        sidecar=json.loads(sidecar_path.read_text(encoding="utf-8"))
        resolved=json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Prepared v56 identity owner is invalid") from exc
    sediment=resolved.get("sediment")
    if (project.get(SECONDARY_FOUNDATION_BINDING_KEY)!=binding or
            sidecar.get(SECONDARY_FOUNDATION_BINDING_KEY)!=binding or
            sidecar.get(SECONDARY_FOUNDATION_EXECUTION_KEY)!=identity or
            sidecar.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=route or
            not isinstance(sediment,dict) or
            sediment.get(SECONDARY_FOUNDATION_BINDING_KEY)!=binding or
            sediment.get(SECONDARY_FOUNDATION_EXECUTION_KEY)!=identity or
            sediment.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=route):
        raise ValueError("Prepared v56 secondary identity owners differ")
    diagnostic_names=(
        "channel_composition.txt","channel_composition_boundary.txt",
        "channel_composition_face.txt","secondary_flow.txt")
    expected_headers={
        "secondary_foundation_identity_envelope":"56",
        "secondary_foundation_execution_fingerprint":identity["fingerprint"],
    }
    for name in diagnostic_names:
        path=run_root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Native v56 secondary result receipt is missing: {name}")
        headers={}
        try:lines=path.read_text(encoding="ascii").splitlines()
        except (OSError,UnicodeError) as exc:
            raise ValueError(
                f"Native v56 secondary result receipt is invalid: {name}") from exc
        for raw in lines:
            if not raw.startswith("# "):continue
            parts=raw[2:].split()
            if len(parts)==2 and parts[0] in expected_headers:
                if parts[0] in headers:
                    raise ValueError(
                        f"Native v56 result repeats a secondary identity header: {name}")
                headers[parts[0]]=parts[1]
        if headers!=expected_headers:
            raise ValueError(
                f"Native v56 secondary headers disagree with prepared identity: {name}")
    return {
        "envelope_version":56,
        SECONDARY_FOUNDATION_EXECUTION_KEY:identity,
        SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:route,
        "secondary_foundation_binding_fingerprint":binding_fingerprint,
        "canonical_preimage_sha256":preimage_sha,
        "capacity_execution_sidecar_sha256":marker["capacity_execution_sha256"],
        "resolved_inputs_sha256":marker["resolved_inputs_sha256"],
        "native_result_headers_verified":True,
        "native_result_diagnostics":list(diagnostic_names),
        "case_metadata_fingerprint":case.get("fingerprint"),
    }


def _prognostic_secondary_result_contract(
    run_root: Path,case: dict,marker: dict,
) -> dict | None:
    """Authenticate every durable Q3-H3/v57 owner and native receipt."""
    receipt=case.get("prognostic_secondary_identity")
    if case.get("format")!="xbraids_cutcell-v57":
        if receipt is not None:
            raise ValueError(
                "Historical native result unexpectedly claims a v57 "
                "prognostic-secondary identity")
        return None
    receipt_keys={
        "envelope_version","canonical_preimage_ascii",
        "canonical_preimage_sha256",
        "prognostic_secondary_execution_fingerprint",
        PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY,
        PROGNOSTIC_SECONDARY_STATE_UNITS_KEY,
        PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY,
        PROGNOSTIC_SECONDARY_EXECUTION_KEY,
        PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    }
    if (not isinstance(receipt,dict) or set(receipt)!=receipt_keys or
            receipt.get("envelope_version")!=57):
        raise ValueError(
            "Native v57 result lacks its prognostic-secondary execution identity")
    try:
        identity=validate_prognostic_secondary_execution_document(
            receipt[PROGNOSTIC_SECONDARY_EXECUTION_KEY])
        preimage=json.dumps(identity,sort_keys=True,separators=(",",":"),
                            ensure_ascii=True,allow_nan=False)
        preimage_sha=hashlib.sha256(preimage.encode("ascii")).hexdigest()
    except (KeyError,TypeError,ValueError,UnicodeEncodeError) as exc:
        raise ValueError(
            f"Native v57 prognostic-secondary execution identity is invalid: {exc}"
        ) from exc
    if (receipt.get("canonical_preimage_ascii")!=preimage or
            receipt.get("canonical_preimage_sha256")!=preimage_sha or
            receipt.get("prognostic_secondary_execution_fingerprint")!=
                identity["fingerprint"] or
            receipt.get(PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY)!=
                identity["bed_boundary_rotation_policy"] or
            receipt.get(PROGNOSTIC_SECONDARY_STATE_UNITS_KEY)!="m2_s" or
            receipt.get(PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY)!=
                identity["integrated_ledger_units"]):
        raise ValueError(
            "Native v57 prognostic-secondary canonical preimage differs")
    route={
        "capacity_execution_fingerprint":identity[
            "capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":identity[
            "capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":identity[
            "hiding_execution_fingerprint"],
        "prognostic_secondary_execution_fingerprint":identity["fingerprint"],
    }
    if receipt.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=route:
        raise ValueError(
            "Native v57 prognostic-secondary ordered route differs")
    base=case.get("capacity_route_identity")
    hiding=case.get("hiding_foundation_identity")
    if (not isinstance(base,dict) or base.get("envelope_version")!=54 or
            base.get("capacity_execution_fingerprint")!=
                identity["capacity_execution_fingerprint"] or
            base.get("capacity_profile_fingerprint")!=
                identity["capacity_profile_fingerprint"] or
            (identity["base_envelope_version"]==55 and (
                not isinstance(hiding,dict) or
                hiding.get("envelope_version")!=55 or
                hiding.get(HIDING_FOUNDATION_EXECUTION_KEY,{}).get(
                    "fingerprint")!=identity["hiding_execution_fingerprint"])) or
            (identity["base_envelope_version"]==54 and hiding is not None)):
        raise ValueError(
            "Native v57 prognostic-secondary route differs from its nested foundation")
    binding=marker.get(PROGNOSTIC_SECONDARY_BINDING_KEY)
    try:
        binding=_validate_prognostic_secondary_binding_document(binding)
    except ValueError as exc:
        raise ValueError(
            "Prepared v57 prognostic-secondary declaration is invalid") from exc
    binding_fingerprint=marker.get(
        "prognostic_secondary_binding_fingerprint")
    binding_fields=(
        "method_id","model","capacity_law","capacity_profile_fingerprint",
        "base_envelope_version","roughness_method_id","ordered_fraction_ids",
        "transport_direction","inflow_mode","coriolis_disposition",
        "coriolis_constant_s_inverse","coriolis_source","rmincf","chzmin",
        "espir","betac","scalar_diffusivity_m2_s","von_karman_constant",
        "gpu_checkpoint_schema")
    if (binding_fingerprint!=binding["fingerprint"] or
            binding["execution_scope_id"]!=identity["scope_id"] or
            binding["execution_semantic_id"]!=identity["semantic_id"] or
            any(binding[key]!=identity[key] for key in binding_fields) or
            binding["bed_layer_count"]!=
                identity["runtime_layout"]["bed_layer_count"] or
            marker.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)!=identity or
            marker.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=route or
            marker.get(PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY)!=
                identity["bed_boundary_rotation_policy"] or
            marker.get(PROGNOSTIC_SECONDARY_STATE_UNITS_KEY)!="m2_s" or
            marker.get(PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY)!=
                identity["integrated_ledger_units"] or
            marker.get("fingerprint")!=case.get("fingerprint") or
            marker.get(SECONDARY_FOUNDATION_BINDING_KEY) is not None):
        raise ValueError(
            "Prepared v57 prognostic-secondary declaration differs from its "
            "execution identity")
    sha=re.compile(r"[0-9a-f]{64}\Z")
    project_path=run_root/"project.json"
    sidecar_path=run_root/"capacity_execution.json"
    resolved_path=run_root/"resolved_inputs.json"
    required=(
        (project_path,marker.get("project_sha256"),"project"),
        (sidecar_path,marker.get("capacity_execution_sha256"),"capacity sidecar"),
        (resolved_path,marker.get("resolved_inputs_sha256"),"resolved inputs"),
    )
    for path,expected,label in required:
        if (not isinstance(expected,str) or sha.fullmatch(expected) is None or
                not path.is_file() or path.is_symlink() or _digest(path)!=expected):
            raise ValueError(f"Prepared v57 {label} changed after preparation")
    try:
        project=json.loads(project_path.read_text(encoding="utf-8"))
        sidecar=json.loads(sidecar_path.read_text(encoding="utf-8"))
        resolved=json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Prepared v57 identity owner is invalid") from exc
    sediment=resolved.get("sediment")
    for owner,label in (
            (sidecar,"capacity sidecar"),(sediment,"resolved inputs")):
        if (not isinstance(owner,dict) or
                owner.get(PROGNOSTIC_SECONDARY_BINDING_KEY)!=binding or
                owner.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)!=identity or
                owner.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=route or
                owner.get(PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY)!=
                    identity["bed_boundary_rotation_policy"] or
                owner.get(PROGNOSTIC_SECONDARY_STATE_UNITS_KEY)!="m2_s" or
                owner.get(PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY)!=
                    identity["integrated_ledger_units"]):
            raise ValueError(
                f"Prepared v57 prognostic-secondary {label} identity differs")
    if project.get(PROGNOSTIC_SECONDARY_BINDING_KEY)!=binding:
        raise ValueError(
            "Saved v57 project declaration differs from its prepared marker")
    diagnostic_names=(
        "channel_composition.txt","channel_composition_boundary.txt",
        "channel_composition_face.txt","secondary_flow.txt",
        "secondary_flow_budget.txt","secondary_momentum_budget.txt")
    expected_headers={
        "prognostic_secondary_identity_envelope":"57",
        "prognostic_secondary_execution_fingerprint":identity["fingerprint"],
        PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY:
            identity["bed_boundary_rotation_policy"],
        PROGNOSTIC_SECONDARY_STATE_UNITS_KEY:"m2_s",
        PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY:
            identity["integrated_ledger_units"],
    }
    for name in diagnostic_names:
        path=run_root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"Native v57 prognostic-secondary result receipt is missing: {name}")
        headers={}
        try:
            lines=path.read_text(encoding="ascii").splitlines()
        except (OSError,UnicodeError) as exc:
            raise ValueError(
                f"Native v57 prognostic-secondary result receipt is invalid: {name}"
            ) from exc
        for raw in lines:
            if not raw.startswith("# "):
                continue
            parts=raw[2:].split()
            if len(parts)==2 and parts[0] in expected_headers:
                if parts[0] in headers:
                    raise ValueError(
                        "Native v57 result repeats a prognostic-secondary "
                        f"identity header: {name}")
                headers[parts[0]]=parts[1]
        if headers!=expected_headers:
            raise ValueError(
                "Native v57 prognostic-secondary headers disagree with "
                f"prepared identity: {name}")
        if name=="secondary_flow_budget.txt" and (
                "# time_s integral_hI_m4_s initial_hI_m4_s "
                "boundary_hI_m4_s source_hI_m4_s residual_hI_m4_s"
                not in lines):
            raise ValueError(
                "Native v57 prognostic-secondary budget units differ from "
                "signed hI area-integral units m4/s")
        if name=="secondary_momentum_budget.txt" and (
                "# time_s current_x_m4_s current_y_m4_s initial_x_m4_s "
                "initial_y_m4_s ordinary_boundary_x_m4_s "
                "ordinary_boundary_y_m4_s secondary_stress_boundary_x_m4_s "
                "secondary_stress_boundary_y_m4_s "
                "interior_source_reconciliation_x_m4_s "
                "interior_source_reconciliation_y_m4_s residual_x_m4_s "
                "residual_y_m4_s" not in lines):
            raise ValueError(
                "Native v57 prognostic-secondary momentum budget units "
                "differ from area-integrated momentum units m4/s")
    return {
        "envelope_version":57,
        PROGNOSTIC_SECONDARY_EXECUTION_KEY:identity,
        PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:route,
        "prognostic_secondary_binding_fingerprint":binding_fingerprint,
        PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY:
            identity["bed_boundary_rotation_policy"],
        PROGNOSTIC_SECONDARY_STATE_UNITS_KEY:"m2_s",
        PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY:
            identity["integrated_ledger_units"],
        "canonical_preimage_sha256":preimage_sha,
        "capacity_execution_sidecar_sha256":marker["capacity_execution_sha256"],
        "resolved_inputs_sha256":marker["resolved_inputs_sha256"],
        "native_result_headers_verified":True,
        "native_result_diagnostics":list(diagnostic_names),
        "case_metadata_fingerprint":case.get("fingerprint"),
    }


def _checked_execution_component_documents(
    profile, documents: object, *, label: str,
) -> dict[str, Mapping[str, object]]:
    """Authenticate exact component documents against one execution profile."""
    expected={row.component_id:row for row in profile.component_profiles}
    if not isinstance(documents,dict) or set(documents)!=set(expected):
        raise ValueError(f"Prepared v52 {label} component inventory differs")
    checked={}
    for component_id in sorted(documents):
        document=documents[component_id]
        if (not isinstance(document,Mapping) or
                bind_component_profile(component_id,document)!=expected[component_id]):
            raise ValueError(
                f"Prepared v52 {label} component differs: {component_id}")
        checked[component_id]=document
    return checked


def _checked_v53_component_receipts(profile, receipts: object) -> dict[str, dict]:
    """Reconstruct the exact canonical component documents carried by v53."""
    expected={row.component_id:row for row in profile.component_profiles}
    if not isinstance(receipts,dict) or set(receipts)!=set(expected):
        raise ValueError("Native v53 component-document inventory differs")
    receipt_keys={
        "canonical_preimage_ascii","document_sha256","fingerprint","schema",
        "scope_id","semantic_id",
    }
    checked={}
    for component_id in sorted(receipts):
        row=receipts[component_id]
        if not isinstance(row,dict) or set(row)!=receipt_keys:
            raise ValueError(
                f"Native v53 {component_id} component receipt is invalid")
        preimage=row.get("canonical_preimage_ascii")
        try:
            encoded=preimage.encode("ascii")
            document=json.loads(preimage)
            canonical=json.dumps(
                document,sort_keys=True,separators=(",",":"),
                ensure_ascii=True,allow_nan=False)
            binding=bind_component_profile(component_id,document)
        except (AttributeError,UnicodeEncodeError,ValueError,TypeError,
                ExecutionProfileError) as exc:
            raise ValueError(
                f"Native v53 {component_id} component receipt is invalid: {exc}") from exc
        expected_binding=expected[component_id]
        if (not isinstance(document,dict) or canonical!=preimage or
                hashlib.sha256(encoded).hexdigest()!=row.get("document_sha256") or
                binding!=expected_binding or
                row.get("document_sha256")!=expected_binding.document_sha256 or
                row.get("fingerprint")!=expected_binding.fingerprint or
                row.get("schema")!=expected_binding.schema or
                row.get("scope_id")!=expected_binding.scope_id or
                row.get("semantic_id")!=expected_binding.semantic_id):
            raise ValueError(
                f"Native v53 {component_id} component receipt differs")
        checked[component_id]=document
    return checked


def _capacity_component_physics(document: Mapping[str, object], *, backend):
    """Validate a capacity component and omit only its route identity fields."""
    try:
        profile=load_capacity_profile(document)
        checked=capacity_profile_document(profile)
    except CapacityProfileError as exc:
        raise ValueError(f"Prepared v52 capacity component is invalid: {exc}") from exc
    if profile.backend!=backend:
        raise ValueError("Prepared v52 capacity component backend differs")
    return {
        key:value for key,value in checked.items()
        if key not in {"backend","fingerprint"}
    }


def _execution_profile_result_contract(
    run_root: Path, case: dict, marker: dict,
) -> dict | None:
    """Authenticate the nested v52/v53 profile through native output.

    The native solver already hashes and reconstructs both canonical preimages
    before its first step.  Result loading independently repeats the managed
    validation and then requires the identities printed by the native process;
    a valid deck receipt alone is not treated as evidence that this executable
    consumed it.
    """
    receipt=case.get("execution_profile_v2")
    channel=case.get("channel_composition_identity")
    if receipt is None:
        if (case.get("format") in {"xbraids_cutcell-v52","xbraids_cutcell-v53",
                                   "xbraids_cutcell-v54","xbraids_cutcell-v55",
                                   "xbraids_cutcell-v56","xbraids_cutcell-v57"} or
                marker.get("execution_profile_v2_state")=="canonical" or
                isinstance(channel,dict) and channel.get("envelope_version") in (52,53)):
            raise ValueError("Native v52/v53 channel result lacks its execution profile")
        return None
    if not isinstance(receipt,dict):
        raise ValueError("Native v52/v53 execution-profile receipt is invalid")
    envelope_version=receipt.get("envelope_version")
    if envelope_version not in (52,53):
        raise ValueError("Native v52/v53 execution-profile receipt is invalid")
    receipt_keys={
        "envelope_version","runtime_backend","profile_backend","schema",
        "semantic_id","fingerprint","canonical_preimage_sha256",
        "canonical_preimage_ascii","profile",
    }
    if envelope_version==53:
        receipt_keys.add("component_documents")
    channel_keys={
        "envelope_version","wire_schema","canonical_wire_order",
        "class_resolution","profile_fingerprint_authority",
        "native_profile_fingerprint_verification","native_profile_preimage",
        "native_profile_preimage_sha256","ordered_fraction_identifiers",
        "native_verification_followup","factor_domain","global_factor_owner",
        "profile","execution_profile",
    }
    nested_keys={
        "schema","semantic_id","fingerprint","canonical_preimage_ascii","profile",
    }
    expected_authority=(
        "native_sha256_verified_component_preimages_v53"
        if envelope_version==53 else
        "native_sha256_verified_canonical_preimage_v52")
    case_format=case.get("format")
    q3c_outer_v54=(
        envelope_version==53 and case_format=="xbraids_cutcell-v54" and
        isinstance(case.get("capacity_route_identity"),dict) and
        case["capacity_route_identity"].get("envelope_version")==54)
    q3h_outer_v55=(
        envelope_version==53 and case_format=="xbraids_cutcell-v55" and
        isinstance(case.get("capacity_route_identity"),dict) and
        case["capacity_route_identity"].get("envelope_version")==54 and
        isinstance(case.get("hiding_foundation_identity"),dict) and
        case["hiding_foundation_identity"].get("envelope_version")==55)
    q3h2_outer_v56=(
        envelope_version==53 and case_format=="xbraids_cutcell-v56" and
        isinstance(case.get("capacity_route_identity"),dict) and
        case["capacity_route_identity"].get("envelope_version")==54 and
        isinstance(case.get("secondary_foundation_identity"),dict) and
        case["secondary_foundation_identity"].get("envelope_version")==56)
    q3h3_outer_v57=(
        envelope_version==53 and case_format=="xbraids_cutcell-v57" and
        isinstance(case.get("capacity_route_identity"),dict) and
        case["capacity_route_identity"].get("envelope_version")==54 and
        isinstance(case.get("prognostic_secondary_identity"),dict) and
        case["prognostic_secondary_identity"].get("envelope_version")==57)
    q3c_foundation_outer=(
        q3c_outer_v54 or q3h_outer_v55 or q3h2_outer_v56 or q3h3_outer_v57)
    if (set(receipt)!=receipt_keys or
            (case_format!=f"xbraids_cutcell-v{envelope_version}" and
             not q3c_foundation_outer) or
            not isinstance(channel,dict) or set(channel)!=channel_keys or
            receipt.get("runtime_backend")!="cpu" or
            receipt.get("schema")!=2 or
            receipt.get("semantic_id")!=execution_profile_semantic_id() or
            channel.get("envelope_version")!=envelope_version or
            channel.get("wire_schema")!=2 or
            channel.get("native_profile_fingerprint_verification") is not True or
            channel.get("profile_fingerprint_authority")!=expected_authority):
        raise ValueError("Native v52/v53 execution-profile receipt is invalid")
    try:
        profile=load_execution_profile(receipt["profile"])
        channel_profile=load_channel_composition_profile(channel["profile"])
        channel_wire=channel_composition_native_wire_document(channel_profile)
    except (ExecutionProfileError,ChannelCompositionError,KeyError,TypeError) as exc:
        raise ValueError(f"Native v52/v53 result profile is invalid: {exc}") from exc
    preimage=execution_profile_payload_json(profile)
    preimage_sha=hashlib.sha256(preimage.encode("ascii")).hexdigest()
    nested=channel.get("execution_profile")
    if (receipt.get("profile")!=execution_profile_document(profile) or
            receipt.get("fingerprint")!=profile.fingerprint or
            receipt.get("canonical_preimage_sha256")!=preimage_sha or
            receipt.get("canonical_preimage_ascii")!=preimage or
            receipt.get("profile_backend")!=
                (("backend_neutral_physical_profile" if envelope_version==53 else
                  "neutral_pending_q3s_route_admission")
                 if profile.backend is None else profile.backend) or
            channel.get("native_profile_preimage")!=channel_wire["preimage_ascii"] or
            channel.get("native_profile_preimage_sha256")!=
                channel_wire["preimage_sha256"] or
            channel.get("ordered_fraction_identifiers")!=[
                row.fraction_id for row in channel_profile.fractions] or
            not isinstance(nested,dict) or set(nested)!=nested_keys or
            nested.get("schema")!=profile.schema or
            nested.get("semantic_id")!=profile.semantic_id or
            nested.get("fingerprint")!=profile.fingerprint or
            nested.get("canonical_preimage_ascii")!=preimage or
            nested.get("profile")!=receipt["profile"]):
        raise ValueError("Native v52/v53 channel and execution-profile identities differ")
    channel_component=[row for row in profile.component_profiles
                       if row.component_id=="channel_composition"]
    if (len(channel_component)!=1 or
            channel_component[0]!=bind_component_profile(
                "channel_composition",channel["profile"]) or
            tuple(row.fraction_id for row in profile.fraction_layout)!=
                tuple(row.fraction_id for row in channel_profile.fractions) or
            tuple(row.transport_class for row in profile.fraction_layout)!=
                tuple(row.transport_class for row in channel_profile.fractions)):
        raise ValueError("Native v52/v53 execution profile does not bind its channel profile")
    native_component_documents=(
        _checked_v53_component_receipts(profile,receipt["component_documents"])
        if envelope_version==53 else None)
    backend_document=marker.get("execution_profile_v2_backend_profile")
    internal_admission=marker.get("execution_profile_v2_internal_admission")
    backend_admission="runtime"
    if backend_document is None:
        if (internal_admission!="validation_internal_neutral" or
                profile.backend is not None):
            raise ValueError("Prepared v52/v53 backend profile is missing")
        backend_profile=None
        backend_admission="validation_internal_neutral"
    else:
        if internal_admission is not None:
            raise ValueError(
                "Prepared v52/v53 runtime profile carries a validation-only admission")
        try:
            backend_profile=load_execution_profile(backend_document)
            neutral_documents=_checked_execution_component_documents(
                profile,marker.get("execution_profile_v2_components"),
                label="neutral")
            if (native_component_documents is not None and
                    native_component_documents!=neutral_documents):
                raise ValueError(
                    "Prepared v53 component documents differ from the native receipt")
            backend_documents=_checked_execution_component_documents(
                backend_profile,
                marker.get("execution_profile_v2_backend_components"),
                label="CPU")
            for component_id in neutral_documents:
                if component_id=="capacity" and envelope_version==52:
                    if (_capacity_component_physics(
                            neutral_documents[component_id],backend=None)!=
                            _capacity_component_physics(
                                backend_documents[component_id],backend="cpu")):
                        raise ValueError(
                            "Prepared v52/v53 CPU and neutral capacity components differ")
                elif neutral_documents[component_id]!=backend_documents[component_id]:
                    raise ValueError(
                        "Prepared v52/v53 CPU and neutral components differ: "+component_id)
            if is_frozen_q3sp_execution_profile(profile):
                # The pre-Q3-C seven-selector profile intentionally retains
                # its frozen ledger semantic id.  Re-resolving it against the
                # current ledger relabels otherwise identical historical
                # physics, so compare the two independently authenticated
                # documents after removing only backend and fingerprint.
                if not is_frozen_q3sp_execution_profile(backend_profile):
                    raise ValueError(
                        "Prepared frozen Q3-S/P backend profile lost its identity")
                def physical_profile(value):
                    document=execution_profile_document(value)
                    return {key:item for key,item in document.items()
                            if key not in {"backend","fingerprint"}}
                neutralized_matches=(
                    profile.backend is None and
                    physical_profile(backend_profile)==physical_profile(profile))
            else:
                neutralized=resolve_execution_profile(
                    fraction_layout=backend_profile.fraction_layout,
                    method_selections=backend_profile.method_selections,
                    parameter_bindings=backend_profile.parameter_bindings,
                    control_bindings=backend_profile.control_bindings,
                    component_profiles=profile.component_profiles,
                    backend=None,
                )
                neutralized_matches=neutralized==profile
        except (ExecutionProfileError,TypeError,KeyError) as exc:
            raise ValueError(f"Prepared v52/v53 backend profile is invalid: {exc}") from exc
        if (backend_profile.backend!="cpu" or not neutralized_matches or
                marker.get("execution_profile_v2_runtime_backend")!="cpu" or
                marker.get("execution_profile_v2_runtime_fingerprint")!=
                    backend_profile.fingerprint or
                marker.get("execution_profile_v2_project_fingerprint")!=
                    profile.fingerprint or
                marker.get("fingerprint")!=case.get("fingerprint")):
            raise ValueError("Prepared v52/v53 backend profile differs from the native profile")
        case_path=run_root/"case.data"
        expected_case_sha=marker.get("case_sha256")
        if (not isinstance(expected_case_sha,str) or len(expected_case_sha)!=64 or
                any(character not in "0123456789abcdef"
                    for character in expected_case_sha) or
                not case_path.is_file() or case_path.is_symlink() or
                _digest(case_path)!=expected_case_sha):
            raise ValueError("Prepared v52/v53 native case bytes differ from the marker")
    if (marker.get("execution_profile_v2_state")!="canonical" or
            marker.get("execution_profile_v2")!=receipt["profile"] or
            marker.get("execution_profile_v2_fingerprint")!=profile.fingerprint or
            marker.get("execution_profile_v2_semantic_id")!=profile.semantic_id):
        raise ValueError("Native v52/v53 deck and prepared marker identities differ")
    result_path=run_root/"channel_composition.txt"
    if not result_path.is_file() or result_path.is_symlink():
        raise ValueError("Native v52/v53 execution-profile result receipt is missing")
    wanted={
        "native_identity_envelope","canonical_preimage_sha256",
        "execution_profile_semantic_id","execution_profile_fingerprint",
    }
    headers={}
    component_envelopes=[]
    result_component_hashes={}
    foundation_headers={}
    foundation_wanted={
        "capacity_forcing_identity_envelope",
        "capacity_forcing_fingerprint","capacity_forcing_raw_sha256",
        "capacity_foundation_execution_fingerprint",
    }
    for raw in result_path.read_text(encoding="ascii").splitlines():
        if not raw.startswith("# "):
            continue
        parts=raw[2:].split()
        if len(parts)==2 and parts[0]=="component_identity_envelope":
            component_envelopes.append(parts[1])
            continue
        if len(parts)==3 and parts[0]=="component_document_sha256":
            if parts[1] in result_component_hashes:
                raise ValueError("Native v53 result repeats a component identity header")
            result_component_hashes[parts[1]]=parts[2]
            continue
        if len(parts)==2 and parts[0] in foundation_wanted:
            if parts[0] in foundation_headers:
                raise ValueError("Native v54 result repeats a forcing identity header")
            foundation_headers[parts[0]]=parts[1]
            continue
        if len(parts)==2 and parts[0] in wanted:
            if parts[0] in headers:
                raise ValueError("Native v52/v53 result repeats an identity header")
            headers[parts[0]]=parts[1]
    expected={
        "native_identity_envelope":"52",
        "canonical_preimage_sha256":channel_profile.fingerprint,
        "execution_profile_semantic_id":profile.semantic_id,
        "execution_profile_fingerprint":profile.fingerprint,
    }
    if headers!=expected:
        raise ValueError("Native v52/v53 result headers disagree with prepared identities")
    if envelope_version==53:
        expected_component_hashes={
            row.component_id:row.document_sha256
            for row in profile.component_profiles
        }
        if (component_envelopes!=["53"] or
                result_component_hashes!=expected_component_hashes):
            raise ValueError(
                "Native v53 result component headers disagree with prepared identities")
    elif component_envelopes or result_component_hashes:
        raise ValueError("Native v52 result unexpectedly claims v53 component identities")
    if q3c_foundation_outer:
        forcing=marker.get("typed_forcing_binding")
        route=case["capacity_route_identity"]
        array=(forcing.get("roughness_parameter_array")
               if isinstance(forcing,dict) else None)
        expected_foundation={
            "capacity_forcing_identity_envelope":"54",
            "capacity_forcing_fingerprint":(
                forcing.get("fingerprint") if isinstance(forcing,dict) else None),
            "capacity_forcing_raw_sha256":(
                array.get("sha256") if isinstance(array,dict) else None),
            "capacity_foundation_execution_fingerprint":
                route.get("capacity_execution_fingerprint"),
        }
        if (None in expected_foundation.values() or
                foundation_headers!=expected_foundation):
            raise ValueError(
                "Native v54/v55 result forcing headers disagree with prepared identities")
    elif foundation_headers:
        raise ValueError("Historical native result unexpectedly claims v54 forcing identity")
    return {
        "envelope_version":envelope_version,
        "execution_profile_semantic_id":profile.semantic_id,
        "execution_profile_fingerprint":profile.fingerprint,
        "selector_ledger_semantic_id":profile.ledger_semantic_id,
        "channel_composition_fingerprint":channel_profile.fingerprint,
        "host_backend_admission":backend_admission,
        "runtime_execution_profile_fingerprint":(
            None if backend_profile is None else backend_profile.fingerprint),
        "native_result_headers_verified":True,
        "case_metadata_fingerprint":case.get("fingerprint"),
    }


def _report(progress, value: float) -> None:
    if progress is not None:
        progress(float(value))


def _raise_if_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("BRAIDS result preparation cancelled")


def _file_stamp(path: Path) -> dict[str, int | str]:
    stat=path.stat()
    return {"name":path.name,"size":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns)}


def _read_binary_frames(path: Path) -> list[dict[str, Any]]:
    """Read the fixed-grid v1 stream emitted beside the ``frames.txt`` marker."""
    header_bytes = 8 + 7 * 4
    size = path.stat().st_size
    if size < header_bytes:
        raise ValueError("Native binary frames file is truncated")
    with path.open("rb") as handle:
        if handle.read(8) != b"BRDFRM01":
            raise ValueError("Native binary frames magic is invalid")
        version, nc, nb, nfrac, bed_layers, nstate, nledger = np.fromfile(
            handle, dtype="<i4", count=7
        ).tolist()
        if version != 1:
            raise ValueError(f"Unsupported native binary frames version: {version}")
        expected_state = 3 + (1 + bed_layers) * nfrac
        expected_ledger = 1 + 2 * nfrac
        if (min(nc, nfrac, bed_layers) < 1 or nb < 0 or
                nstate != expected_state or nledger != expected_ledger or bed_layers > 33):
            raise ValueError("Native binary frames header is invalid")
        frame_bytes = 16 + 8 * (nstate * nc + nledger * nb)
        payload = size - header_bytes
        if payload <= 0 or payload % frame_bytes:
            raise ValueError("Native binary frames file ends inside a frame")
        frames = []
        for _ in range(payload // frame_bytes):
            time_s = np.fromfile(handle, dtype="<f8", count=1)
            steps = np.fromfile(handle, dtype="<i8", count=1)
            raw_q = np.fromfile(handle, dtype="<f8", count=nstate * nc)
            raw_ledger = np.fromfile(handle, dtype="<f8", count=nledger * nb)
            q = raw_q.reshape((nstate, nc), order="F").T
            ledger = raw_ledger.reshape((nledger, nb), order="F").T
            if not np.isfinite(q).all() or not np.isfinite(ledger).all():
                raise ValueError("Native binary frame contains nonfinite data")
            frames.append({"time_s": float(time_s[0]), "steps": int(steps[0]),
                           "q": q, "ledger": ledger, "nfrac": int(nfrac),
                           "bed_layers": int(bed_layers)})
    return frames


def _frames_payload_path(marker: Path) -> Path:
    """Return the state-bearing artifact for ASCII or redirected binary frames."""
    try:
        with marker.open("r", encoding="utf-8") as handle:
            first = handle.readline().split()
    except (OSError, UnicodeDecodeError):
        return marker
    binary = marker.with_name("frames.bin")
    return binary if first == ["-23", "1"] and binary.is_file() else marker


def read_frames(path: str | Path) -> list[dict[str, Any]]:
    """Read ASCII, its binary redirect marker, or a direct binary payload."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic == b"BRDFRM01" or path.suffix.lower() == ".bin":
        return _read_binary_frames(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Native frames file is empty")
    header = list(map(int, lines[0].split()))
    if header == [-23, 1]:
        binary = path.with_name("frames.bin")
        if not binary.is_file():
            raise ValueError("Native frames marker refers to a missing frames.bin")
        return _read_binary_frames(binary)
    adaptive = len(header) == 4 and header[0] == -22
    if adaptive:
        _, nb, nfrac, bed_layers = header
        nc = 1
        if not 1 <= bed_layers <= 33:
            raise ValueError('Adaptive frame bed-layer count is invalid')
    elif len(header) == 2:
        nc, nb = header
        nfrac = bed_layers = 1
    elif len(header) == 3:
        nc, nb, nfrac = header
        bed_layers = 1
    elif len(header) == 4:
        nc, nb, nfrac, bed_layers = header
        if bed_layers < 2 or bed_layers > 33:
            raise ValueError("Layered native frames must contain one active and 1 to 32 substrate layers")
    else:
        raise ValueError("Native frames header must contain nc, nb, optional fraction count and optional bed-layer count")
    if min(nc, nfrac, bed_layers) < 1 or nb < 0:
        raise ValueError("Native frames header is invalid")
    nstate = 3 + (1 + bed_layers) * nfrac
    nledger = 1 + 2 * nfrac
    cursor = 1
    frames = []
    while cursor < len(lines):
        if adaptive:
            time_s, steps, count, mesh_id = lines[cursor].split()
            nc = int(count)
            if nc < 1:
                raise ValueError('Adaptive frame contains no leaf cells')
        if cursor + 1 + nc + nb > len(lines):
            raise ValueError("Native frames file ends inside a frame")
        if not adaptive:
            time_s, steps = lines[cursor].split()
        cursor += 1
        q = np.array([list(map(float, row.split())) for row in lines[cursor:cursor + nc]])
        if adaptive:
            node_ids = q[:, 0].astype(np.int64)
            if np.any(q[:, 0] != node_ids) or np.any(node_ids < 0) or len(set(node_ids)) != nc:
                raise ValueError('Adaptive frame contains invalid hierarchy node IDs')
            q = q[:, 1:]
        cursor += nc
        ledger = np.array([list(map(float, row.split())) for row in lines[cursor:cursor + nb]]).reshape(nb, nledger)
        cursor += nb
        if not np.isfinite(q).all() or not np.isfinite(ledger).all() or q.shape != (nc, nstate):
            raise ValueError("Native frame contains invalid data")
        frames.append({"time_s": float(time_s), "steps": int(steps), "q": q, "ledger": ledger,
                       "nfrac": nfrac, "bed_layers": bed_layers})
        if adaptive:
            frames[-1].update(node_ids=node_ids, mesh_id=int(mesh_id))
    if not frames:
        raise ValueError("Native frames file contains no result frames")
    return frames


def _write_boundary_budgets(run_root: Path, result: Path, case: dict[str, Any],
                            frames: list[dict[str, Any]] | None = None) -> str:
    """Materialize the authoritative accepted boundary ledger as a small CSV.

    Live-preview result preparation deliberately avoids decoding every saved
    state. Boundary diagnostics nevertheless need the native cumulative
    ledger, so stream past state rows and retain only the few boundary rows.
    Positive values are net transport into the computational domain; exports
    through outlets are negative.
    """
    inverse = {int(value): key for key, value in case["string_to_numeric_id"].items()}
    ledger_columns = case.get(
        "ledger_columns", ["water_volume_m3", "suspended_mass_kg", "bedload_mass_kg"])
    target = result / "boundary_budgets.csv"
    temporary = target.with_name(f".{uuid4().hex}.partial")
    result.mkdir(exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "bc_id", *ledger_columns])
        if frames is not None:
            for frame in frames:
                for row, values in enumerate(frame["ledger"], 1):
                    writer.writerow([frame["time_s"], inverse[row], *values])
        elif (run_root / "boundary_ledger.txt").is_file():
            with (run_root / "boundary_ledger.txt").open("r", encoding="utf-8") as source:
                header = list(map(int, source.readline().split()))
                if header != [len(inverse), len(ledger_columns)]:
                    raise ValueError("Native boundary ledger header disagrees with case metadata")
                while True:
                    time_row = source.readline()
                    if not time_row:
                        break
                    if not time_row.strip():
                        continue
                    time_values = time_row.split()
                    if len(time_values) != 1:
                        raise ValueError("Native boundary ledger time row is invalid")
                    for row in range(1, len(inverse) + 1):
                        ledger = source.readline().split()
                        if len(ledger) != len(ledger_columns):
                            raise ValueError("Native boundary ledger ends inside a saved frame")
                        writer.writerow([float(time_values[0]), inverse[row], *map(float, ledger)])
        else:
            with (run_root / "frames.txt").open("r", encoding="utf-8") as source:
                header = list(map(int, source.readline().split()))
                adaptive = len(header) == 4 and header[0] == -22
                if adaptive:
                    _, boundary_count, fraction_count, _ = header
                    fixed_cell_count = None
                elif len(header) in (2, 3, 4):
                    fixed_cell_count, boundary_count = header[:2]
                    fraction_count = 1 if len(header) == 2 else header[2]
                else:
                    raise ValueError("Native frames header is invalid")
                if boundary_count != len(inverse):
                    raise ValueError("Frame and case boundary counts disagree")
                ledger_width = 1 + 2 * fraction_count
                while True:
                    frame_header = source.readline()
                    if not frame_header:
                        break
                    values = frame_header.split()
                    if not values:
                        continue
                    if adaptive:
                        if len(values) != 4:
                            raise ValueError("Adaptive frame header is invalid")
                        time_s, _, cell_count, _ = values
                        cell_count = int(cell_count)
                    else:
                        if len(values) != 2:
                            raise ValueError("Frame header is invalid")
                        time_s, _ = values
                        cell_count = int(fixed_cell_count)
                    for _ in range(cell_count):
                        if not source.readline():
                            raise ValueError("Native frames file ends inside a state frame")
                    for row in range(1, boundary_count + 1):
                        ledger = source.readline().split()
                        if len(ledger) != ledger_width:
                            raise ValueError("Native frames file ends inside a boundary ledger")
                        writer.writerow([float(time_s), inverse[row], *map(float, ledger)])
    _replace_with_windows_retry(temporary, target)
    return target.name


def _read_final_adaptive_mesh(path: Path) -> dict[str, Any] | None:
    """Read just the final AMR topology, without retaining result fields.

    ``frames.txt`` is intentionally portable ASCII, but a completed AMR run
    can contain hundreds of MB of historical states.  The completed-results
    mesh needs only the final active hierarchy-node IDs.  Search backward from
    EOF so old, already-obsolete adaptive frames are not even read.
    """
    with path.open("r", encoding="utf-8") as handle:
        header = [int(value) for value in handle.readline().split()]
    if len(header) != 4 or header[0] != -22:
        return None
    _, boundary_count, _, bed_layers = header
    if boundary_count < 0 or not 1 <= bed_layers <= 33:
        raise ValueError("Adaptive frames header is invalid")
    size = path.stat().st_size
    window = min(size, 4 * 1024 * 1024)
    while True:
        with path.open("rb") as handle:
            handle.seek(max(0, size - window))
            tail = handle.read().decode("utf-8")
        # A window that starts partway through a line has a deliberately
        # discarded first fragment.  Every later row is complete.
        lines = tail.splitlines()
        if window < size:
            lines = lines[1:]
        for index in range(len(lines) - 1, -1, -1):
            parts = lines[index].split()
            if len(parts) != 4:
                continue
            try:
                time_s, steps, count, mesh_id = parts
                count = int(count)
                # A valid final header leaves precisely the active-cell and
                # boundary records through EOF; field rows use a different
                # column count, so this disambiguates it from data.
                if count < 1 or index + 1 + count + boundary_count != len(lines):
                    continue
                node_ids = np.asarray([int(row.split()[0]) for row in lines[index + 1:index + 1 + count]],
                                      dtype=np.int64)
            except (TypeError, ValueError, IndexError):
                continue
            if np.any(node_ids < 0) or len(np.unique(node_ids)) != count:
                raise ValueError("Adaptive frame contains invalid hierarchy node IDs")
            return {"time_s": float(time_s), "steps": int(steps),
                    "mesh_id": int(mesh_id), "node_ids": node_ids}
        if window >= size:
            break
        window = min(size, window * 2)
    raise ValueError("Adaptive frames file contains no complete final frame")


def _read_adaptive_mesh_frames(path: Path, *, progress=None, cancelled=None,
                               progress_range=(0.,100.)) -> list[dict[str, Any]] | None:
    """Stream AMR times and leaf IDs for the optional animated mesh product."""
    with path.open("r", encoding="utf-8") as handle:
        header = [int(value) for value in handle.readline().split()]
        if len(header) != 4 or header[0] != -22:
            return None
        _, boundary_count, _, bed_layers = header
        if boundary_count < 0 or not 1 <= bed_layers <= 33:
            raise ValueError("Adaptive frames header is invalid")
        frames=[];size=max(1,path.stat().st_size);first_progress,last_progress=map(float,progress_range)
        rows_read=0
        while True:
            parts=handle.readline().split()
            if not parts:break
            if len(parts)!=4:raise ValueError("Adaptive frames file ends inside a frame header")
            time_s,steps,count,mesh_id=parts;count=int(count)
            if count<1:raise ValueError("Adaptive frame contains no leaf cells")
            node_ids=np.empty(count,dtype=np.int64)
            for index in range(count):
                row=handle.readline().split()
                if not row:raise ValueError("Adaptive frames file ends inside cell data")
                node_ids[index]=int(row[0])
                rows_read+=1
                if rows_read % 4096 == 0:
                    _raise_if_cancelled(cancelled)
                    _report(progress,first_progress+(last_progress-first_progress)*min(1.,handle.tell()/size))
            for _ in range(boundary_count):
                if not handle.readline():raise ValueError("Adaptive frames file ends inside boundary data")
            if np.any(node_ids<0) or len(np.unique(node_ids))!=count:
                raise ValueError("Adaptive frame contains invalid hierarchy node IDs")
            frames.append({"time_s":float(time_s),"steps":int(steps),"mesh_id":int(mesh_id),"node_ids":node_ids})
            _raise_if_cancelled(cancelled)
            _report(progress,first_progress+(last_progress-first_progress)*min(1.,handle.tell()/size))
    if not frames:raise ValueError("Adaptive frames file contains no result frames")
    return frames


def _local_polygon(vertices, xlower: float, ylower: float, holes=()):
    polygon = ogr.Geometry(ogr.wkbPolygon)
    for points in (vertices,*holes):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in points:
            ring.AddPoint_2D(float(x) - xlower, float(y) - ylower)
        polygon.AddGeometry(ring)
    return polygon


def _pixel_box(i: int, j: int, dx: float, dy: float):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((i * dx, j * dy), ((i + 1) * dx, j * dy),
                 ((i + 1) * dx, (j + 1) * dy), (i * dx, (j + 1) * dy),
                 (i * dx, j * dy)):
        ring.AddPoint_2D(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def _pixel_contributions(geometry: dict[str, Any], subdivisions: int = 1):
    """Map cut-cell values to display pixels with exact overlap weights.

    A standard raster pixel cannot show only part of a solver background cell.
    Subdividing the display grid preserves the native control-volume values
    while resolving domain and obstacle cut fragments in the QGIS mask.
    """
    if isinstance(subdivisions, bool) or int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("Raster display subdivisions must be a positive integer")
    subdivisions = int(subdivisions)
    grid = geometry["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xlower, ylower = float(grid["xlower_m"]), float(grid["ylower_m"])
    dx, dy = float(grid["dx_m"]), float(grid["dy_m"])
    display_dx, display_dy = dx / subdivisions, dy / subdivisions
    contributions: dict[tuple[int, int], list[tuple[int, float]]] = {}
    tolerance = 1e-10 * display_dx * display_dy
    for cell in geometry["cells"]:
        cell_id = int(cell["cell_id"])
        members = cell.get("member_indices") or ()
        indices = {(int(row[0]), int(row[1])) for row in members}
        if not indices:
            indices = {(int(cell["i"]), int(cell["j"]))}
        polygon = _local_polygon(cell["vertices_xy"], xlower, ylower,
                                 cell.get("holes_xy") or ())
        accounted = 0.0
        for i, j in indices:
            if not (0 <= i < nx and 0 <= j < ny):
                raise ValueError(f"Cut cell {cell_id} references a pixel outside the solver grid")
            parent = polygon.Intersection(_pixel_box(i, j, dx, dy))
            parent_area = 0.0 if parent is None or parent.IsEmpty() else float(parent.GetArea())
            if parent_area <= tolerance:
                continue
            if subdivisions == 1:
                contributions.setdefault((i, j), []).append((cell_id, parent_area))
                accounted += parent_area
                continue
            # Full Cartesian members are overwhelmingly common.  Populate all
            # child pixels directly and reserve geometry intersections for the
            # actual domain/obstacle fragments.
            if math.isclose(parent_area, dx * dy, rel_tol=1e-10,
                            abs_tol=max(tolerance, 1e-10 * dx * dy)):
                child_area = display_dx * display_dy
                for jj in range(j * subdivisions, (j + 1) * subdivisions):
                    for ii in range(i * subdivisions, (i + 1) * subdivisions):
                        contributions.setdefault((ii, jj), []).append((cell_id, child_area))
                accounted += dx * dy
                continue
            for jj in range(j * subdivisions, (j + 1) * subdivisions):
                for ii in range(i * subdivisions, (i + 1) * subdivisions):
                    overlap = parent.Intersection(_pixel_box(ii, jj, display_dx, display_dy))
                    area = 0.0 if overlap is None or overlap.IsEmpty() else float(overlap.GetArea())
                    if area > tolerance:
                        accounted += area
                        contributions.setdefault((ii, jj), []).append((cell_id, area))
        expected = float(cell["area_m2"])
        if not math.isclose(accounted, expected, rel_tol=2e-8, abs_tol=max(tolerance, 1e-8)):
            raise ValueError(f"Raster mapping does not preserve cut-cell {cell_id} area")
    return contributions


def _raster_array(values: np.ndarray, contributions, nx: int, ny: int):
    output = np.full((ny, nx), NODATA, dtype=np.float32)
    for (i, j), rows in contributions.items():
        valid_rows=[(float(values[cell_id]),weight) for cell_id,weight in rows
                    if math.isfinite(float(values[cell_id]))]
        if not valid_rows:continue
        area = math.fsum(weight for _, weight in valid_rows)
        value = math.fsum(value * weight for value, weight in valid_rows) / area
        output[ny - 1 - j, i] = np.float32(value)
    return output


def _coverage_alpha(contributions, nx: int, ny: int, pixel_area: float):
    """Return a north-up 0..255 alpha mask from exact fluid area per display pixel."""
    output=np.zeros((ny,nx),dtype=np.float32)
    for (i,j),rows in contributions.items():
        fraction=min(1.,max(0.,math.fsum(weight for _,weight in rows)/pixel_area))
        # Preserve very small but real fragments after eight-bit display conversion.
        output[ny-1-j,i]=max(1.,round(255.*fraction)) if fraction>0 else 0.
    return output


def _safe_raster_name(variable: str):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", variable).strip("._") or "result"
    if stem != variable:
        stem += "_" + hashlib.sha256(variable.encode("utf-8")).hexdigest()[:8]
    return f"temporal_{stem}.tif"


def _frame_fields(frame, q0, rock, roughness_parameter, bulk, fraction_ids, bed_layers,
                  sediment_enabled=True, wet_depth_m=WET_DEPTH_M,
                  roughness_variable="manning_n"):
    q = frame["q"]
    nfrac = len(fraction_ids)
    depth = q[:, 0]
    speed = np.divide(np.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2), depth,
                      out=np.zeros(len(q)), where=depth > wet_depth_m)
    suspended = q[:, 3:3 + nfrac]
    active_inventory = q[:, 3 + nfrac:3 + 2 * nfrac]
    substrate_layers = (np.zeros((len(q),0,nfrac)) if bed_layers == 1 else
        q[:,3+2*nfrac:].reshape(len(q),bed_layers-1,nfrac))
    substrate_inventory=substrate_layers.sum(axis=1)
    bed_inventory = active_inventory + substrate_inventory
    concentrations = np.divide(suspended, depth[:, None], out=np.zeros_like(suspended),
                               where=depth[:, None] > wet_depth_m)
    bed = rock + (bed_inventory / bulk).sum(axis=1)
    stage=np.where(depth>wet_depth_m,depth+bed,np.nan)
    if not sediment_enabled:
        return {"depth_m":depth,"velocity_m_s":speed,"stage_m":stage}
    initial_bed_inventory = q0[:, 3 + nfrac:3 + 2 * nfrac]
    if bed_layers >= 2:
        initial_bed_inventory = initial_bed_inventory + q0[:,3+2*nfrac:].reshape(
            len(q0),bed_layers-1,nfrac).sum(axis=1)
    active_volume = active_inventory / bulk
    active_volume_total = active_volume.sum(axis=1)
    active_fraction = np.divide(active_volume, active_volume_total[:, None],
                                out=np.zeros_like(active_volume),
                                where=active_volume_total[:, None] > 0)
    fields = {
        "depth_m": depth,
        "velocity_m_s": speed,
        "stage_m": stage,
        roughness_variable: roughness_parameter,
        "concentration_kg_m3": concentrations.sum(axis=1),
        "bed_elevation_m": bed,
        "bed_change_m": ((bed_inventory - initial_bed_inventory) / bulk).sum(axis=1),
        "bedrock_elevation_m": rock,
        "erodible_remaining_m": (bed_inventory / bulk).sum(axis=1),
    }
    fields.update({f"concentration_{name}_kg_m3": concentrations[:, index]
                   for index, name in enumerate(fraction_ids)})
    fields.update({f"bed_inventory_{name}_kg_m2": bed_inventory[:, index]
                   for index, name in enumerate(fraction_ids)})
    if frame.get('bedform') is not None:
        fields.update((variable, np.asarray(frame['bedform'][source], dtype=np.float64))
                      for source, variable, _unit in GPU_BEDFORM_RESULT_FIELDS)
    if bed_layers >= 2:
        fields.update({f"active_inventory_{name}_kg_m2": active_inventory[:, index]
                       for index, name in enumerate(fraction_ids)})
        fields.update({f"substrate_inventory_{name}_kg_m2": substrate_inventory[:, index]
                       for index, name in enumerate(fraction_ids)})
        fields.update({f"active_volume_fraction_{name}": active_fraction[:, index]
                       for index, name in enumerate(fraction_ids)})
        for layer in range(bed_layers-1):
            inventory=substrate_layers[:,layer,:]
            volume=inventory/bulk
            total=volume.sum(axis=1)
            composition=np.divide(volume,total[:,None],out=np.zeros_like(volume),where=total[:,None]>0)
            fields.update({f"substrate_layer_{layer+1}_inventory_{name}_kg_m2":inventory[:,index]
                           for index,name in enumerate(fraction_ids)})
            fields.update({f"substrate_layer_{layer+1}_volume_fraction_{name}":composition[:,index]
                           for index,name in enumerate(fraction_ids)})
    return fields


def _variable_units(fraction_ids, bed_layers,sediment_enabled=True, *,
                    roughness_variable="manning_n",
                    roughness_units="s/m^(1/3)"):
    if not sediment_enabled:return {"depth_m":"m","velocity_m_s":"m/s","stage_m":"m"}
    units = {
        "depth_m": "m", "velocity_m_s": "m/s", "stage_m": "m",
        roughness_variable: roughness_units, "concentration_kg_m3": "kg/m3",
        "bed_elevation_m": "m", "bed_change_m": "m",
        "bedrock_elevation_m": "m", "erodible_remaining_m": "m",
    }
    units.update({f"concentration_{name}_kg_m3": "kg/m3" for name in fraction_ids})
    units.update({f"bed_inventory_{name}_kg_m2": "kg/m2" for name in fraction_ids})
    if bed_layers >= 2:
        units.update({f"active_inventory_{name}_kg_m2": "kg/m2" for name in fraction_ids})
        units.update({f"substrate_inventory_{name}_kg_m2": "kg/m2" for name in fraction_ids})
        units.update({f"active_volume_fraction_{name}": "1" for name in fraction_ids})
        for layer in range(1,bed_layers):
            units.update({f"substrate_layer_{layer}_inventory_{name}_kg_m2":"kg/m2" for name in fraction_ids})
            units.update({f"substrate_layer_{layer}_volume_fraction_{name}":"1" for name in fraction_ids})
    return units


def _write_temporal_rasters(result: Path, geometry, frames, fields_by_frame, units,
                            *, progress=None, cancelled=None):
    grid = geometry["grid"]
    source_nx, source_ny = int(grid["nx"]), int(grid["ny"])
    xlower, ylower = float(grid["xlower_m"]), float(grid["ylower_m"])
    source_dx, source_dy = float(grid["dx_m"]), float(grid["dy_m"])
    subdivisions = CUT_CELL_DISPLAY_SUBDIVISIONS
    nx, ny = source_nx * subdivisions, source_ny * subdivisions
    dx, dy = source_dx / subdivisions, source_dy / subdivisions
    snapshot_geometries={int(row['mesh_id']):{**geometry,'cells':row['cells']}
                         for row in geometry.get('mesh_snapshots',[])}
    contributions = (None if snapshot_geometries else
                     _pixel_contributions(geometry, subdivisions))
    snapshot_contributions={}
    options = list(FAST_GEOTIFF_OPTIONS)
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("QGIS GDAL installation does not provide the GeoTIFF driver")
    datasets = {}
    partials = {}
    ranges = {name: [math.inf, -math.inf] for name in units}
    completed_writes=0
    expected_writes=len(frames)*len(units)
    built=False
    try:
        for variable in units:
            _raise_if_cancelled(cancelled)
            final = result / _safe_raster_name(variable)
            # A unique, short staging name avoids colliding with a stale file
            # still held by QGIS, GDAL, antivirus or an indexing service.
            partial = result / f".{uuid4().hex}.partial.tif"
            partials[variable] = (partial, final)
            alpha_band=len(frames)+1
            dataset = driver.Create(os.fspath(partial), nx, ny, alpha_band, gdal.GDT_Float32,
                                    options=options)
            if dataset is None:
                raise RuntimeError(f"Could not create result raster {final.name}")
            dataset.SetGeoTransform((xlower, dx, 0.0, ylower + ny * dy, 0.0, -dy))
            dataset.SetProjection(str(geometry["crs_wkt"]))
            dataset.SetMetadataItem("BRAIDS_TEMPORAL_AXIS", "elapsed simulation seconds")
            dataset.SetMetadataItem("BRAIDS_RASTERIZATION", "area-weighted cut-cell values plus fluid-area alpha on refined display grid")
            dataset.SetMetadataItem("BRAIDS_CUT_CELL_DISPLAY_SUBDIVISIONS", str(subdivisions))
            dataset.SetMetadataItem("BRAIDS_VARIABLE", variable)
            datasets[variable] = dataset
        for frame_index, (frame, fields) in enumerate(zip(frames, fields_by_frame), 1):
            frame_contributions=contributions
            if 'mesh_id' in frame:
                mesh_id=int(frame['mesh_id'])
                if mesh_id not in snapshot_contributions:
                    snapshot_contributions[mesh_id]=_pixel_contributions(snapshot_geometries[mesh_id],subdivisions)
                frame_contributions=snapshot_contributions[mesh_id]
            for variable, values in fields.items():
                _raise_if_cancelled(cancelled)
                array = _raster_array(values, frame_contributions, nx, ny)
                valid = array[array != NODATA]
                if valid.size:
                    ranges[variable][0] = min(ranges[variable][0], float(valid.min()))
                    ranges[variable][1] = max(ranges[variable][1], float(valid.max()))
                band = datasets[variable].GetRasterBand(frame_index)
                band.SetNoDataValue(float(NODATA))
                band.SetDescription(f"{variable} at t={frame['time_s']:.17g} s")
                band.SetMetadataItem("BRAIDS_SIMULATION_TIME_SECONDS", f"{frame['time_s']:.17g}")
                band.SetMetadataItem("BRAIDS_NATIVE_STEPS", str(frame["steps"]))
                if band.WriteArray(array) != gdal.CE_None:
                    raise RuntimeError(f"Could not write band {frame_index} of {variable}")
                band.FlushCache()
                band = None
                completed_writes+=1
                if expected_writes:
                    _report(progress,100.*completed_writes/expected_writes)
        alpha_contributions=contributions
        if alpha_contributions is None:
            first_mesh_id=int(frames[0]['mesh_id'])
            if first_mesh_id not in snapshot_contributions:
                snapshot_contributions[first_mesh_id]=_pixel_contributions(
                    snapshot_geometries[first_mesh_id],subdivisions)
            alpha_contributions=snapshot_contributions[first_mesh_id]
        alpha=_coverage_alpha(alpha_contributions,nx,ny,dx*dy)
        for dataset in datasets.values():
            band=dataset.GetRasterBand(len(frames)+1)
            band.SetDescription("Exact fluid-area coverage alpha (0-255)")
            band.SetMetadataItem("BRAIDS_ALPHA_ROLE","cut-cell fluid-area coverage")
            if band.WriteArray(alpha)!=gdal.CE_None:
                raise RuntimeError("Could not write cut-cell alpha mask")
            band.FlushCache();band=None
        built=True
    finally:
        # ``FlushCache`` writes GDAL's pending blocks but does not close a
        # dataset.  On Windows an open dataset prevents ``os.replace`` below
        # from atomically promoting its partial TIFF.  Pop every reference and
        # clear the loop variable explicitly so the last-created raster (often
        # ``stage_m`` in a hydraulic-only run) is closed before the rename.
        while datasets:
            _, dataset = datasets.popitem()
            try:
                dataset.FlushCache()
            finally:
                dataset = None
        if not built:
            for partial,_ in partials.values():
                try:partial.unlink(missing_ok=True)
                except OSError:pass
    products = {}
    for variable, (partial, final) in partials.items():
        _replace_with_windows_retry(partial, final)
        limits = ranges[variable]
        if not all(math.isfinite(value) for value in limits):
            limits = [0.0, 0.0]
        products[variable] = {
            "path": final.name,
            "unit": units[variable],
            "range": limits,
            "band_count": len(frames),
            "alpha_band": len(frames)+1,
            "bands": [{"band": index + 1, "time_s": frame["time_s"], "steps": frame["steps"]}
                      for index, frame in enumerate(frames)],
            "sha256": _digest(final),
        }
    return products, {"nx": nx, "ny": ny, "xlower_m": xlower, "ylower_m": ylower,
                      "dx_m": dx, "dy_m": dy, "nodata": float(NODATA),
                      "source_nx": source_nx, "source_ny": source_ny,
                      "source_dx_m": source_dx, "source_dy_m": source_dy,
                      "display_subdivisions": subdivisions,
                      "method": "area-weighted cut-cell values with exact fluid-area alpha on refined display grid",
                      "populated_pixel_count": len(alpha_contributions)}


def _live_preview_rows(root: Path) -> list[tuple[int, float, int]]:
    """Return complete native preview frames, accepting a trailing partial line."""
    index = root / LIVE_PREVIEW_DIRECTORY / "frames.csv"
    rows = {}
    if not index.is_file():
        return []
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            frame, time_s, steps = line.split()
            rows[int(frame)] = (int(frame), float(time_s), int(steps))
        except (TypeError, ValueError):
            continue
    return [rows[key] for key in sorted(rows)]


def _write_live_vrt(path: Path, variable: str, rows, grid, crs_wkt: str) -> None:
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xlower, ylower = float(grid["xlower_m"]), float(grid["ylower_m"])
    dx, dy = float(grid["dx_m"]), float(grid["dy_m"])
    bands = []
    for band, (frame, time_s, steps) in enumerate(rows, 1):
        source = f"../{LIVE_PREVIEW_DIRECTORY}/{variable}_{frame:08d}.f32"
        bands.append(
            f'  <VRTRasterBand dataType="Float32" band="{band}" subClass="VRTRawRasterBand">\n'
            f'    <Description>{html.escape(variable)} at t={time_s:.17g} s</Description>\n'
            '    <NoDataValue>-3.4028234e+38</NoDataValue>\n'
            f'    <Metadata><MDI key="BRAIDS_SIMULATION_TIME_SECONDS">{time_s:.17g}</MDI>'
            f'<MDI key="BRAIDS_NATIVE_STEPS">{steps}</MDI></Metadata>\n'
            f'    <SourceFilename relativeToVRT="1">{source}</SourceFilename>\n'
            '    <ImageOffset>0</ImageOffset><PixelOffset>4</PixelOffset>'
            f'<LineOffset>{4 * nx}</LineOffset><ByteOrder>LSB</ByteOrder>\n'
            '  </VRTRasterBand>')
    path.write_text(
        f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">\n'
        f'  <SRS>{html.escape(str(crs_wkt))}</SRS>\n'
        f'  <GeoTransform>{xlower:.17g}, {dx:.17g}, 0, {ylower + ny * dy:.17g}, 0, {-dy:.17g}</GeoTransform>\n'
        + "\n".join(bands) + "\n</VRTDataset>\n", encoding="utf-8")


def _live_preview_range(root: Path, variable: str, rows, nx: int, ny: int,
                        existing: dict[str, Any] | None = None) -> list[float]:
    """Update extrema from only new raw frames, never rerasterizing output."""
    limits = [math.inf, -math.inf];start=0
    if existing is not None:
        count=int(existing.get("band_count",0));previous=existing.get("range",())
        if 0<=count<=len(rows) and len(previous)==2 and all(math.isfinite(float(value)) for value in previous):
            limits=[float(previous[0]),float(previous[1])];start=count
    for frame, _, _ in rows[start:]:
        path = root / LIVE_PREVIEW_DIRECTORY / f"{variable}_{frame:08d}.f32"
        if not path.is_file() or path.stat().st_size != 4 * nx * ny:
            continue
        values = np.memmap(path, dtype="<f4", mode="r", shape=(ny, nx))
        valid = values[np.isfinite(values) & (values > NODATA / 2)]
        if valid.size:
            limits[0] = min(limits[0], float(valid.min()))
            limits[1] = max(limits[1], float(valid.max()))
        del values
    return [0.0, 0.0] if not all(math.isfinite(value) for value in limits) else limits


def materialize_live_preview_results(run_root: str | Path) -> dict[str, Any] | None:
    """Create instant QGIS VRT time stacks from native-grid preview frames."""
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    rows = _live_preview_rows(root)
    if not rows:
        return None
    geometry_path = geometry_artifact(root, "geometry")
    geometry = load_geometry(geometry_path, cell_ids=[], include_faces=False)
    grid = geometry["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    available = {
        variable: unit for variable, unit in LIVE_PREVIEW_VARIABLES.items()
        if all((root / LIVE_PREVIEW_DIRECTORY / f"{variable}_{frame:08d}.f32").is_file()
               for frame, _, _ in rows)
    }
    if not available:
        return None
    result = root / "qgis_results";result.mkdir(exist_ok=True)
    previous={}
    try:
        previous=json.loads((result/"results.json").read_text(encoding="utf-8"))
    except (OSError,ValueError,json.JSONDecodeError):pass
    previous_rasters=previous.get("rasters",{}) if isinstance(previous,dict) else {}
    rasters = {}
    for variable, unit in available.items():
        path = result / _safe_raster_name(variable).replace(".tif", ".vrt")
        _write_live_vrt(path, variable, rows, grid, geometry["crs_wkt"])
        rasters[variable] = {"path": path.name, "unit": unit,
                             "range": _live_preview_range(root, variable, rows, nx, ny,
                                                          previous_rasters.get(variable)),
                             "band_count": len(rows), "alpha_band": 0,
                             "bands": [{"band": index + 1, "time_s": time_s, "steps": steps}
                                       for index, (_, time_s, steps) in enumerate(rows)]}
    manifest = {
        "format": "braids-qgis-live-vrt-v1", "source_run": str(root), "crs_wkt": geometry["crs_wkt"],
        "variables": available, "rasters": rasters, "mesh": {},
        "raster_grid": {"nx": nx, "ny": ny, "xlower_m": float(grid["xlower_m"]),
                        "ylower_m": float(grid["ylower_m"]), "dx_m": float(grid["dx_m"]),
                        "dy_m": float(grid["dy_m"]), "nodata": float(NODATA),
                        "method": "native-grid preview; raw Float32 bands referenced by QGIS VRT"},
        "raster_storage": {"driver": "VRT/RawRasterBand", "layout": "native-grid temporal bands",
                           "compression": "none", "profile": "live-qgis-vrt-v1"},
        "frames": [{"frame_index": index, "band": index + 1, "time_s": time_s, "steps": steps}
                   for index, (_, time_s, steps) in enumerate(rows)],
        "source_sha256": {"live_index": _digest(root / LIVE_PREVIEW_DIRECTORY / "frames.csv"),
                          "geometry": _digest(geometry_path)},
    }
    _write_text_atomic(result / "results.json",
                       json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _gpu_live_vrt_payload(raw_name: str, variable: str,
                          grid: dict, crs_wkt: str) -> str:
    """Describe one committed Float32 GPU preview without copying its pixels."""
    nx, ny = int(grid["nx"]), int(grid["ny"])
    xlower, ylower = float(grid["xlower_m"]), float(grid["ylower_m"])
    dx, dy = float(grid["dx_m"]), float(grid["dy_m"])
    return (
        f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">\n'
        f'  <SRS>{html.escape(crs_wkt)}</SRS>\n'
        f'  <GeoTransform>{xlower:.17g}, {dx:.17g}, 0, {ylower + ny * dy:.17g}, 0, {-dy:.17g}</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">\n'
        f'    <Description>{html.escape(variable)}</Description>\n'
        f'    <NoDataValue>{float(NODATA):.17g}</NoDataValue>\n'
        f'    <SourceFilename relativeToVRT="1">{html.escape(raw_name)}</SourceFilename>\n'
        f'    <ImageOffset>0</ImageOffset><PixelOffset>4</PixelOffset>'
        f'<LineOffset>{4 * nx}</LineOffset><ByteOrder>LSB</ByteOrder>\n'
        '  </VRTRasterBand>\n</VRTDataset>\n')


def _write_gpu_live_vrt(path: Path, payload: str) -> None:
    _write_text_atomic(path, payload)


def materialize_gpu_live_preview(run_root: str | Path, variable: str, *,
                                 prepared_root: str | Path | None = None,
                                 after_index: int = -1,
                                 cache: dict[str, Any] | None = None,
                                 cancelled=None) -> dict[str, Any] | None:
    """Prepare just the latest authenticated GPU frame for a QGIS live layer.

    This is intended for a background QGIS task. It skips uncommitted frames,
    never decodes earlier frame payloads, and rasterizes only the selected
    selected variable at the native grid resolution. The cut-cell area map
    and initial bed are reusable across outputs of the same fixed-grid run.
    """
    if variable not in GPU_LIVE_PREVIEW_VARIABLES:
        raise ValueError("GPU live preview variable is unsupported")
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    if (root / "run.json").is_file():
        kind = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if ((kind.get("schema"), kind.get("input_kind")) in
                {(2, "amr-prepared"), (3, "amr-on-demand"),
                 (3, "amr-on-demand-general")}):
            return materialize_gpu_amr_live_preview(root, variable,
                prepared_root=prepared_root, after_index=after_index,
                cache=cache, cancelled=cancelled)
    from .gpu_frame_io import read_gpu_live_frame
    from .gpu_prepared_io import read_gpu_prepared

    _raise_if_cancelled(cancelled)
    verified = read_gpu_live_frame(root, prepared_root=prepared_root,
                                   after_index=after_index)
    if verified is None:
        return None
    _raise_if_cancelled(cancelled)
    frame = verified["frame"]
    prepared = verified["prepared_root"]
    metadata = verified["geometry"]
    identity = verified["manifest"]["input_preparation_sha256"]
    reuse = (isinstance(cache, dict) and cache.get("run_root") == str(root)
             and cache.get("input_identity") == identity
             and cache.get("geometry_sha256") == metadata["sha256"]
             and cache.get("cell_count") == metadata["cell_count"])
    if reuse:
        mapping = cache
    else:
        geometry = load_geometry(geometry_artifact(prepared, "geometry"),
                                 include_faces=False)
        cells = geometry["cells"]
        if (len(cells) != metadata["cell_count"] or
                any(int(cell["cell_id"]) != index for index, cell in enumerate(cells)) or
                not np.allclose([cell["area_m2"] for cell in cells],
                                metadata["area_m2"], rtol=1e-10, atol=1e-12)):
            raise ValueError("GPU preview cell order/area differs from authenticated geometry")
        source = read_gpu_prepared(prepared, receipt_sha256=identity)
        if int(source["nc"]) != len(cells):
            raise ValueError("GPU preview prepared cell count differs")
        nfrac = int(source["nfrac"])
        nlayer = int(source["nbed_layers"])
        initial_inventory = np.asarray(source["state"][3 + nfrac:
            3 + (1 + nlayer) * nfrac], dtype=np.float64).reshape(
                nlayer, nfrac, len(cells)).sum(axis=0).T
        initial_bed = np.asarray(source["rock"], dtype=np.float64) + (
            initial_inventory / np.asarray(source["bulk_fraction"],
                                           dtype=np.float64)[None, :]).sum(axis=1)
        if not np.isfinite(initial_bed).all():
            raise ValueError("GPU preview prepared initial bed is invalid")
        mapping = {"run_root": str(root), "input_identity": identity,
                   "geometry_sha256": metadata["sha256"],
                   "cell_count": len(cells), "grid": geometry["grid"],
                   "crs_wkt": geometry["crs_wkt"],
                   "contributions": _pixel_contributions(geometry, 1),
                   "wet_depth_m": float(source["dry_tolerance"]),
                   "initial_bed_elevation_m": initial_bed}
    grid = mapping["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    q = frame["q"]
    depth = q[:, 0]
    wet_depth = mapping["wet_depth_m"]
    if variable == "depth_m":
        values = depth
    elif variable == "velocity_m_s":
        values = np.divide(np.hypot(q[:, 1], q[:, 2]), depth,
                           out=np.zeros(len(depth), dtype=np.float64),
                           where=depth > wet_depth)
    elif variable == "stage_m":
        values = np.where(depth > wet_depth,
                          depth + frame["bed_elevation"], np.nan)
    elif variable == "bed_elevation_m":
        values = frame["bed_elevation"]
    elif variable == "bed_change_m":
        values = frame["bed_elevation"] - mapping["initial_bed_elevation_m"]
    else:
        nfrac = int(frame["nfrac"])
        values = np.divide(q[:, 3:3 + nfrac].sum(axis=1), depth,
                           out=np.zeros(len(depth), dtype=np.float64),
                           where=depth > wet_depth)
    pixels = _raster_array(values, mapping["contributions"], nx, ny)
    _raise_if_cancelled(cancelled)
    folder = root / "gpu_live_preview"
    folder.mkdir(exist_ok=True)
    digest = verified["manifest"]["frames"][verified["frame_index"]]["sha256"]
    stem = f'{variable}_{verified["frame_index"]:06d}_{digest[:12]}'
    raw = folder / f"{stem}.f32"
    vrt = folder / f"{stem}.vrt"
    raw_bytes = np.asarray(pixels, dtype="<f4").tobytes()
    if not raw.is_file():
        temporary = folder / f'.{raw.name}.{uuid4().hex}.partial'
        try:
            with temporary.open("xb") as stream:
                stream.write(raw_bytes)
            _replace_with_windows_retry(temporary, raw)
        finally:
            temporary.unlink(missing_ok=True)
    if _digest(raw) != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("GPU preview raw raster differs from authenticated frame")
    vrt_payload = _gpu_live_vrt_payload(raw.name, variable, grid,
                                        mapping["crs_wkt"])
    if not vrt.is_file():
        _write_gpu_live_vrt(vrt, vrt_payload)
    elif vrt.read_text(encoding="utf-8") != vrt_payload:
        raise ValueError("GPU preview VRT differs from authenticated frame mapping")
    valid = pixels[np.isfinite(pixels) & (pixels > NODATA / 2)]
    limits = [float(valid.min()), float(valid.max())] if valid.size else [0., 0.]
    return {"path": str(vrt), "range": limits,
            "unit": GPU_LIVE_PREVIEW_VARIABLES[variable], "band_count": 1,
            "variable": variable, "frame_index": verified["frame_index"],
            "time_s": frame["time_s"], "steps": frame["steps"],
            "crs_wkt": mapping["crs_wkt"], "cache": mapping,
            "source_frame_sha256": digest}


def materialize_gpu_amr_live_preview(run_root: str | Path, variable: str, *,
                                     prepared_root: str | Path | None = None,
                                     after_index: int = -1,
                                     cache: dict[str, Any] | None = None,
                                     cancelled=None) -> dict[str, Any] | None:
    """Rasterize only the latest committed AMR frame on its own topology."""
    if variable not in GPU_LIVE_PREVIEW_VARIABLES:
        raise ValueError("GPU live preview variable is unsupported")
    from .gpu_amr_frame_io import read_gpu_amr_live_frame

    root = filesystem_path(Path(run_root).expanduser()).resolve()
    _raise_if_cancelled(cancelled)
    verified = read_gpu_amr_live_frame(root, prepared_root=prepared_root,
                                       after_index=after_index)
    if verified is None:
        return None
    _raise_if_cancelled(cancelled)
    frame = verified["frame"]
    topology = verified["topology"]
    identity = verified["manifest"]["input_preparation_sha256"]
    operator = topology["operator_sha256"]
    reuse = (isinstance(cache, dict) and cache.get("run_root") == str(root)
             and cache.get("input_identity") == identity)
    mappings = cache.get("topologies", {}) if reuse else {}
    mapping = mappings.get(operator)
    if mapping is None:
        geometry = load_geometry(topology["folder"] / "geometry.npz",
                                 include_faces=False)
        cells = geometry["cells"]
        auth = topology["geometry"]
        if (len(cells) != len(frame["node_ids"]) or
                geometry["crs_wkt"] != auth["declared_crs_wkt"] or
                any(int(cell["cell_id"]) != index for index, cell in enumerate(cells)) or
                not np.allclose([cell["area_m2"] for cell in cells],
                                auth["area_m2"], rtol=1e-11, atol=1e-11) or
                not np.allclose([cell["centroid_xy"] for cell in cells],
                                auth["centroid_xy"], rtol=1e-11, atol=1e-11)):
            raise ValueError("GPU AMR preview cell geometry differs from authenticated topology")
        source = topology["initial_data"]
        nfrac = int(source["nfrac"])
        nlayer = int(source["nbed_layers"])
        masses = np.asarray(source["state"], dtype=np.float64)[
            3+nfrac:3+nfrac+nlayer*nfrac].reshape(nlayer, nfrac, len(cells))
        initial_bed = np.asarray(source["rock"], dtype=np.float64) + (
            masses.sum(axis=0).T /
            np.asarray(source["bulk_fraction"], dtype=np.float64)[None, :]).sum(axis=1)
        mapping = {"grid": geometry["grid"], "crs_wkt": geometry["crs_wkt"],
                   "contributions": _pixel_contributions(geometry, 1),
                   "wet_depth_m": float(source["dry_tolerance"]),
                   "initial_bed_elevation_m": initial_bed}
        mappings[operator] = mapping
    cache = {"run_root": str(root), "input_identity": identity,
             "topologies": mappings}
    grid = mapping["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    q = frame["q"]
    depth = q[:, 0]
    wet_depth = mapping["wet_depth_m"]
    if variable == "depth_m":
        values = depth
    elif variable == "velocity_m_s":
        values = np.divide(np.hypot(q[:, 1], q[:, 2]), depth,
                           out=np.zeros(len(depth), dtype=np.float64),
                           where=depth > wet_depth)
    elif variable == "stage_m":
        values = np.where(depth > wet_depth,
                          depth + frame["bed_elevation"], np.nan)
    elif variable == "bed_elevation_m":
        values = frame["bed_elevation"]
    elif variable == "bed_change_m":
        values = frame["bed_elevation"] - mapping["initial_bed_elevation_m"]
    else:
        nfrac = int(frame["nfrac"])
        values = np.divide(q[:, 3:3+nfrac].sum(axis=1), depth,
                           out=np.zeros(len(depth), dtype=np.float64),
                           where=depth > wet_depth)
    pixels = _raster_array(values, mapping["contributions"], nx, ny)
    _raise_if_cancelled(cancelled)
    folder = root / "gpu_live_preview"
    folder.mkdir(exist_ok=True)
    digest = verified["manifest"]["frames"][verified["frame_index"]]["sha256"]
    stem = f'{variable}_{verified["frame_index"]:06d}_{digest[:12]}'
    raw = folder / f"{stem}.f32"
    vrt = folder / f"{stem}.vrt"
    raw_bytes = np.asarray(pixels, dtype="<f4").tobytes()
    if not raw.is_file():
        temporary = folder / f'.{raw.name}.{uuid4().hex}.partial'
        try:
            with temporary.open("xb") as stream:
                stream.write(raw_bytes)
            _replace_with_windows_retry(temporary, raw)
        finally:
            temporary.unlink(missing_ok=True)
    if _digest(raw) != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("GPU AMR preview raster differs from authenticated frame")
    vrt_payload = _gpu_live_vrt_payload(raw.name, variable, grid,
                                        mapping["crs_wkt"])
    if not vrt.is_file():
        _write_gpu_live_vrt(vrt, vrt_payload)
    elif vrt.read_text(encoding="utf-8") != vrt_payload:
        raise ValueError("GPU AMR preview VRT differs from authenticated frame")
    valid = pixels[np.isfinite(pixels) & (pixels > NODATA / 2)]
    limits = [float(valid.min()), float(valid.max())] if valid.size else [0., 0.]
    return {"path": str(vrt), "range": limits,
            "unit": GPU_LIVE_PREVIEW_VARIABLES[variable], "band_count": 1,
            "variable": variable, "frame_index": verified["frame_index"],
            "time_s": frame["time_s"], "steps": frame["steps"],
            "crs_wkt": mapping["crs_wkt"], "cache": cache,
            "topology_name": frame["topology_name"],
            "source_frame_sha256": digest}


def _dynamic_mesh_sources(frames_path: Path, geometry_path: Path,
                          hierarchy_path: Path) -> list[dict[str, int | str]]:
    return [_file_stamp(path) for path in (frames_path,geometry_path,hierarchy_path)]


def _cached_dynamic_mesh(result: Path, sources) -> dict[str, Any] | None:
    cache_path=result/DYNAMIC_MESH_CACHE
    try:
        cache=json.loads(cache_path.read_text(encoding="utf-8"))
        product=cache["product"]
        mesh_path=result/str(product["path"])
        if cache.get("sources")!=sources or not mesh_path.is_file() or mesh_path.stat().st_size<=0:
            return None
        if not bool(product.get("animated",False)):
            return None
        return product
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):
        return None


def _save_dynamic_mesh_cache(result: Path, sources, product: dict[str, Any]) -> None:
    cache_path=result/DYNAMIC_MESH_CACHE
    _write_text_atomic(cache_path, json.dumps({"format":"braids-adaptive-mesh-cache-v1",
        "sources":sources,"product":product},indent=2,sort_keys=True)+"\n")


def _preserve_manifest_dynamic_mesh(run_root: Path) -> None:
    """Migrate an already-built animated mesh into the persistent cache."""
    result=run_root/"qgis_results";manifest_path=result/"results.json"
    try:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        product=manifest["mesh"]
        if not bool(product.get("animated",False)) or not (result/str(product["path"])).is_file():
            return
        frames_path=run_root/"frames.txt";geometry_path=geometry_artifact(run_root,"geometry")
        hierarchy_path=geometry_artifact(run_root,"amr_hierarchy")
        _save_dynamic_mesh_cache(result,
            _dynamic_mesh_sources(frames_path,geometry_path,hierarchy_path),product)
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):
        return


def _materialize_completed_live_mesh(run_root: Path, manifest: dict[str, Any], *,
                                    animated: bool = False, progress=None,
                                    cancelled=None) -> dict[str, Any]:
    """Build final or explicitly requested animated AMR mesh for VRT results."""
    geometry_path=geometry_artifact(run_root,"geometry")
    frames_path=run_root/"frames.txt"
    if animated:
        _raise_if_cancelled(cancelled);_report(progress,1.)
        adaptive_frames=False
        if frames_path.is_file():
            with frames_path.open("r",encoding="utf-8") as handle:
                header=handle.readline().split()
            adaptive_frames=bool(header and header[0]=="-22")
        hierarchy_path=geometry_artifact(run_root,"amr_hierarchy") if adaptive_frames else None
        sources=(None if hierarchy_path is None else
                 _dynamic_mesh_sources(frames_path,geometry_path,hierarchy_path))
        result=run_root/"qgis_results"
        cached=None if sources is None else _cached_dynamic_mesh(result,sources)
        if cached is not None:
            _report(progress,100.);return cached
        frames=(_read_adaptive_mesh_frames(frames_path,progress=progress,cancelled=cancelled,
                                           progress_range=(3.,35.))
                if adaptive_frames else None)
        if frames is not None:
            _raise_if_cancelled(cancelled);_report(progress,38.)
            used=np.unique(np.concatenate([frame["node_ids"] for frame in frames]))
            selected=load_geometry(hierarchy_path,cell_ids=used,include_faces=False)["cells"]
            _raise_if_cancelled(cancelled);_report(progress,50.)
            cells_by_node={int(node):cell for node,cell in zip(used,selected)}
            snapshots={}
            for frame_index,frame in enumerate(frames):
                _raise_if_cancelled(cancelled)
                if frame["mesh_id"] not in snapshots:
                    snapshots[frame["mesh_id"]]={"mesh_id":frame["mesh_id"],"time_s":frame["time_s"],
                        "cells":[{**cells_by_node[int(node)],"cell_id":index}
                                 for index,node in enumerate(frame["node_ids"])]}
                _report(progress,50.+15.*(frame_index+1)/len(frames))
            geometry=load_geometry(geometry_path,cell_ids=[],include_faces=False)
            geometry={**geometry,"mesh_snapshots":list(snapshots.values()),
                      "frame_mesh_ids":[frame["mesh_id"] for frame in frames]}
            topology=hashlib.sha256((_digest(geometry_path)+_digest(hierarchy_path)).encode("ascii"))
            for frame_index,frame in enumerate(frames):
                _raise_if_cancelled(cancelled)
                topology.update(np.asarray([frame["mesh_id"]],dtype="<i8").tobytes())
                topology.update(np.asarray(frame["node_ids"],dtype="<i8").tobytes())
                _report(progress,65.+5.*(frame_index+1)/len(frames))
            product=write_result_mesh(result/f"computational_mesh_dynamic_{topology.hexdigest()[:12]}.gpkg",
                                      geometry,[float(frame["time_s"]) for frame in manifest["frames"]],
                                      progress=progress,cancelled=cancelled,progress_range=(70.,99.))
            product["animated"]=True
            _save_dynamic_mesh_cache(result,sources,product);_report(progress,100.)
            return product
    final=_read_final_adaptive_mesh(frames_path) if frames_path.is_file() else None
    if final is not None:
        hierarchy_path=geometry_artifact(run_root,"amr_hierarchy")
        # A full GPKG copy for every AMR output can contain tens of millions of
        # polygons. The completed-results overlay deliberately represents the
        # final adaptive mesh: it is useful, bounded, and opens promptly.
        used=final["node_ids"]
        selected=load_geometry(hierarchy_path,cell_ids=used,include_faces=False)["cells"]
        mesh_id=int(final["mesh_id"]);snapshots=[{"mesh_id":mesh_id,"time_s":final["time_s"],
            "cells":[{**cell,"cell_id":row} for row,cell in enumerate(selected)]}]
        geometry=load_geometry(geometry_path,cell_ids=[],include_faces=False)
        times=[float(frame["time_s"]) for frame in manifest["frames"]]
        geometry={**geometry,"mesh_snapshots":snapshots,
                  "frame_mesh_ids":[mesh_id]*len(times)}
        # IDs plus hierarchy version fully identify the final overlay; do not
        # hash the entire historical ASCII field archive on every load.
        topology=hashlib.sha256((_digest(geometry_path)+_digest(hierarchy_path)).encode("ascii"))
        topology.update(np.asarray([mesh_id],dtype="<i8").tobytes())
        topology.update(np.asarray(used,dtype="<i8").tobytes())
        mesh_hash=topology.hexdigest()
    else:
        geometry=load_geometry(geometry_path,include_faces=False)
        mesh_hash=_digest(geometry_path);times=[frame["time_s"] for frame in manifest["frames"]]
    result=run_root/"qgis_results"
    product=write_result_mesh(result/f"computational_mesh_{mesh_hash[:12]}.gpkg",geometry,times)
    product["animated"]=False
    return product


def _result_dataset(manifest: dict[str, Any], variable: str):
    product=manifest.get("rasters",{}).get(variable)
    if not product:raise ValueError(f"Result variable is unavailable: {variable}")
    path=Path(manifest["source_run"])/"qgis_results"/str(product["path"])
    dataset=gdal.Open(str(path))
    if dataset is None:raise RuntimeError(f"GDAL could not open {path}")
    return dataset,product


def _grid_samples(dataset,xy: np.ndarray,band: int) -> np.ndarray:
    """Nearest native-grid values, with outside/dry cells reported as NaN."""
    gt=dataset.GetGeoTransform();array=dataset.GetRasterBand(int(band)).ReadAsArray()
    if array is None:raise RuntimeError("GDAL could not read result band")
    col=np.floor((xy[:,0]-gt[0])/gt[1]).astype(int)
    row=np.floor((xy[:,1]-gt[3])/gt[5]).astype(int)
    values=np.full(len(xy),np.nan,dtype=float)
    valid=(col>=0)&(col<dataset.RasterXSize)&(row>=0)&(row<dataset.RasterYSize)
    values[valid]=np.asarray(array,dtype=float)[row[valid],col[valid]]
    values[values<=float(NODATA)/2]=np.nan
    return values


def sample_result_probe(manifest: dict[str, Any], x: float, y: float,
                        variables=("depth_m","velocity_m_s","stage_m")) -> dict[str, Any]:
    """Read a point's saved-time series from QGIS result bands."""
    xy=np.array([[float(x),float(y)]])
    times=np.array([float(row["time_s"]) for row in manifest["frames"]])
    result={"time_s":times,"x":float(x),"y":float(y),"series":{}}
    for variable in variables:
        if variable not in manifest.get("rasters",{}):continue
        dataset,product=_result_dataset(manifest,variable)
        try:
            result["series"][variable]={"unit":product["unit"],"values":np.array(
                [_grid_samples(dataset,xy,band)[0] for band in range(1,len(times)+1)])}
        finally:dataset=None
    if not result["series"]:raise ValueError("No probe-compatible result variables are available")
    return result


def integrate_result_transect(manifest: dict[str, Any], vertices_xy) -> dict[str, Any]:
    """Integrate signed discharge across a polyline at every saved time.

    Positive flow is toward the left side of the digitized line.  The two
    native unit-discharge components retain direction, unlike speed alone.
    """
    points=np.asarray(vertices_xy,dtype=float)
    if points.ndim!=2 or points.shape[0]<2 or points.shape[1]!=2:raise ValueError("A transect needs at least two points")
    hu,product=_result_dataset(manifest,"discharge_x_m2_s");hv,_=_result_dataset(manifest,"discharge_y_m2_s")
    try:
        gt=hu.GetGeoTransform();spacing=.5*min(abs(gt[1]),abs(gt[5]))
        samples=[];normals=[];weights=[]
        for start,end in zip(points[:-1],points[1:]):
            delta=end-start;length=float(np.hypot(*delta))
            if length<=0:continue
            count=max(1,math.ceil(length/spacing));distance=length/count
            fraction=(np.arange(count)+.5)/count
            samples.append(start+fraction[:,None]*delta)
            normals.append(np.tile(np.array([delta[1],-delta[0]])/length,(count,1)))
            weights.append(np.full(count,distance))
        if not samples:raise ValueError("Transect segments must have positive length")
        xy=np.vstack(samples);normal=np.vstack(normals);lengths=np.concatenate(weights)
        times=np.array([float(row["time_s"]) for row in manifest["frames"]]);flow=[]
        for band in range(1,len(times)+1):
            qx=_grid_samples(hu,xy,band);qy=_grid_samples(hv,xy,band)
            normal_flow=np.nan_to_num(qx*normal[:,0]+qy*normal[:,1],nan=0.)
            flow.append(float(normal_flow@lengths))
        return {"time_s":times,"flow_m3_s":np.asarray(flow),"unit":"m3/s",
                "sign_convention":"positive toward the left side of the digitized transect"}
    finally:hu=None;hv=None


def materialize_gpu_results(run_root: str | Path, *, animated_mesh: bool = False,
                            progress=None, cancelled=None,
                            prepared_root: str | Path | None = None) -> dict[str, Any]:
    """Turn a completed, authenticated fixed-grid GPU run into QGIS products.

    The GPU worker owns the numerical output.  Its preparation folder supplies
    the original georeferenced cut-cell polygons and material properties; face
    midpoints in the worker output alone are not a substitute for that input.
    """
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    worker_kind = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if ((worker_kind.get("schema"), worker_kind.get("input_kind")) in
            {(2, "amr-prepared"), (3, "amr-on-demand"),
             (3, "amr-on-demand-general")}):
        return materialize_gpu_amr_results(root, animated_mesh=animated_mesh,
            progress=progress, cancelled=cancelled, prepared_root=prepared_root)
    if animated_mesh:
        raise ValueError("Adaptive GPU mesh results are not yet supported")
    from .gpu_frame_io import read_gpu_run
    from .gpu_prepared_io import read_gpu_prepared
    from .sediment_product import validate_gpu_result_product

    _raise_if_cancelled(cancelled);_report(progress, 0.)
    prepared_for_product = (filesystem_path(Path(prepared_root).expanduser()).resolve()
        if prepared_root is not None else filesystem_path(
            Path(worker_kind.get("input_folder", root.parent / "gpu_prepared"))
            .expanduser()).resolve())
    validate_gpu_result_product(root, prepared_for_product)
    verified = read_gpu_run(root, prepared_root=prepared_root)
    worker = verified["manifest"]
    if worker.get("backend") != "gpu" or worker.get("status") != "completed":
        raise ValueError("Only a completed GPU BRAIDS run can be loaded")
    prepared = (filesystem_path(Path(prepared_root).expanduser()).resolve()
                if prepared_root is not None else
                filesystem_path(Path(worker["input_folder"]).expanduser()).resolve())
    geometry_path = geometry_artifact(prepared, "geometry")
    geometry = load_geometry(geometry_path, include_faces=False)
    source = read_gpu_prepared(prepared,
        receipt_sha256=worker["input_preparation_sha256"])
    cells = geometry["cells"]
    count = len(cells)
    if count != int(worker["cells"]) or count != int(source["nc"]):
        raise ValueError("GPU frame and prepared cut-cell counts disagree")
    with np.load(root / "geometry.npz", allow_pickle=False) as gpu_geometry:
        xy = np.asarray(gpu_geometry["xy"], dtype=np.float64).T
        area = np.asarray(gpu_geometry["area"], dtype=np.float64)
    cell_xy = np.asarray([cell["centroid_xy"] for cell in cells], dtype=np.float64)
    cell_area = np.asarray([cell["area_m2"] for cell in cells], dtype=np.float64)
    if (xy.shape != cell_xy.shape or area.shape != cell_area.shape or
            not np.allclose(xy, cell_xy, rtol=0., atol=1e-9) or
            not np.allclose(area, cell_area, rtol=1e-10, atol=1e-12)):
        raise ValueError("GPU cell order or area differs from georeferenced preparation")
    if any(int(cell["cell_id"]) != index for index, cell in enumerate(cells)):
        raise ValueError("Prepared cut-cell IDs are not in GPU state order")

    case_path = prepared / "case.data.json"
    origin = verified["prepared_receipt"].get("origin", {})
    declared_files = origin.get("files", {}) if isinstance(origin, dict) else {}
    expected_case_sha = declared_files.get(case_path.name)
    if not isinstance(expected_case_sha, str) or _digest(case_path) != expected_case_sha:
        raise ValueError("GPU prepared case metadata differs from authenticated input")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    prepared_receipt=verified["prepared_receipt"]
    q3c_result_identity=_gpu_q3c_result_identity(
        worker,source,
        secondary_receipt=prepared_receipt.get(
            "secondary_foundation_identity"),
        prognostic_receipt=prepared_receipt.get(
            "prognostic_secondary_identity"),
        prepared_receipt=prepared_receipt)
    roughness_contract=_roughness_result_contract(case=case,source=source)
    nfrac = int(source["nfrac"])
    bed_layers = int(source["nbed_layers"])
    fractions = case["materials"].get("fractions")
    fraction_ids = ([str(row["fraction_id"]) for row in fractions] if fractions is not None
                    else ["sediment"])
    bulk = np.asarray(source["bulk_fraction"], dtype=np.float64)
    if (len(fraction_ids) != nfrac or bulk.shape != (nfrac,) or
            not np.isfinite(bulk).all() or np.any(bulk <= 0)):
        raise ValueError("GPU preparation has inconsistent grain material")
    base_state = 3 + (1 + bed_layers) * nfrac
    q0 = np.asarray(source["state"], dtype=np.float64)[:base_state].T
    rock = np.asarray(source["rock"], dtype=np.float64)
    roughness_parameter = np.asarray(source["manning_cell"], dtype=np.float64)
    wet_depth = float(source["dry_tolerance"])
    sediment_enabled = bool(int(source["bedlaw"]) != 0 or int(source["exchange_mode"]) != 0)
    frames = verified["frames"]
    if not frames:
        raise ValueError("Completed GPU run has no frames")
    has_bank = frames[0].get("bank_history") is not None
    if any((frame.get("bank_history") is not None) != has_bank for frame in frames):
        raise ValueError("GPU bank-history availability changed during run")
    has_bedform = frames[0].get('bedform') is not None
    if any((frame.get('bedform') is not None) != has_bedform for frame in frames):
        raise ValueError('GPU bedform diagnostic availability changed during run')
    method = worker.get("method", {})
    prognostic_secondary = isinstance(method, dict) and method.get(
        "secondary_flow_model") == 2
    fields_by_frame = []
    for index, frame in enumerate(frames):
        _raise_if_cancelled(cancelled)
        if frame["q"].shape != q0.shape:
            raise ValueError("GPU and prepared numerical state layouts disagree")
        fields = _frame_fields(frame, q0, rock, roughness_parameter, bulk, fraction_ids,
                               bed_layers, sediment_enabled, wet_depth,
                               roughness_contract["variable"])
        layer_mass = frame["q"][:, 3 + nfrac:].reshape(count, bed_layers, nfrac)
        reconstructed_bed = rock + (layer_mass / bulk[None, None, :]).sum(axis=(1, 2))
        if not np.allclose(reconstructed_bed, frame["bed_elevation"],
                           rtol=0., atol=1e-8):
            raise ValueError(f"GPU bed elevation disagrees with conserved layers in frame {index}")
        fields["discharge_x_m2_s"] = np.asarray(frame["q"][:, 1], dtype=np.float64)
        fields["discharge_y_m2_s"] = np.asarray(frame["q"][:, 2], dtype=np.float64)
        if prognostic_secondary:
            fields["secondary_hI_m2_s"] = np.asarray(frame["secondary"], dtype=np.float64)
        if has_bank:
            for source_key, variable, _unit in GPU_BANK_RESULT_FIELDS:
                fields[variable] = np.asarray(frame["bank_history"][source_key], dtype=np.float64)
        fields_by_frame.append(fields)
        _report(progress, 10. + 25. * (index + 1) / len(frames))
    units = _variable_units(
        fraction_ids,bed_layers,sediment_enabled,
        roughness_variable=roughness_contract["variable"],
        roughness_units=roughness_contract["units"])
    units.update(discharge_x_m2_s="m2/s", discharge_y_m2_s="m2/s")
    if prognostic_secondary:
        units["secondary_hI_m2_s"] = "m2/s"
    if has_bank:
        units.update((variable, unit) for _source, variable, unit in GPU_BANK_RESULT_FIELDS)
    if has_bedform:
        units.update((variable, unit) for _source, variable, unit in GPU_BEDFORM_RESULT_FIELDS)

    result = root / "qgis_results"
    result.mkdir(exist_ok=True)
    rasters, raster_grid = _write_temporal_rasters(result, geometry, frames,
        fields_by_frame, units,
        progress=lambda value:_report(progress,35.+.5*value),cancelled=cancelled)
    _raise_if_cancelled(cancelled);_report(progress, 85.)
    geometry_sha = _digest(geometry_path)
    mesh = write_result_mesh(result / f"computational_mesh_{geometry_sha[:12]}.gpkg",
                             geometry, [float(frame["time_s"]) for frame in frames])
    products = [{"frame_index": index, "band": index + 1,
                 "time_s": float(frame["time_s"]), "steps": int(frame["steps"])}
                for index, frame in enumerate(frames)]
    manifest = {
        "format": "braids-qgis-gpu-results-v1", "source_run": str(root),
        "crs_wkt": geometry["crs_wkt"], "variables": units, "rasters": rasters,
        "raster_grid": raster_grid, "raster_storage": {"driver": "GTiff",
            "layout": "tiled", "compression": "none", "profile": "fast-qgis-geotiff-v1"},
        "mesh": mesh, "dry_tolerance_m": wet_depth,
        "hydraulic_roughness":roughness_contract,
        "sediment_enabled": sediment_enabled, "sediment_fractions": fraction_ids,
        "bed_layers": bed_layers, "frames": products, "boundary_budgets": None,
        "source_sha256": {"gpu_run": _digest(root / "run.json"),
            "gpu_geometry": _digest(root / "geometry.npz"),
            "prepared_geometry": geometry_sha, "prepared_case": _digest(case_path)},
        "source_artifacts": {"prepared_geometry": str(geometry_path),
            "input_preparation_sha256": worker["input_preparation_sha256"]},
    }
    if q3c_result_identity is not None:
        manifest.update(q3c_result_identity)
    _write_text_atomic(result / "results.json",
                       json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _report(progress, 100.)
    return manifest


def materialize_gpu_amr_results(run_root: str | Path, *, animated_mesh: bool = False,
                                progress=None, cancelled=None,
                                prepared_root: str | Path | None = None) -> dict[str, Any]:
    """Render authenticated, topology-changing GPU frames into QGIS products.

    This reader does not select or qualify an AMR numerical route. A future
    public worker must publish the schema-2 frame contract before it is used
    for a real run; the failed G4 dynamic secondary-flow gate remains separate.
    """
    from .gpu_amr_frame_io import read_gpu_amr_run
    from .sediment_product import validate_gpu_result_product

    root = filesystem_path(Path(run_root).expanduser()).resolve()
    _raise_if_cancelled(cancelled);_report(progress, 0.)
    run_envelope=json.loads((root/"run.json").read_text(encoding="utf-8"))
    prepared_for_product = (filesystem_path(Path(prepared_root).expanduser()).resolve()
        if prepared_root is not None else filesystem_path(
            Path(run_envelope.get("input_folder", root.parent / "gpu_prepared"))
            .expanduser()).resolve())
    validate_gpu_result_product(root, prepared_for_product)
    verified = read_gpu_amr_run(root, prepared_root=prepared_root)
    worker = verified["manifest"]
    if worker.get("status") != "completed":
        raise ValueError("Only a completed GPU AMR run can be loaded")
    index = verified["prepared_index"]
    frames = verified["frames"]
    has_bedform = frames[0].get('bedform') is not None
    if any((frame.get('bedform') is not None) != has_bedform for frame in frames):
        raise ValueError('GPU AMR bedform diagnostic availability changed during run')
    has_bank = frames[0].get("bank_history") is not None
    if any((frame.get("bank_history") is not None) != has_bank
           for frame in frames):
        raise ValueError("GPU AMR bank-history availability changed during run")
    topologies = index["topologies"]
    names = list(topologies)
    result = root / "qgis_results"
    source_sha = {"gpu_run": _digest(root / "run.json"),
                  "amr_prepared": index["manifest_sha256"]}
    cached_path = result / "results.json"
    if cached_path.is_file():
        try:
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            expected_frames = [(frame["time_s"], frame["steps"], frame["mesh_id"])
                               for frame in frames]
            cached_frames = [(row["time_s"], row["steps"], row["mesh_id"])
                             for row in cached["frames"]]
            raster_rows = cached["rasters"]
            cached_roughness = cached.get("hydraulic_roughness")
            if (not isinstance(cached_roughness, dict) or
                    cached_roughness.get("mode") not in ROUGHNESS_RESULT_CONTRACTS or
                    cached_roughness.get("variable") !=
                    ROUGHNESS_RESULT_CONTRACTS[cached_roughness["mode"]]["variable"] or
                    cached_roughness.get("units") !=
                    ROUGHNESS_RESULT_CONTRACTS[cached_roughness["mode"]]["units"]):
                raise ValueError("GPU AMR raster cache lacks a typed roughness identity")
            required_rasters = {"depth_m", "velocity_m_s", "stage_m"}
            if cached.get("sediment_enabled", True):
                required_rasters.add(cached_roughness["variable"])
            if has_bedform:
                required_rasters.update(variable for _source, variable, _unit
                                        in GPU_BEDFORM_RESULT_FIELDS)
            if has_bank:
                required_rasters.update(variable for _source, variable, _unit
                                        in GPU_BANK_RESULT_FIELDS)
            if (not isinstance(raster_rows, dict) or
                    not required_rasters <= raster_rows.keys() or
                    any(not isinstance(row, dict) or
                        not isinstance(row.get("path"), str) or
                        Path(row["path"]).name != row["path"] or
                        row.get("band_count") != len(frames) or
                        row.get("alpha_band") != len(frames)+1
                        for row in raster_rows.values())):
                raise ValueError("GPU AMR raster cache inventory differs")
            mesh_name = cached["mesh"]["path"]
            if not isinstance(mesh_name, str) or Path(mesh_name).name != mesh_name:
                raise ValueError("GPU AMR mesh cache path is unsafe")
            raster_files = [result / row["path"] for row in raster_rows.values()]
            raster_hashes = [row["sha256"] for row in raster_rows.values()]
            mesh_file = result / mesh_name
            reusable = (cached.get("format") == "braids-qgis-gpu-amr-results-v1" and
                cached.get("render_method") == "braids-gpu-amr-qgis-render-v1" and
                cached.get("source_sha256") == source_sha and
                expected_frames == cached_frames and
                cached["mesh"].get("mesh_count") == len(names) and
                cached["mesh"].get("frame_mesh_ids") ==
                    [frame["mesh_id"] for frame in frames] and
                mesh_file.is_file() and
                _digest(mesh_file) == cached["mesh"].get("sha256") and
                all(path.is_file() and _digest(path) == digest
                    for path, digest in zip(raster_files, raster_hashes)))
            if reusable:
                # ``animated`` records the requested time-synchronised display
                # mode, not whether AMR happened to publish multiple unique
                # topologies.  A bounded run may retain one topology at every
                # saved time; the user's animation selection must still stay
                # active and advance with the result frames.
                cached["mesh"]["animated"] = bool(animated_mesh)
                _write_text_atomic(cached_path,
                    json.dumps(cached, indent=2, sort_keys=True) + "\n")
                _report(progress, 100.)
                return cached
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            # A partial or old cache is not evidence about the live run.
            # Full frame/source authentication already succeeded above.
            pass
    geometries = {}
    materials = {}
    grid = None
    crs = None
    fraction_ids = None
    nfrac = frames[0]["nfrac"]
    bed_layers = frames[0]["bed_layers"]
    sediment_enabled = None
    roughness_contract = None
    for mesh_id, name in enumerate(names):
        _raise_if_cancelled(cancelled)
        row = topologies[name]
        packed = load_geometry(row["folder"] / "geometry.npz", include_faces=False)
        cells = packed["cells"]
        auth = row["geometry"]
        if (len(cells) != len(row["node_ids"]) or
                packed["crs_wkt"] != auth["declared_crs_wkt"] or
                any(int(cell["cell_id"]) != cell_id for cell_id, cell in enumerate(cells)) or
                not np.allclose([cell["area_m2"] for cell in cells],
                                auth["area_m2"], rtol=1e-11, atol=1e-11) or
                not np.allclose([cell["centroid_xy"] for cell in cells],
                                auth["centroid_xy"], rtol=1e-11, atol=1e-11)):
            raise ValueError("GPU AMR displayed cell geometry differs from authenticated nodes")
        if grid is None:
            grid = packed["grid"]
            crs = packed["crs_wkt"]
        elif grid != packed["grid"] or crs != packed["crs_wkt"]:
            raise ValueError("GPU AMR topologies do not share a display grid/CRS")
        initial = row["initial_data"]
        if (int(initial["nfrac"]) != nfrac or
                int(initial["nbed_layers"]) != bed_layers):
            raise ValueError("GPU AMR topology material layout changed")
        # The AMR exporter currently carries numerical fraction records, but
        # not the human-readable case JSON inside each topology folder.  Use
        # stable generic labels without following a machine-specific original
        # deck path. A later portable package can include authenticated names.
        case_path = row["folder"] / "case.data.json"
        case_metadata={}
        ids = (["sediment"] if nfrac == 1 else
               [f"grain_{fraction+1}" for fraction in range(nfrac)])
        if case_path.is_file():
            receipt = json.loads((row["folder"] / "preparation_receipt.json").read_text(
                encoding="utf-8"))
            origin = receipt.get("origin", {})
            expected_case = origin.get("files", {}).get(case_path.name)
            if not isinstance(expected_case, str) or _digest(case_path) != expected_case:
                raise ValueError("GPU AMR case metadata differs from authenticated input")
            case_metadata = json.loads(case_path.read_text(encoding="utf-8"))
            fractions = case_metadata["materials"].get("fractions")
            ids = ([str(item["fraction_id"]) for item in fractions]
                   if fractions is not None else ["sediment"])
        if fraction_ids is None:
            fraction_ids = ids
        elif ids != fraction_ids:
            raise ValueError("GPU AMR fraction names differ across topologies")
        bulk = np.asarray(initial["bulk_fraction"], dtype=np.float64)
        q0 = np.asarray(initial["state"], dtype=np.float64)[:
            3+nfrac+bed_layers*nfrac].T
        rock = np.asarray(initial["rock"], dtype=np.float64)
        manning = np.asarray(initial["manning_cell"], dtype=np.float64)
        topology_roughness=_roughness_result_contract(
            case=case_metadata,source=initial)
        if roughness_contract is None:roughness_contract=topology_roughness
        elif topology_roughness!=roughness_contract:
            raise ValueError("GPU AMR hydraulic roughness identity changed across topologies")
        enabled = bool(int(initial["bedlaw"]) != 0 or
                       int(initial["exchange_mode"]) != 0)
        if sediment_enabled is None:
            sediment_enabled = enabled
        elif enabled != sediment_enabled:
            raise ValueError("GPU AMR sediment switch differs across topologies")
        geometries[mesh_id] = packed
        materials[mesh_id] = (q0, rock, manning, bulk,
                              float(initial["dry_tolerance"]))
    if len(fraction_ids) != nfrac:
        raise ValueError("GPU AMR grain fraction count differs from frames")
    if len({entry[4] for entry in materials.values()}) != 1:
        raise ValueError("GPU AMR wet/dry thresholds differ across topologies")
    wet_depth = materials[0][4]
    fields_by_frame = []
    for frame_number, frame in enumerate(frames):
        _raise_if_cancelled(cancelled)
        q0, rock, manning, bulk, _ = materials[frame["mesh_id"]]
        if frame["q"].shape != q0.shape:
            raise ValueError("GPU AMR frame/initial state layout differs")
        fields = _frame_fields(frame, q0, rock, manning, bulk, fraction_ids,
                               bed_layers, sediment_enabled, wet_depth,
                               roughness_contract["variable"])
        fields["discharge_x_m2_s"] = frame["q"][:, 1]
        fields["discharge_y_m2_s"] = frame["q"][:, 2]
        fields["secondary_hI_m2_s"] = frame["secondary"]
        if has_bank:
            for source_key, variable, _unit in GPU_BANK_RESULT_FIELDS:
                fields[variable] = frame["bank_history"][source_key]
        fields_by_frame.append(fields)
        _report(progress, 10. + 25. * (frame_number + 1) / len(frames))
    units = _variable_units(
        fraction_ids,bed_layers,sediment_enabled,
        roughness_variable=roughness_contract["variable"],
        roughness_units=roughness_contract["units"])
    units.update(discharge_x_m2_s="m2/s", discharge_y_m2_s="m2/s",
                 secondary_hI_m2_s="m2/s")
    if has_bank:
        units.update((variable, unit) for _, variable, unit in GPU_BANK_RESULT_FIELDS)
    if has_bedform:
        units.update((variable, unit) for _, variable, unit in GPU_BEDFORM_RESULT_FIELDS)
    snapshots = [{"mesh_id": mesh_id, "time_s": next(
        (frame["time_s"] for frame in frames if frame["mesh_id"] == mesh_id), 0.),
        "cells": geometries[mesh_id]["cells"]}
        for mesh_id in range(len(names))]
    geometry = {**geometries[0], "mesh_snapshots": snapshots,
                "frame_mesh_ids": [frame["mesh_id"] for frame in frames]}
    result.mkdir(exist_ok=True)
    rasters, raster_grid = _write_temporal_rasters(result, geometry, frames,
        fields_by_frame, units, progress=lambda value: _report(progress, 35. + .5*value),
        cancelled=cancelled)
    _raise_if_cancelled(cancelled);_report(progress, 85.)
    topology_digest = hashlib.sha256(index["manifest_sha256"].encode("ascii"))
    for frame in frames:
        topology_digest.update(np.asarray([frame["mesh_id"]], dtype="<i8").tobytes())
        topology_digest.update(np.asarray(frame["node_ids"], dtype="<i8").tobytes())
    mesh = write_result_mesh(result / f"computational_mesh_{topology_digest.hexdigest()[:12]}.gpkg",
                             geometry, [frame["time_s"] for frame in frames])
    mesh["animated"] = bool(animated_mesh)
    mesh["sha256"] = _digest(result / mesh["path"])
    products = [{"frame_index": number, "band": number+1,
                 "time_s": frame["time_s"], "steps": frame["steps"],
                 "mesh_id": frame["mesh_id"]}
                for number, frame in enumerate(frames)]
    manifest = {"format": "braids-qgis-gpu-amr-results-v1",
        "render_method": "braids-gpu-amr-qgis-render-v1", "source_run": str(root),
        "crs_wkt": crs, "variables": units, "rasters": rasters,
        "raster_grid": raster_grid, "raster_storage": {"driver": "GTiff",
            "layout": "tiled", "compression": "none",
            "profile": "fast-qgis-geotiff-v1"},
        "mesh": mesh, "dry_tolerance_m": wet_depth,
        "hydraulic_roughness":roughness_contract,
        "sediment_enabled": sediment_enabled,
        "sediment_fractions": fraction_ids, "bed_layers": bed_layers,
        "frames": products, "boundary_budgets": None,
        "source_sha256": source_sha,
        "source_artifacts": {"prepared_amr": str(index["root"]),
            "input_preparation_sha256": worker["input_preparation_sha256"]}}
    _write_text_atomic(result / "results.json",
                       json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _report(progress, 100.)
    return manifest


def materialize_results(run_root: str | Path, *, animated_mesh: bool = False,
                        progress=None, cancelled=None,
                        prepared_root: str | Path | None = None) -> dict[str, Any]:
    run_root = filesystem_path(Path(run_root).expanduser()).resolve()
    gpu_manifest = run_root / "run.json"
    if gpu_manifest.is_file():
        candidate = json.loads(gpu_manifest.read_text(encoding="utf-8"))
        if candidate.get("backend") == "gpu":
            return materialize_gpu_results(run_root, animated_mesh=animated_mesh,
                                           progress=progress, cancelled=cancelled,
                                           prepared_root=prepared_root)
    _raise_if_cancelled(cancelled);_report(progress,0.)
    _preserve_manifest_dynamic_mesh(run_root)
    marker = json.loads((run_root / RUN_MARKER).read_text(encoding="utf-8"))
    if marker.get("status") != "completed":
        raise ValueError("Only a completed BRAIDS run can be loaded")
    saved_project={}
    if (run_root / "project.json").is_file():
        saved_project=json.loads((run_root / "project.json").read_text(encoding="utf-8"))
    sediment_enabled=bool(saved_project.get("numerics",{}).get("sediment_enabled",True))
    # Native previews keep raster result preparation instant. Completed runs
    # still need their computational mesh on demand for the Results overlay.
    live_manifest = materialize_live_preview_results(run_root)
    if live_manifest is not None and (not sediment_enabled or not (run_root / "frames.txt").is_file()):
        case = json.loads((run_root / "case.data.json").read_text(encoding="utf-8"))
        live_manifest["capacity_route_identity"]=_capacity_result_contract(
            run_root,case,marker)
        live_manifest["execution_profile_v2"]=_execution_profile_result_contract(
            run_root,case,marker)
        hiding_result_contract=_hiding_foundation_result_contract(
            run_root,case,marker)
        if hiding_result_contract is not None:
            live_manifest["hiding_foundation_identity"]=hiding_result_contract
        secondary_result_contract=_secondary_foundation_result_contract(
            run_root,case,marker)
        if secondary_result_contract is not None:
            live_manifest["secondary_foundation_identity"]=\
                secondary_result_contract
        prognostic_result_contract=_prognostic_secondary_result_contract(
            run_root,case,marker)
        if prognostic_result_contract is not None:
            live_manifest["prognostic_secondary_identity"]=\
                prognostic_result_contract
        live_manifest["source_artifacts"] = dict(marker.get("sidecars", {}))
        live_manifest["boundary_budgets"] = _write_boundary_budgets(
            run_root, run_root / "qgis_results", case)
        live_manifest["boundary_flux_sign"] = "nonnegative magnitude; boundary role identifies inflow or outflow"
        live_manifest["mesh"]=_materialize_completed_live_mesh(run_root,live_manifest,animated=animated_mesh,
                                                               progress=progress,cancelled=cancelled)
        _write_text_atomic(run_root/"qgis_results"/"results.json",
            json.dumps(live_manifest,indent=2,sort_keys=True)+"\n")
        return live_manifest
    frames = read_frames(run_root / "frames.txt")
    adaptive = 'node_ids' in frames[0]
    geometry_path = geometry_artifact(run_root, "geometry")
    geometry = load_geometry(geometry_path, cell_ids=[] if adaptive else None,
                             include_faces=False)
    case = json.loads((run_root / "case.data.json").read_text(encoding="utf-8"))
    capacity_result_contract=_capacity_result_contract(run_root,case,marker)
    execution_profile_result_contract=_execution_profile_result_contract(
        run_root,case,marker)
    hiding_result_contract=_hiding_foundation_result_contract(
        run_root,case,marker)
    secondary_result_contract=_secondary_foundation_result_contract(
        run_root,case,marker)
    prognostic_result_contract=_prognostic_secondary_result_contract(
        run_root,case,marker)
    roughness_contract=_roughness_result_contract(
        case=case,project=saved_project,source=marker)
    default_roughness = case.get("hydraulic_roughness", {}).get("default_parameter")
    if default_roughness is None and saved_project:
        default_roughness = saved_project.get("roughness", {}).get("default_parameter",
            saved_project.get("roughness", {}).get("default_manning_n",
            saved_project.get("numerics", {}).get("manning", 0.0)))
    if default_roughness is None:
        default_roughness = case.get("manning", {}).get("default_manning_n",0.0)
    with np.load(run_root / "initial_fields.npz") as initial:
        q0 = initial["q"].copy()
        rock = initial["bedrock_m"].copy()
        roughness_parameter = (
            initial["roughness_parameter"].copy()
            if "roughness_parameter" in initial.files else
            initial["manning_n"].copy() if "manning_n" in initial.files else
            np.full(len(q0), float(default_roughness)))
    nfrac, bed_layers = frames[0]["nfrac"], frames[0]["bed_layers"]
    material_fractions = case["materials"].get("fractions")
    if material_fractions is None:
        fraction_ids = ["sediment"]
        bulk = np.array([float(case["materials"]["bulk_kg_m3"])])
    else:
        fraction_ids = [row["fraction_id"] for row in material_fractions]
        bulk = np.array([float(row["bulk_kg_m3"]) for row in material_fractions])
    if len(fraction_ids) != nfrac:
        raise ValueError("Frame and material fraction counts disagree")
    units = _variable_units(
        fraction_ids,bed_layers,sediment_enabled,
        roughness_variable=roughness_contract["variable"],
        roughness_units=roughness_contract["units"])
    wet_depth_m=float(case.get("dry_tolerance_m",WET_DEPTH_M))
    hierarchy_path = None
    if adaptive:
        hierarchy_path=geometry_artifact(run_root,'amr_hierarchy')
        index=load_hierarchy_index(hierarchy_path)
        parent=np.asarray(index['parent']);offsets=np.asarray(index['offsets'])
        node_count=len(parent);fine_offset=int(offsets[-2]);areas=np.asarray(index['area_m2'])
        children=[[] for _ in range(node_count)]
        for n,p in enumerate(parent):
            if p>=0:children[p].append(n)
        def restrict(values,maximum=False):
            values=np.asarray(values)
            out=np.empty((node_count,*values.shape[1:]));out[fine_offset:]=values
            for node in range(fine_offset-1,-1,-1):
                child=children[node]
                out[node]=(np.max(out[child],axis=0) if maximum else
                    np.tensordot(areas[child],out[child],axes=(0,0))/math.fsum(areas[child]))
            return out
        node_q0,node_rock,node_roughness=(
            restrict(q0),restrict(rock),restrict(roughness_parameter,True))
        fields_by_frame=[];snapshots={}
        used_nodes=np.unique(np.concatenate([np.asarray(frame['node_ids'],dtype=np.int64)
                                             for frame in frames]))
        selected=load_geometry(hierarchy_path,cell_ids=used_nodes,include_faces=False)['cells']
        cells_by_node={int(node):cell for node,cell in zip(used_nodes,selected)}
        for frame in frames:
            ids=frame['node_ids'];mesh_id=frame['mesh_id']
            fields_by_frame.append(_frame_fields(
                frame,node_q0[ids],node_rock[ids],node_roughness[ids],
                bulk,fraction_ids,bed_layers,sediment_enabled,wet_depth_m,
                roughness_contract["variable"]))
            if mesh_id not in snapshots:
                snapshots[mesh_id]={'mesh_id':mesh_id,'time_s':frame['time_s'],
                    'cells':[{**cells_by_node[int(node)],'cell_id':row}
                             for row,node in enumerate(ids)]}
        geometry={**geometry,'mesh_snapshots':list(snapshots.values()),
                  'frame_mesh_ids':[f['mesh_id'] for f in frames]}
    else:
        fields_by_frame = [_frame_fields(
            frame,q0,rock,roughness_parameter,bulk,fraction_ids,bed_layers,
            sediment_enabled,wet_depth_m,roughness_contract["variable"])
                           for frame in frames]
    result = run_root / "qgis_results"
    result.mkdir(exist_ok=True)
    for stale in [*result.glob("frame_*.geojson"), *result.glob(".temporal_*.partial.tif")]:
        stale.unlink(missing_ok=True)
    rasters, raster_grid = _write_temporal_rasters(result, geometry, frames, fields_by_frame, units)
    geometry_sha = _digest(geometry_path)
    mesh_sha=geometry_sha
    if 'node_ids' in frames[0]:
        # A re-run can reuse mesh IDs and counts while refining different cells.
        # Include the actual frame topology in the filename of the cached mesh.
        topology=hashlib.sha256(geometry_sha.encode('ascii'))
        topology.update(_digest(hierarchy_path).encode('ascii'))
        for frame in frames:
            topology.update(np.asarray([frame['mesh_id']],dtype='<i8').tobytes())
            topology.update(np.asarray(frame['node_ids'],dtype='<i8').tobytes())
        mesh_sha=topology.hexdigest()
    mesh = write_result_mesh(result / f"computational_mesh_{mesh_sha[:12]}.gpkg", geometry,
                             [frame["time_s"] for frame in frames])
    active_rasters={row["path"] for row in rasters.values()}
    for stale in result.glob("temporal_*.tif"):
        if stale.name not in active_rasters:stale.unlink(missing_ok=True)
    boundary_budget_name = _write_boundary_budgets(run_root, result, case, frames)
    products = [{"frame_index": index, "band": index + 1, "time_s": frame["time_s"],
                 "steps": frame["steps"]} for index, frame in enumerate(frames)]
    manifest = {
        "format": "braids-qgis-results-v2",
        "source_run": str(run_root),
        "crs_wkt": geometry["crs_wkt"],
        "variables": units,
        "rasters": rasters,
        "raster_grid": raster_grid,
        "raster_storage": {"driver": "GTiff", "layout": "tiled",
                           "compression": "none", "profile": "fast-qgis-geotiff-v1"},
        "mesh": mesh,
        "hydraulic_roughness":roughness_contract,
        "capacity_route_identity":capacity_result_contract,
        "execution_profile_v2":execution_profile_result_contract,
        "dry_tolerance_m":wet_depth_m,
        "sediment_enabled":sediment_enabled,
        "sediment_fractions": fraction_ids,
        "bed_layers": bed_layers,
        "frames": products,
        "boundary_budgets": boundary_budget_name,
        "boundary_flux_sign": "nonnegative magnitude; boundary role identifies inflow or outflow",
        "source_sha256": {"frames": _digest(_frames_payload_path(run_root / "frames.txt")),
                          "frames_marker": _digest(run_root / "frames.txt"),
                          "geometry": geometry_sha,
                          "initial": _digest(run_root / "initial_fields.npz"),
                          "case_metadata": _digest(run_root / "case.data.json")},
        "source_artifacts": {"geometry": geometry_path.name,
                             "amr_hierarchy": None if hierarchy_path is None else hierarchy_path.name},
    }
    if hiding_result_contract is not None:
        manifest["hiding_foundation_identity"]=hiding_result_contract
    if secondary_result_contract is not None:
        manifest["secondary_foundation_identity"]=secondary_result_contract
    if prognostic_result_contract is not None:
        manifest["prognostic_secondary_identity"]=\
            prognostic_result_contract
    _write_text_atomic(result / "results.json",
                       json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _report(progress,100.)
    return manifest
