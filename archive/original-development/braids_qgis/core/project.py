"""Persistent, QGIS-independent preparation for native BRAIDS polygon runs."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict,replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from osgeo import gdal, ogr, osr

from braids_core.boundaries.cut_cell_io import (CutBoundary, CutMorphologicalFactorSeries,
                                               CutSediment, CutSedimentFraction,
                                               write_cut_case)
from braids_core.boundaries.cut_cells import CutCellMesh, cut_cell_polygon, map_cut_cells
from braids_core.boundaries.amr import AMRSettings, build_hierarchy
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain, boundary_polyline
from braids_core.boundaries.gis import section_from_polyline
from braids_core.boundaries.inlet_io import NativeGrid
from .bed_inputs import (ERODIBLE_LAYER_NAME, load_erodible_zones,
                         map_erodible_to_cells)
from .bank_inputs import (BANK_LAYER_NAME, load_bank_zones,
                          map_bank_zones_to_cells)
from .structure_inputs import OBSTACLE_LAYER_NAME, load_solid_obstacles
from .refinement_inputs import REFINEMENT_LAYER_NAME,load_refinement_zones,map_refinement_zones
from .roughness import MANNING_LAYER_NAME, load_manning_zones, map_manning_to_cells
from .typed_roughness_zones import (
    load_typed_roughness_zones,
    map_typed_roughness_to_cells,
)
from .water_body import WATER_BODY_LAYER_NAME, map_water_body_to_cells
from .inlet_diagnostics import DIAGNOSTICS_FILE, InletPreparationError, represented_inlets
from .boundary_terrain import sample_inlet_terrain
from .raster_surface import surface_averages
from braids_core.boundaries.agglomeration import agglomerate_small_cells
from braids_core.sediment.capacity_profile import (
    SUSPENDED_CHANNEL_ZERO,
    CapacityProfileError,
    capacity_profile_document,
    capacity_profile_semantic_id,
    load_capacity_profile,
    resolve_capacity_profile_from_code,
)
from braids_core.sediment.capacity_execution_identity import (
    CAPACITY_EXECUTION_SCOPE_ID,
    EH_DELFT_METHOD_ID,
    EH_LEGACY_METHOD_ID,
    MPM_DELFT_METHOD_ID,
    MPM_LEGACY_METHOD_ID,
    CapacityExecutionIdentityError,
    capacity_execution_document,
    capacity_execution_semantic_id,
    load_capacity_execution_identity,
    migrate_legacy_capacity_controls,
)
from braids_core.sediment.settling import DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3
from braids_core.sediment.method_selection import resolve_sediment_methods
from braids_core.sediment.execution_profile_binding import (
    EXECUTION_PROFILE_BINDING_KEY,
    ExecutionProfileError,
    legacy_execution_profile_binding,
    load_execution_profile_binding,
    runtime_execution_profile,
)
from braids_core.sediment.execution_profile_v2 import (
    BACKEND_CPU,
    BACKEND_GPU,
    FROZEN_Q3SP_LEDGER_SEMANTIC_ID,
    bind_component_profile,
    execution_profile_document,
    is_frozen_q3sp_execution_profile,
    load_execution_profile,
    resolve_execution_profile,
)
from braids_core.sediment.capacity_foundation_profile import (
    CAPACITY_FOUNDATION_RESOLUTION_KEY,
    GENERAL_METHOD_ID,
    CapacityFoundationProfileError,
    resolve_project_capacity_foundation_profile,
    validate_capacity_foundation_resolution,
)
from braids_core.sediment.capacity_foundation_execution import (
    capacity_foundation_execution_document,
)
from braids_core.sediment.hiding_foundation_execution import (
    COMPOSITION_ORDER as HIDING_FOUNDATION_COMPOSITION_ORDER,
    EXECUTION_KEY as HIDING_FOUNDATION_EXECUTION_KEY,
    METHOD_BY_MODE as HIDING_FOUNDATION_METHOD_BY_MODE,
    ORDERED_ROUTE_KEY as HIDING_FOUNDATION_ORDERED_ROUTE_KEY,
    SCOPE_ID as HIDING_FOUNDATION_SCOPE_ID,
    SEMANTIC_ID as HIDING_FOUNDATION_SEMANTIC_ID,
    hiding_foundation_execution_document,
    validate_hiding_foundation_execution_document,
)
from braids_core.sediment.secondary_foundation_execution import (
    COMPOSITION_ORDER as SECONDARY_FOUNDATION_COMPOSITION_ORDER,
    EXECUTION_KEY as SECONDARY_FOUNDATION_EXECUTION_KEY,
    METHOD_ID as SECONDARY_FOUNDATION_METHOD_ID,
    ORDERED_ROUTE_KEY as SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY,
    SCOPE_ID as SECONDARY_FOUNDATION_SCOPE_ID,
    SEMANTIC_ID as SECONDARY_FOUNDATION_SEMANTIC_ID,
    secondary_foundation_execution_document,
    validate_secondary_foundation_execution_document,
)
from braids_core.sediment.prognostic_secondary_execution import (
    COMPOSITION_ORDER as PROGNOSTIC_SECONDARY_COMPOSITION_ORDER,
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    GPU_CHECKPOINT_SCHEMA as PROGNOSTIC_SECONDARY_GPU_CHECKPOINT_SCHEMA,
    INFLOW_MODE as PROGNOSTIC_SECONDARY_INFLOW_MODE,
    METHOD_ID as PROGNOSTIC_SECONDARY_METHOD_ID,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    SCOPE_ID as PROGNOSTIC_SECONDARY_SCOPE_ID,
    SEMANTIC_ID as PROGNOSTIC_SECONDARY_SEMANTIC_ID,
    prognostic_secondary_execution_document,
    validate_prognostic_secondary_execution_document,
)
from braids_core.sediment_method_catalog import (
    CATALOG_SCHEMA,
    CATALOG_SCOPE_ID,
    legacy_code_map,
    semantic_catalog_id,
)
from .sediment_product import (
    PROJECT_KEY as QGIS_PRODUCT_KEY,
    bind_qgis_product,
    validate_qgis_product,
    write_prepared_product,
)
from .recursive_product import write_recursive_prepared_product

LEGACY_PROJECT_FORMAT = "braids-qgis-project-v1"
PROJECT_FORMAT = "braids-qgis-project-v2"
RUN_FORMAT = "braids-qgis-run-v1"
RUN_MARKER = ".braids_qgis_run.json"
COMPACT_SIDECAR_CELL_THRESHOLD = 250_000
PREPARATION_CACHE_FORMAT = "braids-preparation-cache-v1"
CAPACITY_BINDING_SCHEMA = 1
CAPACITY_BINDING_KEY = "capacity_profile_binding"
CAPACITY_EXECUTION_BINDING_SCHEMA = 1
CAPACITY_EXECUTION_BINDING_KEY = "capacity_execution_binding"
CAPACITY_EXECUTION_SIDECAR = "capacity_execution.json"
EXECUTION_PROFILE_V2_SIDECAR = "execution_profile_v2.json"
HIDING_FOUNDATION_BINDING_KEY = "hiding_foundation_binding"
HIDING_FOUNDATION_BINDING_SCHEMA = 1
HIDING_FOUNDATION_BINDING_SCOPE_ID = (
    "braids-qgis-q3h1-hiding-foundation-deferred-binding-v1"
)
HIDING_FOUNDATION_BINDING_SEMANTIC_ID = hashlib.sha256(
    HIDING_FOUNDATION_BINDING_SCOPE_ID.encode("ascii")
).hexdigest()
SECONDARY_FOUNDATION_BINDING_KEY = "secondary_foundation_binding"
SECONDARY_FOUNDATION_BINDING_SCHEMA = 1
SECONDARY_FOUNDATION_BINDING_SCOPE_ID = (
    "braids-qgis-q3h2-secondary-foundation-deferred-binding-v1"
)
SECONDARY_FOUNDATION_BINDING_SEMANTIC_ID = hashlib.sha256(
    SECONDARY_FOUNDATION_BINDING_SCOPE_ID.encode("ascii")
).hexdigest()
PROGNOSTIC_SECONDARY_BINDING_KEY = "prognostic_secondary_binding"
PROGNOSTIC_SECONDARY_BINDING_SCHEMA = 1
PROGNOSTIC_SECONDARY_BINDING_SCOPE_ID = (
    "braids-qgis-q3h3-prognostic-secondary-deferred-binding-v1"
)
PROGNOSTIC_SECONDARY_BINDING_SEMANTIC_ID = hashlib.sha256(
    PROGNOSTIC_SECONDARY_BINDING_SCOPE_ID.encode("ascii")
).hexdigest()
PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID = PROGNOSTIC_SECONDARY_METHOD_ID
PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE = (
    "delft3d4_flow2dh_classic_sub1_i_equili_n"
)
PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY = (
    "prognostic_secondary_bed_boundary_rotation_policy"
)
PROGNOSTIC_SECONDARY_STATE_UNITS_KEY = "prognostic_secondary_state_units"
PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY = (
    "prognostic_secondary_integrated_ledger_units"
)
PROGNOSTIC_SECONDARY_NATIVE_SOURCE = (
    "solver/wave/braids_flow2dh_prognostic_secondary_module.f90"
)
SPLIT_CAPACITY_LAWS = frozenset((11, 12, 13, 14))
LEGACY_SECONDARY_FLOW_MODEL_LABEL = (
    "D-Flow FM equilibrium streamline curvature and spiral intensity"
)

_CAPACITY_SOURCE_BY_CODE = {
    code: row.source_id for code, row in legacy_code_map("capacity").items()
    if isinstance(code, int)
}
_SLOPE_SOURCE_BY_CODE = {
    code: row.source_id for code, row in legacy_code_map("bed_slope").items()
    if isinstance(code, int)
}
_PROFILE_SHEAR_METHOD = {
    "total": "shear.total_roughness",
    "soulsby_clarke_skin": "shear.soulsby_clarke_skin",
}
_PROFILE_HIDING_METHOD = {
    0: "hiding.none",
    2: "hiding.egiazaroff",
    3: "hiding.ashida_michiue",
    4: "hiding.parker_klm",
    5: "hiding.wu_wang_jia",
}
_PROFILE_SLOPE_METHOD = {
    0: "bed_slope.off",
    3: "bed_slope.delft_koch_flokstra",
    5: "bed_slope.delft_bagnold_ikeda",
    6: "bed_slope.delft_parker_andrews",
}
_Q2_CAPACITY_METHODS = frozenset((
    EH_LEGACY_METHOD_ID, EH_DELFT_METHOD_ID,
    MPM_LEGACY_METHOD_ID, MPM_DELFT_METHOD_ID,
))
_LEGACY_Q2_CAPACITY_METHODS = frozenset((
    EH_LEGACY_METHOD_ID, MPM_LEGACY_METHOD_ID,
))
_ROUGHNESS_PUBLIC = {
    "chezy": ("chezy_C", "m^0.5/s", "Chezy C"),
    "manning": ("manning_n", "s/m^(1/3)", "Manning n"),
    "white_colebrook": ("nikuradse_ks", "m", "Nikuradse ks"),
}
_HISTORICAL_Q3SP_METHOD_IDS = frozenset((
    "bed_slope.off", "capacity.general", "hiding.none", "secondary.off",
    "shear.total_roughness", "suspension.off", "transport_route.total",
))
_HISTORICAL_Q3SP_NOTE = (
    "Loaded the frozen historical Q3-S/P seven-selector law-11 profile. Its "
    "typed C/n/ks forcing remains authenticated by the existing forcing "
    "contract; new Q3-C profiles include that roughness dependency in schema v2."
)


def _is_historical_q3sp_general_profile(
    binding: Mapping[str, Any],
) -> bool:
    """Recognize the frozen seven-selector law-11 admission only.

    That historical profile predates the Q3-C discovery that TraFrm=4 also
    directly consumes the resolved hydraulic Chezy.  Existing runs remain
    loadable and keep their separately authenticated typed-forcing contract;
    new automatic profiles include the roughness selector instead.
    """
    profile=binding.get("profile")
    if not isinstance(profile,Mapping) or profile.get("backend") is not None:
        return False
    if profile.get("ledger_semantic_id")!=FROZEN_Q3SP_LEDGER_SEMANTIC_ID:
        return False
    selections=profile.get("method_selections")
    layout=profile.get("fraction_layout")
    if not isinstance(selections,list) or not isinstance(layout,list) or len(layout)!=1:
        return False
    method_ids=[row.get("entry_id") for row in selections
                if isinstance(row,Mapping)]
    return (len(method_ids)==len(selections)==7 and
            frozenset(method_ids)==_HISTORICAL_Q3SP_METHOD_IDS and
            layout[0].get("transport_class")==1)


def _capacity_fraction_layout(project: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    rows = project.get("sediment_fractions")
    if rows is None:
        return ("single",), (project["sediment"],)
    return tuple(row["fraction_id"] for row in rows), tuple(rows)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,sort_keys=True,separators=(",",":"),ensure_ascii=True,
        allow_nan=False).encode("ascii")


def _hiding_foundation_mode(project: Mapping[str, Any]) -> int:
    numerics=project.get("numerics",{})
    if not isinstance(numerics,Mapping):return 0
    mode=numerics.get("hiding_exposure",0)
    return mode if type(mode) is int else 0


def _hiding_foundation_binding_document(
    *,mode: int,capacity_law: int,fraction_ids: Sequence[str],
    parker_alpha: float,wu_exponent: float,
    capacity_profile_fingerprint: str,
) -> dict[str, Any]:
    """Return the canonical project-owned, prepare-deferred Q3-H1 binding."""
    if type(mode) is not int or mode not in HIDING_FOUNDATION_METHOD_BY_MODE:
        raise ValueError("Q3-H1 hiding mode must be the integer 2, 3, 4 or 5")
    if type(capacity_law) is not int or capacity_law not in (2,11):
        raise ValueError("Q3-H1 hiding supports corrected MPM law 2 or general law 11")
    ids=tuple(fraction_ids)
    if (len(ids)!=2 or len(set(ids))!=2 or
            any(not isinstance(value,str) or not value or
                value!=value.strip() for value in ids)):
        raise ValueError("Q3-H1 hiding requires two ordered unique fraction identities")
    if (isinstance(parker_alpha,bool) or isinstance(wu_exponent,bool) or
            not isinstance(parker_alpha,(int,float)) or
            not isinstance(wu_exponent,(int,float)) or
            not math.isfinite(float(parker_alpha)) or
            not math.isfinite(float(wu_exponent))):
        raise ValueError("Q3-H1 Parker/Wu controls must be finite")
    if (not isinstance(capacity_profile_fingerprint,str) or
            len(capacity_profile_fingerprint)!=64 or
            any(value not in "0123456789abcdef"
                for value in capacity_profile_fingerprint)):
        raise ValueError("Q3-H1 base capacity fingerprint is invalid")
    body={
        "schema":HIDING_FOUNDATION_BINDING_SCHEMA,
        "scope_id":HIDING_FOUNDATION_BINDING_SCOPE_ID,
        "semantic_id":HIDING_FOUNDATION_BINDING_SEMANTIC_ID,
        "state":"deferred_prepared_capacity_execution",
        "execution_scope_id":HIDING_FOUNDATION_SCOPE_ID,
        "execution_semantic_id":HIDING_FOUNDATION_SEMANTIC_ID,
        "mode":mode,
        "method_id":HIDING_FOUNDATION_METHOD_BY_MODE[mode],
        "capacity_law":capacity_law,
        "fraction_ids":list(ids),
        "parker_alpha":float(parker_alpha),
        "wu_exponent":float(wu_exponent),
        "capacity_profile_fingerprint":capacity_profile_fingerprint,
        "base_hiding_method_id":"hiding.none",
        "composition_order":list(HIDING_FOUNDATION_COMPOSITION_ORDER),
        "preparation_requirement":(
            "bind the canonical Q3-C execution fingerprint after typed forcing preparation"),
        "backend_selection":"explicit_user_choice",
        "automatic_cpu_fallback":False,
    }
    return {**body,"fingerprint":hashlib.sha256(
        _canonical_json_bytes(body)).hexdigest()}


def _validate_hiding_foundation_binding_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild an untrusted deferred declaration and require exact bytes."""
    if not isinstance(value,Mapping):
        raise ValueError("Q3-H1 hiding project binding must be an object")
    try:
        rebuilt=_hiding_foundation_binding_document(
            mode=value["mode"],capacity_law=value["capacity_law"],
            fraction_ids=value["fraction_ids"],
            parker_alpha=value["parker_alpha"],
            wu_exponent=value["wu_exponent"],
            capacity_profile_fingerprint=value[
                "capacity_profile_fingerprint"])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError("Invalid Q3-H1 hiding project binding") from exc
    if _canonical_json_bytes(value)!=_canonical_json_bytes(rebuilt):
        raise ValueError("Q3-H1 hiding project binding is stale or forged")
    return rebuilt


def _build_hiding_foundation_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the deferred project declaration for additive Q3-H1.

    The final canonical execution identity cannot be created until typed
    roughness and Q3-C forcing bytes exist.  This declaration binds every
    project-owned input plus the neutral Q3-C capacity component and is
    materialized into ``hiding_foundation_execution_document`` at Prepare Run.
    """
    mode=_hiding_foundation_mode(project)
    if mode==0:return None
    if mode not in HIDING_FOUNDATION_METHOD_BY_MODE:
        raise ValueError("Q3-H1 hiding mode must be the integer 2, 3, 4 or 5")
    numerics=project["numerics"]
    law=numerics.get("bedload") if numerics.get("sediment_enabled",True) else 0
    if type(law) is not int or law not in (2,11):
        raise ValueError("Q3-H1 hiding supports corrected MPM law 2 or general law 11")
    if numerics.get("exchange")!=0:
        raise ValueError("Q3-H1 hiding requires exchange off and class-1 fractions")
    if numerics.get("bed_slope")!=0:
        raise ValueError("Q3-H1 hiding requires bed-slope transport correction off")
    secondary=project.get("internal_conditions",{}).get(
        "secondary_flow",{"enabled":False})
    if not isinstance(secondary,Mapping):
        raise ValueError("Q3-H1 secondary-flow declaration is invalid")
    if secondary.get("enabled",False):
        # A v56 equilibrium or sibling v57 prognostic operator may wrap this
        # v55 foundation.  The nested profile itself remains secondary.off;
        # neither outer route is allowed to reinterpret it.
        secondary_model=_resolved_secondary_flow_model(
            secondary.get("model","equilibrium"))
        if (secondary_model not in ("equilibrium","prognostic") or
                secondary.get("transport_direction",True) is not True or
                _finite(secondary.get("momentum_beta",0.),
                        "Q3-H momentum Beta")!=0.0):
            raise ValueError(
                "Q3-H1 can compose only with the Q3-H2/H3 zero-Betac secondary routes")
    rows=project.get("sediment_fractions")
    if not isinstance(rows,(list,tuple)) or len(rows)!=2:
        raise ValueError("Q3-H1 hiding requires exactly two class-1 fractions")
    fraction_ids=tuple(row.get("fraction_id") for row in rows
                       if isinstance(row,Mapping))
    if (len(fraction_ids)!=2 or len(set(fraction_ids))!=2 or
            any(not isinstance(value,str) or not value for value in fraction_ids)):
        raise ValueError("Q3-H1 hiding requires two ordered unique fraction identities")
    if any(row.get("sediment_type","sand")!="sand" for row in rows):
        raise ValueError("Q3-H1 hiding class-1 fractions must be noncohesive")
    initial=project.get("initial",{})
    if (not isinstance(initial,Mapping) or
            float(initial.get("active_layer_thickness_m",0.))<=0.):
        raise ValueError("Q3-H1 hiding requires an active surface bed layer")
    binding=project.get(CAPACITY_BINDING_KEY)
    capacity=(binding.get("capacity_runtime_profile")
              if isinstance(binding,Mapping) and
              binding.get("state")=="canonical" else None)
    if not isinstance(capacity,Mapping):
        raise ValueError("Q3-H1 hiding requires the canonical Q3-C capacity component")
    if capacity.get("hiding",{}).get("method_id")!="hiding.none":
        raise ValueError(
            "Q3-H1 outer hiding must retain hiding.none in the nested Q3-C capacity component")
    capacity_fingerprint=capacity.get("fingerprint")
    if (not isinstance(capacity_fingerprint,str) or
            len(capacity_fingerprint)!=64 or
            any(value not in "0123456789abcdef" for value in capacity_fingerprint)):
        raise ValueError("Q3-H1 base capacity fingerprint is invalid")
    return _hiding_foundation_binding_document(
        mode=mode,capacity_law=law,fraction_ids=fraction_ids,
        parker_alpha=numerics.get("hiding_parker_alpha",.6),
        wu_exponent=numerics.get("hiding_wu_exponent",-.6),
        capacity_profile_fingerprint=capacity_fingerprint)


def _validated_hiding_foundation_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected=_build_hiding_foundation_binding(project)
    supplied=project.get(HIDING_FOUNDATION_BINDING_KEY)
    if expected is None:
        if supplied is not None:
            raise ValueError(
                "Q3-H1 hiding binding is present while generic hiding is disabled")
        return None
    if supplied is None:
        return expected
    try:
        supplied=_validate_hiding_foundation_binding_document(supplied)
    except ValueError as exc:
        raise ValueError(
            "Q3-H1 hiding project binding is stale, forged, or disagrees with project inputs") from exc
    if supplied!=expected:
        raise ValueError(
            "Q3-H1 hiding project binding is stale, forged, or disagrees with project inputs")
    return expected


def _secondary_roughness_method(project: Mapping[str, Any]) -> str:
    roughness=project.get("roughness",{})
    if not isinstance(roughness,Mapping):
        raise ValueError("Q3-H2 requires an explicit typed roughness declaration")
    mode=roughness.get("mode","manning")
    method={
        "chezy":"roughness.chezy",
        "manning":"roughness.manning",
        "white_colebrook":(
            "roughness.white_colebrook_waqua"
            if roughness.get("white_colebrook_chzk2d",False) else
            "roughness.white_colebrook"),
    }.get(mode)
    if method is None:
        raise ValueError("Q3-H2 requires Chezy, Manning, or White-Colebrook roughness")
    return method


def _uses_secondary_foundation(project: Mapping[str, Any]) -> bool:
    """Return whether this project is in the first exact Q3-H2 live slice."""
    internal=project.get("internal_conditions",{})
    secondary=(internal.get("secondary_flow",{"enabled":False})
               if isinstance(internal,Mapping) else {"enabled":False})
    numerics=project.get("numerics",{})
    amr=numerics.get("amr",{}) if isinstance(numerics,Mapping) else {}
    bank=(internal.get("bank_morphology",{})
          if isinstance(internal,Mapping) else {})
    bank_failure=(internal.get("bank_failure",{})
                  if isinstance(internal,Mapping) else {})
    general_avalanche=(internal.get("general_avalanche",{})
                       if isinstance(internal,Mapping) else {})
    try:
        model=_resolved_secondary_flow_model(
            secondary.get("model","equilibrium"))
        beta=_finite(secondary.get("momentum_beta",0.),"secondary-flow Beta")
        viscosity=(_finite(numerics.get("horizontal_viscosity_m2_s",0.),
                           "horizontal viscosity",nonnegative=True)
                   if isinstance(numerics,Mapping) else float("nan"))
    except (AttributeError,TypeError,ValueError):
        return False
    return bool(
        isinstance(secondary,Mapping) and secondary.get("enabled",False) and
        model=="equilibrium" and
        secondary.get("transport_direction",True) is True and beta==0. and
        isinstance(numerics,Mapping) and
        numerics.get("sediment_enabled",True) is True and
        type(numerics.get("bedload")) is int and
        numerics.get("bedload") in (2,8,11) and
        numerics.get("exchange")==0 and numerics.get("bed_slope")==0 and
        viscosity==0. and
        not (isinstance(amr,Mapping) and amr.get("enabled",False)) and
        isinstance(bank,Mapping) and not bank.get("enabled",False) and
        isinstance(bank_failure,Mapping) and
        not bank_failure.get("enabled",False) and
        not bank_failure.get("front_tracking_enabled",False) and
        isinstance(general_avalanche,Mapping) and
        not general_avalanche.get("enabled",False) and
        project.get("sediment_shear",{}).get("mode","total")=="total" and
        not internal.get("bedload_adaptation",{}).get("enabled",False) and
        not internal.get("bedform_roughness",{}).get("enabled",False))


def _secondary_foundation_binding_document(
    *,capacity_law: int,capacity_profile_fingerprint: str,
    hiding_binding_fingerprint: str | None,roughness_method_id: str,
    rmincf: float,chzmin: float,espir: float,betac: float,
) -> dict[str, Any]:
    """Return the project-owned declaration whose forcing is prepare-time data."""
    if type(capacity_law) is not int or capacity_law not in (2,8,11):
        raise ValueError("Q3-H2 supports capacity laws 2, 8, or 11")
    for value,name in ((capacity_profile_fingerprint,"capacity profile"),
                       (hiding_binding_fingerprint,"hiding binding")):
        if value is not None and (not isinstance(value,str) or len(value)!=64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"Q3-H2 {name} fingerprint is invalid")
    if hiding_binding_fingerprint is not None and capacity_law not in (2,11):
        raise ValueError("Q3-H2 law 8 cannot wrap Q3-H1 hiding")
    controls={
        "rmincf":_finite(rmincf,"Q3-H2 Rmincf",positive=True),
        "chzmin":_finite(chzmin,"Q3-H2 Chzmin",positive=True),
        "espir":_finite(espir,"Q3-H2 Espir",nonnegative=True),
        "betac":_finite(betac,"Q3-H2 Betac"),
    }
    body={
        "schema":SECONDARY_FOUNDATION_BINDING_SCHEMA,
        "scope_id":SECONDARY_FOUNDATION_BINDING_SCOPE_ID,
        "semantic_id":SECONDARY_FOUNDATION_BINDING_SEMANTIC_ID,
        "state":"deferred_prepared_mesh_typed_forcing",
        "execution_scope_id":SECONDARY_FOUNDATION_SCOPE_ID,
        "execution_semantic_id":SECONDARY_FOUNDATION_SEMANTIC_ID,
        "method_id":SECONDARY_FOUNDATION_METHOD_ID,
        "model":1,
        "capacity_law":capacity_law,
        "capacity_profile_fingerprint":capacity_profile_fingerprint,
        "base_envelope_version":55 if hiding_binding_fingerprint is not None else 54,
        "hiding_binding_fingerprint":hiding_binding_fingerprint,
        "roughness_method_id":roughness_method_id,
        "coriolis_disposition":"zero",
        "coriolis_constant_s_inverse":0.0,
        "coriolis_source":"constant_zero_planar",
        **controls,
        "base_secondary_method_id":"secondary.off",
        "composition_order":list(SECONDARY_FOUNDATION_COMPOSITION_ORDER),
        "preparation_requirement":(
            "bind canonical Q3-C/Q3-H1 execution, typed forcing, and fixed-grid mesh"),
        "backend_selection":"explicit_user_choice",
        "automatic_cpu_fallback":False,
    }
    return {**body,"fingerprint":hashlib.sha256(
        _canonical_json_bytes(body)).hexdigest()}


def _validate_secondary_foundation_binding_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value,Mapping):
        raise ValueError("Q3-H2 secondary project binding must be an object")
    try:
        rebuilt=_secondary_foundation_binding_document(
            capacity_law=value["capacity_law"],
            capacity_profile_fingerprint=value["capacity_profile_fingerprint"],
            hiding_binding_fingerprint=value["hiding_binding_fingerprint"],
            roughness_method_id=value["roughness_method_id"],
            rmincf=value["rmincf"],chzmin=value["chzmin"],
            espir=value["espir"],betac=value["betac"])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError("Invalid Q3-H2 secondary project binding") from exc
    if _canonical_json_bytes(value)!=_canonical_json_bytes(rebuilt):
        raise ValueError("Q3-H2 secondary project binding is stale or forged")
    return rebuilt


def _build_secondary_foundation_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    secondary=project.get("internal_conditions",{}).get(
        "secondary_flow",{"enabled":False})
    if not isinstance(secondary,Mapping) or not secondary.get("enabled",False):
        return None
    model=_resolved_secondary_flow_model(secondary.get("model","equilibrium"))
    if model!="equilibrium" or not _uses_secondary_foundation(project):
        return None
    numerics=project.get("numerics",{})
    law=numerics.get("bedload")
    capacity=project.get(CAPACITY_BINDING_KEY,{}).get(
        "capacity_runtime_profile")
    if not isinstance(capacity,Mapping):
        raise ValueError("Q3-H2 requires the canonical Q3-C capacity component")
    if capacity.get("secondary_flow",{}).get("method_id")!="secondary.off":
        raise ValueError("Q3-H2 nested Q3-C capacity must retain secondary.off")
    hiding=project.get(HIDING_FOUNDATION_BINDING_KEY)
    if hiding is not None:
        hiding=_validate_hiding_foundation_binding_document(hiding)
        if hiding["capacity_law"]!=law:
            raise ValueError("Q3-H2 hiding and capacity laws differ")
    return _secondary_foundation_binding_document(
        capacity_law=law,
        capacity_profile_fingerprint=capacity["fingerprint"],
        hiding_binding_fingerprint=(None if hiding is None else hiding["fingerprint"]),
        roughness_method_id=_secondary_roughness_method(project),
        rmincf=secondary.get("rmincf",2.5),
        chzmin=secondary.get("chzmin",20.0),
        espir=secondary.get("espir",1.0),
        betac=secondary.get("momentum_beta",0.0))


def _validated_secondary_foundation_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected=_build_secondary_foundation_binding(project)
    supplied=project.get(SECONDARY_FOUNDATION_BINDING_KEY)
    if expected is None:
        if supplied is not None:
            raise ValueError(
                "Q3-H2 secondary binding is present while classic equilibrium secondary flow is disabled")
        return None
    if supplied is None:return expected
    try:supplied=_validate_secondary_foundation_binding_document(supplied)
    except ValueError as exc:
        raise ValueError(
            "Q3-H2 secondary project binding is stale, forged, or disagrees with project inputs") from exc
    if supplied!=expected:
        raise ValueError(
            "Q3-H2 secondary project binding is stale, forged, or disagrees with project inputs")
    return expected


def _uses_prognostic_secondary_foundation(project: Mapping[str, Any]) -> bool:
    """Return whether this project is in the first exact Q3-H3 live slice."""
    internal=project.get("internal_conditions",{})
    secondary=(internal.get("secondary_flow",{"enabled":False})
               if isinstance(internal,Mapping) else {"enabled":False})
    numerics=project.get("numerics",{})
    amr=numerics.get("amr",{}) if isinstance(numerics,Mapping) else {}
    rows=project.get("sediment_fractions")
    hiding=(numerics.get("hiding_exposure",0)
            if isinstance(numerics,Mapping) else None)
    try:
        model=_resolved_secondary_flow_model(
            secondary.get("model","equilibrium"))
        method_id=secondary.get("method_id")
        provenance=secondary.get("method_provenance")
        betac=_finite(secondary.get("classic_betac",0.),
                      "Q3-H3 classic Betac")
        legacy_beta=_finite(secondary.get("momentum_beta",0.),
                            "legacy prognostic momentum Beta",nonnegative=True)
        scalar_diffusivity=_finite(
            secondary.get("scalar_diffusivity_m2_s",0.),
            "Q3-H3 scalar diffusivity",nonnegative=True)
        viscosity=_finite(numerics.get("horizontal_viscosity_m2_s",0.),
                          "horizontal viscosity",nonnegative=True)
    except (AttributeError,TypeError,ValueError):
        return False
    exact_fraction_scope=(
        (hiding==0 and rows is None) or
        (hiding in HIDING_FOUNDATION_METHOD_BY_MODE and
         isinstance(rows,(list,tuple)) and len(rows)==2))
    return bool(
        isinstance(secondary,Mapping) and secondary.get("enabled",False) and
        model=="prognostic" and
        method_id==PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID and
        provenance==PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE and
        secondary.get("transport_direction",True) is True and betac==0. and
        legacy_beta==0. and scalar_diffusivity==0. and
        isinstance(numerics,Mapping) and
        numerics.get("sediment_enabled",True) is True and
        type(numerics.get("bedload")) is int and
        numerics.get("bedload") in (2,8,11) and
        numerics.get("exchange")==0 and numerics.get("bed_slope")==0 and
        viscosity==0. and exact_fraction_scope and
        (hiding==0 or numerics.get("bedload") in (2,11)) and
        not (isinstance(amr,Mapping) and amr.get("enabled",False)) and
        project.get("sediment_shear",{}).get("mode","total")=="total" and
        not internal.get("bedload_adaptation",{}).get("enabled",False) and
        not internal.get("bedform_roughness",{}).get("enabled",False))


def _prognostic_secondary_binding_document(
    *,capacity_law: int,capacity_profile_fingerprint: str,
    hiding_binding_fingerprint: str | None,roughness_method_id: str,
    fraction_ids: Sequence[str],bed_layer_count: int,
    transport_direction: bool,inflow_mode: int,
    rmincf: float,chzmin: float,espir: float,betac: float,
    scalar_diffusivity_m2_s: float,von_karman_constant: float,
) -> dict[str, Any]:
    """Return the deferred project declaration for additive Q3-H3/v57."""
    if type(capacity_law) is not int or capacity_law not in (2,8,11):
        raise ValueError("Q3-H3 supports capacity laws 2, 8, or 11")
    for value,name in ((capacity_profile_fingerprint,"capacity profile"),
                       (hiding_binding_fingerprint,"hiding binding")):
        if value is not None and (not isinstance(value,str) or len(value)!=64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"Q3-H3 {name} fingerprint is invalid")
    if hiding_binding_fingerprint is not None and capacity_law not in (2,11):
        raise ValueError("Q3-H3 law 8 cannot wrap Q3-H1 hiding")
    if (not isinstance(fraction_ids,Sequence) or
            isinstance(fraction_ids,(str,bytes)) or not fraction_ids or
            len(set(fraction_ids))!=len(fraction_ids) or
            any(not isinstance(value,str) or not value for value in fraction_ids)):
        raise ValueError("Q3-H3 requires ordered unique fraction identities")
    if type(bed_layer_count) is not int or bed_layer_count<1:
        raise ValueError("Q3-H3 bed-layer count must be a positive integer")
    if transport_direction is not True:
        raise ValueError("First Q3-H3 live scope requires transported-state direction coupling")
    if type(inflow_mode) is not int or inflow_mode!=PROGNOSTIC_SECONDARY_INFLOW_MODE:
        raise ValueError("Q3-H3 requires adjacent-interior equilibrium inflow mode 1")
    controls={
        "rmincf":_finite(rmincf,"Q3-H3 Rmincf",positive=True),
        "chzmin":_finite(chzmin,"Q3-H3 Chzmin",positive=True),
        "espir":_finite(espir,"Q3-H3 Espir",nonnegative=True),
        "betac":_finite(betac,"Q3-H3 classic Betac"),
        "scalar_diffusivity_m2_s":_finite(
            scalar_diffusivity_m2_s,"Q3-H3 scalar diffusivity",nonnegative=True),
        "von_karman_constant":_finite(
            von_karman_constant,"Q3-H3 von Karman constant",positive=True),
    }
    body={
        "schema":PROGNOSTIC_SECONDARY_BINDING_SCHEMA,
        "scope_id":PROGNOSTIC_SECONDARY_BINDING_SCOPE_ID,
        "semantic_id":PROGNOSTIC_SECONDARY_BINDING_SEMANTIC_ID,
        "state":"deferred_prepared_mesh_typed_forcing_and_runtime_layout",
        "execution_scope_id":PROGNOSTIC_SECONDARY_SCOPE_ID,
        "execution_semantic_id":PROGNOSTIC_SECONDARY_SEMANTIC_ID,
        "method_id":PROGNOSTIC_SECONDARY_METHOD_ID,
        "model":2,
        "capacity_law":capacity_law,
        "capacity_profile_fingerprint":capacity_profile_fingerprint,
        "base_envelope_version":55 if hiding_binding_fingerprint is not None else 54,
        "hiding_binding_fingerprint":hiding_binding_fingerprint,
        "roughness_method_id":roughness_method_id,
        "ordered_fraction_ids":list(fraction_ids),
        "bed_layer_count":bed_layer_count,
        "transport_direction":transport_direction,
        "inflow_mode":inflow_mode,
        "coriolis_disposition":"zero",
        "coriolis_constant_s_inverse":0.0,
        "coriolis_source":"constant_zero_planar",
        **controls,
        "gpu_checkpoint_schema":PROGNOSTIC_SECONDARY_GPU_CHECKPOINT_SCHEMA,
        "base_secondary_method_id":"secondary.off",
        "equilibrium_rotation_executed":False,
        "composition_order":list(PROGNOSTIC_SECONDARY_COMPOSITION_ORDER),
        "preparation_requirement":(
            "bind canonical Q3-C/Q3-H1 execution, typed forcing, runtime layout, and fixed-grid mesh"),
        "backend_selection":"explicit_user_choice",
        "automatic_cpu_fallback":False,
    }
    return {**body,"fingerprint":hashlib.sha256(
        _canonical_json_bytes(body)).hexdigest()}


def _validate_prognostic_secondary_binding_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value,Mapping):
        raise ValueError("Q3-H3 prognostic-secondary project binding must be an object")
    try:
        rebuilt=_prognostic_secondary_binding_document(
            capacity_law=value["capacity_law"],
            capacity_profile_fingerprint=value["capacity_profile_fingerprint"],
            hiding_binding_fingerprint=value["hiding_binding_fingerprint"],
            roughness_method_id=value["roughness_method_id"],
            fraction_ids=value["ordered_fraction_ids"],
            bed_layer_count=value["bed_layer_count"],
            transport_direction=value["transport_direction"],
            inflow_mode=value["inflow_mode"],
            rmincf=value["rmincf"],chzmin=value["chzmin"],
            espir=value["espir"],betac=value["betac"],
            scalar_diffusivity_m2_s=value["scalar_diffusivity_m2_s"],
            von_karman_constant=value["von_karman_constant"])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError("Invalid Q3-H3 prognostic-secondary project binding") from exc
    if _canonical_json_bytes(value)!=_canonical_json_bytes(rebuilt):
        raise ValueError("Q3-H3 prognostic-secondary project binding is stale or forged")
    return rebuilt


def _build_prognostic_secondary_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    secondary=project.get("internal_conditions",{}).get(
        "secondary_flow",{"enabled":False})
    if not isinstance(secondary,Mapping) or not secondary.get("enabled",False):
        return None
    if (_resolved_secondary_flow_model(secondary.get("model","equilibrium"))!=
            "prognostic"):
        return None
    # Historical prognostic projects did not carry a method declaration and
    # therefore remain on their byte-stable v37 D-Flow-FM/BetaSpiral route.
    # A classic Delft3D4 request is explicit, and can never silently fall back
    # to that different formulation when an H3 admission restriction fails.
    method_id=secondary.get("method_id")
    provenance=secondary.get("method_provenance")
    if method_id is None and provenance is None:
        return None
    if (method_id!=PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID or
            provenance!=PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE):
        raise ValueError(
            "Prognostic secondary-flow method/provenance is not a supported "
            "classic Delft3D4 FLOW declaration")
    if not _uses_prognostic_secondary_foundation(project):
        raise ValueError(
            "Classic Delft3D4 prognostic secondary flow is outside the first "
            "Q3-H3 live scope; it cannot fall back to the legacy v37 route")
    numerics=project.get("numerics",{})
    law=numerics.get("bedload")
    capacity=project.get(CAPACITY_BINDING_KEY,{}).get("capacity_runtime_profile")
    if not isinstance(capacity,Mapping):
        raise ValueError("Q3-H3 requires the canonical Q3-C capacity component")
    if capacity.get("secondary_flow",{}).get("method_id")!="secondary.off":
        raise ValueError("Q3-H3 nested Q3-C capacity must retain secondary.off")
    hiding=project.get(HIDING_FOUNDATION_BINDING_KEY)
    if hiding is not None:
        hiding=_validate_hiding_foundation_binding_document(hiding)
        if hiding["capacity_law"]!=law:
            raise ValueError("Q3-H3 hiding and capacity laws differ")
    fraction_ids,_=_capacity_fraction_layout(project)
    initial=project.get("initial",{})
    return _prognostic_secondary_binding_document(
        capacity_law=law,
        capacity_profile_fingerprint=capacity["fingerprint"],
        hiding_binding_fingerprint=(None if hiding is None else hiding["fingerprint"]),
        roughness_method_id=_secondary_roughness_method(project),
        fraction_ids=fraction_ids,
        bed_layer_count=(
            1+int(initial.get("substrate_layer_count",1))
            if float(initial.get("active_layer_thickness_m",0.))>0. else 1),
        transport_direction=secondary.get("transport_direction",True),
        inflow_mode=PROGNOSTIC_SECONDARY_INFLOW_MODE,
        rmincf=secondary.get("rmincf",2.5),
        chzmin=secondary.get("chzmin",20.0),
        espir=secondary.get("espir",1.0),
        betac=secondary.get("classic_betac",0.0),
        scalar_diffusivity_m2_s=secondary.get(
            "scalar_diffusivity_m2_s",0.0),
        von_karman_constant=.41)


def _validated_prognostic_secondary_binding(
    project: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected=_build_prognostic_secondary_binding(project)
    supplied=project.get(PROGNOSTIC_SECONDARY_BINDING_KEY)
    if expected is None:
        if supplied is not None:
            raise ValueError(
                "Q3-H3 prognostic-secondary binding is present outside its exact live scope")
        return None
    if supplied is None:return expected
    try:supplied=_validate_prognostic_secondary_binding_document(supplied)
    except ValueError as exc:
        raise ValueError(
            "Q3-H3 prognostic-secondary project binding is stale, forged, or disagrees with project inputs") from exc
    if supplied!=expected:
        raise ValueError(
            "Q3-H3 prognostic-secondary project binding is stale, forged, or disagrees with project inputs")
    return expected


def _capacity_profile_methods(project: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    """Translate only method identities represented by the canonical profile.

    BRAIDS legacy slope code 2 is intentionally absent: it is not Delft3D
    ``ISlope=2`` and must remain an explicitly unprofiled legacy route.
    Secondary flow enters this profile only when it steers capacity direction;
    a momentum-only prognostic state therefore resolves to ``secondary.off``.
    """
    numerics = project["numerics"]
    shear = project.get("sediment_shear", {"mode": "total"}).get("mode", "total")
    hiding = numerics.get("hiding_exposure", 0)
    slope = numerics.get("bed_slope", 0)
    if shear not in _PROFILE_SHEAR_METHOD:
        return None, "unrepresented_sediment_shear"
    if hiding not in _PROFILE_HIDING_METHOD:
        return None, "unrepresented_hiding_method"
    if slope not in _PROFILE_SLOPE_METHOD:
        return None, ("braids_legacy_bed_slope" if slope == 2 else
                      "unrepresented_bed_slope")
    secondary = project.get("internal_conditions", {}).get(
        "secondary_flow", {"enabled": False})
    secondary_method = "secondary.off"
    if secondary.get("enabled", False) and secondary.get("transport_direction", True):
        model = _resolved_secondary_flow_model(secondary.get("model", "equilibrium"))
        # Classic equilibrium is an additive Q3-H2 accepted-state operator.
        # Its nested Q3-C (and optional Q3-H1) foundation remains exactly
        # secondary.off; the separately authenticated v56 envelope owns the
        # direction rotation.  Prognostic secondary flow is still represented
        # directly by the older execution profile.
        secondary_method = ("secondary.off" if
                            ((model=="equilibrium" and
                              _uses_secondary_foundation(project)) or
                             (model=="prognostic" and
                              _uses_prognostic_secondary_foundation(project))) else
                            "secondary.equilibrium" if model=="equilibrium" else
                            "secondary.prognostic")
    # Q3-H1 is an additive accepted-state operator.  Its nested Q3-C capacity
    # component must stay byte/semantically equivalent to the accepted
    # hiding.none foundation; the separately authenticated outer declaration
    # owns the selected mode and controls.
    profile_hiding=(0 if hiding in HIDING_FOUNDATION_METHOD_BY_MODE else hiding)
    return {
        "sediment_shear": _PROFILE_SHEAR_METHOD[shear],
        "hiding": _PROFILE_HIDING_METHOD[profile_hiding],
        "bed_slope": _PROFILE_SLOPE_METHOD[slope],
        "secondary_flow": secondary_method,
    }, None


def _legacy_capacity_parameters(
    project: Mapping[str, Any], formula_code: int,
    fraction_ids: Sequence[str], materials: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float | int]] | None, str | None]:
    """Return source-keyword rows exactly equivalent to the legacy runtime.

    This is a migration description, not a new numerical default.  Where the
    legacy runtime has a state-dependent coefficient, or an extra multiplier
    absent from the locked FLOW formula contract, it remains unprofiled.
    """
    calibration = float(project["numerics"].get("transport_calibration_factor", 1.))

    def every(values: Mapping[str, float | int]) -> dict[str, dict[str, float | int]]:
        return {fraction_id: dict(values) for fraction_id in fraction_ids}

    if formula_code == 1:
        return every({}), None
    if formula_code == 2:
        # Both established MPM closures have the fixed FLOW calibration one;
        # their distinct shear identity remains outside ACal.
        return every({"ACal": 1.}), None
    if formula_code == 3:
        return every({
            "ACal": 17. * calibration,
            "ThetaC": .05,
            "PowerM": 1.5,
            "PowerP": 1.,
            "PowerQ": 1.,
        }), None
    if formula_code == 4:
        if any(row.get("roughness_mode", "fixed") != "fixed" for row in materials):
            return None, "law4_dynamic_sediment_roughness"
        return {
            fraction_id: {
                "ACal": calibration,
                "ModInd": 1,
                "RatioD50Z0": float(material["d50_m"]) / float(material["z0_m"]),
            }
            for fraction_id, material in zip(fraction_ids, materials)
        }, None
    if formula_code == 5:
        return every({"ACal": calibration}), None
    if formula_code == 6:
        if calibration != 1.:
            return None, "law6_legacy_calibration_not_in_formula_contract"
        return every({"ThetaC0": .021, "Alpha0": .33}), None
    if formula_code == 7:
        if calibration != 1.:
            return None, "law7_legacy_calibration_not_in_formula_contract"
        return every({"ThetaC0": .03, "Alpha0": .30}), None
    if formula_code == 8:
        return every({"ACal": calibration, "RouKs": 1., "SusFac": 0.}), None
    return None, "formula_not_in_capacity_profile"


def _profile_parameter_rows(document: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    return {
        row["fraction_id"]: {
            coefficient["keyword"]: coefficient["value"]
            for coefficient in row["coefficients"]
        }
        for row in document["fractions"]
    }


def _split_output_contract(document: Mapping[str, Any], formula_code: int) -> str:
    if formula_code not in SPLIT_CAPACITY_LAWS:
        return "bed_only"
    computed = any(
        row["output"]["suspended_channel"] != SUSPENDED_CHANNEL_ZERO
        for row in document["fractions"]
    )
    return "retain_bed_and_suspended" if computed else "bed_only"


def _inactive_capacity_binding() -> dict[str, Any]:
    return {
        "schema": CAPACITY_BINDING_SCHEMA,
        "state": "inactive",
        "formula_code": 0,
        "ordered_fraction_ids": [],
        "capacity_parameters_by_fraction": {},
        "capacity_output_contract": "bed_only",
        "capacity_runtime_profile": None,
        "inactive_identity": {
            "method_id": "capacity.off",
            "source_id": "none",
            "semantic_catalog_id": semantic_catalog_id(),
            "capacity_profile_semantic_id": capacity_profile_semantic_id(),
        },
    }


def _legacy_unprofiled_capacity_binding(
    formula_code: int, fraction_ids: Sequence[str], reason_code: str,
) -> dict[str, Any]:
    return {
        "schema": CAPACITY_BINDING_SCHEMA,
        "state": ("legacy_dynamic_unprofiled"
                  if reason_code == "law4_dynamic_sediment_roughness"
                  else "legacy_unprofiled"),
        "formula_code": formula_code,
        "ordered_fraction_ids": list(fraction_ids),
        "capacity_parameters_by_fraction": None,
        "capacity_output_contract": "bed_only",
        "capacity_runtime_profile": None,
        "reason_code": reason_code,
    }


def _resolve_project_capacity_profile(
    project: Mapping[str, Any], parameters: Mapping[str, Any], *,
    backend: str | None,
):
    numerics = project["numerics"]
    fraction_ids, _ = _capacity_fraction_layout(project)
    methods, reason = _capacity_profile_methods(project)
    if methods is None:
        raise CapacityProfileError(
            f"Project methods cannot be represented by the canonical capacity profile: {reason}",
            category="scope_restriction",
        )
    return resolve_capacity_profile_from_code(
        numerics["bedload"], fraction_ids, parameters,
        backend=backend, method_id=numerics.get("capacity_method_id"), **methods,
    )


def _build_capacity_binding(
    project: Mapping[str, Any],
    supplied_parameters: Mapping[str, Any] | None,
    *, require_split_parameters: bool,
) -> dict[str, Any]:
    numerics = project["numerics"]
    formula_code = numerics["bedload"] if numerics.get("sediment_enabled", True) else 0
    if formula_code == 0:
        if supplied_parameters not in (None, {}):
            raise ValueError("Capacity parameters are invalid when bed load is inactive")
        return _inactive_capacity_binding()

    fraction_ids, materials = _capacity_fraction_layout(project)
    methods, method_reason = _capacity_profile_methods(project)
    legacy_parameters, legacy_reason = _legacy_capacity_parameters(
        project, formula_code, fraction_ids, materials)

    if formula_code in SPLIT_CAPACITY_LAWS:
        if supplied_parameters is None:
            if require_split_parameters:
                raise ValueError(
                    f"Capacity law {formula_code} requires exact source-keyword "
                    "parameters for every sediment fraction")
            return _legacy_unprofiled_capacity_binding(
                formula_code, fraction_ids, "missing_capacity_parameters")
        parameters = supplied_parameters
    else:
        if legacy_parameters is None:
            if supplied_parameters is not None:
                raise ValueError(
                    "This legacy capacity route cannot be represented by fixed "
                    "source-keyword parameters")
            return _legacy_unprofiled_capacity_binding(
                formula_code, fraction_ids, legacy_reason or "unrepresented_legacy_formula")
        parameters = legacy_parameters if supplied_parameters is None else supplied_parameters

    if methods is None:
        if formula_code in SPLIT_CAPACITY_LAWS:
            raise ValueError(
                "The selected capacity auxiliaries are outside the canonical profile: " +
                str(method_reason))
        return _legacy_unprofiled_capacity_binding(
            formula_code, fraction_ids, method_reason or "unrepresented_legacy_methods")

    try:
        profile = _resolve_project_capacity_profile(project, parameters, backend=None)
        document = capacity_profile_document(profile)
        corrected_scalar_method=numerics.get("capacity_method_id") in (
            MPM_DELFT_METHOD_ID,EH_DELFT_METHOD_ID)
        if (formula_code not in SPLIT_CAPACITY_LAWS and
                supplied_parameters is not None and
                not corrected_scalar_method):
            legacy_profile = _resolve_project_capacity_profile(
                project, legacy_parameters or {}, backend=None)
            if (_profile_parameter_rows(document) !=
                    _profile_parameter_rows(capacity_profile_document(legacy_profile))):
                raise ValueError(
                    "Explicit capacity parameters do not match the established legacy runtime")
    except CapacityProfileError as exc:
        if formula_code in SPLIT_CAPACITY_LAWS:
            raise ValueError(str(exc)) from exc
        return _legacy_unprofiled_capacity_binding(
            formula_code, fraction_ids, f"capacity_profile_{exc.category}")

    return {
        "schema": CAPACITY_BINDING_SCHEMA,
        "state": "canonical",
        "formula_code": formula_code,
        "ordered_fraction_ids": list(fraction_ids),
        "capacity_parameters_by_fraction": _profile_parameter_rows(document),
        "capacity_output_contract": _split_output_contract(document, formula_code),
        "capacity_runtime_profile": document,
    }


def _validated_capacity_binding(project: Mapping[str, Any]) -> dict[str, Any]:
    binding = project.get(CAPACITY_BINDING_KEY)
    if not isinstance(binding, Mapping):
        raise ValueError("BRAIDS v2 project is missing its capacity-profile binding")
    if binding.get("schema") != CAPACITY_BINDING_SCHEMA:
        raise ValueError("Unsupported capacity-profile binding schema")
    state = binding.get("state")
    supplied = (binding.get("capacity_parameters_by_fraction")
                if state == "canonical" else None)
    expected = _build_capacity_binding(
        project, supplied, require_split_parameters=False)
    if dict(binding) != expected:
        # Load the document separately so stale fingerprints and semantic IDs
        # retain the capacity-profile module's precise fail-closed diagnosis.
        document = binding.get("capacity_runtime_profile")
        if document is not None:
            try:
                load_capacity_profile(document)
            except CapacityProfileError as exc:
                raise ValueError(str(exc)) from exc
        raise ValueError(
            "Capacity-profile binding is stale, forged, or disagrees with project inputs")
    if state == "canonical":
        try:
            profile = load_capacity_profile(binding["capacity_runtime_profile"])
        except CapacityProfileError as exc:
            raise ValueError(str(exc)) from exc
        if profile.backend is not None:
            raise ValueError("Project capacity profiles must remain backend-independent")
    return expected


def _prepared_capacity_document(project: Mapping[str, Any], backend: str) -> dict[str, Any] | None:
    binding = project[CAPACITY_BINDING_KEY]
    if binding["state"] == "inactive":
        return None
    if binding["state"] != "canonical":
        if binding["formula_code"] in SPLIT_CAPACITY_LAWS:
            raise ValueError(
                "This migrated split-capacity project has no exact coefficient profile; "
                "edit and save complete source-keyword parameters before preparation")
        return None
    try:
        profile = _resolve_project_capacity_profile(
            project, binding["capacity_parameters_by_fraction"], backend=backend)
        return capacity_profile_document(profile)
    except CapacityProfileError as exc:
        raise ValueError(
            f"The selected {backend} capacity route is not available: {exc}") from exc


def _neutral_execution_capacity_document(
    project: Mapping[str, Any], backend_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact neutral capacity component behind a CPU admission.

    Capacity profiles include their selected backend in the fingerprint.  The
    CPU execution profile must bind that backend-specific document, while the
    native v52 wire deliberately remains backend neutral.  Require the two
    documents to differ only by backend/fingerprint before writing either
    identity so a CPU admission cannot change physical parameters behind the
    neutral project profile.
    """
    binding = project[CAPACITY_BINDING_KEY]
    try:
        backend_profile = load_capacity_profile(backend_document)
        neutral_profile = _resolve_project_capacity_profile(
            project, binding["capacity_parameters_by_fraction"], backend=None)
        checked_backend = capacity_profile_document(backend_profile)
        neutral_document = capacity_profile_document(neutral_profile)
    except (CapacityProfileError, KeyError, TypeError) as exc:
        raise ValueError(
            f"CPU and neutral capacity component identities are invalid: {exc}") from exc
    if backend_profile.backend != "cpu" or neutral_profile.backend is not None:
        raise ValueError(
            "CPU admission and native v52 capacity components have invalid backends")

    def physical(document: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in document.items()
            if key not in {"backend", "fingerprint"}
        }

    if physical(checked_backend) != physical(neutral_document):
        raise ValueError(
            "CPU and neutral capacity components describe different physics")
    return neutral_document


def _capacity_execution_method_id(numerics: Mapping[str, Any]) -> str | None:
    """Resolve the physical capacity closure without granting route support."""
    enabled = numerics.get("sediment_enabled", True)
    law = numerics.get("bedload", 0) if enabled else 0
    if isinstance(law, bool) or not isinstance(law, int):
        raise ValueError("Bed-load law must be an integer")
    explicit = numerics.get("capacity_method_id")
    if law not in range(9):
        if explicit is not None:
            raise ValueError(
                "An explicit Q2 capacity identity is valid only for bed-load laws 0 to 8")
        return None
    try:
        return resolve_sediment_methods(
            law, slope_enabled=False, slope_method=0,
            exchange_mode=numerics.get("exchange", 0) if enabled else 0,
            mpm_friction_mode=(numerics.get("mpm_friction_mode", 0)
                               if enabled and law == 2 else 0),
            capacity_method_id=explicit,
        ).capacity_method_id
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _build_capacity_execution_binding(
    project: Mapping[str, Any], *, legacy_missing_method_identity: bool = False,
) -> dict[str, Any]:
    """Build the project-level declaration for the frozen Q2 identity.

    A corrected Delft identity depends on the exact per-cell roughness bytes,
    which do not exist until preparation.  The project therefore records an
    authenticated semantic declaration and defers only that byte binding.
    Historical closures are fully resolvable at project-save time.
    """
    numerics = project["numerics"]
    enabled = bool(numerics.get("sediment_enabled", True))
    law = int(numerics["bedload"]) if enabled else 0
    method_id = _capacity_execution_method_id(numerics)
    common = {
        "schema": CAPACITY_EXECUTION_BINDING_SCHEMA,
        "scope_id": CAPACITY_EXECUTION_SCOPE_ID,
        "semantic_id": capacity_execution_semantic_id(),
        "formula_code": law,
        "method_id": method_id,
        "native_deck_method_identity_serialized": False,
    }
    if law not in (2, 8):
        return {
            **common,
            "state": "not_applicable",
            "selection_origin": "explicit" if method_id is not None else None,
            "execution_identity": None,
            "preparation_requirement": None,
        }
    if method_id not in _Q2_CAPACITY_METHODS:
        raise ValueError("Unknown Q2 capacity execution identity")
    origin = ("legacy_missing_method_identity"
              if legacy_missing_method_identity else "explicit")
    if (method_id==EH_DELFT_METHOD_ID and
            "engelund_hansen_chezy" in numerics):
        raise ValueError(
            "Corrected Delft Engelund-Hansen cannot retain the historical "
            "independent Chezy control")
    if (method_id==MPM_DELFT_METHOD_ID and not legacy_missing_method_identity and
            "mpm_cf" in numerics):
        raise ValueError(
            "Corrected Delft MPM cannot retain the historical specified-Cf control")
    if method_id in _LEGACY_Q2_CAPACITY_METHODS:
        controls = dict(numerics)
        if legacy_missing_method_identity:
            controls.pop("capacity_method_id", None)
        try:
            identity = migrate_legacy_capacity_controls(controls)
            document = capacity_execution_document(identity)
        except CapacityExecutionIdentityError as exc:
            raise ValueError(str(exc)) from exc
        return {
            **common,
            "state": ("resolved_legacy_migration" if legacy_missing_method_identity
                      else "resolved"),
            "selection_origin": origin,
            "execution_identity": document,
            "preparation_requirement": "authenticate host sidecar against prepared deck",
        }
    return {
        **common,
        "state": "deferred_typed_roughness",
        "selection_origin": origin,
        "execution_identity": None,
        "preparation_requirement": (
            "bind canonical C/n/ks bytes and forcing fingerprint after mesh preparation"),
    }


def _validated_capacity_execution_binding(project: Mapping[str, Any]) -> dict[str, Any]:
    binding = project.get(CAPACITY_EXECUTION_BINDING_KEY)
    if not isinstance(binding, Mapping):
        raise ValueError("BRAIDS project is missing its capacity-execution binding")
    if binding.get("schema") != CAPACITY_EXECUTION_BINDING_SCHEMA:
        raise ValueError("Unsupported capacity-execution binding schema")
    migrated = binding.get("selection_origin") == "legacy_missing_method_identity"
    expected = _build_capacity_execution_binding(
        project, legacy_missing_method_identity=migrated)
    if dict(binding) != expected:
        document = binding.get("execution_identity")
        if document is not None:
            try:
                load_capacity_execution_identity(document)
            except CapacityExecutionIdentityError as exc:
                raise ValueError(str(exc)) from exc
        raise ValueError(
            "Capacity-execution binding is stale, forged, or disagrees with project inputs")
    return expected


def _prepared_capacity_execution(
    project: Mapping[str, Any], metadata: Mapping[str, Any],
    roughness_parameter: np.ndarray,
    prepared_capacity: Mapping[str, Any] | None,
    *, q3c_profile_owned: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the exact host identity after raw roughness bytes exist."""
    # The authenticated numerical worker is installed separately and is not
    # part of the QGIS Python path.  Preparation therefore uses the shared,
    # NumPy-only document owner shipped in ``braids_core``.
    from braids_core.sediment.forcing_profile import forcing_binding_from_metadata

    shear = project.get("sediment_shear", {})
    native_skin = {
        "kinematic_viscosity_m2_s": float(shear.get("kinematic_viscosity_m2_s", 1.e-6)),
        "sand_roughness_m": float(shear.get("sand_roughness_m", .003)),
        "silt_roughness_m": float(shear.get("silt_roughness_m", .0003)),
        "critical_mud_fraction_low": float(shear.get("critical_mud_fraction_low", .2)),
        "critical_mud_fraction_high": float(shear.get("critical_mud_fraction_high", .8)),
        "fixed_fluff_fraction": float(shear.get("fluff_fraction", 0.)),
    }
    try:
        forcing = forcing_binding_from_metadata(
            metadata, np.ascontiguousarray(roughness_parameter, dtype=np.float64),
            native_skin_controls=native_skin)
    except ValueError as exc:
        raise ValueError(f"Cannot authenticate typed roughness/shear forcing: {exc}") from exc

    binding = project[CAPACITY_EXECUTION_BINDING_KEY]
    method_id = binding.get("method_id")
    execution = None
    state = binding["state"]
    if method_id in _Q2_CAPACITY_METHODS:
        try:
            controls=dict(project["numerics"])
            if binding.get("selection_origin")=="legacy_missing_method_identity":
                controls.pop("capacity_method_id",None)
            identity = migrate_legacy_capacity_controls(
                controls, forcing_binding=forcing)
            execution = capacity_execution_document(
                identity,
                forcing_binding=(forcing if identity.typed_roughness is not None else None))
        except CapacityExecutionIdentityError as exc:
            raise ValueError(str(exc)) from exc
        state = "resolved_host_sidecar"
    route_identity = None
    if method_id in _Q2_CAPACITY_METHODS:
        if execution is None or prepared_capacity is None:
            raise ValueError(
                "A Q2 capacity route requires both execution and capacity-profile identities")
        if method_id==MPM_DELFT_METHOD_ID:
            if prepared_capacity.get("hiding",{}).get("method_id")!="hiding.none":
                raise ValueError(
                    "Corrected Delft MPM generic hiding is not connected in the Q3-0 route")
            for fraction in prepared_capacity.get("fractions",[]):
                coefficients={row.get("keyword"):row.get("value")
                              for row in fraction.get("coefficients",[])}
                if (fraction.get("method_id")!=MPM_DELFT_METHOD_ID or
                        isinstance(coefficients.get("ACal"),bool) or
                        not isinstance(coefficients.get("ACal"),(int,float)) or
                        not np.isfinite(float(coefficients["ACal"])) or
                        float(coefficients["ACal"])<0.):
                    raise ValueError(
                        "Corrected Delft MPM requires an authenticated nonnegative ACal profile")
        route_identity = {
            "ordered_parts": [
                "capacity_execution_fingerprint",
                "capacity_profile_fingerprint",
            ],
            "capacity_execution_fingerprint": execution["fingerprint"],
            "capacity_profile_fingerprint": prepared_capacity["fingerprint"],
        }
    q3c_execution = None
    q3c_route_identity = None
    if q3c_profile_owned and prepared_capacity is not None:
        fractions = prepared_capacity.get("fractions", [])
        methods = {row.get("method_id") for row in fractions
                   if isinstance(row, Mapping)}
        formulas = {row.get("formula_code") for row in fractions
                    if isinstance(row, Mapping)}
        expected = {2: MPM_DELFT_METHOD_ID, 8: EH_DELFT_METHOD_ID,
                    11: GENERAL_METHOD_ID}
        if (len(methods) == 1 and len(formulas) == 1 and
                expected.get(next(iter(formulas))) == next(iter(methods))):
            q3c_execution = capacity_foundation_execution_document(
                method_id=next(iter(methods)),
                formula_code=next(iter(formulas)),
                forcing_fingerprint=forcing["fingerprint"],
                roughness_method_id=forcing["roughness_method_id"],
                capacity_profile_fingerprint=prepared_capacity["fingerprint"])
            q3c_route_identity = {
                "ordered_parts": [
                    "capacity_execution_fingerprint",
                    "capacity_profile_fingerprint",
                ],
                "capacity_execution_fingerprint": q3c_execution["fingerprint"],
                "capacity_profile_fingerprint": prepared_capacity["fingerprint"],
            }
    sidecar = {
        "format": "braids-capacity-execution-sidecar-v1",
        "state": state,
        "project_binding": binding,
        "forcing_binding": forcing,
        "execution_identity": execution,
        "ordered_route_identity": route_identity,
        "capacity_foundation_execution_identity": q3c_execution,
        "capacity_foundation_ordered_route_identity": q3c_route_identity,
        "native_deck_identity_serialized": (
            q3c_route_identity is not None or method_id in _Q2_CAPACITY_METHODS),
        "native_deck_envelope_version": (
            54 if q3c_route_identity is not None else
            50 if method_id in _Q2_CAPACITY_METHODS else None),
        "native_deck_prerequisite": (
            "corrected Delft Engelund-Hansen still requires its live selected-"
            "hydraulic-Chezy kernel" if method_id == EH_DELFT_METHOD_ID else None),
    }
    return sidecar, forcing


def _prepared_hiding_foundation_execution(
    project: Mapping[str, Any],capacity_execution: Mapping[str, Any],
    prepared_capacity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None,dict[str, Any] | None]:
    """Materialize and independently rebuild the canonical outer identity."""
    binding=_validated_hiding_foundation_binding(project)
    if binding is None:return None,None
    base=capacity_execution.get("capacity_foundation_execution_identity")
    base_route=capacity_execution.get(
        "capacity_foundation_ordered_route_identity")
    if not isinstance(base,Mapping) or not isinstance(base_route,Mapping):
        raise ValueError(
            "Q3-H1 hiding requires the authenticated Q3-C foundation identity")
    if (not isinstance(prepared_capacity,Mapping) or
            prepared_capacity.get("hiding",{}).get("method_id")!="hiding.none" or
            prepared_capacity.get("fingerprint")!=
                binding["capacity_profile_fingerprint"]):
        raise ValueError(
            "Q3-H1 prepared capacity must be the nested hiding.none component")
    if base_route!={
            "ordered_parts":["capacity_execution_fingerprint",
                             "capacity_profile_fingerprint"],
            "capacity_execution_fingerprint":base.get("fingerprint"),
            "capacity_profile_fingerprint":prepared_capacity.get("fingerprint")}:
        raise ValueError("Q3-H1 base Q3-C ordered route identity is invalid")
    identity=hiding_foundation_execution_document(
        mode=binding["mode"],method_id=binding["method_id"],
        capacity_law=binding["capacity_law"],
        fraction_ids=binding["fraction_ids"],
        parker_alpha=binding["parker_alpha"],
        wu_exponent=binding["wu_exponent"],
        capacity_execution_fingerprint=base["fingerprint"],
        capacity_profile_fingerprint=prepared_capacity["fingerprint"])
    identity=validate_hiding_foundation_execution_document(identity)
    route={
        "ordered_parts":[
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
        ],
        "capacity_execution_fingerprint":base["fingerprint"],
        "capacity_profile_fingerprint":prepared_capacity["fingerprint"],
        "hiding_execution_fingerprint":identity["fingerprint"],
    }
    return identity,route


def _prepared_secondary_foundation_execution(
    project: Mapping[str, Any],capacity_execution: Mapping[str, Any],
    prepared_capacity: Mapping[str, Any] | None,
    forcing_binding: Mapping[str, Any],
    hiding_identity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None,dict[str, Any] | None]:
    """Materialize v56 after exact capacity/hiding and forcing bytes exist."""
    binding=_validated_secondary_foundation_binding(project)
    if binding is None:return None,None
    capacity=capacity_execution.get("capacity_foundation_execution_identity")
    capacity_route=capacity_execution.get(
        "capacity_foundation_ordered_route_identity")
    if (not isinstance(capacity,Mapping) or
            not isinstance(capacity_route,Mapping) or
            not isinstance(prepared_capacity,Mapping)):
        raise ValueError("Q3-H2 requires an authenticated Q3-C foundation")
    if (capacity.get("fingerprint")!=
            capacity_route.get("capacity_execution_fingerprint") or
            prepared_capacity.get("fingerprint")!=
            capacity_route.get("capacity_profile_fingerprint") or
            binding["capacity_profile_fingerprint"]!=
            prepared_capacity.get("fingerprint")):
        raise ValueError("Q3-H2 Q3-C foundation fingerprints differ")
    hiding_fingerprint=None
    if binding["base_envelope_version"]==55:
        if not isinstance(hiding_identity,Mapping):
            raise ValueError("Q3-H2 v55 base lacks its Q3-H1 execution identity")
        hiding_identity=validate_hiding_foundation_execution_document(
            hiding_identity)
        hiding_fingerprint=hiding_identity["fingerprint"]
        if (hiding_identity["capacity_execution_fingerprint"]!=
                capacity["fingerprint"] or
                hiding_identity["capacity_profile_fingerprint"]!=
                prepared_capacity["fingerprint"]):
            raise ValueError("Q3-H2 Q3-H1 foundation differs from nested Q3-C")
    elif hiding_identity is not None:
        raise ValueError("Q3-H2 v54 base unexpectedly carries Q3-H1 hiding")
    if (not isinstance(forcing_binding,Mapping) or
            forcing_binding.get("roughness_method_id")!=
                binding["roughness_method_id"] or
            not isinstance(forcing_binding.get("fingerprint"),str)):
        raise ValueError("Q3-H2 typed roughness forcing differs")
    identity=secondary_foundation_execution_document(
        method_id=binding["method_id"],model=binding["model"],
        capacity_law=binding["capacity_law"],
        capacity_execution_fingerprint=capacity["fingerprint"],
        capacity_profile_fingerprint=prepared_capacity["fingerprint"],
        hiding_execution_fingerprint=hiding_fingerprint,
        roughness_method_id=binding["roughness_method_id"],
        roughness_fingerprint=forcing_binding["fingerprint"],
        coriolis_disposition=binding["coriolis_disposition"],
        coriolis_constant_s_inverse=binding["coriolis_constant_s_inverse"],
        coriolis_source=binding["coriolis_source"],
        rmincf=binding["rmincf"],chzmin=binding["chzmin"],
        espir=binding["espir"],betac=binding["betac"])
    identity=validate_secondary_foundation_execution_document(identity)
    if identity["base_envelope_version"]!=binding["base_envelope_version"]:
        raise ValueError("Q3-H2 prepared base envelope differs from declaration")
    route={
        "ordered_parts":[
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
            "secondary_execution_fingerprint"],
        "capacity_execution_fingerprint":capacity["fingerprint"],
        "capacity_profile_fingerprint":prepared_capacity["fingerprint"],
        "hiding_execution_fingerprint":hiding_fingerprint,
        "secondary_execution_fingerprint":identity["fingerprint"],
    }
    return identity,route


def _prepared_prognostic_secondary_execution(
    project: Mapping[str, Any],capacity_execution: Mapping[str, Any],
    prepared_capacity: Mapping[str, Any] | None,
    forcing_binding: Mapping[str, Any],
    hiding_identity: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None,dict[str, Any] | None]:
    """Materialize v57 after exact capacity/hiding/forcing bytes exist."""
    binding=_validated_prognostic_secondary_binding(project)
    if binding is None:return None,None
    capacity=capacity_execution.get("capacity_foundation_execution_identity")
    capacity_route=capacity_execution.get(
        "capacity_foundation_ordered_route_identity")
    if (not isinstance(capacity,Mapping) or
            not isinstance(capacity_route,Mapping) or
            not isinstance(prepared_capacity,Mapping)):
        raise ValueError("Q3-H3 requires an authenticated Q3-C foundation")
    if (capacity.get("fingerprint")!=
            capacity_route.get("capacity_execution_fingerprint") or
            prepared_capacity.get("fingerprint")!=
            capacity_route.get("capacity_profile_fingerprint") or
            binding["capacity_profile_fingerprint"]!=
            prepared_capacity.get("fingerprint")):
        raise ValueError("Q3-H3 Q3-C foundation fingerprints differ")
    hiding_fingerprint=None
    if binding["base_envelope_version"]==55:
        if not isinstance(hiding_identity,Mapping):
            raise ValueError("Q3-H3 v55 base lacks its Q3-H1 execution identity")
        hiding_identity=validate_hiding_foundation_execution_document(
            hiding_identity)
        hiding_fingerprint=hiding_identity["fingerprint"]
        if (hiding_identity["capacity_execution_fingerprint"]!=
                capacity["fingerprint"] or
                hiding_identity["capacity_profile_fingerprint"]!=
                prepared_capacity["fingerprint"]):
            raise ValueError("Q3-H3 Q3-H1 foundation differs from nested Q3-C")
    elif hiding_identity is not None:
        raise ValueError("Q3-H3 v54 base unexpectedly carries Q3-H1 hiding")
    if (not isinstance(forcing_binding,Mapping) or
            forcing_binding.get("roughness_method_id")!=
                binding["roughness_method_id"] or
            not isinstance(forcing_binding.get("fingerprint"),str)):
        raise ValueError("Q3-H3 typed roughness forcing differs")
    identity=prognostic_secondary_execution_document(
        method_id=binding["method_id"],model=binding["model"],
        capacity_law=binding["capacity_law"],
        capacity_execution_fingerprint=capacity["fingerprint"],
        capacity_profile_fingerprint=prepared_capacity["fingerprint"],
        hiding_execution_fingerprint=hiding_fingerprint,
        roughness_method_id=binding["roughness_method_id"],
        roughness_fingerprint=forcing_binding["fingerprint"],
        fraction_ids=binding["ordered_fraction_ids"],
        bed_layer_count=binding["bed_layer_count"],
        transport_direction=binding["transport_direction"],
        inflow_mode=binding["inflow_mode"],
        coriolis_disposition=binding["coriolis_disposition"],
        coriolis_constant_s_inverse=binding["coriolis_constant_s_inverse"],
        coriolis_source=binding["coriolis_source"],
        rmincf=binding["rmincf"],chzmin=binding["chzmin"],
        espir=binding["espir"],betac=binding["betac"],
        scalar_diffusivity_m2_s=binding["scalar_diffusivity_m2_s"],
        von_karman_constant=binding["von_karman_constant"],
        gpu_checkpoint_schema=binding["gpu_checkpoint_schema"])
    identity=validate_prognostic_secondary_execution_document(identity)
    if identity["base_envelope_version"]!=binding["base_envelope_version"]:
        raise ValueError("Q3-H3 prepared base envelope differs from declaration")
    route={
        "capacity_execution_fingerprint":capacity["fingerprint"],
        "capacity_profile_fingerprint":prepared_capacity["fingerprint"],
        "hiding_execution_fingerprint":hiding_fingerprint,
        "prognostic_secondary_execution_fingerprint":identity["fingerprint"],
    }
    return identity,route


def _prognostic_secondary_flat_receipt(
    identity: Mapping[str, Any],
) -> dict[str, str]:
    """Expose physical routing and units beside the full v57 preimage."""
    return {
        PROGNOSTIC_SECONDARY_BED_BOUNDARY_POLICY_KEY:
            str(identity["bed_boundary_rotation_policy"]),
        PROGNOSTIC_SECONDARY_STATE_UNITS_KEY:"m2_s",
        PROGNOSTIC_SECONDARY_LEDGER_UNITS_KEY:
            str(identity["integrated_ledger_units"]),
    }


def _capacity_profile_uniform_acal(document: Mapping[str, Any]) -> float:
    """Return the v50 compatibility sentinel without flattening Q3-C.

    Historical v50 identities carried one scalar ``ACal``.  The additive v53
    capacity component is authoritative for Q3-C and binds the complete,
    ordered per-fraction coefficient vector.  Keep the first value in the
    nested v50 slot so old readers retain a deterministic scalar, while never
    requiring the remaining authenticated fractions to equal it.
    """
    values=[]
    for fraction in document.get("fractions",[]):
        coefficients={row.get("keyword"):row.get("value")
                      for row in fraction.get("coefficients",[])}
        value=coefficients.get("ACal")
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError("Capacity profile lacks a numeric ACal coefficient")
        value=float(value)
        if not np.isfinite(value) or value<0.:
            raise ValueError("Capacity profile ACal must be finite and nonnegative")
        values.append(value)
    if not values:
        raise ValueError("Capacity profile lacks an ACal coefficient")
    return values[0]


def _capacity_forcing_metadata_before_deck(
    roughness_mode: str, roughness_settings: Mapping[str, Any],
    roughness_parameter: np.ndarray, shear_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact typed declaration needed to authenticate a v50 deck.

    The forcing fingerprint includes the physical C/n/ks array bytes, so it
    must be resolved after spatial roughness mapping but before the native
    text deck is written.  This compact declaration mirrors fields emitted by
    ``write_cut_case``; the final case receipt is cross-checked below before it
    becomes an execution trust anchor.
    """
    white=bool(roughness_settings.get("white_colebrook_chzk2d",False))
    work_depth=(.01 if roughness_mode!="manning" or white else 0.)
    shear_mode=shear_settings.get("mode","total")
    active=shear_mode!="total"
    return {
        "hydraulic_roughness": {
            "mode":roughness_mode,
            "parameter_units":_ROUGHNESS_PUBLIC[roughness_mode][1],
            "default_parameter":float(roughness_settings["default_parameter"]),
            "minimum_parameter":float(np.min(roughness_parameter)),
            "maximum_parameter":float(np.max(roughness_parameter)),
            "white_colebrook_chzk2d":white,
            "minimum_work_depth_m":work_depth,
        },
        "sediment_shear": {
            "mode":shear_mode,
            "kinematic_viscosity_m2_s":(
                float(shear_settings.get("kinematic_viscosity_m2_s",1.e-6))
                if active else None),
            "sand_roughness_m":(
                float(shear_settings.get("sand_roughness_m",.003))
                if active else None),
            "silt_roughness_m":(
                float(shear_settings.get("silt_roughness_m",.0003))
                if active else None),
            "critical_mud_fraction_low":(
                float(shear_settings.get("critical_mud_fraction_low",.2))
                if active else None),
            "critical_mud_fraction_high":(
                float(shear_settings.get("critical_mud_fraction_high",.8))
                if active else None),
            "fixed_fluff_fraction":(
                float(shear_settings.get("fluff_fraction",0.))
                if active else None),
        },
        "manning":{"friction_chezy_minimum_depth_m":0.},
    }


def sediment_method_metadata(numerics: Mapping[str, Any]) -> dict[str, Any]:
    """Describe actual saved methods; never use provenance to rewrite physics.

    Old projects are explicit. Automatic provenance is retained only when the
    recorded identities still match the effective configuration. Edited/stale
    metadata is regenerated from the numerical settings, not trusted as input.
    This does not choose a backend or admit additional runtime combinations.
    """
    enabled=numerics.get("sediment_enabled",True)
    law=numerics["bedload"] if enabled else 0
    slope=numerics["bed_slope"] if enabled else 0
    options={"exchange_mode":numerics["exchange"] if enabled else 0,
             "mpm_friction_mode":numerics.get("mpm_friction_mode",0) if enabled and law==2 else 0,
             "capacity_method_id":numerics.get("capacity_method_id")}
    if law not in range(9) or slope not in (0,2,3,5):
        try:
            capacity_source=_CAPACITY_SOURCE_BY_CODE[law]
            slope_source=_SLOPE_SOURCE_BY_CODE[slope]
        except KeyError as exc:
            raise ValueError("Saved sediment method has no catalogue identity") from exc
        return {
            "bedload_law":law,"bed_slope":slope,"selection_origin":"explicit",
            "capacity_method_id":options["capacity_method_id"],
            "capacity_source_id":capacity_source,"slope_source_id":slope_source,
            "exchange_mode":options["exchange_mode"],
            "mpm_friction_mode":options["mpm_friction_mode"],
            "messages":[
                f"Preserved explicit bed-slope method {slope} for bed-load law {law}."],
            "restrictions":[
                "Project, material, CPU/GPU route and validation restrictions still apply; "
                "this identity does not grant runtime support."],
            "catalog_schema":CATALOG_SCHEMA,"catalog_scope_id":CATALOG_SCOPE_ID,
            "semantic_catalog_id":semantic_catalog_id(),
            "selection_components":[
                "capacity","bed_slope","suspension","shear_closure"],
        }
    selection=resolve_sediment_methods(law,slope_enabled=bool(slope),
                                      slope_method=slope,**options)
    recorded=numerics.get("sediment_method_selection")
    if isinstance(recorded,Mapping) and recorded.get("selection_origin")=="automatic":
        automatic=resolve_sediment_methods(law,slope_enabled=bool(slope),**options)
        identity=("bedload_law","bed_slope","capacity_method_id",
                  "capacity_source_id","slope_source_id",
                  "exchange_mode","mpm_friction_mode","catalog_schema",
                  "catalog_scope_id","semantic_catalog_id","selection_components")
        resolved=automatic.as_dict()
        if all(recorded.get(key)==resolved[key] for key in identity):
            selection=automatic
    return selection.as_dict()


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        requirement = "positive finite" if positive else ("nonnegative finite" if nonnegative else "finite")
        raise ValueError(f"{name} must be {requirement}")
    return result


def _resolved_secondary_flow_model(value: Any) -> str:
    """Resolve old project metadata without mutating the persisted payload."""
    if value == LEGACY_SECONDARY_FLOW_MODEL_LABEL:
        return "equilibrium"
    if value not in ("equilibrium", "prognostic"):
        raise ValueError("Secondary-flow model must be equilibrium or prognostic")
    return value


def _digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_prepared_inputs(run_root: str | Path,
                             marker: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Authenticate the prepared deck and resolved profile before execution.

    The native deck carries a configuration fingerprint for checkpoint
    identity, but that text cannot authenticate itself after a user or another
    process edits the file.  The preparation marker is therefore the external
    trust anchor for managed CPU execution.  Both launchers call this function
    before starting the process, so a modified coefficient deck or resolved
    profile cannot silently enter a fresh run or a restart.

    Older prepared runs without these digests remain readable result archives;
    they must be prepared again before execution.
    """
    root = Path(run_root).expanduser().resolve()
    if marker is None:
        marker_path = root / RUN_MARKER
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Prepared run marker is missing or invalid") from exc
    if not isinstance(marker, Mapping):
        raise ValueError("Prepared run marker is missing or invalid")
    if "execution_profile_v2_internal_admission" in marker:
        raise ValueError(
            "Validation-only internal execution profiles cannot use a normal "
            "BRAIDS CPU launcher")
    preparation_backend = marker.get(
        "preparation_execution_backend", BACKEND_CPU)
    if preparation_backend != BACKEND_CPU:
        raise ValueError(
            "A GPU-owned prepared deck cannot use a normal BRAIDS CPU launcher")

    result: dict[str, Path] = {}
    authenticated = [("case.data", "case_sha256"),
                     ("resolved_inputs.json", "resolved_inputs_sha256")]
    if marker.get("capacity_execution_sidecar") is not None:
        if marker.get("capacity_execution_sidecar") != CAPACITY_EXECUTION_SIDECAR:
            raise ValueError("Prepared capacity-execution sidecar name is invalid")
        authenticated.append((CAPACITY_EXECUTION_SIDECAR,
                              "capacity_execution_sha256"))
    for name, marker_key in authenticated:
        expected = marker.get(marker_key)
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(character not in "0123456789abcdef" for character in expected)):
            raise ValueError(
                f"Prepared run has no valid {name} identity; Prepare Run again")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Prepared run input is missing or linked: {path}")
        if _digest(path) != expected:
            raise ValueError(
                f"Prepared run input changed after preparation: {name}; "
                "Prepare Run again")
        result[name] = path
    hiding_binding_fingerprint=marker.get(
        "hiding_foundation_binding_fingerprint")
    if hiding_binding_fingerprint is not None:
        if (not isinstance(hiding_binding_fingerprint,str) or
                len(hiding_binding_fingerprint)!=64 or
                any(value not in "0123456789abcdef"
                    for value in hiding_binding_fingerprint)):
            raise ValueError("Prepared Q3-H1 project binding identity is invalid")
        try:
            sidecar=json.loads(result[CAPACITY_EXECUTION_SIDECAR].read_text(
                encoding="utf-8"))
            marker_binding=_validate_hiding_foundation_binding_document(
                marker[HIDING_FOUNDATION_BINDING_KEY])
            sidecar_binding=_validate_hiding_foundation_binding_document(
                sidecar[HIDING_FOUNDATION_BINDING_KEY])
            identity=validate_hiding_foundation_execution_document(
                sidecar[HIDING_FOUNDATION_EXECUTION_KEY])
            route=sidecar[HIDING_FOUNDATION_ORDERED_ROUTE_KEY]
            base=sidecar["capacity_foundation_execution_identity"]
            base_route=sidecar[
                "capacity_foundation_ordered_route_identity"]
        except (KeyError,OSError,TypeError,ValueError,json.JSONDecodeError) as exc:
            raise ValueError(
                "Prepared Q3-H1 hiding execution identity is missing, stale, or forged") from exc
        if (marker_binding!=sidecar_binding or
                marker_binding["fingerprint"]!=hiding_binding_fingerprint or
                identity["mode"]!=marker_binding["mode"] or
                identity["method_id"]!=marker_binding["method_id"] or
                identity["capacity_law"]!=marker_binding["capacity_law"] or
                identity["fraction_ids"]!=marker_binding["fraction_ids"] or
                identity["parker_alpha"]!=marker_binding["parker_alpha"] or
                identity["wu_exponent"]!=marker_binding["wu_exponent"] or
                identity["capacity_profile_fingerprint"]!=marker_binding[
                    "capacity_profile_fingerprint"] or
                not isinstance(base,Mapping) or
                identity["capacity_execution_fingerprint"]!=base.get(
                    "fingerprint") or
                base_route!={
                    "ordered_parts":[
                        "capacity_execution_fingerprint",
                        "capacity_profile_fingerprint"],
                    "capacity_execution_fingerprint":base.get("fingerprint"),
                    "capacity_profile_fingerprint":identity[
                        "capacity_profile_fingerprint"]}):
            raise ValueError(
                "Prepared Q3-H1 hiding declaration differs from its execution identity")
        expected_route={
            "ordered_parts":[
                "capacity_execution_fingerprint",
                "capacity_profile_fingerprint",
                "hiding_execution_fingerprint"],
            "capacity_execution_fingerprint":identity[
                "capacity_execution_fingerprint"],
            "capacity_profile_fingerprint":identity[
                "capacity_profile_fingerprint"],
            "hiding_execution_fingerprint":identity["fingerprint"],
        }
        if (route!=expected_route or
                marker.get(HIDING_FOUNDATION_EXECUTION_KEY)!=identity or
                marker.get(HIDING_FOUNDATION_ORDERED_ROUTE_KEY)!=route):
            raise ValueError(
                "Prepared Q3-H1 hiding route differs from its authenticated marker")
    secondary_binding_fingerprint=marker.get(
        "secondary_foundation_binding_fingerprint")
    if secondary_binding_fingerprint is not None:
        if (not isinstance(secondary_binding_fingerprint,str) or
                len(secondary_binding_fingerprint)!=64 or
                any(value not in "0123456789abcdef"
                    for value in secondary_binding_fingerprint)):
            raise ValueError("Prepared Q3-H2 project binding identity is invalid")
        try:
            sidecar=json.loads(result[CAPACITY_EXECUTION_SIDECAR].read_text(
                encoding="utf-8"))
            marker_binding=_validate_secondary_foundation_binding_document(
                marker[SECONDARY_FOUNDATION_BINDING_KEY])
            sidecar_binding=_validate_secondary_foundation_binding_document(
                sidecar[SECONDARY_FOUNDATION_BINDING_KEY])
            identity=validate_secondary_foundation_execution_document(
                sidecar[SECONDARY_FOUNDATION_EXECUTION_KEY])
            route=sidecar[SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY]
            base=sidecar["capacity_foundation_execution_identity"]
        except (KeyError,OSError,TypeError,ValueError,json.JSONDecodeError) as exc:
            raise ValueError(
                "Prepared Q3-H2 secondary execution identity is missing, stale, or forged") from exc
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
        if (marker_binding!=sidecar_binding or
                marker_binding["fingerprint"]!=secondary_binding_fingerprint or
                identity["capacity_law"]!=marker_binding["capacity_law"] or
                identity["capacity_profile_fingerprint"]!=
                    marker_binding["capacity_profile_fingerprint"] or
                identity["base_envelope_version"]!=
                    marker_binding["base_envelope_version"] or
                identity["roughness_method_id"]!=
                    marker_binding["roughness_method_id"] or
                identity["rmincf"]!=marker_binding["rmincf"] or
                identity["chzmin"]!=marker_binding["chzmin"] or
                identity["espir"]!=marker_binding["espir"] or
                identity["betac"]!=marker_binding["betac"] or
                not isinstance(base,Mapping) or
                identity["capacity_execution_fingerprint"]!=
                    base.get("fingerprint") or route!=expected_route or
                marker.get(SECONDARY_FOUNDATION_EXECUTION_KEY)!=identity or
                marker.get(SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY)!=route):
            raise ValueError(
                "Prepared Q3-H2 secondary declaration differs from its execution identity")
    prognostic_binding_fingerprint=marker.get(
        "prognostic_secondary_binding_fingerprint")
    if prognostic_binding_fingerprint is not None:
        if secondary_binding_fingerprint is not None:
            raise ValueError(
                "Prepared run cannot carry both Q3-H2 and Q3-H3 secondary operators")
        if (not isinstance(prognostic_binding_fingerprint,str) or
                len(prognostic_binding_fingerprint)!=64 or
                any(value not in "0123456789abcdef"
                    for value in prognostic_binding_fingerprint)):
            raise ValueError("Prepared Q3-H3 project binding identity is invalid")
        try:
            sidecar=json.loads(result[CAPACITY_EXECUTION_SIDECAR].read_text(
                encoding="utf-8"))
            marker_binding=_validate_prognostic_secondary_binding_document(
                marker[PROGNOSTIC_SECONDARY_BINDING_KEY])
            sidecar_binding=_validate_prognostic_secondary_binding_document(
                sidecar[PROGNOSTIC_SECONDARY_BINDING_KEY])
            identity=validate_prognostic_secondary_execution_document(
                sidecar[PROGNOSTIC_SECONDARY_EXECUTION_KEY])
            route=sidecar[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY]
            base=sidecar["capacity_foundation_execution_identity"]
            forcing=sidecar["forcing_binding"]
        except (KeyError,OSError,TypeError,ValueError,json.JSONDecodeError) as exc:
            raise ValueError(
                "Prepared Q3-H3 prognostic-secondary execution identity is "
                "missing, stale, or forged") from exc
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
        binding_fields=(
            "method_id","model","capacity_law","capacity_profile_fingerprint",
            "base_envelope_version","roughness_method_id","ordered_fraction_ids",
            "transport_direction","inflow_mode","coriolis_disposition",
            "coriolis_constant_s_inverse","coriolis_source","rmincf","chzmin",
            "espir","betac","scalar_diffusivity_m2_s","von_karman_constant",
            "gpu_checkpoint_schema")
        if (marker_binding!=sidecar_binding or
                marker_binding["fingerprint"]!=prognostic_binding_fingerprint or
                any(identity[field]!=marker_binding[field]
                    for field in binding_fields) or
                identity["runtime_layout"]["bed_layer_count"]!=
                    marker_binding["bed_layer_count"] or
                not isinstance(base,Mapping) or
                identity["capacity_execution_fingerprint"]!=
                    base.get("fingerprint") or
                not isinstance(forcing,Mapping) or
                identity["roughness_fingerprint"]!=forcing.get("fingerprint") or
                route!=expected_route or
                marker.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)!=identity or
                marker.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=route or
                any(sidecar.get(key)!=value or marker.get(key)!=value
                    for key,value in
                    _prognostic_secondary_flat_receipt(identity).items())):
            raise ValueError(
                "Prepared Q3-H3 prognostic-secondary declaration differs "
                "from its execution identity")
    return result


def verified_native_executable(run_root: str | Path,
                               marker: Mapping[str, Any] | None = None) -> Path:
    """Return the exact native executable authenticated at preparation time.

    Older prepared folders did not always record an executable digest.  They
    remain valid result archives, but cannot be executed or resumed safely:
    the executable at the recorded path may have been replaced by a plugin
    update.  Re-preparing is the only fail-closed migration because it creates
    a deck and marker bound to the currently installed binary.
    """
    root = Path(run_root).expanduser().resolve()
    if marker is None:
        marker_path = root / RUN_MARKER
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Prepared run marker is missing or invalid") from exc
    if not isinstance(marker, Mapping):
        raise ValueError("Prepared run marker is missing or invalid")
    executable_value = marker.get("native_executable")
    if not isinstance(executable_value, str) or not executable_value:
        raise ValueError("Prepared run marker has no native executable; Prepare Run again")
    executable = Path(executable_value).expanduser().resolve()
    if not executable.is_file():
        raise ValueError(f"Native executable is missing: {executable}")
    expected = marker.get("executable_sha256")
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError(
            "Prepared run has no valid native executable identity; Prepare Run again")
    if _digest(executable) != expected:
        raise ValueError(
            "Native executable changed since this run was prepared; Prepare Run again")
    return executable


def grid_for_domain(domain: PolygonDomain, spacing_m: float) -> NativeGrid:
    spacing = _finite(spacing_m, "grid spacing", positive=True)
    vertices = np.asarray(domain.vertices_xy)
    xlower = math.floor(float(vertices[:, 0].min()) / spacing) * spacing
    ylower = math.floor(float(vertices[:, 1].min()) / spacing) * spacing
    nx = max(1, math.ceil((float(vertices[:, 0].max()) - xlower) / spacing))
    ny = max(1, math.ceil((float(vertices[:, 1].max()) - ylower) / spacing))
    return NativeGrid(nx, ny, xlower, ylower, spacing, spacing)


def _validate_geometry_contract(
    domain: PolygonDomain,
    assignments: Sequence[BoundaryAssignment],
    boundaries: Mapping[str, CutBoundary],
    grid: NativeGrid,
) -> None:
    """Validate persistent geometry rules without constructing cut cells."""
    domain.boundary_segments(assignments)
    active = {value.bc_id: value.kind for value in assignments
              if value.kind in {"inlet", "outlet"}}
    if set(active) != set(boundaries) or not all(
            isinstance(value, CutBoundary) for value in boundaries.values()):
        raise ValueError("Every selected inlet/outlet ID needs exactly one boundary configuration")
    for name, kind in active.items():
        mode = boundaries[name].mode
        if boundaries[name].is_inlet != (kind == "inlet"):
            raise ValueError(f"Boundary {name!r} kind and hydraulic mode disagree")
    vertices = np.asarray(domain.vertices_xy)
    xmax = grid.xlower_m + grid.nx * grid.dx_m
    ymax = grid.ylower_m + grid.ny * grid.dy_m
    tolerance = 64 * np.finfo(float).eps * max(
        abs(grid.xlower_m), abs(grid.ylower_m), abs(xmax), abs(ymax), 1.)
    if (vertices[:, 0] < grid.xlower_m - tolerance).any() or \
            (vertices[:, 0] > xmax + tolerance).any() or \
            (vertices[:, 1] < grid.ylower_m - tolerance).any() or \
            (vertices[:, 1] > ymax + tolerance).any():
        raise ValueError("The native grid must contain the complete domain polygon")


def create_project(
    *, workspace: str | Path, executable: str | Path, domain: PolygonDomain,
    assignments: Sequence[BoundaryAssignment], boundaries: Mapping[str, CutBoundary],
    dem_path: str | Path, grid: NativeGrid, initial_stage_m: float | None,
    lake_seed_xy: Sequence[float] | None, erodible_thickness_m: float,
    erodible_minimum_depth_m: float, end_time_s: float, output_interval_s: float,
    dt_max_s: float, cfl: float = .45, threads: int = 1, sediment: CutSediment | None = None,
    sediment_fractions: Sequence[CutSedimentFraction] | None = None,
    active_layer_thickness_m: float = 0.,
    active_layer_mode: str = "constant",
    active_layer_depth_fraction: float = 0.,
    active_layer_minimum_thickness_m: float | None = None,
    substrate_layer_count: int = 1, substrate_layer_thickness_m: float = .1,
    scheme: int = 5, bedload: int = 1, exchange: int = 2,
    engelund_hansen_chezy: float = 28.8,
    settling_velocity_m_s: float = .01, manning: float = .03,
    hydraulic_roughness_mode: str = "manning",
    white_colebrook_chzk2d: bool = False,
    sediment_shear_mode: str = "total",
    skin_kinematic_viscosity_m2_s: float = 1.e-6,
    skin_sand_roughness_m: float = .003,
    skin_silt_roughness_m: float = .0003,
    skin_critical_mud_fraction_low: float = .2,
    skin_critical_mud_fraction_high: float = .8,
    skin_fluff_fraction: float = 0.,
    horizontal_viscosity_m2_s: float = 0.,
    horizontal_viscosity_formulation: str = 'depth_conservative',
    bed_slope: int = 0, manning_zones_path: str | Path | None = None,
    manning_zones_layer: str = MANNING_LAYER_NAME,
    mpm_cf: float = .01, mpm_critical_shields: float = 0., mpm_friction_mode: int = 0,
    hiding_exposure: int = 0,
    hiding_parker_alpha: float = .6,
    hiding_wu_exponent: float = -.6,
    critical_mud_fraction: float = .5,
    transport_calibration_factor: float = 1.,
    capacity_method_id: str | None = None,
    capacity_parameters_by_fraction: Mapping[str, Any] | None = None,
    settling_mode: int = 0,
    hindered_settling_reference_concentration_kg_m3: float = DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3,
    bed_slope_alfabs: float = 1., bed_slope_alfabn: float = 1.5,
    bed_friction_angle_deg: float = 30.,
    secondary_flow_enabled: bool = False,
    secondary_flow_espir: float = 1.,
    secondary_flow_model: str = "equilibrium",
    secondary_flow_method_id: str | None = None,
    secondary_flow_method_provenance: str | None = None,
    secondary_flow_transport_direction: bool = True,
    secondary_flow_momentum_beta: float = 0.,
    secondary_flow_classic_betac: float = 0.,
    secondary_flow_scalar_diffusivity_m2_s: float = 0.,
    bedload_adaptation_enabled: bool = False,
    bedload_adaptation_length_diameters: float = 100.,
    morphological_acceleration_enabled: bool = False,
    morphological_factor: float = 1.,
    morphological_factor_series: CutMorphologicalFactorSeries | Mapping[str,Any] | None = None,
    bedform_roughness_enabled: bool = False,
    bedrock_raster_path: str | Path | None = None,
    erodible_thickness_raster_path: str | Path | None = None,
    erodible_zones_path: str | Path | None = None,
    erodible_zones_layer: str = ERODIBLE_LAYER_NAME,
    bank_morphology_enabled: bool = False,
    bank_zones_path: str | Path | None = None,
    bank_zones_layer: str = BANK_LAYER_NAME,
    bank_erosion_fraction: float = 0.5,
    bank_avalanche: bool = True,
    bank_wet_slope: float = 10.,
    bank_dry_slope: float = 1.,
    bank_avalanche_time_s: float = 86400.,
    bank_relaxation_length_m: float = 1.,
    bank_wet_depth_switch_m: float = .1,
    bank_failure_enabled: bool = False,
    bank_toe_erodibility_m_pa_s: float = 1.3e-6,
    bank_toe_critical_shear_pa: float = 32.5,
    bank_toe_exponent: float = 1.,
    bank_critical_undercut_m: float = .1,
    bank_toe_erosion_mode: str = "cohesive_excess_shear",
    bank_toe_capacity_fraction: float = 1.,
    bank_distributed_transport_fraction: float = .1,
    bank_distributed_critical_slope: float = .35,
    bank_front_tracking: bool = False,
    general_avalanche_enabled: bool = False,
    general_avalanche_wet_slope: float = 10.,
    general_avalanche_dry_slope: float = 1.,
    general_avalanche_time_s: float = 86400.,
    general_avalanche_relaxation_length_m: float = 1.,
    general_avalanche_wet_depth_switch_m: float = .1,
    solid_obstacles_path: str | Path | None = None,
    solid_obstacles_layer: str = OBSTACLE_LAYER_NAME,
    minimum_boundary_cell_area_fraction: float = 0.,
    section_vertical_tolerance_m: float = 0.,
    section_dem_path: str | Path | None = None,
    terrain_interpretation: str = "bilinear_surface",
    compact_sidecars: bool | None = None,
    source_layers: Mapping[str, str] | None = None,
    water_body_path: str | Path | None = None,
    water_body_layer: str = WATER_BODY_LAYER_NAME,
    sediment_enabled: bool = True,
    amr: AMRSettings | None = None,
    refinement_zones_path: str | Path | None = None,
    refinement_zones_layer: str = REFINEMENT_LAYER_NAME,
    execution_backend: str | None = None,
    morphology_output_settings: Mapping[str, object] | None = None,
    apply_morphology_output_defaults: bool = False,
) -> dict[str, Any]:
    """Create the complete persisted contract consumed by ``prepare_run``."""
    if not isinstance(domain, PolygonDomain) or not isinstance(grid, NativeGrid):
        raise ValueError("PolygonDomain and NativeGrid are required")
    assignments = tuple(assignments)
    amr = AMRSettings() if amr is None else amr
    if not isinstance(amr, AMRSettings):
        raise ValueError('AMR settings must be an AMRSettings record')
    _validate_geometry_contract(domain, assignments, boundaries, grid)
    workspace = Path(workspace).expanduser().resolve()
    executable = Path(executable).expanduser().resolve()
    dem_path = Path(dem_path).expanduser().resolve()
    section_dem = (None if section_dem_path is None else
                   Path(section_dem_path).expanduser().resolve())
    zones_path = None if manning_zones_path is None else Path(manning_zones_path).expanduser().resolve()
    bedrock_path = None if bedrock_raster_path is None else Path(bedrock_raster_path).expanduser().resolve()
    thickness_path = None if erodible_thickness_raster_path is None else Path(erodible_thickness_raster_path).expanduser().resolve()
    erodible_path = None if erodible_zones_path is None else Path(erodible_zones_path).expanduser().resolve()
    bank_path = None if bank_zones_path is None else Path(bank_zones_path).expanduser().resolve()
    obstacle_path = None if solid_obstacles_path is None else Path(solid_obstacles_path).expanduser().resolve()
    water_path = None if water_body_path is None else Path(water_body_path).expanduser().resolve()
    if not executable.is_file():
        raise ValueError(f"Native BRAIDS executable does not exist: {executable}")
    if not dem_path.is_file():
        raise ValueError(f"DEM does not exist: {dem_path}")
    if section_dem is not None and not section_dem.is_file():
        raise ValueError(f"Inlet-section DEM does not exist: {section_dem}")
    if terrain_interpretation not in ("bilinear_surface", "cell_average_raster"):
        raise ValueError(
            "Terrain interpretation must be bilinear_surface or cell_average_raster"
        )
    if compact_sidecars is not None and not isinstance(compact_sidecars, bool):
        raise ValueError("Compact sidecars must be true, false or automatic")
    if zones_path is not None and not zones_path.is_file():
        raise ValueError(f"Roughness-zone dataset does not exist: {zones_path}")
    if hydraulic_roughness_mode not in ("chezy","manning","white_colebrook"):
        raise ValueError("Hydraulic roughness mode must be chezy, manning, or white_colebrook")
    if not isinstance(white_colebrook_chzk2d,bool):
        raise ValueError("White-Colebrook ChzK2D switch must be true or false")
    if white_colebrook_chzk2d and hydraulic_roughness_mode!="white_colebrook":
        raise ValueError("ChzK2D is valid only with White-Colebrook roughness")
    if sediment_shear_mode not in ("total","soulsby_clarke_skin"):
        raise ValueError("Sediment shear mode must be total or soulsby_clarke_skin")
    skin_values=(
        _finite(skin_kinematic_viscosity_m2_s,"skin-shear kinematic viscosity",positive=True),
        _finite(skin_sand_roughness_m,"skin-shear sand roughness",positive=True),
        _finite(skin_silt_roughness_m,"skin-shear silt roughness",positive=True),
        _finite(skin_critical_mud_fraction_low,"skin-shear lower mud fraction",nonnegative=True),
        _finite(skin_critical_mud_fraction_high,"skin-shear upper mud fraction",positive=True),
        _finite(skin_fluff_fraction,"skin-shear fluff fraction",nonnegative=True))
    if not skin_values[3]<skin_values[4]<=1 or skin_values[5]>1:
        raise ValueError("Invalid Soulsby/Clarke skin-shear mud/fluff fractions")
    roughness_parameter=_finite(
        manning,"hydraulic roughness parameter",
        positive=hydraulic_roughness_mode!="manning",
        nonnegative=hydraulic_roughness_mode=="manning")
    for path, label in ((bedrock_path, "Bedrock raster"), (thickness_path, "Erodible-thickness raster"),
                        (erodible_path, "Erodible-zone dataset"), (bank_path, "Erodible-bank dataset"),
                        (obstacle_path, "Solid-obstacle dataset"),
                        (water_path, "Initial-water polygon dataset")):
        if path is not None and not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if thickness_path is not None and erodible_path is not None:
        raise ValueError("Choose an erodible-thickness raster or polygon zones, not both")
    if not isinstance(manning_zones_layer, str) or not manning_zones_layer.strip():
        raise ValueError("Roughness-zone layer name is required")
    if water_path is not None and (not isinstance(water_body_layer,str) or not water_body_layer.strip()):
        raise ValueError("Initial-water polygon layer name is required")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    seed = None
    if lake_seed_xy is not None:
        if len(lake_seed_xy) != 2:
            raise ValueError("lake_seed_xy requires x and y")
        seed = [_finite(value, "lake seed coordinate") for value in lake_seed_xy]
    if seed is not None and water_path is not None:
        raise ValueError("Choose an initial-water polygon or a legacy lake seed, not both")
    fractions=None if sediment_fractions is None else tuple(sediment_fractions)
    if fractions is not None:
        if sediment is not None:
            raise ValueError("sediment_fractions replaces the single sediment material")
        if len(fractions)<2 or not all(isinstance(value,CutSedimentFraction) for value in fractions):
            raise ValueError("At least two CutSedimentFraction values are required")
        material=CutSediment(**{name:getattr(fractions[0],name) for name in CutSediment.__dataclass_fields__})
    else:
        material = CutSediment() if sediment is None else sediment
    if not isinstance(material, CutSediment):
        raise ValueError("sediment must be CutSediment")
    active_layer=_finite(active_layer_thickness_m,"active-layer thickness",nonnegative=True)
    if active_layer>0 and fractions is None:
        raise ValueError("Active layers require at least two sediment fractions")
    if active_layer_mode not in ("constant","water_depth"):
        raise ValueError("Active-layer mode must be constant or water_depth")
    active_depth_fraction=_finite(active_layer_depth_fraction,"active-layer depth fraction",nonnegative=True)
    active_minimum=(active_layer if active_layer_minimum_thickness_m is None else
                    _finite(active_layer_minimum_thickness_m,"minimum active-layer thickness",nonnegative=True))
    if active_layer_mode=="water_depth" and (active_layer<=0 or active_depth_fraction<=0 or active_minimum<=0):
        raise ValueError("Water-depth active layers require positive layer, depth fraction, and minimum thickness")
    if isinstance(substrate_layer_count,bool) or not isinstance(substrate_layer_count,int) or not 1<=substrate_layer_count<=32:
        raise ValueError("Substrate layer count must be an integer from 1 to 32")
    substrate_thickness=_finite(substrate_layer_thickness_m,"preserved substrate layer thickness",nonnegative=True)
    if active_layer>0 and substrate_layer_count>1 and substrate_thickness<=0:
        raise ValueError("Preserved substrate layer thickness must be positive when multiple layers are enabled")
    if not isinstance(sediment_enabled,bool):
        raise ValueError("sediment_enabled must be true or false")
    if sediment_shear_mode=="soulsby_clarke_skin":
        if not sediment_enabled:
            raise ValueError("Soulsby/Clarke skin shear requires sediment transport")
        if bedload==9:
            raise ValueError("Delft3D BsSkin cannot replace the TELEMAC-GAIA skin correction")
        cohesive_exchange_consumer=(
            exchange==3 and fractions is not None and
            any(value.sediment_type=="mud" for value in fractions))
        if bedload not in (5,6,7) and not cohesive_exchange_consumer:
            raise ValueError(
                "Soulsby/Clarke skin shear requires an RP_TAUB consumer: law 5/6/7 "
                "or exchange mode 3 with an actually cohesive fraction")
    if isinstance(settling_mode,bool) or not isinstance(settling_mode,int) or settling_mode not in (0,1):
        raise ValueError("Settling treatment must be constant (0) or concentration-dependent (1)")
    hindered_reference=_finite(hindered_settling_reference_concentration_kg_m3,
                               "hindered-settling reference concentration",positive=True)
    corrected_eh_requested=(
        bedload==8 and capacity_method_id==EH_DELFT_METHOD_ID)
    # The corrected FLOW closure consumes cell/trace-local Chezy resolved from
    # the selected typed C/n/ks field.  The historical independent constant is
    # deliberately absent from that project identity, even though the legacy
    # UI argument retains its backward-compatible default.
    eh_chezy=(None if corrected_eh_requested else
              _finite(engelund_hansen_chezy,
                      "Engelund-Hansen constant Chezy",positive=True))
    if bedload==8:
        if exchange!=0:
            raise ValueError(
                "Engelund-Hansen law 8 is qualified only with suspended exchange off (mode 0)")
        if bed_slope not in (0,2,3,5):
            raise ValueError(
                "Engelund-Hansen law 8 slope correction must be off, legacy 2, Koch-Flokstra 3, or Delft3D FLOW 5")
    if bed_slope in (3,5) and bedload not in (2,8):
        raise ValueError("Delft3D FLOW slope input contract requires MPM or Engelund-Hansen (law 2/8)")
    viscosity=_finite(horizontal_viscosity_m2_s,'horizontal viscosity',nonnegative=True)
    if horizontal_viscosity_formulation not in ('depth_conservative','velocity_laplacian'):
        raise ValueError('Unsupported horizontal viscosity formulation')
    if horizontal_viscosity_formulation=='velocity_laplacian' and viscosity<=0.:
        raise ValueError('Velocity-Laplacian viscosity requires a positive coefficient')
    if (not isinstance(bank_morphology_enabled,bool) or not isinstance(bank_avalanche,bool)
            or not isinstance(bank_failure_enabled,bool)
            or not isinstance(bank_front_tracking,bool)
            or not isinstance(general_avalanche_enabled,bool) or not isinstance(secondary_flow_enabled,bool)
            or not isinstance(bedload_adaptation_enabled,bool)
            or not isinstance(morphological_acceleration_enabled,bool)
            or not isinstance(bedform_roughness_enabled,bool)):
        raise ValueError("Morphodynamic switches must be true or false")
    if not isinstance(secondary_flow_transport_direction,bool):
        raise ValueError("Secondary-flow transport-direction switch must be true or false")
    bank_fraction=_finite(bank_erosion_fraction,"bank erosion supply fraction",nonnegative=True)
    if bank_fraction>1:raise ValueError("Bank erosion supply fraction must not exceed 1")
    bank_wet=_finite(bank_wet_slope,"wet-bank maximum slope",positive=True)
    bank_dry=_finite(bank_dry_slope,"dry-bank maximum slope",positive=True)
    bank_time=_finite(bank_avalanche_time_s,"bank avalanche time",positive=True)
    bank_length=_finite(bank_relaxation_length_m,"bank relaxation length",positive=True)
    bank_switch=_finite(bank_wet_depth_switch_m,"bank wet-depth switch",nonnegative=True)
    if bank_toe_erosion_mode not in (
            "cohesive_excess_shear", "noncohesive_transport_capacity",
            "distributed_slope_transport"):
        raise ValueError(
            "Bank-toe erosion mode must be cohesive_excess_shear, "
            "noncohesive_transport_capacity, or distributed_slope_transport")
    bank_toe_capacity=_finite(bank_toe_capacity_fraction,"bank-toe capacity fraction",nonnegative=True)
    if bank_toe_capacity>1:raise ValueError("Bank-toe capacity fraction must not exceed 1")
    bank_distributed_fraction=_finite(
        bank_distributed_transport_fraction,
        "distributed bank transport fraction", nonnegative=True)
    if bank_distributed_fraction>1:
        raise ValueError("Distributed bank transport fraction must not exceed 1")
    bank_distributed_slope=_finite(
        bank_distributed_critical_slope,
        "distributed bank critical slope", positive=True)
    bank_toe_erodibility=_finite(bank_toe_erodibility_m_pa_s,"bank toe erodibility",
                                 positive=bank_toe_erosion_mode=="cohesive_excess_shear",
                                 nonnegative=bank_toe_erosion_mode!="cohesive_excess_shear")
    bank_toe_critical_shear=_finite(bank_toe_critical_shear_pa,"bank toe critical shear",nonnegative=True)
    bank_toe_power=_finite(bank_toe_exponent,"bank toe erosion exponent",positive=True)
    bank_critical_undercut=_finite(bank_critical_undercut_m,"bank critical undercut",positive=True)
    general_wet=_finite(general_avalanche_wet_slope,"general wet maximum slope",positive=True)
    general_dry=_finite(general_avalanche_dry_slope,"general dry maximum slope",positive=True)
    general_time=_finite(general_avalanche_time_s,"general avalanche time",positive=True)
    general_length=_finite(general_avalanche_relaxation_length_m,"general avalanche relaxation length",positive=True)
    general_switch=_finite(general_avalanche_wet_depth_switch_m,"general avalanche wet-depth switch",nonnegative=True)
    section_tolerance=_finite(
        section_vertical_tolerance_m, "inlet-section vertical tolerance",
        nonnegative=True,
    )
    secondary_model=_resolved_secondary_flow_model(secondary_flow_model)
    secondary_espir=_finite(secondary_flow_espir,"secondary-flow Espir",nonnegative=True)
    secondary_beta=_finite(secondary_flow_momentum_beta,"secondary-flow momentum Beta",nonnegative=True)
    secondary_classic_betac=_finite(
        secondary_flow_classic_betac,"classic FLOW secondary Betac")
    secondary_scalar_diffusivity=_finite(
        secondary_flow_scalar_diffusivity_m2_s,
        "classic FLOW secondary scalar diffusivity",nonnegative=True)
    if secondary_flow_method_id is not None and (
            not isinstance(secondary_flow_method_id,str) or
            not secondary_flow_method_id.strip()):
        raise ValueError("Secondary-flow method id must be nonempty text or null")
    if secondary_flow_method_provenance is not None and (
            not isinstance(secondary_flow_method_provenance,str) or
            not secondary_flow_method_provenance.strip()):
        raise ValueError(
            "Secondary-flow method provenance must be nonempty text or null")
    if ((secondary_flow_method_id is None)!=
            (secondary_flow_method_provenance is None)):
        raise ValueError(
            "Secondary-flow method id and provenance must be declared together")
    if secondary_flow_method_id is not None and (
            secondary_model!="prognostic" or
            secondary_flow_method_id!=PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID or
            secondary_flow_method_provenance!=
                PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE):
        raise ValueError(
            "Classic prognostic secondary-flow method/provenance declaration is invalid")
    if secondary_flow_method_id is None and (
            secondary_classic_betac!=0. or secondary_scalar_diffusivity!=0.):
        raise ValueError(
            "Classic Betac/diffusivity require an explicit classic prognostic "
            "secondary-flow method declaration")
    adaptation_length=_finite(bedload_adaptation_length_diameters,
                              "bed-load adaptation length in grain diameters",positive=True)
    morfac=_finite(morphological_factor,"morphological acceleration factor",positive=True)
    if not 1<=morfac<=20:
        raise ValueError("Morphological acceleration factor must be within 1 to 20")
    if isinstance(morphological_factor_series,Mapping):
        morphological_factor_series=CutMorphologicalFactorSeries(**morphological_factor_series)
    if morphological_factor_series is not None and not isinstance(
            morphological_factor_series,CutMorphologicalFactorSeries):
        raise ValueError("Morphological-factor series is invalid")
    if bank_path is not None and (not isinstance(bank_zones_layer,str) or not bank_zones_layer.strip()):
        raise ValueError("Erodible-bank layer name is required")
    if obstacle_path is not None and (not isinstance(solid_obstacles_layer,str) or not solid_obstacles_layer.strip()):
        raise ValueError("Solid-obstacle layer name is required")
    if bank_morphology_enabled:
        if not sediment_enabled:raise ValueError("Bank morphology requires sediment transport")
        if fractions is None or active_layer<=0:
            raise ValueError("Bank morphology requires multiple fractions and a positive active-layer thickness")
        if bank_path is None:raise ValueError("Bank morphology requires an erodible-bank polygon layer")
        if bank_fraction==0 and not bank_avalanche:
            raise ValueError("Enable dry-bank erosion, bank avalanching, or both")
    if bank_failure_enabled:
        if bank_morphology_enabled:
            raise ValueError("Choose process-separated bank failure or legacy bank morphology, not both")
        if not sediment_enabled:raise ValueError("Bank failure requires sediment transport")
        if fractions is None or active_layer<=0:
            raise ValueError("Bank failure requires multiple fractions and a positive active-layer thickness")
        if bank_path is None:raise ValueError("Bank failure requires an erodible-bank polygon layer")
        if bank_toe_erosion_mode=="cohesive_excess_shear":
            _finite(manning,"Manning coefficient for bank failure",positive=True)
        else:
            if bedload==0:raise ValueError("Noncohesive bank failure requires an active bed-load formulation")
            if any(value.sediment_type!="sand" for value in fractions):
                raise ValueError("Noncohesive bank failure requires noncohesive sediment fractions")
            if exchange!=0:
                raise ValueError("Noncohesive bank failure requires bed load without suspended exchange")
    if bank_front_tracking:
        raise ValueError("Persistent bank-front tracking is retired; use bed-elevation bank evolution")
    if general_avalanche_enabled:
        if not sediment_enabled:
            raise ValueError("Domain-wide avalanching requires sediment transport")
        if fractions is None or active_layer<=0:
            raise ValueError("Domain-wide avalanching requires multiple fractions and a positive active-layer thickness")
        if bank_morphology_enabled and bank_avalanche:
            raise ValueError("Domain-wide and bank-zone avalanching cannot both be enabled")
    if secondary_flow_enabled:
        if secondary_model=="equilibrium":
            if not secondary_flow_transport_direction or secondary_beta!=0:
                raise ValueError(
                    "Legacy equilibrium secondary flow has fixed direction-only coupling and zero momentum Beta")
            if not sediment_enabled:
                raise ValueError("Secondary-flow sediment steering requires sediment transport")
            if bedload not in (2,3,5,6,7,8,11,12,14):
                raise ValueError("Secondary-flow sediment steering requires a magnitude-based bed-load formulation")
        elif secondary_flow_transport_direction:
            if not sediment_enabled:
                raise ValueError("Secondary-flow sediment steering requires sediment transport")
            if bedload not in (1,2,3,5,6,7,8,11,12,14):
                raise ValueError(
                    "Prognostic secondary-flow transport direction requires a supported bed-load formulation")
        _finite(manning,"Manning coefficient for secondary flow",positive=True)
    if bedload_adaptation_enabled:
        if not sediment_enabled or bedload==0:
            raise ValueError("Bed-load adaptation requires an active bed-load formulation")
    if morphological_acceleration_enabled:
        if not sediment_enabled or bedload==0:
            raise ValueError("Morphological acceleration requires an active bed-load formulation")
        if exchange!=0:
            raise ValueError("Bed-load morphological acceleration is not qualified with suspended exchange")
        if bank_morphology_enabled or bank_failure_enabled:
            raise ValueError("Morphological acceleration is not qualified with bank erosion or bank failure")
        if any(boundary.hydraulic_series is not None for boundary in boundaries.values()):
            raise ValueError("Morphological acceleration is not qualified with time-varying hydraulic boundaries")
    composed_boundary = next(iter(boundaries.values())) if len(boundaries) == 1 else None
    recursive_bank_bedform = (
        amr.enabled and amr.subcycling and
        bank_failure_enabled and
        bank_toe_erosion_mode in (
            "noncohesive_transport_capacity",
            "distributed_slope_transport") and
        not bank_front_tracking and general_avalanche_enabled and
        not bank_morphology_enabled and not secondary_flow_enabled and
        not bedload_adaptation_enabled and not morphological_acceleration_enabled and
        bedload == 2 and exchange == 0 and mpm_friction_mode == 1 and
        bed_slope == 0 and hiding_exposure == 0 and viscosity == 0. and
        active_layer_mode == "constant" and substrate_layer_count == 1 and
        composed_boundary is not None and composed_boundary.mode == "stage" and
        composed_boundary.hydraulic_series is None and
        composed_boundary.sediment_series is None and
        not composed_boundary.fraction_sediment_series and
        not composed_boundary.fraction_concentration_kg_m3 and
        composed_boundary.concentration_kg_m3 == 0. and
        composed_boundary.bedload_boundary_mode != "equilibrium_capacity" and
        composed_boundary.bed_input_kg_m_s == 0. and
        composed_boundary.bed_input_kg_s in (None, 0.) and
        not composed_boundary.fraction_bed_input_kg_s)
    if bedform_roughness_enabled:
        if hydraulic_roughness_mode!="manning":
            raise ValueError("The current bedform-feedback operator requires Manning hydraulic roughness")
        if not sediment_enabled or bedload!=2 or mpm_friction_mode!=1:
            raise ValueError("Bedform roughness requires roughness-corrected MPM bed load")
        if exchange!=0:
            raise ValueError("Bedform roughness is not qualified with suspended exchange")
        if (any((secondary_flow_enabled,bedload_adaptation_enabled,morphological_acceleration_enabled,
                 bank_morphology_enabled,bank_failure_enabled,general_avalanche_enabled)) and
                not recursive_bank_bedform):
            raise ValueError("Bedform roughness is not qualified with another experimental morphology process")
        if fractions is not None and any(value.sediment_type!='sand' for value in fractions):
            raise ValueError("Bedform roughness requires noncohesive sediment fractions")
        _finite(manning,"Manning coefficient for bedform roughness",positive=True)
        if any(boundary.mode=='inlet_normal' for boundary in boundaries.values()):
            raise ValueError("Bedform roughness is not qualified with normal-depth inlet control")
    initial_stage=(None if initial_stage_m is None else _finite(initial_stage_m,"initial stage"))
    if initial_stage is None and (water_path is not None or seed is not None):
        raise ValueError("An initial water level is required for an initial-water polygon or legacy seed")
    payload = {
        "format": PROJECT_FORMAT,
        "workspace": str(workspace),
        "native_executable": str(executable),
        "domain": {"domain_id": domain.domain_id, "crs_wkt": domain.crs_wkt,
                   "vertices_xy": [list(point) for point in domain.vertices_xy]},
        "grid": asdict(grid),
        "assignments": [asdict(value) for value in assignments],
        "boundaries": {name: asdict(value) for name, value in sorted(boundaries.items())},
        "terrain": {
            "dem_path": str(dem_path),
            "section_dem_path": None if section_dem is None else str(section_dem),
            "method": "exact source-pixel area average",
            "interpretation": terrain_interpretation,
            "section_vertical_tolerance_m": section_tolerance,
        },
        "roughness": {"default_manning_n": roughness_parameter,
                      "mode": hydraulic_roughness_mode,
                      "default_parameter": roughness_parameter,
                      "parameter_name": _ROUGHNESS_PUBLIC[hydraulic_roughness_mode][0],
                      "parameter_units": _ROUGHNESS_PUBLIC[hydraulic_roughness_mode][1],
                      "parameter_label": _ROUGHNESS_PUBLIC[hydraulic_roughness_mode][2],
                      "legacy_storage": {
                          "field": "default_manning_n",
                          "physical_type": hydraulic_roughness_mode=="manning",
                          "note": ("physical Manning n" if hydraulic_roughness_mode=="manning" else
                                   "compatibility alias containing the selected raw parameter; not Manning n")},
                      "white_colebrook_chzk2d": white_colebrook_chzk2d,
                      "zones_path": None if zones_path is None else str(zones_path),
                      "zones_layer": manning_zones_layer,
                      "zones_schema": (
                          "legacy_manning_v1" if zones_path is not None and
                          manning_zones_layer==MANNING_LAYER_NAME else
                          "typed_roughness_v1"),
                      "cell_rule": "whole retained cell on positive-area polygon intersection",
                      "overlap_policy": (
                          "shared-node/edge cell takes highest value; other co-occupancy rejected"
                          if zones_path is not None and manning_zones_layer==MANNING_LAYER_NAME else
                          "highest integer priority; equal-priority unequal raw values reject")},
        "sediment_shear": {
            "mode": sediment_shear_mode,
            "kinematic_viscosity_m2_s": skin_values[0],
            "sand_roughness_m": skin_values[1],
            "silt_roughness_m": skin_values[2],
            "critical_mud_fraction_low": skin_values[3],
            "critical_mud_fraction_high": skin_values[4],
            "fluff_fraction": skin_values[5],
            "scope": "sediment RP_TAUB consumers only; hydraulic friction remains total roughness"},
        "initial": {"stage_m": initial_stage, "lake_seed_xy": seed,
                    "water_body_path": None if water_path is None else str(water_path),
                    "water_body_layer": water_body_layer,
                    "water_body_cell_rule": "whole retained cell on positive-area polygon intersection",
                    "erodible_thickness_m": _finite(erodible_thickness_m, "erodible thickness", nonnegative=True),
                    "erodible_minimum_depth_m": _finite(erodible_minimum_depth_m, "minimum erodible depth", nonnegative=True),
                    "active_layer_thickness_m": active_layer,
                    "active_layer_mode": active_layer_mode,
                    "active_layer_depth_fraction": active_depth_fraction,
                    "active_layer_minimum_thickness_m": active_minimum,
                    "substrate_layer_count": substrate_layer_count,
                    "substrate_layer_thickness_m": substrate_thickness},
        "bed_inputs": {
            "bedrock_raster_path": None if bedrock_path is None else str(bedrock_path),
            "thickness_raster_path": None if thickness_path is None else str(thickness_path),
            "zones_path": None if erodible_path is None else str(erodible_path),
            "zones_layer": erodible_zones_layer,
            "cell_rule": "whole retained cell on positive-area polygon intersection",
            "overlap_policy": "different values on one cell rejected",
        },
        "internal_conditions": {"solid_obstacles": {
            "path": None if obstacle_path is None else str(obstacle_path),
            "layer": solid_obstacles_layer,
            "geometry": "exact polygon subtraction from retained water area",
            "wall_behavior": "fixed impermeable reflective shallow-water wall",
            "sediment_behavior": "zero suspended and bed-load flux"},
            "bank_morphology": {
            "enabled": bank_morphology_enabled,
            "zones_path": None if bank_path is None else str(bank_path),
            "zones_layer": bank_zones_layer,
            "cell_rule": "whole retained cell on positive-area polygon intersection",
            "erosion_supply_fraction": bank_fraction,
            "avalanche_enabled": bank_avalanche,
            "wet_maximum_slope": bank_wet,
            "dry_maximum_slope": bank_dry,
            "avalanche_time_s": bank_time,
            "relaxation_length_m": bank_length,
            "wet_depth_switch_m": bank_switch,
            "domain_boundary": "fixed"},
            "bank_failure": {
            "enabled": bank_failure_enabled,
            "qualified": bank_toe_erosion_mode=="cohesive_excess_shear",
            "qualification_evidence": (
                "validation/sediment/jamieson2013_bend/reports/crossed_bank_forcing_oracle_v1/RESULTS.md"
                if bank_toe_erosion_mode=="distributed_slope_transport" else
                "validation/sediment/bank_toe_geometry/SUPPLY_RESULTS_V1.md"
                if bank_toe_erosion_mode=="noncohesive_transport_capacity" else
                "validation/sediment/delft3d_current/bank_failure/reports/metrics.json"),
            "zones_path": None if bank_path is None else str(bank_path),
            "zones_layer": bank_zones_layer,
            "cell_rule": "whole retained cell on positive-area polygon intersection",
            "toe_erosion_mode": bank_toe_erosion_mode,
            "toe_capacity_fraction": bank_toe_capacity,
            "distributed_transport_fraction": bank_distributed_fraction,
            "distributed_critical_slope": bank_distributed_slope,
            "front_tracking_enabled": bank_front_tracking,
            "toe_erodibility_m_pa_s": bank_toe_erodibility,
            "toe_critical_shear_pa": bank_toe_critical_shear,
            "toe_exponent": bank_toe_power,
            "critical_undercut_m": bank_critical_undercut,
            "repose_redistribution": "qualified domain-wide avalanche when enabled",
            "domain_boundary": "fixed"},
            "general_avalanche": {
            "enabled": general_avalanche_enabled,
            "qualified": True,
            "qualification_evidence": "validation/sediment/delft3d_current/general_avalanche/reports/metrics.json",
            "wet_maximum_slope": general_wet,
            "dry_maximum_slope": general_dry,
            "avalanche_time_s": general_time,
            "relaxation_length_m": general_length,
            "wet_depth_switch_m": general_switch,
            "domain_boundary": "fixed"},
            "secondary_flow": {
            "enabled": secondary_flow_enabled,
            "qualified": bool(secondary_flow_enabled and secondary_model=="equilibrium"),
            "experimental": bool(secondary_flow_enabled and secondary_model=="prognostic"),
            "qualification_evidence": (
                "validation/sediment/delft3d_current/secondary_flow/reports/metrics.json"
                if secondary_flow_enabled and secondary_model=="equilibrium" else None),
            "model": secondary_model,
            "method_id": secondary_flow_method_id,
            "method_provenance": secondary_flow_method_provenance,
            "transport_direction": secondary_flow_transport_direction,
            "momentum_beta": secondary_beta,
            "classic_betac": secondary_classic_betac,
            "scalar_diffusivity_m2_s": secondary_scalar_diffusivity,
            "espir": secondary_espir,
            "transport_effect": (
                "bed-load direction only; scalar magnitude is preserved"
                if secondary_flow_transport_direction else "disabled"),
            "momentum_effect": (
                f"experimental BetaSpiral feedback with Beta={secondary_beta:.17g}"
                if secondary_model=="prognostic" and secondary_beta>0 else "disabled")},
            "bedload_adaptation": {
            "enabled": bedload_adaptation_enabled,
            "qualified": True,
            "qualification_evidence": "validation/sediment/delft3d_current/adaptation_length/reports/metrics.json",
            "model": "steady first-order streamline advection-relaxation",
            "length_diameters": adaptation_length,
            "length_rule": "one class-specific length: factor times D50",
            "default": "disabled"},
            "morphological_acceleration": {
            "enabled": morphological_acceleration_enabled,
            "qualified": True,
            "qualification_evidence": "validation/sediment/delft3d_current/morphological_acceleration/reports/metrics.json",
            "constant_factor": morfac,
            "series": None if morphological_factor_series is None else asdict(morphological_factor_series),
            "interpolation": "Delft3D-compatible block interpolation",
            "clock": "hydraulic seconds; morphological time is integral of factor",
            "published_regime_limits": {"quasi_steady_subcritical_bedform": 20,
                                         "wet_dry_hydraulic_transient": 1},
            "default": "disabled"},
            "bedform_roughness": {
            "enabled": bedform_roughness_enabled,
            "qualified": True,
            "qualification_evidence": "validation/sediment/delft3d_current/bedform_roughness_assessment/reports/metrics.json",
            "model": "Fredsoe MPM equilibrium height with Van Rijn 1984 length and roughness",
            "equilibrium": "deterministic local fixed point",
            "default": "disabled"}},
        "sediment": asdict(material),
        "numerics": {"amr": asdict(amr), "end_time_s": _finite(end_time_s, "end time", positive=True),
                     "output_interval_s": _finite(output_interval_s, "output interval", positive=True),
                     "dt_max_s": _finite(dt_max_s, "maximum timestep", positive=True),
                     "cfl": _finite(cfl, "CFL limit", positive=True),
                     "threads": threads, "compact_sidecars": compact_sidecars,
                      "scheme": int(scheme), "sediment_enabled":sediment_enabled,
                      "bedload": int(bedload),
                     **({} if corrected_eh_requested else
                        {"engelund_hansen_chezy": eh_chezy}),
                     "minimum_boundary_cell_area_fraction": _finite(minimum_boundary_cell_area_fraction, "minimum boundary-cell area fraction", nonnegative=True),
                     "exchange": int(exchange),
                     "settling_velocity_m_s": _finite(settling_velocity_m_s, "settling velocity", nonnegative=True),
                     "settling_mode": settling_mode,
                     "hindered_settling_reference_concentration_kg_m3": hindered_reference,
                     "manning": roughness_parameter,
                     **({} if viscosity == 0. and horizontal_viscosity_formulation == 'depth_conservative' else
                        {"horizontal_viscosity_m2_s": viscosity,
                         "horizontal_viscosity_formulation": horizontal_viscosity_formulation}),
                     "bed_slope": int(bed_slope),
                     **({} if bedload==2 and mpm_friction_mode==1 else
                        {"mpm_cf": _finite(mpm_cf, "MPM friction coefficient", nonnegative=True)}),
                     "mpm_critical_shields": _finite(mpm_critical_shields, "MPM critical Shields", nonnegative=True),
                     "mpm_friction_mode": int(mpm_friction_mode),
                     "hiding_exposure": hiding_exposure,
                     "hiding_parker_alpha": _finite(hiding_parker_alpha, "Parker hiding/exposure alpha"),
                     "hiding_wu_exponent": _finite(hiding_wu_exponent, "Wu hiding/exposure exponent"),
                     "critical_mud_fraction": _finite(critical_mud_fraction, "critical mud fraction", nonnegative=True),
                     "transport_calibration_factor": _finite(transport_calibration_factor, "transport calibration factor", nonnegative=True),
                     "bed_slope_alfabs": _finite(bed_slope_alfabs, "AlfaBs", nonnegative=True),
                     "bed_slope_alfabn": _finite(bed_slope_alfabn, "AlfaBn", nonnegative=True),
                     "bed_friction_angle_deg": _finite(bed_friction_angle_deg, "bed friction angle")},
        "source_layers": dict(source_layers or {}),
    }
    if sediment_enabled and bedload in range(9):
        payload["numerics"]["capacity_method_id"] = resolve_sediment_methods(
            bedload, slope_enabled=False, slope_method=0,
            exchange_mode=exchange,
            mpm_friction_mode=(mpm_friction_mode if bedload==2 else 0),
            capacity_method_id=capacity_method_id,
        ).capacity_method_id
    elif capacity_method_id is not None:
        raise ValueError(
            "An explicit capacity execution identity is unavailable for this bed-load law")
    payload['mesh_refinement']={'zones_path':None if refinement_zones_path is None else
        str(Path(refinement_zones_path).expanduser().resolve()),'zones_layer':refinement_zones_layer}
    if fractions is not None:payload["sediment_fractions"]=[asdict(value) for value in fractions]
    payload[CAPACITY_BINDING_KEY]=_build_capacity_binding(
        payload,capacity_parameters_by_fraction,require_split_parameters=True)
    payload[CAPACITY_EXECUTION_BINDING_KEY]=_build_capacity_execution_binding(payload)
    hiding_foundation_binding=_build_hiding_foundation_binding(payload)
    if hiding_foundation_binding is not None:
        payload[HIDING_FOUNDATION_BINDING_KEY]=hiding_foundation_binding
    secondary_foundation_binding=_build_secondary_foundation_binding(payload)
    if secondary_foundation_binding is not None:
        payload[SECONDARY_FOUNDATION_BINDING_KEY]=secondary_foundation_binding
    prognostic_secondary_binding=_build_prognostic_secondary_binding(payload)
    if prognostic_secondary_binding is not None:
        payload[PROGNOSTIC_SECONDARY_BINDING_KEY]=prognostic_secondary_binding
    try:
        foundation=resolve_project_capacity_foundation_profile(payload)
    except CapacityFoundationProfileError as exc:
        # A selected later-slice auxiliary (for example generic hiding or
        # secondary steering) remains representable by the existing project
        # schema, but must not be relabelled as this complete initial profile.
        # All missing identities/dependencies still fail immediately.
        if exc.category!="combination_not_implemented":
            raise ValueError(str(exc)) from exc
        foundation=None
    if foundation is not None:
        payload[EXECUTION_PROFILE_BINDING_KEY]=dict(foundation.project_binding)
        payload[CAPACITY_FOUNDATION_RESOLUTION_KEY]=dict(
            foundation.resolution_receipt)
    payload=_validated_project(payload)
    if execution_backend is not None:
        payload[QGIS_PRODUCT_KEY]=bind_qgis_product(
            payload, backend=execution_backend,
            output_settings=morphology_output_settings,
            apply_output_defaults=apply_morphology_output_defaults,
        )
        # Re-enter the ordinary validator so the portable product document is
        # checked by the same load/save path as every other project identity.
        payload=_validated_project(payload)
    payload["numerics"]["sediment_method_selection"]=sediment_method_metadata(payload["numerics"])
    return payload


def _validated_project(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, Mapping) or
            payload.get("format") not in (PROJECT_FORMAT,LEGACY_PROJECT_FORMAT)):
        raise ValueError(f"Expected {PROJECT_FORMAT} or {LEGACY_PROJECT_FORMAT}")
    legacy_project=payload.get("format")==LEGACY_PROJECT_FORMAT
    legacy_eh_defaulted=False
    if legacy_project and isinstance(payload.get("numerics"),Mapping):
        # v1 predates the explicit Engelund-Hansen Chezy control.  Its UI and
        # runtime default was 28.8, so record that dormant value during the
        # v2 migration instead of letting a load/edit/save cycle introduce it
        # implicitly.  It has no physical effect unless law 8 is selected.
        migrated_numerics=dict(payload["numerics"])
        if "engelund_hansen_chezy" not in migrated_numerics:
            migrated_numerics["engelund_hansen_chezy"]=28.8
            legacy_eh_defaulted=True
        payload={**payload,"numerics":migrated_numerics}
    try:
        domain_data = payload["domain"]
        domain = PolygonDomain(domain_data["domain_id"], domain_data["vertices_xy"], domain_data["crs_wkt"])
        grid = NativeGrid(**payload["grid"])
        AMRSettings(**payload['numerics'].get('amr', {}))
        refinement=payload.get('mesh_refinement',{})
        if refinement.get('zones_path') is not None and not Path(refinement['zones_path']).is_file():
            raise ValueError('Refinement-zone dataset does not exist')
        if not isinstance(refinement.get('zones_layer',REFINEMENT_LAYER_NAME),str):
            raise ValueError('Refinement-zone layer name must be text')
        assignments = [BoundaryAssignment(**row) for row in payload["assignments"]]
        boundaries = {name: CutBoundary(**row) for name, row in payload["boundaries"].items()}
        material = CutSediment(**payload["sediment"])
        fraction_rows=payload.get("sediment_fractions")
        fractions=None if fraction_rows is None else tuple(CutSedimentFraction(**row) for row in fraction_rows)
        if fractions is not None:
            if len(fractions)<2 or len(fractions)>32:
                raise ValueError("Multi-fraction projects require 2 to 32 fractions")
            if len({value.fraction_id for value in fractions})!=len(fractions):
                raise ValueError("Sediment fraction IDs must be unique")
            if not math.isclose(sum(value.initial_volume_fraction for value in fractions),1.,rel_tol=0,abs_tol=1e-12):
                raise ValueError("Initial sediment volume fractions must sum to one")
            active_composition=[value.initial_volume_fraction if value.initial_active_volume_fraction is None
                                else value.initial_active_volume_fraction for value in fractions]
            if not math.isclose(sum(active_composition),1.,rel_tol=0,abs_tol=1e-12):
                raise ValueError("Initial active-layer volume fractions must sum to one")
            if any(value.water_density_kg_m3!=fractions[0].water_density_kg_m3 for value in fractions[1:]):
                raise ValueError("All sediment fractions must use the same water density")
        active_layer=_finite(payload["initial"].get("active_layer_thickness_m",0.),
                             "active-layer thickness",nonnegative=True)
        if active_layer>0 and fractions is None:
            raise ValueError("Active layers require multiple sediment fractions")
        active_mode=payload["initial"].get("active_layer_mode","constant")
        if active_mode not in ("constant","water_depth"):
            raise ValueError("Active-layer mode must be constant or water_depth")
        active_depth_fraction=_finite(payload["initial"].get("active_layer_depth_fraction",0.),
                                      "active-layer depth fraction",nonnegative=True)
        active_minimum=_finite(payload["initial"].get("active_layer_minimum_thickness_m",active_layer),
                               "minimum active-layer thickness",nonnegative=True)
        if active_mode=="water_depth" and (active_layer<=0 or active_depth_fraction<=0 or active_minimum<=0):
            raise ValueError("Water-depth active layers require positive layer, depth fraction, and minimum thickness")
        substrate_count=payload["initial"].get("substrate_layer_count",1)
        if isinstance(substrate_count,bool) or not isinstance(substrate_count,int) or not 1<=substrate_count<=32:
            raise ValueError("Substrate layer count must be an integer from 1 to 32")
        substrate_thickness=_finite(payload["initial"].get("substrate_layer_thickness_m",active_layer),
                                    "preserved substrate layer thickness",nonnegative=True)
        if active_layer>0 and substrate_count>1 and substrate_thickness<=0:
            raise ValueError("Preserved substrate layer thickness must be positive when multiple layers are enabled")
        initial=payload["initial"];water_path=initial.get("water_body_path");initial_stage=initial.get("stage_m")
        if initial_stage is not None:_finite(initial_stage,"initial stage")
        if water_path is not None and (not isinstance(water_path,str) or not water_path.strip()):
            raise ValueError("Initial-water polygon path must be a nonempty path or null")
        water_layer=initial.get("water_body_layer",WATER_BODY_LAYER_NAME)
        if water_path is not None and (not isinstance(water_layer,str) or not water_layer.strip()):
            raise ValueError("Initial-water polygon layer name must be nonempty text")
        if water_path is not None and initial.get("lake_seed_xy") is not None:
            raise ValueError("Choose an initial-water polygon or a legacy lake seed, not both")
        if initial_stage is None and (water_path is not None or initial.get("lake_seed_xy") is not None):
            raise ValueError("An initial water level is required for an initial-water polygon or legacy seed")
        known=set() if fractions is None else {value.fraction_id for value in fractions}
        for name,boundary in boundaries.items():
            supplied=(set(boundary.fraction_concentration_kg_m3)|set(boundary.fraction_bed_input_kg_s)|
                      set(boundary.fraction_sediment_series))
            if fractions is None and supplied:
                raise ValueError(f"Boundary {name} has fraction forcing but the project has one material")
            if supplied-known:
                raise ValueError(f"Boundary {name} names an unknown sediment fraction")
            if fractions is not None and (boundary.concentration_kg_m3 or boundary.bed_input_kg_m_s or
                                           boundary.bed_input_kg_s is not None or boundary.sediment_series is not None):
                raise ValueError("Multi-fraction boundaries require per-fraction sediment fields")
        numerics = payload["numerics"]
        if not isinstance(numerics.get("sediment_enabled",True),bool):
            raise ValueError("sediment_enabled must be true or false")
        if (numerics.get("compact_sidecars") is not None and
                not isinstance(numerics.get("compact_sidecars"),bool)):
            raise ValueError("Compact sidecars must be true, false or automatic")
        roughness = payload.get("roughness", {"default_manning_n": numerics["manning"],
                                                "zones_path": None,
                                                "zones_layer": MANNING_LAYER_NAME})
        roughness_mode=roughness.get("mode","manning")
        if roughness_mode not in ("chezy","manning","white_colebrook"):
            raise ValueError("Hydraulic roughness mode must be chezy, manning, or white_colebrook")
        chzk2d=roughness.get("white_colebrook_chzk2d",False)
        if not isinstance(chzk2d,bool):
            raise ValueError("White-Colebrook ChzK2D switch must be true or false")
        if chzk2d and roughness_mode!="white_colebrook":
            raise ValueError("ChzK2D is valid only with White-Colebrook roughness")
        roughness_default=_finite(
            roughness["default_manning_n"],"default hydraulic roughness parameter",
            positive=roughness_mode!="manning",nonnegative=roughness_mode=="manning")
        recorded_parameter=_finite(
            roughness.get("default_parameter",roughness_default),
            "default hydraulic roughness parameter",
            positive=roughness_mode!="manning",nonnegative=roughness_mode=="manning")
        if recorded_parameter!=roughness_default:
            raise ValueError("Saved hydraulic roughness parameters disagree")
        legacy_default=_finite(
            numerics["manning"],"legacy hydraulic roughness parameter",
            positive=roughness_mode!="manning",nonnegative=roughness_mode=="manning")
        if "roughness" in payload and roughness_default!=legacy_default:
            raise ValueError("Saved default hydraulic roughness values disagree")
        zones_path=roughness.get("zones_path")
        if zones_path is not None and (not isinstance(zones_path,str) or not zones_path.strip()):
            raise ValueError("Roughness-zone path must be a nonempty path or null")
        if not isinstance(roughness.get("zones_layer", MANNING_LAYER_NAME), str) or not roughness.get("zones_layer","").strip():
            raise ValueError("Roughness-zone layer name must be nonempty text")
        shear=payload.get("sediment_shear",{"mode":"total"})
        shear_mode=shear.get("mode","total")
        if shear_mode not in ("total","soulsby_clarke_skin"):
            raise ValueError("Sediment shear mode must be total or soulsby_clarke_skin")
        skin_values=(
            _finite(shear.get("kinematic_viscosity_m2_s",1.e-6),"skin-shear kinematic viscosity",positive=True),
            _finite(shear.get("sand_roughness_m",.003),"skin-shear sand roughness",positive=True),
            _finite(shear.get("silt_roughness_m",.0003),"skin-shear silt roughness",positive=True),
            _finite(shear.get("critical_mud_fraction_low",.2),"skin-shear lower mud fraction",nonnegative=True),
            _finite(shear.get("critical_mud_fraction_high",.8),"skin-shear upper mud fraction",positive=True),
            _finite(shear.get("fluff_fraction",0.),"skin-shear fluff fraction",nonnegative=True))
        if not skin_values[3]<skin_values[4]<=1 or skin_values[5]>1:
            raise ValueError("Invalid Soulsby/Clarke skin-shear mud/fluff fractions")
        bed_inputs = payload.get("bed_inputs", {})
        bed_paths = {name: bed_inputs.get(name) for name in
                     ("bedrock_raster_path", "thickness_raster_path", "zones_path")}
        for name, value in bed_paths.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty path or null")
        if bed_paths["thickness_raster_path"] is not None and bed_paths["zones_path"] is not None:
            raise ValueError("Choose an erodible-thickness raster or polygon zones, not both")
        if not isinstance(bed_inputs.get("zones_layer", ERODIBLE_LAYER_NAME), str) or not bed_inputs.get("zones_layer", ERODIBLE_LAYER_NAME).strip():
            raise ValueError("Erodible-zone layer name must be nonempty text")
        internal=payload.get("internal_conditions",{})
        bank=internal.get("bank_morphology",{"enabled":False})
        bank_failure=internal.get("bank_failure",{"enabled":False})
        general=internal.get("general_avalanche",{"enabled":False})
        secondary=internal.get("secondary_flow",{"enabled":False})
        adaptation=internal.get("bedload_adaptation",{"enabled":False})
        acceleration=internal.get("morphological_acceleration",{"enabled":False})
        bedform=internal.get("bedform_roughness",{"enabled":False})
        obstacle=payload.get("internal_conditions",{}).get("solid_obstacles",{"path":None})
        obstacle_path=obstacle.get("path")
        if obstacle_path is not None and (not isinstance(obstacle_path,str) or not obstacle_path.strip()):
            raise ValueError("Solid-obstacle path must be a nonempty path or null")
        if not isinstance(obstacle.get("layer",OBSTACLE_LAYER_NAME),str) or not obstacle.get("layer",OBSTACLE_LAYER_NAME).strip():
            raise ValueError("Solid-obstacle layer name must be nonempty text")
        if not isinstance(bank.get("enabled",False),bool) or not isinstance(bank.get("avalanche_enabled",True),bool):
            raise ValueError("Bank-morphology switches must be true or false")
        if not isinstance(bank_failure.get("enabled",False),bool):
            raise ValueError("Bank-failure switch must be true or false")
        if not isinstance(bank_failure.get("front_tracking_enabled",False),bool):
            raise ValueError("Persistent bank-front tracking switch must be true or false")
        if not isinstance(general.get("enabled",False),bool):
            raise ValueError("Domain-wide avalanche switch must be true or false")
        if not isinstance(secondary.get("enabled",False),bool):
            raise ValueError("Secondary-flow switch must be true or false")
        secondary_model=_resolved_secondary_flow_model(secondary.get("model","equilibrium"))
        secondary_direction=secondary.get("transport_direction",True)
        if not isinstance(secondary_direction,bool):
            raise ValueError("Secondary-flow transport-direction switch must be true or false")
        secondary_beta=_finite(
            secondary.get("momentum_beta",0.),"secondary-flow momentum Beta",nonnegative=True)
        secondary_method_id=secondary.get("method_id")
        secondary_method_provenance=secondary.get("method_provenance")
        if ((secondary_method_id is None)!=
                (secondary_method_provenance is None)):
            raise ValueError(
                "Secondary-flow method id and provenance must be declared together")
        if secondary_method_id is not None and (
                secondary_model!="prognostic" or
                secondary_method_id!=PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID or
                secondary_method_provenance!=
                    PROGNOSTIC_SECONDARY_PROJECT_PROVENANCE):
            raise ValueError(
                "Classic prognostic secondary-flow method/provenance declaration is invalid")
        classic_betac=_finite(
            secondary.get("classic_betac",0.),"classic FLOW secondary Betac")
        secondary_diffusivity=_finite(
            secondary.get("scalar_diffusivity_m2_s",0.),
            "classic FLOW secondary scalar diffusivity",nonnegative=True)
        if secondary_method_id is None and (
                classic_betac!=0. or secondary_diffusivity!=0.):
            raise ValueError(
                "Classic Betac/diffusivity require an explicit classic "
                "prognostic secondary-flow method declaration")
        if (secondary_method_id==PROGNOSTIC_SECONDARY_PROJECT_METHOD_ID and
                (bank.get("enabled",False) or
                 bank_failure.get("enabled",False) or
                 bank_failure.get("front_tracking_enabled",False) or
                 general.get("enabled",False))):
            raise ValueError(
                "Classic Delft3D4 prognostic secondary flow is outside the "
                "first Q3-H3 live scope with bank erosion, bank failure, "
                "front tracking, or avalanching; it cannot fall back to the "
                "legacy v37 route")
        if not isinstance(adaptation.get("enabled",False),bool):
            raise ValueError("Bed-load adaptation switch must be true or false")
        if not isinstance(acceleration.get("enabled",False),bool):
            raise ValueError("Morphological-acceleration switch must be true or false")
        if not isinstance(bedform.get("enabled",False),bool):
            raise ValueError("Bedform-roughness switch must be true or false")
        bank_path=bank.get("zones_path")
        failure_path=bank_failure.get("zones_path",bank_path)
        if bank_path is not None and (not isinstance(bank_path,str) or not bank_path.strip()):
            raise ValueError("Erodible-bank path must be a nonempty path or null")
        if not isinstance(bank.get("zones_layer",BANK_LAYER_NAME),str) or not bank.get("zones_layer",BANK_LAYER_NAME).strip():
            raise ValueError("Erodible-bank layer name must be nonempty text")
        if failure_path is not None and (not isinstance(failure_path,str) or not failure_path.strip()):
            raise ValueError("Bank-failure zone path must be a nonempty path or null")
        if not isinstance(bank_failure.get("zones_layer",BANK_LAYER_NAME),str) or not bank_failure.get("zones_layer",BANK_LAYER_NAME).strip():
            raise ValueError("Bank-failure zone layer name must be nonempty text")
        bank_fraction=_finite(bank.get("erosion_supply_fraction",.5),"bank erosion supply fraction",nonnegative=True)
        if bank_fraction>1:raise ValueError("Bank erosion supply fraction must not exceed 1")
        _finite(bank.get("wet_maximum_slope",10.),"wet-bank maximum slope",positive=True)
        _finite(bank.get("dry_maximum_slope",1.),"dry-bank maximum slope",positive=True)
        _finite(bank.get("avalanche_time_s",86400.),"bank avalanche time",positive=True)
        _finite(bank.get("relaxation_length_m",1.),"bank relaxation length",positive=True)
        _finite(bank.get("wet_depth_switch_m",.1),"bank wet-depth switch",nonnegative=True)
        if bank.get("enabled",False):
            if not numerics.get("sediment_enabled",True):raise ValueError("Bank morphology requires sediment transport")
            if fractions is None or active_layer<=0:
                raise ValueError("Bank morphology requires multiple fractions and a positive active-layer thickness")
            if bank_path is None:raise ValueError("Bank morphology requires an erodible-bank polygon layer")
            if bank_fraction==0 and not bank.get("avalanche_enabled",True):
                raise ValueError("Enable dry-bank erosion, bank avalanching, or both")
        failure_mode=bank_failure.get("toe_erosion_mode","cohesive_excess_shear")
        if failure_mode not in (
                "cohesive_excess_shear", "noncohesive_transport_capacity",
                "distributed_slope_transport"):
            raise ValueError("Bank-toe erosion mode is invalid")
        failure_capacity=_finite(bank_failure.get("toe_capacity_fraction",1.),
                                 "bank-toe capacity fraction",nonnegative=True)
        if failure_capacity>1:raise ValueError("Bank-toe capacity fraction must not exceed 1")
        failure_distributed_fraction=_finite(
            bank_failure.get("distributed_transport_fraction",.1),
            "distributed bank transport fraction",nonnegative=True)
        if failure_distributed_fraction>1:
            raise ValueError("Distributed bank transport fraction must not exceed 1")
        _finite(bank_failure.get("distributed_critical_slope",.35),
                "distributed bank critical slope",positive=True)
        _finite(bank_failure.get("toe_erodibility_m_pa_s",1.3e-6),"bank toe erodibility",
                positive=failure_mode=="cohesive_excess_shear",
                nonnegative=failure_mode!="cohesive_excess_shear")
        _finite(bank_failure.get("toe_critical_shear_pa",32.5),"bank toe critical shear",nonnegative=True)
        _finite(bank_failure.get("toe_exponent",1.),"bank toe erosion exponent",positive=True)
        _finite(bank_failure.get("critical_undercut_m",.1),"bank critical undercut",positive=True)
        if bank_failure.get("enabled",False):
            if bank.get("enabled",False):
                raise ValueError("Choose process-separated bank failure or legacy bank morphology, not both")
            if not numerics.get("sediment_enabled",True):raise ValueError("Bank failure requires sediment transport")
            if fractions is None or active_layer<=0:
                raise ValueError("Bank failure requires multiple fractions and a positive active-layer thickness")
            if failure_path is None:raise ValueError("Bank failure requires an erodible-bank polygon layer")
            if failure_mode=="cohesive_excess_shear":
                _finite(roughness_default,"Manning coefficient for bank failure",positive=True)
            else:
                if numerics.get("bedload",0)==0:
                    raise ValueError("Noncohesive bank failure requires an active bed-load formulation")
                if any(value.sediment_type!="sand" for value in fractions):
                    raise ValueError("Noncohesive bank failure requires noncohesive sediment fractions")
                if numerics.get("exchange",0)!=0:
                    raise ValueError(
                        "Noncohesive bank failure requires bed load without suspended exchange")
        if bank_failure.get("front_tracking_enabled",False) and (
                not bank_failure.get("enabled",False) or
                failure_mode!="distributed_slope_transport"):
            raise ValueError(
                "Persistent bank-front tracking requires enabled "
                "distributed_slope_transport bank failure")
        _finite(general.get("wet_maximum_slope",10.),"general wet maximum slope",positive=True)
        _finite(general.get("dry_maximum_slope",1.),"general dry maximum slope",positive=True)
        _finite(general.get("avalanche_time_s",86400.),"general avalanche time",positive=True)
        _finite(general.get("relaxation_length_m",1.),"general avalanche relaxation length",positive=True)
        _finite(general.get("wet_depth_switch_m",.1),"general avalanche wet-depth switch",nonnegative=True)
        if general.get("enabled",False):
            if not numerics.get("sediment_enabled",True):
                raise ValueError("Domain-wide avalanching requires sediment transport")
            if fractions is None or active_layer<=0:
                raise ValueError("Domain-wide avalanching requires multiple fractions and a positive active-layer thickness")
            if bank.get("enabled",False) and bank.get("avalanche_enabled",True):
                raise ValueError("Domain-wide and bank-zone avalanching cannot both be enabled")
        _finite(secondary.get("espir",1.),"secondary-flow Espir",nonnegative=True)
        _finite(adaptation.get("length_diameters",100.),"bed-load adaptation length in grain diameters",positive=True)
        factor=_finite(acceleration.get("constant_factor",1.),"morphological acceleration factor",positive=True)
        if not 1<=factor<=20:raise ValueError("Morphological acceleration factor must be within 1 to 20")
        series=acceleration.get("series")
        if series is not None:CutMorphologicalFactorSeries(**series)
        if acceleration.get("enabled",False):
            if not numerics.get("sediment_enabled",True) or numerics.get("bedload",0)==0:
                raise ValueError("Morphological acceleration requires an active bed-load formulation")
            if numerics.get("exchange",0)!=0:
                raise ValueError("Bed-load morphological acceleration is not qualified with suspended exchange")
            if bank.get("enabled",False) or bank_failure.get("enabled",False):
                raise ValueError("Morphological acceleration is not qualified with bank erosion or bank failure")
            if any(row.get("hydraulic_series") is not None for row in payload.get("boundaries",{}).values()):
                raise ValueError("Morphological acceleration is not qualified with time-varying hydraulic boundaries")
        create_values = [numerics[name] for name in ("end_time_s", "output_interval_s", "dt_max_s")]
        if any(_finite(value, "time setting", positive=True) <= 0 for value in create_values):
            raise ValueError
        threads = numerics["threads"]
        if isinstance(threads, bool) or int(threads) != threads or int(threads) < 1:
            raise ValueError("threads must be a positive integer")
        cfl=_finite(numerics.get("cfl",.45),"CFL limit",positive=True)
        if cfl>.9:raise ValueError("CFL limit must not exceed 0.9")
        if (numerics["scheme"] not in (1,5,6) or
                numerics["bedload"] not in (*range(9),11,12,13,14) or
                numerics["exchange"] not in (0,1,2,3)):
            raise ValueError("Unsupported reconstruction or sediment transport mode")
        if shear_mode=="soulsby_clarke_skin":
            if not numerics.get("sediment_enabled",True):
                raise ValueError("Soulsby/Clarke skin shear requires sediment transport")
            if numerics["bedload"]==9:
                raise ValueError("Delft3D BsSkin cannot replace the TELEMAC-GAIA skin correction")
            cohesive_exchange_consumer=(
                numerics["exchange"]==3 and fractions is not None and
                any(value.sediment_type=="mud" for value in fractions))
            if numerics["bedload"] not in (5,6,7) and not cohesive_exchange_consumer:
                raise ValueError(
                    "Soulsby/Clarke skin shear requires an RP_TAUB consumer: law 5/6/7 "
                    "or exchange mode 3 with an actually cohesive fraction")
        if secondary.get("enabled",False):
            if secondary_model=="equilibrium":
                if not secondary_direction or secondary_beta!=0:
                    raise ValueError(
                        "Legacy equilibrium secondary flow has fixed direction-only coupling and zero momentum Beta")
                if not numerics.get("sediment_enabled",True):
                    raise ValueError("Secondary-flow sediment steering requires sediment transport")
                if numerics["bedload"] not in (2,3,5,6,7,8,11,12,14):
                    raise ValueError(
                        "Secondary-flow sediment steering requires a magnitude-based bed-load formulation")
            elif secondary_direction:
                if not numerics.get("sediment_enabled",True):
                    raise ValueError("Secondary-flow sediment steering requires sediment transport")
                if numerics["bedload"] not in (1,2,3,5,6,7,8,11,12,14):
                    raise ValueError(
                        "Prognostic secondary-flow transport direction requires a supported bed-load formulation")
            _finite(payload.get("roughness",{}).get("default_manning_n",numerics["manning"]),
                    "Manning coefficient for secondary flow",positive=True)
        if adaptation.get("enabled",False) and (not numerics.get("sediment_enabled",True) or numerics["bedload"]==0):
            raise ValueError("Bed-load adaptation requires an active bed-load formulation")
        mud_fraction=_finite(numerics.get("critical_mud_fraction",.5),"critical mud fraction",nonnegative=True)
        if mud_fraction>1:raise ValueError("Critical mud fraction must not exceed 1")
        transport_calibration=_finite(
            numerics.get("transport_calibration_factor",1.),
            "transport calibration factor",nonnegative=True)
        if numerics["bedload"] in SPLIT_CAPACITY_LAWS and transport_calibration!=1.:
            raise ValueError(
                "Parameterized split-capacity laws use per-formula coefficients; "
                "the legacy global transport calibration must remain 1")
        if numerics["bedload"] in SPLIT_CAPACITY_LAWS and numerics["exchange"]!=0:
            raise ValueError(
                "The current split-capacity route retains the suspended formula "
                "channel but does not yet couple it; suspended exchange must be off")
        if numerics["bedload"]==8 and transport_calibration!=1.:
            raise ValueError("Engelund-Hansen T2 scope currently requires transport calibration 1")
        eh_chezy=numerics.get("engelund_hansen_chezy")
        if numerics["bedload"]==8:
            if numerics.get("capacity_method_id")==EH_DELFT_METHOD_ID:
                if "engelund_hansen_chezy" in numerics:
                    raise ValueError(
                        "Corrected Delft Engelund-Hansen cannot retain an "
                        "independent Chezy field")
            else:
                _finite(eh_chezy,"Engelund-Hansen constant Chezy",positive=True)
            if numerics["exchange"]!=0:
                raise ValueError(
                    "Engelund-Hansen law 8 is qualified only with suspended exchange off (mode 0)")
            if numerics["bed_slope"] not in (0,2,3,5):
                raise ValueError(
                    "Engelund-Hansen law 8 slope correction must be off, legacy 2, Koch-Flokstra 3, or Delft3D FLOW 5")
        elif eh_chezy is not None:
            _finite(eh_chezy,"Engelund-Hansen constant Chezy",positive=True)
        if fractions is None and numerics["exchange"]==3:
            raise ValueError("Cohesive or mixed exchange requires at least two sediment fractions")
        if (numerics["bedload"] in (3,4,5,6,7) or numerics["exchange"]==3) and (fractions is None or active_layer<=0):
            raise ValueError("Extended current-driven or cohesive sediment laws require multiple fractions and a positive active-layer thickness")
        if fractions is not None and any(value.sediment_type=='mud' for value in fractions) and numerics["exchange"]!=3:
            raise ValueError("Mud fractions require cohesive or mixed suspended exchange")
        hiding=numerics.get("hiding_exposure",0)
        if isinstance(hiding,bool) or not isinstance(hiding,int) or hiding not in (0,2,3,4,5):
            raise ValueError("Hiding/exposure must be disabled (0) or a Delft3D mode from 2 to 5")
        _finite(numerics.get("hiding_parker_alpha",.6),"Parker hiding/exposure alpha")
        _finite(numerics.get("hiding_wu_exponent",-.6),"Wu hiding/exposure exponent")
        if hiding and (fractions is None or active_layer<=0):
            raise ValueError(
                "Q3-H1 hiding requires exactly two class-1 fractions and an active surface bed layer")
        if hiding and numerics["bedload"] not in (2,11):
            raise ValueError(
                "Q3-H1 hiding supports corrected MPM law 2 or general law 11")
        if hiding and fractions is not None and len(fractions)!=2:
            raise ValueError("Q3-H1 hiding requires exactly two class-1 fractions")
        if hiding and numerics["exchange"]!=0:
            raise ValueError("Q3-H1 hiding requires exchange off and class-1 fractions")
        if hiding and numerics["bed_slope"]!=0:
            raise ValueError(
                "Q3-H1 hiding requires bed-slope transport correction off")
        if hiding and secondary.get("enabled",False) and (
                secondary_model not in ("equilibrium","prognostic") or
                not secondary_direction or secondary_beta!=0.):
            raise ValueError(
                "Q3-H1 can compose only with the Q3-H2/H3 zero-Betac secondary routes")
        if numerics["bed_slope"] not in (0,2,3,5,6):
            raise ValueError("Slope-correction mode must be off, legacy Bagnold/Ikeda 2, Koch-Flokstra 3, Delft3D FLOW 5, or Parker-Andrews 6")
        fraction=_finite(numerics.get("minimum_boundary_cell_area_fraction",0.),"minimum boundary-cell area fraction",nonnegative=True)
        if fraction>.5:
            raise ValueError("Minimum boundary-cell area fraction must not exceed 0.5")
        mpm_cf=_finite(numerics.get("mpm_cf",.01),"MPM friction coefficient",nonnegative=True)
        mpm_critical=_finite(numerics.get("mpm_critical_shields",0.),"MPM critical Shields",nonnegative=True)
        mpm_mode=numerics.get("mpm_friction_mode",0)
        if isinstance(mpm_mode,bool) or not isinstance(mpm_mode,int) or mpm_mode not in (0,1):
            raise ValueError("MPM friction mode must be specified coefficient (0) or roughness-corrected (1)")
        if numerics.get("sediment_enabled",True) and mpm_mode and numerics["bedload"]!=2:
            raise ValueError("Roughness-corrected MPM requires MPM bed load")
        if numerics.get("sediment_enabled",True) and mpm_mode and not math.isclose(mpm_critical,.047,rel_tol=0,abs_tol=1e-14):
            raise ValueError("Roughness-corrected MPM requires the source critical Shields value 0.047")
        if bedform.get("enabled",False):
            if roughness_mode!="manning":
                raise ValueError("The current bedform-feedback operator requires Manning hydraulic roughness")
            if not numerics.get("sediment_enabled",True) or numerics["bedload"]!=2 or mpm_mode!=1:
                raise ValueError("Bedform roughness requires roughness-corrected MPM bed load")
            if numerics["exchange"]!=0:
                raise ValueError("Bedform roughness is not qualified with suspended exchange")
            composed_boundary = (next(iter(payload.get("boundaries", {}).values()))
                                 if len(payload.get("boundaries", {})) == 1 else None)
            recursive_bank_bedform = (
                numerics.get("amr", {}).get("enabled") is True and
                numerics.get("amr", {}).get("subcycling") is True and
                bank_failure.get("enabled") is True and
                failure_mode in (
                    "noncohesive_transport_capacity",
                    "distributed_slope_transport") and
                bank_failure.get("front_tracking_enabled", False) is False and
                general.get("enabled") is True and bank.get("enabled") is False and
                secondary.get("enabled") is False and adaptation.get("enabled") is False and
                acceleration.get("enabled") is False and
                numerics["bedload"] == 2 and numerics["exchange"] == 0 and
                mpm_mode == 1 and numerics.get("bed_slope", 0) == 0 and
                numerics.get("hiding_exposure", 0) == 0 and
                numerics.get("horizontal_viscosity_m2_s", 0.) == 0. and
                payload.get("initial", {}).get("active_layer_mode", "constant") == "constant" and
                payload.get("initial", {}).get("substrate_layer_count") == 1 and
                composed_boundary is not None and composed_boundary.get("mode") == "stage" and
                composed_boundary.get("hydraulic_series") is None and
                composed_boundary.get("sediment_series") is None and
                not composed_boundary.get("fraction_sediment_series") and
                not composed_boundary.get("fraction_concentration_kg_m3") and
                composed_boundary.get("concentration_kg_m3", 0.) == 0. and
                composed_boundary.get("bedload_boundary_mode", "prescribed") !=
                    "equilibrium_capacity" and
                composed_boundary.get("bed_input_kg_m_s", 0.) == 0. and
                composed_boundary.get("bed_input_kg_s") in (None, 0.) and
                not composed_boundary.get("fraction_bed_input_kg_s"))
            if (any(setting.get("enabled",False) for setting in
                    (secondary,adaptation,acceleration,bank,bank_failure,general)) and
                    not recursive_bank_bedform):
                raise ValueError("Bedform roughness is not qualified with another experimental morphology process")
            if fractions is not None and any(value.sediment_type!='sand' for value in fractions):
                raise ValueError("Bedform roughness requires noncohesive sediment fractions")
            _finite(roughness_default,"Manning coefficient for bedform roughness",positive=True)
            if any(row.get("mode")=='inlet_normal' for row in payload.get("boundaries",{}).values()):
                raise ValueError("Bedform roughness is not qualified with normal-depth inlet control")
        alfabs=_finite(numerics.get("bed_slope_alfabs",1.),"AlfaBs",nonnegative=True)
        alfabn=_finite(numerics.get("bed_slope_alfabn",1.5),"AlfaBn",nonnegative=True)
        friction_angle=_finite(numerics.get("bed_friction_angle_deg",30.),"bed friction angle")
        _finite(numerics["settling_velocity_m_s"],"settling velocity",nonnegative=True)
        viscosity=_finite(numerics.get('horizontal_viscosity_m2_s',0.),'horizontal viscosity',nonnegative=True)
        formulation=numerics.get('horizontal_viscosity_formulation','depth_conservative')
        if formulation not in ('depth_conservative','velocity_laplacian'):
            raise ValueError('Unsupported horizontal viscosity formulation')
        if formulation=='velocity_laplacian' and viscosity<=0.:
            raise ValueError('Velocity-Laplacian viscosity requires a positive coefficient')
        settling_mode=numerics.get("settling_mode",0)
        if isinstance(settling_mode,bool) or not isinstance(settling_mode,int) or settling_mode not in (0,1):
            raise ValueError("Settling treatment must be constant (0) or concentration-dependent (1)")
        _finite(numerics.get("hindered_settling_reference_concentration_kg_m3",
                             DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3),
                "hindered-settling reference concentration",positive=True)
        if numerics["bed_slope"] in (2,5) and not 0<friction_angle<90:
            raise ValueError("Bed friction angle must lie between 0 and 90 degrees")
        if numerics["bed_slope"]==3 and friction_angle<0:
            raise ValueError("Koch-Flokstra BShld must be nonnegative")
        if numerics["bed_slope"] and numerics["bedload"]==0:
            raise ValueError("Slope correction requires an active bed-load formulation")
        if numerics["bed_slope"] in (3,5) and numerics["bedload"] not in (2,8):
            raise ValueError("Delft3D FLOW slope input contract requires MPM or Engelund-Hansen (law 2/8)")
        if numerics.get("sediment_enabled",True) and numerics["bedload"]==8:
            _finite(roughness_default,"Manning coefficient for Engelund-Hansen",positive=True)
        _validate_geometry_contract(domain, assignments, boundaries, grid)
        if not isinstance(material, CutSediment):
            raise ValueError
        for key in ("workspace", "native_executable"):
            if not isinstance(payload[key], str) or not payload[key].strip():
                raise ValueError(f"{key} must be a path")
        if not isinstance(payload["terrain"]["dem_path"], str):
            raise ValueError("DEM path is missing")
        section_dem_path = payload["terrain"].get("section_dem_path")
        if section_dem_path is not None and not isinstance(section_dem_path, str):
            raise ValueError("Inlet-section DEM path must be a path")
        if payload["terrain"].get("interpretation", "bilinear_surface") not in (
                "bilinear_surface", "cell_average_raster"):
            raise ValueError(
                "Terrain interpretation must be bilinear_surface or cell_average_raster"
            )
        _finite(
            payload["terrain"].get("section_vertical_tolerance_m", 0.),
            "inlet-section vertical tolerance", nonnegative=True,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc):
            raise
        raise ValueError("Invalid BRAIDS project is incomplete or invalid") from exc
    validated=dict(payload)
    raw_numerics=payload["numerics"]
    method_identity_missing="capacity_method_id" not in raw_numerics
    resolved_capacity_method=_capacity_execution_method_id(raw_numerics)
    if CAPACITY_EXECUTION_BINDING_KEY in payload:
        capacity_execution_binding=_validated_capacity_execution_binding(payload)
    else:
        capacity_execution_binding=_build_capacity_execution_binding(
            payload,legacy_missing_method_identity=method_identity_missing)
    validated["numerics"]={**raw_numerics}
    if resolved_capacity_method is not None:
        validated["numerics"]["capacity_method_id"]=resolved_capacity_method
    validated[CAPACITY_EXECUTION_BINDING_KEY]=capacity_execution_binding
    roughness=dict(validated.get("roughness",{}))
    roughness_mode=roughness.get("mode","manning")
    parameter_name,parameter_units,parameter_label=_ROUGHNESS_PUBLIC[roughness_mode]
    roughness.setdefault("default_manning_n",validated["numerics"]["manning"])
    roughness.setdefault("default_parameter",roughness["default_manning_n"])
    roughness.update(
        parameter_name=parameter_name,parameter_units=parameter_units,
        parameter_label=parameter_label,
        legacy_storage={
            "field":"default_manning_n",
            "physical_type":roughness_mode=="manning",
            "note":("physical Manning n" if roughness_mode=="manning" else
                    "compatibility alias containing the selected raw parameter; not Manning n")})
    validated["roughness"]=roughness
    if legacy_project:
        validated["format"]=PROJECT_FORMAT
        validated[CAPACITY_BINDING_KEY]=_build_capacity_binding(
            validated,None,require_split_parameters=False)
        existing_notes=validated.get("compatibility_notes",[])
        if not isinstance(existing_notes,list) or not all(
                isinstance(value,str) for value in existing_notes):
            raise ValueError("Project compatibility notes must be a list of text")
        state=validated[CAPACITY_BINDING_KEY]["state"]
        note=(
            "Migrated braids-qgis-project-v1 capacity settings to a backend-neutral "
            "canonical runtime profile. CPU/GPU route support will be checked again "
            "at preparation time."
            if state=="canonical" else
            "Migrated braids-qgis-project-v1 with sediment capacity inactive. No "
            "formula profile or synthetic fraction identity was created."
            if state=="inactive" else
            "Migrated braids-qgis-project-v1 while preserving its legacy capacity "
            f"route as {state}; no fixed Delft3D coefficient profile was forged."
        )
        migration_notes=[note]
        if legacy_eh_defaulted:
            migration_notes.append(
                "Migrated braids-qgis-project-v1 with the historical inactive "
                "Engelund-Hansen Chezy default 28.8 recorded explicitly; it affects "
                "physics only when bed-load law 8 is selected.")
        if method_identity_missing and resolved_capacity_method in _Q2_CAPACITY_METHODS:
            migration_notes.append(
                "Migrated an integer-only capacity setting to explicit execution identity "
                f"{resolved_capacity_method}; its historical numerical closure was preserved.")
        validated["compatibility_notes"]=[*existing_notes,
            *(value for value in migration_notes if value not in existing_notes)]
    else:
        if CAPACITY_EXECUTION_BINDING_KEY not in payload:
            old_binding=payload.get(CAPACITY_BINDING_KEY,{})
            supplied=(old_binding.get("capacity_parameters_by_fraction")
                      if isinstance(old_binding,Mapping) and
                      old_binding.get("state")=="canonical" else None)
            validated[CAPACITY_BINDING_KEY]=_build_capacity_binding(
                validated,supplied,require_split_parameters=False)
        else:
            validated[CAPACITY_BINDING_KEY]=_validated_capacity_binding(validated)
        if CAPACITY_EXECUTION_BINDING_KEY not in payload:
            notes=validated.get("compatibility_notes",[])
            if not isinstance(notes,list) or not all(isinstance(value,str) for value in notes):
                raise ValueError("Project compatibility notes must be a list of text")
            note=(
                "Migrated an integer-only capacity setting to explicit execution identity "
                f"{resolved_capacity_method}; its historical numerical closure was preserved."
                if resolved_capacity_method in _Q2_CAPACITY_METHODS else
                "Added the Q2 capacity-execution scope declaration; this formula is outside "
                "the historically ambiguous MPM/Engelund-Hansen pair.")
            validated["compatibility_notes"]=[*notes,
                *(value for value in (note,) if value not in notes)]
    hiding_foundation_binding=_validated_hiding_foundation_binding(validated)
    if hiding_foundation_binding is None:
        validated.pop(HIDING_FOUNDATION_BINDING_KEY,None)
    else:
        validated[HIDING_FOUNDATION_BINDING_KEY]=hiding_foundation_binding
    secondary_foundation_binding=_validated_secondary_foundation_binding(
        validated)
    if secondary_foundation_binding is None:
        validated.pop(SECONDARY_FOUNDATION_BINDING_KEY,None)
    else:
        validated[SECONDARY_FOUNDATION_BINDING_KEY]=secondary_foundation_binding
    prognostic_secondary_binding=_validated_prognostic_secondary_binding(
        validated)
    if prognostic_secondary_binding is None:
        validated.pop(PROGNOSTIC_SECONDARY_BINDING_KEY,None)
    else:
        validated[PROGNOSTIC_SECONDARY_BINDING_KEY]=prognostic_secondary_binding
    raw_execution_profile=payload.get(EXECUTION_PROFILE_BINDING_KEY)
    if raw_execution_profile is None:
        try:
            migrated_foundation=resolve_project_capacity_foundation_profile(validated)
        except CapacityFoundationProfileError as exc:
            if exc.category!="combination_not_implemented":
                raise ValueError(str(exc)) from exc
            migrated_foundation=None
        if migrated_foundation is not None:
            validated[EXECUTION_PROFILE_BINDING_KEY]=dict(
                migrated_foundation.project_binding)
            validated[CAPACITY_FOUNDATION_RESOLUTION_KEY]=dict(
                migrated_foundation.resolution_receipt)
        else:
            validated[EXECUTION_PROFILE_BINDING_KEY]=legacy_execution_profile_binding(
                "schema_v1_project" if legacy_project else
                "project_predates_explicit_schema_v2_profile")
            notes=validated.get("compatibility_notes",[])
            if not isinstance(notes,list) or not all(isinstance(value,str) for value in notes):
                raise ValueError("Project compatibility notes must be a list of text")
            note=(
                "Added an additive schema-v2 sediment execution-profile placeholder. "
                "Existing schema-v1 numerical settings and restart identities remain unchanged; "
                "an explicit complete profile is required only for v52 authenticated routes."
            )
            validated["compatibility_notes"]=[*notes,
                *(value for value in (note,) if value not in notes)]
    else:
        try:
            checked_execution_profile,execution_profile=load_execution_profile_binding(
                raw_execution_profile)
        except ExecutionProfileError as exc:
            raise ValueError(str(exc)) from exc
        if execution_profile is not None:
            documents=checked_execution_profile["component_documents"]
            capacity_document=validated[CAPACITY_BINDING_KEY].get(
                "capacity_runtime_profile")
            if ("capacity" in documents and
                    documents["capacity"]!=capacity_document):
                raise ValueError(
                    "Schema-v2 capacity component differs from the project capacity profile")
        validated[EXECUTION_PROFILE_BINDING_KEY]=checked_execution_profile
        if execution_profile is None:
            try:
                migrated_foundation=resolve_project_capacity_foundation_profile(
                    validated)
            except CapacityFoundationProfileError as exc:
                if exc.category!="combination_not_implemented":
                    raise ValueError(str(exc)) from exc
                migrated_foundation=None
            if migrated_foundation is not None:
                validated[EXECUTION_PROFILE_BINDING_KEY]=dict(
                    migrated_foundation.project_binding)
                validated[CAPACITY_FOUNDATION_RESOLUTION_KEY]=dict(
                    migrated_foundation.resolution_receipt)
        if _is_historical_q3sp_general_profile(checked_execution_profile):
            try:
                historical_capacity=resolve_project_capacity_foundation_profile(
                    validated)
            except CapacityFoundationProfileError as exc:
                raise ValueError(str(exc)) from exc
            if (historical_capacity is None or
                    historical_capacity.resolution_receipt.get(
                        "capacity_method_id")!=GENERAL_METHOD_ID):
                raise ValueError(
                    "Historical Q3-S/P law-11 profile differs from the project capacity")
            notes=validated.get("compatibility_notes",[])
            if not isinstance(notes,list) or not all(
                    isinstance(value,str) for value in notes):
                raise ValueError("Project compatibility notes must be a list of text")
            validated["compatibility_notes"]=[*notes,
                *(value for value in (_HISTORICAL_Q3SP_NOTE,) if value not in notes)]
    resolution_receipt=payload.get(CAPACITY_FOUNDATION_RESOLUTION_KEY)
    if resolution_receipt is not None:
        try:
            automatic=resolve_project_capacity_foundation_profile(validated)
        except CapacityFoundationProfileError as exc:
            raise ValueError(str(exc)) from exc
        if automatic is None:
            raise ValueError(
                "Capacity-foundation receipt is present for an ineligible formula")
        automatic_binding=dict(automatic.project_binding)
        if validated[EXECUTION_PROFILE_BINDING_KEY]==automatic_binding:
            try:
                validate_capacity_foundation_resolution(
                    validated,resolution_receipt)
            except CapacityFoundationProfileError as exc:
                raise ValueError(str(exc)) from exc
            validated[CAPACITY_FOUNDATION_RESOLUTION_KEY]=dict(
                automatic.resolution_receipt)
        else:
            # A complete explicitly authored schema-v2 profile is a legitimate
            # override.  Its own fingerprint remains authoritative, while the
            # obsolete automatic-choice explanation is removed rather than
            # misdescribing the saved physics.
            required_roughness=automatic.resolution_receipt.get(
                "direct_roughness_dependency")
            historical_q3sp=(
                automatic.resolution_receipt.get("capacity_method_id")==
                    GENERAL_METHOD_ID and
                _is_historical_q3sp_general_profile(checked_execution_profile))
            if required_roughness is not None:
                explicit_profile=checked_execution_profile.get("profile")
                if not isinstance(explicit_profile,Mapping):
                    raise ValueError(
                        "Q3-C capacity cannot discard its complete "
                        "schema-v2 execution profile")
                selected_roughness={
                    row.get("entry_id") for row in
                    explicit_profile.get("method_selections",[])
                    if isinstance(row,Mapping) and
                    isinstance(row.get("entry_id"),str) and
                    row["entry_id"].startswith("roughness.")}
                if (selected_roughness!={required_roughness} and
                        not historical_q3sp):
                    raise ValueError(
                        "Schema-v2 Q3-C capacity roughness dependency "
                        "differs from the selected project C/n/ks method")
            validated.pop(CAPACITY_FOUNDATION_RESOLUTION_KEY,None)
            notes=validated.get("compatibility_notes",[])
            if not isinstance(notes,list) or not all(
                    isinstance(value,str) for value in notes):
                raise ValueError("Project compatibility notes must be a list of text")
            note=(
                _HISTORICAL_Q3SP_NOTE
                if historical_q3sp else
                "An explicitly authored schema-v2 sediment profile superseded "
                "the automatic Q3-C capacity-foundation resolution receipt.")
            validated["compatibility_notes"]=[*notes,
                *(value for value in (note,) if value not in notes)]
    # A canonical eight-selector Q3-C profile must remain tied to the project
    # C/n/ks method even when a caller removes the optional explanation
    # receipt.  The profile itself is authoritative; only the historical
    # seven-selector Q3-S/P law-11 identity is exempt.
    checked=validated.get(EXECUTION_PROFILE_BINDING_KEY)
    if (isinstance(checked,Mapping) and checked.get("state")=="canonical" and
            not _is_historical_q3sp_general_profile(checked)):
        try:
            automatic=resolve_project_capacity_foundation_profile(validated)
        except CapacityFoundationProfileError as exc:
            if exc.category!="combination_not_implemented":
                raise ValueError(str(exc)) from exc
            automatic=None
        if automatic is not None:
            required=automatic.resolution_receipt.get(
                "direct_roughness_dependency")
            selected={row.get("entry_id") for row in
                      checked.get("profile",{}).get("method_selections",[])
                      if isinstance(row,Mapping) and
                      isinstance(row.get("entry_id"),str) and
                      row["entry_id"].startswith("roughness.")}
            if required is not None and selected!={required}:
                raise ValueError(
                    "Schema-v2 Q3-C capacity roughness dependency differs "
                    "from the selected project C/n/ks method")
    if QGIS_PRODUCT_KEY in validated:
        validate_qgis_product(validated)
    return validated


def save_project(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validated = _validated_project(payload)
    validated["numerics"]={**validated["numerics"],
        "sediment_method_selection":sediment_method_metadata(validated["numerics"])}
    path = Path(path).expanduser().resolve();path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def reject_raw_gpu_persistent_front(path: str | Path) -> None:
    """Reject a GPU-incompatible raw request before legacy load migration.

    CPU project loading intentionally preserves its historical compatibility
    migration. GPU preparation must inspect the saved request *before* that
    migration can silently turn persistent-front tracking off.
    """
    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read BRAIDS project: {path}") from exc
    if payload.get("internal_conditions", {}).get("bank_failure", {}).get(
            "front_tracking_enabled", False):
        raise ValueError("GPU persistent bank-front tracking is excluded; the saved project requests it explicitly")


def load_project(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read BRAIDS project: {path}") from exc
    validated = _validated_project(payload)
    validated["numerics"]={**validated["numerics"],
        "sediment_method_selection":sediment_method_metadata(validated["numerics"])}
    bank = validated.get("internal_conditions", {}).get("bank_failure", {})
    if bank.get("front_tracking_enabled", False):
        bank["front_tracking_enabled"] = False
        validated.setdefault("compatibility_notes", []).append(
            "Retired persistent bank-front tracking was disabled. New runs use "
            "conservative bed-elevation bank evolution; archived v41 runs and "
            "checkpoints remain unchanged and cannot be resumed with this solver.")
    return validated


def _ogr_polygon(vertices, holes=()) -> ogr.Geometry:
    polygon = ogr.Geometry(ogr.wkbPolygon)
    for points in (vertices,*holes):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in points: ring.AddPoint_2D(float(x), float(y))
        polygon.AddGeometry(ring)
    return polygon


def terrain_averages(mesh: CutCellMesh, dem_path: str | Path,
                     progress: Callable[[float], None] | None = None,
                     cancelled: Callable[[], bool] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact area average of a north-up source raster over every cut cell."""
    dataset = gdal.OpenEx(str(dem_path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None or dataset.RasterCount < 1:
        raise ValueError(f"Cannot open DEM: {dem_path}")
    transform = dataset.GetGeoTransform(can_return_null=True)
    if transform is None or transform[1] <= 0 or transform[5] >= 0 or transform[2] != 0 or transform[4] != 0:
        raise ValueError("The first QGIS path requires a north-up DEM without rotation or shear")
    spatial = dataset.GetSpatialRef();mesh_spatial = osr.SpatialReference();mesh_spatial.ImportFromWkt(mesh.crs_wkt)
    if spatial is None or not spatial.IsSame(mesh_spatial):
        raise ValueError("DEM and polygon CRS must match")
    band = dataset.GetRasterBand(1);values = band.ReadAsArray()
    mask = None if band.GetMaskFlags() & gdal.GMF_ALL_VALID else band.GetMaskBand().ReadAsArray()
    nodata = band.GetNoDataValue();scale = 1. if band.GetScale() is None else band.GetScale();offset = 0. if band.GetOffset() is None else band.GetOffset()
    pixel_width,pixel_height=transform[1],-transform[5];pixel_area=pixel_width*pixel_height
    result=np.empty(len(mesh.cells));partition=[];fast_cells=0
    for index,cell in enumerate(mesh.cells):
        if cancelled and cancelled(): raise InterruptedError("BRAIDS preparation cancelled")
        coordinates=np.asarray(cell.vertices_xy)
        pixel=np.column_stack(((coordinates[:,0]-transform[0])/pixel_width,(transform[3]-coordinates[:,1])/pixel_height))
        lo=np.floor(pixel.min(axis=0)+1e-12).astype(int);hi=np.ceil(pixel.max(axis=0)-1e-12).astype(int)
        if np.any(lo<0) or hi[0]>dataset.RasterXSize or hi[1]>dataset.RasterYSize:
            raise ValueError(f"DEM does not cover cut cell {cell.cell_id}")
        xmin,ymin=pixel.min(axis=0);xmax,ymax=pixel.max(axis=0)
        rectangular=(math.isclose(cell.area_m2,(xmax-xmin)*(ymax-ymin)*pixel_area,
                                  rel_tol=2e-12,abs_tol=2e-12*pixel_area)
                     and all((math.isclose(x,xmin,abs_tol=1e-11) or math.isclose(x,xmax,abs_tol=1e-11))
                             and (math.isclose(y,ymin,abs_tol=1e-11) or math.isclose(y,ymax,abs_tol=1e-11))
                             for x,y in pixel))
        weighted=0.;weight_sum=0.
        if rectangular:
            # Rectangular solver cells have separable, exact source-pixel
            # overlaps. This avoids GEOS calls for the bulk of a large domain.
            columns=np.arange(lo[0],hi[0]);rows=np.arange(lo[1],hi[1])
            if len(columns)==1 and len(rows)==1:
                raw=float(values[rows[0],columns[0]])
                invalid=(not math.isfinite(raw) or
                         (nodata is not None and raw==nodata) or
                         (mask is not None and mask[rows[0],columns[0]]==0))
                if invalid:
                    raise ValueError(f"DEM contains invalid data beneath cut cell {cell.cell_id}")
                # An aligned cell-average raster is already the exact cell
                # integral divided by area. Preserve its bits directly and
                # avoid needless multiply/divide roundoff; this path is also
                # faster for benchmark-sized Cartesian grids.
                result[index]=raw*scale+offset
                partition.append(0.);fast_cells+=1
                if progress and (index%max(1,len(mesh.cells)//100)==0 or index+1==len(mesh.cells)):
                    progress(100*(index+1)/len(mesh.cells))
                continue
            xweights=np.maximum(0.,np.minimum(columns+1.,xmax)-np.maximum(columns,xmin))*pixel_width
            yweights=np.maximum(0.,np.minimum(rows+1.,ymax)-np.maximum(rows,ymin))*pixel_height
            weights=np.outer(yweights,xweights);raw=np.asarray(values[lo[1]:hi[1],lo[0]:hi[0]],dtype=float)
            invalid=~np.isfinite(raw)
            if nodata is not None:invalid |= raw==nodata
            if mask is not None:invalid |= mask[lo[1]:hi[1],lo[0]:hi[0]]==0
            if np.any(invalid & (weights>0)):
                raise ValueError(f"DEM contains invalid data beneath cut cell {cell.cell_id}")
            weighted=float(np.sum(weights*(raw*scale+offset)));weight_sum=float(weights.sum());fast_cells+=1
        else:
            hole_pixels=[np.column_stack(((np.asarray(hole)[:,0]-transform[0])/pixel_width,
                                          (transform[3]-np.asarray(hole)[:,1])/pixel_height))
                         for hole in cell.holes_xy]
            polygon=_ogr_polygon(pixel,hole_pixels)
            for row in range(lo[1],hi[1]):
                for column in range(lo[0],hi[0]):
                    box=_ogr_polygon([(column,row),(column+1,row),(column+1,row+1),(column,row+1),(column,row)])
                    weight=polygon.Intersection(box).GetArea()*pixel_area
                    if weight<=0:continue
                    raw=float(values[row,column])
                    if (not math.isfinite(raw) or (nodata is not None and raw==nodata) or
                            (mask is not None and mask[row,column]==0)):
                        raise ValueError(f"DEM contains invalid data beneath cut cell {cell.cell_id}")
                    weighted += weight*(raw*scale+offset);weight_sum += weight
        if weight_sum<=0:raise ValueError(f"DEM overlap is empty for cut cell {cell.cell_id}")
        partition.append(weight_sum-cell.area_m2);result[index]=weighted/weight_sum
        if progress and (index%max(1,len(mesh.cells)//100)==0 or index+1==len(mesh.cells)):
            progress(100*(index+1)/len(mesh.cells))
    return result,{"method":"exact source-pixel area average (analytic rectangles; intersections for cut cells)",
                   "analytic_rectangle_cells":fast_cells,"intersection_cells":len(mesh.cells)-fast_cells,
                   "maximum_partition_error_m2":max(map(abs,partition)),
                   "minimum_bed_m":float(result.min()),"maximum_bed_m":float(result.max()),
                   "gdal_scale":scale,"gdal_offset":offset}


def initial_state(mesh: CutCellMesh, bed: np.ndarray, settings: Mapping[str, Any], material: CutSediment,
                  *, thickness_by_cell: np.ndarray | None = None,
                  bedrock_by_cell: np.ndarray | None = None,
                  sediment_fractions: Sequence[CutSedimentFraction] | None = None,
                  water_body_mask: np.ndarray | None = None):
    stage_value=settings.get("stage_m")
    stage=None if stage_value is None else _finite(stage_value,"initial stage")
    wet=np.zeros(bed.shape,dtype=bool) if stage is None else bed<stage;seed=settings.get("lake_seed_xy")
    connected=wet.copy()
    if water_body_mask is not None:
        mask=np.asarray(water_body_mask)
        if mask.shape!=wet.shape:
            raise ValueError("Initial-water polygon mask must contain one value per retained cell")
        connected=wet & mask.astype(bool)
        if not connected.any():
            raise ValueError("Initial-water polygons contain no retained cell below the initial water level")
    elif seed is not None:
        point=ogr.Geometry(ogr.wkbPoint);point.AddPoint_2D(*map(float,seed))
        sx,sy=map(float,seed);i=math.floor((sx-mesh.grid.xlower_m)/mesh.grid.dx_m)
        j=math.floor((sy-mesh.grid.ylower_m)/mesh.grid.dy_m)
        candidates=(cell for cell in mesh.cells if cell.i==i and cell.j==j)
        seeds=[cell.cell_id for cell in candidates
               if cut_cell_polygon(cell).Intersects(point) and wet[cell.cell_id]]
        if not seeds:raise ValueError("Lake seed point is not inside a retained cell below the initial stage")
        adjacency=[[] for _ in mesh.cells]
        for face in mesh.faces:
            if face.neighbour>=0:
                adjacency[face.owner].append(face.neighbour);adjacency[face.neighbour].append(face.owner)
        connected=np.zeros(len(mesh.cells),dtype=bool);queue=deque(seeds);connected[seeds]=True
        while queue:
            for neighbour in adjacency[queue.popleft()]:
                if wet[neighbour] and not connected[neighbour]:connected[neighbour]=True;queue.append(neighbour)
    fractions=None if sediment_fractions is None else tuple(sediment_fractions)
    active_layer=_finite(settings.get("active_layer_thickness_m",0.),"active-layer thickness",nonnegative=True)
    layered=fractions is not None and active_layer>0
    if active_layer>0 and fractions is None:raise ValueError("Active layers require multiple sediment fractions")
    active_mode=settings.get("active_layer_mode","constant")
    if active_mode not in ("constant","water_depth"):
        raise ValueError("Active-layer mode must be constant or water_depth")
    active_depth_fraction=_finite(settings.get("active_layer_depth_fraction",0.),
                                  "active-layer depth fraction",nonnegative=True)
    active_minimum=_finite(settings.get("active_layer_minimum_thickness_m",active_layer),
                           "minimum active-layer thickness",nonnegative=True)
    if active_mode=="water_depth" and (active_layer<=0 or active_depth_fraction<=0 or active_minimum<=0):
        raise ValueError("Water-depth active layers require positive layer, depth fraction, and minimum thickness")
    substrate_count=settings.get("substrate_layer_count",1)
    if isinstance(substrate_count,bool) or not isinstance(substrate_count,int) or not 1<=substrate_count<=32:
        raise ValueError("Substrate layer count must be an integer from 1 to 32")
    substrate_target=_finite(settings.get("substrate_layer_thickness_m",active_layer),
                             "preserved substrate layer thickness",nonnegative=True)
    if layered and substrate_count>1 and substrate_target<=0:
        raise ValueError("Preserved substrate layer thickness must be positive when multiple layers are enabled")
    bed_layers=1+(substrate_count if layered else 0)
    q=np.zeros((len(mesh.cells),5 if fractions is None else 3+(1+bed_layers)*len(fractions)))
    q[:,0]=0. if stage is None else np.where(connected,stage-bed,0.)
    active_target=(np.maximum(active_depth_fraction*np.maximum(q[:,0],0.),active_minimum)
                   if active_mode=="water_depth" else np.full(len(mesh.cells),active_layer))
    if thickness_by_cell is None and bedrock_by_cell is None:
        thickness=_finite(settings["erodible_thickness_m"],"erodible thickness",nonnegative=True)
        minimum=_finite(settings["erodible_minimum_depth_m"],"minimum erodible depth",nonnegative=True)
        # A physically dry start still has a mobile sediment bed.  Retain the
        # legacy wet-water selection for positive minimum-depth thresholds,
        # while a zero threshold explicitly means domain-wide erodibility.
        eligible = connected & (q[:, 0] >= minimum)
        if stage is None and minimum == 0.:
            eligible = np.ones(len(mesh.cells), dtype=bool)
        derived=np.where(eligible,thickness,0.)
        rock=bed-derived
    else:
        thickness = None if thickness_by_cell is None else np.asarray(thickness_by_cell, dtype=float)
        rock = None if bedrock_by_cell is None else np.asarray(bedrock_by_cell, dtype=float)
        for values, name in ((thickness, "erodible thickness"), (rock, "bedrock elevation")):
            if values is not None and (values.shape != bed.shape or not np.isfinite(values).all()):
                raise ValueError(f"Spatial {name} must contain one finite value per retained cell")
        if thickness is not None and (thickness < 0.).any():
            raise ValueError("Spatial erodible thickness cannot be negative")
        tolerance=1e-8+64*np.finfo(float).eps*np.maximum(1.,np.maximum(np.abs(bed),0. if rock is None else np.abs(rock)))
        if rock is not None and (rock > bed+tolerance).any():
            cell=int(np.flatnonzero(rock>bed+tolerance)[0])
            raise ValueError(f"Bedrock elevation is above the initial bed in retained cell {cell}")
        derived=np.maximum(0.,bed-rock) if thickness is None else thickness
        if rock is None:
            rock=bed-derived
        elif thickness is not None and not np.all(np.abs((bed-rock)-thickness)<=tolerance):
            cell=int(np.flatnonzero(np.abs((bed-rock)-thickness)>tolerance)[0])
            raise ValueError(f"Bedrock and erodible-thickness inputs disagree in retained cell {cell}")
    if fractions is None:
        q[:,4]=derived*material.bulk_kg_m3
    elif layered:
        active_thickness=np.minimum(derived,active_target);remaining=derived-active_thickness
        active_composition=np.array([value.initial_volume_fraction if value.initial_active_volume_fraction is None
                                     else value.initial_active_volume_fraction for value in fractions])
        if not math.isclose(float(active_composition.sum()),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError("Initial active-layer volume fractions must sum to one")
        for index,fraction in enumerate(fractions):
            q[:,3+len(fractions)+index]=active_thickness*active_composition[index]*fraction.bulk_kg_m3
        for layer in range(substrate_count):
            layer_thickness=remaining if layer==substrate_count-1 else np.minimum(remaining,substrate_target)
            for index,fraction in enumerate(fractions):
                q[:,3+(2+layer)*len(fractions)+index]=layer_thickness*fraction.initial_volume_fraction*fraction.bulk_kg_m3
            remaining=remaining-layer_thickness
    else:
        for index,fraction in enumerate(fractions):
            q[:,3+len(fractions)+index]=derived*fraction.initial_volume_fraction*fraction.bulk_kg_m3
    return q,rock,connected


def _spatial_bed_values(mesh: CutCellMesh, bed: np.ndarray, project: Mapping[str, Any],
                        progress: Callable[[float], None] | None = None,
                        cancelled: Callable[[], bool] | None = None):
    """Resolve optional bedrock/thickness sources without changing legacy projects."""
    inputs=project.get("bed_inputs",{})
    bedrock_path=inputs.get("bedrock_raster_path")
    thickness_path=inputs.get("thickness_raster_path")
    zones_path=inputs.get("zones_path")
    if not any((bedrock_path,thickness_path,zones_path)):
        return None,None,{"mode":"uniform wet-depth default","spatial":False}
    bedrock=None;thickness=None;source_audits={}
    if bedrock_path:
        bedrock,audit=surface_averages(mesh,bedrock_path,progress=progress,cancelled=cancelled)
        source_audits["bedrock_raster"]={**audit,"path":str(Path(bedrock_path).resolve()),
            "minimum_elevation_m":float(bedrock.min()),"maximum_elevation_m":float(bedrock.max())}
    if thickness_path:
        thickness,audit=terrain_averages(mesh,thickness_path,progress=progress,cancelled=cancelled)
        if (thickness<0.).any():
            raise ValueError(f"Erodible-thickness raster is negative in retained cell {int(np.flatnonzero(thickness<0.)[0])}")
        source_audits["thickness_raster"]={**audit,"path":str(Path(thickness_path).resolve()),
            "minimum_thickness_m":float(thickness.min()),"maximum_thickness_m":float(thickness.max())}
    elif zones_path:
        zones=load_erodible_zones(zones_path,project["domain"]["crs_wkt"],
            inputs.get("zones_layer",ERODIBLE_LAYER_NAME))
        thickness,audit=map_erodible_to_cells(mesh,project["initial"]["erodible_thickness_m"],zones)
        source_audits["thickness_zones"]={**audit,"path":str(Path(zones_path).resolve()),
            "layer":inputs.get("zones_layer",ERODIBLE_LAYER_NAME)}
    return thickness,bedrock,{"mode":"spatial bed inventory","spatial":True,"sources":source_audits}


def executable_identity(executable: str | Path) -> dict[str, str]:
    executable=Path(executable).resolve();manifest=executable.parent/"build.json"
    if manifest.is_file():
        try:data=json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):data={}
        if data.get("executable_sha256") and data["executable_sha256"]!=_digest(executable):
            raise ValueError("Native executable does not match its build manifest")
        source=data.get("source_sha256")
        if isinstance(source,dict) and source:return {str(key):str(value) for key,value in source.items()}
    return {"native_executable":_digest(executable)}


def _require_cpu_v57_native_backend(executable: str | Path) -> None:
    """Reject classic H3 before deck publication on an older CPU binary."""
    executable=Path(executable).expanduser().resolve()
    manifest=executable.parent/"build.json"
    try:
        build=json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(
            "Classic Delft3D4 prognostic secondary flow requires a native "
            "CPU v57 build receipt; this executable cannot be authenticated") from exc
    source=build.get("source_sha256")
    expected_executable=build.get("executable_sha256")
    supported=build.get("supported_case_envelopes")
    if (not isinstance(source,Mapping) or
            PROGNOSTIC_SECONDARY_NATIVE_SOURCE not in source or
            not isinstance(source.get(PROGNOSTIC_SECONDARY_NATIVE_SOURCE),str) or
            build.get("prognostic_secondary_v57") is not True or
            not isinstance(supported,list) or 57 not in supported or
            not isinstance(expected_executable,str) or
            expected_executable!=_digest(executable)):
        raise ValueError(
            "Classic Delft3D4 prognostic secondary flow requires an "
            "authenticated native CPU v57 backend; no legacy fallback is used")


def _write_marker(run_root: Path, payload: Mapping[str, Any]) -> None:
    path=run_root/RUN_MARKER;temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8");temporary.replace(path)


def _prepared_cache_candidate(
    workspace: Path, project_sha256: str, executable_sha256: str,
    capacity_profile_fingerprint: str | None,
    capacity_profile_semantic_id_value: str,
    execution_backend: str,
    hiding_foundation_binding_fingerprint: str | None = None,
    secondary_foundation_binding_fingerprint: str | None = None,
    prognostic_secondary_binding_fingerprint: str | None = None,
) -> tuple[Path, dict] | None:
    """Return an immutable, byte-identical prepared run that can be reused."""
    runs=workspace/"braids_runs"
    if not runs.is_dir():return None
    candidates=[]
    for marker_path in runs.glob(f"*/{RUN_MARKER}"):
        try:marker=json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError,ValueError,json.JSONDecodeError):continue
        root=marker_path.parent
        if (marker.get("status") not in {"prepared","completed"} or
                marker.get("project_sha256")!=project_sha256 or
                marker.get("executable_sha256")!=executable_sha256 or
                marker.get("preparation_execution_backend", BACKEND_CPU)!=
                    execution_backend or
                marker.get("capacity_profile_fingerprint")!=
                    capacity_profile_fingerprint or
                marker.get("capacity_profile_semantic_id")!=
                    capacity_profile_semantic_id_value or
                marker.get("hiding_foundation_binding_fingerprint")!=
                    hiding_foundation_binding_fingerprint or
                marker.get("secondary_foundation_binding_fingerprint")!=
                    secondary_foundation_binding_fingerprint or
                marker.get("prognostic_secondary_binding_fingerprint")!=
                    prognostic_secondary_binding_fingerprint):continue
        if hiding_foundation_binding_fingerprint is not None:
            try:
                binding=_validate_hiding_foundation_binding_document(
                    marker[HIDING_FOUNDATION_BINDING_KEY])
                if binding["fingerprint"]!=hiding_foundation_binding_fingerprint:
                    continue
                identity=validate_hiding_foundation_execution_document(
                    marker[HIDING_FOUNDATION_EXECUTION_KEY])
                route=marker[HIDING_FOUNDATION_ORDERED_ROUTE_KEY]
                if (identity["mode"]!=binding["mode"] or
                        identity["method_id"]!=binding["method_id"] or
                        identity["capacity_law"]!=binding["capacity_law"] or
                        identity["fraction_ids"]!=binding["fraction_ids"] or
                        identity["parker_alpha"]!=binding["parker_alpha"] or
                        identity["wu_exponent"]!=binding["wu_exponent"] or
                        identity["capacity_profile_fingerprint"]!=binding[
                            "capacity_profile_fingerprint"]):
                    continue
                if route!={
                        "ordered_parts":[
                            "capacity_execution_fingerprint",
                            "capacity_profile_fingerprint",
                            "hiding_execution_fingerprint"],
                        "capacity_execution_fingerprint":identity[
                            "capacity_execution_fingerprint"],
                        "capacity_profile_fingerprint":identity[
                            "capacity_profile_fingerprint"],
                        "hiding_execution_fingerprint":identity["fingerprint"]}:
                    continue
            except (KeyError,TypeError,ValueError):
                continue
        if secondary_foundation_binding_fingerprint is not None:
            try:
                binding=_validate_secondary_foundation_binding_document(
                    marker[SECONDARY_FOUNDATION_BINDING_KEY])
                identity=validate_secondary_foundation_execution_document(
                    marker[SECONDARY_FOUNDATION_EXECUTION_KEY])
                route=marker[SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY]
                if (binding["fingerprint"]!=
                        secondary_foundation_binding_fingerprint or
                        identity["capacity_law"]!=binding["capacity_law"] or
                        identity["capacity_profile_fingerprint"]!=
                            binding["capacity_profile_fingerprint"] or
                        identity["roughness_method_id"]!=
                            binding["roughness_method_id"] or
                        identity["rmincf"]!=binding["rmincf"] or
                        identity["chzmin"]!=binding["chzmin"] or
                        identity["espir"]!=binding["espir"] or
                        identity["betac"]!=binding["betac"] or
                        route.get("secondary_execution_fingerprint")!=
                            identity["fingerprint"]):
                    continue
            except (KeyError,TypeError,ValueError):
                continue
        if prognostic_secondary_binding_fingerprint is not None:
            try:
                binding=_validate_prognostic_secondary_binding_document(
                    marker[PROGNOSTIC_SECONDARY_BINDING_KEY])
                identity=validate_prognostic_secondary_execution_document(
                    marker[PROGNOSTIC_SECONDARY_EXECUTION_KEY])
                route=marker[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY]
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
                if (binding["fingerprint"]!=
                        prognostic_secondary_binding_fingerprint or
                        identity["method_id"]!=binding["method_id"] or
                        identity["model"]!=binding["model"] or
                        identity["capacity_law"]!=binding["capacity_law"] or
                        identity["capacity_profile_fingerprint"]!=
                            binding["capacity_profile_fingerprint"] or
                        identity["roughness_method_id"]!=
                            binding["roughness_method_id"] or
                        identity["ordered_fraction_ids"]!=
                            binding["ordered_fraction_ids"] or
                        identity["runtime_layout"]["bed_layer_count"]!=
                            binding["bed_layer_count"] or
                        identity["transport_direction"]!=
                            binding["transport_direction"] or
                        identity["inflow_mode"]!=binding["inflow_mode"] or
                        identity["betac"]!=binding["betac"] or
                        identity["scalar_diffusivity_m2_s"]!=
                            binding["scalar_diffusivity_m2_s"] or
                        route!=expected_route or
                        any(marker.get(key)!=value for key,value in
                            _prognostic_secondary_flat_receipt(identity).items())):
                    continue
            except (KeyError,TypeError,ValueError):
                continue
        sidecars=marker.get("sidecars",{})
        required=("case.data","case.data.json","initial_fields.npz","project.json",
                  "resolved_inputs.json",DIAGNOSTICS_FILE,sidecars.get("geometry","geometry.npz"),
                  sidecars.get("native_binary"),marker.get("capacity_execution_sidecar"))
        required=tuple(name for name in required if name is not None)
        if all(isinstance(name,str) and (root/name).is_file() for name in required):
            candidates.append((str(marker.get("updated_at", "")),root,marker))
    if not candidates:return None
    _,root,marker=max(candidates,key=lambda item:(item[0],str(item[1])))
    return root,marker


def _reuse_prepared_run(source: Path, target: Path, marker: Mapping[str, Any]) -> list[str]:
    """Hard-link immutable preparation products, falling back to normal copies."""
    sidecars=marker.get("sidecars",{})
    names=["case.data","case.data.json","initial_fields.npz","project.json",
           "resolved_inputs.json",DIAGNOSTICS_FILE,sidecars.get("geometry","geometry.npz"),
           sidecars.get("amr_hierarchy","amr_hierarchy.npz"),sidecars.get("native_binary"),
           marker.get("capacity_execution_sidecar"),
           EXECUTION_PROFILE_V2_SIDECAR,
           "agglomeration.json","refinement_zones.json"]
    reused=[]
    for name in dict.fromkeys(name for name in names if isinstance(name,str)):
        origin=source/name
        if not origin.is_file():continue
        destination=target/name
        try:os.link(origin,destination)
        except OSError:shutil.copy2(origin,destination)
        reused.append(name)
    return reused


def update_run_status(run_root: str | Path, status: str, **fields: Any) -> dict[str, Any]:
    run_root=Path(run_root).resolve();path=run_root/RUN_MARKER
    payload=json.loads(path.read_text(encoding="utf-8"));payload.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**fields)
    _write_marker(run_root,payload);return payload


def prepare_run(project_path: str | Path, run_root: str | Path | None = None,
                progress: Callable[[float], None] | None = None,
                cancelled: Callable[[], bool] | None = None, *,
                execution_backend: str = BACKEND_CPU,
                product_backend: str | None = None) -> Path:
    """Materialize a self-describing backend-owned preparation deck.

    Normal callers retain the CPU default.  The QGIS GPU preparation path may
    explicitly request ``implicit_fwave_midpoint_gpu`` for a canonical
    schema-v2 profile.  That request changes admission identities only: the
    native wire remains a backend-neutral physical contract, and a GPU-owned
    deck is rejected by the managed CPU launchers.
    """
    project_path=Path(project_path).resolve();project=load_project(project_path)
    if execution_backend not in (BACKEND_CPU, BACKEND_GPU):
        raise ValueError("Preparation execution backend must be cpu or implicit_fwave_midpoint_gpu")
    product_profile=validate_qgis_product(project)
    requested_product_backend=(execution_backend if product_backend is None
                               else product_backend)
    if requested_product_backend not in (BACKEND_CPU,BACKEND_GPU):
        raise ValueError("Product backend must be cpu or implicit_fwave_midpoint_gpu")
    if (product_profile is not None and
            product_profile.backend!=requested_product_backend):
        raise ValueError(
            "Prepared backend differs from the explicit saved Q10 product backend")
    capacity_execution_binding=project[CAPACITY_EXECUTION_BINDING_KEY]
    execution_profile_binding=project[EXECUTION_PROFILE_BINDING_KEY]
    hiding_foundation_binding=project.get(HIDING_FOUNDATION_BINDING_KEY)
    hiding_foundation_binding_fingerprint=(
        hiding_foundation_binding.get("fingerprint")
        if isinstance(hiding_foundation_binding,Mapping) else None)
    secondary_foundation_binding=project.get(SECONDARY_FOUNDATION_BINDING_KEY)
    secondary_foundation_binding_fingerprint=(
        secondary_foundation_binding.get("fingerprint")
        if isinstance(secondary_foundation_binding,Mapping) else None)
    prognostic_secondary_binding=project.get(
        PROGNOSTIC_SECONDARY_BINDING_KEY)
    prognostic_secondary_binding_fingerprint=(
        prognostic_secondary_binding.get("fingerprint")
        if isinstance(prognostic_secondary_binding,Mapping) else None)
    if (secondary_foundation_binding is not None and
            prognostic_secondary_binding is not None):
        raise ValueError(
            "A project cannot compose equilibrium and prognostic secondary "
            "foundation operators")
    if (prognostic_secondary_binding is not None and
            execution_backend==BACKEND_CPU):
        _require_cpu_v57_native_backend(project["native_executable"])
    project_execution_profile=(execution_profile_binding.get("profile")
                               if execution_profile_binding.get("state")=="canonical"
                               else None)
    try:
        foundation=resolve_project_capacity_foundation_profile(project)
    except CapacityFoundationProfileError as exc:
        if exc.category!="combination_not_implemented":
            raise ValueError(str(exc)) from exc
        foundation=None
    q3c_profile_owned=(
        foundation is not None and
        execution_profile_binding==dict(foundation.project_binding) and
        project["numerics"].get("bedload") in (2,8,11) and
        project["numerics"].get("exchange")==0)
    historical_q3sp_owned=(
        _is_historical_q3sp_general_profile(execution_profile_binding) and
        project["numerics"].get("bedload")==11 and
        project["numerics"].get("exchange")==0)
    profile_owned_public_wire=(
        project["numerics"].get("sediment_enabled",True) is True and
        (q3c_profile_owned or historical_q3sp_owned))
    if (execution_backend==BACKEND_GPU and
            not profile_owned_public_wire):
        raise ValueError(
            "GPU-owned native preparation requires the admitted canonical "
            "schema-v2 law-11 total-load profile; no CPU fallback was used")
    if (execution_backend==BACKEND_CPU and
            capacity_execution_binding.get("method_id") in (
            EH_DELFT_METHOD_ID,MPM_DELFT_METHOD_ID) and
            not q3c_profile_owned):
        selected=capacity_execution_binding["method_id"]
        if selected==EH_DELFT_METHOD_ID:
            reason=("the live selected-hydraulic-Chezy capacity kernel is not "
                    "connected. A historical constant-Chezy closure was not substituted.")
        else:
            reason=("the authenticated v50 component gate is not product admission; "
                    "live capacity-profile ACal and generic hiding integration remain "
                    "to be completed. A historical specified-Cf closure was not substituted.")
        raise ValueError(
            f"Corrected Delft capacity method {selected} is declared but cannot be prepared: "
            +reason)
    legacy_cpu_capacity_resolution=(
        execution_backend==BACKEND_CPU and not profile_owned_public_wire)
    if legacy_cpu_capacity_resolution:
        prepared_capacity=_prepared_capacity_document(project,"cpu")
    else:
        # Schema-v2 is the admission authority for the public law-11 path on
        # both CPU and GPU.  Preserve the exact neutral capacity component
        # carried by the project instead of relabelling it through the older
        # schema-v1 backend catalogue.
        prepared_capacity=execution_profile_binding["component_documents"].get(
            "capacity")
        if prepared_capacity is not None:
            try:
                prepared_capacity=capacity_profile_document(
                    load_capacity_profile(prepared_capacity))
            except (CapacityProfileError,TypeError,KeyError) as exc:
                raise ValueError(
                    f"Schema-v2 capacity component is invalid: {exc}") from exc
            if prepared_capacity.get("backend") is not None:
                raise ValueError(
                    "Schema-v2 project capacity component must remain backend neutral")
        try:
            runtime_execution_profile(
                execution_profile_binding, backend=execution_backend,
                component_documents=execution_profile_binding["component_documents"])
        except ExecutionProfileError as exc:
            raise ValueError(
                f"The schema-v2 {execution_backend} execution profile is unavailable: {exc}") from exc
    capacity_binding=project[CAPACITY_BINDING_KEY]
    workspace=Path(project["workspace"]);workspace.mkdir(parents=True,exist_ok=True)
    if run_root is None:
        stem="braids_"+datetime.now().strftime("%Y%m%d_%H%M%S");run_root=workspace/"braids_runs"/stem
        counter=1
        while run_root.exists():counter+=1;run_root=workspace/"braids_runs"/f"{stem}_{counter:02d}"
    run_root=Path(run_root).expanduser().resolve()
    if run_root.exists():raise ValueError(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    product_sidecar=write_prepared_product(run_root,project)
    recursive_sidecar=None
    if (product_profile is not None and
            product_profile.worker_requirements.get("required_worker_variant")==
            "recursive_bank_amr_v118"):
        recursive_sidecar=write_recursive_prepared_product(run_root,project)
    # The native solver writes immutable native-grid preview frames here while
    # it runs.  QGIS reads them through lightweight VRT descriptors.
    (run_root / "live_preview").mkdir()
    marker={"format":RUN_FORMAT,"status":"preparing","created_at":datetime.now(timezone.utc).isoformat(),
            "project_path":str(project_path),"native_executable":project["native_executable"],
            "capacity_profile_state":capacity_binding["state"],
            "capacity_runtime_profile":prepared_capacity,
            "capacity_profile_fingerprint":(
                None if prepared_capacity is None else prepared_capacity["fingerprint"]),
            "capacity_profile_semantic_id":(
                capacity_profile_semantic_id() if prepared_capacity is None else
                prepared_capacity["semantic_id"]),
            "capacity_execution_binding":capacity_execution_binding,
            "capacity_execution_semantic_id":capacity_execution_semantic_id(),
            "preparation_execution_backend":execution_backend,
            "product_execution_backend":requested_product_backend,
            "qgis_product_fingerprint":(
                None if product_profile is None else product_profile.fingerprint),
            "qgis_product_sidecar":(
                None if product_sidecar is None else product_sidecar.name),
            "recursive_worker_v118_sidecar":(
                None if recursive_sidecar is None else recursive_sidecar.name),
            "execution_profile_v2_state":execution_profile_binding["state"],
            "execution_profile_v2_project_fingerprint":(
                None if project_execution_profile is None else
                project_execution_profile["fingerprint"])}
    if hiding_foundation_binding is not None:
        marker[HIDING_FOUNDATION_BINDING_KEY]=hiding_foundation_binding
        marker["hiding_foundation_binding_fingerprint"]=\
            hiding_foundation_binding_fingerprint
    if secondary_foundation_binding is not None:
        marker[SECONDARY_FOUNDATION_BINDING_KEY]=secondary_foundation_binding
        marker["secondary_foundation_binding_fingerprint"]=\
            secondary_foundation_binding_fingerprint
    if prognostic_secondary_binding is not None:
        marker[PROGNOSTIC_SECONDARY_BINDING_KEY]=prognostic_secondary_binding
        marker["prognostic_secondary_binding_fingerprint"]=\
            prognostic_secondary_binding_fingerprint
    _write_marker(run_root,marker)
    try:
        # A run deck is self-contained and immutable until the native solver
        # starts. Reuse it when every project byte and executable byte match,
        # which makes repeated Prepare Run presses effectively instantaneous
        # without a service, database, or machine-specific installation.
        project_sha256=_digest(project_path);executable_sha256=_digest(project["native_executable"])
        cached=_prepared_cache_candidate(
            workspace,project_sha256,executable_sha256,
            marker["capacity_profile_fingerprint"],
            marker["capacity_profile_semantic_id"],execution_backend,
            hiding_foundation_binding_fingerprint,
            secondary_foundation_binding_fingerprint,
            prognostic_secondary_binding_fingerprint)
        if cached is not None:
            cached_root,cached_marker=cached
            reused=_reuse_prepared_run(cached_root,run_root,cached_marker)
            marker.update({key:value for key,value in cached_marker.items()
                           if key not in {"status","created_at","updated_at","checkpoint","returncode"}})
            marker.update(status="prepared",created_at=marker["created_at"],
                updated_at=datetime.now(timezone.utc).isoformat(),project_path=str(project_path),
                native_executable=project["native_executable"],project_sha256=project_sha256,
                executable_sha256=executable_sha256,
                preparation_cache={"format":PREPARATION_CACHE_FORMAT,"source_run":str(cached_root),
                                   "reused_files":reused})
            _write_marker(run_root,marker)
            verified_prepared_inputs(run_root,marker)
            if progress:progress(100.)
            return run_root
        # Preparation can spend meaningful time before sampling the DEM. Give
        # those stages their own share of the single task percentage.
        def check_cancelled():
            if cancelled and cancelled():
                raise InterruptedError("Run preparation cancelled")

        if progress:progress(0.)
        check_cancelled()
        domain_data=project["domain"];domain=PolygonDomain(domain_data["domain_id"],domain_data["vertices_xy"],domain_data["crs_wkt"])
        assignments=[BoundaryAssignment(**row) for row in project["assignments"]]
        grid=NativeGrid(**project["grid"])
        if progress:progress(3.)
        check_cancelled()
        obstacle_settings=project.get("internal_conditions",{}).get("solid_obstacles",{"path":None})
        obstacle_path=obstacle_settings.get("path")
        obstacles=() if obstacle_path is None else load_solid_obstacles(
            obstacle_path,domain.crs_wkt,obstacle_settings.get("layer",OBSTACLE_LAYER_NAME))
        if progress:progress(6.)
        check_cancelled()
        amr_settings=AMRSettings(**project['numerics'].get('amr',{}))
        hierarchy=None
        amr_minimum=None
        if amr_settings.enabled:
            hierarchy_options={"minimum_area_fraction":project['numerics'].get(
                'minimum_boundary_cell_area_fraction',0.),
                "levels":amr_settings.levels,
                "refinement_ratio":amr_settings.refinement_ratio}
            if progress:
                hierarchy_options["progress"]=lambda value: progress(6.+.16*value)
            # A plugin reload can leave an older bundled braids_core module in
            # QGIS's process.  Use its cancellation support when available,
            # but do not fail a preparation merely because that stale module
            # predates this optional keyword.
            if "cancelled" in inspect.signature(build_hierarchy).parameters:
                hierarchy_options["cancelled"]=cancelled
            hierarchy=build_hierarchy(domain,assignments,grid,obstacles,**hierarchy_options)
            mesh=hierarchy.finest
            refinement=project.get('mesh_refinement',{})
            if refinement.get('zones_path'):
                zones=load_refinement_zones(refinement['zones_path'],domain.crs_wkt,
                    refinement.get('zones_layer',REFINEMENT_LAYER_NAME),
                    maximum_level=amr_settings.levels-1)
                amr_minimum,audit=map_refinement_zones(hierarchy,zones)
                (run_root/'refinement_zones.json').write_text(json.dumps(audit,indent=2)+'\n')
        else:
            mesh=map_cut_cells(domain,assignments,grid,obstacles,
                progress=(lambda value: progress(6+.16*value)) if progress else None,
                cancelled=cancelled)
        if progress:progress(22.)
        check_cancelled()
        agglomeration=None
        if hierarchy is None and project["numerics"].get("minimum_boundary_cell_area_fraction",0.)>0:
            mesh,_,agglomeration=agglomerate_small_cells(
                mesh,project["numerics"]["minimum_boundary_cell_area_fraction"],consume_input=True)
            (run_root/"agglomeration.json").write_text(json.dumps(agglomeration,indent=2,allow_nan=False)+"\n")
        if progress:progress(25.)
        terrain_progress=(lambda value: progress(25+.35*value)) if progress else None
        terrain_interpretation=project["terrain"].get("interpretation", "bilinear_surface")
        if terrain_interpretation == "cell_average_raster":
            bed,terrain=terrain_averages(
                mesh,project["terrain"]["dem_path"],
                progress=terrain_progress,cancelled=cancelled)
        else:
            bed,terrain=surface_averages(
                mesh,project["terrain"]["dem_path"],
                progress=terrain_progress,cancelled=cancelled)
        terrain["interpretation"]=terrain_interpretation
        check_cancelled()
        material=CutSediment(**project["sediment"])
        fractions=(None if "sediment_fractions" not in project else
                   tuple(CutSedimentFraction(**row) for row in project["sediment_fractions"]))
        thickness_by_cell,bedrock_by_cell,bed_inputs=_spatial_bed_values(mesh,bed,project,
            progress=(lambda value: progress(60+.10*value)) if progress else None,cancelled=cancelled)
        if progress:progress(70.)
        check_cancelled()
        initial_stage=project["initial"].get("stage_m")
        water_body=None;water_body_audit={"mode":("dry domain" if initial_stage is None else "all cells below constant level")}
        water_path=project["initial"].get("water_body_path")
        if water_path is not None:
            water_body,water_body_audit=map_water_body_to_cells(mesh,water_path,domain.crs_wkt,
                project["initial"].get("water_body_layer",WATER_BODY_LAYER_NAME))
        elif project["initial"].get("lake_seed_xy") is not None:
            water_body_audit={"mode":"legacy connected lake seed"}
        q,rock,lake=initial_state(mesh,bed,project["initial"],material,
            thickness_by_cell=thickness_by_cell,bedrock_by_cell=bedrock_by_cell,
            sediment_fractions=fractions,water_body_mask=water_body)
        water_body_audit.update(stage_m=project["initial"]["stage_m"],wet_cell_count=int(lake.sum()))
        if fractions is None:
            resolved_thickness=q[:,4]/material.bulk_kg_m3
            initial_erodible_mass=float(np.dot(q[:,4],np.array([cell.area_m2 for cell in mesh.cells])))
        else:
            layered=project["initial"].get("active_layer_thickness_m",0.)>0
            bed_layers=1+(project["initial"].get("substrate_layer_count",1) if layered else 0)
            resolved_thickness=sum(sum(q[:,3+(1+layer)*len(fractions)+index]
                for layer in range(bed_layers))/fraction.bulk_kg_m3 for index,fraction in enumerate(fractions))
            initial_erodible_mass=float(sum(np.dot(sum(q[:,3+(1+layer)*len(fractions)+index]
                for layer in range(bed_layers)),np.array([cell.area_m2 for cell in mesh.cells]))
                for index in range(len(fractions))))
        bed_inputs.update(minimum_thickness_m=float(resolved_thickness.min()),
            maximum_thickness_m=float(resolved_thickness.max()),
            erodible_cell_count=int(np.count_nonzero(resolved_thickness>0.)),
            initial_erodible_mass_kg=initial_erodible_mass,
            minimum_bedrock_elevation_m=float(rock.min()),maximum_bedrock_elevation_m=float(rock.max()))
        boundaries={name:CutBoundary(**row) for name,row in project["boundaries"].items()};numerics=project["numerics"]
        sediment_enabled=numerics.get("sediment_enabled",True)
        if not sediment_enabled:
            boundaries={name:replace(value,concentration_kg_m3=0.,bed_input_kg_m_s=0.,bed_input_kg_s=None,
                bedload_boundary_mode='prescribed',sediment_series=None,fraction_concentration_kg_m3={},
                fraction_bed_input_kg_s={},fraction_sediment_series={}) for name,value in boundaries.items()}
        roughness_settings=project.get("roughness",{"default_manning_n":numerics["manning"],
            "zones_path":None,"zones_layer":MANNING_LAYER_NAME})
        roughness_mode=roughness_settings.get("mode","manning")
        shear_settings=project.get("sediment_shear",{"mode":"total"})
        zones_path=roughness_settings.get("zones_path")
        zones_layer=roughness_settings.get("zones_layer",MANNING_LAYER_NAME)
        if zones_path is None and roughness_mode=="manning":
            # Preserve the historical no-zone Manning audit verbatim.
            roughness_parameter,roughness=map_manning_to_cells(
                mesh,roughness_settings["default_parameter"],())
            roughness["legacy_manning_conversion"]=False
        elif zones_path is None:
            # Typed C/ks still receive a mode-explicit raw-field identity even
            # when every cell uses the default parameter.
            roughness_parameter,roughness=map_typed_roughness_to_cells(
                mesh,roughness_mode,roughness_settings["default_parameter"],())
        elif zones_layer==MANNING_LAYER_NAME:
            # Historical Manning GeoPackages remain byte/physics compatible.
            # They cannot be relabelled as Chezy or Nikuradse inputs because
            # their schema does not record a physical parameter type.
            if roughness_mode!="manning":
                raise ValueError(
                    "A legacy Manning-zone layer cannot be used as Chezy or "
                    "White-Colebrook roughness; create typed roughness zones")
            zones=load_manning_zones(zones_path,domain.crs_wkt,zones_layer)
            roughness_parameter,roughness=map_manning_to_cells(
                mesh,roughness_settings["default_parameter"],zones)
            roughness["roughness_field_binding"]={
                "schema":0,"mode":"manning","parameter_name":"manning_n",
                "parameter_units":"s/m^(1/3)",
                "provenance":"legacy Manning-zone mapping",
            }
            roughness["legacy_manning_conversion"]=False
        else:
            zones=load_typed_roughness_zones(
                zones_path,domain.crs_wkt,roughness_mode,zones_layer)
            roughness_parameter,roughness=map_typed_roughness_to_cells(
                mesh,roughness_mode,roughness_settings["default_parameter"],zones)
        # ``manning_field`` is a historical internal storage name consumed by
        # the native wire.  For C/W it contains the exact raw C/ks bytes and is
        # never an equivalent Manning coefficient.
        manning_field=roughness_parameter
        parameter_name,parameter_units,parameter_label=_ROUGHNESS_PUBLIC[roughness_mode]
        roughness.update(
            mode=roughness_mode,
            parameter_name=parameter_name,parameter_units=parameter_units,
            parameter_label=parameter_label,
            default_parameter=float(roughness_settings["default_parameter"]),
            minimum_parameter=float(np.min(manning_field)),
            maximum_parameter=float(np.max(manning_field)),
            white_colebrook_chzk2d=roughness_settings.get("white_colebrook_chzk2d",False),
            legacy_storage={"field":"manning_cell","physical_type":roughness_mode=="manning",
                "note":("physical Manning n" if roughness_mode=="manning" else
                        "native compatibility storage containing the selected raw parameter; not Manning n")})
        if roughness_mode!="manning":
            for key in ("default_manning_n","minimum_manning_n","maximum_manning_n"):
                roughness.pop(key,None)
            if roughness.get("source")=="default Manning":
                roughness["source"]="default typed hydraulic roughness parameter"
        if progress:progress(72.)
        check_cancelled()
        section_dem_path=(project["terrain"].get("section_dem_path") or
                          project["terrain"]["dem_path"])
        mesh,sampled_sections,boundary_terrain=sample_inlet_terrain(
            mesh,domain,assignments,boundaries,section_dem_path,manning_field,
            section_vertical_tolerance_m=project["terrain"].get(
                "section_vertical_tolerance_m",0.))
        if hierarchy is not None:
            # Every active level must represent the same physical DEM section.
            # Safety-capped inlet fragments would otherwise substitute their
            # owner-cell mean bed for the surveyed edge terrain.
            sampled_levels=[]
            for level_mesh in hierarchy.meshes[:-1]:
                sampled_level,_,_=sample_inlet_terrain(
                    level_mesh,domain,assignments,boundaries,
                    section_dem_path,
                    section_vertical_tolerance_m=project["terrain"].get(
                        "section_vertical_tolerance_m",0.))
                sampled_levels.append(sampled_level)
            hierarchy=replace(hierarchy,meshes=(*sampled_levels,mesh))
        if progress:progress(76.)
        check_cancelled()
        sections={name:section for name,section in sampled_sections.items() if boundaries[name].needs_section}
        bank_settings=project.get("internal_conditions",{}).get("bank_morphology",{"enabled":False})
        bank_failure_settings=project.get("internal_conditions",{}).get("bank_failure",{"enabled":False})
        general_settings=project.get("internal_conditions",{}).get("general_avalanche",{"enabled":False})
        secondary_settings=project.get("internal_conditions",{}).get("secondary_flow",{"enabled":False})
        adaptation_settings=project.get("internal_conditions",{}).get("bedload_adaptation",{"enabled":False})
        acceleration_settings=project.get("internal_conditions",{}).get("morphological_acceleration",{"enabled":False})
        bedform_settings=project.get("internal_conditions",{}).get("bedform_roughness",{"enabled":False})
        bank_path=(bank_failure_settings.get("zones_path") if bank_failure_settings.get("enabled",False)
                   else bank_settings.get("zones_path"))
        bank_layer=(bank_failure_settings.get("zones_layer",BANK_LAYER_NAME)
                    if bank_failure_settings.get("enabled",False)
                    else bank_settings.get("zones_layer",BANK_LAYER_NAME))
        bank_zones=() if bank_path is None else load_bank_zones(
            bank_path,domain.crs_wkt,bank_layer)
        bank_mask,bank_audit=map_bank_zones_to_cells(mesh,bank_zones)
        bank_audit.update(enabled=bool(bank_settings.get("enabled",False) or bank_failure_settings.get("enabled",False)),
            mode=("process_separated" if bank_failure_settings.get("enabled",False) else "legacy"),
            domain_boundary="fixed; exterior faces are excluded from bank transfer")
        compact_setting=numerics.get("compact_sidecars")
        if compact_setting is not None and not isinstance(compact_setting,bool):
            raise ValueError("Compact sidecars must be true, false or automatic")
        compact_sidecars=(len(mesh.cells)>=COMPACT_SIDECAR_CELL_THRESHOLD
                          if compact_setting is None else compact_setting)
        if progress:progress(78.)
        case=run_root/"case.data"
        capacity_writer_options={}
        if sediment_enabled and numerics["bedload"] in SPLIT_CAPACITY_LAWS:
            if prepared_capacity is None:
                raise ValueError(
                    "Split-capacity preparation requires a resolved runtime profile")
            capacity_writer_options={
                "capacity_parameters_by_fraction":
                    capacity_binding["capacity_parameters_by_fraction"],
                "capacity_output_contract":capacity_binding["capacity_output_contract"],
            }
        forcing_metadata=_capacity_forcing_metadata_before_deck(
            roughness_mode,roughness_settings,manning_field,shear_settings)
        capacity_execution,forcing_binding=_prepared_capacity_execution(
            project,forcing_metadata,manning_field,prepared_capacity,
            q3c_profile_owned=q3c_profile_owned)
        hiding_foundation_execution,hiding_foundation_route=\
            _prepared_hiding_foundation_execution(
                project,capacity_execution,prepared_capacity)
        if hiding_foundation_execution is not None:
            capacity_execution={
                **capacity_execution,
                HIDING_FOUNDATION_BINDING_KEY:hiding_foundation_binding,
                HIDING_FOUNDATION_EXECUTION_KEY:hiding_foundation_execution,
                HIDING_FOUNDATION_ORDERED_ROUTE_KEY:hiding_foundation_route,
            }
        secondary_foundation_execution,secondary_foundation_route=\
            _prepared_secondary_foundation_execution(
                project,capacity_execution,prepared_capacity,forcing_binding,
                hiding_foundation_execution)
        if secondary_foundation_execution is not None:
            capacity_execution={
                **capacity_execution,
                SECONDARY_FOUNDATION_BINDING_KEY:secondary_foundation_binding,
                SECONDARY_FOUNDATION_EXECUTION_KEY:
                    secondary_foundation_execution,
                SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:
                    secondary_foundation_route,
            }
        prognostic_secondary_execution,prognostic_secondary_route=\
            _prepared_prognostic_secondary_execution(
                project,capacity_execution,prepared_capacity,forcing_binding,
                hiding_foundation_execution)
        if (secondary_foundation_execution is not None and
                prognostic_secondary_execution is not None):
            raise ValueError(
                "Prepared execution cannot compose Q3-H2 and Q3-H3 together")
        if prognostic_secondary_execution is not None:
            capacity_execution={
                **capacity_execution,
                PROGNOSTIC_SECONDARY_BINDING_KEY:
                    prognostic_secondary_binding,
                PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                    prognostic_secondary_execution,
                PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:
                    prognostic_secondary_route,
                **_prognostic_secondary_flat_receipt(
                    prognostic_secondary_execution),
            }
        runtime_execution_profile_document=None
        backend_execution_profile_document=None
        runtime_execution_component_documents={}
        backend_execution_component_documents={}
        if execution_profile_binding["state"]=="canonical":
            runtime_execution_component_documents=dict(
                execution_profile_binding["component_documents"])
            backend_execution_component_documents=dict(
                runtime_execution_component_documents)
            if (legacy_cpu_capacity_resolution and
                    "capacity" in backend_execution_component_documents):
                if prepared_capacity is None:
                    raise ValueError(
                        "Schema-v2 execution profile declares capacity but the runtime "
                        "capacity profile is inactive")
                backend_execution_component_documents["capacity"]=prepared_capacity
            try:
                backend_execution_profile_value,backend_execution_component_documents=\
                    runtime_execution_profile(
                        execution_profile_binding,backend=execution_backend,
                        component_documents=backend_execution_component_documents)
            except ExecutionProfileError as exc:
                raise ValueError(
                    f"The schema-v2 {execution_backend} execution profile is unavailable: {exc}") from exc
            backend_execution_profile_document=execution_profile_document(
                backend_execution_profile_value)
            if (legacy_cpu_capacity_resolution and
                    "capacity" in backend_execution_component_documents):
                neutral_capacity=_neutral_execution_capacity_document(
                    project,backend_execution_component_documents["capacity"])
                if neutral_capacity!=runtime_execution_component_documents["capacity"]:
                    raise ValueError(
                        "Prepared neutral capacity component differs from the project profile")
                runtime_execution_component_documents["capacity"]=neutral_capacity
            if (set(backend_execution_component_documents)!=
                    set(runtime_execution_component_documents) or any(
                        (not legacy_cpu_capacity_resolution or component_id!="capacity") and
                        backend_execution_component_documents[component_id]!=
                            runtime_execution_component_documents[component_id]
                        for component_id in runtime_execution_component_documents)):
                raise ValueError(
                    "Backend execution components differ from the neutral project components")
            # The native wire authenticates one backend-neutral physical
            # profile; the marker separately records the explicitly admitted
            # runtime resolution.  Historical schema-v1 CPU capacity routes
            # retain their backend-specific component in that marker only.
            neutral_component_profiles=tuple(
                bind_component_profile(
                    row["component_id"],
                    runtime_execution_component_documents[row["component_id"]])
                for row in execution_profile_binding["profile"][
                    "component_profiles"])
            if is_frozen_q3sp_execution_profile(
                    backend_execution_profile_value):
                # Preserve the exact 0d40... profile consumed by frozen v13.
                # Re-resolving it against the current catalogue would create a
                # forbidden current-ledger seven-selector hybrid.
                runtime_execution_profile_value=load_execution_profile(
                    execution_profile_binding["profile"])
                if (tuple(runtime_execution_profile_value.component_profiles)!=
                        neutral_component_profiles):
                    raise ValueError(
                        "Frozen Q3-S/P components differ from the project profile")
            else:
                runtime_execution_profile_value=resolve_execution_profile(
                    fraction_layout=backend_execution_profile_value.fraction_layout,
                    method_selections=backend_execution_profile_value.method_selections,
                    parameter_bindings=backend_execution_profile_value.parameter_bindings,
                    control_bindings=backend_execution_profile_value.control_bindings,
                    component_profiles=neutral_component_profiles,
                    backend=None)
            runtime_execution_profile_document=execution_profile_document(
                runtime_execution_profile_value)
            if runtime_execution_profile_document!=execution_profile_binding["profile"]:
                raise ValueError(
                    "Prepared native execution profile differs from the neutral project profile")
        capacity_route_writer_options={}
        # The additive Q3-C foundation identity belongs only to the exact
        # eight-selector profile resolved above.  A frozen seven-selector
        # Q3-S/P law-11 profile remains a distinct v53/v13 route: it must not
        # acquire the Q3-C capacity document merely because the current
        # project defaults can independently resolve one.
        q3c_execution=(capacity_execution.get(
            "capacity_foundation_execution_identity")
            if q3c_profile_owned else None)
        route=(capacity_execution.get(
            "capacity_foundation_ordered_route_identity")
            if q3c_profile_owned else None)
        execution_document=q3c_execution
        if route is None:
            route=capacity_execution.get("ordered_route_identity")
            execution_document=capacity_execution.get("execution_identity")
        if route is not None:
            if not isinstance(execution_document,Mapping):
                raise ValueError("Resolved capacity route lacks its execution identity")
            capacity_route_writer_options={
                "capacity_method_id":execution_document["method_id"],
                "capacity_execution_semantic_id_value":
                    execution_document["semantic_id"],
                "capacity_execution_fingerprint":
                    route["capacity_execution_fingerprint"],
                "capacity_runtime_profile_semantic_id":
                    prepared_capacity["semantic_id"],
                "capacity_profile_fingerprint":
                    route["capacity_profile_fingerprint"],
                "capacity_profile_acal":
                    _capacity_profile_uniform_acal(prepared_capacity),
            }
        execution_profile_writer_options={}
        if runtime_execution_profile_document is not None:
            channel_document=runtime_execution_component_documents.get(
                "channel_composition")
            if channel_document is None:
                raise ValueError(
                    "Schema-v2 native execution requires a channel-composition component")
            execution_profile_writer_options={
                "channel_composition_profile":channel_document,
                "capacity_runtime_profile":
                    (runtime_execution_component_documents.get("capacity")
                     if q3c_profile_owned else None),
                "capacity_foundation_execution_profile":
                    (q3c_execution if q3c_profile_owned else None),
                "capacity_forcing_profile":
                    (forcing_binding if q3c_profile_owned else None),
                # Existing schema-v1 CPU preparation stays on its byte-stable
                # v52 contract.  The public schema-v2 law-11 route uses v53 on
                # both CPU and GPU so the native reader independently verifies
                # canonical component-document preimages, not only hashes.
                "channel_composition_wire_version":(
                    53 if profile_owned_public_wire else 52),
                "execution_profile_v2":runtime_execution_profile_document,
                "execution_runtime_backend":execution_backend,
            }
            if hiding_foundation_execution is not None:
                execution_profile_writer_options[
                    "hiding_foundation_execution_profile"]=\
                    hiding_foundation_execution
            if secondary_foundation_execution is not None:
                execution_profile_writer_options[
                    "secondary_foundation_execution_profile"]=\
                    secondary_foundation_execution
        if prognostic_secondary_execution is not None:
            execution_profile_writer_options["prognostic_secondary_identity"]={
                PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                    prognostic_secondary_execution,
            }
        metadata=write_cut_case(case,mesh,q,rock,boundaries,end_time_s=numerics["end_time_s"],
            dt_max_s=numerics["dt_max_s"],output_interval_s=numerics["output_interval_s"],cfl=numerics.get("cfl",.45),
            scheme=numerics["scheme"],bedload=numerics["bedload"] if sediment_enabled else 0,
            exchange=numerics["exchange"] if sediment_enabled else 0,
            settling_velocity_m_s=numerics["settling_velocity_m_s"],manning=roughness_settings["default_manning_n"],
            hydraulic_roughness_mode=roughness_mode,
            white_colebrook_chzk2d=roughness_settings.get("white_colebrook_chzk2d",False),
            sediment_shear_mode=(shear_settings.get("mode","total") if sediment_enabled else "total"),
            skin_kinematic_viscosity_m2_s=shear_settings.get("kinematic_viscosity_m2_s",1.e-6),
            skin_sand_roughness_m=shear_settings.get("sand_roughness_m",.003),
            skin_silt_roughness_m=shear_settings.get("silt_roughness_m",.0003),
            skin_critical_mud_fraction_low=shear_settings.get("critical_mud_fraction_low",.2),
            skin_critical_mud_fraction_high=shear_settings.get("critical_mud_fraction_high",.8),
            skin_fluff_fraction=shear_settings.get("fluff_fraction",0.),
            horizontal_viscosity_m2_s=numerics.get('horizontal_viscosity_m2_s',0.),
            horizontal_viscosity_formulation=numerics.get('horizontal_viscosity_formulation','depth_conservative'),
            manning_by_cell=manning_field if roughness_settings.get("zones_path") is not None else None,sections=sections,
            sediment=None if fractions is not None else material,sediment_fractions=fractions,
            bed_slope=numerics["bed_slope"] if sediment_enabled else 0,
            engelund_hansen_chezy=(numerics.get("engelund_hansen_chezy")
                                   if sediment_enabled and numerics["bedload"]==8 else None),
            mpm_cf=numerics.get("mpm_cf",.01),mpm_critical_shields=numerics.get("mpm_critical_shields",0.),
            mpm_friction_mode=numerics.get("mpm_friction_mode",0) if sediment_enabled else 0,
            hiding_exposure=numerics.get("hiding_exposure",0) if sediment_enabled else 0,
            hiding_parker_alpha=numerics.get("hiding_parker_alpha",.6),
            hiding_wu_exponent=numerics.get("hiding_wu_exponent",-.6),
            bed_slope_alfabs=numerics.get("bed_slope_alfabs",1.),bed_slope_alfabn=numerics.get("bed_slope_alfabn",1.5),
            bed_friction_angle_deg=numerics.get("bed_friction_angle_deg",30.),
            active_layer_thickness_m=project["initial"].get("active_layer_thickness_m",0.),
            active_layer_mode=project["initial"].get("active_layer_mode","constant"),
            active_layer_depth_fraction=project["initial"].get("active_layer_depth_fraction",0.),
            active_layer_minimum_thickness_m=project["initial"].get(
                "active_layer_minimum_thickness_m",project["initial"].get("active_layer_thickness_m",0.)),
            substrate_layer_count=project["initial"].get("substrate_layer_count",1),
            substrate_layer_thickness_m=project["initial"].get("substrate_layer_thickness_m",
                project["initial"].get("active_layer_thickness_m",0.)),
            critical_mud_fraction=numerics.get("critical_mud_fraction",.5),
            transport_calibration_factor=numerics.get("transport_calibration_factor",1.),
            settling_mode=numerics.get("settling_mode",0) if sediment_enabled else 0,
            hindered_settling_reference_concentration_kg_m3=numerics.get(
                "hindered_settling_reference_concentration_kg_m3",
                DEFAULT_HINDERED_REFERENCE_CONCENTRATION_KG_M3),
            bank_erodible_by_cell=(bank_mask if (bank_settings.get("enabled",False) or
                                                  bank_failure_settings.get("enabled",False)) else None),
            bank_erosion_fraction=(bank_settings.get("erosion_supply_fraction",.5)
                                   if bank_settings.get("enabled",False) else 0.),
            bank_avalanche=(bank_settings.get("avalanche_enabled",True)
                            if bank_settings.get("enabled",False) else False),
            bank_wet_slope=bank_settings.get("wet_maximum_slope",10.),
            bank_dry_slope=bank_settings.get("dry_maximum_slope",1.),
            bank_avalanche_time_s=bank_settings.get("avalanche_time_s",86400.),
            bank_relaxation_length_m=bank_settings.get("relaxation_length_m",1.),
            bank_wet_depth_switch_m=bank_settings.get("wet_depth_switch_m",.1),
            bank_failure=(bank_failure_settings.get("enabled",False) if sediment_enabled else False),
            bank_toe_erodibility_m_pa_s=bank_failure_settings.get("toe_erodibility_m_pa_s",1.3e-6),
            bank_toe_critical_shear_pa=bank_failure_settings.get("toe_critical_shear_pa",32.5),
            bank_toe_exponent=bank_failure_settings.get("toe_exponent",1.),
            bank_critical_undercut_m=bank_failure_settings.get("critical_undercut_m",.1),
            bank_toe_erosion_mode=bank_failure_settings.get(
                "toe_erosion_mode","cohesive_excess_shear"),
            bank_toe_capacity_fraction=bank_failure_settings.get("toe_capacity_fraction",1.),
            bank_distributed_transport_fraction=bank_failure_settings.get(
                "distributed_transport_fraction",.1),
            bank_distributed_critical_slope=bank_failure_settings.get(
                "distributed_critical_slope",.35),
            bank_front_tracking=bank_failure_settings.get(
                "front_tracking_enabled",False),
            general_avalanche=(general_settings.get("enabled",False) if sediment_enabled else False),
            general_avalanche_wet_slope=general_settings.get("wet_maximum_slope",10.),
            general_avalanche_dry_slope=general_settings.get("dry_maximum_slope",1.),
            general_avalanche_time_s=general_settings.get("avalanche_time_s",86400.),
            general_avalanche_relaxation_length_m=general_settings.get("relaxation_length_m",1.),
            general_avalanche_wet_depth_switch_m=general_settings.get("wet_depth_switch_m",.1),
            # Q3-H2 is an additive v56 operator around a foundation that must
            # remain secondary.off.  Do not also activate the historical
            # secondary-flow switch and apply the rotation twice.
            secondary_flow=(secondary_settings.get("enabled",False)
                            if (secondary_foundation_execution is None and
                                prognostic_secondary_execution is None)
                            else False),
            secondary_flow_espir=secondary_settings.get("espir",1.),
            secondary_flow_model=_resolved_secondary_flow_model(
                secondary_settings.get("model","equilibrium")),
            secondary_flow_transport_direction=secondary_settings.get("transport_direction",True),
            secondary_flow_momentum_beta=secondary_settings.get("momentum_beta",0.),
            bedload_adaptation=(adaptation_settings.get("enabled",False) if sediment_enabled else False),
            bedload_adaptation_length_diameters=adaptation_settings.get("length_diameters",100.),
            morphological_factor=(acceleration_settings.get("constant_factor",1.)
                                  if acceleration_settings.get("enabled",False) else 1.),
            morphological_factor_series=(acceleration_settings.get("series")
                                         if acceleration_settings.get("enabled",False) else None),
            bedform_roughness=(bedform_settings.get("enabled",False) if sediment_enabled else False),
            amr_hierarchy=hierarchy,amr_settings=amr_settings,amr_minimum_levels=amr_minimum,
            source_sha256=executable_identity(project["native_executable"]),
            compact_sidecars=compact_sidecars,
            sediment_catalog_id=semantic_catalog_id(),
            **capacity_writer_options,**capacity_route_writer_options,
            **execution_profile_writer_options)
        # The pre-deck declaration above is an ordering necessity, not a
        # second source of truth.  Rebuild both documents from the final case
        # receipt and require exact agreement before publishing any sidecar.
        checked_capacity_execution,checked_forcing_binding=\
            _prepared_capacity_execution(
                project,metadata,manning_field,prepared_capacity,
                q3c_profile_owned=q3c_profile_owned)
        checked_hiding_execution,checked_hiding_route=\
            _prepared_hiding_foundation_execution(
                project,checked_capacity_execution,prepared_capacity)
        if checked_hiding_execution is not None:
            checked_capacity_execution={
                **checked_capacity_execution,
                HIDING_FOUNDATION_BINDING_KEY:hiding_foundation_binding,
                HIDING_FOUNDATION_EXECUTION_KEY:checked_hiding_execution,
                HIDING_FOUNDATION_ORDERED_ROUTE_KEY:checked_hiding_route,
            }
        checked_secondary_execution,checked_secondary_route=\
            _prepared_secondary_foundation_execution(
                project,checked_capacity_execution,prepared_capacity,
                checked_forcing_binding,checked_hiding_execution)
        if checked_secondary_execution is not None:
            checked_capacity_execution={
                **checked_capacity_execution,
                SECONDARY_FOUNDATION_BINDING_KEY:secondary_foundation_binding,
                SECONDARY_FOUNDATION_EXECUTION_KEY:checked_secondary_execution,
                SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:
                    checked_secondary_route,
            }
        checked_prognostic_execution,checked_prognostic_route=\
            _prepared_prognostic_secondary_execution(
                project,checked_capacity_execution,prepared_capacity,
                checked_forcing_binding,checked_hiding_execution)
        if checked_prognostic_execution is not None:
            native_prognostic=metadata.get("prognostic_secondary_identity")
            if (not isinstance(native_prognostic,Mapping) or
                    native_prognostic.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)!=
                        checked_prognostic_execution or
                    native_prognostic.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)!=
                        checked_prognostic_route or
                    any(native_prognostic.get(key)!=value for key,value in
                        _prognostic_secondary_flat_receipt(
                            checked_prognostic_execution).items())):
                raise ValueError(
                    "Final native v57 receipt differs from its authenticated "
                    "Q3-H3 prognostic-secondary identity")
            checked_capacity_execution={
                **checked_capacity_execution,
                PROGNOSTIC_SECONDARY_BINDING_KEY:
                    prognostic_secondary_binding,
                PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                    checked_prognostic_execution,
                PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:
                    checked_prognostic_route,
                **_prognostic_secondary_flat_receipt(
                    checked_prognostic_execution),
            }
        if (checked_capacity_execution!=capacity_execution or
                checked_forcing_binding!=forcing_binding):
            raise ValueError(
                "Final native case metadata differs from its authenticated "
                "capacity/forcing identity")
        capacity_execution_path=run_root/CAPACITY_EXECUTION_SIDECAR
        capacity_execution_path.write_text(
            json.dumps(capacity_execution,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        if backend_execution_profile_document is not None and \
                execution_backend==BACKEND_GPU:
            # The zero-step GPU initializer authenticates a backend-specific
            # schema-v2 sidecar next to the backend-neutral native deck.  Put
            # it in the prepared run transaction itself so direct and QGIS-
            # managed launches consume the same exact profile; the later
            # worker-scope gate may verify this file but must not recreate a
            # different physical identity.
            execution_profile_sidecar={
                "schema":1,
                "backend":BACKEND_GPU,
                "admission":"runtime",
                "profile":backend_execution_profile_document,
                "component_documents":backend_execution_component_documents,
            }
            (run_root/EXECUTION_PROFILE_V2_SIDECAR).write_text(
                json.dumps(execution_profile_sidecar,sort_keys=True,
                           separators=(",",":"),ensure_ascii=True,
                           allow_nan=False),
                encoding="ascii")
        if progress:progress(90.)
        mesh.write_geometry(run_root/metadata["sidecars"]["geometry"])
        if progress:progress(95.)
        initial_fields={"q":q,"bed_m":bed,"bedrock_m":rock,
            "erodible_thickness_m":resolved_thickness,"initial_lake":lake,
            "area_m2":np.array([cell.area_m2 for cell in mesh.cells]),
            "roughness_parameter":manning_field,"erodible_bank_mask":bank_mask}
        if roughness_mode=="manning":initial_fields["manning_n"]=manning_field
        np.savez_compressed(run_root/"initial_fields.npz",**initial_fields)
        (run_root/"project.json").write_text(json.dumps(project,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        inlet_diagnostics=represented_inlets(mesh,bed,q,boundaries,metadata,numerics["end_time_s"])
        (run_root/DIAGNOSTICS_FILE).write_text(json.dumps(inlet_diagnostics,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        resolved_inputs={"format":"braids-resolved-inputs-v1",
                         "compatibility_notes":project.get("compatibility_notes",[]),
            "qgis_product":(None if product_profile is None else
                            product_profile.as_dict()),
            "mesh":{"cell_count":len(mesh.cells),"grid":asdict(mesh.grid),"base_grid":project['grid'],
                "amr":asdict(amr_settings),"reconstruction_scheme":numerics["scheme"],
                "sidecars":metadata["sidecars"],
                "solid_obstacles":{"count":len(mesh.obstacle_ids),"ids":list(mesh.obstacle_ids),
                    "excluded_area_m2":mesh.solid_area_m2,
                    "wall_fragment_count":mesh.metrics()["internal_wall_fragment_count"],
                    "wall_length_m":mesh.metrics()["internal_wall_length_m"],
                    "behavior":"fixed impermeable reflective wall"},
                "boundary_cell_agglomeration":None if agglomeration is None else {k:v for k,v in agglomeration.items() if k!='clusters'}},
            "initial_water":water_body_audit,
            "roughness":roughness,
            "sediment":{"material":metadata["materials"],"settling_velocity_m_s":numerics["settling_velocity_m_s"],
                "fraction_ids":metadata["fraction_ids"],
                "enabled":sediment_enabled,
                "suspended_exchange_mode":numerics["exchange"] if sediment_enabled else 0,
                "bedload_law":numerics["bedload"] if sediment_enabled else 0,
                "method_selection":sediment_method_metadata(numerics),
                "semantic_catalog_id":metadata["semantic_catalog_id"],
                "capacity_profile_state":capacity_binding["state"],
                "capacity_parameters_by_fraction":
                    capacity_binding["capacity_parameters_by_fraction"],
                "capacity_output_contract":capacity_binding["capacity_output_contract"],
                "capacity_runtime_profile":prepared_capacity,
                "capacity_profile_fingerprint":(
                    None if prepared_capacity is None else prepared_capacity["fingerprint"]),
                "capacity_profile_semantic_id":(
                    capacity_profile_semantic_id() if prepared_capacity is None else
                    prepared_capacity["semantic_id"]),
                "capacity_execution":capacity_execution,
                "capacity_execution_sidecar":CAPACITY_EXECUTION_SIDECAR,
                "capacity_execution_sha256":_digest(capacity_execution_path),
                "execution_profile_v2_state":execution_profile_binding["state"],
                "execution_profile_v2":runtime_execution_profile_document,
                "execution_profile_v2_fingerprint":(
                    None if runtime_execution_profile_document is None else
                    runtime_execution_profile_document["fingerprint"]),
                "execution_profile_v2_semantic_id":(
                    None if runtime_execution_profile_document is None else
                    runtime_execution_profile_document["semantic_id"]),
                "execution_profile_v2_components":
                    runtime_execution_component_documents,
                "execution_profile_v2_backend_profile":
                    backend_execution_profile_document,
                "execution_profile_v2_backend_components":
                    backend_execution_component_documents,
                "typed_forcing_binding":forcing_binding,
                "engelund_hansen_chezy":metadata["engelund_hansen_chezy"],
                "suspended_exchange_name":metadata["suspended_exchange_name"],
                "bedload_law_name":metadata["bedload_law_name"],
                "mpm_cf":metadata["mpm_cf"],"mpm_critical_shields":metadata["mpm_critical_shields"],
                "mpm_friction_mode":metadata["mpm_friction_mode"],
                "hiding_exposure":metadata["hiding_exposure"],
                "critical_mud_fraction":metadata["critical_mud_fraction"],
                "transport_calibration_factor":metadata["transport_calibration_factor"],
                "settling_treatment":metadata["settling_treatment"],
                "bed_slope":metadata["bed_slope"],"initial_bed":{**project["initial"],"resolved":bed_inputs},
                "bank_morphology":{**metadata["bank_morphology"],"mapping":bank_audit},
                "bank_failure":metadata.get("bank_failure",{"enabled":False}),
                "general_avalanche":metadata["general_avalanche"],
                "secondary_flow":metadata["secondary_flow"],
                "bedload_adaptation":metadata["bedload_adaptation"],
                "morphological_acceleration":metadata["morphological_acceleration"],
                "bedform_roughness":metadata["bedform_roughness"]},
            "boundaries":{"ids":metadata["string_to_numeric_id"],"settings":project["boundaries"],
                "terrain_representation":boundary_terrain,
                "represented_inlet_diagnostics":DIAGNOSTICS_FILE,
                "initial_inlet_screen":{name:entry["status"] for name,entry in inlet_diagnostics["inlets"].items()},
                "initial_outlet_screen":{name:entry["status"] for name,entry in inlet_diagnostics["outlets"].items()},
                "resolved_targets":metadata["resolved_boundary_targets"],
                "resolved_hydraulic_series":metadata["resolved_hydraulic_series"]},
            "time":{"end_time_s":numerics["end_time_s"],"dt_max_s":numerics["dt_max_s"],"cfl":numerics.get("cfl",.45),
                    "dry_tolerance_m":metadata["dry_tolerance_m"],
                    "output_interval_s":numerics["output_interval_s"],"threads":numerics["threads"]}}
        if hiding_foundation_execution is not None:
            resolved_inputs["sediment"].update({
                HIDING_FOUNDATION_BINDING_KEY:hiding_foundation_binding,
                HIDING_FOUNDATION_EXECUTION_KEY:hiding_foundation_execution,
                HIDING_FOUNDATION_ORDERED_ROUTE_KEY:hiding_foundation_route,
            })
        if secondary_foundation_execution is not None:
            resolved_inputs["sediment"].update({
                SECONDARY_FOUNDATION_BINDING_KEY:secondary_foundation_binding,
                SECONDARY_FOUNDATION_EXECUTION_KEY:
                    secondary_foundation_execution,
                SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:
                    secondary_foundation_route,
            })
        if prognostic_secondary_execution is not None:
            resolved_inputs["sediment"].update({
                PROGNOSTIC_SECONDARY_BINDING_KEY:
                    prognostic_secondary_binding,
                PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                    prognostic_secondary_execution,
                PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:
                    prognostic_secondary_route,
                **_prognostic_secondary_flat_receipt(
                    prognostic_secondary_execution),
            })
        resolved_path=run_root/"resolved_inputs.json"
        resolved_path.write_text(json.dumps(resolved_inputs,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        if progress:progress(98.)
        if inlet_diagnostics["errors"]:
            raise InletPreparationError("\n".join(inlet_diagnostics["errors"]),run_root)
        marker.update(status="prepared",updated_at=datetime.now(timezone.utc).isoformat(),threads=numerics["threads"],
            fingerprint=metadata["fingerprint"],cell_count=len(mesh.cells),boundary_ids=metadata["string_to_numeric_id"],
            terrain=terrain,roughness=roughness,bed_inputs=bed_inputs,initial_water=water_body_audit,
            bank_morphology=bank_audit,solid_obstacles=resolved_inputs["mesh"]["solid_obstacles"],
            sidecars=metadata["sidecars"],
            semantic_catalog_id=metadata["semantic_catalog_id"],
            capacity_profile_state=capacity_binding["state"],
            capacity_runtime_profile=prepared_capacity,
            capacity_profile_fingerprint=(
                None if prepared_capacity is None else prepared_capacity["fingerprint"]),
            capacity_profile_semantic_id=(
                capacity_profile_semantic_id() if prepared_capacity is None else
                prepared_capacity["semantic_id"]),
            capacity_execution_sidecar=CAPACITY_EXECUTION_SIDECAR,
            capacity_execution_sha256=_digest(capacity_execution_path),
            capacity_execution_state=capacity_execution["state"],
            capacity_execution_identity=capacity_execution["execution_identity"],
            capacity_foundation_execution_identity=capacity_execution.get(
                "capacity_foundation_execution_identity"),
            capacity_foundation_ordered_route_identity=capacity_execution.get(
                "capacity_foundation_ordered_route_identity"),
            execution_profile_v2_state=execution_profile_binding["state"],
            execution_profile_v2=runtime_execution_profile_document,
            execution_profile_v2_fingerprint=(
                None if runtime_execution_profile_document is None else
                runtime_execution_profile_document["fingerprint"]),
            execution_profile_v2_semantic_id=(
                None if runtime_execution_profile_document is None else
                runtime_execution_profile_document["semantic_id"]),
            execution_profile_v2_backend_profile=backend_execution_profile_document,
            execution_profile_v2_components=runtime_execution_component_documents,
            execution_profile_v2_backend_components=
                backend_execution_component_documents,
            execution_profile_v2_runtime_backend=(
                None if backend_execution_profile_document is None else
                execution_backend),
            execution_profile_v2_runtime_fingerprint=(
                None if backend_execution_profile_document is None else
                backend_execution_profile_document["fingerprint"]),
            typed_forcing_binding=forcing_binding,
            project_sha256=_digest(project_path),case_sha256=_digest(case),
            resolved_inputs_sha256=_digest(resolved_path),executable_sha256=_digest(project["native_executable"]))
        if hiding_foundation_execution is not None:
            marker.update({
                HIDING_FOUNDATION_BINDING_KEY:hiding_foundation_binding,
                "hiding_foundation_binding_fingerprint":
                    hiding_foundation_binding_fingerprint,
                HIDING_FOUNDATION_EXECUTION_KEY:hiding_foundation_execution,
                HIDING_FOUNDATION_ORDERED_ROUTE_KEY:hiding_foundation_route,
            })
        if secondary_foundation_execution is not None:
            marker.update({
                SECONDARY_FOUNDATION_BINDING_KEY:secondary_foundation_binding,
                "secondary_foundation_binding_fingerprint":
                    secondary_foundation_binding_fingerprint,
                SECONDARY_FOUNDATION_EXECUTION_KEY:
                    secondary_foundation_execution,
                SECONDARY_FOUNDATION_ORDERED_ROUTE_KEY:
                    secondary_foundation_route,
            })
        if prognostic_secondary_execution is not None:
            marker.update({
                PROGNOSTIC_SECONDARY_BINDING_KEY:
                    prognostic_secondary_binding,
                "prognostic_secondary_binding_fingerprint":
                    prognostic_secondary_binding_fingerprint,
                PROGNOSTIC_SECONDARY_EXECUTION_KEY:
                    prognostic_secondary_execution,
                PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY:
                    prognostic_secondary_route,
                **_prognostic_secondary_flat_receipt(
                    prognostic_secondary_execution),
            })
        _write_marker(run_root,marker)
        if progress:progress(100.)
        return run_root
    except BaseException as exc:
        for partial in run_root.glob("*.partial"):
            partial.unlink(missing_ok=True)
        marker.update(status="cancelled" if isinstance(exc,InterruptedError) else "failed",
                      updated_at=datetime.now(timezone.utc).isoformat(),preparation_error=str(exc))
        _write_marker(run_root,marker);raise
