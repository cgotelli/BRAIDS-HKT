"""Profile-authenticated law-11 bridge for the implicit resident GPU route.

This module is the authenticated admission boundary for the common M1
cell-capacity dispatcher.  Historical direct fixtures retain their named
research override, while the public route is admitted only by the exact
schema-v2 execution profile carried by the zero-step preparation sidecar.
The diagnostic route keeps the suspended equilibrium-capacity channel
observable only.  Explicit exchange mode 4 composes that same channel with
the separately authenticated M2 FLOW 2DH advection/settling/exchange path.

Every host contract is checked before the first CuPy import.  Unsupported
profiles therefore fail closed before CUDA initialisation and never fall back
to a CPU formula.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping

import numpy as np

from braids_core.sediment.capacity_profile import (
    CapacityProfileError,
    CapacityRuntimeProfile,
    capacity_profile_document,
    load_capacity_profile,
)
from braids_core.sediment.channel_composition import (
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionError,
    channel_composition_profile_document,
    channel_composition_semantic_id,
    load_channel_composition_profile,
)
from braids_core.sediment.suspension_profile import (
    SuspensionProfileError,
    load_suspension_profile,
    suspension_profile_document,
    suspension_profile_semantic_id,
)
from braids_core.sediment.execution_profile_v2 import (
    BACKEND_GPU, bind_component_profile)
from .law11_soursin_exchange import (
    CAPACITY_ROUTE as M2_SUSPENDED_ROUTE,
    EXCHANGE_MODE as M2_EXCHANGE_MODE,
    SUSPENSION_FINGERPRINT_KEY,
    SUSPENSION_PROFILE_KEY,
    method_identity as suspension_method_identity,
    validate_scope as validate_suspension_scope,
)
from braids_core.sediment.capacity_foundation_execution import (
    capacity_foundation_execution_document,
    load_capacity_foundation_execution,
)
from braids_core.sediment.hiding_foundation_execution import (
    EXECUTION_KEY as HIDING_EXECUTION_KEY,
    METHOD_BY_MODE as HIDING_METHOD_BY_MODE,
    ORDERED_ROUTE_KEY as HIDING_ORDERED_ROUTE_KEY,
    hiding_foundation_execution_document,
    validate_hiding_foundation_execution_document,
)
from braids_core.sediment.secondary_foundation_execution import (
    EXECUTION_KEY as SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as SECONDARY_ORDERED_ROUTE_KEY,
    validate_secondary_foundation_execution_document,
)
from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)
from braids_core.sediment.bed_slope_execution import (
    EXECUTION_KEY as BED_SLOPE_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as BED_SLOPE_ORDERED_ROUTE_KEY,
    METHODS as BED_SLOPE_METHODS,
    UPSTREAM_STAGES as BED_SLOPE_UPSTREAM_STAGES,
    bed_slope_execution_document,
    validate_bed_slope_execution_document,
)


PROFILE_RUNTIME_SCHEMA = 2
PROFILE_GENERATION = "implicit_m1_law11_profile_v2"
SCALAR_PROFILE_GENERATION = "implicit_q3c_scalar_capacity_profile_v1"
HIDING_PROFILE_GENERATION = "implicit_q3h1_hiding_capacity_profile_v1"
SECONDARY_PROFILE_GENERATION = (
    "implicit_q3h2_classic_flow_equilibrium_secondary_profile_v1"
)
PROGNOSTIC_SECONDARY_PROFILE_GENERATION = (
    "implicit_q3h3_classic_flow_prognostic_secondary_profile_v1"
)
BED_SLOPE_PROFILE_GENERATION = (
    "implicit_q3h4_classic_flow_face_bed_slope_profile_v1"
)
PROTOTYPE_OVERRIDE = "research_law11_vertical_slice_v1"
PUBLIC_ADMISSION_KEY = "m1_profile_public_admission"
PUBLIC_ADMISSION_SCHEMA = 1
PUBLIC_ADMISSION_ID = "q3sp.law11_total_single_fraction_v1"
SCALAR_PUBLIC_ADMISSION_ID = "q3c.scalar_capacity_total_v1"
HIDING_PUBLIC_ADMISSION_ID = "q3h1.scalar_capacity_hiding_v1"
EXECUTION_PROFILE_CONTRACT_KEY = "_execution_profile_v2_contract"
SUSPENDED_ROUTE = "resident_capacity_diagnostic_only_m2_pending"
ROUGHNESS_METHOD = "roughness.manning"
ROUGHNESS_METHODS = frozenset((
    "roughness.chezy", "roughness.manning", "roughness.white_colebrook",
    "roughness.white_colebrook_waqua"))
PROFILE_KEY = "capacity_runtime_profile"
CHANNEL_PROFILE_KEY = "channel_composition_profile"
CHANNEL_FINGERPRINT_KEY = "channel_composition_fingerprint"
_SCALAR_METHODS = {
    2: "capacity.mpm",
    8: "capacity.engelund_hansen",
    11: "capacity.general",
}


def _is_hiding_public(data: Mapping[str, object]) -> bool:
    marker = data.get(PUBLIC_ADMISSION_KEY)
    return (isinstance(marker, Mapping) and
             marker.get("admission_id") == HIDING_PUBLIC_ADMISSION_ID)


def _is_secondary_public(data: Mapping[str, object]) -> bool:
    """Recognize only the additive v56 authority, never numeric model 1."""
    return SECONDARY_EXECUTION_KEY in data


def _is_prognostic_secondary_public(data: Mapping[str, object]) -> bool:
    """Recognize only the additive v57 authority, never numeric model 2."""
    return PROGNOSTIC_SECONDARY_EXECUTION_KEY in data


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(data, key):
    value = np.asarray(data.get(key))
    _require(value.shape == (), "Law-11 profile missing scalar " + key)
    return value.item()


def _suspended_route(data: Mapping[str, object]) -> str:
    return (M2_SUSPENDED_ROUTE if int(_scalar(data, "exchange_mode")) ==
            M2_EXCHANGE_MODE else SUSPENDED_ROUTE)


def public_law11_admission_document(profile, documents: Mapping[str, object]) -> dict:
    """Build the public marker from an already authenticated profile bundle.

    The marker is deliberately not a free-standing feature flag.  It names
    the complete execution-profile fingerprint, its ledger and both component
    document hashes.  :func:`validate_law11_prepared` recomputes it from the
    receipt-authenticated sidecar on every public read.
    """
    _require(getattr(profile, "backend", None) == BACKEND_GPU,
             "Public law-11 admission requires the GPU execution profile")
    layout = tuple(getattr(profile, "fraction_layout", ()))
    _require(len(layout) == 1 and layout[0].transport_class == TRA_BEDLOAD,
             "Public law-11 admission requires one class-1 fraction")
    fraction_id = layout[0].fraction_id
    selected = {(row.entry_id, tuple(row.scope))
                for row in getattr(profile, "method_selections", ())}
    expected = {
        ("capacity.general", (("fraction_id", fraction_id),)),
        ("transport_route.total", (("fraction_id", fraction_id),)),
        ("suspension.off", (("fraction_id", fraction_id),)),
        ("shear.total_roughness", (("model_id", "model"),)),
        ("hiding.none", (("model_id", "model"),)),
        ("bed_slope.off", (("model_id", "model"),)),
        ("secondary.off", (("model_id", "model"),)),
    }
    _require(selected == expected,
             "Public law-11 admission has a different operator chain")
    _require(set(documents) == {"capacity", "channel_composition"},
             "Public law-11 admission has a different component inventory")
    try:
        capacity = load_capacity_profile(documents["capacity"])
        channel = load_channel_composition_profile(
            documents["channel_composition"])
    except (CapacityProfileError, ChannelCompositionError, TypeError) as error:
        raise ValueError(
            "Public law-11 component authentication failed: " + str(error)
        ) from error
    _require(capacity.backend is None and len(capacity.fractions) == 1 and
             capacity.fractions[0].fraction_id == fraction_id and
             capacity.fractions[0].method_id == "capacity.general" and
             capacity.fractions[0].formula_code == 11 and
             capacity.sediment_shear.method_id == "shear.total_roughness" and
             capacity.hiding.method_id == "hiding.none" and
             capacity.bed_slope.method_id == "bed_slope.off" and
             capacity.secondary_flow.method_id == "secondary.off",
             "Public law-11 capacity component differs")
    _require(len(channel.fractions) == 1 and
             channel.fractions[0].fraction_id == fraction_id and
             channel.fractions[0].transport_class == TRA_BEDLOAD and
             channel.epsilon == 1.0e-6,
             "Public law-11 channel component differs")
    bindings = {row.component_id: row
                for row in getattr(profile, "component_profiles", ())}
    _require(set(bindings) == set(documents) and all(
        bind_component_profile(component_id, documents[component_id]) ==
        bindings[component_id] for component_id in documents),
        "Public law-11 component hashes differ from the execution profile")
    return {
        "schema": PUBLIC_ADMISSION_SCHEMA,
        "admission_id": PUBLIC_ADMISSION_ID,
        "backend": BACKEND_GPU,
        "fraction_id": fraction_id,
        "execution_profile_fingerprint": profile.fingerprint,
        "execution_profile_ledger_semantic_id": profile.ledger_semantic_id,
        "capacity_document_sha256": bindings["capacity"].document_sha256,
        "channel_composition_document_sha256":
            bindings["channel_composition"].document_sha256,
    }


def public_scalar_admission_document(profile, documents: Mapping[str, object]) -> dict:
    """Authenticate the complete multi-fraction Q3-C GPU operator chain."""
    _require(getattr(profile, "backend", None) == BACKEND_GPU,
             "Public Q3-C admission requires the GPU execution profile")
    layout = tuple(getattr(profile, "fraction_layout", ()))
    _require(layout and all(row.transport_class == TRA_BEDLOAD for row in layout),
             "Public Q3-C admission requires class-1 fractions")
    fraction_ids = tuple(row.fraction_id for row in layout)
    _require(len(set(fraction_ids)) == len(fraction_ids),
             "Public Q3-C fraction identities are not unique")
    _require(set(documents) == {"capacity", "channel_composition"},
             "Public Q3-C component inventory differs")
    try:
        capacity = load_capacity_profile(documents["capacity"])
        channel = load_channel_composition_profile(
            documents["channel_composition"])
    except (CapacityProfileError, ChannelCompositionError, TypeError) as error:
        raise ValueError(
            "Public Q3-C component authentication failed: " + str(error)
        ) from error
    laws = {row.formula_code for row in capacity.fractions}
    methods = {row.method_id for row in capacity.fractions}
    _require(len(laws) == 1 and len(methods) == 1,
             "Public Q3-C requires one scalar capacity formula")
    law = next(iter(laws))
    method_id = next(iter(methods))
    _require(_SCALAR_METHODS.get(law) == method_id and
             capacity.backend is None and
             tuple(row.fraction_id for row in capacity.fractions) == fraction_ids and
             capacity.sediment_shear.method_id == "shear.total_roughness" and
             capacity.hiding.method_id == "hiding.none" and
             capacity.bed_slope.method_id == "bed_slope.off" and
             capacity.secondary_flow.method_id == "secondary.off",
             "Public Q3-C capacity component differs")
    _require(tuple(row.fraction_id for row in channel.fractions) == fraction_ids and
             all(row.transport_class == TRA_BEDLOAD for row in channel.fractions) and
             channel.epsilon == 1.0e-6,
             "Public Q3-C channel component differs")
    selected = {(row.entry_id, tuple(row.scope))
                for row in getattr(profile, "method_selections", ())}
    expected = set()
    for fraction_id in fraction_ids:
        scope = (("fraction_id", fraction_id),)
        expected.update(((method_id, scope),
                         ("transport_route.total", scope),
                         ("suspension.off", scope)))
    expected.update({
        ("shear.total_roughness", (("model_id", "model"),)),
        ("hiding.none", (("model_id", "model"),)),
        ("bed_slope.off", (("model_id", "model"),)),
        ("secondary.off", (("model_id", "model"),)),
    })
    roughness = {row.entry_id for row in profile.method_selections
                 if row.entry_id in ROUGHNESS_METHODS and
                 tuple(row.scope) == (("model_id", "model"),)}
    _require(len(roughness) == 1,
             "Public Q3-C requires one typed hydraulic-roughness selector")
    expected.add((next(iter(roughness)), (("model_id", "model"),)))
    _require(selected == expected,
             "Public Q3-C admission has a different operator chain")
    bindings = {row.component_id: row
                for row in getattr(profile, "component_profiles", ())}
    _require(set(bindings) == set(documents) and all(
        bind_component_profile(component_id, documents[component_id]) ==
        bindings[component_id] for component_id in documents),
        "Public Q3-C component hashes differ from the execution profile")
    return {
        "schema": PUBLIC_ADMISSION_SCHEMA,
        "admission_id": SCALAR_PUBLIC_ADMISSION_ID,
        "backend": BACKEND_GPU,
        "formula_code": law,
        "method_id": method_id,
        "fraction_ids": list(fraction_ids),
        "roughness_method_id": next(iter(roughness)),
        "execution_profile_fingerprint": profile.fingerprint,
        "execution_profile_ledger_semantic_id": profile.ledger_semantic_id,
        "capacity_document_sha256": bindings["capacity"].document_sha256,
        "channel_composition_document_sha256":
            bindings["channel_composition"].document_sha256,
    }


def public_hiding_admission_document(
        profile, documents: Mapping[str, object],
        hiding_identity: Mapping[str, object]) -> dict:
    """Authenticate Q3-H1 as an additive operator around frozen Q3-C.

    ``profile`` and ``documents`` remain the exact accepted Q3-C foundation
    (whose capacity component names ``hiding.none``).  The separately
    fingerprinted identity owns the actual hiding method and active-surface
    composition.  This avoids mutating or ambiguously reinterpreting a Q3-C
    profile/restart while still making every Q3-H control fail closed.
    """
    base = public_scalar_admission_document(profile, documents)
    identity = validate_hiding_foundation_execution_document(hiding_identity)
    _require(base["formula_code"] in (2, 11) and
             identity["capacity_law"] == base["formula_code"],
             "Q3-H1 supports corrected MPM 2 or general law 11 only")
    _require(len(base["fraction_ids"]) == 2 and
             identity["fraction_ids"] == base["fraction_ids"],
             "Q3-H1 requires the exact two ordered Q3-C fractions")
    _require(identity["capacity_profile_fingerprint"] ==
                 documents["capacity"]["fingerprint"],
             "Q3-H1 capacity profile fingerprint differs from Q3-C")
    return {
        **base,
        "admission_id": HIDING_PUBLIC_ADMISSION_ID,
        "base_admission_id": SCALAR_PUBLIC_ADMISSION_ID,
        "hiding_mode": identity["mode"],
        "hiding_method_id": identity["method_id"],
        "hiding_execution_fingerprint": identity["fingerprint"],
        "hiding_execution_semantic_id": identity["semantic_id"],
    }


def bind_public_law11_runtime(data: dict, profile,
                              documents: Mapping[str, object]) -> dict:
    """Install only deterministic adapters derived from authenticated inputs."""
    _require(isinstance(data, dict),
             "Public law-11 binding requires a mutable prepared mapping")
    marker = public_law11_admission_document(profile, documents)
    _require("m1_profile_research_override" not in data,
             "Public law-11 input cannot carry the research override")
    from .forcing_profile import BINDING_FIELD, forcing_identity
    _require(BINDING_FIELD in data,
             "Public law-11 input lacks authenticated roughness/shear forcing")
    forcing = forcing_identity(data)
    _require(forcing["sediment_shear_method_id"] == "shear.total_roughness" and
             forcing["hydraulic_roughness_method_id"] in ROUGHNESS_METHODS,
             "Public law-11 forcing selects a different roughness/shear route")
    binding = data[BINDING_FIELD]
    viscosity = binding["skin_shear_controls"]["kinematic_viscosity_m2_s"]
    work_depth = forcing["roughness_minimum_work_depth_m"]
    _require(np.isfinite(viscosity) and viscosity > 0.0 and
             np.isfinite(work_depth) and work_depth >= 0.0,
             "Public law-11 viscosity/work-depth identity is invalid")
    derived = {
        "m1_profile_runtime_schema": np.int32(PROFILE_RUNTIME_SCHEMA),
        "capacity_suspended_route": SUSPENDED_ROUTE,
        "capacity_roughness_method_id":
            forcing["hydraulic_roughness_method_id"],
        "capacity_kinematic_viscosity_m2_s": np.float64(viscosity),
        "capacity_roughness_work_depth_m": np.float64(work_depth),
        PUBLIC_ADMISSION_KEY: marker,
    }
    for key, value in derived.items():
        if key in data:
            if isinstance(value, np.generic):
                same = np.array_equal(data[key], value)
            else:
                same = data[key] == value
            _require(same, "Prepared public law-11 runtime field differs: " + key)
        data[key] = value
    return marker


def bind_public_scalar_runtime(data: dict, profile,
                               documents: Mapping[str, object]) -> dict:
    """Install the Q3-C adapter from profile and typed-forcing identities."""
    _require(isinstance(data, dict),
             "Public Q3-C binding requires a mutable prepared mapping")
    marker = public_scalar_admission_document(profile, documents)
    _require(int(np.asarray(data.get("bedlaw")).item()) == marker["formula_code"],
             "Prepared Q3-C law differs from its execution profile")
    _require("m1_profile_research_override" not in data,
             "Public Q3-C input cannot carry the research override")
    boundary_modes=np.asarray(data.get("mode", ()))
    _require(boundary_modes.ndim == 1 and boundary_modes.dtype.kind in "iu" and
             np.isin(boundary_modes, (1, 2, 4, 6, 8, 9)).all(),
             "Public Q3-C GPU boundaries support only modes 1/2/4/6/8/9; "
             "supercritical mode 3 and weir mode 5 remain fail-closed")
    from .forcing_profile import BINDING_FIELD, forcing_identity
    _require(BINDING_FIELD in data,
             "Public Q3-C input lacks authenticated roughness/shear forcing")
    forcing = forcing_identity(data)
    _require(forcing["sediment_shear_method_id"] == "shear.total_roughness" and
             forcing["hydraulic_roughness_method_id"] ==
                 marker["roughness_method_id"],
             "Public Q3-C forcing differs from the selected operator chain")
    binding = data[BINDING_FIELD]
    viscosity = binding["skin_shear_controls"]["kinematic_viscosity_m2_s"]
    work_depth = forcing["roughness_minimum_work_depth_m"]
    _require(np.isfinite(viscosity) and viscosity > 0.0 and
             np.isfinite(work_depth) and work_depth >= 0.0,
             "Public Q3-C viscosity/work-depth identity is invalid")
    execution_identity = capacity_foundation_execution_document(
        method_id=marker["method_id"], formula_code=marker["formula_code"],
        forcing_fingerprint=forcing["delft_forcing_fingerprint"],
        roughness_method_id=forcing["hydraulic_roughness_method_id"],
        capacity_profile_fingerprint=documents["capacity"]["fingerprint"])
    route_identity = {
        "ordered_parts": ["capacity_execution_fingerprint",
                          "capacity_profile_fingerprint"],
        "capacity_execution_fingerprint": execution_identity["fingerprint"],
        "capacity_profile_fingerprint": documents["capacity"]["fingerprint"],
    }
    derived = {
        "m1_profile_runtime_schema": np.int32(PROFILE_RUNTIME_SCHEMA),
        "capacity_suspended_route": SUSPENDED_ROUTE,
        "capacity_roughness_method_id":
            forcing["hydraulic_roughness_method_id"],
        "capacity_kinematic_viscosity_m2_s": np.float64(viscosity),
        "capacity_roughness_work_depth_m": np.float64(work_depth),
        PUBLIC_ADMISSION_KEY: marker,
        "capacity_foundation_execution_identity": execution_identity,
        "capacity_foundation_ordered_route_identity": route_identity,
    }
    for key, value in derived.items():
        if key in data:
            same = (np.array_equal(data[key], value) if isinstance(value, np.generic)
                    else data[key] == value)
            _require(same, "Prepared public Q3-C runtime field differs: " + key)
        data[key] = value
    return marker


def bind_public_hiding_runtime(
        data: dict, profile, documents: Mapping[str, object], *,
        mode: int, method_id: str, parker_alpha: float = 0.6,
        wu_exponent: float = -0.6) -> dict:
    """Bind Q3-H1 on top of, never in place of, the Q3-C identity."""
    _require(isinstance(data, dict),
             "Public Q3-H1 binding requires a mutable prepared mapping")
    # Construct and authenticate the frozen foundation first.  It installs
    # the exact typed-forcing execution identity consumed below.
    base_marker = bind_public_scalar_runtime(data, profile, documents)
    execution = data["capacity_foundation_execution_identity"]
    identity = hiding_foundation_execution_document(
        mode=mode, method_id=method_id,
        capacity_law=base_marker["formula_code"],
        fraction_ids=base_marker["fraction_ids"],
        parker_alpha=parker_alpha, wu_exponent=wu_exponent,
        capacity_execution_fingerprint=execution["fingerprint"],
        capacity_profile_fingerprint=documents["capacity"]["fingerprint"],
    )
    marker = public_hiding_admission_document(profile, documents, identity)
    native_mode = np.asarray(data.get("hiding_exposure"))
    _require(native_mode.shape == () and native_mode.dtype.kind in "iu",
             "Prepared Q3-H1 hiding mode is not an integer scalar")
    derived = {
        # Retain the native record width so the authenticated NPZ/native
        # archive comparison remains exact after this additive host binding.
        "hiding_exposure": native_mode.dtype.type(mode),
        "hiding_parker_alpha": np.float64(parker_alpha),
        "hiding_wu_exponent": np.float64(wu_exponent),
        HIDING_EXECUTION_KEY: identity,
        HIDING_ORDERED_ROUTE_KEY: {
            "ordered_parts": [
                "capacity_execution_fingerprint",
                "capacity_profile_fingerprint",
                "hiding_execution_fingerprint",
            ],
            "capacity_execution_fingerprint": execution["fingerprint"],
            "capacity_profile_fingerprint": documents["capacity"][
                "fingerprint"],
            "hiding_execution_fingerprint": identity["fingerprint"],
        },
    }
    # The public marker is intentionally replaced only after the additive
    # identity is complete.  Every other field remains the Q3-C base byte.
    data[PUBLIC_ADMISSION_KEY] = marker
    for key, value in derived.items():
        if key in data:
            same = (np.array_equal(data[key], value)
                    if isinstance(value, np.generic) else data[key] == value)
            _require(same, "Prepared public Q3-H1 runtime field differs: " + key)
        data[key] = value
    return marker


def _validate_public_law11_admission(data: Mapping[str, object]) -> dict:
    contract = data.get(EXECUTION_PROFILE_CONTRACT_KEY)
    _require(isinstance(contract, Mapping),
             "Public law-11 marker lacks its authenticated execution profile")
    profile = contract.get("profile")
    documents = contract.get("component_documents")
    identities = contract.get("identities")
    _require(profile is not None and isinstance(documents, Mapping) and
             isinstance(identities, Mapping),
             "Public law-11 execution-profile contract differs")
    expected = public_law11_admission_document(profile, documents)
    _require(data.get(PUBLIC_ADMISSION_KEY) == expected and
             identities.get("execution_profile_v2_admission") == "runtime" and
             identities.get("execution_profile_v2_backend") == BACKEND_GPU and
             identities.get("execution_profile_v2_profile_backend") ==
                 BACKEND_GPU and
             identities.get("execution_profile_v2_fingerprint") ==
                 profile.fingerprint and
             identities.get("execution_profile_v2_ledger_semantic_id") ==
                 profile.ledger_semantic_id,
             "Public law-11 admission marker differs from its authenticated profile")
    _require(data.get(PROFILE_KEY) == documents["capacity"] and
             data.get(CHANNEL_PROFILE_KEY) == documents["channel_composition"],
             "Public law-11 runtime components differ from the authenticated profile")
    return expected


def _validate_public_scalar_admission(data: Mapping[str, object]) -> dict:
    contract = data.get(EXECUTION_PROFILE_CONTRACT_KEY)
    _require(isinstance(contract, Mapping),
             "Public Q3-C marker lacks its authenticated execution profile")
    profile = contract.get("profile")
    documents = contract.get("component_documents")
    identities = contract.get("identities")
    _require(profile is not None and isinstance(documents, Mapping) and
             isinstance(identities, Mapping),
             "Public Q3-C execution-profile contract differs")
    expected = public_scalar_admission_document(profile, documents)
    _require(data.get(PUBLIC_ADMISSION_KEY) == expected and
             identities.get("execution_profile_v2_admission") == "runtime" and
             identities.get("execution_profile_v2_backend") == BACKEND_GPU and
             identities.get("execution_profile_v2_profile_backend") == BACKEND_GPU and
             identities.get("execution_profile_v2_fingerprint") ==
                 profile.fingerprint and
             identities.get("execution_profile_v2_ledger_semantic_id") ==
                 profile.ledger_semantic_id,
             "Public Q3-C admission differs from its authenticated profile")
    _require(data.get(PROFILE_KEY) == documents["capacity"] and
             data.get(CHANNEL_PROFILE_KEY) == documents["channel_composition"],
             "Public Q3-C runtime components differ from the authenticated profile")
    return expected


def _validate_public_hiding_admission(data: Mapping[str, object]) -> dict:
    contract = data.get(EXECUTION_PROFILE_CONTRACT_KEY)
    _require(isinstance(contract, Mapping),
             "Public Q3-H1 marker lacks its authenticated Q3-C profile")
    profile = contract.get("profile")
    documents = contract.get("component_documents")
    identities = contract.get("identities")
    identity = validate_hiding_foundation_execution_document(
        data.get(HIDING_EXECUTION_KEY))
    _require(profile is not None and isinstance(documents, Mapping) and
             isinstance(identities, Mapping),
             "Public Q3-H1 execution-profile contract differs")
    expected = public_hiding_admission_document(profile, documents, identity)
    _require(data.get(PUBLIC_ADMISSION_KEY) == expected and
             identities.get("execution_profile_v2_admission") == "runtime" and
             identities.get("execution_profile_v2_backend") == BACKEND_GPU and
             identities.get("execution_profile_v2_profile_backend") ==
                 BACKEND_GPU and
             identities.get("execution_profile_v2_fingerprint") ==
                 profile.fingerprint and
             identities.get("execution_profile_v2_ledger_semantic_id") ==
                 profile.ledger_semantic_id,
             "Public Q3-H1 admission differs from its Q3-C foundation")
    _require(data.get(PROFILE_KEY) == documents["capacity"] and
             data.get(CHANNEL_PROFILE_KEY) == documents["channel_composition"],
             "Public Q3-H1 runtime components differ from Q3-C")
    base_execution = data.get("capacity_foundation_execution_identity")
    route = data.get(HIDING_ORDERED_ROUTE_KEY)
    _require(isinstance(base_execution, Mapping) and
             identity["capacity_execution_fingerprint"] ==
                 base_execution.get("fingerprint") and
             route == {
                 "ordered_parts": [
                     "capacity_execution_fingerprint",
                     "capacity_profile_fingerprint",
                     "hiding_execution_fingerprint",
                 ],
                 "capacity_execution_fingerprint": base_execution.get(
                     "fingerprint"),
                 "capacity_profile_fingerprint": documents["capacity"][
                     "fingerprint"],
                 "hiding_execution_fingerprint": identity["fingerprint"],
             },
             "Public Q3-H1 ordered route differs")
    return expected


def validate_secondary_composition(
        data: Mapping[str, object]) -> dict | None:
    """Authenticate the bounded classic FLOW secondary outer composition.

    The nested Q3-C/v54 or Q3-H1/v55 marker is deliberately left untouched.
    Numeric ``secondary_flow_model=1`` without this exact v56 document and
    ordered route is not an admission mechanism.
    """
    if not _is_secondary_public(data):
        return None
    identity = validate_secondary_foundation_execution_document(
        data.get(SECONDARY_EXECUTION_KEY))
    base = load_capacity_foundation_execution(
        data.get("capacity_foundation_execution_identity"))
    hiding = (validate_hiding_foundation_execution_document(
        data.get(HIDING_EXECUTION_KEY)) if _is_hiding_public(data) else None)
    route = data.get(SECONDARY_ORDERED_ROUTE_KEY)
    expected_route = {
        "ordered_parts": [
            "capacity_execution_fingerprint",
            "capacity_profile_fingerprint",
            "hiding_execution_fingerprint",
            "secondary_execution_fingerprint",
        ],
        "capacity_execution_fingerprint": base["fingerprint"],
        "capacity_profile_fingerprint": data.get(
            "capacity_profile_fingerprint"),
        "hiding_execution_fingerprint": (
            None if hiding is None else hiding["fingerprint"]),
        "secondary_execution_fingerprint": identity["fingerprint"],
    }
    marker = data.get(PUBLIC_ADMISSION_KEY)
    expected_marker = (HIDING_PUBLIC_ADMISSION_ID if hiding is not None else
                       SCALAR_PUBLIC_ADMISSION_ID)
    _require(isinstance(marker, Mapping) and
             marker.get("admission_id") == expected_marker and
             route == expected_route,
             "Q3-H2 ordered v56 route differs from its frozen foundation")
    from .forcing_profile import forcing_identity
    forcing = forcing_identity(data)
    _require(
        identity["capacity_execution_fingerprint"] == base["fingerprint"] and
        identity["capacity_profile_fingerprint"] ==
            data.get("capacity_profile_fingerprint") and
        identity["hiding_execution_fingerprint"] ==
            (None if hiding is None else hiding["fingerprint"]) and
        identity["base_envelope_version"] == (55 if hiding is not None else 54) and
        identity["capacity_law"] == int(_scalar(data, "bedlaw")) and
        identity["roughness_method_id"] ==
            forcing["hydraulic_roughness_method_id"] and
        identity["roughness_fingerprint"] ==
            forcing["delft_forcing_fingerprint"] ==
            data.get("delft_forcing_fingerprint"),
        "Q3-H2 identity differs from capacity, hiding, or typed roughness")
    _require(
        identity["coriolis_disposition"] == "zero" and
        identity["coriolis_constant_s_inverse"] == 0.0 and
        identity["coriolis_source"] == "constant_zero_planar" and
        identity["betac"] == 0.0 and
        int(_scalar(data, "secondary_flow_model")) == 1 and
        int(_scalar(data, "secondary_flow")) == 1 and
        int(_scalar(data, "secondary_flow_transport_direction")) == 1 and
        float(_scalar(data, "secondary_flow_momentum_beta")) == 0.0 and
        float(_scalar(data, "secondary_flow_espir")) == identity["espir"] and
        (int(_scalar(data, "bed_slope")) == 0 or
         BED_SLOPE_EXECUTION_KEY in data) and
        int(_scalar(data, "exchange_mode")) == 0 and
        int(_scalar(data, "bedload_adaptation")) == 0 and
        int(_scalar(data, "bedform_roughness")) == 0 and
        float(_scalar(data, "horizontal_viscosity")) == 0.0 and
        not bool(np.any(data.get("amr_enabled", 0))),
        "Q3-H2 first live route requires fixed-grid equilibrium secondary, "
        "zero Coriolis/Betac, and slope/exchange/adaptation/bedform/viscosity off")
    return identity


def validate_prognostic_secondary_composition(
        data: Mapping[str, object]) -> dict | None:
    """Authenticate the bounded v57 classic FLOW prognostic composition."""
    if not _is_prognostic_secondary_public(data):
        return None
    _require(not _is_secondary_public(data),
             "Q3-H3 cannot execute the Q3-H2 equilibrium rotation")
    identity = validate_prognostic_secondary_execution_document(
        data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
    base = load_capacity_foundation_execution(
        data.get("capacity_foundation_execution_identity"))
    hiding = (validate_hiding_foundation_execution_document(
        data.get(HIDING_EXECUTION_KEY)) if _is_hiding_public(data) else None)
    route = data.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)
    expected_route = {
        "capacity_execution_fingerprint": base["fingerprint"],
        "capacity_profile_fingerprint": data.get(
            "capacity_profile_fingerprint"),
        "hiding_execution_fingerprint": (
            None if hiding is None else hiding["fingerprint"]),
        "prognostic_secondary_execution_fingerprint":
            identity["fingerprint"],
    }
    marker = data.get(PUBLIC_ADMISSION_KEY)
    expected_marker = (HIDING_PUBLIC_ADMISSION_ID if hiding is not None else
                       SCALAR_PUBLIC_ADMISSION_ID)
    _require(isinstance(marker, Mapping) and
             marker.get("admission_id") == expected_marker and
             route == expected_route,
             "Q3-H3 ordered v57 route differs from its frozen foundation")
    from .forcing_profile import forcing_identity
    forcing = forcing_identity(data)
    layout = identity["runtime_layout"]
    _require(
        identity["capacity_execution_fingerprint"] == base["fingerprint"] and
        identity["capacity_profile_fingerprint"] ==
            data.get("capacity_profile_fingerprint") and
        identity["hiding_execution_fingerprint"] ==
            (None if hiding is None else hiding["fingerprint"]) and
        identity["base_envelope_version"] ==
            (55 if hiding is not None else 54) and
        identity["capacity_law"] == int(_scalar(data, "bedlaw")) and
        identity["roughness_method_id"] ==
            forcing["hydraulic_roughness_method_id"] and
        identity["roughness_fingerprint"] ==
            forcing["delft_forcing_fingerprint"] ==
            data.get("delft_forcing_fingerprint") and
        layout["fraction_count"] == int(_scalar(data, "nfrac")) and
        layout["bed_layer_count"] == int(_scalar(data, "nbed_layers")),
        "Q3-H3 identity differs from capacity, hiding, layout, or roughness")
    _require(
        identity["transport_direction"] is True and
        identity["inflow_mode"] == 1 and
        identity["coriolis_disposition"] == "zero" and
        identity["coriolis_constant_s_inverse"] == 0.0 and
        identity["coriolis_source"] == "constant_zero_planar" and
        identity["betac"] == 0.0 and
        identity["scalar_diffusivity_m2_s"] == 0.0 and
        int(_scalar(data, "secondary_flow_model")) == 2 and
        int(_scalar(data, "secondary_flow")) == 1 and
        int(_scalar(data, "secondary_flow_transport_direction")) == 1 and
        int(_scalar(data, "secondary_flow_inflow_mode")) == 1 and
        float(_scalar(data, "secondary_flow_momentum_beta")) == 0.0 and
        float(_scalar(data, "secondary_flow_espir")) == identity["espir"] and
        (int(_scalar(data, "bed_slope")) == 0 or
         BED_SLOPE_EXECUTION_KEY in data) and
        int(_scalar(data, "exchange_mode")) == 0 and
        int(_scalar(data, "bedload_adaptation")) == 0 and
        int(_scalar(data, "bedform_roughness")) == 0 and
        float(_scalar(data, "horizontal_viscosity")) == 0.0 and
        not bool(np.any(data.get("amr_enabled", 0))) and
        not bool(np.any(data.get("bank_failure", 0))) and
        not bool(np.any(data.get("bank_front_tracking", 0))) and
        not bool(np.any(data.get("bank_toe_erosion_mode", 0))) and
        not bool(np.any(data.get("bank_avalanche", 0))) and
        not bool(np.any(data.get("general_avalanche", 0))) and
        not bool(np.any(data.get("bank_erosion_fraction", 0.0))),
        "Q3-H3 first live route requires fixed-grid prognostic secondary, "
        "zero Coriolis/Betac/diffusivity, and slope/exchange/adaptation/"
        "bedform/viscosity/bank/avalanche physics off")
    # Reuse the exact low-level first-admission gate, including state/ledger
    # indices and the canonical ordered route.  This remains host-only.
    from .flow2dh_prognostic_secondary import validate_scope
    validate_scope(dict(data))
    return identity


def _bed_slope_stage_hash(stage: str, payload: Mapping[str, object]) -> str:
    """Stable fingerprint for an explicitly inactive/upstream H4 stage."""
    encoded = json.dumps(
        {"stage": stage, **dict(payload)}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def bed_slope_upstream_fingerprints(
        data: Mapping[str, object]) -> dict[str, str]:
    """Resolve the exact accepted operators feeding the H4 face seam.

    This helper deliberately does not inspect ``bed_slope`` itself, so it can
    be used both to create and to revalidate the additive identity.  Every
    inactive auxiliary is represented by a typed fingerprint, never by a
    missing or wildcard field.
    """
    base = load_capacity_foundation_execution(
        data.get("capacity_foundation_execution_identity"))
    channel = load_channel_composition_profile(data.get(CHANNEL_PROFILE_KEY))
    hiding = (validate_hiding_foundation_execution_document(
        data.get(HIDING_EXECUTION_KEY)) if _is_hiding_public(data) else None)
    secondary = None
    secondary_stage = "secondary.off"
    if _is_secondary_public(data):
        secondary = validate_secondary_foundation_execution_document(
            data.get(SECONDARY_EXECUTION_KEY))
        secondary_stage = secondary["method_id"]
    if _is_prognostic_secondary_public(data):
        _require(secondary is None,
                 "Q3-H4 upstream secondary owners are mutually exclusive")
        secondary = validate_prognostic_secondary_execution_document(
            data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
        secondary_stage = secondary["method_id"]
    face_source = Path(__file__).with_name("m1_profile_faces.cu")
    _require(face_source.is_file() and not face_source.is_symlink(),
             "Q3-H4 reconstruction source is missing or linked")
    capacity_fingerprint = str(base["fingerprint"])
    return {
        "capacity_execution": capacity_fingerprint,
        "secondary_steering_execution": (
            str(secondary["fingerprint"]) if secondary is not None else
            _bed_slope_stage_hash("secondary_steering_execution", {
                "method_id": secondary_stage,
                "capacity_execution_fingerprint": capacity_fingerprint,
            })
        ),
        "channel_composition_execution": str(channel.fingerprint),
        "face_reconstruction_execution": _bed_slope_stage_hash(
            "face_reconstruction_execution", {
                "method_id": "braids.scalar_capacity_reconstruction_v1",
                "source_sha256": hashlib.sha256(
                    face_source.read_bytes()).hexdigest(),
            }),
        "hiding_execution": (
            str(hiding["fingerprint"]) if hiding is not None else
            _bed_slope_stage_hash("hiding_execution", {
                "method_id": "hiding.none",
                "capacity_execution_fingerprint": capacity_fingerprint,
            })
        ),
    }


def bind_public_bed_slope_runtime(
        data: dict, *, method_id: str, alfabs: float, alfabn: float,
        ashld: float, bshld: float, cshld: float, dshld: float,
        alfpa: float, thcrpa: float) -> dict:
    """Bind an explicit classic FLOW slope choice to an accepted Q3-C chain."""
    _require(isinstance(data, dict),
             "Q3-H4 binding requires a mutable prepared mapping")
    # Authenticate the complete frozen foundation before changing its numeric
    # legacy selector.  The nested capacity profile remains slope-off; this
    # separate identity owns the additive face operator.
    profile = validate_scalar_capacity_prepared(data)
    _require(BED_SLOPE_EXECUTION_KEY not in data and
             BED_SLOPE_ORDERED_ROUTE_KEY not in data and
             int(_scalar(data, "bed_slope")) == 0,
             "Q3-H4 binding requires an unbound slope-off foundation")
    upstream = bed_slope_upstream_fingerprints(data)
    identity = bed_slope_execution_document(
        method_id=method_id,
        fraction_ids=tuple(row.fraction_id for row in profile.fractions),
        upstream_fingerprints=upstream,
        alfabs=alfabs, alfabn=alfabn, ashld=ashld, bshld=bshld,
        cshld=cshld, dshld=dshld, alfpa=alfpa, thcrpa=thcrpa,
    )
    native = np.asarray(data["bed_slope"])
    _require(native.shape == () and native.dtype.kind in "iu",
             "Q3-H4 native bed-slope selector is not an integer scalar")
    data["bed_slope"] = native.dtype.type(identity["persistence_code"])
    data[BED_SLOPE_EXECUTION_KEY] = identity
    data[BED_SLOPE_ORDERED_ROUTE_KEY] = {
        "ordered_parts": list(BED_SLOPE_UPSTREAM_STAGES) + [
            "bed_slope_execution"],
        **upstream,
        "bed_slope_execution": identity["fingerprint"],
    }
    return identity


def validate_bed_slope_composition(
        data: Mapping[str, object]) -> dict | None:
    """Authenticate the live H4 operator or require an exact slope-off state."""
    present = BED_SLOPE_EXECUTION_KEY in data
    if not present:
        _require(BED_SLOPE_ORDERED_ROUTE_KEY not in data and
                 int(_scalar(data, "bed_slope")) == 0,
                 "Active Q3-H4 slope requires its explicit execution identity")
        return None
    identity = validate_bed_slope_execution_document(
        data.get(BED_SLOPE_EXECUTION_KEY))
    _require(int(_scalar(data, "bed_slope")) ==
                 identity["persistence_code"],
             "Q3-H4 method identity differs from the saved numeric selector")
    fraction_ids = data.get("capacity_fraction_ids")
    _require(isinstance(fraction_ids, (tuple, list)) and
             list(fraction_ids) == identity["ordered_fraction_ids"],
             "Q3-H4 ordered fractions differ from capacity")
    upstream = bed_slope_upstream_fingerprints(data)
    supplied = {row["stage"]: row["fingerprint"]
                for row in identity["upstream_identities"]}
    expected_route = {
        "ordered_parts": list(BED_SLOPE_UPSTREAM_STAGES) + [
            "bed_slope_execution"],
        **upstream,
        "bed_slope_execution": identity["fingerprint"],
    }
    _require(supplied == upstream and
             data.get(BED_SLOPE_ORDERED_ROUTE_KEY) == expected_route,
             "Q3-H4 upstream/operator route identity differs")
    _require(int(_scalar(data, "bedlaw")) in _SCALAR_METHODS and
             int(_scalar(data, "exchange_mode")) == 0 and
             int(_scalar(data, "bedload_adaptation")) == 0,
             "Q3-H4 first live route requires scalar bedload with exchange "
             "and adaptation off")
    return identity


def _channel_profile(data: Mapping[str, object], profile: CapacityRuntimeProfile):
    """Authenticate the single post-density owner for Bed/Sus/class order."""
    try:
        channel = load_channel_composition_profile(data.get(CHANNEL_PROFILE_KEY))
    except ChannelCompositionError as error:
        raise ValueError(
            "Law-11 channel-composition profile authentication failed: " +
            str(error)
        ) from error
    _require(data.get(CHANNEL_FINGERPRINT_KEY) == channel.fingerprint,
             "Law-11 prepared/channel-composition fingerprint differs")
    fraction_ids = tuple(value.fraction_id for value in profile.fractions)
    expected_class = (TRA_COMBINE if int(_scalar(data, "exchange_mode")) ==
                      M2_EXCHANGE_MODE else TRA_BEDLOAD)
    _require(tuple(value.fraction_id for value in channel.fractions) ==
             fraction_ids and
             all(value.transport_class == expected_class
                 for value in channel.fractions),
             "Law-11 transport classes differ from exchange ownership")
    _require(channel.semantic_id == channel_composition_semantic_id(),
             "Law-11 channel-composition semantic identity differs")
    _require(channel.bed_factor == float(_scalar(data, "transport_calibration")),
             "Law-11 global Bed factor differs from the prepared transport calibration")
    expected_sus = float(data.get("suspended_calibration_factor", 1.0))
    _require(channel.suspended_factor == expected_sus,
             "Law-11 global Sus factor differs from the prepared suspension calibration")
    _require(channel.epsilon == 1.0e-6,
             "Law-11 first live composer slice requires FLOW epsilon 1e-6")
    return channel


def validate_law11_prepared(data: Mapping[str, object], *,
                            require_prototype_override: bool = True
                            ) -> CapacityRuntimeProfile:
    """Authenticate the narrow law-11 prepared contract without importing CUDA."""
    _require(isinstance(data, Mapping), "Law-11 prepared input must be a mapping")
    _require(int(_scalar(data, "bedlaw")) == 11,
             "Law-11 profile bridge cannot serve another capacity law")
    _require(int(_scalar(data, "m1_profile_runtime_schema")) == PROFILE_RUNTIME_SCHEMA,
             "Law-11 profile runtime schema differs")
    if require_prototype_override:
        research = (
            data.get("m1_profile_research_override") == PROTOTYPE_OVERRIDE and
            PUBLIC_ADMISSION_KEY not in data)
        public = (
            "m1_profile_research_override" not in data and
            PUBLIC_ADMISSION_KEY in data)
        _require(research != public,
                 "Law-11 runtime requires exactly one authenticated admission")
        if public:
            _validate_public_law11_admission(data)
    exchange_mode = int(_scalar(data, "exchange_mode"))
    _require(exchange_mode in (0, M2_EXCHANGE_MODE),
             "Law-11 GPU profile supports diagnostic exchange OFF or explicit mode 4")
    suspended_route = _suspended_route(data)
    _require(data.get("capacity_suspended_route") == suspended_route,
             "Law-11 suspended-capacity route differs from the selected exchange")
    _require(data.get("capacity_roughness_method_id") in ROUGHNESS_METHODS,
             "Law-11 vertical slice lacks an authenticated C/M/W route")
    from .forcing_profile import BINDING_FIELD, forcing_identity
    if BINDING_FIELD in data:
        forcing = forcing_identity(data)
        _require(forcing["hydraulic_roughness_method_id"] ==
                 data.get("capacity_roughness_method_id") and
                 forcing["sediment_shear_method_id"] == "shear.total_roughness",
                 "Law-11 forcing/profile identities differ")
    else:
        _require(data.get("capacity_roughness_method_id") == ROUGHNESS_METHOD,
                 "Historical law-11 input is Manning-only")
    _require(PROFILE_KEY in data, "Law-11 prepared input lacks a capacity profile")
    try:
        profile = load_capacity_profile(data[PROFILE_KEY])
    except CapacityProfileError as error:
        raise ValueError("Law-11 capacity profile authentication failed: " + str(error)) from error
    _require(profile.backend is None,
             "Prototype law-11 profile must retain semantic backend=None identity")
    _require(data.get("capacity_profile_fingerprint") == profile.fingerprint,
             "Law-11 prepared/profile fingerprint differs")
    nfrac = int(_scalar(data, "nfrac"))
    fraction_ids = data.get("capacity_fraction_ids")
    _require(isinstance(fraction_ids, (tuple, list)) and
             all(isinstance(value, str) for value in fraction_ids) and
             len(fraction_ids) == nfrac and len(set(fraction_ids)) == nfrac,
             "Law-11 prepared fraction identities differ")
    _require(tuple(value.fraction_id for value in profile.fractions) == tuple(fraction_ids),
             "Law-11 capacity profile fraction order differs from prepared state")
    _require(all(value.formula_code == 11 and
                 value.method_id == "capacity.general" and
                 value.reference_selector == 4 and
                 value.output.bed_channel == "scalar_magnitude" and
                 value.output.suspended_channel == "scalar_magnitude" and
                 value.output.bed_direction == "external_scalar_direction" and
                 value.output.suspended_direction ==
                 "depth_averaged_flow_direction"
                 for value in profile.fractions),
             "Law-11 profile does not contain complete general-formula channels")
    _require(profile.sediment_shear.method_id == "shear.total_roughness" and
             profile.hiding.method_id == "hiding.none" and
             profile.bed_slope.method_id == "bed_slope.off" and
             profile.secondary_flow.method_id == "secondary.off",
             "Law-11 vertical slice requires total shear, hiding/slope/secondary OFF")
    _require(int(_scalar(data, "hiding_exposure")) == 0 and
             int(_scalar(data, "bed_slope")) == 0 and
             int(_scalar(data, "secondary_flow_model")) == 0,
             "Law-11 slice excludes hiding, slope and secondary flow")
    if exchange_mode == 0:
        _require(not bool(np.any(data.get("suspended_transport_active", False))) and
                 not bool(np.any(data.get("suspended_mass", 0))) and
                 not bool(np.any(data.get("fraction_concentration", 0))) and
                 not bool(np.any(data.get("fraction_series_concentration", 0))),
                 "Law-11 diagnostic route excludes suspended M2 state")
    else:
        _require(nfrac == 1 and all(
            float(dict((item.keyword, item.value)
                       for item in fraction.coefficients)["SusACal"]) > 0.0
            for fraction in profile.fractions),
            "Law-11 exchange mode 4 requires one fraction with positive SusACal")
        try:
            suspension_profile = load_suspension_profile(
                data.get(SUSPENSION_PROFILE_KEY))
        except SuspensionProfileError as error:
            raise ValueError(
                "Law-11 suspension-profile authentication failed: " + str(error)
            ) from error
        _require(data.get(SUSPENSION_FINGERPRINT_KEY) ==
                 suspension_profile.fingerprint and
                 suspension_profile.capacity_profile.fingerprint ==
                 profile.fingerprint,
                 "Law-11 suspension and capacity profile identities differ")
        validate_suspension_scope(dict(data))
    modes = np.asarray(data.get("mode"))
    bed_modes = np.asarray(data.get("bed_boundary_mode"))
    boundary_count = int(_scalar(data, "nb"))
    _require(modes.shape == (boundary_count,) and
             bed_modes.shape == (boundary_count,),
             "Law-11 M1 slice requires explicit boundary mode vectors")
    inlets = np.isin(modes, (1, 6, 7, 8, 9))
    _require(np.all(bed_modes[inlets] == 1) and
             np.all(bed_modes[~inlets] == 0),
             "Law-11 M1 slice supports equilibrium-capacity inlets and stage outlets only")
    for key in ("bc_bed_input", "fraction_bed_input", "series_bed_input",
                "fraction_series_bed_input"):
        _require(not np.any(data.get(key, 0)),
                 "Law-11 M1 slice excludes prescribed bed feed: " + key)
    viscosity = float(_scalar(data, "capacity_kinematic_viscosity_m2_s"))
    work_depth = float(_scalar(data, "capacity_roughness_work_depth_m"))
    _require(np.isfinite(viscosity) and viscosity > 0.0 and
             np.isfinite(work_depth) and work_depth >= 0.0,
             "Law-11 viscosity/work-depth controls are invalid")
    bed_factor = float(_scalar(data, "transport_calibration"))
    _require(np.isfinite(bed_factor) and bed_factor >= 0.0,
             "Law-11 global Bed factor must be finite and nonnegative")
    _channel_profile(data, profile)
    return profile


def validate_scalar_capacity_prepared(
        data: Mapping[str, object]) -> CapacityRuntimeProfile:
    """Authenticate Q3-C and the additive Q3-H1/Q3-H2 compositions."""
    _require(isinstance(data, Mapping),
             "Q3-C/Q3-H1 prepared input must be a mapping")
    law = int(_scalar(data, "bedlaw"))
    _require(law in _SCALAR_METHODS,
             "Q3-C bridge supports corrected laws 2, 8 and 11 only")
    _require(int(_scalar(data, "m1_profile_runtime_schema")) ==
             PROFILE_RUNTIME_SCHEMA,
             "Q3-C profile runtime schema differs")
    hiding_public = _is_hiding_public(data)
    marker = (_validate_public_hiding_admission(data) if hiding_public
              else _validate_public_scalar_admission(data))
    _require(marker["formula_code"] == law,
             "Q3-C admission formula differs from prepared state")
    _require(int(_scalar(data, "exchange_mode")) == 0 and
             data.get("capacity_suspended_route") == SUSPENDED_ROUTE,
             "Q3-C total route requires diagnostic-only formula suspension")
    from .forcing_profile import BINDING_FIELD, forcing_identity
    _require(BINDING_FIELD in data,
             "Q3-C prepared state lacks typed roughness forcing")
    forcing = forcing_identity(data)
    _require(data.get("capacity_roughness_method_id") ==
                 marker["roughness_method_id"] ==
                 forcing["hydraulic_roughness_method_id"] and
             forcing["sediment_shear_method_id"] == "shear.total_roughness",
              "Q3-C forcing/profile identities differ")
    secondary_identity = validate_secondary_composition(data)
    prognostic_secondary_identity = \
        validate_prognostic_secondary_composition(data)
    bed_slope_identity = validate_bed_slope_composition(data)
    _require(not (secondary_identity is not None and
                  prognostic_secondary_identity is not None),
             "Q3-H2 and Q3-H3 secondary owners are mutually exclusive")
    try:
        profile = load_capacity_profile(data[PROFILE_KEY])
    except (CapacityProfileError, KeyError) as error:
        raise ValueError("Q3-C capacity profile authentication failed: " +
                         str(error)) from error
    nfrac = int(_scalar(data, "nfrac"))
    fraction_ids = data.get("capacity_fraction_ids")
    _require(profile.backend is None and
             data.get("capacity_profile_fingerprint") == profile.fingerprint and
             isinstance(fraction_ids, (tuple, list)) and
             tuple(row.fraction_id for row in profile.fractions) ==
                 tuple(fraction_ids) and len(profile.fractions) == nfrac and
             all(row.formula_code == law and
                 row.method_id == _SCALAR_METHODS[law]
                 for row in profile.fractions),
             "Q3-C capacity profile differs from prepared state")
    hiding_mode = int(_scalar(data, "hiding_exposure"))
    _require(profile.sediment_shear.method_id == "shear.total_roughness" and
             # The nested component is the frozen Q3-C foundation; Q3-H1's
             # selected physical hiding method is owned by its outer identity.
             profile.hiding.method_id == "hiding.none" and
             profile.bed_slope.method_id == "bed_slope.off" and
              profile.secondary_flow.method_id == "secondary.off" and
              int(_scalar(data, "native_sediment_shear_mode")) == 0 and
              int(_scalar(data, "bed_slope")) == (
                  0 if bed_slope_identity is None else
                  bed_slope_identity["persistence_code"]) and
              int(_scalar(data, "secondary_flow_model")) ==
                  (2 if prognostic_secondary_identity is not None else
                   1 if secondary_identity is not None else 0),
              "Q3-C/Q3-H1/Q3-H2/Q3-H3/Q3-H4 auxiliary operator chain differs")
    if hiding_public:
        identity = validate_hiding_foundation_execution_document(
            data.get(HIDING_EXECUTION_KEY))
        _require(law in (2, 11) and nfrac == 2 and
                 hiding_mode == marker["hiding_mode"] == identity["mode"] and
                 marker["hiding_method_id"] == identity["method_id"] ==
                     HIDING_METHOD_BY_MODE[hiding_mode] and
                 float(_scalar(data, "hiding_parker_alpha")) ==
                     identity["parker_alpha"] and
                 float(_scalar(data, "hiding_wu_exponent")) ==
                     identity["wu_exponent"],
                 "Q3-H1 hiding controls differ from their additive identity")
    else:
        _require(hiding_mode == 0,
                 "Q3-C auxiliary operator chain requires hiding off")
    _require(not bool(np.any(data.get("suspended_transport_active", False))) and
             not bool(np.any(data.get("suspended_mass", 0))) and
             not bool(np.any(data.get("fraction_concentration", 0))) and
             not bool(np.any(data.get("fraction_series_concentration", 0))),
             "Q3-C total route excludes an independent suspended reservoir")
    modes = np.asarray(data.get("mode"))
    bed_modes = np.asarray(data.get("bed_boundary_mode"))
    boundary_count = int(_scalar(data, "nb"))
    _require(modes.shape == (boundary_count,) and
             bed_modes.shape == (boundary_count,),
             "Q3-C requires explicit boundary mode vectors")
    inlets = np.isin(modes, (1, 6, 7, 8, 9))
    _require(np.all(bed_modes[inlets] == 1) and
             np.all(bed_modes[~inlets] == 0),
             "Q3-C supports equilibrium-capacity inlets and stage outlets")
    for key in ("bc_bed_input", "fraction_bed_input", "series_bed_input",
                "fraction_series_bed_input"):
        _require(not np.any(data.get(key, 0)),
                 "Q3-C excludes prescribed bed feed: " + key)
    viscosity = float(_scalar(data, "capacity_kinematic_viscosity_m2_s"))
    work_depth = float(_scalar(data, "capacity_roughness_work_depth_m"))
    _require(np.isfinite(viscosity) and viscosity > 0.0 and
             np.isfinite(work_depth) and work_depth >= 0.0,
             "Q3-C viscosity/work-depth controls are invalid")
    _channel_profile(data, profile)
    return profile


def _is_scalar_public(data: Mapping[str, object]) -> bool:
    marker = data.get(PUBLIC_ADMISSION_KEY)
    return (isinstance(marker, Mapping) and
            marker.get("admission_id") in (
                SCALAR_PUBLIC_ADMISSION_ID, HIDING_PUBLIC_ADMISSION_ID))


def _validated_profile(data: Mapping[str, object]) -> CapacityRuntimeProfile:
    return (validate_scalar_capacity_prepared(data) if _is_scalar_public(data)
            else validate_law11_prepared(data))


def _profile_generation(data: Mapping[str, object]) -> str:
    if BED_SLOPE_EXECUTION_KEY in data:
        return BED_SLOPE_PROFILE_GENERATION
    if _is_prognostic_secondary_public(data):
        return PROGNOSTIC_SECONDARY_PROFILE_GENERATION
    if _is_secondary_public(data):
        return SECONDARY_PROFILE_GENERATION
    if _is_hiding_public(data):
        return HIDING_PROFILE_GENERATION
    return SCALAR_PROFILE_GENERATION if _is_scalar_public(data) else PROFILE_GENERATION


def _profile_admission(data: Mapping[str, object]) -> str:
    marker = data.get(PUBLIC_ADMISSION_KEY)
    return (marker["admission_id"] if isinstance(marker, Mapping)
            else PROTOTYPE_OVERRIDE)


def profile_identity(data: Mapping[str, object]) -> dict:
    profile = _validated_profile(data)
    channel = _channel_profile(data, profile)
    law = int(_scalar(data, "bedlaw"))
    result = dict(
        m1_profile_runtime_schema=PROFILE_RUNTIME_SCHEMA,
        m1_profile_generation=_profile_generation(data),
        capacity_profile_fingerprint=profile.fingerprint,
        capacity_profile_semantic_id=profile.semantic_id,
        capacity_parameter_catalog_id=profile.parameter_catalog_id,
        capacity_formula=_SCALAR_METHODS[law],
        capacity_formula_code=law,
        capacity_bed_route="resident_reconstructed_face_exner_v1",
        capacity_suspended_route=_suspended_route(data),
        channel_composition_schema=channel.schema,
        channel_composition_scope_id=channel.scope_id,
        channel_composition_semantic_id=channel.semantic_id,
        channel_composition_fingerprint=channel.fingerprint,
        channel_composition_bed_factor=channel.bed_factor,
        channel_composition_suspended_factor=channel.suspended_factor,
        channel_composition_epsilon=channel.epsilon,
        channel_transport_classes=[value.transport_class
                                   for value in channel.fractions],
        m1_profile_admission=_profile_admission(data),
        automatic_CPU_fallback=False,
    )
    if int(_scalar(data, "exchange_mode")) == M2_EXCHANGE_MODE:
        result.update(suspension_method_identity(data))
    if _is_hiding_public(data):
        identity = validate_hiding_foundation_execution_document(
            data.get(HIDING_EXECUTION_KEY))
        result.update(
            hiding_execution_schema=identity["schema"],
            hiding_execution_scope_id=identity["scope_id"],
            hiding_execution_semantic_id=identity["semantic_id"],
            hiding_execution_fingerprint=identity["fingerprint"],
            hiding_exposure_mode=identity["mode"],
            hiding_method_id=identity["method_id"],
            hiding_parker_alpha=identity["parker_alpha"],
            hiding_wu_exponent=identity["wu_exponent"],
            hiding_active_volume_source=identity["active_volume_source"],
            hiding_composition_order=identity["composition_order"],
        )
    secondary = validate_secondary_composition(data)
    prognostic = validate_prognostic_secondary_composition(data)
    if secondary is not None:
        result.update(_secondary_receipt(secondary))
    if prognostic is not None:
        result.update(_prognostic_secondary_receipt(prognostic))
    bed_slope = validate_bed_slope_composition(data)
    if bed_slope is not None:
        result.update(
            bed_slope_execution_schema=bed_slope["schema"],
            bed_slope_execution_scope_id=bed_slope["scope_id"],
            bed_slope_execution_semantic_id=bed_slope["semantic_id"],
            bed_slope_execution_fingerprint=bed_slope["fingerprint"],
            bed_slope_method_id=bed_slope["method_id"],
            bed_slope_runtime_mode=bed_slope["runtime_mode"],
            bed_slope_persistence_code=bed_slope["persistence_code"],
            bed_slope_parameters=bed_slope["parameters"],
            bed_slope_composition_order=bed_slope["composition_order"],
        )
    return result


def _secondary_receipt(identity: Mapping[str, object]) -> dict:
    return dict(
        secondary_execution_schema=identity["schema"],
        secondary_execution_scope_id=identity["scope_id"],
        secondary_execution_semantic_id=identity["semantic_id"],
        secondary_execution_fingerprint=identity["fingerprint"],
        secondary_method_id=identity["method_id"],
        secondary_model=identity["model"],
        secondary_base_envelope_version=identity["base_envelope_version"],
        secondary_capacity_law=identity["capacity_law"],
        secondary_rmincf=identity["rmincf"],
        secondary_chzmin=identity["chzmin"],
        secondary_espir=identity["espir"],
        secondary_betac=identity["betac"],
        secondary_curvature_discretization=identity[
            "curvature_discretization"],
        secondary_curvature_cap_policy=identity["curvature_cap_policy"],
        secondary_metric_policy=identity["metric_policy"],
        secondary_roughness_policy=identity["roughness_policy"],
        secondary_refresh_timing=identity["refresh_timing"],
        secondary_boundary_policy=identity["boundary_policy"],
        secondary_composition_order=identity["composition_order"],
        secondary_raw_bed_rotation_scope=identity["raw_bed_rotation_scope"],
        suspended_channel_rotation=identity["suspended_channel_rotation"],
        secondary_backend_selection=identity["backend_selection"],
    )


def _prognostic_secondary_receipt(identity: Mapping[str, object]) -> dict:
    return dict(
        prognostic_secondary_execution_schema=identity["schema"],
        prognostic_secondary_execution_scope_id=identity["scope_id"],
        prognostic_secondary_execution_semantic_id=identity["semantic_id"],
        prognostic_secondary_execution_fingerprint=identity["fingerprint"],
        prognostic_secondary_method_id=identity["method_id"],
        prognostic_secondary_model=identity["model"],
        prognostic_secondary_state_semantics=identity["state_semantics"],
        prognostic_secondary_integrated_ledger_units=
            identity["integrated_ledger_units"],
        prognostic_secondary_base_envelope_version=
            identity["base_envelope_version"],
        prognostic_secondary_capacity_law=identity["capacity_law"],
        prognostic_secondary_rmincf=identity["rmincf"],
        prognostic_secondary_chzmin=identity["chzmin"],
        prognostic_secondary_espir=identity["espir"],
        prognostic_secondary_betac=identity["betac"],
        prognostic_secondary_scalar_diffusivity_m2_s=
            identity["scalar_diffusivity_m2_s"],
        prognostic_secondary_source_sink_policy=
            identity["source_sink_policy"],
        prognostic_secondary_open_boundary_policy=
            identity["open_boundary_policy"],
        prognostic_secondary_bed_boundary_rotation_policy=
            identity["bed_boundary_rotation_policy"],
        prognostic_secondary_depth_pair_policy=
            identity["depth_pair_policy"],
        prognostic_secondary_target_forcing_policy=
            identity["target_forcing_policy"],
        prognostic_secondary_composition_order=identity["composition_order"],
        prognostic_secondary_checkpoint_schema=
            identity["gpu_checkpoint_schema"],
        prognostic_secondary_backend_selection=identity["backend_selection"],
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_source_files(root: Path, data=None) -> tuple[Path, ...]:
    names = [
        "braids_gpu/m1_profile_transport.py",
        "braids_gpu/m1_profile_faces.cu",
        "braids_gpu/m1_capacity_dispatch.py",
        "braids_gpu/m1_capacity_dispatch.cu",
        "braids_gpu/scalar_capacity_runtime.py",
        "braids_gpu/scalar_capacity_runtime.cu",
        "braids_gpu/channel_composition.py",
        "braids_gpu/channel_composition.cu",
        "braids_gpu/hydraulic_roughness.py",
        "braids_gpu/hydraulic_roughness.cu",
        "braids_gpu/delft_current_capacity.py",
        "braids_gpu/delft_current_capacity.cu",
        "braids_core/sediment/capacity_profile.py",
        "braids_core/sediment/capacity_parameters.py",
        "braids_core/sediment/capacity_foundation_execution.py",
        "braids_core/sediment/channel_composition.py",
        "braids_core/sediment_method_catalog.py",
    ]
    if data is not None and _is_hiding_public(data):
        names.extend((
            "braids_gpu/hiding_exposure_runtime.py",
            "braids_gpu/hiding_exposure_runtime.cu",
            "braids_core/sediment/hiding_foundation_execution.py",
        ))
    if data is not None and _is_secondary_public(data):
        names.extend((
            "braids_gpu/flow2dh_secondary_operator.py",
            "braids_gpu/flow2dh_secondary_operator.cu",
            "braids_gpu/flow2dh_secondary_live.cu",
            "braids_core/sediment/secondary_foundation_execution.py",
        ))
    if data is not None and _is_prognostic_secondary_public(data):
        names.extend((
            "braids_gpu/flow2dh_secondary_operator.py",
            "braids_gpu/flow2dh_secondary_operator.cu",
            "braids_gpu/flow2dh_secondary_live.cu",
            "braids_gpu/flow2dh_prognostic_secondary.py",
            "braids_gpu/flow2dh_prognostic_secondary.cu",
            "braids_core/sediment/prognostic_secondary_execution.py",
        ))
    if data is not None and BED_SLOPE_EXECUTION_KEY in data:
        names.extend((
            "braids_gpu/flow_bed_slope_runtime.py",
            "braids_gpu/flow_bed_slope_runtime.cu",
            "braids_gpu/flow_bed_slope_composition.py",
            "braids_core/sediment/bed_slope_execution.py",
        ))
    if data is not None and int(_scalar(data, "exchange_mode")) == M2_EXCHANGE_MODE:
        names.extend((
            "braids_gpu/law11_soursin_exchange.py",
            "braids_gpu/law11_soursin_exchange.cu",
            "braids_gpu/flow2dh_boundary_diffusion.py",
            "braids_gpu/flow2dh_boundary_diffusion.cu",
            "braids_core/sediment/flow2dh_exchange.py",
            "braids_core/sediment/flow2dh_boundary_diffusion.py",
            "braids_core/sediment/flow2dh_settling.py",
            "braids_core/sediment/suspension_profile.py",
        ))
    return tuple(root/name for name in names)


def archive_profile_sources(folder, data, *, root):
    """Create an additive authenticated generation beside the base archive."""
    folder, root = Path(folder).resolve(), Path(root).resolve()
    profile = _validated_profile(data)
    channel = _channel_profile(data, profile)
    records = {}
    for source in profile_source_files(root, data):
        _require(source.is_file() and not source.is_symlink() and
                 source.resolve().is_relative_to(root),
                 "Law-11 profile source is missing, linked, or outside the repository")
        relative = source.relative_to(root).as_posix()
        target = (folder/source.name if source.suffix == ".cu" else
                  folder/"python_snapshot"/relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[relative] = dict(sha256=_sha(source), archived=target.relative_to(folder).as_posix())
    receipt = dict(
        schema=PROFILE_RUNTIME_SCHEMA,
        generation=_profile_generation(data),
        capacity_profile=capacity_profile_document(profile),
        capacity_profile_fingerprint=profile.fingerprint,
        capacity_profile_semantic_id=profile.semantic_id,
        capacity_parameter_catalog_id=profile.parameter_catalog_id,
        channel_composition_profile=channel_composition_profile_document(channel),
        channel_composition_fingerprint=channel.fingerprint,
        channel_composition_semantic_id=channel.semantic_id,
        sources=records,
        bed_channel="resident_reconstructed_face_exner_v1",
        suspended_channel=_suspended_route(data),
        m1_profile_admission=_profile_admission(data),
        automatic_CPU_fallback=False,
        Delft3D_source_modified=False,
    )
    if int(_scalar(data, "exchange_mode")) == M2_EXCHANGE_MODE:
        suspension_profile = load_suspension_profile(data[SUSPENSION_PROFILE_KEY])
        receipt.update(
            suspension_profile=suspension_profile_document(suspension_profile),
            suspension_profile_semantic_id=suspension_profile_semantic_id(),
            suspension_profile_fingerprint=suspension_profile.fingerprint,
        )
    if _is_hiding_public(data):
        receipt[HIDING_EXECUTION_KEY] = \
            validate_hiding_foundation_execution_document(
                data[HIDING_EXECUTION_KEY])
    secondary = validate_secondary_composition(data)
    if secondary is not None:
        receipt[SECONDARY_EXECUTION_KEY] = secondary
    prognostic = validate_prognostic_secondary_composition(data)
    if prognostic is not None:
        receipt[PROGNOSTIC_SECONDARY_EXECUTION_KEY] = prognostic
    bed_slope = validate_bed_slope_composition(data)
    if bed_slope is not None:
        receipt[BED_SLOPE_EXECUTION_KEY] = bed_slope
        receipt[BED_SLOPE_ORDERED_ROUTE_KEY] = data[
            BED_SLOPE_ORDERED_ROUTE_KEY]
    path = folder/"m1_profile_sources.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_profile_sources(folder, data, root=root)
    return receipt


def verify_profile_sources(folder, data, *, root):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    profile = _validated_profile(data)
    channel = _channel_profile(data, profile)
    path = folder/"m1_profile_sources.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require(receipt.get("schema") == PROFILE_RUNTIME_SCHEMA and
             receipt.get("generation") == _profile_generation(data) and
             receipt.get("capacity_profile_fingerprint") == profile.fingerprint and
             receipt.get("capacity_profile_semantic_id") == profile.semantic_id and
             receipt.get("capacity_parameter_catalog_id") ==
             profile.parameter_catalog_id and
             receipt.get("capacity_profile") == capacity_profile_document(profile) and
             receipt.get("channel_composition_profile") ==
             channel_composition_profile_document(channel) and
             receipt.get("channel_composition_fingerprint") ==
             channel.fingerprint and
             receipt.get("channel_composition_semantic_id") ==
             channel.semantic_id and
             receipt.get("bed_channel") == "resident_reconstructed_face_exner_v1" and
             receipt.get("suspended_channel") == _suspended_route(data) and
             receipt.get("m1_profile_admission") == _profile_admission(data) and
             receipt.get("automatic_CPU_fallback") is False and
             receipt.get("Delft3D_source_modified") is False,
             "Law-11 profile source receipt differs")
    if int(_scalar(data, "exchange_mode")) == M2_EXCHANGE_MODE:
        suspension_profile = load_suspension_profile(data[SUSPENSION_PROFILE_KEY])
        _require(receipt.get("suspension_profile") ==
                 suspension_profile_document(suspension_profile) and
                 receipt.get("suspension_profile_semantic_id") ==
                 suspension_profile_semantic_id() and
                 receipt.get("suspension_profile_fingerprint") ==
                 suspension_profile.fingerprint,
                 "Law-11 suspended source receipt differs")
    if _is_hiding_public(data):
        _require(receipt.get(HIDING_EXECUTION_KEY) ==
                 validate_hiding_foundation_execution_document(
                     data[HIDING_EXECUTION_KEY]),
                  "Q3-H1 hiding source receipt differs")
    secondary = validate_secondary_composition(data)
    if secondary is not None:
        _require(receipt.get(SECONDARY_EXECUTION_KEY) == secondary,
                 "Q3-H2 secondary source receipt differs")
    prognostic = validate_prognostic_secondary_composition(data)
    if prognostic is not None:
        _require(receipt.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY) == prognostic,
                 "Q3-H3 prognostic-secondary source receipt differs")
    bed_slope = validate_bed_slope_composition(data)
    if bed_slope is not None:
        _require(receipt.get(BED_SLOPE_EXECUTION_KEY) == bed_slope and
                 receipt.get(BED_SLOPE_ORDERED_ROUTE_KEY) ==
                     data[BED_SLOPE_ORDERED_ROUTE_KEY],
                 "Q3-H4 bed-slope source receipt differs")
    expected = {path.relative_to(root).as_posix(): path
                for path in profile_source_files(root, data)}
    _require(set(receipt.get("sources", {})) == set(expected),
             "Law-11 profile source inventory differs")
    for relative, source in expected.items():
        row = receipt["sources"][relative]
        archived = folder/row["archived"]
        _require(not archived.is_symlink() and archived.is_file() and
                 row["sha256"] == _sha(source) == _sha(archived),
                 "Law-11 profile source drift: " + relative)
    return receipt


def make_law11_transport(Base):
    """Compose the common 2/8/11 dispatcher with the existing face/Exner ABI.

    The historical name is retained as a source/archive compatibility alias.
    """
    class ResidentLaw11Transport(Base):
        profile_runtime_schema = PROFILE_RUNTIME_SCHEMA

        def __init__(self, data, *args, **kwargs):
            # This call is host-only and intentionally precedes Base's CuPy
            # import and every CUDA allocation/compilation.
            public_marker = data.get(PUBLIC_ADMISSION_KEY)
            scalar_public = _is_scalar_public(data)
            hiding_public = _is_hiding_public(data)
            profile = (validate_scalar_capacity_prepared(data) if scalar_public
                       else validate_law11_prepared(data))
            secondary_identity = validate_secondary_composition(data)
            prognostic_secondary_identity = \
                validate_prognostic_secondary_composition(data)
            bed_slope_identity = validate_bed_slope_composition(data)
            capacity_law = int(_scalar(data, "bedlaw"))
            channel_profile = _channel_profile(data, profile)
            source_value = kwargs.get("source_path")
            _require(source_value is not None,
                     "Law-11 profile bridge requires an authenticated source archive")
            source = Path(source_value).resolve()
            source_folder = source.parent
            _require(source.is_file() and not source.is_symlink() and
                     all((source_folder/name).is_file() and
                         not (source_folder/name).is_symlink()
                          for name in ("m1_capacity_dispatch.cu",
                                       "scalar_capacity_runtime.cu",
                                      "hydraulic_roughness.cu",
                                      "delft_current_capacity.cu",
                                       "channel_composition.cu",
                                       "m1_profile_faces.cu",
                                      *(("flow2dh_secondary_operator.cu",
                                          "flow2dh_secondary_live.cu")
                                        if (secondary_identity is not None or
                                            prognostic_secondary_identity is not None)
                                        else ()),
                                      *(("flow2dh_prognostic_secondary.cu",)
                                        if prognostic_secondary_identity is not None
                                        else ()),
                                       *(("hiding_exposure_runtime.cu",)
                                        if hiding_public else ()),
                                      *(('flow2dh_boundary_diffusion.cu',)
                                        if _suspended_route(data) ==
                                        M2_SUSPENDED_ROUTE else ()),
                                      *(("flow_bed_slope_runtime.cu",)
                                        if bed_slope_identity is not None
                                        else ()))),
                     "Law-11 archived CUDA source inventory differs")
            single_active_reservoir = (
                int(data.get("exchange_mode", -1)) in (0, M2_EXCHANGE_MODE) and
                int(data.get("nfrac", -1)) == 1 and
                int(data.get("nbed_layers", -1)) == 1 and
                int(data.get("nsubstrate_layers", -1)) == 0)
            if (single_active_reservoir and
                    int(data.get("exchange_mode", -1)) == M2_EXCHANGE_MODE):
                # v49's ordinary one-fraction deck owns one real active-bed
                # row.  Authenticate the M2-C route before using a temporary
                # two-layer *constructor shadow* for the older generic CUDA
                # geometry allocator; no substrate row is added to state.
                from .implicit_suspension_amr import contract_from_data
                contract_from_data(data)
            shadow = dict(data, bedlaw=np.int32(2), bed_slope=np.int32(0),
                          exchange_mode=np.int32(0),
                          settling_mode=np.int32(0),
                          hiding_exposure=np.int32(0),
                          secondary_flow_model=np.int32(0),
                          secondary_flow=np.int32(0),
                          secondary_flow_transport_direction=np.int32(0),
                          mpm_friction_mode=np.int32(0))
            if prognostic_secondary_identity is not None:
                # Private link consumed only by closed-wall boundary
                # validation while the generic allocator is deliberately
                # neutralized.  It never enables a numerical operator.
                shadow['_q3h3_prognostic_constructor_shadow'] = \
                    prognostic_secondary_identity['fingerprint']
            if single_active_reservoir:
                shadow.update(
                    deck_version=np.int32(max(13, int(data["deck_version"]))),
                    nsubstrate_layers=np.int32(1),
                    active_layer_thickness=np.float64(1.0))
            super().__init__(shadow, *args, **kwargs)
            from .scalar_capacity_runtime import ResidentScalarCapacityRuntime
            from .channel_composition import ResidentChannelComposer
            cp = self.cp
            a = self.geometry

            self.profile = profile
            self.capacity_law = capacity_law
            self.scalar_public = scalar_public
            self.hiding_public = hiding_public
            self.secondary_identity = secondary_identity
            self.prognostic_secondary_identity = \
                prognostic_secondary_identity
            self.bed_slope_identity = bed_slope_identity
            self.channel_profile = channel_profile
            self.single_active_reservoir = single_active_reservoir
            if single_active_reservoir:
                self.nl = 1
                self.config["nsubstrate_layers"] = 0
                self.config["target"] = 0.0
            self.suspended_route = _suspended_route(data)
            self.suspension_identity = (
                suspension_method_identity(data)
                if self.suspended_route == M2_SUSPENDED_ROUTE else {})
            self.capacity_profile_fingerprint = profile.fingerprint
            self.profile_generation = _profile_generation(data)
            self.profile_admission = (
                public_marker["admission_id"] if isinstance(public_marker, Mapping)
                else PROTOTYPE_OVERRIDE)
            forcing = data.get("delft_forcing_binding")
            self.capacity_roughness_mode = (
                forcing["hydraulic_roughness_mode"] if forcing is not None
                else "manning")
            self.capacity_roughness_waqua = bool(
                forcing["white_colebrook_chzk2d"] if forcing is not None
                else False)
            # Base owns the already-qualified high-order reconstruction and
            # Exner transaction.  Keep its scalar shadow on law 2; the actual
            # constitutive identity is exclusively ``capacity_law`` and never
            # enters the legacy coupled-MPM override.
            self.config["bedlaw"] = 2
            self.config["bed_slope"] = 0
            self.config["hiding_exposure"] = 0
            self.config["secondary_flow_model"] = 0
            self.coupled_mpm = self.coupled_gaia = self.coupled_bedload = False
            self.coupled_kind = 0
            self.capacity_parameters = MappingProxyType({
                coefficient.keyword: cp.ascontiguousarray(cp.asarray(
                    [dict((value.keyword, value.value) for value in fraction.coefficients)[coefficient.keyword]
                     for fraction in profile.fractions], dtype=(
                         cp.int32 if isinstance(coefficient.value, int) and
                         not isinstance(coefficient.value, bool) else cp.float64)))
                for coefficient in profile.fractions[0].coefficients
            })
            self.scalar_runtime = ResidentScalarCapacityRuntime(
                capacity_law, d50=a["d50"], d90=a["d90"],
                solid_density=a["rhos"], parameters=self.capacity_parameters,
                water_density=float(self.config["rho"]),
                gravity=float(self.config["grav"]),
                roughness_mode=self.capacity_roughness_mode,
                minimum_work_depth=float(data["capacity_roughness_work_depth_m"]),
                waqua_white_colebrook=self.capacity_roughness_waqua,
                source_path=source_folder/"scalar_capacity_runtime.cu",
                dispatch_source_path=source_folder/"m1_capacity_dispatch.cu",
                roughness_source_path=source_folder/"hydraulic_roughness.cu",
                delft_source_path=source_folder/"delft_current_capacity.cu")
            self.dispatch = self.scalar_runtime.dispatch
            self.channel_composer = ResidentChannelComposer(
                source_path=source_folder/"channel_composition.cu")
            if bed_slope_identity is not None:
                from .flow_bed_slope_runtime import FlowBedSlopeCUDA
                from .flow_bed_slope_composition import (
                    ResidentFlowBedSlopeFaceComposer,
                )
                slope_operator = FlowBedSlopeCUDA(
                    source_path=source_folder/"flow_bed_slope_runtime.cu")
                self.bed_slope_composer = ResidentFlowBedSlopeFaceComposer(
                    bed_slope_identity, operator=slope_operator)
            else:
                self.bed_slope_composer = None
            if secondary_identity is not None:
                from .flow2dh_secondary_operator import (
                    ResidentClassicFlow2DHSecondary,
                )
                self.classic_secondary = ResidentClassicFlow2DHSecondary(
                    data, a, source_folder=source_folder,
                    execution_identity=secondary_identity)
            else:
                self.classic_secondary = None
            # The solver controller binds the sole v57 transaction owner
            # after the generic scalar reconstruction kernels exist.
            self.prognostic_secondary = None
            if hiding_public:
                from .hiding_exposure_runtime import (
                    HidingExposureCUDA, STATUS_DEPLETED)
                self.hiding_identity = \
                    validate_hiding_foundation_execution_document(
                        data[HIDING_EXECUTION_KEY])
                self.hiding_mode = int(data["hiding_exposure"])
                self.hiding_parker_alpha = float(data["hiding_parker_alpha"])
                self.hiding_wu_exponent = float(data["hiding_wu_exponent"])
                self.hiding_runtime = HidingExposureCUDA(
                    source_path=source_folder/"hiding_exposure_runtime.cu")
                self.hiding_depleted_status = int(STATUS_DEPLETED)
            else:
                self.hiding_identity = None
                self.hiding_mode = 0
                self.hiding_parker_alpha = 0.6
                self.hiding_wu_exponent = -0.6
                self.hiding_runtime = None
                self.hiding_depleted_status = 2
            self.channel_transport_class = cp.ascontiguousarray(cp.asarray(
                [value.transport_class for value in channel_profile.fractions],
                dtype=cp.int32))
            face_source = ((source_folder/"hydraulic_roughness.cu").read_text(
                encoding="utf-8") + "\n" +
                (source_folder/"m1_profile_faces.cu").read_text(
                    encoding="utf-8"))
            self.profile_face_module = cp.RawModule(
                code=face_source,
                options=("--std=c++11", "--fmad=false", "--prec-div=true",
                         "--prec-sqrt=true", "--ftz=false"),
                name_expressions=("m1_law11_boundary_faces",
                                  "m1_project_composed_boundary_faces",
                                  "m1_reconstruct_composed_faces"),
            )
            self.profile_face_kernel = self.profile_face_module.get_function(
                "m1_law11_boundary_faces")
            self.profile_face_projection_kernel = (
                self.profile_face_module.get_function(
                    "m1_project_composed_boundary_faces"))
            self.profile_face_reconstruction_kernel = (
                self.profile_face_module.get_function(
                    "m1_reconstruct_composed_faces"))
            self.capacity_viscosity = cp.full(
                self.nc, float(data["capacity_kinematic_viscosity_m2_s"]),
                dtype=cp.float64)
            self.capacity_work_depth = float(data["capacity_roughness_work_depth_m"])
            b = self.buffers
            b["raw_bed_capacity"] = cp.zeros(
                (2, self.ng, self.nc), dtype=cp.float64)
            b["raw_suspended_capacity"] = cp.zeros(
                (2, self.ng, self.nc), dtype=cp.float64)
            b["suspended_capacity"] = cp.zeros((2, self.ng, self.nc), dtype=cp.float64)
            b["equilibrium_suspended_capacity"] = cp.zeros(
                (self.ng, self.nc), dtype=cp.float64)
            b["signed_suspended_capacity"] = cp.zeros(
                (self.ng, self.nc), dtype=cp.float64)
            b["channel_composition_status"] = cp.zeros(
                (self.ng, self.nc), dtype=cp.int32)
            b["capacity_base_velocity"] = cp.zeros(
                (2, self.nc), dtype=cp.float64)
            b["capacity_composition_depth"] = cp.ones(
                self.nc, dtype=cp.float64)
            b["capacity_absolute_shields"] = cp.zeros((self.ng, self.nc), dtype=cp.float64)
            b["capacity_critical_shields"] = cp.zeros((self.ng, self.nc), dtype=cp.float64)
            b["capacity_stress_ratio"] = cp.zeros((self.ng, self.nc), dtype=cp.float64)
            b["capacity_formula_auxiliary"] = cp.zeros((2, self.ng, self.nc), dtype=cp.float64)
            b["capacity_status"] = cp.zeros(self.nc, dtype=cp.int32)
            b["hiding_factors"] = cp.ones(
                (self.ng, self.nc), dtype=cp.float64)
            b["hiding_mean_diameter"] = cp.zeros(
                self.nc, dtype=cp.float64)
            b["hiding_status"] = cp.zeros(self.nc, dtype=cp.int32)
            b["boundary_raw_bed_capacity"] = cp.zeros(
                (2, self.ng, self.nf), dtype=cp.float64)
            b["boundary_raw_suspended_capacity"] = cp.zeros(
                (2, self.ng, self.nf), dtype=cp.float64)
            b["boundary_bed_capacity"] = cp.zeros(
                (2, self.ng, self.nf), dtype=cp.float64)
            b["boundary_suspended_capacity"] = cp.zeros(
                (2, self.ng, self.nf), dtype=cp.float64)
            b["boundary_equilibrium_suspended_capacity"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.float64)
            b["boundary_signed_suspended_capacity"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.float64)
            b["boundary_channel_composition_status"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.int32)
            b["boundary_channel_role"] = cp.zeros(
                self.nf, dtype=cp.int32)
            b["boundary_base_velocity"] = cp.zeros(
                (2, self.nf), dtype=cp.float64)
            b["boundary_composition_depth"] = cp.ones(
                self.nf, dtype=cp.float64)
            b["boundary_absolute_shields"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.float64)
            b["boundary_stress_ratio"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.float64)
            b["bed_slope_role"] = cp.zeros(self.nf, dtype=cp.int32)
            b["bed_slope_downhill"] = cp.zeros(
                (2, self.nf), dtype=cp.float64)
            b["bed_slope_stress"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.float64)
            b["bed_slope_depth"] = cp.ones(
                (self.ng, self.nf), dtype=cp.float64)
            b["bed_slope_mean_diameter"] = cp.ones(
                (self.ng, self.nf), dtype=cp.float64)
            b["bed_slope_donor"] = cp.zeros(
                (self.ng, self.nf), dtype=cp.int32)
            self.last_capacity_result = None
            self.last_capacity_points = None
            self.last_bed_slope_faces = None

        def prepare(self, q, bed, layers, secondary_mass, active):
            cp, c, a, b = self.cp, self.config, self.geometry, self.buffers
            self._array(q, (3, self.nc)); self._array(bed, (self.nc,))
            self._array(layers, (self.nl, self.ng, self.nc))
            self._array(secondary_mass, (self.nc,))
            self._array(active, (self.nc,), cp.int32)
            secondary_status = (self.classic_secondary.refresh(q, active)
                                if self.classic_secondary is not None else None)
            surface_volume = layers[0]/a["bulk"][:, None]
            volume = cp.sum(surface_volume, axis=0)
            if self.hiding_public:
                hiding = self.hiding_runtime.evaluate(
                    cp.ascontiguousarray(surface_volume), a["d50"],
                    mode=self.hiding_mode,
                    parker_alpha=self.hiding_parker_alpha,
                    wu_exponent=self.hiding_wu_exponent,
                    reject_invalid=False,
                )
                cp.copyto(b["hiding_factors"], hiding.factors)
                cp.copyto(b["hiding_mean_diameter"], hiding.mean_diameter)
                cp.copyto(b["hiding_status"], hiding.status)
            else:
                b["hiding_factors"].fill(1.0)
                b["hiding_mean_diameter"][...] = cp.where(
                    volume > 0.0,
                    cp.sum(surface_volume*a["d50"][:, None], axis=0)/
                        cp.where(volume > 0.0, volume, 1.0),
                    0.0,
                )
                b["hiding_status"].fill(0)
            fill = (cp.where(volume > 0.0, 1.0, 0.0)
                    if self.single_active_reservoir else
                    cp.minimum(1.0, volume/np.float64(c["target"])))
            b["availability"][...] = cp.where(
                volume[None, :] > 0.0,
                surface_volume/cp.where(volume[None, :] > 0.0,
                                        volume[None, :], 1.0)*fill[None, :],
                0.0)
            present = surface_volume > 0.0
            local_d90 = cp.max(cp.where(present, a["d90"][:, None], 0.0), axis=0)
            b["mixture_d90"][...] = cp.where(
                local_d90 > 0.0, local_d90, cp.max(a["d90"]))
            wet = (q[0] > np.float64(c["dry"])) & (active != 0)
            denominator = cp.where(wet, q[0], 1.0)
            b["capacity_base_velocity"][0] = cp.where(
                wet, q[1]/denominator, 0.0)
            b["capacity_base_velocity"][1] = cp.where(
                wet, q[2]/denominator, 0.0)
            b["capacity_composition_depth"][...] = cp.where(wet, q[0], 1.0)
            points = self.scalar_runtime.cells(
                q=q, roughness_parameter=a["manning_cell"],
                kinematic_viscosity=self.capacity_viscosity, active=active,
                dry_tolerance=float(c["dry"]),
                mixture_d90=(b["mixture_d90"] if self.capacity_law == 2
                             else None),
                hiding=(b["hiding_factors"]
                        if self.hiding_public else None))
            result = points.result
            b["raw_bed_capacity"][...] = cp.where(
                wet[None, None, :], points.bed_mass, 0.0)
            b["raw_suspended_capacity"][...] = cp.where(
                wet[None, None, :], points.suspended_mass, 0.0)
            rotation_status = (self.classic_secondary.rotate_cells(
                b["raw_bed_capacity"])
                if self.classic_secondary is not None else
                self.prognostic_secondary.rotate_cells(
                    q, b["capacity_base_velocity"], result.chezy,
                    b["raw_bed_capacity"], active)
                if self.prognostic_secondary is not None else None)
            composed = self.channel_composer.compose(
                raw_bed_mass=b["raw_bed_capacity"],
                raw_suspended_mass=b["raw_suspended_capacity"],
                base_velocity=b["capacity_base_velocity"],
                depth=b["capacity_composition_depth"],
                transport_class=self.channel_transport_class,
                bed_factor=self.channel_profile.bed_factor,
                suspended_factor=self.channel_profile.suspended_factor,
                speed_epsilon=self.channel_profile.epsilon,
                reject_invalid=False,
            )
            cp.copyto(b["capacity"], composed.composed_bed_mass)
            cp.copyto(b["suspended_capacity"],
                      composed.retained_suspended_mass)
            cp.copyto(b["equilibrium_suspended_capacity"],
                      composed.equilibrium_suspended_mass_concentration)
            cp.copyto(b["signed_suspended_capacity"],
                      composed.signed_suspended_mass_rate)
            cp.copyto(b["channel_composition_status"], composed.status)
            cp.copyto(b["capacity_absolute_shields"], result.absolute_shields)
            cp.copyto(b["capacity_critical_shields"], result.critical_shields)
            cp.copyto(b["capacity_stress_ratio"], result.stress_ratio)
            cp.copyto(b["capacity_formula_auxiliary"], result.formula_auxiliary)
            cp.copyto(b["capacity_status"], points.status)
            # A genuinely depleted surface is physically neutral in a dry or
            # inactive cell.  Malformed input, a singular relation, or an
            # invalid output is never made acceptable merely by dryness.
            hiding_cell_status = cp.where(
                b["hiding_status"] == self.hiding_depleted_status,
                cp.where(wet, b["hiding_status"], cp.int32(0)),
                b["hiding_status"])
            b["cell_status"][...] = cp.maximum(
                hiding_cell_status, cp.maximum(
                    points.status,
                    cp.any(composed.status != 0, axis=0).astype(cp.int32)))
            if secondary_status is not None:
                b["cell_status"][...] = cp.maximum(
                    b["cell_status"], secondary_status)
            if rotation_status is not None:
                b["cell_status"][...] = cp.maximum(
                    b["cell_status"], rotation_status)
            b["unrotated_status"].fill(0)
            self.last_capacity_result = result
            self.last_capacity_points = points
            self._launch("st_gradients", self.nc, (
                np.int32(self.nc), np.int32(self.nf), np.int32(self.na),
                np.int32(self.ng), b["capacity"], bed, a["bed_ls_matrix"],
                a["bed_ls_delta"], a["bed_ls_weight"], a["xy"], a["mid"],
                a["owner"], a["neighbour"], a["offset"], a["faces"],
                a["signs"], active, b["gradient"], b["downslope"]))
            return b["unrotated_status"]

        def rebalance(self, layers):
            if not self.single_active_reservoir:
                return super().rebalance(layers)
            self._array(layers, (1, self.ng, self.nc))
            status = self.buffers["rebalance_status"]
            status[...] = self.cp.where(
                self.cp.all(self.cp.isfinite(layers) & (layers >= 0.0), axis=(0, 1)),
                self.cp.int32(0), self.cp.int32(1))
            return status

        def bed_elevation(self, layers, rock, out=None):
            if not self.single_active_reservoir:
                return super().bed_elevation(layers, rock, out=out)
            self._array(layers, (1, self.ng, self.nc))
            self._array(rock, (self.nc,))
            out = self.buffers["bed"] if out is None else out
            self._array(out, (self.nc,))
            out[...] = rock + self.cp.sum(
                layers[0]/self.geometry["bulk"][:, None], axis=0)
            return out

        def _compose_bed_slope_faces(self, q, bed, hydraulic):
            """Own the complete H4 reconstructed-face to Exner flux seam."""
            from .scalar_capacity_runtime import (
                FACE_INTERNAL, FACE_OWNER_OUTLET,
                FACE_TRACE_INLET, FACE_TRACE_OUTLET,
            )

            cp, c, a, b = self.cp, self.config, self.geometry, self.buffers
            owner = a["owner"]
            neighbour = a["neighbour"]
            closed = hydraulic["closed"] != 0
            open_internal = (neighbour >= 0) & ~closed
            # The frozen st_faces path contains a separate Riemann wet/dry
            # opening test.  H4 does not approximate that policy: its first
            # live slice is explicitly all-wet and rejects before Exner when
            # such a face is encountered.
            wet_pair = ((hydraulic["trace"][0] > np.float64(c["dry"])) &
                        (hydraulic["trace"][4] > np.float64(c["dry"])))
            if bool(cp.any(open_internal & ~wet_pair).item()):
                raise ValueError(
                    "Q3-H4 first live route excludes internal wet/dry faces")

            self.profile_face_reconstruction_kernel(
                ((self.nf+127)//128,), (128,), (
                    np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
                    b["capacity"], b["gradient"], a["xy"], a["mid"],
                    a["rectangle_width"], a["normal"],
                    a["face_displacement"], a["weno_cells"],
                    a["open_boundary_distance"], owner, neighbour,
                    a["offset"], a["faces"], a["signs"],
                    b["face_left"], b["face_right"]),
            )

            # A hydraulic wall can exist on an internal topology edge.  Map
            # it to the composer's explicit closed role without changing the
            # immutable geometry used by Exner adjacency.
            compose_neighbour = cp.ascontiguousarray(cp.where(
                closed, cp.int32(-1), neighbour).astype(cp.int32, copy=False))
            internal = compose_neighbour >= 0
            safe_neighbour = cp.clip(
                compose_neighbour, 0, max(self.nc-1, 0))
            role = cp.where(
                internal, cp.int32(FACE_INTERNAL),
                cp.where(closed, cp.int32(FACE_INTERNAL),
                         b["boundary_channel_role"]),
            ).astype(cp.int32, copy=False)
            cp.copyto(b["bed_slope_role"], cp.ascontiguousarray(role))

            nx, ny = a["normal"][0], a["normal"][1]
            dx = a["xy"][0, safe_neighbour]-a["xy"][0, owner]
            dy = a["xy"][1, safe_neighbour]-a["xy"][1, owner]
            normal_distance = dx*nx+dy*ny
            tolerance = np.float64(1.0e-12*max(float(c["dx"]),
                                                float(c["dy"])))
            invalid_distance = internal & (
                ~cp.isfinite(normal_distance) | (normal_distance <= tolerance))
            if bool(cp.any(invalid_distance).item()):
                raise ValueError("Q3-H4 encountered invalid face bed-slope geometry")
            safe_distance = cp.where(internal, normal_distance, 1.0)
            normal_slope = cp.where(
                internal,
                -(bed[safe_neighbour]-bed[owner])/safe_distance, 0.0)
            tangential_internal = 0.5*(
                -(b["downslope"][0, owner]+
                  b["downslope"][0, safe_neighbour])*ny+
                 (b["downslope"][1, owner]+
                  b["downslope"][1, safe_neighbour])*nx)
            tangential_boundary = (-b["downslope"][0, owner]*ny+
                                   b["downslope"][1, owner]*nx)
            tangential = cp.where(
                internal, tangential_internal, tangential_boundary)
            b["bed_slope_downhill"][0] = normal_slope*nx-tangential*ny
            b["bed_slope_downhill"][1] = normal_slope*ny+tangential*nx

            trace_role = ((role == FACE_TRACE_INLET) |
                          (role == FACE_TRACE_OUTLET))
            owner_role = role == FACE_OWNER_OUTLET
            mode = int(self.bed_slope_identity["runtime_mode"])
            cell_stress = (b["capacity_stress_ratio"] if mode == 2 else
                           b["capacity_absolute_shields"])
            boundary_stress = (b["boundary_stress_ratio"] if mode == 2 else
                               b["boundary_absolute_shields"])
            stress = cp.where(
                internal[None, :],
                0.5*(cell_stress[:, owner]+
                     cell_stress[:, safe_neighbour]),
                cell_stress[:, owner])
            stress = cp.where(trace_role[None, :], boundary_stress, stress)
            cp.copyto(b["bed_slope_stress"], cp.ascontiguousarray(stress))

            depth = cp.where(
                internal,
                0.5*(q[0, owner]+q[0, safe_neighbour]), q[0, owner])
            depth = cp.where(trace_role, hydraulic["trace"][0], depth)
            b["bed_slope_depth"][...] = depth[None, :]
            owner_mean = b["hiding_mean_diameter"][owner]
            neighbour_mean = b["hiding_mean_diameter"][safe_neighbour]
            # FLOW's internal Dm is geometric; boundaries own only the cell
            # surface composition and therefore repeat its value.
            mean = cp.where(
                internal,
                cp.sqrt(cp.maximum(0.0, owner_mean*neighbour_mean)),
                owner_mean)
            b["bed_slope_mean_diameter"][...] = mean[None, :]

            owner_bed = cp.ascontiguousarray(b["capacity"][:, :, owner])
            trace_bed = cp.ascontiguousarray(b["boundary_bed_capacity"])
            # Preserve role-3 diagnostics even though the composer consumes
            # the owner vector directly and never composes it twice.
            b["boundary_bed_capacity"][...] = cp.where(
                owner_role[None, None, :], owner_bed,
                b["boundary_bed_capacity"])
            slope = self.bed_slope_composer.compose(
                left_bed=b["face_left"], right_bed=b["face_right"],
                owner_bed=owner_bed, trace_bed=trace_bed,
                owner=owner, neighbour=compose_neighbour,
                role=b["bed_slope_role"], normal=a["normal"],
                availability=b["availability"],
                downhill=b["bed_slope_downhill"],
                stress=b["bed_slope_stress"], depth=b["bed_slope_depth"],
                d50=a["d50"],
                mean_diameter=b["bed_slope_mean_diameter"],
                hiding=b["hiding_factors"], reject_invalid=False)
            cp.copyto(b["raw"], slope.flux)
            cp.copyto(b["bed_slope_donor"], slope.donor)
            b["face_status"][...] = cp.maximum(
                b["face_status"], slope.status)
            self.last_bed_slope_faces = slope

        def fluxes(self, q, bed, layers, secondary_mass, hydraulic, *,
                   prepared=False):
            """Reconstruct internal faces, then recompute open traces with law 11."""
            cp, c, a, b = self.cp, self.config, self.geometry, self.buffers
            if self.bed_slope_composer is None:
                result = super().fluxes(
                    q, bed, layers, secondary_mass, hydraulic,
                    prepared=prepared)
            else:
                _require(not prepared,
                         "Q3-H4 slope owns one uncoupled capacity-to-Exner pass")
                self._array(q, (3, self.nc)); self._array(bed, (self.nc,))
                self._array(layers, (self.nl, self.ng, self.nc))
                self._array(secondary_mass, (self.nc,))
                for name, shape, dtype in (
                    ("trace", (10, self.nf), cp.float64),
                    ("raw_fl", (3, self.nf), cp.float64),
                    ("closed", (self.nf,), cp.int32),
                    ("active", (self.nc,), cp.int32),
                ):
                    self._array(hydraulic[name], shape, dtype)
                secondary_status = self.prepare(
                    q, bed, layers, secondary_mass, hydraulic["active"])
                b["face_status"].fill(0)
                result = {
                    "raw": b["raw"], "cfl": b["cfl"],
                    "status": (b["cell_status"], b["face_status"],
                               secondary_status),
                }
            b["boundary_bed_capacity"].fill(0)
            b["boundary_suspended_capacity"].fill(0)
            b["boundary_raw_bed_capacity"].fill(0)
            b["boundary_raw_suspended_capacity"].fill(0)
            b["boundary_channel_role"].fill(0)
            b["boundary_absolute_shields"].fill(0)
            b["boundary_stress_ratio"].fill(0)
            parameters = self.capacity_parameters
            if self.scalar_public:
                owner, neighbour, bid = a["owner"], a["neighbour"], a["bid"]
                valid_bid = (bid >= 0) & (bid < self.nb)
                safe_bid = cp.clip(bid, 0, max(self.nb-1, 0))
                safe_owner = cp.clip(owner, 0, max(self.nc-1, 0))
                boundary = (neighbour < 0) & valid_bid & (hydraulic["closed"] == 0)
                # A genuinely closed selected mesh has no logical boundary
                # rows.  ``safe_bid`` is still a face-sized vector for the
                # branch-free role calculation, but must not index an empty
                # table.  Zero is the inert role/mode for every such face.
                mode = (cp.zeros(self.nf, dtype=cp.int32) if self.nb == 0
                        else a["mode"][safe_bid])
                bed_mode = (cp.zeros(self.nf, dtype=cp.int32) if self.nb == 0
                            else a["bed_boundary_mode"][safe_bid])
                # ``raw_fl`` is the three-equation hydraulic face-flux
                # matrix; boundary ownership is determined by its water-mass
                # component only, matching the CUDA projection kernels.
                water = hydraulic["raw_fl"][0]
                inlet = cp.isin(mode, cp.asarray((1, 6, 7, 8, 9), dtype=cp.int32))
                equilibrium = boundary & inlet & (water < 0.0) & (bed_mode == 1)
                h = q[0, safe_owner]
                normal_velocity_mass = (q[1, safe_owner]*a["normal"][0] +
                                        q[2, safe_owner]*a["normal"][1])
                subcritical = ((h > np.float64(c["dry"])) &
                    (cp.abs(normal_velocity_mass) <
                     h*cp.sqrt(np.float64(c["grav"])*h)))
                stage = boundary & (mode == 2) & (water > 0.0) & subcritical
                owner_outlet = (boundary & cp.isin(
                    mode, cp.asarray((2, 4), dtype=cp.int32)) &
                    (water > 0.0) & ~stage)
                role = cp.where(equilibrium, 1,
                       cp.where(stage, 2, cp.where(owner_outlet, 3, 0))).astype(
                           cp.int32)
                cp.copyto(b["boundary_channel_role"], role)
                trace_points = self.scalar_runtime.traces(
                    # The hydraulic boundary workspace also carries seven
                    # auxiliary rows.  The scalar constitutive ABI consumes
                    # only conserved (h, hu, hv), exactly as the native and
                    # CUDA boundary kernels do.
                    trace=hydraulic["trace"][:3], owner=owner, role=role,
                    roughness_parameter=a["manning_cell"],
                    kinematic_viscosity=self.capacity_viscosity,
                    dry_tolerance=float(c["dry"]),
                    mixture_d90=(b["mixture_d90"]
                                 if self.capacity_law == 2 else None),
                    hiding=(b["hiding_factors"]
                            if self.hiding_public else None))
                cp.copyto(b["boundary_raw_bed_capacity"],
                          trace_points.bed_mass)
                cp.copyto(b["boundary_raw_suspended_capacity"],
                          trace_points.suspended_mass)
                cp.copyto(b["boundary_absolute_shields"],
                          trace_points.result.absolute_shields)
                cp.copyto(b["boundary_stress_ratio"],
                          trace_points.result.stress_ratio)
                hiding_trace_status = cp.where(
                    (role == 1) | (role == 2),
                    b["hiding_status"][safe_owner], cp.int32(0))
                b["face_status"][...] = cp.maximum(
                    b["face_status"], cp.maximum(
                        trace_points.status, hiding_trace_status))
            else:
                self.profile_face_kernel(
                    ((self.nf+127)//128,), (128,), (
                        np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
                        np.int32(self.nb), hydraulic["trace"], q,
                        hydraulic["raw_fl"], hydraulic["closed"], a["owner"],
                        a["neighbour"], a["bid"], a["mode"],
                        a["bed_boundary_mode"], a["normal"],
                        a["manning_cell"], a["d50"], a["rhos"], a["min_depth"],
                        np.float64(c["rho"]),
                        np.float64(c["grav"]), np.float64(c["dry"]),
                        np.float64(self.capacity_work_depth),
                        np.int32({"chezy": 1, "manning": 2,
                                  "white_colebrook": 3}[
                                      self.capacity_roughness_mode]),
                        np.int32(self.capacity_roughness_waqua),
                        parameters["ACal"], parameters["PowerB"],
                        parameters["PowerC"], parameters["RipFac"],
                        parameters["ThetaC"], parameters["SusACal"],
                        parameters["SusPowerB"], parameters["SusPowerC"],
                        parameters["SusRipFac"], parameters["SusThetaC"],
                        b["boundary_raw_bed_capacity"],
                        b["boundary_raw_suspended_capacity"],
                        b["boundary_channel_role"],
                        b["face_status"]),
                )
            if self.classic_secondary is not None:
                secondary_face_status = self.classic_secondary.rotate_boundaries(
                    hydraulic["trace"][:3], a["owner"],
                    b["boundary_channel_role"],
                    b["boundary_raw_bed_capacity"],
                    b["raw_bed_capacity"])
                b["face_status"][...] = cp.maximum(
                    b["face_status"], secondary_face_status)
            elif self.prognostic_secondary is not None:
                secondary_face_status = \
                    self.prognostic_secondary.rotate_boundaries(
                        hydraulic["trace"][:3], a["owner"],
                        b["boundary_channel_role"],
                        b["boundary_raw_bed_capacity"],
                        b["raw_bed_capacity"])
                b["face_status"][...] = cp.maximum(
                    b["face_status"], secondary_face_status)
            trace_depth = hydraulic["trace"][0]
            trace_wet = trace_depth > np.float64(c["dry"])
            safe_trace_depth = cp.where(trace_wet, trace_depth, 1.0)
            b["boundary_base_velocity"][0] = cp.where(
                trace_wet, hydraulic["trace"][1]/safe_trace_depth, 0.0)
            b["boundary_base_velocity"][1] = cp.where(
                trace_wet, hydraulic["trace"][2]/safe_trace_depth, 0.0)
            b["boundary_composition_depth"][...] = cp.where(
                trace_wet, trace_depth, 1.0)
            composed = self.channel_composer.compose(
                raw_bed_mass=b["boundary_raw_bed_capacity"],
                raw_suspended_mass=b["boundary_raw_suspended_capacity"],
                base_velocity=b["boundary_base_velocity"],
                depth=b["boundary_composition_depth"],
                transport_class=self.channel_transport_class,
                bed_factor=self.channel_profile.bed_factor,
                suspended_factor=self.channel_profile.suspended_factor,
                speed_epsilon=self.channel_profile.epsilon,
                reject_invalid=False,
            )
            cp.copyto(b["boundary_bed_capacity"],
                      composed.composed_bed_mass)
            cp.copyto(b["boundary_suspended_capacity"],
                      composed.retained_suspended_mass)
            cp.copyto(b["boundary_equilibrium_suspended_capacity"],
                      composed.equilibrium_suspended_mass_concentration)
            cp.copyto(b["boundary_signed_suspended_capacity"],
                      composed.signed_suspended_mass_rate)
            cp.copyto(b["boundary_channel_composition_status"],
                      composed.status)
            b["face_status"][...] = cp.maximum(
                b["face_status"],
                cp.any(composed.status != 0, axis=0).astype(cp.int32))
            if self.bed_slope_composer is None:
                self.profile_face_projection_kernel(
                    ((self.nf+127)//128,), (128,), (
                        np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
                        np.int32(self.nb), hydraulic["raw_fl"],
                        hydraulic["closed"], a["owner"], a["neighbour"],
                        a["bid"], a["mode"], a["bed_boundary_mode"],
                        a["normal"], b["availability"],
                        b["boundary_channel_role"],
                        b["raw_bed_capacity"],
                        b["raw_suspended_capacity"], b["capacity"],
                        b["suspended_capacity"],
                        b["signed_suspended_capacity"],
                        b["equilibrium_suspended_capacity"],
                        b["channel_composition_status"],
                        b["boundary_raw_bed_capacity"],
                        b["boundary_raw_suspended_capacity"],
                        b["boundary_bed_capacity"],
                        b["boundary_suspended_capacity"],
                        b["boundary_signed_suspended_capacity"],
                        b["boundary_equilibrium_suspended_capacity"],
                        b["boundary_channel_composition_status"], b["raw"],
                        b["face_status"]),
                )
            else:
                self._compose_bed_slope_faces(q, bed, hydraulic)
            result["boundary_bed_capacity"] = b["boundary_bed_capacity"]
            result["boundary_suspended_capacity"] = b[
                "boundary_suspended_capacity"]
            result["boundary_formula_code"] = self.capacity_law
            result["boundary_trace_recomputed"] = True
            result["boundary_channel_role"] = b["boundary_channel_role"]
            return result

        def capacity_diagnostic(self):
            """Return resident stable buffers; suspended capacity is not discarded."""
            b = self.buffers
            return dict(
                schema=PROFILE_RUNTIME_SCHEMA,
                fingerprint=self.capacity_profile_fingerprint,
                channel_composition_fingerprint=self.channel_profile.fingerprint,
                channel_composition_semantic_id=self.channel_profile.semantic_id,
                transport_class=self.channel_transport_class,
                raw_bed_mass=b["raw_bed_capacity"],
                raw_suspended_mass=b["raw_suspended_capacity"],
                bed_mass=b["capacity"],
                pre_face_composed_bed_mass=b["capacity"],
                suspended_mass=b["suspended_capacity"],
                signed_suspended_mass_rate=b["signed_suspended_capacity"],
                equilibrium_suspended_mass_concentration=
                    b["equilibrium_suspended_capacity"],
                absolute_shields=b["capacity_absolute_shields"],
                critical_shields=b["capacity_critical_shields"],
                stress_ratio=b["capacity_stress_ratio"],
                formula_auxiliary=b["capacity_formula_auxiliary"],
                availability=b["availability"],
                hiding_factors=b["hiding_factors"],
                hiding_mean_diameter=b["hiding_mean_diameter"],
                hiding_status=b["hiding_status"],
                status=b["capacity_status"],
                channel_composition_status=b[
                    "channel_composition_status"],
                boundary_bed_mass=b["boundary_bed_capacity"],
                boundary_suspended_mass=b["boundary_suspended_capacity"],
                boundary_raw_bed_mass=b["boundary_raw_bed_capacity"],
                boundary_raw_suspended_mass=
                    b["boundary_raw_suspended_capacity"],
                boundary_signed_suspended_mass_rate=
                    b["boundary_signed_suspended_capacity"],
                boundary_equilibrium_suspended_mass_concentration=
                    b["boundary_equilibrium_suspended_capacity"],
                boundary_channel_composition_status=b[
                    "boundary_channel_composition_status"],
                boundary_channel_role=b["boundary_channel_role"],
                bed_slope_face_flux=b["raw"],
                bed_slope_face_role=b["bed_slope_role"],
                bed_slope_face_donor=b["bed_slope_donor"],
                bed_slope_execution_fingerprint=(
                    None if self.bed_slope_identity is None else
                    self.bed_slope_identity["fingerprint"]),
                boundary_formula_code=self.capacity_law,
                boundary_trace_recomputed=True,
                suspended_route=self.suspended_route,
                classic_secondary=(None if self.classic_secondary is None else
                                   self.classic_secondary.diagnostic()),
                prognostic_secondary=(
                    None if self.prognostic_secondary is None else
                    self.prognostic_secondary.diagnostic()),
            )

        def profile_receipt(self):
            result = dict(
                m1_profile_runtime_schema=PROFILE_RUNTIME_SCHEMA,
                # Preserve the historical law-11 label only for the historical
                # profile route.  Public Q3-C laws 2/8/11 share the scalar
                # dispatcher generation authenticated by their source archive.
                m1_profile_generation=self.profile_generation,
                capacity_profile_fingerprint=self.capacity_profile_fingerprint,
                capacity_profile_semantic_id=self.profile.semantic_id,
                capacity_parameter_catalog_id=self.profile.parameter_catalog_id,
                capacity_formula=_SCALAR_METHODS[self.capacity_law],
                capacity_formula_code=self.capacity_law,
                capacity_bed_route="resident_reconstructed_face_exner_v1",
                capacity_suspended_route=self.suspended_route,
                channel_composition_schema=self.channel_profile.schema,
                channel_composition_scope_id=self.channel_profile.scope_id,
                channel_composition_semantic_id=self.channel_profile.semantic_id,
                channel_composition_fingerprint=self.channel_profile.fingerprint,
                channel_composition_bed_factor=self.channel_profile.bed_factor,
                channel_composition_suspended_factor=
                    self.channel_profile.suspended_factor,
                channel_composition_epsilon=self.channel_profile.epsilon,
                channel_transport_classes=[
                    value.transport_class
                    for value in self.channel_profile.fractions],
                m1_profile_admission=self.profile_admission,
                automatic_CPU_fallback=False,
            )
            if self.hiding_public:
                identity = self.hiding_identity
                result.update(
                    hiding_execution_schema=identity["schema"],
                    hiding_execution_scope_id=identity["scope_id"],
                    hiding_execution_semantic_id=identity["semantic_id"],
                    hiding_execution_fingerprint=identity["fingerprint"],
                    hiding_exposure_mode=identity["mode"],
                    hiding_method_id=identity["method_id"],
                    hiding_parker_alpha=identity["parker_alpha"],
                    hiding_wu_exponent=identity["wu_exponent"],
                    hiding_active_volume_source=identity[
                        "active_volume_source"],
                )
            if self.secondary_identity is not None:
                result.update(_secondary_receipt(self.secondary_identity))
            if self.prognostic_secondary_identity is not None:
                result.update(_prognostic_secondary_receipt(
                    self.prognostic_secondary_identity))
            if self.bed_slope_identity is not None:
                identity = self.bed_slope_identity
                result.update(
                    bed_slope_execution_schema=identity["schema"],
                    bed_slope_execution_scope_id=identity["scope_id"],
                    bed_slope_execution_semantic_id=identity["semantic_id"],
                    bed_slope_execution_fingerprint=identity["fingerprint"],
                    bed_slope_method_id=identity["method_id"],
                    bed_slope_runtime_mode=identity["runtime_mode"],
                    bed_slope_persistence_code=identity["persistence_code"],
                    bed_slope_parameters=identity["parameters"],
                    bed_slope_composition_order=identity[
                        "composition_order"],
                    bed_slope_automatic_CPU_fallback=False,
                )
            if self.suspended_route == M2_SUSPENDED_ROUTE:
                result.update(self.suspension_identity)
            return result

    ResidentLaw11Transport.__name__ = "ResidentLaw11Transport"
    return ResidentLaw11Transport


__all__ = [
    "PROFILE_RUNTIME_SCHEMA", "PROFILE_GENERATION", "SCALAR_PROFILE_GENERATION",
    "HIDING_PROFILE_GENERATION", "SECONDARY_PROFILE_GENERATION",
    "PROGNOSTIC_SECONDARY_PROFILE_GENERATION",
    "BED_SLOPE_PROFILE_GENERATION",
    "PROTOTYPE_OVERRIDE",
    "PUBLIC_ADMISSION_KEY", "PUBLIC_ADMISSION_SCHEMA", "PUBLIC_ADMISSION_ID",
    "SCALAR_PUBLIC_ADMISSION_ID", "HIDING_PUBLIC_ADMISSION_ID",
    "SECONDARY_EXECUTION_KEY", "SECONDARY_ORDERED_ROUTE_KEY",
    "PROGNOSTIC_SECONDARY_EXECUTION_KEY",
    "PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY",
    "BED_SLOPE_EXECUTION_KEY", "BED_SLOPE_ORDERED_ROUTE_KEY",
    "public_scalar_admission_document", "public_hiding_admission_document",
    "EXECUTION_PROFILE_CONTRACT_KEY", "public_law11_admission_document",
    "bind_public_law11_runtime", "bind_public_scalar_runtime",
    "bind_public_hiding_runtime", "bind_public_bed_slope_runtime",
    "SUSPENDED_ROUTE", "M2_SUSPENDED_ROUTE", "ROUGHNESS_METHOD",
    "ROUGHNESS_METHODS", "PROFILE_KEY", "CHANNEL_PROFILE_KEY",
    "CHANNEL_FINGERPRINT_KEY",
    "validate_law11_prepared", "validate_scalar_capacity_prepared",
    "validate_secondary_composition",
    "validate_prognostic_secondary_composition",
    "validate_bed_slope_composition", "bed_slope_upstream_fingerprints",
    "profile_identity", "profile_source_files",
    "archive_profile_sources", "verify_profile_sources", "make_law11_transport",
]
