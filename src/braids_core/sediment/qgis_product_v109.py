"""Q10 product contract generated from the closed sediment catalogues.

This module is intentionally free of QGIS and CUDA imports.  It is the shared
pre-allocation resolver used by the desktop plugin, the external worker and
release tests.  Constitutive provenance, product-route admission and physical
compatibility are kept separate: a component kernel is not a public product
route merely because it exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from braids_core.sediment_method_catalog import (
    ADMITTED_ROUTE_STATUSES,
    CATALOG_SCOPE_ID,
    DELFT_SOURCE_COMMIT,
    METHOD_BY_ID,
    METHODS,
    SCOPE_BRAIDS_EXTENSION,
    SCOPE_BRAIDS_LEGACY,
    SCOPE_EXCLUDED,
    SCOPE_REFERENCE_DISABLED,
    SedimentCatalogError,
    method,
    resolve_profile,
    semantic_catalog_id,
)
from .cross_physics_v108 import (
    INTEGRATOR,
    cross_physics_catalogue_v108,
    output_registry_v108,
    resolve_output_selection_v108,
)


SCHEMA = "braids.qgis_product.v109"
CATALOGUE_SCHEMA = "braids.qgis_product_catalogue.v109"
PRODUCT_ID = "braids.natural_river_2dh.v1"
CPU_BACKEND = "cpu"
GPU_BACKEND = "implicit_fwave_midpoint_gpu"
BACKENDS = (CPU_BACKEND, GPU_BACKEND)
CPU_INTEGRATOR = "native_qualified_auto"
PRODUCT_SIDECAR = "qgis_product_v109.json"

# M19 admits only the exact recursive layer/bedform/bank bundle qualified by
# v115--v117.  The semantic catalogue deliberately retains the older
# component evidence labels; product connectivity is a separate concern.
# Adding these rows here does not make any other bedform/layer combination
# public and does not alter saved projects which do not select the bundle.
_RECURSIVE_BANK_CONNECTED = frozenset((
    "bed.layered",
    "transport_layer.constant",
    "bedform.enabled",
    "bedform_height.fredsoe_mpm",
    "bedform_length.van_rijn_1984",
    "bedform_roughness.van_rijn_2007",
    "bedform_relaxation.equilibrium",
))
RECURSIVE_BANK_WORKER_VARIANT = "recursive_bank_amr_v118"

ERROR_CATEGORIES = frozenset((
    "input_contract",
    "physical_incompatibility",
    "scope_exclusion",
    "backend_route_unimplemented",
    "validation_restriction",
    "identity_mismatch",
))

# Existing connected product route.  Later Q10 revisions can add an identity
# here only together with a real prepared-input/worker consumer.  Entries not
# listed remain catalogued but fail closed as component-only evidence.
_CONNECTED_METHODS = {
    CPU_BACKEND: frozenset(
        row.method_id for row in METHODS
        if row.cpu_route in ADMITTED_ROUTE_STATUSES
    ),
    GPU_BACKEND: frozenset(
        row.method_id for row in METHODS
        if row.implicit_gpu_route in ADMITTED_ROUTE_STATUSES
    ),
}


def _recursive_bank_bundle(method_ids) -> bool:
    selected = frozenset(method_ids)
    return (_RECURSIVE_BANK_CONNECTED <= selected and
            "redistribution.braids_toe_collapse" in selected and
            "redistribution.braids_conservative_wet_dry_avalanche" in selected)


def _connected_methods(backend: str, method_ids) -> frozenset[str]:
    connected = _CONNECTED_METHODS[backend]
    return (connected | _RECURSIVE_BANK_CONNECTED
            if _recursive_bank_bundle(method_ids) else connected)

_ROUGHNESS = {
    ("manning", False): "roughness.manning",
    ("chezy", False): "roughness.chezy",
    ("white_colebrook", False): "roughness.white_colebrook",
    ("white_colebrook", True): "roughness.white_colebrook_waqua",
}
_SHEAR = {
    "total": "shear.total_roughness",
    "soulsby_clarke_skin": "shear.soulsby_clarke_skin",
}
_CAPACITY_BY_CODE = {
    0: "capacity.off",
    1: "capacity.van_rijn_1993",
    2: "capacity.mpm_specified_cf_legacy",
    3: "capacity.ashida_michiue_generalized",
    4: "capacity.soulsby",
    5: "capacity.wilcock_crowe",
    6: "capacity.wilcock_crowe_modified",
    7: "capacity.gaeuman",
    8: "capacity.engelund_hansen_constant_chezy_legacy",
    9: "capacity.gaia_research",
    10: "capacity.van_rijn_2007",
    11: "capacity.general",
    12: "capacity.van_rijn_1984_modified",
    13: "capacity.soulsby_van_rijn",
    14: "capacity.wang_fredsoe",
}
_SLOPE = {
    0: "bed_slope.off",
    2: "bed_slope.braids_legacy_bagnold",
    3: "bed_slope.delft_koch_flokstra",
    4: "bed_slope.gaia_talmon",
    5: "bed_slope.delft_bagnold_ikeda",
    6: "bed_slope.delft_parker_andrews",
}
_HIDING = {
    0: "hiding.none",
    2: "hiding.egiazaroff",
    3: "hiding.ashida_michiue",
    4: "hiding.parker_klm",
    5: "hiding.wu_wang_jia",
}
_SUSPENSION = {
    0: "suspension.off",
    1: "suspension.settling_only",
    2: "suspension.van_rijn_exchange",
    3: "suspension.cohesive_mixed",
}


class QgisProductError(ValueError):
    """Stable product-preflight error shown before mesh/array allocation."""

    def __init__(self, message: str, *, category: str):
        if category not in ERROR_CATEGORIES:
            raise ValueError("Unknown Q10 error category")
        super().__init__(message)
        self.category = category


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def canonical_backend(value: str) -> str:
    """Map the QGIS token without ever choosing a backend automatically."""
    if value == "gpu":
        return GPU_BACKEND
    if value in BACKENDS:
        return value
    raise QgisProductError(
        "Choose the CPU or implicit_fwave_midpoint GPU backend explicitly",
        category="input_contract",
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QgisProductError(f"{label} must be a mapping",
                               category="input_contract")
    return value


def _integer(value: object, label: str, choices: Mapping[int, str]) -> int:
    if type(value) is not int or value not in choices:
        raise QgisProductError(
            f"{label} must be one of {sorted(choices)}",
            category="input_contract",
        )
    return value


def _walk_for_fildad(value: object, path: str = "project") -> str | None:
    """Return the first managed-sediment key before any numerical allocation."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.casefold() in {
                    "fildad", "dredging", "dumping", "sand_mining",
                    "sandmining", "nourishment"} and child not in (
                        None, False, "", (), [], {}):
                return child_path
            found = _walk_for_fildad(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found = _walk_for_fildad(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def reject_managed_sediment_v109(project: Mapping[str, Any]) -> None:
    found = _walk_for_fildad(project)
    if found is not None:
        raise QgisProductError(
            f"{found} requests managed Fildad dredging/dumping/mining/nourishment; "
            "that active Delft3D4 capability is outside the natural-river product",
            category="scope_exclusion",
        )


@dataclass(frozen=True)
class ProductResolutionV109:
    schema: str
    product_id: str
    backend: str
    integrator: str
    selected_method_ids: tuple[str, ...]
    automatic_method_ids: tuple[str, ...]
    explanations: tuple[str, ...]
    delft3d4_method_ids: tuple[str, ...]
    braids_legacy_method_ids: tuple[str, ...]
    braids_extension_method_ids: tuple[str, ...]
    semantic_catalog_id: str
    cross_physics_catalogue_id: str
    output_registry_id: str
    output_selection: Mapping[str, Any]
    worker_requirements: Mapping[str, Any]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
                "selected_method_ids", "automatic_method_ids", "explanations",
                "delft3d4_method_ids", "braids_legacy_method_ids",
                "braids_extension_method_ids"):
            result[key] = list(result[key])
        return result


def _project_methods(project: Mapping[str, Any]) -> tuple[set[str], set[str], list[str]]:
    numerics = _mapping(project.get("numerics"), "project numerics")
    roughness = _mapping(project.get("roughness", {}), "project roughness")
    shear = _mapping(project.get("sediment_shear", {"mode": "total"}),
                     "sediment shear")
    internal = _mapping(project.get("internal_conditions", {}),
                        "internal conditions")
    initial = _mapping(project.get("initial", {}), "initial conditions")

    sediment_enabled = numerics.get("sediment_enabled", True)
    if type(sediment_enabled) is not bool:
        raise QgisProductError("sediment_enabled must be boolean",
                               category="input_contract")
    bedload = _integer(numerics.get("bedload", 0) if sediment_enabled else 0,
                       "bed-load law", _CAPACITY_BY_CODE)
    exchange = _integer(numerics.get("exchange", 0) if sediment_enabled else 0,
                        "suspended route", _SUSPENSION)
    slope = _integer(numerics.get("bed_slope", 0) if sediment_enabled else 0,
                     "bed-slope method", _SLOPE)
    hiding = _integer(numerics.get("hiding_exposure", 0) if sediment_enabled else 0,
                      "hiding/exposure method", _HIDING)

    mode = roughness.get("mode", "manning")
    chzk2d = roughness.get("white_colebrook_chzk2d", False)
    if type(chzk2d) is not bool or (mode, chzk2d) not in _ROUGHNESS:
        raise QgisProductError("Invalid hydraulic roughness selection",
                               category="input_contract")
    shear_mode = shear.get("mode", "total") if sediment_enabled else "total"
    if shear_mode not in _SHEAR:
        raise QgisProductError("Invalid sediment shear selection",
                               category="input_contract")

    selected = {
        _ROUGHNESS[(mode, chzk2d)], _SHEAR[shear_mode],
        _CAPACITY_BY_CODE[bedload], _SLOPE[slope], _HIDING[hiding],
        _SUSPENSION[exchange],
    }
    automatic = set()
    explanations = []

    if bedload == 0:
        if exchange != 0 or slope != 0 or hiding != 0:
            raise QgisProductError(
                "Transport OFF requires suspension, slope and hiding OFF",
                category="physical_incompatibility",
            )
        selected.add("transport_route.off")
        automatic.add("transport_route.off")
        explanations.append("Transport OFF automatically selects the zero-transport route.")
    elif exchange == 0:
        selected.add("transport_route.total")
        automatic.add("transport_route.total")
        explanations.append("No independent suspension automatically selects the algebraic total-load route.")
    else:
        selected.add("transport_route.combine")
        automatic.add("transport_route.combine")
        explanations.append("Active suspension automatically selects the bed-plus-water-column route.")

    secondary = _mapping(internal.get("secondary_flow", {}), "secondary flow")
    if not secondary.get("enabled", False):
        secondary_id = "secondary.off"
    elif secondary.get("model", "equilibrium") == "equilibrium":
        secondary_id = "secondary.equilibrium"
    elif secondary.get("model") == "prognostic":
        provenance = secondary.get("method_provenance")
        secondary_id = ("secondary.prognostic" if provenance is not None
                        else "secondary.prognostic")
    else:
        raise QgisProductError("Invalid secondary-flow method",
                               category="input_contract")
    selected.add(secondary_id)

    bedform = _mapping(internal.get("bedform_roughness", {}), "bedform roughness")
    if bedform.get("enabled", False):
        selected.update((
            "bedform.enabled", "bedform_height.fredsoe_mpm",
            "bedform_length.van_rijn_1984",
            "bedform_roughness.van_rijn_2007",
            "bedform_relaxation.equilibrium",
        ))
        automatic.update(selected.intersection({
            "bedform_height.fredsoe_mpm", "bedform_length.van_rijn_1984",
            "bedform_roughness.van_rijn_2007",
            "bedform_relaxation.equilibrium",
        }))
        explanations.append(
            "The QGIS bedform switch resolves the qualified Fredsoe-MPM height, "
            "Van Rijn length/roughness and equilibrium relaxation bundle."
        )
    else:
        selected.add("bedform.off")

    fractions = project.get("sediment_fractions")
    active_layer = float(initial.get("active_layer_thickness_m", 0.0))
    if fractions is not None or active_layer > 0:
        selected.add("bed.layered")
        layer_mode = initial.get("active_layer_mode", "constant")
        if layer_mode == "constant":
            selected.add("transport_layer.constant")
        elif layer_mode == "water_depth":
            selected.add("transport_layer.water_depth")
        else:
            raise QgisProductError("Invalid active-layer rule",
                                   category="input_contract")
        automatic.update(("bed.layered", next(value for value in selected
                           if value.startswith("transport_layer."))))
        explanations.append("Explicit fractions/layer thickness automatically activate the layered-bed owner.")

    redistributions = []
    general = _mapping(internal.get("general_avalanche", {}), "general avalanche")
    bank = _mapping(internal.get("bank_failure", {}), "bank failure")
    legacy_bank = _mapping(internal.get("bank_morphology", {}), "legacy bank morphology")
    if general.get("enabled", False):
        redistributions.append("redistribution.braids_conservative_wet_dry_avalanche")
    if bank.get("enabled", False):
        redistributions.append("redistribution.braids_toe_collapse")
    if legacy_bank.get("enabled", False):
        redistributions.append("redistribution.dry_bank")
    if not redistributions:
        redistributions.append("redistribution.off")
    selected.update(redistributions)

    # Preserve explicit capacity identities that distinguish historical BRAIDS
    # closure parameters from the locked FLOW closure.
    recorded = numerics.get("capacity_method_id")
    if recorded is not None:
        if not isinstance(recorded, str) or recorded not in METHOD_BY_ID:
            raise QgisProductError("Saved capacity identity is unknown",
                                   category="identity_mismatch")
        current = next(value for value in selected if value.startswith("capacity."))
        if recorded != current:
            selected.remove(current)
            selected.add(recorded)
            explanations.append("The explicit saved capacity identity overrides the legacy integer facade.")

    return selected, automatic, explanations


def _classify(method_ids: set[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    delft, legacy, extension = [], [], []
    for method_id in sorted(method_ids):
        row = method(method_id)
        if row.disposition == SCOPE_BRAIDS_LEGACY:
            legacy.append(method_id)
        elif row.disposition == SCOPE_BRAIDS_EXTENSION:
            extension.append(method_id)
        else:
            delft.append(method_id)
    return tuple(delft), tuple(legacy), tuple(extension)


def _worker_requirements(backend: str, catalogue_id: str,
                         output_id: str,
                         selected_method_ids=()) -> dict[str, Any]:
    selected = frozenset(selected_method_ids)
    recursive_bank = _recursive_bank_bundle(selected)
    result = {
        "backend": backend,
        "integrator": CPU_INTEGRATOR if backend == CPU_BACKEND else INTEGRATOR,
        "automatic_backend_fallback": False,
        "product_catalogue_id": catalogue_id,
        "semantic_catalog_id": semantic_catalog_id(),
        "cross_physics_catalogue_id": cross_physics_catalogue_v108()["identity"],
        "output_registry_id": output_id,
        "required_worker_variant": (
            RECURSIVE_BANK_WORKER_VARIANT if recursive_bank else
            None if backend == CPU_BACKEND else
            "live_normal_critical_transport_ssp33"),
    }
    if recursive_bank:
        result["recursive_bank_contract"] = "braids.recursive_worker.v118"
    return result


def resolve_qgis_product_v109(
    project: Mapping[str, Any], *, backend: str,
    output_selection: Mapping[str, object] | None = None,
    apply_output_defaults: bool = False,
) -> ProductResolutionV109:
    """Resolve one portable QGIS product profile before mesh allocation."""
    project = _mapping(project, "project")
    reject_managed_sediment_v109(project)
    backend = canonical_backend(backend)
    selected, automatic, explanations = _project_methods(project)

    for method_id in sorted(selected):
        row = method(method_id)
        if row.disposition in (SCOPE_EXCLUDED, SCOPE_REFERENCE_DISABLED):
            raise QgisProductError(
                f"{row.label} is outside the current-only natural-river scope",
                category="scope_exclusion",
            )
        if method_id == "capacity.gaia_research" or method_id == "bed_slope.gaia_talmon":
            raise QgisProductError(
                f"{row.label} is a labelled TELEMAC-GAIA research extension and "
                "is excluded from Delft3D4 product coverage",
                category="scope_exclusion",
            )
    try:
        physical = resolve_profile(selected)
    except SedimentCatalogError as error:
        category = ("scope_exclusion" if error.category == "scope_restriction"
                    else "physical_incompatibility")
        raise QgisProductError(str(error), category=category) from error

    missing = sorted(set(physical.selected_method_ids) - _connected_methods(
        backend, physical.selected_method_ids))
    if missing:
        labels = ", ".join(method(value).label for value in missing)
        raise QgisProductError(
            f"The selected {backend} product route is not connected for: {labels}. "
            "Component tests do not constitute QGIS/runtime admission.",
            category="backend_route_unimplemented",
        )

    outputs = resolve_output_selection_v108(
        {} if output_selection is None else output_selection,
        apply_source_defaults=apply_output_defaults,
    )
    registry = output_registry_v108()
    cross = cross_physics_catalogue_v108()
    catalogue_id = product_catalogue_v109()["identity"]
    worker = _worker_requirements(
        backend, catalogue_id, registry["identity"],
        physical.selected_method_ids)
    delft, legacy, extension = _classify(set(physical.selected_method_ids))
    integrator = CPU_INTEGRATOR if backend == CPU_BACKEND else INTEGRATOR
    body = {
        "schema": SCHEMA,
        "product_id": PRODUCT_ID,
        "backend": backend,
        "integrator": integrator,
        "selected_method_ids": list(physical.selected_method_ids),
        "automatic_method_ids": sorted(automatic),
        "explanations": explanations + list(physical.reasons),
        "delft3d4_method_ids": list(delft),
        "braids_legacy_method_ids": list(legacy),
        "braids_extension_method_ids": list(extension),
        "semantic_catalog_id": semantic_catalog_id(),
        "cross_physics_catalogue_id": cross["identity"],
        "output_registry_id": registry["identity"],
        "output_selection": outputs,
        "worker_requirements": worker,
    }
    runtime_body = dict(body)
    for key in (
            "selected_method_ids", "automatic_method_ids", "explanations",
            "delft3d4_method_ids", "braids_legacy_method_ids",
            "braids_extension_method_ids"):
        runtime_body[key] = tuple(runtime_body[key])
    return ProductResolutionV109(
        **runtime_body, fingerprint=_digest(body),
    )


def validate_product_document_v109(
    project: Mapping[str, Any], document: Mapping[str, Any],
) -> ProductResolutionV109:
    """Re-resolve a saved identity; never trust mutable labels or reasons."""
    document = _mapping(document, "QGIS product document")
    if document.get("schema") != SCHEMA or document.get("product_id") != PRODUCT_ID:
        raise QgisProductError("Unknown QGIS product document",
                               category="identity_mismatch")
    selection = _mapping(document.get("output_selection"), "output selection")
    settings = selection.get("settings")
    if not isinstance(settings, list):
        raise QgisProductError("Saved output settings are malformed",
                               category="identity_mismatch")
    raw_outputs = {}
    for row in settings:
        if not isinstance(row, Mapping) or set(row) != {"entry_id", "value"}:
            raise QgisProductError("Saved output setting is malformed",
                                   category="identity_mismatch")
        raw_outputs[str(row["entry_id"])] = row["value"]
    resolved = resolve_qgis_product_v109(
        project, backend=str(document.get("backend")),
        output_selection=raw_outputs,
        apply_output_defaults=bool(selection.get("apply_source_defaults", False)),
    )
    if resolved.as_dict() != dict(document):
        raise QgisProductError(
            "Saved QGIS product identity differs from the current project/catalogue",
            category="identity_mismatch",
        )
    return resolved


def load_product_document_v109(document: Mapping[str, Any]) -> ProductResolutionV109:
    """Validate a prepared/checkpoint identity without reopening a QGIS project."""
    document = _mapping(document, "QGIS product document")
    expected_keys = set(ProductResolutionV109.__dataclass_fields__)
    if set(document) != expected_keys:
        raise QgisProductError("Prepared QGIS product fields differ",
                               category="identity_mismatch")
    raw = dict(document)
    fingerprint = raw.pop("fingerprint")
    if (raw.get("schema") != SCHEMA or raw.get("product_id") != PRODUCT_ID or
            _digest(raw) != fingerprint):
        raise QgisProductError("Prepared QGIS product fingerprint differs",
                               category="identity_mismatch")
    backend = canonical_backend(str(raw.get("backend")))
    expected_integrator = CPU_INTEGRATOR if backend == CPU_BACKEND else INTEGRATOR
    if raw.get("integrator") != expected_integrator:
        raise QgisProductError("Prepared QGIS product integrator differs",
                               category="identity_mismatch")
    if (raw.get("semantic_catalog_id") != semantic_catalog_id() or
            raw.get("cross_physics_catalogue_id") !=
                cross_physics_catalogue_v108()["identity"] or
            raw.get("output_registry_id") != output_registry_v108()["identity"]):
        raise QgisProductError("Prepared QGIS product catalogue differs",
                               category="identity_mismatch")
    selected = raw.get("selected_method_ids")
    if not isinstance(selected, list) or not all(isinstance(v, str) for v in selected):
        raise QgisProductError("Prepared method inventory is malformed",
                               category="identity_mismatch")
    try:
        physical = resolve_profile(selected)
    except SedimentCatalogError as error:
        raise QgisProductError(str(error),
                               category="physical_incompatibility") from error
    if list(physical.selected_method_ids) != selected:
        raise QgisProductError("Prepared methods are not canonical",
                               category="identity_mismatch")
    missing = sorted(set(selected) - _connected_methods(backend, selected))
    if missing:
        raise QgisProductError(
            "Prepared product contains an unconnected method route: " +
            ", ".join(missing), category="backend_route_unimplemented")
    output = _mapping(raw.get("output_selection"), "prepared output selection")
    settings = output.get("settings")
    if not isinstance(settings, list):
        raise QgisProductError("Prepared output settings are malformed",
                               category="identity_mismatch")
    requested = {}
    for row in settings:
        if not isinstance(row, Mapping) or set(row) != {"entry_id", "value"}:
            raise QgisProductError("Prepared output setting is malformed",
                                   category="identity_mismatch")
        requested[str(row["entry_id"])] = row["value"]
    rebuilt = resolve_output_selection_v108(
        requested,
        apply_source_defaults=bool(output.get("apply_source_defaults", False)))
    if rebuilt != output:
        raise QgisProductError("Prepared output identity differs",
                               category="identity_mismatch")
    catalogue_id = product_catalogue_v109()["identity"]
    if raw.get("worker_requirements") != _worker_requirements(
            backend, catalogue_id, output_registry_v108()["identity"],
            selected):
        raise QgisProductError("Prepared worker requirements differ",
                               category="identity_mismatch")
    try:
        return ProductResolutionV109(
            schema=str(raw["schema"]), product_id=str(raw["product_id"]),
            backend=backend, integrator=str(raw["integrator"]),
            selected_method_ids=tuple(raw["selected_method_ids"]),
            automatic_method_ids=tuple(raw["automatic_method_ids"]),
            explanations=tuple(raw["explanations"]),
            delft3d4_method_ids=tuple(raw["delft3d4_method_ids"]),
            braids_legacy_method_ids=tuple(raw["braids_legacy_method_ids"]),
            braids_extension_method_ids=tuple(raw["braids_extension_method_ids"]),
            semantic_catalog_id=str(raw["semantic_catalog_id"]),
            cross_physics_catalogue_id=str(raw["cross_physics_catalogue_id"]),
            output_registry_id=str(raw["output_registry_id"]),
            output_selection=raw["output_selection"],
            worker_requirements=raw["worker_requirements"],
            fingerprint=str(fingerprint),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise QgisProductError("Prepared QGIS product fields are malformed",
                               category="identity_mismatch") from error


def verify_worker_capabilities_v109(
    profile: ProductResolutionV109 | Mapping[str, Any],
    worker_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify package capabilities without importing CUDA or allocating arrays."""
    profile_doc = profile.as_dict() if isinstance(profile, ProductResolutionV109) else dict(profile)
    required = _mapping(profile_doc.get("worker_requirements"), "worker requirements")
    if required.get("backend") != GPU_BACKEND:
        return {"status": "not_required", "backend": CPU_BACKEND}
    advertised = _mapping(worker_scope.get("qgis_product_v109"),
                          "worker Q10 capability")
    expected = {
        "schema": SCHEMA,
        "product_id": PRODUCT_ID,
        "product_catalogue_id": required["product_catalogue_id"],
        "semantic_catalog_id": required["semantic_catalog_id"],
        "cross_physics_catalogue_id": required["cross_physics_catalogue_id"],
        "output_registry_id": required["output_registry_id"],
        "backend": GPU_BACKEND,
        "integrator": INTEGRATOR,
        "automatic_backend_fallback": False,
    }
    if dict(advertised) != expected:
        raise QgisProductError(
            "The selected GPU worker does not advertise the exact prepared Q10 product identity",
            category="backend_route_unimplemented",
        )
    recursive_required=required.get("recursive_bank_contract")
    if recursive_required is not None:
        recursive=_mapping(worker_scope.get("recursive_worker_v118"),
                           "worker recursive-v118 capability")
        if recursive != {
                "schema": recursive_required,
                "required_worker_variant": RECURSIVE_BANK_WORKER_VARIANT,
                "source": "braids_core/sediment/recursive_worker_v118.py",
                "source_sha256": recursive.get("source_sha256"),
                "automatic_backend_fallback": False,
        } or not isinstance(recursive.get("source_sha256"),str) or \
                len(recursive["source_sha256"])!=64:
            raise QgisProductError(
                "The selected GPU worker does not advertise the required v118 recursive lifecycle",
                category="backend_route_unimplemented")
    capability=dict(expected)
    if recursive_required is not None:
        capability["recursive_worker_v118"]=dict(recursive)
    return {"status": "pass", "capability": capability,
            "profile_fingerprint": profile_doc.get("fingerprint")}


def product_catalogue_v109() -> dict[str, Any]:
    """Return the deterministic catalogue used to generate QGIS/manual assets."""
    registry = output_registry_v108()
    rows = []
    for row in sorted(METHODS, key=lambda value: (value.component, value.label, value.method_id)):
        if row.disposition == SCOPE_BRAIDS_LEGACY:
            provenance = "BRAIDS legacy"
            delft_coverage = False
        elif row.disposition == SCOPE_BRAIDS_EXTENSION:
            provenance = "BRAIDS/GAIA extension"
            delft_coverage = False
        elif row.disposition in (SCOPE_EXCLUDED, SCOPE_REFERENCE_DISABLED):
            provenance = "Delft3D4 inventoried exclusion"
            delft_coverage = False
        else:
            provenance = "Delft3D4 FLOW 2DH"
            delft_coverage = True
        rows.append({
            "method_id": row.method_id,
            "component": row.component,
            "label": row.label,
            "reference_selector": row.reference_selector,
            "disposition": row.disposition,
            "provenance": provenance,
            "counts_as_delft3d4_coverage": delft_coverage,
            "product_backend_status": {
                CPU_BACKEND: ("connected" if row.method_id in _CONNECTED_METHODS[CPU_BACKEND]
                              else "not_connected"),
                GPU_BACKEND: ("connected" if row.method_id in _CONNECTED_METHODS[GPU_BACKEND]
                              else "not_connected"),
            },
            "source_id": row.source_id,
            "requires": list(row.requires),
            "notes": list(row.notes),
        })
    document = {
        "schema": CATALOGUE_SCHEMA,
        "product_id": PRODUCT_ID,
        "delft3d4_source_commit": DELFT_SOURCE_COMMIT,
        "semantic_catalog_id": semantic_catalog_id(),
        "cross_physics_catalogue_id": cross_physics_catalogue_v108()["identity"],
        "output_registry_id": registry["identity"],
        "backends": list(BACKENDS),
        "gpu_integrator": INTEGRATOR,
        "automatic_backend_selection": False,
        "managed_fildad": "scope_exclusion_before_allocation",
        "method_choices": rows,
        "output_choices": registry["mappings"],
        "error_categories": sorted(ERROR_CATEGORIES),
    }
    document["identity"] = _digest(document)
    return document


def product_manual_v109() -> str:
    """Generate the shipped method manual from the same immutable catalogue."""
    catalogue = product_catalogue_v109()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in catalogue["method_choices"]:
        grouped.setdefault(row["component"], []).append(row)
    lines = [
        "# BRAIDS Q10 sediment-method manual",
        "",
        f"Product: `{PRODUCT_ID}`  ",
        f"Catalogue identity: `{catalogue['identity']}`  ",
        f"Locked Delft3D4 source commit: `{DELFT_SOURCE_COMMIT}`",
        "",
        "The CPU/GPU backend is always selected by the user. The GPU product route uses "
        "`implicit_fwave_midpoint`; there is no mesh-size selector or silent CPU fallback. "
        "Auxiliary methods are resolved automatically from the saved physical choices.",
        "",
        "A **connected** row has a complete route through the current product preflight. "
        "A component-only or prototype row remains **not connected** and is rejected before "
        "mesh/array allocation. This distinction is deliberate.",
        "",
        "Managed Delft3D4 `Fildad` dredging, dumping, mining and nourishment are active "
        "reference capabilities but outside the natural/unmanaged-river product. Wave, "
        "roller, 3D-only, source-disabled and FM-only choices also fail closed. BRAIDS/GAIA "
        "extensions are labelled and never counted as Delft3D4 coverage.",
        "",
        "## Method catalogue",
        "",
    ]
    for component in sorted(grouped):
        lines.extend((f"### {component.replace('_', ' ').title()}", "",
                      "| Method | Provenance | CPU | implicit GPU | Reference selector |", 
                      "|---|---|---|---|---|"))
        for row in grouped[component]:
            status = row["product_backend_status"]
            selector = row["reference_selector"] or "—"
            lines.append(
                f"| `{row['method_id']}` — {row['label']} | {row['provenance']} | "
                f"{status[CPU_BACKEND]} | {status[GPU_BACKEND]} | `{selector}` |"
            )
        lines.append("")
    lines.extend((
        "## Output selectors",
        "",
        "The Results surface is generated from the 69 audited morphology selectors. "
        "Sixty-two map to result/checkpoint destinations. Five wave-current selectors, "
        "the FM-only morphology-statistics interval and the classic parser key with no "
        "writer are shown as explicit exclusions.",
        "",
        "## Rejection categories",
        "",
        "- `physical_incompatibility`: the selected equations would double-count or require "
        "  mutually incompatible physics.",
        "- `scope_exclusion`: the feature exists outside this natural-river/Delft3D4 scope.",
        "- `backend_route_unimplemented`: a component exists, but the selected product route "
        "  is not connected end to end.",
        "- `validation_restriction`: the route exists but lacks the required validation scope.",
        "- `input_contract` / `identity_mismatch`: malformed or stale saved/prepared input.",
        "",
    ))
    return "\n".join(lines)


__all__ = [
    "BACKENDS", "CATALOGUE_SCHEMA", "CPU_BACKEND", "CPU_INTEGRATOR",
    "GPU_BACKEND", "PRODUCT_ID", "PRODUCT_SIDECAR", "QgisProductError",
    "ProductResolutionV109", "SCHEMA", "canonical_backend",
    "product_catalogue_v109", "product_manual_v109",
    "load_product_document_v109", "reject_managed_sediment_v109",
    "resolve_qgis_product_v109",
    "validate_product_document_v109", "verify_worker_capabilities_v109",
]
