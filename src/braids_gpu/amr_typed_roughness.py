"""Typed raw hydraulic-roughness ownership across AMR topology changes.

The native deck keeps Chezy C, Manning n and Nikuradse ks in the historical
``manning_cell`` column.  That name is storage only.  This module authenticates
one immutable raw parameter on every hierarchy node and makes the physical
method identity common to every selected topology in an AMR run.

Restriction is the area mean of the *raw* intensive material parameter.  It is
never the historical maximum child Manning value and never a depth-dependent
equivalent-n conversion.  Prolongation restores the hierarchy-owned child raw
values, so refine/coarsen cycles do not erase the original spatial field.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from .forcing_profile import (BINDING_FIELD as LOCAL_BINDING_FIELD,
                              apply_forcing_binding, forcing_identity,
                              rebind_forcing_parameter)


SCHEMA = 1
BINDING_FIELD = "amr_typed_roughness_binding"
RESTRICTION_POLICY = "area_weighted_raw_parameter_v1"
PROLONGATION_POLICY = "hierarchy_owned_immutable_child_raw_parameter_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _array(value, dtype, *, ndim=1, name="AMR array"):
    result = np.asarray(value)
    _require(result.ndim == ndim and result.dtype == np.dtype(dtype) and
             result.flags.c_contiguous and
             (result.dtype.kind != "f" or np.isfinite(result).all()),
             name + " must be contiguous, finite, and have its canonical dtype")
    return result


def _identity(value):
    result = np.asarray(value)
    return dict(shape=[int(item) for item in result.shape],
                dtype=result.dtype.str,
                sha256=hashlib.sha256(result.tobytes(order="C")).hexdigest())


def _tree(tree):
    _require(isinstance(tree, Mapping), "AMR roughness tree must be a mapping")
    parent = _array(tree.get("parent"), np.int64, name="AMR parent")
    level = _array(tree.get("node_level"), np.int64, name="AMR level")
    area = _array(tree.get("area"), np.float64, name="AMR area")
    raw = _array(tree.get("roughness_parameter"), np.float64,
                 name="AMR raw roughness parameter")
    size = len(parent)
    child = parent >= 0
    _require(size > 0 and level.shape == area.shape == raw.shape == (size,) and
             np.all(area > 0.0) and np.any(~child) and
             np.all(parent[~child] == -1) and np.all(level[~child] == 0) and
             np.all(parent[child] < np.arange(size)[child]) and
             np.all(level >= 0) and
             np.all(level[child] == level[parent[child]] + 1),
             "AMR raw roughness hierarchy differs")
    return parent, level, area, raw


def _children(parent):
    return tuple(np.flatnonzero(parent == node).astype(np.int64)
                 for node in range(len(parent)))


def verify_raw_restriction(tree, native_mode):
    """Verify every parent uses bounded area-weighted raw C/n/ks restriction."""
    parent, level, area, raw = _tree(tree)
    _require(native_mode in (1, 2, 3),
             "AMR raw roughness method identity is invalid")
    _require(np.all(raw >= 0.0) if native_mode == 2 else np.all(raw > 0.0),
             "AMR raw roughness values are invalid for the selected method")
    groups = _children(parent)
    for node, child in enumerate(groups):
        if not len(child):
            continue
        total = float(np.sum(area[child]))
        expected = float(np.dot(area[child], raw[child]) / total)
        scale = max(1.0, abs(expected), float(np.max(np.abs(raw[child]))))
        _require(abs(float(raw[node]) - expected) <=
                 32.0 * np.finfo(np.float64).eps * scale and
                 float(np.min(raw[child])) <= float(raw[node]) <=
                 float(np.max(raw[child])),
                 "AMR parent roughness is not the bounded raw-parameter area mean")
    return True


def build_binding(local_finest_binding: Mapping[str, object], tree) -> dict:
    """Sign one topology-invariant raw-parameter identity for an AMR tree."""
    parent, level, area, raw = _tree(tree)
    finest = np.flatnonzero(level == np.max(level)).astype(np.int64)
    _require(len(finest) > 0, "AMR hierarchy has no finest raw parameters")
    # Rebinding to the same bytes both validates schema 2 and proves the
    # metadata was constructed from the native finest cells.
    rebound = rebind_forcing_parameter(local_finest_binding, raw[finest])
    _require(rebound["fingerprint"] == local_finest_binding.get("fingerprint"),
             "Finest AMR raw parameter differs from the case forcing binding")
    native_mode = int(local_finest_binding["native_hydraulic_roughness_mode"])
    if "raw_roughness_mode" in tree:
        observed = np.asarray(tree["raw_roughness_mode"])
        _require(observed.shape == () and observed.dtype.kind in "iu" and
                 int(observed) == native_mode,
                 "Native AMR tree roughness method identity differs")
    verify_raw_restriction(tree, native_mode)
    body = dict(
        schema=SCHEMA,
        local_forcing_binding=dict(local_finest_binding),
        roughness_method_id=local_finest_binding["roughness_method_id"],
        hydraulic_roughness_mode=local_finest_binding[
            "hydraulic_roughness_mode"],
        native_hydraulic_roughness_mode=native_mode,
        roughness_parameter_name=local_finest_binding[
            "roughness_parameter_name"],
        roughness_parameter_units=local_finest_binding[
            "roughness_parameter_units"],
        roughness_storage_field="manning_cell",
        hierarchy_node_count=len(parent),
        hierarchy_parent_array=_identity(parent),
        hierarchy_level_array=_identity(level),
        hierarchy_area_array=_identity(area),
        hierarchy_roughness_parameter_array=_identity(raw),
        finest_node_ids_array=_identity(finest),
        restriction_policy=RESTRICTION_POLICY,
        prolongation_policy=PROLONGATION_POLICY,
        raw_parameter_state_independent=True,
        equivalent_manning_conversion=False,
        historical_max_child_manning=False,
        automatic_roughness_substitution=False,
        automatic_CPU_fallback=False,
    )
    return {**body, "fingerprint":
            hashlib.sha256(_canonical(body)).hexdigest()}


def verify_binding(binding, tree) -> dict:
    _require(isinstance(binding, Mapping),
             "AMR typed roughness binding must be a mapping")
    expected = {
        "schema", "local_forcing_binding", "roughness_method_id",
        "hydraulic_roughness_mode", "native_hydraulic_roughness_mode",
        "roughness_parameter_name", "roughness_parameter_units",
        "roughness_storage_field", "hierarchy_node_count",
        "hierarchy_parent_array", "hierarchy_level_array",
        "hierarchy_area_array", "hierarchy_roughness_parameter_array",
        "finest_node_ids_array", "restriction_policy",
        "prolongation_policy", "raw_parameter_state_independent",
        "equivalent_manning_conversion", "historical_max_child_manning",
        "automatic_roughness_substitution", "automatic_CPU_fallback",
        "fingerprint",
    }
    _require(set(binding) == expected and binding.get("schema") == SCHEMA,
             "AMR typed roughness binding fields differ")
    body = {key: binding[key] for key in binding if key != "fingerprint"}
    _require(binding["fingerprint"] ==
             hashlib.sha256(_canonical(body)).hexdigest(),
             "AMR typed roughness binding fingerprint differs")
    parent, level, area, raw = _tree(tree)
    finest = np.flatnonzero(level == np.max(level)).astype(np.int64)
    local = binding["local_forcing_binding"]
    same = rebind_forcing_parameter(local, raw[finest])
    _require(same["fingerprint"] == local.get("fingerprint") and
             binding["roughness_method_id"] == local["roughness_method_id"] and
             binding["hydraulic_roughness_mode"] ==
             local["hydraulic_roughness_mode"] and
             binding["native_hydraulic_roughness_mode"] ==
             local["native_hydraulic_roughness_mode"] and
             binding["roughness_parameter_name"] ==
             local["roughness_parameter_name"] and
             binding["roughness_parameter_units"] ==
             local["roughness_parameter_units"] and
             binding["roughness_storage_field"] == "manning_cell" and
             binding["hierarchy_node_count"] == len(parent) and
             binding["hierarchy_parent_array"] == _identity(parent) and
             binding["hierarchy_level_array"] == _identity(level) and
             binding["hierarchy_area_array"] == _identity(area) and
             binding["hierarchy_roughness_parameter_array"] == _identity(raw) and
             binding["finest_node_ids_array"] == _identity(finest) and
             binding["restriction_policy"] == RESTRICTION_POLICY and
             binding["prolongation_policy"] == PROLONGATION_POLICY and
             binding["raw_parameter_state_independent"] is True and
             binding["equivalent_manning_conversion"] is False and
             binding["historical_max_child_manning"] is False and
             binding["automatic_roughness_substitution"] is False and
             binding["automatic_CPU_fallback"] is False,
             "AMR typed roughness semantic or byte identity differs")
    verify_raw_restriction(tree,
                           binding["native_hydraulic_roughness_mode"])
    return dict(binding)


def apply_amr_forcing_binding(data: dict, node_ids, tree,
                              binding: Mapping[str, object]):
    """Authenticate a selected topology and install its exact raw alias."""
    _require(isinstance(data, dict),
             "AMR typed forcing binding requires mutable operator data")
    binding = verify_binding(binding, tree)
    parent, level, area, raw = _tree(tree)
    ids = _array(node_ids, np.int64, name="AMR selected node IDs")
    _require(len(ids) == int(np.asarray(data.get("nc"))) and
             np.all((ids >= 0) & (ids < len(parent))) and
             len(np.unique(ids)) == len(ids),
             "AMR selected roughness node IDs differ")
    selected = np.ascontiguousarray(raw[ids])
    observed = _array(data.get("manning_cell"), np.float64,
                      name="Selected native raw roughness")
    _require(np.array_equal(observed, selected),
             "Selected native raw roughness differs from the hierarchy")
    local = rebind_forcing_parameter(binding["local_forcing_binding"],
                                     selected)
    apply_forcing_binding(data, local)
    data[BINDING_FIELD] = binding
    data["amr_typed_roughness_fingerprint"] = binding["fingerprint"]
    data["amr_roughness_node_ids"] = ids
    data["amr_roughness_restriction_policy"] = RESTRICTION_POLICY
    data["amr_roughness_prolongation_policy"] = PROLONGATION_POLICY
    _require(data["roughness_parameter"] is data["manning_cell"],
             "AMR canonical raw roughness is not the native parameter bytes")
    return data


def amr_forcing_identity(data: Mapping[str, object]) -> dict:
    """Return the topology-invariant forcing identity used by restart."""
    binding = data.get(BINDING_FIELD)
    _require(isinstance(binding, Mapping),
             "Prepared AMR typed roughness binding is absent")
    local = forcing_identity(data)
    _require(binding.get("local_forcing_binding", {}).get(
                 "roughness_method_id") ==
             local["hydraulic_roughness_method_id"] and
             data.get("amr_typed_roughness_fingerprint") ==
             binding.get("fingerprint"),
             "Prepared AMR local/common roughness identities differ")
    return dict(
        amr_typed_roughness_schema=SCHEMA,
        amr_typed_roughness_fingerprint=binding["fingerprint"],
        hydraulic_roughness_method_id=binding["roughness_method_id"],
        roughness_parameter_name=binding["roughness_parameter_name"],
        roughness_parameter_units=binding["roughness_parameter_units"],
        roughness_hierarchy_parameter_array=dict(
            binding["hierarchy_roughness_parameter_array"]),
        roughness_restriction_policy=binding["restriction_policy"],
        roughness_prolongation_policy=binding["prolongation_policy"],
        automatic_roughness_substitution=False,
        automatic_CPU_fallback=False,
    )


def propose_resident_transfer(binding, tree, old_ids, new_ids, accepted_raw,
                              *, backend="cuda"):
    """Gather immutable target values on CUDA and audit refine/coarsen closure.

    Raw roughness is immutable geometry, not a conserved evolving state row.
    Coarsening therefore selects the already authenticated area-mean parent;
    refinement restores the original child field.  CUDA checks both relations
    without a CPU numerical fallback and leaves ``accepted_raw`` unchanged.
    """
    _require(backend == "cuda", "AMR typed roughness backend must be CUDA")
    import cupy as cp

    binding = verify_binding(binding, tree)
    parent, level, area, raw = _tree(tree)
    old_ids = _array(old_ids, np.int64, name="Old AMR roughness node IDs")
    new_ids = _array(new_ids, np.int64, name="New AMR roughness node IDs")
    _require(isinstance(accepted_raw, cp.ndarray) and
             accepted_raw.shape == old_ids.shape and
             accepted_raw.dtype == cp.float64 and
             accepted_raw.flags.c_contiguous and
             bool(cp.all(cp.isfinite(accepted_raw))),
             "Accepted AMR roughness must be resident contiguous FP64")
    expected_old = cp.asarray(raw[old_ids])
    _require(bool(cp.array_equal(accepted_raw, expected_old)),
             "Accepted resident raw roughness differs from the hierarchy")
    before = accepted_raw.copy()
    resident_tree = cp.asarray(raw)
    candidate = cp.ascontiguousarray(resident_tree[cp.asarray(new_ids)])
    old_set, new_set = set(map(int, old_ids)), set(map(int, new_ids))
    groups = _children(parent)
    coarsened, refined = [], []
    for node in sorted(new_set - old_set):
        if parent[node] in old_set:
            # Child introduced by refinement; its owning parent is audited in
            # the old-minus-new loop below.
            continue
        child = groups[node]
        _require(len(child) > 0 and set(map(int, child)).issubset(old_set),
                 "AMR roughness coarsening is not a complete child group")
        child_position = cp.asarray([int(np.flatnonzero(old_ids == item)[0])
                                     for item in child])
        weights = cp.asarray(area[child])
        restricted = cp.sum(weights * accepted_raw[child_position]) / cp.sum(weights)
        target_position = int(np.flatnonzero(new_ids == node)[0])
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, abs(float(raw[node])))
        _require(abs(float((candidate[target_position] - restricted).item())) <=
                 tolerance,
                 "CUDA AMR raw-parameter restriction differs")
        coarsened.append(node)
    for node in sorted(old_set - new_set):
        child = groups[node]
        if len(child) and set(map(int, child)).issubset(new_set):
            pass
        elif parent[node] in new_set:
            # Child removed by coarsening; its new parent was audited above.
            continue
        child_position = cp.asarray([int(np.flatnonzero(new_ids == item)[0])
                                     for item in child])
        weights = cp.asarray(area[child])
        restored_mean = cp.sum(weights * candidate[child_position]) / cp.sum(weights)
        old_position = int(np.flatnonzero(old_ids == node)[0])
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, abs(float(raw[node])))
        _require(abs(float((accepted_raw[old_position] - restored_mean).item())) <=
                 tolerance,
                 "CUDA AMR raw-parameter prolongation closure differs")
        refined.append(node)
    explained_old = set(old_set & new_set)
    explained_new = set(old_set & new_set)
    for node in coarsened:
        explained_new.add(node)
        explained_old.update(map(int, groups[node]))
    for node in refined:
        explained_old.add(node)
        explained_new.update(map(int, groups[node]))
    _require(explained_old == old_set and explained_new == new_set and
             bool(cp.array_equal(accepted_raw, before)) and
             bool(cp.array_equal(candidate, cp.asarray(raw[new_ids]))),
             "AMR typed roughness transition is incomplete or mutated accepted data")
    receipt = dict(
        amr_typed_roughness_fingerprint=binding["fingerprint"],
        old_leaves=len(old_ids), new_leaves=len(new_ids),
        coarsened_parent_ids=coarsened, refined_parent_ids=refined,
        restriction_policy=RESTRICTION_POLICY,
        prolongation_policy=PROLONGATION_POLICY,
        transfer_executed_on_CUDA=True,
        candidate_only_no_accepted_mutation=True,
        automatic_CPU_fallback=False,
    )
    return candidate, receipt


__all__ = [
    "SCHEMA", "BINDING_FIELD", "RESTRICTION_POLICY", "PROLONGATION_POLICY",
    "verify_raw_restriction", "build_binding", "verify_binding",
    "apply_amr_forcing_binding", "amr_forcing_identity",
    "propose_resident_transfer",
]
