"""Generalized conservative AMR rebuild for v115/v116 bank compositions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .bed_columns_v101 import host
from .cohesive_banks_v107 import remap_cohesive_bank_owner_v107
from .dependent_layers_v103 import remap_dependent_owner_v103
from .distributed_banks_v117 import (
    DistributedBankOwnerV117, remap_distributed_bank_owner_v117,
)
from .dynamic_bedforms_v104 import remap_dynamic_bedform_owner_v104
from .recursive_cross_physics_v115 import (
    CoupledAMRRecursiveCrossPhysicsV115,
)
from .recursive_physical_toe_v116 import (
    CoupledAMRRecursivePhysicalToeV116, PhysicalToeFaceLedgerV116,
)
from .recursive_surface_feedback_v114 import (
    CoupledAMRRecursiveSurfaceFeedbackV114,
)


SCOPE = "braids-recursive-bank-amr-v117"
LEVELS = ("coarse", "fine")


class RecursiveBankAMRV117Error(ValueError):
    """A conservative cell/face map or rebuilt owner is inconsistent."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ConservativeCellMapV117:
    offsets: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    target_area: np.ndarray
    identity: str

    @classmethod
    def create(cls, offsets, indices, weights, target_area):
        off = np.ascontiguousarray(offsets, dtype=np.int32)
        src = np.ascontiguousarray(indices, dtype=np.int32)
        w = np.ascontiguousarray(weights, dtype=np.float64)
        area = np.ascontiguousarray(target_area, dtype=np.float64)
        if (area.ndim != 2 or np.any(area <= 0.0) or
                not np.isfinite(area).all() or
                off.shape != (area.size+1,) or off[0] != 0 or
                off[-1] != src.size or src.shape != w.shape or
                np.any(np.diff(off) <= 0) or np.any(src < 0) or
                np.any(w < 0.0) or not np.isfinite(w).all()):
            raise RecursiveBankAMRV117Error("Invalid conservative cell map")
        h = hashlib.sha256()
        for value in (off, src, w, area): h.update(value.tobytes())
        identity = _digest({"scope": SCOPE, "arrays": h.hexdigest()})
        for value in (off, src, w, area): value.setflags(write=False)
        return cls(off, src, w, area, identity)


@dataclass(frozen=True)
class FaceLedgerMapV117:
    old_face: np.ndarray
    new_face: np.ndarray
    weight: np.ndarray
    orientation: np.ndarray
    identity: str

    @classmethod
    def create(cls, old_face, new_face, weight, orientation=None):
        old = np.ascontiguousarray(old_face, dtype=np.int64)
        new = np.ascontiguousarray(new_face, dtype=np.int64)
        w = np.ascontiguousarray(weight, dtype=np.float64)
        orient = np.ascontiguousarray(
            np.ones(old.shape, dtype=np.int8) if orientation is None
            else orientation, dtype=np.int8)
        if (old.ndim != 1 or not old.size or new.shape != old.shape or
                w.shape != old.shape or orient.shape != old.shape or
                np.any(old < 0) or np.any(new < 0) or
                np.any(w <= 0.0) or not np.isfinite(w).all() or
                not np.isin(orient, (-1, 1)).all()):
            raise RecursiveBankAMRV117Error("Invalid face-ledger map")
        h = hashlib.sha256()
        for value in (old, new, w, orient): h.update(value.tobytes())
        identity = _digest({"scope": SCOPE, "arrays": h.hexdigest()})
        for value in (old, new, w, orient): value.setflags(write=False)
        return cls(old, new, w, orient, identity)


@dataclass(frozen=True)
class RecursiveBankAMRReceiptV117:
    route: str
    backend: str
    source_owner_identity: str
    target_owner_identity: str
    cell_map_sha256: str
    face_map_sha256: str | None
    source_cell_count: int
    target_cell_count: int
    maximum_class_mass_residual_kg: float
    maximum_q5_bank_mirror_error_kg_m2: float
    maximum_toe_gross_ledger_residual_kg: float
    topology_epoch: int
    automatic_cpu_fallback: bool
    scope: str
    receipt_sha256: str


def _class_total(bank):
    area = np.asarray(host(bank.state["area"]), dtype=np.float64)
    return (np.sum(host(bank.state["layers"])*area, axis=(0, 2, 3))+
            np.sum(host(bank.state["suspended_mass_areal"])*area,
                   axis=(1, 2))+
            np.sum(host(bank.state["fluff_mass_areal"])*area,
                   axis=(1, 2)))


def _mapped_scalar(value, mapping, source_shape):
    flat = np.asarray(host(value), dtype=np.float64).reshape(-1)
    if flat.size != int(np.prod(source_shape)):
        raise RecursiveBankAMRV117Error("AMR scalar source shape differs")
    out = np.empty(mapping.target_area.size, dtype=np.float64)
    for target in range(out.size):
        ids = mapping.indices[mapping.offsets[target]:mapping.offsets[target+1]]
        weights = mapping.weights[
            mapping.offsets[target]:mapping.offsets[target+1]]
        out[target] = np.sum(flat[ids]*weights)
    return out.reshape(mapping.target_area.shape)


def _surface_from(old_surface, columns):
    settings = old_surface.levels["coarse"]
    for name in LEVELS:
        candidate = old_surface.levels[name]
        if (candidate.hiding_mode != settings.hiding_mode or
                candidate.parker_alpha != settings.parker_alpha or
                candidate.wu_exponent != settings.wu_exponent or
                tuple(item.fingerprint for item in candidate.distributions) !=
                tuple(item.fingerprint for item in settings.distributions)):
            raise RecursiveBankAMRV117Error(
                "Recursive levels use different surface methods")
    return CoupledAMRRecursiveSurfaceFeedbackV114(
        columns["coarse"], columns["fine"],
        hiding_mode=settings.hiding_mode,
        parker_alpha=settings.parker_alpha,
        wu_exponent=settings.wu_exponent,
        distributions=settings.distributions)


def _transfer_toe_ledger(old, new, face_maps):
    maximum_residual = 0.0
    for name in LEVELS:
        source = old.ledgers[name]
        target = new.ledgers[name]
        mapping = face_maps[name]
        if (np.any(mapping.old_face >= source.mapping.face_count) or
                np.any(mapping.new_face >= target.mapping.face_count)):
            raise RecursiveBankAMRV117Error(
                "Face-ledger remap index is outside topology")
        coverage = np.bincount(
            mapping.old_face, weights=mapping.weight,
            minlength=source.mapping.face_count)
        old_gross = np.asarray(host(source.gross_kg))
        active = np.any(old_gross != 0.0, axis=0)
        if np.any(np.abs(coverage[active]-1.0) > 1.e-13):
            raise RecursiveBankAMRV117Error(
                "Accepted nonzero toe ledger lacks complete face coverage")
        xp = target.xp
        old_signed = xp.asarray(host(source.signed_kg))
        old_gross_device = xp.asarray(old_gross)
        for index in range(mapping.old_face.size):
            source_face = int(mapping.old_face[index])
            target_face = int(mapping.new_face[index])
            weight = float(mapping.weight[index])
            orientation = int(mapping.orientation[index])
            target.signed_kg[:, target_face] += (
                old_signed[:, source_face]*weight*orientation)
            target.gross_kg[:, target_face] += (
                old_gross_device[:, source_face]*weight)
        residual = np.max(np.abs(
            np.sum(host(target.gross_kg), axis=1)-
            np.sum(old_gross, axis=1)))
        maximum_residual = max(maximum_residual, float(residual))
        if residual > 2.e-10:
            raise RecursiveBankAMRV117Error(
                "Toe gross ledger changed during conservative regrid")
    return maximum_residual


def regrid_recursive_bank_v117(owner, *, cell_maps,
                               target_column_runtimes,
                               target_bedform_runtimes,
                               target_bank_components,
                               face_maps=None):
    """Rebuild a private v115/v116 owner, then return it for caller commit.

    The accepted source is never mutated.  ``target_bank_components`` contains
    surveyed maps for mode 2 or target distributed runtimes for mode 3.
    """
    physical = isinstance(owner, CoupledAMRRecursivePhysicalToeV116)
    cross = owner.cross if physical else owner
    if not isinstance(cross, CoupledAMRRecursiveCrossPhysicsV115):
        raise TypeError("A versioned v115/v116 recursive owner is required")
    dictionaries = (cell_maps, target_column_runtimes,
                    target_bedform_runtimes, target_bank_components)
    if any(not isinstance(value, Mapping) or set(value) != set(LEVELS)
           for value in dictionaries):
        raise RecursiveBankAMRV117Error(
            "AMR rebuild requires explicit coarse and fine components")
    if physical and (not isinstance(face_maps, Mapping) or
                     set(face_maps) != set(LEVELS)):
        raise RecursiveBankAMRV117Error(
            "Physical-toe AMR requires both face-ledger maps")
    before = {name: _class_total(cross.banks[name]) for name in LEVELS}
    columns = {}; banks = {}; bedforms = {}
    for name in LEVELS:
        mapping = cell_maps[name]
        if not isinstance(mapping, ConservativeCellMapV117):
            raise RecursiveBankAMRV117Error("Versioned cell map required")
        old_column = cross.surface.levels[name].columns
        columns[name] = remap_dependent_owner_v103(
            old_column, target_column_runtimes[name], mapping.offsets,
            mapping.indices, mapping.weights, mapping.target_area.ravel())
        bedforms[name] = remap_dynamic_bedform_owner_v104(
            cross.bedforms[name], target_bedform_runtimes[name],
            mapping.offsets, mapping.indices, mapping.weights,
            mapping.target_area)
        old_bank = cross.banks[name]
        if physical:
            surveyed = target_bank_components[name]
            mask, target, length = surveyed.bank_fields(
                mapping.target_area.shape)
            height = _mapped_scalar(
                old_bank.state["bank_height_m"], mapping,
                old_bank.state["area"].shape)
            banks[name] = remap_cohesive_bank_owner_v107(
                old_bank, mapping.offsets, mapping.indices, mapping.weights,
                mapping.target_area, bank_mask=mask, toe_target=target,
                bank_height_m=height, bank_face_length_m=length)
        else:
            if not isinstance(old_bank, DistributedBankOwnerV117):
                raise RecursiveBankAMRV117Error(
                    "Mode-3 generalized regrid requires v117 bank owners")
            banks[name] = remap_distributed_bank_owner_v117(
                old_bank, target_bank_components[name], mapping.offsets,
                mapping.indices, mapping.weights, mapping.target_area)
    surface = _surface_from(cross.surface, columns)
    new_cross = CoupledAMRRecursiveCrossPhysicsV115(
        surface, banks, bedforms)
    if physical:
        result = CoupledAMRRecursivePhysicalToeV116(
            new_cross, target_bank_components)
        ledger_residual = _transfer_toe_ledger(owner, result, face_maps)
        route = "physical_toe_mode2"
        face_sha = _digest({name: face_maps[name].identity
                            for name in LEVELS})
        epoch = max(item.topology_epoch
                    for item in target_bank_components.values())
    else:
        result = new_cross; ledger_residual = 0.0
        route = "distributed_bank_mode3"; face_sha = None
        epoch = max(item.topology.topology_epoch
                    for item in target_bank_components.values())
    after = {name: _class_total(new_cross.banks[name]) for name in LEVELS}
    mass_residual = max(float(np.max(np.abs(after[name]-before[name])))
                        for name in LEVELS)
    mirror = 0.0
    for name in LEVELS:
        mirror = max(mirror, float(np.max(np.abs(
            host(new_cross.banks[name].state["layers"]).reshape(
                columns[name].mass.shape)-host(columns[name].mass)))))
    if mass_residual > 2.e-10 or mirror > 2.e-12:
        raise RecursiveBankAMRV117Error(
            "Rebuilt recursive bank owner changed accepted mass/mirror")
    payload = {
        "route": route, "backend": result.backend,
        "source_owner_identity": owner.identity,
        "target_owner_identity": result.identity,
        "cell_map_sha256": _digest({name: cell_maps[name].identity
                                    for name in LEVELS}),
        "face_map_sha256": face_sha,
        "source_cell_count": sum(
            cross.surface.levels[name].columns.runtime.nc for name in LEVELS),
        "target_cell_count": sum(
            columns[name].runtime.nc for name in LEVELS),
        "maximum_class_mass_residual_kg": mass_residual,
        "maximum_q5_bank_mirror_error_kg_m2": mirror,
        "maximum_toe_gross_ledger_residual_kg": ledger_residual,
        "topology_epoch": epoch,
        "automatic_cpu_fallback": False, "scope": SCOPE,
    }
    receipt = RecursiveBankAMRReceiptV117(
        **payload, receipt_sha256=_digest(payload))
    return result, receipt


__all__ = [
    "SCOPE", "ConservativeCellMapV117", "FaceLedgerMapV117",
    "RecursiveBankAMRReceiptV117", "RecursiveBankAMRV117Error",
    "regrid_recursive_bank_v117",
]
