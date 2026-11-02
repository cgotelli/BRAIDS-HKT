"""Surveyed physical-toe mode-2 composition over the v115 owner.

The v115 transaction remains the sole bed/layer publisher.  This module adds
an authenticated face/cell map and accepted signed/gross diagnostics around
that transaction.  It does not repeat bank removal or deposition.  NumPy and
CuPy owners follow the same path; the selected backend is never changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .bed_columns_v101 import array_sha, host
from .recursive_cross_physics_v115 import (
    CoupledAMRRecursiveCrossPhysicsV115,
)


SCOPE = "braids-recursive-physical-toe-v116"
LEVELS = ("coarse", "fine")


class RecursivePhysicalToeV116Error(ValueError):
    """The surveyed toe map or its atomic transaction is inconsistent."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _readonly(value, dtype):
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SurveyedPhysicalToeMapV116:
    """One stable surveyed bank-toe face per active mode-2 bank cell."""

    cell_count: int
    face_count: int
    bank_cell: np.ndarray
    toe_cell: np.ndarray
    face_id: np.ndarray
    face_length_m: np.ndarray
    orientation: np.ndarray
    source_identity: str
    topology_epoch: int
    array_sha256: str
    identity: str
    scope: str = SCOPE

    @classmethod
    def create(cls, cell_count, face_count, bank_cell, toe_cell, face_id,
               face_length_m, *, orientation=None, source_identity,
               topology_epoch=0):
        nc, nf = int(cell_count), int(face_count)
        bank = _readonly(bank_cell, np.int64)
        toe = _readonly(toe_cell, np.int64)
        faces = _readonly(face_id, np.int64)
        length = _readonly(face_length_m, np.float64)
        orient = _readonly(
            np.ones(bank.shape, np.int8) if orientation is None
            else orientation, np.int8)
        if (nc < 1 or nf < 1 or bank.ndim != 1 or not bank.size or
                toe.shape != bank.shape or faces.shape != bank.shape or
                length.shape != bank.shape or orient.shape != bank.shape or
                np.any(bank < 0) or np.any(bank >= nc) or
                np.any(toe < 0) or np.any(toe >= nc) or
                np.any(bank == toe) or np.any(faces < 0) or
                np.any(faces >= nf) or not np.isfinite(length).all() or
                np.any(length <= 0.0) or not np.isin(orient, (-1, 1)).all() or
                len(np.unique(bank)) != bank.size or
                len(np.unique(faces)) != faces.size or
                type(source_identity) is not str or not source_identity or
                int(topology_epoch) < 0):
            raise RecursivePhysicalToeV116Error(
                "Invalid surveyed physical-toe face/cell map")
        arrays = array_sha(bank, toe, faces, length, orient)
        payload = {
            "scope": SCOPE, "cell_count": nc, "face_count": nf,
            "source_identity": source_identity,
            "topology_epoch": int(topology_epoch),
            "array_sha256": arrays,
        }
        return cls(nc, nf, bank, toe, faces, length, orient,
                   source_identity, int(topology_epoch), arrays,
                   _digest(payload))

    def validate(self):
        expected = SurveyedPhysicalToeMapV116.create(
            self.cell_count, self.face_count, self.bank_cell, self.toe_cell,
            self.face_id, self.face_length_m,
            orientation=self.orientation, source_identity=self.source_identity,
            topology_epoch=self.topology_epoch)
        if (self.scope != SCOPE or self.array_sha256 != expected.array_sha256 or
                self.identity != expected.identity):
            raise RecursivePhysicalToeV116Error(
                "Surveyed physical-toe map was mutated")
        return self

    def bank_fields(self, shape):
        if int(np.prod(shape)) != self.cell_count:
            raise RecursivePhysicalToeV116Error(
                "Surveyed physical-toe map and bank patch differ")
        mask = np.zeros(self.cell_count, dtype=bool)
        target = np.full(self.cell_count, -1, dtype=np.int64)
        length = np.ones(self.cell_count, dtype=np.float64)
        mask[self.bank_cell] = True
        target[self.bank_cell] = self.toe_cell
        length[self.bank_cell] = self.face_length_m
        return (mask.reshape(shape), target.reshape(shape),
                length.reshape(shape))

    def document(self):
        return {
            "scope": self.scope, "cell_count": self.cell_count,
            "face_count": self.face_count,
            "bank_cell": self.bank_cell.tolist(),
            "toe_cell": self.toe_cell.tolist(),
            "face_id": self.face_id.tolist(),
            "face_length_m": self.face_length_m.tolist(),
            "orientation": self.orientation.tolist(),
            "source_identity": self.source_identity,
            "topology_epoch": self.topology_epoch,
            "array_sha256": self.array_sha256, "identity": self.identity,
        }

    @classmethod
    def from_document(cls, value):
        result = cls.create(
            value.get("cell_count"), value.get("face_count"),
            value.get("bank_cell"), value.get("toe_cell"),
            value.get("face_id"), value.get("face_length_m"),
            orientation=value.get("orientation"),
            source_identity=value.get("source_identity"),
            topology_epoch=value.get("topology_epoch"))
        if (value.get("scope") != SCOPE or
                value.get("array_sha256") != result.array_sha256 or
                value.get("identity") != result.identity):
            raise RecursivePhysicalToeV116Error(
                "Surveyed physical-toe restart map differs")
        return result


class PhysicalToeFaceLedgerV116:
    """Accepted per-class face mass ledgers; no sediment is owned here."""

    def __init__(self, mapping, grain_count, xp=np, *, time_s=0., updates=0):
        self.mapping = mapping.validate()
        self.xp = xp
        self.grain_count = int(grain_count)
        if self.grain_count < 1:
            raise RecursivePhysicalToeV116Error("Toe ledger needs fractions")
        self.signed_kg = xp.zeros(
            (self.grain_count, mapping.face_count), dtype=xp.float64)
        self.gross_kg = xp.zeros_like(self.signed_kg)
        self.time_s = float(time_s)
        self.updates = int(updates)
        self.pending = None

    def propose(self, bank_trial, dt_s):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error("Toe ledger trial is open")
        xp = self.xp
        removed = bank_trial.get("toe_removed_mass_by_cell_kg")
        if removed is None:
            raise RecursivePhysicalToeV116Error(
                "Bank trial has no cell-resolved toe transfer")
        removed = xp.asarray(removed, dtype=xp.float64).reshape(
            (self.grain_count, -1))
        if (removed.shape[1] != self.mapping.cell_count or
                bool(xp.any(~xp.isfinite(removed))) or
                bool(xp.any(removed < 0.0))):
            raise RecursivePhysicalToeV116Error(
                "Invalid cell-resolved toe transfer")
        bank = xp.asarray(self.mapping.bank_cell)
        face = xp.asarray(self.mapping.face_id)
        orientation = xp.asarray(self.mapping.orientation, dtype=xp.float64)
        delta = removed[:, bank]
        signed = self.signed_kg.copy()
        gross = self.gross_kg.copy()
        xp.add.at(signed, (slice(None), face), delta*orientation[None, :])
        xp.add.at(gross, (slice(None), face), xp.abs(delta))
        self.pending = (signed, gross, float(dt_s), delta)
        return delta

    def commit(self):
        if self.pending is None:
            raise RecursivePhysicalToeV116Error("No toe ledger trial")
        self.signed_kg, self.gross_kg, dt_s, _ = self.pending
        self.pending = None
        self.time_s += dt_s
        self.updates += 1

    def rollback(self):
        self.pending = None

    @property
    def state_sha256(self):
        return array_sha(host(self.signed_kg), host(self.gross_kg),
                         np.asarray([self.time_s, self.updates], np.float64))

    def checkpoint(self):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "Toe ledger checkpoint requires committed state")
        raw = {
            "schema": "braids.physical_toe_face_ledger.v116",
            "map_identity": self.mapping.identity,
            "grain_count": self.grain_count,
            "time_s": self.time_s, "updates": self.updates,
            "signed_kg": host(self.signed_kg).tolist(),
            "gross_kg": host(self.gross_kg).tolist(),
            "state_sha256": self.state_sha256,
        }
        return {**raw, "checkpoint_sha256": _digest(raw)}

    def restore(self, value):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "Toe ledger restore requires committed state")
        raw = {key: item for key, item in value.items()
               if key != "checkpoint_sha256"}
        if (_digest(raw) != value.get("checkpoint_sha256") or
                value.get("schema") !=
                "braids.physical_toe_face_ledger.v116" or
                value.get("map_identity") != self.mapping.identity or
                int(value.get("grain_count", -1)) != self.grain_count):
            raise RecursivePhysicalToeV116Error(
                "Physical-toe ledger restart identity differs")
        signed = np.asarray(value["signed_kg"], dtype=np.float64)
        gross = np.asarray(value["gross_kg"], dtype=np.float64)
        shape = (self.grain_count, self.mapping.face_count)
        if (signed.shape != shape or gross.shape != shape or
                not np.isfinite(signed).all() or
                not np.isfinite(gross).all() or np.any(gross < 0.0)):
            raise RecursivePhysicalToeV116Error("Invalid toe ledger restart")
        self.signed_kg = self.xp.asarray(signed)
        self.gross_kg = self.xp.asarray(gross)
        self.time_s = float(value["time_s"])
        self.updates = int(value["updates"])
        if self.state_sha256 != value.get("state_sha256"):
            raise RecursivePhysicalToeV116Error(
                "Physical-toe ledger restart digest differs")
        return self


@dataclass(frozen=True)
class RecursivePhysicalToeReceiptV116:
    backend: str
    macro_dt_s: float
    cross_physics_receipt_sha256: str
    accepted_state_sha256: str
    topology_epoch: int
    active_toe_face_count: int
    toe_signed_increment_kg: tuple[float, ...]
    toe_gross_increment_kg: tuple[float, ...]
    collapse_event_count: int
    accepted_bed_layer_owner_count: int
    automatic_cpu_fallback: bool
    scope: str
    receipt_sha256: str


class CoupledAMRRecursivePhysicalToeV116:
    """Atomic v115 plus surveyed mode-2 face-ledger composition."""

    def __init__(self, cross, topologies):
        if not isinstance(cross, CoupledAMRRecursiveCrossPhysicsV115):
            raise TypeError("Versioned v115 cross-physics owner required")
        if (not isinstance(topologies, Mapping) or
                set(topologies) != set(LEVELS)):
            raise RecursivePhysicalToeV116Error(
                "Explicit coarse and fine surveyed toe maps are required")
        self.cross = cross
        self.topologies = {name: topologies[name].validate()
                           for name in LEVELS}
        self.ledgers = {}
        for name in LEVELS:
            bank = cross.banks[name]
            mapping = self.topologies[name]
            shape = bank.state["area"].shape
            mask, target, length = mapping.bank_fields(shape)
            comparisons = (
                (bank.state["bank_mask"], mask, "bank mask"),
                (bank.state["toe_target"], target, "toe targets"),
                (bank.state["bank_face_length_m"], length, "face length"),
            )
            for actual, expected, label in comparisons:
                if not np.array_equal(host(actual), expected):
                    raise RecursivePhysicalToeV116Error(
                        f"{name} surveyed map and Q8b {label} differ")
            xp = bank.runtime.xp
            self.ledgers[name] = PhysicalToeFaceLedgerV116(
                mapping, bank.state["layers"].shape[1], xp,
                time_s=bank.time_s, updates=bank.updates)
        self.pending = None
        self._set_identity()
        self._validate()

    @property
    def backend(self):
        return self.cross.backend

    @property
    def time_s(self):
        return self.cross.time_s

    @property
    def updates(self):
        return self.cross.updates

    def _set_identity(self):
        self.identity = _digest({
            "scope": SCOPE, "cross": self.cross.identity,
            "topologies": {name: self.topologies[name].identity
                           for name in LEVELS},
            "backend": self.backend,
        })

    def _validate(self):
        self.cross._validate_accepted()
        clocks = {(ledger.time_s, ledger.updates)
                  for ledger in self.ledgers.values()}
        if clocks != {(self.time_s, self.updates)}:
            raise RecursivePhysicalToeV116Error(
                "Physical-toe ledgers and v115 clocks differ")

    def begin(self, parent, **kwargs):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "A recursive physical-toe macro is already open")
        checkpoint = self.checkpoint()
        result = self.cross.begin(parent, **kwargs)
        self.pending = {"checkpoint": checkpoint, "prepared": False}
        return result

    def consume_surface(self, name):
        return self.cross.consume_surface(name)

    def consume_bedform(self, name):
        return self.cross.consume_bedform(name)

    def prepare(self, candidate, **kwargs):
        if self.pending is None or self.pending["prepared"]:
            raise RecursivePhysicalToeV116Error(
                "Recursive physical-toe macro cannot be prepared")
        try:
            targets = self.cross.prepare(candidate, **kwargs)
            increments = {}
            for name in LEVELS:
                increments[name] = self.ledgers[name].propose(
                    self.cross.banks[name]._trial,
                    self.cross.pending["dt_s"])
            self.pending.update(prepared=True, increments=increments)
            return targets
        except BaseException:
            for ledger in self.ledgers.values():
                ledger.rollback()
            if self.cross.pending is not None:
                self.cross.rollback()
            self.pending = None
            raise

    def commit(self, *, cancelled=False, **kwargs):
        if self.pending is None or not self.pending["prepared"]:
            raise RecursivePhysicalToeV116Error(
                "Recursive physical-toe macro has no prepared trial")
        pending = self.pending
        checkpoint = pending["checkpoint"]
        try:
            cross_receipt, state = self.cross.commit(
                cancelled=cancelled, **kwargs)
            for ledger in self.ledgers.values():
                ledger.commit()
            self._validate()
            signed = np.zeros(self.cross.banks["coarse"].state[
                "layers"].shape[1], dtype=np.float64)
            gross = np.zeros_like(signed)
            for name in LEVELS:
                delta = np.asarray(host(pending["increments"][name]))
                orientation = self.topologies[name].orientation
                signed += np.sum(delta*orientation[None, :], axis=1)
                gross += np.sum(np.abs(delta), axis=1)
            epochs = {item.topology_epoch for item in self.topologies.values()}
            payload = {
                "backend": self.backend,
                "macro_dt_s": cross_receipt.macro_dt_s,
                "cross_physics_receipt_sha256":
                    cross_receipt.receipt_sha256,
                "accepted_state_sha256":
                    state["joined"].state_sha256,
                "topology_epoch": max(epochs),
                "active_toe_face_count": sum(
                    item.bank_cell.size for item in self.topologies.values()),
                "toe_signed_increment_kg": tuple(float(x) for x in signed),
                "toe_gross_increment_kg": tuple(float(x) for x in gross),
                "collapse_event_count": cross_receipt.collapse_event_count,
                "accepted_bed_layer_owner_count":
                    cross_receipt.accepted_bed_layer_owner_count,
                "automatic_cpu_fallback": False,
                "scope": SCOPE,
            }
            receipt = RecursivePhysicalToeReceiptV116(
                **payload, receipt_sha256=_digest(payload))
            self.pending = None
            return receipt, state
        except BaseException:
            self._restore_checkpoint(checkpoint)
            self.pending = None
            raise

    def rollback(self):
        if self.pending is None:
            raise RecursivePhysicalToeV116Error(
                "No recursive physical-toe macro is open")
        if self.cross.pending is not None:
            self.cross.rollback()
        for ledger in self.ledgers.values():
            ledger.rollback()
        self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "Physical-toe checkpoint requires committed state")
        cross = self.cross.checkpoint()
        ledgers = {name: ledger.checkpoint()
                   for name, ledger in self.ledgers.items()}
        maps = {name: item.document()
                for name, item in self.topologies.items()}
        seals = {
            "cross": cross["checkpoint_sha256"],
            "ledgers": {name: item["checkpoint_sha256"]
                        for name, item in ledgers.items()},
            "maps": {name: item["identity"] for name, item in maps.items()},
        }
        meta = {"scope": SCOPE, "identity": self.identity,
                "backend": self.backend, "seals": seals}
        return {**meta, "cross": cross, "ledgers": ledgers,
                "maps": maps, "checkpoint_sha256": _digest(meta)}

    def _restore_checkpoint(self, value):
        if self.cross.pending is not None:
            self.cross.rollback()
        for ledger in self.ledgers.values():
            ledger.rollback()
        self.cross.restore(value["cross"])
        for name in LEVELS:
            self.ledgers[name].restore(value["ledgers"][name])

    def restore(self, value):
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "Physical-toe restore requires committed state")
        maps = {name: SurveyedPhysicalToeMapV116.from_document(
                    value.get("maps", {}).get(name, {}))
                for name in LEVELS}
        seals = {
            "cross": value.get("cross", {}).get("checkpoint_sha256"),
            "ledgers": {name: value.get("ledgers", {}).get(name, {}).get(
                "checkpoint_sha256") for name in LEVELS},
            "maps": {name: maps[name].identity for name in LEVELS},
        }
        meta = {"scope": value.get("scope"),
                "identity": value.get("identity"),
                "backend": value.get("backend"), "seals": seals}
        if (value.get("scope") != SCOPE or
                value.get("identity") != self.identity or
                value.get("backend") != self.backend or
                value.get("seals") != seals or
                value.get("checkpoint_sha256") != _digest(meta) or
                any(maps[name].identity != self.topologies[name].identity
                    for name in LEVELS)):
            raise RecursivePhysicalToeV116Error(
                "Recursive physical-toe checkpoint identity differs")
        old = self.checkpoint()
        try:
            self._restore_checkpoint(value)
            self._validate()
        except BaseException:
            self._restore_checkpoint(old)
            raise
        return self

    def rebuild_maps(self, topologies):
        """Install a rebuilt face identity and retain matching diagnostics.

        Cell ownership must be unchanged here.  A cell-layout change belongs
        to the conservative M18 regrid transaction.
        """
        if self.pending is not None:
            raise RecursivePhysicalToeV116Error(
                "Toe-map rebuild requires committed state")
        if not isinstance(topologies, Mapping) or set(topologies) != set(LEVELS):
            raise RecursivePhysicalToeV116Error(
                "Toe-map rebuild requires both levels")
        new_maps = {name: topologies[name].validate() for name in LEVELS}
        new_ledgers = {}
        for name in LEVELS:
            old_map = self.topologies[name]
            new_map = new_maps[name]
            if (new_map.cell_count != old_map.cell_count or
                    not np.array_equal(new_map.bank_cell, old_map.bank_cell) or
                    not np.array_equal(new_map.toe_cell, old_map.toe_cell) or
                    not np.array_equal(new_map.face_length_m,
                                       old_map.face_length_m)):
                raise RecursivePhysicalToeV116Error(
                    "Cell-changing toe-map rebuild requires M18 regrid")
            old = self.ledgers[name]
            new = PhysicalToeFaceLedgerV116(
                new_map, old.grain_count, old.xp,
                time_s=old.time_s, updates=old.updates)
            old_by_id = {int(face): i
                         for i, face in enumerate(old_map.face_id)}
            for j, face in enumerate(new_map.face_id):
                i = old_by_id.get(int(face))
                if i is None:
                    continue
                factor = int(new_map.orientation[j])/int(old_map.orientation[i])
                new.signed_kg[:, int(face)] = old.signed_kg[:, int(face)]*factor
                new.gross_kg[:, int(face)] = old.gross_kg[:, int(face)]
            new_ledgers[name] = new
        self.topologies = new_maps
        self.ledgers = new_ledgers
        self._set_identity()
        self._validate()
        return self


__all__ = [
    "SCOPE", "SurveyedPhysicalToeMapV116",
    "PhysicalToeFaceLedgerV116", "RecursivePhysicalToeReceiptV116",
    "CoupledAMRRecursivePhysicalToeV116",
    "RecursivePhysicalToeV116Error",
]
