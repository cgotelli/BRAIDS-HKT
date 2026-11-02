"""Recursive Q5/Q7b/Q8b cross-physics composition for joined AMR.

The accepted Q5b columns are the sole bed-layer owner.  Q3/Q4 first prepare
their transport update without publishing it.  Q8b then advances bank toe
erosion, collapse, parent-owned repose and fines from those pending layers.
The final Q8b layer result replaces the pending Q5 trial and the outer owner
publishes Q5, Q8b and Q7b atomically.  The next macro therefore consumes both
the newly sorted active surface and newly accepted dynamic-bedform feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .bed_columns_v101 import array_sha, host
from .dynamic_bedforms_v104 import FEEDBACK_CONSUMERS
from .recursive_surface_feedback_v114 import (
    ALL_SURFACE_CONSUMERS,
    CoupledAMRRecursiveSurfaceFeedbackV114,
)


SCOPE = "braids-recursive-cross-physics-v115"
LEVELS = ("coarse", "fine")


class RecursiveCrossPhysicsV115Error(ValueError):
    """A cross-physics owner, clock, backend, or transaction differs."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _cancelled(value):
    return bool(value() if callable(value) else value)


def _is_gpu(value):
    name = str(value).lower()
    return "gpu" in name or "cuda" in name


def _bank_total(state, runtime):
    area = np.asarray(host(state["area"]), dtype=np.float64)
    layers = np.asarray(host(state["layers"]), dtype=np.float64)
    water = np.asarray(host(state["suspended_mass_areal"]), dtype=np.float64)
    fluff = np.asarray(host(state["fluff_mass_areal"]), dtype=np.float64)
    return (np.sum(layers*area, axis=(0, 2, 3)) +
            np.sum(water*area, axis=(1, 2)) +
            np.sum(fluff*area, axis=(1, 2)))


class _Proxy:
    def __init__(self, source, **values):
        self._source = source
        self.__dict__.update(values)

    def __getattr__(self, name):
        return getattr(self._source, name)


@dataclass(frozen=True)
class AcceptedJoinedCrossPhysicsStateV115:
    """Joined accepted-state view carrying the final cross-physics bed."""

    coarse_numerical_parent: object
    fine_numerical_parent: object
    accepted_macro_updates: int
    time_s: float
    source_state_sha256: str | None
    state_sha256: str
    scope: str = SCOPE


def _accepted_joined(candidate, targets, component_sha):
    leaves = {}
    for name in LEVELS:
        leaf = getattr(candidate, f"{name}_numerical_parent")
        active = leaf.active_state
        base = active.base_state
        target = targets[name]
        base_view = _Proxy(
            base, bed_mass_kg_m2=np.ascontiguousarray(target["stock"]))
        active_view = _Proxy(active, base_state=base_view)
        leaves[name] = _Proxy(
            leaf, active_state=active_view,
            bed_elevation_m=np.ascontiguousarray(target["bed"]),
            time_s=float(candidate.time_s))
    payload = {
        "scope": SCOPE,
        "source_state_sha256": getattr(candidate, "state_sha256", None),
        "accepted_macro_updates": int(candidate.accepted_macro_updates),
        "time_s": float(candidate.time_s),
        "targets_sha256": array_sha(
            targets["coarse"]["stock"], targets["coarse"]["bed"],
            targets["fine"]["stock"], targets["fine"]["bed"]),
        "components_sha256": component_sha,
    }
    return AcceptedJoinedCrossPhysicsStateV115(
        leaves["coarse"], leaves["fine"],
        payload["accepted_macro_updates"], payload["time_s"],
        payload["source_state_sha256"], _digest(payload))


@dataclass(frozen=True)
class RecursiveCrossPhysicsReceiptV115:
    backend: str
    macro_dt_s: float
    surface_receipt_sha256: str
    accepted_state_sha256: str
    maximum_q5_projection_error_kg_m2: float
    maximum_bed_feedback_error_m: float
    maximum_bank_fines_budget_residual_kg: float
    maximum_bank_q5_mirror_error_kg_m2: float
    maximum_bedform_height_change_m: float
    wet_cell_count: int
    dry_cell_count: int
    collapse_event_count: int
    parent_avalanche_level_count: int
    active_surface_consumer_count: int
    active_bedform_consumer_count: int
    accepted_bed_layer_owner_count: int
    automatic_cpu_fallback: bool
    scope: str
    receipt_sha256: str


class CoupledAMRRecursiveCrossPhysicsV115:
    """Atomic two-level recursive surface, bedform, bank and fines owner."""

    def __init__(self, surface: CoupledAMRRecursiveSurfaceFeedbackV114,
                 banks, bedforms):
        if not isinstance(surface, CoupledAMRRecursiveSurfaceFeedbackV114):
            raise TypeError("Versioned recursive surface owner required")
        if (not isinstance(banks, Mapping) or set(banks) != set(LEVELS) or
                not isinstance(bedforms, Mapping) or
                set(bedforms) != set(LEVELS)):
            raise RecursiveCrossPhysicsV115Error(
                "Explicit coarse and fine bank/bedform owners are required")
        self.surface = surface
        self.banks = dict(banks)
        self.bedforms = dict(bedforms)
        self.pending = None
        self._validate_accepted()
        self.identity = _digest({
            "scope": SCOPE,
            "surface": {name: owner.identity
                        for name, owner in surface.levels.items()},
            "banks": {name: owner.runtime.identity
                      for name, owner in self.banks.items()},
            "bedforms": {name: owner.runtime.identity
                         for name, owner in self.bedforms.items()},
            "backend": self.backend,
        })

    @property
    def backend(self):
        value = self.surface.backend
        variants = {_is_gpu(value)}
        variants.update(_is_gpu(owner.runtime.backend)
                        for owner in self.banks.values())
        variants.update(_is_gpu(owner.runtime.backend)
                        for owner in self.bedforms.values())
        if len(variants) != 1:
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics owners differ from the explicit backend")
        return value

    @property
    def time_s(self):
        return self.surface.columns[0].time_s

    @property
    def updates(self):
        return self.surface.columns[0].updates

    def _validate_accepted(self):
        columns = dict(zip(LEVELS, self.surface.columns))
        clocks = set()
        updates = set()
        for name in LEVELS:
            column = columns[name]
            column.validate()
            bank = self.banks[name]
            bedform = self.bedforms[name]
            if bank._trial is not None or bedform.pending is not None:
                raise RecursiveCrossPhysicsV115Error(
                    f"{name} cross-physics owner has a pending trial")
            shape = bank.state["area"].shape
            if int(np.prod(shape)) != column.runtime.nc:
                raise RecursiveCrossPhysicsV115Error(
                    f"{name} Q8b topology and Q5 column count differ")
            comparisons = (
                (host(bank.state["layers"]).reshape(column.mass.shape),
                 host(column.mass), "layers"),
                (host(bank.state["solid_fraction"]).reshape(column.sv.shape),
                 host(column.sv), "solid fraction"),
                (host(bank.state["area"]).ravel(), host(column.area), "area"),
                (host(bank.state["rock"]).ravel(), host(column.rock), "rock"),
                (host(bedform.area).ravel(), host(column.area),
                 "bedform area"),
            )
            for actual, expected, label in comparisons:
                if not np.allclose(actual, expected, rtol=0.0, atol=1.e-12):
                    raise RecursiveCrossPhysicsV115Error(
                        f"{name} Q5/Q8b {label} mirror differs")
            clocks.update((float(column.time_s), float(bank.time_s),
                           float(bedform.time_s)))
            updates.update((int(column.updates), int(bank.updates),
                            int(bedform.updates)))
        if len(clocks) != 1 or len(updates) != 1:
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics accepted clocks or update counts differ")
        # Accessing this property performs the explicit CPU/GPU consistency
        # check after all component shapes have been verified.
        self.backend

    def begin(self, parent, *,
              active_surface_consumers=ALL_SURFACE_CONSUMERS,
              active_bedform_consumers=FEEDBACK_CONSUMERS,
              cancelled=False):
        if self.pending is not None:
            raise RecursiveCrossPhysicsV115Error(
                "A recursive cross-physics macro is already open")
        if _cancelled(cancelled):
            raise InterruptedError("Recursive cross-physics macro cancelled")
        self._validate_accepted()
        surface_consumers = tuple(active_surface_consumers)
        consumers = tuple(active_bedform_consumers)
        if (len(consumers) != len(set(consumers)) or
                any(name not in FEEDBACK_CONSUMERS for name in consumers)):
            raise RecursiveCrossPhysicsV115Error(
                "Bedform consumers are unknown or duplicated")
        checkpoint = self.checkpoint()
        surfaces = self.surface.begin(
            parent, active_consumers=surface_consumers,
            cancelled=cancelled)
        feedback = {name: owner.feedback()
                    for name, owner in self.bedforms.items()}
        self.pending = {
            "parent": parent,
            "checkpoint": checkpoint,
            "surfaces": surfaces,
            "bedform_before": feedback,
            "bedform_consumers": consumers,
            "bedform_counts": {name: 0 for name in consumers},
            "surface_consumer_count": len(surface_consumers),
            "prepared": False,
        }
        return {"surface": surfaces, "bedform": feedback}

    def consume_surface(self, name):
        if self.pending is None:
            raise RecursiveCrossPhysicsV115Error(
                "No recursive cross-physics macro is open")
        return self.surface.consume(name)

    def consume_bedform(self, name):
        if self.pending is None:
            raise RecursiveCrossPhysicsV115Error(
                "No recursive cross-physics macro is open")
        counts = self.pending["bedform_counts"]
        if name not in counts:
            raise RecursiveCrossPhysicsV115Error(
                f"Bedform consumer {name!r} is not active")
        if counts[name]:
            raise RecursiveCrossPhysicsV115Error(
                f"Bedform consumer {name!r} was applied more than once")
        counts[name] = 1
        return self.pending["bedform_before"]

    def _rollback_trials(self):
        if self.surface.pending is not None:
            self.surface.rollback()
        for owner in self.banks.values():
            owner.discard()
        for owner in self.bedforms.values():
            owner.rollback()

    def prepare(self, candidate, *, bank_forcings, bedform_forcings,
                resident_deltas=None, cancelled=False):
        if self.pending is None or self.pending["prepared"]:
            raise RecursiveCrossPhysicsV115Error(
                "Recursive cross-physics macro cannot be prepared")
        missing = [name for name, count in
                   self.pending["bedform_counts"].items() if count != 1]
        if missing:
            raise RecursiveCrossPhysicsV115Error(
                "Bedform consumers were not applied exactly once: " +
                ", ".join(missing))
        if (not isinstance(bank_forcings, Mapping) or
                set(bank_forcings) != set(LEVELS) or
                not isinstance(bedform_forcings, Mapping) or
                set(bedform_forcings) != set(LEVELS)):
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics forcings require coarse and fine mappings")
        dt_s = float(candidate.time_s-self.pending["parent"].time_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics macro interval must be positive")
        column_inputs = {}
        for name in LEVELS:
            feedback = self.pending["bedform_before"][name]
            column_inputs[name] = {
                "dune_height": feedback.height_m.reshape(-1),
                "dune_length": feedback.length_m.reshape(-1),
            }
        bank_residuals = {}
        collapse_events = {}
        try:
            self.surface.prepare(
                candidate, resident_deltas=resident_deltas,
                column_inputs=column_inputs, cancelled=cancelled)
            projection = self.surface.pending["projection"]
            levels = {}
            for name, column in zip(LEVELS, self.surface.columns):
                if _cancelled(cancelled):
                    raise InterruptedError(
                        f"Recursive cross-physics cancelled before {name}")
                bed_args = dict(bedform_forcings[name])
                bank_args = dict(bank_forcings[name])
                forbidden = set(bed_args).intersection({"dt", "dt_s"})
                forbidden |= set(bank_args).intersection(
                    {"dt", "dt_s", "time_s", "layers",
                     "solid_fraction"})
                if forbidden:
                    raise RecursiveCrossPhysicsV115Error(
                        "The recursive owner controls " +
                        ", ".join(sorted(forbidden)))
                self.bedforms[name].propose(dt_s=dt_s, **bed_args)
                bank = self.banks[name]
                shape = bank.state["area"].shape
                nl, ng = column.mass.shape[:2]
                pending_mass = column.pending[0].reshape((nl, ng)+shape)
                pending_solid = column.pending[1].reshape((nl,)+shape)
                input_state = dict(bank.state)
                input_state["layers"] = pending_mass
                input_state["solid_fraction"] = pending_solid
                before = _bank_total(input_state, bank.runtime)
                trial = bank.propose_from_column_state(
                    pending_mass, pending_solid, dt_s=dt_s, **bank_args)
                after = _bank_total(trial, bank.runtime)
                bank_residuals[name] = after-before
                collapse_events[name] = int(np.sum(
                    host(trial["collapse_events"])))
                levels[name] = {
                    "mass": trial["layers"].reshape(column.mass.shape),
                    "solid_fraction": trial["solid_fraction"].reshape(
                        column.sv.shape),
                }
            targets = projection.compose_pending_layers(
                levels, cancelled=cancelled)
            self.pending.update({
                "prepared": True,
                "candidate": candidate,
                "dt_s": dt_s,
                "targets": targets,
                "bank_residuals": bank_residuals,
                "collapse_events": collapse_events,
                "projection": projection,
            })
            return targets
        except BaseException:
            self._rollback_trials()
            self.pending = None
            raise

    def commit(self, *, cancelled=False, after_q5_publish=None,
               after_bank_publish=None):
        if self.pending is None or not self.pending["prepared"]:
            raise RecursiveCrossPhysicsV115Error(
                "Recursive cross-physics macro has no prepared trial")
        if _cancelled(cancelled):
            self.rollback()
            raise InterruptedError(
                "Recursive cross-physics publication cancelled")
        pending = self.pending
        checkpoint = pending["checkpoint"]
        targets = pending["targets"]
        try:
            surface_receipt, after_surface = self.surface.commit(
                cancelled=cancelled,
                after_coarse_publish=after_q5_publish)
            for name in LEVELS:
                self.banks[name].commit()
            if after_bank_publish is not None:
                after_bank_publish()
            for name in LEVELS:
                self.bedforms[name].commit()
            self._validate_accepted()
            after_bedform = {name: owner.feedback()
                             for name, owner in self.bedforms.items()}
            component_sha = _digest({
                "surface": surface_receipt.receipt_sha256,
                "banks": {name: owner.state_sha256
                          for name, owner in self.banks.items()},
                "bedforms": {name: owner.state_sha256
                             for name, owner in self.bedforms.items()},
            })
            accepted = _accepted_joined(
                pending["candidate"], targets, component_sha)
            mirror_error = 0.0
            height_change = 0.0
            wet = dry = collapse = avalanche = 0
            for name, column in zip(LEVELS, self.surface.columns):
                bank = self.banks[name]
                mirror_error = max(mirror_error, float(np.max(np.abs(
                    host(bank.state["layers"]).reshape(column.mass.shape)-
                    host(column.mass)))))
                height_change = max(height_change, float(np.max(np.abs(
                    host(after_bedform[name].height_m)-
                    host(pending["bedform_before"][name].height_m)))))
                depth = np.asarray(host(bank.state["depth"]))
                wet += int(np.count_nonzero(depth > 0.0))
                dry += int(np.count_nonzero(depth <= 0.0))
                collapse += pending["collapse_events"][name]
                # Commit clears _trial; use the cumulative accepted count as
                # evidence that the bank/repose route remained active.
                avalanche += int(bank.runtime.methods.avalanche)
            max_bank_residual = max(
                abs(float(value))
                for values in pending["bank_residuals"].values()
                for value in values)
            payload = {
                "backend": self.backend,
                "macro_dt_s": pending["dt_s"],
                "surface_receipt_sha256": surface_receipt.receipt_sha256,
                "accepted_state_sha256": accepted.state_sha256,
                "maximum_q5_projection_error_kg_m2":
                    surface_receipt.maximum_column_projection_error_kg_m2,
                "maximum_bed_feedback_error_m":
                    surface_receipt.maximum_bed_feedback_error_m,
                "maximum_bank_fines_budget_residual_kg":max_bank_residual,
                "maximum_bank_q5_mirror_error_kg_m2":mirror_error,
                "maximum_bedform_height_change_m":height_change,
                "wet_cell_count":wet,
                "dry_cell_count":dry,
                "collapse_event_count":collapse,
                "parent_avalanche_level_count":avalanche,
                "active_surface_consumer_count":pending[
                    "surface_consumer_count"],
                "active_bedform_consumer_count":len(
                    pending["bedform_consumers"]),
                "accepted_bed_layer_owner_count":1,
                "automatic_cpu_fallback":False,
                "scope":SCOPE,
            }
            receipt = RecursiveCrossPhysicsReceiptV115(
                **payload, receipt_sha256=_digest(payload))
            self.pending = None
            return receipt, {
                "surface": after_surface,
                "bedform": after_bedform,
                "joined": accepted,
            }
        except BaseException:
            self._restore_checkpoint(checkpoint)
            self.pending = None
            raise

    def rollback(self):
        if self.pending is None:
            raise RecursiveCrossPhysicsV115Error(
                "No recursive cross-physics macro is open")
        self._rollback_trials()
        self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics checkpoint requires committed state")
        surface = self.surface.checkpoint()
        banks = {name: owner.checkpoint()
                 for name, owner in self.banks.items()}
        bedforms = {name: owner.checkpoint()
                    for name, owner in self.bedforms.items()}
        seals = {
            "surface": surface["checkpoint_sha256"],
            "banks": {name: value["state_sha256"]
                      for name, value in banks.items()},
            "bedforms": {name: value["state_sha256"]
                         for name, value in bedforms.items()},
        }
        meta = {"scope":SCOPE, "identity":getattr(self,"identity",None),
                "backend":self.backend, "seals":seals}
        return {**meta, "surface":surface, "banks":banks,
                "bedforms":bedforms, "checkpoint_sha256":_digest(meta)}

    def _restore_checkpoint(self, value):
        # Internal recovery trusts the snapshot created by this live owner;
        # every child restore still authenticates its own arrays and identity.
        if self.surface.pending is not None:
            self.surface.rollback()
        self.surface.restore(value["surface"])
        for name in LEVELS:
            if self.banks[name]._trial is not None:
                self.banks[name].discard()
            if self.bedforms[name].pending is not None:
                self.bedforms[name].rollback()
            self.banks[name].restore(value["banks"][name])
            self.bedforms[name].restore(value["bedforms"][name])

    def restore(self, value):
        if self.pending is not None:
            raise RecursiveCrossPhysicsV115Error(
                "Cross-physics restore requires committed state")
        seals = {
            "surface": value.get("surface", {}).get("checkpoint_sha256"),
            "banks": {name: value.get("banks", {}).get(name, {}).get(
                "state_sha256") for name in LEVELS},
            "bedforms": {name: value.get("bedforms", {}).get(name, {}).get(
                "state_sha256") for name in LEVELS},
        }
        meta = {"scope":value.get("scope"),
                "identity":value.get("identity"),
                "backend":value.get("backend"), "seals":seals}
        if (value.get("scope") != SCOPE or
                value.get("identity") != self.identity or
                value.get("backend") != self.backend or
                value.get("seals") != seals or
                value.get("checkpoint_sha256") != _digest(meta)):
            raise RecursiveCrossPhysicsV115Error(
                "Recursive cross-physics checkpoint identity differs")
        old = self.checkpoint()
        try:
            self._restore_checkpoint(value)
            self._validate_accepted()
        except BaseException:
            self._restore_checkpoint(old)
            raise
        return self


__all__ = [
    "SCOPE", "AcceptedJoinedCrossPhysicsStateV115",
    "CoupledAMRRecursiveCrossPhysicsV115",
    "RecursiveCrossPhysicsReceiptV115",
    "RecursiveCrossPhysicsV115Error",
]
