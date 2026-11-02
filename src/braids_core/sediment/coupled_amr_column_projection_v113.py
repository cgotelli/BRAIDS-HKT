"""Project one accepted joined-AMR bed update into both layered columns.

The joined Q4 owner has already selected and integrated bedload and suspended
exchange.  This module performs no transport, exchange, Exner, remap, or
capacity calculation.  It only applies the difference between two accepted
joined-leaf states to the authoritative Q5 column inventories exactly once.

The two levels are prepared and validated before either column is published.
If publication, cancellation, or an injected outer-owner check fails, both
levels are restored to their byte-identical checkpoints.  An outer coupled
owner can therefore prepare the Q4 candidate, call ``begin()``, publish the Q4
candidate, and then call ``commit()``; its existing transport checkpoint owns
recovery if publication after the Q4 commit fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from .bed_columns_v101 import host


SCOPE = "braids-q4-q5-joined-amr-column-projection-v113"


class CoupledAMRColumnProjectionV113Error(ValueError):
    """The accepted joined-AMR state or layered projection is inconsistent."""


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


def _leaf(joined, level):
    name = f"{level}_numerical_parent"
    value = getattr(joined, name, None)
    if value is None:
        raise CoupledAMRColumnProjectionV113Error(
            f"Joined state has no {name}")
    return value


def _stock(leaf):
    try:
        value = leaf.active_state.base_state.bed_mass_kg_m2
    except AttributeError as error:
        raise CoupledAMRColumnProjectionV113Error(
            "Joined leaf has no authoritative Q4 bed stock") from error
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or np.any(value < 0.0):
        raise CoupledAMRColumnProjectionV113Error(
            "Joined leaf bed stock is invalid")
    return value


def _depth(leaf):
    value = np.asarray(leaf.q[0], dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or np.any(value < 0.0):
        raise CoupledAMRColumnProjectionV113Error(
            "Joined leaf depth is invalid")
    return value


def _bed(leaf):
    value = np.asarray(leaf.bed_elevation_m, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise CoupledAMRColumnProjectionV113Error(
            "Joined leaf bed geometry is invalid")
    return value


def _column_bed_from_pending(columns):
    out, solid_fraction = columns.pending[:2]
    xp = columns.xp
    density = xp.asarray(columns.runtime.rho)[None, :, None]
    volume = (out / density).sum(axis=1) / solid_fraction
    return columns.rock + volume.sum(axis=0)


def _cancelled(value):
    return bool(value() if callable(value) else value)


@dataclass(frozen=True)
class CoupledAMRColumnProjectionReceiptV113:
    backend: str
    macro_dt_s: float
    coarse_class_budget_residual_kg: tuple[float, ...]
    fine_class_budget_residual_kg: tuple[float, ...]
    coarse_projection_max_abs_kg_m2: float
    fine_projection_max_abs_kg_m2: float
    coarse_bed_feedback_max_abs_m: float
    fine_bed_feedback_max_abs_m: float
    transport_increment_owner_count: int
    coarse_layer_owner_count: int
    fine_layer_owner_count: int
    scope: str
    receipt_sha256: str


class CoupledAMRColumnProjectionV113:
    """A two-level prepare/commit transaction for one accepted Q4 candidate."""

    def __init__(self, coarse_columns, fine_columns, parent, candidate,
                 *, resident_deltas=None, column_inputs=None):
        self.coarse_columns = coarse_columns
        self.fine_columns = fine_columns
        self.parent = parent
        self.candidate = candidate
        self.resident_deltas = resident_deltas
        if column_inputs is None:
            column_inputs = {"coarse": {}, "fine": {}}
        if (not isinstance(column_inputs, Mapping) or
                set(column_inputs) != {"coarse", "fine"} or
                any(not isinstance(value, Mapping)
                    for value in column_inputs.values())):
            raise CoupledAMRColumnProjectionV113Error(
                "Column inputs require coarse and fine mappings")
        self.column_inputs = {
            key: dict(value) for key, value in column_inputs.items()}
        self.pending = None
        self.receipt = None
        self.composed_targets = None
        self.composed_receipts = None
        self._validate_transition()

    @property
    def columns(self):
        return self.coarse_columns, self.fine_columns

    @property
    def backend(self):
        backends = tuple(column.runtime.backend for column in self.columns)
        if len(set(backends)) != 1:
            raise CoupledAMRColumnProjectionV113Error(
                "Both AMR column levels must use the same backend")
        return backends[0]

    def _validate_transition(self):
        parent_time = float(self.parent.time_s)
        candidate_time = float(self.candidate.time_s)
        if (not np.isfinite((parent_time, candidate_time)).all() or
                candidate_time <= parent_time):
            raise CoupledAMRColumnProjectionV113Error(
                "Joined AMR projection requires one positive accepted interval")
        if (hasattr(self.parent, "accepted_macro_updates") and
                hasattr(self.candidate, "accepted_macro_updates") and
                self.candidate.accepted_macro_updates !=
                self.parent.accepted_macro_updates + 1):
            raise CoupledAMRColumnProjectionV113Error(
                "Joined AMR projection requires exactly one macro update")
        for level, columns in zip(("coarse", "fine"), self.columns):
            columns.validate()
            leaf = _leaf(self.parent, level)
            if columns.pending is not None:
                raise CoupledAMRColumnProjectionV113Error(
                    f"{level} column already has an open trial")
            if columns.time_s != parent_time:
                raise CoupledAMRColumnProjectionV113Error(
                    f"{level} column clock differs from the joined parent")
            if not np.allclose(host(columns.total), _stock(leaf),
                               rtol=0.0, atol=1.0e-12):
                raise CoupledAMRColumnProjectionV113Error(
                    f"{level} column stock differs from the joined parent")
            if not np.allclose(host(columns.bed), _bed(leaf),
                               rtol=0.0, atol=1.0e-12):
                raise CoupledAMRColumnProjectionV113Error(
                    f"{level} column bed differs from the joined parent")
        if self.resident_deltas is None and self.backend.endswith("gpu"):
            # The accepted Q4 states are immutable host receipts.  GPU callers
            # explicitly stage their accepted delta on device; the column
            # operation then remains an actual CUDA kernel and never silently
            # executes the CPU numerical implementation.
            raise CoupledAMRColumnProjectionV113Error(
                "GPU AMR column projection requires resident Q4 deltas")

    def checkpoint(self):
        if self.pending is not None:
            raise CoupledAMRColumnProjectionV113Error(
                "Checkpoint requires accepted AMR columns")
        raw = {
            "scope": SCOPE,
            "parent_state_sha256": getattr(self.parent, "state_sha256", None),
            "candidate_state_sha256": getattr(
                self.candidate, "state_sha256", None),
            "coarse": self.coarse_columns.checkpoint(),
            "fine": self.fine_columns.checkpoint(),
        }
        return {**raw, "checkpoint_sha256": _digest(raw)}

    def restore(self, checkpoint):
        if self.pending is not None:
            raise CoupledAMRColumnProjectionV113Error(
                "Restore requires no open AMR column trial")
        raw = {key: value for key, value in checkpoint.items()
               if key != "checkpoint_sha256"}
        if (_digest(raw) != checkpoint.get("checkpoint_sha256") or
                checkpoint.get("scope") != SCOPE or
                checkpoint.get("parent_state_sha256") !=
                getattr(self.parent, "state_sha256", None) or
                checkpoint.get("candidate_state_sha256") !=
                getattr(self.candidate, "state_sha256", None)):
            raise CoupledAMRColumnProjectionV113Error(
                "AMR column checkpoint identity differs")
        old = (self.coarse_columns.checkpoint(),
               self.fine_columns.checkpoint())
        try:
            self.coarse_columns.restore(checkpoint["coarse"])
            self.fine_columns.restore(checkpoint["fine"])
        except BaseException:
            self.coarse_columns.restore(old[0])
            self.fine_columns.restore(old[1])
            raise

    def begin(self, *, cancelled=False):
        if self.pending is not None:
            raise CoupledAMRColumnProjectionV113Error(
                "AMR column projection is already open")
        if _cancelled(cancelled):
            raise InterruptedError("AMR column projection cancelled")
        self._validate_transition()
        checkpoints = (self.coarse_columns.checkpoint(),
                       self.fine_columns.checkpoint())
        parent_leaves = tuple(_leaf(self.parent, name)
                              for name in ("coarse", "fine"))
        candidate_leaves = tuple(_leaf(self.candidate, name)
                                 for name in ("coarse", "fine"))
        host_deltas = tuple(_stock(after) - _stock(before)
                            for before, after in
                            zip(parent_leaves, candidate_leaves))
        deltas = host_deltas
        if self.resident_deltas is not None:
            if len(self.resident_deltas) != 2:
                raise CoupledAMRColumnProjectionV113Error(
                    "Resident AMR projection requires coarse and fine deltas")
            deltas = self.resident_deltas
            for resident, expected in zip(deltas, host_deltas):
                if not np.array_equal(host(resident), expected):
                    raise CoupledAMRColumnProjectionV113Error(
                        "Resident Q4 delta differs from accepted joined states")
        dt_s = float(self.candidate.time_s - self.parent.time_s)
        try:
            for level, columns, delta, leaf in zip(
                    ("coarse", "fine"), self.columns, deltas,
                    candidate_leaves):
                columns.begin(delta, _depth(leaf), dt_s,
                              **self.column_inputs[level])
            if _cancelled(cancelled):
                raise InterruptedError("AMR column projection cancelled")
            for level, columns, leaf in zip(
                    ("coarse", "fine"), self.columns, candidate_leaves):
                projected = host(columns.pending[0].sum(axis=0))
                target = _stock(leaf)
                if not np.allclose(projected, target,
                                   rtol=0.0, atol=1.0e-12):
                    raise CoupledAMRColumnProjectionV113Error(
                        f"{level} accepted bed increment was duplicated")
                if not np.allclose(host(_column_bed_from_pending(columns)),
                                   _bed(leaf), rtol=0.0, atol=1.0e-12):
                    raise CoupledAMRColumnProjectionV113Error(
                        f"{level} layered bed feedback differs from Q4")
                columns.validate_pending()
            self.pending = checkpoints
            return self
        except BaseException:
            for columns in self.columns:
                columns.rollback()
            raise

    def compose_pending_layers(self, levels, *, cancelled=False):
        """Compose absolute bank/repose/fines layers into this open trial.

        The Q4 candidate authenticates the transport part.  A v115 outer
        owner may advance split physical processes from the unpublished Q5
        layers, then replace both level trials here.  The accepted columns and
        clocks are still published only by :meth:`commit`.
        """
        if self.pending is None or self.composed_targets is not None:
            raise CoupledAMRColumnProjectionV113Error(
                "AMR column projection cannot compose split-process layers")
        if _cancelled(cancelled):
            raise InterruptedError("AMR split-process composition cancelled")
        if not isinstance(levels, Mapping) or set(levels)!={"coarse","fine"}:
            raise CoupledAMRColumnProjectionV113Error(
                "Split-process layers require coarse and fine mappings")
        targets={};receipts={}
        try:
            for name,columns in zip(("coarse","fine"),self.columns):
                value=levels[name]
                if (not isinstance(value,Mapping) or
                        set(value)!={"mass","solid_fraction"}):
                    raise CoupledAMRColumnProjectionV113Error(
                        f"{name} split-process state is incomplete")
                receipts[name]=columns.compose_pending_layers(
                    value["mass"],value["solid_fraction"],cancelled=cancelled)
                columns.validate_pending()
                targets[name]={
                    "stock":np.ascontiguousarray(
                        host(columns.pending[0].sum(axis=0))),
                    "bed":np.ascontiguousarray(
                        host(_column_bed_from_pending(columns))),
                }
            self.composed_targets=targets
            self.composed_receipts=receipts
            return targets
        except BaseException:
            for columns in self.columns:
                columns.rollback()
            self.pending=None
            self.composed_targets=None
            self.composed_receipts=None
            raise

    def commit(self, *, cancelled=False, after_coarse_publish=None):
        if self.pending is None:
            raise CoupledAMRColumnProjectionV113Error(
                "No AMR column projection is open")
        checkpoints = self.pending
        if _cancelled(cancelled):
            self.rollback()
            raise InterruptedError("AMR column publication cancelled")
        try:
            residuals = [self.coarse_columns.commit()]
            if after_coarse_publish is not None:
                after_coarse_publish()
            residuals.append(self.fine_columns.commit())
            self.pending = None
            self.receipt = self._make_receipt(residuals)
            return self.receipt
        except BaseException:
            for columns in self.columns:
                if columns.pending is not None:
                    columns.rollback()
            self.coarse_columns.restore(checkpoints[0])
            self.fine_columns.restore(checkpoints[1])
            self.pending = None
            raise

    def rollback(self):
        if self.pending is None:
            raise CoupledAMRColumnProjectionV113Error(
                "No AMR column projection is open")
        for columns in self.columns:
            columns.rollback()
        self.pending = None
        self.composed_targets = None
        self.composed_receipts = None

    def _make_receipt(self, residuals):
        values = {}
        for level, columns, leaf in zip(
                ("coarse", "fine"), self.columns,
                (_leaf(self.candidate, "coarse"),
                 _leaf(self.candidate, "fine"))):
            target=(None if self.composed_targets is None else
                    self.composed_targets[level])
            target_stock=_stock(leaf) if target is None else target["stock"]
            target_bed=_bed(leaf) if target is None else target["bed"]
            values[f"{level}_projection_max_abs_kg_m2"] = float(
                np.max(np.abs(host(columns.total) - target_stock)))
            values[f"{level}_bed_feedback_max_abs_m"] = float(
                np.max(np.abs(host(columns.bed) - target_bed)))
        payload = {
            "backend": self.backend,
            "macro_dt_s": float(self.candidate.time_s-self.parent.time_s),
            "coarse_class_budget_residual_kg": tuple(
                float(x) for x in host(residuals[0])),
            "fine_class_budget_residual_kg": tuple(
                float(x) for x in host(residuals[1])),
            **values,
            "transport_increment_owner_count": 1,
            "coarse_layer_owner_count": 1,
            "fine_layer_owner_count": 1,
            "scope": SCOPE,
        }
        return CoupledAMRColumnProjectionReceiptV113(
            **payload, receipt_sha256=_digest(payload))


__all__ = [
    "SCOPE",
    "CoupledAMRColumnProjectionV113Error",
    "CoupledAMRColumnProjectionReceiptV113",
    "CoupledAMRColumnProjectionV113",
]
