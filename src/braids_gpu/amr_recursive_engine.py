"""Shared 2:1 recursive level scheduling for SSPRK33 AMR steppers.

The scheduler owns only clock traversal and accepted face-register assembly.
Level-local SSPRK33 work and reflux remain callbacks so numerical state,
diagnostics, and transaction rollback stay under the calling stepper's control.
The supplied arrays may be NumPy or device arrays; this module never imports a
numerical backend or copies an integral between host and device.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any


class RecursiveLevelScheduler:
    """Run an arbitrary-depth, two-child 2:1 level schedule.

    ``level_integrals[level]`` is written by ``step_level`` and represents one
    SSPRK33 step at that level. ``subtree_integrals[level]`` is reusable scratch
    storage for the time-integrated result of that level and all descendants.
    ``own_faces[level]`` selects the faces whose final register value belongs to
    that level. All four level sequences use root-to-finest ordering.

    The scheduler is intentionally state-transaction agnostic. A callback that
    raises stops traversal immediately, and the owning stepper is responsible
    for rolling back its state and counters.
    """

    def __init__(
        self,
        *,
        level_has_work: Sequence[object],
        own_faces: Sequence[Any],
        level_integrals: Sequence[Any],
        subtree_integrals: Sequence[Any],
    ) -> None:
        active = tuple(bool(value) for value in level_has_work)
        faces = tuple(own_faces)
        level_values = tuple(level_integrals)
        subtree_values = tuple(subtree_integrals)
        counts = {len(active), len(faces), len(level_values), len(subtree_values)}
        if counts == {0} or len(counts) != 1:
            raise ValueError("Recursive level schedule arrays must have one common nonzero length")

        for level, (indices, value, scratch) in enumerate(
            zip(faces, level_values, subtree_values, strict=True)
        ):
            dtype = getattr(indices, "dtype", None)
            if (getattr(indices, "ndim", None) != 1 or
                    getattr(dtype, "kind", None) not in ("i", "u")):
                raise TypeError(f"Recursive level {level} own-face indices must be a 1-D integer array")
            if not callable(getattr(scratch, "fill", None)):
                raise TypeError(f"Recursive level {level} subtree integral is not mutable array storage")
            if value is scratch:
                raise ValueError(f"Recursive level {level} and subtree integrals must use distinct storage")
            if getattr(value, "shape", None) != getattr(scratch, "shape", None):
                raise ValueError(f"Recursive level {level} integral shapes differ")

        self._level_has_work = active
        self._own_faces = faces
        self._level_integrals = level_values
        self._subtree_integrals = subtree_values
        self._running = False

    @property
    def nlevels(self) -> int:
        """Number of scheduled levels, including intentionally empty levels."""
        return len(self._level_has_work)

    def advance(
        self,
        dt: float,
        current: float,
        step_level: Callable[[int, float, float], None],
        reflux: Callable[[int, Any, Any], None],
    ) -> Any:
        """Advance one root step and return its reusable subtree integral.

        Each non-leaf visit performs exactly two child steps of ``dt / 2``.
        Reflux follows both child steps, then own-face values replace the
        accumulated descendant register. Empty levels skip ``step_level`` but
        retain the same recursive clocks and reflux ordering.
        """
        try:
            dt = float(dt)
            current = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Recursive level clock must be finite with positive dt") from exc
        if not isfinite(dt) or dt <= 0.0 or not isfinite(current):
            raise ValueError("Recursive level clock must be finite with positive dt")
        if not callable(step_level) or not callable(reflux):
            raise TypeError("Recursive level step and reflux callbacks must be callable")
        if self._running:
            raise RuntimeError("Recursive level scheduler cannot be advanced reentrantly")

        self._running = True
        try:
            return self._advance_level(0, dt, current, step_level, reflux)
        finally:
            self._running = False

    def _advance_level(
        self,
        level: int,
        dt: float,
        current: float,
        step_level: Callable[[int, float, float], None],
        reflux: Callable[[int, Any, Any], None],
    ) -> Any:
        integral = self._subtree_integrals[level]
        integral.fill(0.0)
        if self._level_has_work[level]:
            step_level(level, dt, current)

        child_level = level + 1
        if child_level < self.nlevels:
            child_dt = 0.5 * dt
            for substep in range(2):
                child = self._advance_level(
                    child_level,
                    child_dt,
                    current + substep * child_dt,
                    step_level,
                    reflux,
                )
                integral += child
            reflux(level, self._level_integrals[level], integral)

        own = self._own_faces[level]
        integral[own] = self._level_integrals[level][own]
        return integral
