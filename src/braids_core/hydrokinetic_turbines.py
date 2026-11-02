"""Research hydrokinetic-turbine actuator source for depth-averaged flow.

The public Desktop project does not import this module yet.  It deliberately
contains the independently testable pieces needed before UI admission:

* an explicit, strict turbine definition;
* a locator-polyline to physical rotor-plane transformation;
* exact overlap weights for polygonal/cropped finite volumes; and
* the Kramer--Piggott actuator-disc correction applied as an integrated
  shallow-water momentum source.

The locator line gives position and rotor-plane orientation only.  Its drawn
length is never interpreted as rotor diameter.  The derived rotor line is a
permeable mesh constraint, not a wall.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree


SCHEMA = "braids.hydrokinetic-turbines.research.v1"
METHOD = "depth_averaged_actuator_disc.kramer_piggott_2016.v1"


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(name + " must be positive")
    if nonnegative and value < 0:
        raise ValueError(name + " must be nonnegative")
    return value


def _exact(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(name + " requires exactly: " + ", ".join(sorted(fields)))


@dataclass(frozen=True)
class TurbineDefinition:
    turbine_id: str
    locator_xy: tuple[tuple[float, float], ...]
    center_xy: tuple[float, float]
    tangent_xy: tuple[float, float]
    normal_xy: tuple[float, float]
    rotor_diameter_m: float
    hub_height_m: float
    thrust_coefficient: float
    power_coefficient: float
    support_frontal_area_m2: float
    support_drag_coefficient: float
    control: dict

    @property
    def rotor_area_m2(self):
        return math.pi * self.rotor_diameter_m**2 / 4.0

    @property
    def combined_drag_area_m2(self):
        """Return At*Ct + As*Cs, the quadratic-force area."""
        return (self.rotor_area_m2 * self.thrust_coefficient +
                self.support_frontal_area_m2 * self.support_drag_coefficient)

    @property
    def rotor_line_xy(self):
        c = np.asarray(self.center_xy)
        t = np.asarray(self.tangent_xy) * (0.5 * self.rotor_diameter_m)
        return tuple(map(tuple, (c - t, c + t)))


def _locator(points, turbine_id):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError(turbine_id + ": locator polyline needs at least two finite points")
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0):
        raise ValueError(turbine_id + ": locator polyline has repeated consecutive points")
    line = LineString(xy)
    if not line.is_simple:
        raise ValueError(turbine_id + ": locator polyline cannot self-intersect")
    chord = xy[-1] - xy[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 0:
        raise ValueError(turbine_id + ": locator endpoints must differ")
    tangent = chord / chord_length
    # The line is an orientation marker, not a curved physical object.  Permit
    # ordinary hand-click noise while rejecting a locator whose orientation is
    # ambiguous.  A two-point locator has exactly zero deviation.
    normal = np.array([-tangent[1], tangent[0]])
    deviation = float(np.max(np.abs((xy - xy[0]) @ normal)))
    if deviation > 0.05 * chord_length:
        raise ValueError(turbine_id + ": locator must be approximately straight; use a two-point line")
    center = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
    return (tuple(map(tuple, xy)), tuple(center), tuple(tangent), tuple(normal))


def _control(value, turbine_id):
    if not isinstance(value, dict) or "mode" not in value:
        raise ValueError(turbine_id + ": an explicit turbine control is required")
    mode = value["mode"]
    if mode == "constant_ct":
        _exact(value, ("mode", "thrust_coefficient", "power_coefficient"),
               turbine_id + ".control")
        ct = _number(value["thrust_coefficient"], turbine_id + ".thrust_coefficient",
                     nonnegative=True)
        cp = _number(value["power_coefficient"], turbine_id + ".power_coefficient",
                     nonnegative=True)
        return ct, cp, dict(mode=mode, thrust_coefficient=ct, power_coefficient=cp)
    if mode == "prescribed_tsr_table":
        _exact(value, ("mode", "tip_speed_ratio", "performance"), turbine_id + ".control")
        tsr = _number(value["tip_speed_ratio"], turbine_id + ".tip_speed_ratio", positive=True)
        rows = value["performance"]
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError(turbine_id + ": TSR performance table needs at least two rows")
        table = []
        for i, row in enumerate(rows):
            _exact(row, ("tip_speed_ratio", "thrust_coefficient", "power_coefficient"),
                   f"{turbine_id}.performance[{i}]")
            table.append((_number(row["tip_speed_ratio"], "table TSR", positive=True),
                          _number(row["thrust_coefficient"], "table Ct", nonnegative=True),
                          _number(row["power_coefficient"], "table Cp", nonnegative=True)))
        table.sort()
        if any(a[0] == b[0] for a, b in zip(table, table[1:])):
            raise ValueError(turbine_id + ": TSR table abscissae must be unique")
        if not table[0][0] <= tsr <= table[-1][0]:
            raise ValueError(turbine_id + ": prescribed TSR lies outside its lookup table")
        x = np.array([r[0] for r in table])
        ct = float(np.interp(tsr, x, [r[1] for r in table]))
        cp = float(np.interp(tsr, x, [r[2] for r in table]))
        return ct, cp, dict(mode=mode, tip_speed_ratio=tsr,
                            performance=[dict(tip_speed_ratio=a,
                                              thrust_coefficient=b,
                                              power_coefficient=c)
                                         for a, b, c in table])
    raise ValueError(turbine_id + ": control mode must be constant_ct or prescribed_tsr_table")


def validate_turbines(rows):
    """Validate research definitions and return immutable resolved records."""
    if not isinstance(rows, list):
        raise ValueError("Turbines must be a list")
    names = set()
    result = []
    fields = ("id", "points_xy", "rotor_diameter_m", "hub_height_m", "control", "support")
    for row in rows:
        _exact(row, fields, "turbine")
        name = row["id"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Turbine IDs must be nonempty and unique")
        names.add(name)
        locator, center, tangent, normal = _locator(row["points_xy"], name)
        diameter = _number(row["rotor_diameter_m"], name + ".rotor_diameter_m", positive=True)
        hub = _number(row["hub_height_m"], name + ".hub_height_m", positive=True)
        if hub < 0.5 * diameter:
            raise ValueError(name + ": hub height must leave the rotor above the bed")
        _exact(row["support"], ("frontal_area_m2", "drag_coefficient"), name + ".support")
        support_area = _number(row["support"]["frontal_area_m2"], name + ".support area",
                               nonnegative=True)
        support_cd = _number(row["support"]["drag_coefficient"], name + ".support Cd",
                             nonnegative=True)
        ct, cp, control = _control(row["control"], name)
        result.append(TurbineDefinition(name, locator, center, tangent, normal, diameter, hub,
                                        ct, cp, support_area, support_cd, control))
    return tuple(result)


def turbine_breaklines(definitions: Iterable[TurbineDefinition]):
    """Return physical rotor lines for the existing permeable constraint mesh."""
    return [dict(id="turbine::" + d.turbine_id, closed=False,
                 points_xy=[list(p) for p in d.rotor_line_xy]) for d in definitions]


@dataclass(frozen=True)
class TurbineFootprint:
    definition: TurbineDefinition
    cell_ids: np.ndarray
    overlap_area_m2: np.ndarray
    cell_area_m2: np.ndarray
    support_area_m2: float
    support_thickness_m: float
    numerical_cross_stream_width_m: float
    polygon_xy: tuple[tuple[float, float], ...]


def _cell_polygon(cell):
    return Polygon(cell.vertices_xy, getattr(cell, "holes_xy", ()))


def build_footprints(cells, definitions, support_thickness_m):
    """Intersect each finite source rectangle with true solver polygons.

    The support thickness is a numerical integration length along the turbine
    axis.  In the first qualified route it is the finest requested mesh size.
    The transverse width is always the physical rotor diameter.
    """
    thickness = _number(support_thickness_m, "turbine support thickness", positive=True)
    cells = tuple(cells)
    polygons = [_cell_polygon(cell) for cell in cells]
    if not polygons or any(not p.is_valid or p.area <= 0 for p in polygons):
        raise ValueError("Valid polygonal finite volumes are required")
    index = STRtree(polygons)
    result = []
    for definition in definitions:
        c = np.asarray(definition.center_xy)
        tangent = np.asarray(definition.tangent_xy)
        # Kramer--Piggott's modified actuator cross-section is Delta_y*H,
        # where Delta_y is the width over which the numerical state is
        # represented.  If a rotor is unresolved, that width is the containing
        # finite volume rather than the smaller physical diameter.  Once the
        # rotor spans several cells, the physical D is the source width.
        point = Polygon((c+[-1e-12, -1e-12], c+[1e-12, -1e-12],
                         c+[1e-12, 1e-12], c+[-1e-12, 1e-12]))
        containing = [int(raw) for raw in index.query(point)
                      if polygons[int(raw)].distance(point.centroid) <= 1e-10]
        if not containing:
            raise ValueError(definition.turbine_id + ": turbine center is outside the mesh")
        represented_width = max(
            max(np.asarray(polygons[i].exterior.coords) @ tangent) -
            min(np.asarray(polygons[i].exterior.coords) @ tangent)
            for i in containing)
        numerical_width = max(definition.rotor_diameter_m, float(represented_width))
        t = tangent * (0.5 * numerical_width)
        n = np.asarray(definition.normal_xy) * (0.5 * thickness)
        support = Polygon((c-t-n, c+t-n, c+t+n, c-t+n))
        ids, overlap = [], []
        for raw in index.query(support):
            i = int(raw)
            area = float(polygons[i].intersection(support).area)
            if area > 1e-14 * support.area:
                ids.append(i); overlap.append(area)
        order = np.argsort(ids); ids = np.asarray(ids, np.int64)[order]
        overlap = np.asarray(overlap, float)[order]
        covered = float(overlap.sum())
        if not math.isclose(covered, support.area, rel_tol=1e-10,
                            abs_tol=1e-12 * max(1.0, support.area)):
            raise ValueError(definition.turbine_id +
                             ": turbine footprint intersects a wall, solid, or domain boundary")
        areas = np.asarray([float(cells[i].area_m2) for i in ids])
        if np.any(overlap > areas * (1 + 1e-10)):
            raise ValueError(definition.turbine_id + ": footprint overlap exceeds a cell area")
        result.append(TurbineFootprint(definition, ids, overlap, areas, support.area,
                                       thickness, numerical_width,
                                       tuple(map(tuple, support.exterior.coords[:-1]))))
    return tuple(result)


def kramer_piggott_state(definition, local_normal_velocity_m_s, mean_depth_m,
                         numerical_cross_stream_width_m=None):
    """Resolve upstream velocity and corrected force per unit density.

    This is the rectangular-support correction in Kramer & Piggott (2016),
    equations 15--18.  The numerical actuator cross-section is D*H; the
    actual rotor and optional support contribute At*Ct + As*Cs.
    """
    u1 = _number(local_normal_velocity_m_s, "local normal velocity")
    depth = _number(mean_depth_m, "mean turbine depth", positive=True)
    width = (definition.rotor_diameter_m if numerical_cross_stream_width_m is None else
             _number(numerical_cross_stream_width_m, "numerical turbine width", positive=True))
    if width + 1e-12 < definition.rotor_diameter_m:
        raise ValueError(definition.turbine_id + ": numerical width cannot be smaller than the rotor")
    if depth < definition.hub_height_m + 0.5 * definition.rotor_diameter_m:
        raise ValueError(definition.turbine_id + ": rotor is not fully submerged")
    drag_area = definition.combined_drag_area_m2
    modified_ct = drag_area / (width * depth)
    if not 0 <= modified_ct < 1:
        raise ValueError(definition.turbine_id +
                         ": actuator correction requires 0 <= (At Ct + As Cs)/(D H) < 1")
    induction = 0.5 * (1.0 + math.sqrt(1.0 - modified_ct))
    u0 = u1 / induction
    force = 0.5 * drag_area * abs(u0) * u0
    useful_power = 0.5 * definition.rotor_area_m2 * definition.power_coefficient * abs(u0)**3
    tip_speed_ratio = definition.control.get("tip_speed_ratio")
    return dict(local_velocity_m_s=u1, upstream_velocity_m_s=u0,
                mean_depth_m=depth, modified_thrust_coefficient=modified_ct,
                induction_ratio=induction, force_per_density_m4_s2=force,
                useful_power_per_density_m5_s3=useful_power,
                hydraulic_power_removed_per_density_m5_s3=force*u1,
                control_mode=definition.control["mode"],
                numerical_cross_stream_width_m=width,
                tip_speed_ratio=tip_speed_ratio,
                thrust_coefficient=definition.thrust_coefficient,
                power_coefficient=definition.power_coefficient)


class TurbineMomentumSource:
    """Backend-neutral integrated source over polygon-overlap weights."""

    def __init__(self, footprints, *, backend="cpu", dry_depth_m=1e-6):
        if backend not in ("cpu", "gpu"):
            raise ValueError("Choose CPU or GPU explicitly")
        self.backend = backend
        if backend == "gpu":
            import cupy as xp
        else:
            xp = np
        self.xp = xp
        self.dry = _number(dry_depth_m, "dry depth", nonnegative=True)
        self.footprints = tuple(footprints)
        self.device = tuple(dict(ids=xp.asarray(f.cell_ids), overlap=xp.asarray(f.overlap_area_m2),
                                 fractions=xp.asarray(f.overlap_area_m2 / f.cell_area_m2),
                                 normal=xp.asarray(f.definition.normal_xy)) for f in self.footprints)
        self.identity = dict(schema=1, method=METHOD, backend=backend,
                             distribution="exact polygon overlap / actual cell area",
                             velocity="footprint-area-average local depth-averaged normal velocity",
                             depth="footprint-area-average physical water depth",
                             wet_dry="fully wet and fully submerged only",
                             turbulence_source=False, sediment_coupling=False, yaw_qualified=False)

    @staticmethod
    def _scalar(value, xp):
        return float(value) if xp is np else float(xp.asnumpy(value))

    def sample(self, q):
        """Evaluate every turbine on one accepted ``[h, hu, hv]`` state.

        This method does not mutate ``q``.  Output writers must call it on an
        accepted state instead of reusing the most recent source-stage
        diagnostic, because an SSP stage can subsequently be rejected.
        """
        xp = self.xp
        if q.ndim != 2 or q.shape[0] < 3:
            raise ValueError("Turbine source requires a [h, hu, hv] array")
        diagnostics = []
        for footprint, device in zip(self.footprints, self.device):
            ids, overlap, normal = device["ids"], device["overlap"], device["normal"]
            h = q[0, ids]
            if self._scalar(xp.min(h), xp) <= self.dry:
                raise ValueError(footprint.definition.turbine_id + ": turbine footprint contains a dry cell")
            u = q[1, ids] / h; v = q[2, ids] / h
            area = footprint.support_area_m2
            u1 = self._scalar(xp.sum(overlap * (normal[0]*u + normal[1]*v)) / area, xp)
            depth = self._scalar(xp.sum(overlap * h) / area, xp)
            state = kramer_piggott_state(footprint.definition, u1, depth,
                                         footprint.numerical_cross_stream_width_m)
            diagnostics.append(dict(turbine_id=footprint.definition.turbine_id,
                                    support_area_m2=area, **state))
        return diagnostics

    def apply(self, q, rate):
        """Add turbine momentum tendency to ``rate`` and return diagnostics.

        ``q`` and ``rate`` contain ``[h, hu, hv]``.  Water depth is unchanged.
        The normal force reverses with the normal flow; tangential momentum is
        untouched.  The returned force is per unit density.
        """
        if q.shape != rate.shape:
            raise ValueError("Turbine source requires matching [h, hu, hv] arrays")
        diagnostics = self.sample(q)
        for state, device in zip(diagnostics, self.device):
            acceleration = -state["force_per_density_m4_s2"] / state["support_area_m2"]
            ids, normal = device["ids"], device["normal"]
            rate[1, ids] += acceleration * normal[0] * device["fractions"]
            rate[2, ids] += acceleration * normal[1] * device["fractions"]
        return diagnostics


_HISTORY_FIELDS = (
    "time_s", "turbine_id", "control_mode", "local_normal_velocity_m_s",
    "reconstructed_upstream_velocity_m_s", "mean_depth_m", "tip_speed_ratio",
    "numerical_cross_stream_width_m", "thrust_coefficient", "power_coefficient", "normal_thrust_n",
    "thrust_magnitude_n", "useful_power_w", "hydraulic_power_removed_w",
    "cumulative_useful_energy_j", "cumulative_hydraulic_energy_removed_j",
)


class TurbinePerformanceHistory:
    """Accepted-state Ct/Cp, thrust, power and energy history.

    Useful turbine power is based on ``Cp`` and reconstructed upstream speed.
    Hydraulic power removed is the actuator force dotted with the local model
    velocity.  They are intentionally separate: their difference includes
    wake/mixing losses and must not be presented as generated electricity.
    """

    def __init__(self, definitions, *, water_density_kg_m3):
        self.density = _number(water_density_kg_m3, "water density", positive=True)
        self.ids = tuple(d.turbine_id for d in definitions)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Performance history requires unique turbines")
        self.rows = []
        self._last = {}

    def record(self, time_s, diagnostics):
        time_s = _number(time_s, "performance time", nonnegative=True)
        states = {row["turbine_id"]: row for row in diagnostics}
        if set(states) != set(self.ids) or len(states) != len(diagnostics):
            raise ValueError("Performance sample must contain every turbine exactly once")
        result = []
        for turbine_id in self.ids:
            state = states[turbine_id]
            useful = self.density * float(state["useful_power_per_density_m5_s3"])
            removed = self.density * float(state["hydraulic_power_removed_per_density_m5_s3"])
            previous = self._last.get(turbine_id)
            useful_energy = 0.0 if previous is None else previous[3]
            removed_energy = 0.0 if previous is None else previous[4]
            if previous is not None:
                dt = time_s - previous[0]
                if dt <= 0:
                    raise ValueError("Performance times must increase strictly")
                useful_energy += 0.5 * dt * (previous[1] + useful)
                removed_energy += 0.5 * dt * (previous[2] + removed)
            force = self.density * float(state["force_per_density_m4_s2"])
            row = dict(
                time_s=time_s, turbine_id=turbine_id,
                control_mode=state["control_mode"],
                local_normal_velocity_m_s=float(state["local_velocity_m_s"]),
                reconstructed_upstream_velocity_m_s=float(state["upstream_velocity_m_s"]),
                mean_depth_m=float(state["mean_depth_m"]),
                tip_speed_ratio=state["tip_speed_ratio"],
                numerical_cross_stream_width_m=float(state["numerical_cross_stream_width_m"]),
                thrust_coefficient=float(state["thrust_coefficient"]),
                power_coefficient=float(state["power_coefficient"]),
                normal_thrust_n=force, thrust_magnitude_n=abs(force),
                useful_power_w=useful, hydraulic_power_removed_w=removed,
                cumulative_useful_energy_j=useful_energy,
                cumulative_hydraulic_energy_removed_j=removed_energy,
            )
            self.rows.append(row); result.append(dict(row))
            self._last[turbine_id] = (time_s, useful, removed, useful_energy, removed_energy)
        return result

    def state(self):
        """Return JSON-serializable checkpoint state."""
        return dict(schema=1, water_density_kg_m3=self.density,
                    turbine_ids=list(self.ids), rows=[dict(row) for row in self.rows])

    @classmethod
    def restore(cls, definitions, state):
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("Unsupported turbine performance checkpoint")
        result = cls(definitions, water_density_kg_m3=state["water_density_kg_m3"])
        if tuple(state.get("turbine_ids", ())) != result.ids:
            raise ValueError("Turbines differ from the performance checkpoint")
        rows = state.get("rows")
        if not isinstance(rows, list) or len(rows) % len(result.ids):
            raise ValueError("Invalid turbine performance checkpoint rows")
        for offset in range(0, len(rows), len(result.ids)):
            group = rows[offset:offset+len(result.ids)]
            time_s = group[0].get("time_s")
            if any(row.get("time_s") != time_s or row.get("turbine_id") != turbine_id
                   for row, turbine_id in zip(group, result.ids)):
                raise ValueError("Invalid turbine performance checkpoint order")
            # Retain the authenticated reported values exactly.  Reconstruct
            # only the integration cursor used by the next accepted sample.
            for row in group:
                if set(row) != set(_HISTORY_FIELDS):
                    raise ValueError("Invalid turbine performance checkpoint fields")
                copy = dict(row); result.rows.append(copy)
                result._last[row["turbine_id"]] = (
                    float(time_s), float(row["useful_power_w"]),
                    float(row["hydraulic_power_removed_w"]),
                    float(row["cumulative_useful_energy_j"]),
                    float(row["cumulative_hydraulic_energy_removed_j"]),
                )
        return result

    @staticmethod
    def _filename(turbine_id):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", turbine_id).strip("._") or "turbine"
        suffix = hashlib.sha256(turbine_id.encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{suffix}.csv"

    @staticmethod
    def _atomic_csv(path, rows):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial",
                                         dir=path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS)
                writer.writeheader(); writer.writerows(rows)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
            raise

    def write_csv(self, folder):
        """Write one combined CSV and one CSV per turbine atomically."""
        folder = Path(folder); combined = folder / "turbine_performance.csv"
        self._atomic_csv(combined, self.rows)
        individual = {}
        for turbine_id in self.ids:
            path = folder / "turbines" / self._filename(turbine_id)
            self._atomic_csv(path, (row for row in self.rows if row["turbine_id"] == turbine_id))
            individual[turbine_id] = path
        return dict(combined=combined, per_turbine=individual)

