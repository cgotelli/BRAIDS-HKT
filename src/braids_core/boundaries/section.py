"""Hydraulics of a sampled open-channel cross section in SI units.

Only the physical piecewise-linear section is integrated. No analytical shape
formula, raster staircase perimeter or validation oracle is used here. Stage is
horizontal across every submerged portion of the assigned section. An explicit
legacy connected policy remains available for historical verification. Normal
flow supports either uniform roughness or contiguous roughness subsections;
critical flow uses energy alpha=1.
This service computes a hydraulic target, not an accepted SWE boundary flux.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np


_FLOW_HYDRAULIC_KAPPA = float(np.float32(0.41))


def _roughness_chezy(mode: str, depth: float, parameter: float, *,
                     white_colebrook_chzk2d: bool = False,
                     gravity: float = 9.81,
                     minimum_work_depth_m: float = .01) -> float:
    """Locked FLOW C/M/W conversion used by normal-section preparation."""
    if mode not in ("chezy", "manning", "white_colebrook"):
        raise SectionError(f"Unsupported hydraulic roughness mode {mode!r}")
    work = max(_finite(depth, "roughness work depth", positive=True),
               _finite(minimum_work_depth_m, "minimum roughness work depth", nonnegative=True))
    value = _finite(parameter, "hydraulic roughness parameter", positive=True)
    if mode == "chezy":
        return value
    if mode == "manning":
        return work ** (1 / 6) / value
    if white_colebrook_chzk2d:
        return 18 * math.log10(12 * max(.5, work / value))
    return (math.sqrt(_finite(gravity, "gravity", positive=True)) /
            _FLOW_HYDRAULIC_KAPPA *
            math.log1p(work / (math.e * value / 30)))


class SectionError(ValueError):
    """Invalid section, units, stage, or hydraulic input."""


class NoHydraulicSolution(SectionError):
    """No admissible hydraulic root exists, or its tolerance was not reached."""


@dataclass(frozen=True)
class WetInterval:
    start_station_m: float
    end_station_m: float


@dataclass(frozen=True)
class SectionGeometry:
    stage_m: float
    depth_m: float
    area_m2: float
    perimeter_m: float
    top_width_m: float
    hydraulic_radius_m: float
    wet_intervals: tuple[WetInterval, ...]
    excluded_wet_area_m2: float = 0.0

    @property
    def wetted_perimeter_m(self) -> float:
        return self.perimeter_m


@dataclass(frozen=True)
class HydraulicSolution:
    bc_id: str
    mode: str
    stage_m: float
    depth_m: float
    discharge_m3_s: float
    mean_velocity_m_s: float
    geometry: SectionGeometry
    relative_residual: float
    iterations: int
    candidate_stages_m: tuple[float, ...]
    roughness_mode: str = "not applicable"
    subsection_conveyances_m3_s_per_sqrt_slope: tuple[float, ...] = ()


def _finite(value: object, label: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SectionError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SectionError(f"{label} must be a finite number") from error
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        qualifier = "positive " if positive else "nonnegative " if nonnegative else ""
        raise SectionError(f"{label} must be a finite {qualifier}number")
    return number


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real stationary points, with stable subtraction for the quadratic."""
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0:
        return []
    a, b, c = a / scale, b / scale, c / scale
    if a == 0:
        return [-c / b] if b else []
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    q = -.5 * (b + math.copysign(math.sqrt(discriminant), b))
    if q == 0:
        return [-b / (2 * a)]
    return [q / a, c / q]


@dataclass(frozen=True)
class Section:
    bc_id: str
    stations_m: Sequence[float]
    bed_elevations_m: Sequence[float]
    datum_m: float
    inward_normal: tuple[float, float]
    horizontal_units: str = "m"
    vertical_units: str = "m"
    max_stage_m: float | None = None
    seed_station_m: float | None = None
    segment_manning_n: Sequence[float] | None = None
    wetting_policy: str = "all_submerged"
    _s: np.ndarray = field(init=False, repr=False, compare=False)
    _z: np.ndarray = field(init=False, repr=False, compare=False)
    _length: np.ndarray = field(init=False, repr=False, compare=False)
    _segment_length: np.ndarray = field(init=False, repr=False, compare=False)
    _seed_level: float = field(init=False, repr=False, compare=False)
    _segment_manning: np.ndarray | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bc_id, str) or not self.bc_id or any(c.isspace() for c in self.bc_id):
            raise SectionError("bc_id must be a nonempty stable identifier without whitespace")
        if self.horizontal_units != "m" or self.vertical_units != "m":
            raise SectionError(f"{self.bc_id}: section coordinates and elevations must be in metres ('m')")
        datum = _finite(self.datum_m, "datum_m")
        try:
            stations = tuple(_finite(v, "station") for v in self.stations_m)
            elevations = tuple(_finite(v, "bed elevation") for v in self.bed_elevations_m)
            normal = tuple(_finite(v, "inward normal") for v in self.inward_normal)
        except TypeError as error:
            raise SectionError(f"{self.bc_id}: stations, elevations and normal must be sequences") from error
        if len(stations) < 2 or len(elevations) != len(stations):
            raise SectionError(f"{self.bc_id}: at least two matching station/elevation samples are required")
        segment_manning = None
        if self.segment_manning_n is not None:
            try:
                values = tuple(_finite(value, "section Manning n", positive=True)
                               for value in self.segment_manning_n)
            except TypeError as error:
                raise SectionError(
                    f"{self.bc_id}: segment_manning_n must contain one value per terrain segment") from error
            if len(values) != len(stations) - 1:
                raise SectionError(
                    f"{self.bc_id}: segment_manning_n must contain one value per terrain segment")
            segment_manning = np.asarray(values)
        try:
            with np.errstate(over="raise", invalid="raise"):
                s, z = np.asarray(stations), np.asarray(elevations) - datum
                length = np.diff(s)
                segment_length = np.hypot(length, np.diff(z))
        except FloatingPointError as error:
            raise SectionError(f"{self.bc_id}: section differences exceed numerical range") from error
        if np.any(length <= 0):
            raise SectionError(f"{self.bc_id}: stations must be strictly increasing")
        if len(normal) != 2 or not math.isclose(math.hypot(*normal), 1., rel_tol=0, abs_tol=1e-10):
            raise SectionError(f"{self.bc_id}: an explicit unit inward normal (nx, ny) is required")
        highest = min(elevations[0], elevations[-1])
        maximum = highest if self.max_stage_m is None else _finite(self.max_stage_m, "max_stage_m")
        if maximum > highest:
            raise SectionError(f"{self.bc_id}: max_stage_m exceeds a section endpoint; extend the physical section")
        if self.wetting_policy not in ("all_submerged", "connected"):
            raise SectionError(f"{self.bc_id}: unknown section wetting_policy")
        if self.seed_station_m is None:
            indices = np.flatnonzero(z == z.min())
            if np.any(np.diff(indices) > 1):
                if self.wetting_policy == "connected":
                    raise SectionError(f"{self.bc_id}: separate equal minima require an explicit seed_station_m")
                # A seed is provenance only for all-submerged sections. Do not
                # put it on a dry crest between equal, separated minima.
                indices = indices[:1]
            seed = float(s[indices[0]] + (s[indices[-1]] - s[indices[0]]) / 2)
        else:
            seed = _finite(self.seed_station_m, "seed_station_m")
        if not s[0] < seed < s[-1]:
            raise SectionError(f"{self.bc_id}: the wetting seed must be inside the section endpoints")
        seed_level = float(np.interp(seed, s, z))
        # A seed selects a basin minimum, not an arbitrary point on its bank.
        j = int(np.searchsorted(s, seed))
        if s[j] == seed:
            first = last = j
        else:
            first, last = j - 1, j
        is_minimum = z[first] == z[last]
        if is_minimum:
            # Test the whole plateau: a bench that drains at its far end is
            # not a basin minimum, even if the seed has flat immediate neighbors.
            while first > 0 and z[first - 1] == seed_level:
                first -= 1
            while last < len(z) - 1 and z[last + 1] == seed_level:
                last += 1
            is_minimum = (first > 0 and last < len(z) - 1 and
                          z[first - 1] > seed_level and z[last + 1] > seed_level)
        if not is_minimum:
            raise SectionError(f"{self.bc_id}: seed_station_m must lie at a local section minimum")
        if maximum <= datum + seed_level:
            raise SectionError(f"{self.bc_id}: endpoints/max_stage do not enclose a positive-depth section")
        for key, value in (("stations_m", stations), ("bed_elevations_m", elevations),
                           ("datum_m", datum), ("inward_normal", normal),
                           ("max_stage_m", maximum), ("seed_station_m", seed),
                           ("segment_manning_n", None if segment_manning is None else tuple(segment_manning)),
                           ("_seed_level", seed_level if self.wetting_policy == "connected" else float(z.min()))):
            object.__setattr__(self, key, value)
        for key, values in (("_s", s), ("_z", z), ("_length", length),
                            ("_segment_length", segment_length)):
            values.setflags(write=False)
            object.__setattr__(self, key, values)
        if segment_manning is not None:
            segment_manning.setflags(write=False)
        object.__setattr__(self, "_segment_manning", segment_manning)

    @property
    def minimum_stage_m(self) -> float:
        """Dry-state stage at the minimum of the participating section."""
        return self.datum_m + self._seed_level

    def _segment_geometry(self, level: float):
        """Return wetted geometry plus per-segment area and perimeter."""
        stage = self.datum_m + level
        if level <= self._seed_level:
            zeros = np.zeros_like(self._length)
            return SectionGeometry(stage, level, 0., 0., 0., 0., ()), zeros, zeros, 0, 0
        h0, h1 = level - self._z[:-1], level - self._z[1:]
        wet0, wet1 = h0 > 0, h1 > 0
        fraction = np.zeros_like(self._length)
        fraction[wet0 & wet1] = 1.0
        left, right = wet0 & ~wet1, ~wet0 & wet1
        fraction[left] = h0[left] / (h0[left] - h1[left])
        fraction[right] = h1[right] / (h1[right] - h0[right])
        widths = fraction * self._length
        areas = widths * (np.maximum(h0, 0) + np.maximum(h1, 0)) / 2

        if self.wetting_policy == "all_submerged":
            intervals = []
            for j in np.flatnonzero(widths > 0):
                start = float(self._s[j] if wet0[j] else self._s[j+1]-widths[j])
                end = float(self._s[j+1] if wet1[j] else self._s[j]+widths[j])
                # Join across a wet node, never across a dry/just-touched crest.
                if intervals and wet0[j] and intervals[-1].end_station_m == start:
                    intervals[-1] = WetInterval(intervals[-1].start_station_m, end)
                else:
                    intervals.append(WetInterval(start, end))
            area = float(np.sum(areas))
            perimeter = float(np.sum(fraction*self._segment_length))
            width = float(np.sum(widths))
            geometry = SectionGeometry(stage, level, area, perimeter, width,
                area/perimeter if perimeter else 0., tuple(intervals))
            return geometry, areas, fraction*self._segment_length, 0, len(areas)

        # Dry/just-touched crests isolate neighboring depressions. They join
        # only when the common stage rises strictly above their connecting crest.
        left_barriers = np.flatnonzero((self._s < self.seed_station_m) & (self._z >= level))
        right_barriers = np.flatnonzero((self._s > self.seed_station_m) & (self._z >= level))
        if not len(left_barriers) or not len(right_barriers):
            raise SectionError(f"{self.bc_id}: stage leaves the sampled section")
        first, last = int(left_barriers[-1]), int(right_barriers[0])
        area = float(np.sum(areas[first:last]))
        perimeter = float(np.sum((fraction * self._segment_length)[first:last]))
        width = float(np.sum(widths[first:last]))
        start = self._s[first + 1] - widths[first]
        end = self._s[last - 1] + widths[last - 1]
        geometry = SectionGeometry(stage, level, area, perimeter, width,
                                   area / perimeter if perimeter else 0.,
                                   (WetInterval(float(start), float(end)),) if area else (),
                                   max(0., float(np.sum(areas)) - area))
        return geometry, areas, fraction * self._segment_length, first, last

    def _geometry(self, level: float) -> SectionGeometry:
        return self._segment_geometry(level)[0]

    def geometry(self, stage_m: float) -> SectionGeometry:
        """Wetted geometry at one absolute horizontal water stage.

        At a dry stage A/P/T are zero (actual wetted quantities, not one-sided
        geometric limits). ``depth_m`` is stage minus the fixed datum; it is not
        a uniform local water depth. Above-bank extrapolation is rejected.
        """
        stage = _finite(stage_m, "stage_m")
        if stage > self.max_stage_m:
            raise SectionError(f"{self.bc_id}: requested stage exceeds max_stage_m/section endpoints")
        return self._geometry(stage - self.datum_m)

    def _normal_conveyances(self, level: float,
                            n: float | Sequence[float] | None, *,
                            roughness_mode: str = "manning",
                            white_colebrook_chzk2d: bool = False,
                            gravity: float = 9.81,
                            minimum_work_depth_m: float = 0.) -> tuple[float, ...]:
        """Return conveyance for contiguous equal-roughness subsections.

        Artificial vertical lines between subsections are excluded from the
        wetted perimeter. Each group contains only its actual terrain segments.
        """
        geometry, areas, perimeters, first, last = self._segment_geometry(level)
        if geometry.area_m2 <= 0:
            return ()
        spatial = n is None or (not isinstance(n, (str, bytes)) and np.ndim(n) != 0)
        if not spatial:
            coefficient = _finite(n, "hydraulic roughness parameter", positive=True)
            if roughness_mode == "manning" and minimum_work_depth_m == 0.:
                # Preserve archived preparation arithmetic byte for byte.
                return (geometry.area_m2 * geometry.hydraulic_radius_m ** (2 / 3) / coefficient,)
            chezy = _roughness_chezy(
                roughness_mode, geometry.hydraulic_radius_m, coefficient,
                white_colebrook_chzk2d=white_colebrook_chzk2d,
                gravity=gravity, minimum_work_depth_m=minimum_work_depth_m)
            return (geometry.area_m2 * chezy * math.sqrt(geometry.hydraulic_radius_m),)
        if n is None:
            values = self._segment_manning
            if values is None:
                raise SectionError(f"{self.bc_id}: normal depth requires section roughness")
        else:
            try:
                values = np.asarray(tuple(_finite(value, "section roughness parameter", positive=True)
                                          for value in n))
            except TypeError as error:
                raise SectionError(
                    f"{self.bc_id}: spatial Manning requires one value per terrain segment") from error
            if values.shape != self._length.shape:
                raise SectionError(
                    f"{self.bc_id}: spatial Manning requires one value per terrain segment")
        conveyances = []
        start = first
        while start < last:
            stop = start + 1
            while stop < last and values[stop] == values[start]:
                stop += 1
            area = float(np.sum(areas[start:stop]))
            perimeter = float(np.sum(perimeters[start:stop]))
            if area > 0 and perimeter > 0:
                radius = area / perimeter
                if roughness_mode == "manning" and minimum_work_depth_m == 0.:
                    conveyances.append(area * radius ** (2 / 3) / float(values[start]))
                else:
                    chezy = _roughness_chezy(
                        roughness_mode, radius, float(values[start]),
                        white_colebrook_chzk2d=white_colebrook_chzk2d,
                        gravity=gravity, minimum_work_depth_m=minimum_work_depth_m)
                    conveyances.append(area * chezy * math.sqrt(radius))
            start = stop
        return tuple(conveyances)

    def _rate(self, level: float, mode: str,
              n: float | Sequence[float] | None, slope: float, g: float,
              roughness_mode: str = "manning",
              white_colebrook_chzk2d: bool = False,
              minimum_work_depth_m: float = 0.) -> float:
        geometry = self._geometry(level)
        if geometry.area_m2 <= 0:
            return 0.0
        if mode == "normal":
            result = math.sqrt(slope) * math.fsum(self._normal_conveyances(
                level, n, roughness_mode=roughness_mode,
                white_colebrook_chzk2d=white_colebrook_chzk2d,
                gravity=g, minimum_work_depth_m=minimum_work_depth_m))
        else:
            result = geometry.area_m2 * math.sqrt(g * geometry.area_m2 / geometry.top_width_m)
        if not math.isfinite(result):
            raise NoHydraulicSolution(f"{self.bc_id}: hydraulic discharge is outside numerical range")
        return result

    def _smooth_intervals(self, mode: str) -> list[tuple[float, float]]:
        """Split at all terrain knots, connectivity events and rate extrema.

        Between terrain elevations on one connected branch, A is quadratic and
        T/P are linear in stage. Rate extrema follow quadratics: normal flow
        5*T*P - 2*A*P' = 0; critical flow 3*T*T - A*T' = 0. Splitting there
        prevents missing two roots in a wide/compound section without requiring
        a fragile uniform stage scan or assuming global monotonic conveyance.
        """
        maximum = self.max_stage_m - self.datum_m
        knots = sorted({self._seed_level, maximum, *(float(z) for z in self._z if self._seed_level < z < maximum)})
        intervals = []
        for a, b in zip(knots[:-1], knots[1:]):
            mid, quarter = (a + b) / 2, (b - a) / 4
            if mid == a or mid == b:
                continue
            centre, lower, upper = self._geometry(mid), self._geometry(mid - quarter), self._geometry(mid + quarter)
            tp = (upper.top_width_m - lower.top_width_m) / (2 * quarter)
            pp = (upper.perimeter_m - lower.perimeter_m) / (2 * quarter)
            if abs(tp) < 1e-12 * max(centre.top_width_m, 1.) / (b - a):
                tp = 0.
            if abs(pp) < 1e-12 * max(centre.perimeter_m, 1.) / (b - a):
                pp = 0.
            area, top, perimeter = centre.area_m2, centre.top_width_m, centre.perimeter_m
            if mode == "normal":
                extra = _quadratic_roots(4 * tp * pp, 3 * top * pp + 5 * tp * perimeter,
                                         5 * top * perimeter - 2 * area * pp)
            else:
                extra = _quadratic_roots(2.5 * tp * tp, 5 * top * tp, 3 * top * top - area * tp)
            points = sorted({a, b, *(mid + x for x in extra if math.isfinite(x) and a < mid + x < b)})
            intervals.extend(zip(points[:-1], points[1:]))
        return intervals

    def _solve(self, Q: float, mode: str, n: float | Sequence[float] | None,
               slope: float, g: float,
               stage_tolerance_m: float, relative_discharge_tolerance: float,
               max_iterations: int, roughness_mode: str = "manning",
               white_colebrook_chzk2d: bool = False,
               minimum_work_depth_m: float = 0.) -> HydraulicSolution:
        target = _finite(Q, "discharge_m3_s", nonnegative=True)
        stage_tol = _finite(stage_tolerance_m, "stage_tolerance_m", positive=True)
        qtol = _finite(relative_discharge_tolerance, "relative_discharge_tolerance", positive=True)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise SectionError("max_iterations must be a positive integer")
        if target == 0:
            geometry = self.geometry(self.minimum_stage_m)
            spatial = mode == "normal" and (n is None or
                      (not isinstance(n, (str, bytes)) and np.ndim(n) != 0))
            return HydraulicSolution(self.bc_id, mode, geometry.stage_m, geometry.depth_m, 0., 0.,
                                     geometry, 0., 0, (geometry.stage_m,),
                                     "spatial subsections" if spatial else
                                     "uniform" if mode == "normal" else "not applicable", ())
        law = lambda level: self._rate(
            level, mode, n, slope, g, roughness_mode,
            white_colebrook_chzk2d, minimum_work_depth_m)
        roots: list[tuple[float, int]] = []
        brackets = self._smooth_intervals(mode)

        for left, right in brackets:
            # Stay on each continuous branch, avoiding a jump when an isolated
            # depression suddenly joins at a sill. Exact knot roots are added below.
            lo, hi = math.nextafter(left, right), math.nextafter(right, left)
            if lo >= hi:
                continue
            flo, fhi = law(lo) - target, law(hi) - target
            if mode == "critical" and fhi < flo:
                continue  # This branch gives an energy maximum, not a minimum.
            if flo * fhi > 0:
                continue
            for iteration in range(1, max_iterations + 1):
                mid = (lo + hi) / 2
                residual = law(mid) - target
                if abs(residual) / target <= qtol and hi - lo <= stage_tol:
                    roots.append((mid, iteration))
                    break
                if mid == lo or mid == hi:
                    if abs(residual) / target <= qtol:
                        roots.append((mid, iteration))
                        break
                    raise NoHydraulicSolution(f"{self.bc_id}: stage precision cannot satisfy the requested discharge tolerance")
                if (flo <= 0 <= residual) or (residual <= 0 <= flo):
                    hi, fhi = mid, residual
                else:
                    lo, flo = mid, residual
            else:
                raise NoHydraulicSolution(f"{self.bc_id}: {mode} root did not converge in {max_iterations} iterations")

        knots = sorted({p for interval in brackets for p in interval})
        for level in knots:
            if level <= self._seed_level:
                continue
            if abs(law(level) - target) / target <= qtol:
                if mode == "critical":
                    maximum = self.max_stage_m - self.datum_m
                    if level < maximum:
                        # At a joining sill, area (and therefore specific
                        # energy) can jump. Qc derivative signs presume smooth
                        # geometry and cannot prove an energy minimum there.
                        at = self._geometry(level).area_m2
                        above = self._geometry(math.nextafter(level, maximum)).area_m2
                        if above - at > 64 * math.ulp(max(at, above)):
                            roots = [r for r in roots if abs(r[0] - level) > 2 * stage_tol]
                            continue
                    step = min(stage_tol * 10, (level - self._seed_level) / 4)
                    if law(level - step) > target:
                        continue
                    if level < self.max_stage_m - self.datum_m and law(min(level + step, self.max_stage_m - self.datum_m)) < target:
                        continue
                roots.append((level, 0))
        if not roots:
            raise NoHydraulicSolution(f"{self.bc_id}: no admissible {mode} stage for Q={target:g} m3/s below the section endpoints; check section extent, control and connectivity")
        roots.sort()
        distinct = []
        for candidate in roots:
            if not distinct or abs(candidate[0] - distinct[-1][0]) > stage_tol * 2:
                distinct.append(candidate)
        if mode == "critical":
            def energy(pair: tuple[float, int]) -> float:
                geometry = self._geometry(pair[0])
                return pair[0] + target * target / (2 * g * geometry.area_m2 ** 2)
            chosen, iterations = min(distinct, key=energy)
        else:
            chosen, iterations = distinct[0]  # Lowest admissible normal-stage root.
        # Re-evaluate the actual representable absolute stage returned to callers.
        geometry = self.geometry(self.datum_m + chosen)
        actual_q = self._rate(
            geometry.depth_m, mode, n, slope, g, roughness_mode,
            white_colebrook_chzk2d, minimum_work_depth_m)
        relative = abs(actual_q - target) / target
        if relative > qtol:
            raise NoHydraulicSolution(f"{self.bc_id}: absolute-stage rounding exceeds discharge tolerance; use a local vertical datum or achievable tolerance")
        spatial = mode == "normal" and (n is None or
                  (not isinstance(n, (str, bytes)) and np.ndim(n) != 0))
        conveyances = self._normal_conveyances(
            chosen, n, roughness_mode=roughness_mode,
            white_colebrook_chzk2d=white_colebrook_chzk2d,
            gravity=g, minimum_work_depth_m=minimum_work_depth_m) if mode == "normal" else ()
        return HydraulicSolution(self.bc_id, mode, geometry.stage_m, geometry.depth_m,
                                 target, target / geometry.area_m2, geometry, relative,
                                 iterations, tuple(self.datum_m + p[0] for p in distinct),
                                 "spatial subsections" if spatial else
                                 "uniform" if mode == "normal" else "not applicable",
                                 conveyances)

    def normal_stage(self, discharge_m3_s: float,
                     n: float | Sequence[float] | None, energy_slope: float, *,
                     roughness_mode: str = "manning",
                     white_colebrook_chzk2d: bool = False,
                     minimum_work_depth_m: float = 0.,
                     stage_tolerance_m: float = 1e-10,
                     relative_discharge_tolerance: float = 1e-10,
                     max_iterations: int = 200) -> HydraulicSolution:
        """Lowest admissible section Manning stage; positive n and slope.

        Passing ``None`` uses ``segment_manning_n`` and evaluates contiguous
        roughness subsections with a common horizontal water surface.
        """
        if n is None and self._segment_manning is None:
            raise SectionError(f"{self.bc_id}: normal depth requires section roughness")
        if n is not None and (isinstance(n, (str, bytes)) or np.ndim(n) == 0):
            n = _finite(n, "Manning n", positive=True)
        slope = _finite(energy_slope, "energy_slope", positive=True)
        if roughness_mode != "white_colebrook" and white_colebrook_chzk2d:
            raise SectionError("ChzK2D applies only to White-Colebrook roughness")
        return self._solve(discharge_m3_s, "normal", n, slope, 9.81,
                           stage_tolerance_m, relative_discharge_tolerance,
                           max_iterations, roughness_mode,
                           white_colebrook_chzk2d, minimum_work_depth_m)

    def critical_stage(self, discharge_m3_s: float, g: float = 9.81, *,
                       stage_tolerance_m: float = 1e-10,
                       relative_discharge_tolerance: float = 1e-10,
                       max_iterations: int = 200) -> HydraulicSolution:
        """Lowest-energy admissible critical stage, using energy coefficient 1."""
        gravity = _finite(g, "gravity", positive=True)
        return self._solve(discharge_m3_s, "critical", 1., 1., gravity,
                           stage_tolerance_m, relative_discharge_tolerance, max_iterations)


def register_sections(sections: Iterable[Section]) -> dict[str, Section]:
    """Keep stable physical IDs; duplicate IDs cannot overwrite an inlet."""
    result: dict[str, Section] = {}
    for section in sections:
        if not isinstance(section, Section):
            raise SectionError("Only Section instances can be registered")
        if section.bc_id in result:
            raise SectionError(f"Duplicate boundary ID: {section.bc_id}")
        result[section.bc_id] = section
    return result
