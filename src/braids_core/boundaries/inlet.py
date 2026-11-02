"""Immutable inlet requests and whole-face hydraulic target construction.

These are requested states, not accepted SWE fluxes. In particular, a subcritical
inlet must retain outgoing characteristic information when a solver constructs
its compatible boundary state; prescribing this target's stage and discharge
as two independent constraints would overdetermine that boundary.

The first mapping supports straight, axis-aligned, fixed-grid boundaries. Each
Face represents a whole native face with constant native bed. Connectivity is
selected at its station midpoint; the continuous wetted overlap is reported
but never silently used to cut a native face. Refine the grid to reduce this
explicit representation error. A section endpoint cutting a face is rejected.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .section import HydraulicSolution, Section, SectionError


class InletError(SectionError):
    """An invalid inlet request, hydrograph, or whole-face map."""


def _number(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InletError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InletError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        qualifier = "positive " if positive else "nonnegative " if nonnegative else ""
        raise InletError(f"{name} must be a finite {qualifier}number")
    return number


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise InletError(f"{name} must be a nonempty stable identifier without whitespace")
    return value


def _axis_normal(values: Sequence[float]) -> tuple[float, float]:
    try:
        normal = tuple(_number(value, "normal component") for value in values)
    except TypeError as error:
        raise InletError("An axis-aligned inward normal is required") from error
    if normal not in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)):
        raise InletError("Only the four exact axis-aligned inward unit normals are supported")
    return normal


@dataclass(frozen=True)
class Hydrograph:
    """Nonnegative discharge knots; endpoint policy is hold, zero, or error.

    Exact endpoints use their knot value. The policy applies strictly outside
    the supplied time interval; a single knot is valid and follows the same
    rule. Times may be negative, for an explicitly chosen simulation origin.
    """

    times_s: Sequence[float]
    discharges_m3_s: Sequence[float]
    endpoint_policy: str = "hold"

    def __post_init__(self) -> None:
        try:
            times = tuple(_number(value, "hydrograph time") for value in self.times_s)
            values = tuple(_number(value, "hydrograph discharge", nonnegative=True)
                           for value in self.discharges_m3_s)
        except TypeError as error:
            raise InletError("Hydrograph times and discharges must be sequences") from error
        if not times or len(times) != len(values):
            raise InletError("Hydrograph needs matching nonempty time/discharge sequences")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise InletError("Hydrograph times must be strictly increasing")
        if self.endpoint_policy not in ("hold", "zero", "error"):
            raise InletError("Hydrograph endpoint_policy must be hold, zero, or error")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "discharges_m3_s", values)

    def discharge(self, time_s: float) -> float:
        time = _number(time_s, "time_s")
        if time < self.times_s[0] or time > self.times_s[-1]:
            if self.endpoint_policy == "error":
                raise InletError("Requested time lies outside the hydrograph interval")
            if self.endpoint_policy == "zero":
                return 0.
            return self.discharges_m3_s[0 if time < self.times_s[0] else -1]
        index = bisect_right(self.times_s, time) - 1
        if index == len(self.times_s) - 1 or time == self.times_s[index]:
            return self.discharges_m3_s[index]
        t0, t1 = self.times_s[index:index + 2]
        q0, q1 = self.discharges_m3_s[index:index + 2]
        interval = t1 - t0
        if math.isfinite(interval):
            fraction = (time - t0) / interval
        else:
            # Scaling avoids overflow for opposite large finite times.
            scale = max(abs(t0), abs(t1))
            fraction = ((time / scale) - (t0 / scale)) / ((t1 / scale) - (t0 / scale))
        answer = (1. - fraction) * q0 + fraction * q1
        return _number(answer, "interpolated discharge", nonnegative=True)


@dataclass(frozen=True)
class InletSpec:
    """A total-Q request with a hydraulic target and transverse allocation.

    ``value_m`` is an absolute elevation for stage mode, or depth above the
    Section's fixed datum for depth mode. Normal/critical modes derive stage
    from Q and reject an unrelated value. Conveyance allocation uses local
    wide-strip weights L*h^(5/3)/n; it is a target profile, not a compound-section
    Manning closure or a characteristic-boundary formulation.
    """

    bc_id: str
    mode: str
    hydrograph: Hydrograph
    value_m: float | None = None
    manning_n: float = .03
    energy_slope: float = .001
    gravity_m_s2: float = 9.81
    allocation: str = "uniform"

    def __post_init__(self) -> None:
        _identifier(self.bc_id, "bc_id")
        if self.mode not in ("stage", "depth", "normal", "critical"):
            raise InletError("Inlet mode must be stage, depth, normal, or critical")
        if not isinstance(self.hydrograph, Hydrograph):
            raise InletError("Every inlet requires an explicit Hydrograph discharge input")
        if self.allocation not in ("uniform", "conveyance"):
            raise InletError("Allocation must be uniform or conveyance")
        if self.mode in ("stage", "depth"):
            value = _number(self.value_m, "value_m", nonnegative=self.mode == "depth")
            object.__setattr__(self, "value_m", value)
        elif self.value_m is not None:
            raise InletError("Normal/critical targets derive stage from discharge; omit value_m")
        for name in ("manning_n", "energy_slope", "gravity_m_s2"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class Face:
    """One whole native face in increasing canonical section stations.

    ``global_key`` identifies its unique physical ownership across all inlets.
    ``length_m`` is the actual native grid-face length (defaults to the station
    span); a supplied length different from that span rejects a cut/oblique
    face. Bed is the constant native face bed, not a fitted continuous profile.
    Optional local n changes conveyance weights only, not solver friction.
    """

    global_key: str
    station_start_m: float
    station_end_m: float
    bed_elevation_m: float
    inward_normal: tuple[float, float]
    length_m: float | None = None
    manning_n: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.global_key, "global face key")
        start = _number(self.station_start_m, "face start")
        end = _number(self.station_end_m, "face end")
        span = _number(end - start, "face station span", positive=True)
        length = span if self.length_m is None else _number(self.length_m, "native face length", positive=True)
        if not math.isclose(span, length, rel_tol=1e-12, abs_tol=1e-12):
            raise InletError("Station span must equal the whole native face length; cut/oblique faces are unsupported")
        bed = _number(self.bed_elevation_m, "native face bed")
        normal = _axis_normal(self.inward_normal)
        for name, value in (("station_start_m", start), ("station_end_m", end),
                            ("bed_elevation_m", bed), ("inward_normal", normal), ("length_m", length)):
            object.__setattr__(self, name, value)
        if self.manning_n is not None:
            object.__setattr__(self, "manning_n", _number(self.manning_n, "face Manning n", positive=True))


@dataclass(frozen=True)
class FaceState:
    face: Face
    h: float
    hu: float
    hv: float
    normal_velocity_m_s: float
    connected_fraction: float
    midpoint_connected: bool


@dataclass(frozen=True)
class InletTarget:
    bc_id: str
    time_s: float
    mode: str
    allocation: str
    target_stage_m: float
    requested_discharge_m3_s: float
    state_discharge_m3_s: float
    continuous_area_m2: float
    discrete_area_m2: float
    continuous_to_discrete_area_ratio: float | None
    continuous_mean_velocity_m_s: float
    faces: tuple[FaceState, ...]
    hydraulic_solution: HydraulicSolution | None = None


def register_face_ownership(faces_by_id: Mapping[str, Iterable[Face]]) -> dict[str, str]:
    """Return global face key -> inlet ID, rejecting duplicate physical owners."""
    owners = {}
    for bc_id, faces in faces_by_id.items():
        _identifier(bc_id, "bc_id")
        for face in faces:
            if not isinstance(face, Face):
                raise InletError("Inlet maps must contain Face records")
            if face.global_key in owners:
                raise InletError(f"Duplicate ownership of face {face.global_key!r}: {owners[face.global_key]!r} and {bc_id!r}")
            owners[face.global_key] = bc_id
    return owners


def _face_map(section: Section, faces: Iterable[Face]) -> tuple[Face, ...]:
    result = tuple(faces)
    if not result:
        raise InletError("An inlet needs a nonempty whole-face map")
    register_face_ownership({section.bc_id: result})
    normal = _axis_normal(section.inward_normal)
    ordered = tuple(sorted(result, key=lambda face: (face.station_start_m, face.global_key)))
    tolerance = 1e-10
    if (abs(ordered[0].station_start_m - section.stations_m[0]) > tolerance or
            abs(ordered[-1].station_end_m - section.stations_m[-1]) > tolerance):
        raise InletError("Whole faces must tile the section endpoints; boundary cuts/truncation are unsupported")
    for face in ordered:
        if face.inward_normal != normal:
            raise InletError("Every face normal must match the section inward normal")
    for left, right in zip(ordered, ordered[1:]):
        if abs(left.station_end_m - right.station_start_m) > tolerance:
            raise InletError("Whole-face station intervals must tile without gaps or overlaps")
    return ordered


def build_inlet_target(spec: InletSpec, section: Section, faces: Iterable[Face],
                       time_s: float) -> InletTarget:
    """Construct a requested state without solver calls or any budget mutation.

    Native faces are selected by midpoint membership in the wetted intervals. A partially intersected
    wet interval is reported in connected_fraction; state/flux integration still
    uses the full native length. Positive Q without represented wet area fails.
    For uniform allocation U=Q/sum(h*L). For conveyance, U is proportional to
    h^(2/3)/n and normalized so sum(h*U*L)=Q. Neither is an accepted flux claim.
    """
    if not isinstance(spec, InletSpec) or not isinstance(section, Section):
        raise InletError("Expected an InletSpec and Section")
    if spec.bc_id != section.bc_id:
        raise InletError("Inlet and section bc_id must match")
    time = _number(time_s, "time_s")
    ordered = _face_map(section, faces)
    Q = spec.hydrograph.discharge(time)
    solution = None
    if spec.mode == "normal":
        solution = section.normal_stage(Q, spec.manning_n, spec.energy_slope)
        stage = solution.stage_m
    elif spec.mode == "critical":
        solution = section.critical_stage(Q, spec.gravity_m_s2)
        stage = solution.stage_m
    else:
        stage = spec.value_m if spec.mode == "stage" else section.datum_m + spec.value_m
    geometry = section.geometry(stage)
    mapped = []
    for face in ordered:
        midpoint = face.station_start_m + (face.station_end_m - face.station_start_m) / 2
        connected = any(interval.start_station_m < midpoint < interval.end_station_m
                        for interval in geometry.wet_intervals)
        overlap = math.fsum(max(0., min(face.station_end_m, interval.end_station_m) -
                                max(face.station_start_m, interval.start_station_m))
                            for interval in geometry.wet_intervals)
        fraction = min(1., max(0., overlap / face.length_m))
        depth = max(stage - face.bed_elevation_m, 0.) if connected else 0.
        depth = _number(depth, "mapped water depth", nonnegative=True)
        area = _number(depth * face.length_m, "mapped face area", nonnegative=True)
        mapped.append((face, depth, area, connected, fraction))
    try:
        discrete_area = _number(math.fsum(item[2] for item in mapped), "discrete wet area", nonnegative=True)
    except OverflowError as error:
        raise InletError("Discrete wet area exceeds numerical range") from error
    if Q > 0 and (geometry.area_m2 <= 0 or discrete_area <= 0):
        raise InletError("Positive discharge has no represented wet area; refine or revise the face map")
    # Log-scaled conveyance weights avoid overflow for large depth/n ratios.
    logs = [(2 / 3) * math.log(depth) - math.log(face.manning_n or spec.manning_n)
            if depth > 0 else -math.inf for face, depth, *_ in mapped]
    maximum = max(logs)
    weights = ([1. if item[1] > 0 else 0. for item in mapped] if spec.allocation == "uniform"
               else [math.exp(value - maximum) if value != -math.inf else 0. for value in logs])
    denominator = math.fsum(item[2] * weight for item, weight in zip(mapped, weights))
    if Q > 0 and denominator <= 0:
        raise InletError("Represented conveyance is below numerical range")
    scale = _number(Q / denominator, "velocity normalization", nonnegative=True) if Q > 0 else 0.
    states = []
    for (face, depth, _, connected, fraction), weight in zip(mapped, weights):
        velocity = _number(scale * weight, "face velocity", nonnegative=True)
        normal_momentum = _number(depth * velocity, "face normal momentum", nonnegative=True)
        states.append(FaceState(face, depth, normal_momentum * face.inward_normal[0],
                                normal_momentum * face.inward_normal[1], velocity, fraction, connected))
    try:
        state_q = math.fsum((state.hu * state.face.inward_normal[0] +
                            state.hv * state.face.inward_normal[1]) * state.face.length_m
                           for state in states)
    except OverflowError as error:
        raise InletError("Integrated state discharge exceeds numerical range") from error
    if not math.isfinite(state_q) or (Q > 0 and abs(state_q / Q - 1) > 1e-12):
        raise InletError("Mapped state discharge cannot satisfy the requested numerical precision")
    return InletTarget(spec.bc_id, time, spec.mode, spec.allocation, stage, Q, state_q,
                       geometry.area_m2, discrete_area,
                       geometry.area_m2 / discrete_area if discrete_area > 0 else None,
                       Q / geometry.area_m2 if Q > 0 else 0., tuple(states), solution)


def build_inlet_targets(specs: Iterable[InletSpec], sections: Mapping[str, Section],
                        faces_by_id: Mapping[str, Iterable[Face]], time_s: float) -> dict[str, InletTarget]:
    """Build multiple independent IDs, checking global ownership before any Q."""
    requests = tuple(specs)
    ids = [spec.bc_id for spec in requests if isinstance(spec, InletSpec)]
    if len(ids) != len(requests) or not ids or len(set(ids)) != len(ids):
        raise InletError("Expected nonempty inlet requests with unique IDs")
    if set(faces_by_id) != set(ids) or any(bc_id not in sections for bc_id in ids):
        raise InletError("Every requested inlet needs its section and exactly one face map")
    maps = {bc_id: tuple(faces) for bc_id, faces in faces_by_id.items()}
    register_face_ownership(maps)
    by_id = {spec.bc_id: spec for spec in requests}
    return {bc_id: build_inlet_target(by_id[bc_id], sections[bc_id], maps[bc_id], time_s)
            for bc_id in sorted(ids)}
