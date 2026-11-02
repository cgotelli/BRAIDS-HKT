"""Common resident runtime consumers for qualified Q3 capacity vectors.

The constitutive owner is :class:`ResidentM1CapacityDispatch`; this module
does not duplicate any sediment formula.  It gives corrected MPM, corrected
Engelund--Hansen, the parameterized general relation, and the already-qualified
current-only Soulsby and Soulsby/Van Rijn direct-vector laws one downstream ABI
for cell diagnostics, reconstructed internal faces, explicit open-boundary
trace/owner roles and conservative Exner candidates.

This is an additive integration layer.  It has no CPU fallback and performs
no hidden backend or mesh-size selection.  The caller remains responsible for
the existing BRAIDS high-order reconstruction and passes its reconstructed
left/right vectors to :meth:`faces`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .m1_capacity_dispatch import (
    ResidentM1CapacityDispatch,
    ResidentM1CapacityResult,
)


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
SUPPORTED_LAWS = frozenset((2, 4, 8, 11, 13))
FACE_INTERNAL = 0
FACE_TRACE_INLET = 1
FACE_TRACE_OUTLET = 2
FACE_OWNER_OUTLET = 3
FACE_CLOSED = 4
FACE_ROLES = frozenset((FACE_INTERNAL, FACE_TRACE_INLET,
                        FACE_TRACE_OUTLET, FACE_OWNER_OUTLET, FACE_CLOSED))


@dataclass(frozen=True)
class ResidentScalarPoints:
    """One resident constitutive result and its accepted-state mask."""

    law: int
    result: ResidentM1CapacityResult
    bed_mass: object
    suspended_mass: object
    status: object
    active: object


@dataclass(frozen=True)
class ResidentScalarFaces:
    """One conservative bed-mass face flux per class and explicit role."""

    flux: object
    donor: object
    role: object
    status: object
    owner: object
    neighbour: object


@dataclass(frozen=True)
class ResidentScalarExnerCandidate:
    """Uncommitted Exner candidate; construction never mutates old mass."""

    mass: object
    delta: object
    status: object
    boundary_transfer: object
    conservation_residual: object


class ResidentScalarCapacityRuntime:
    """Law-neutral resident consumers for qualified laws 2, 4, 8, 11 and 13.

    The historical class name is retained for compatibility.  Face transport
    consumes the constitutive vector directly, so no scalar-direction
    assumption is made for direct-vector laws 4 and 13.
    """

    def __init__(
        self, law: int, *, d50, d90, solid_density,
        parameters: Mapping[str, object], water_density: float = 1000.0,
        gravity: float = 9.81, roughness_mode: str | int = "manning",
        minimum_work_depth: float = 0.0,
        waqua_white_colebrook: bool = False,
        source_path: str | Path | None = None,
        dispatch_source_path: str | Path | None = None,
        roughness_source_path: str | Path | None = None,
        delft_source_path: str | Path | None = None,
    ):
        import cupy as cp

        if isinstance(law, (bool, np.bool_)) or int(law) not in SUPPORTED_LAWS:
            raise ValueError(
                "Common capacity runtime supports qualified laws 2, 4, 8, 11 and 13"
            )
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.law = int(law)
        self.dispatch = ResidentM1CapacityDispatch(
            source_path=dispatch_source_path,
            roughness_source_path=roughness_source_path,
            delft_source_path=delft_source_path,
        )
        if self.dispatch.device_id != self.device_id:
            raise ValueError("Capacity dispatcher is resident on a different CUDA device")
        self.d50 = self._array("d50", d50, 1)
        self.nclass = int(self.d50.size)
        if self.nclass <= 0:
            raise ValueError("Scalar runtime requires at least one sediment class")
        self.d90 = self._array("d90", d90, 1, (self.nclass,))
        self.solid_density = self._array(
            "solid_density", solid_density, 1, (self.nclass,))
        if not isinstance(parameters, Mapping):
            raise TypeError("Scalar runtime parameters must be a mapping")
        self.parameters = MappingProxyType(dict(parameters))
        if isinstance(water_density, (bool, np.bool_)) or not np.isfinite(
                float(water_density)) or float(water_density) <= 0.0:
            raise ValueError("water_density must be finite and positive")
        if isinstance(gravity, (bool, np.bool_)) or not np.isfinite(
                float(gravity)) or float(gravity) <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if (not np.isfinite(float(minimum_work_depth)) or
                float(minimum_work_depth) < 0.0):
            raise ValueError("minimum_work_depth must be finite and nonnegative")
        self.water_density = float(water_density)
        self.gravity = float(gravity)
        self.roughness_mode = roughness_mode
        self.minimum_work_depth = float(minimum_work_depth)
        self.waqua_white_colebrook = bool(waqua_white_colebrook)

        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        names = ("q3_scalar_face_dispatch", "q3_scalar_exner_candidate")
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=names,
            )
            self.kernels = {name: self.module.get_function(name) for name in names}
        self._maximum_waterdepth_operator = None

    def _array(self, name: str, value: object, ndim: int,
               shape: tuple[int, ...] | None = None, *, integer: bool = False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value

    def _evaluate(self, *, depth, u, v, roughness_parameter,
                  kinematic_viscosity, active, mixture_d90=None, hiding=None
                  ) -> ResidentScalarPoints:
        cp = self.cp
        depth = self._array("depth", depth, 1)
        count = int(depth.size)
        u = self._array("u", u, 1, (count,))
        v = self._array("v", v, 1, (count,))
        roughness_parameter = self._array(
            "roughness_parameter", roughness_parameter, 1, (count,))
        kinematic_viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, 1, (count,))
        active = self._array("active", active, 1, (count,), integer=True)
        if mixture_d90 is not None:
            mixture_d90 = self._array("mixture_d90", mixture_d90, 1, (count,))
        if hiding is not None:
            hiding = self._array(
                "hiding", hiding, 2, (self.nclass, count))
        result = self.dispatch.evaluate(
            self.law, depth=depth, u=u, v=v,
            roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            d50=self.d50, d90=self.d90, solid_density=self.solid_density,
            parameters=self.parameters, roughness_mode=self.roughness_mode,
            water_density=self.water_density, gravity=self.gravity,
            minimum_work_depth=self.minimum_work_depth,
            waqua_white_colebrook=self.waqua_white_colebrook,
            mixture_d90=mixture_d90, hiding=hiding, reject_invalid=False,
        )
        mask = active != 0
        status = cp.maximum(
            result.status, ((active != 0) & (active != 1)).astype(cp.int32))
        bed = cp.ascontiguousarray(cp.where(
            mask[None, None, :], result.bed_mass, 0.0))
        suspended = cp.ascontiguousarray(cp.where(
            mask[None, None, :], result.suspended_mass, 0.0))
        return ResidentScalarPoints(
            self.law, result, bed, suspended,
            cp.ascontiguousarray(status.astype(cp.int32, copy=False)), active,
        )

    def _characteristic_velocity(self, *, state, wet,
                                 maximum_waterdepth: bool,
                                 support_offset, support_depth, support_active):
        """Resolve dwnvel's optional max-depth divisor on the device."""
        cp = self.cp
        if not isinstance(maximum_waterdepth, (bool, np.bool_)):
            raise TypeError("maximum_waterdepth must be Boolean")
        if not maximum_waterdepth:
            if any(value is not None for value in (
                    support_offset, support_depth, support_active)):
                raise ValueError(
                    "MaximumWaterdepth-off rejects inert support topology")
            safe_depth = cp.where(wet, state[0], 1.0)
            return (
                cp.ascontiguousarray(cp.where(wet, state[1]/safe_depth, 0.0)),
                cp.ascontiguousarray(cp.where(wet, state[2]/safe_depth, 0.0)),
                cp.zeros(int(state.shape[1]), dtype=cp.int32),
            )
        if any(value is None for value in (
                support_offset, support_depth, support_active)):
            raise ValueError(
                "MaximumWaterdepth requires offset, depth and activity support")
        if self._maximum_waterdepth_operator is None:
            from .maximum_waterdepth_v66 import MaximumWaterdepthCUDAv66
            self._maximum_waterdepth_operator = MaximumWaterdepthCUDAv66()
            if self._maximum_waterdepth_operator.device_id != self.device_id:
                raise RuntimeError(
                    "MaximumWaterdepth dependency selected a different CUDA device")
        result = self._maximum_waterdepth_operator.apply(
            cell_depth=cp.ascontiguousarray(cp.where(wet, state[0], 0.0)),
            transport_u=cp.ascontiguousarray(cp.where(wet, state[1], 0.0)),
            transport_v=cp.ascontiguousarray(cp.where(wet, state[2], 0.0)),
            active=cp.ascontiguousarray(wet.astype(cp.int32)),
            support_offset=support_offset,
            support_depth=support_depth,
            support_active=support_active,
            enabled=True,
            reject_invalid=False,
        )
        return result.characteristic_u, result.characteristic_v, result.status

    def cells(self, *, q, roughness_parameter, kinematic_viscosity, active,
              dry_tolerance: float, mixture_d90=None, hiding=None,
              maximum_waterdepth: bool = False,
              maximum_waterdepth_offset=None,
              maximum_waterdepth_support_depth=None,
              maximum_waterdepth_support_active=None,
              ) -> ResidentScalarPoints:
        """Evaluate the common capacity ABI from resident accepted cell state."""
        cp = self.cp
        q = self._array("q", q, 2)
        if q.shape[0] != 3:
            raise ValueError("q must have shape (3,ncell)")
        ncell = int(q.shape[1])
        active = self._array("active", active, 1, (ncell,), integer=True)
        if not np.isfinite(float(dry_tolerance)) or float(dry_tolerance) < 0.0:
            raise ValueError("dry_tolerance must be finite and nonnegative")
        wet = (q[0] > np.float64(dry_tolerance)) & (active != 0)
        depth = cp.ascontiguousarray(cp.where(wet, q[0], 0.0))
        u, v, maximum_depth_status = self._characteristic_velocity(
            state=q, wet=wet, maximum_waterdepth=maximum_waterdepth,
            support_offset=maximum_waterdepth_offset,
            support_depth=maximum_waterdepth_support_depth,
            support_active=maximum_waterdepth_support_active,
        )
        result = self._evaluate(
            depth=depth, u=u, v=v, roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            active=cp.ascontiguousarray(wet.astype(cp.int32)),
            mixture_d90=mixture_d90, hiding=hiding,
        )
        # Inactive AMR parents may legitimately retain a finite wet state;
        # inactivity suppresses evaluation but is not malformed physics.
        # Only an actually dry accepted state carrying momentum is rejected.
        malformed = (~cp.all(cp.isfinite(q), axis=0)) | (q[0] < 0.0) | \
            ((active != 0) & (active != 1)) | (
            (q[0] <= np.float64(dry_tolerance)) &
            ((q[1] != 0.0) | (q[2] != 0.0)))
        status = cp.maximum(
            result.status,
            cp.maximum(maximum_depth_status, malformed.astype(cp.int32)))
        return ResidentScalarPoints(
            self.law, result.result, result.bed_mass, result.suspended_mass,
            cp.ascontiguousarray(status), result.active,
        )

    def traces(self, *, trace, owner, role, roughness_parameter,
               kinematic_viscosity, dry_tolerance: float,
               mixture_d90=None, hiding=None,
               maximum_waterdepth: bool = False,
               maximum_waterdepth_offset=None,
               maximum_waterdepth_support_depth=None,
               maximum_waterdepth_support_active=None) -> ResidentScalarPoints:
        """Evaluate only explicitly trace-owned inlet/outlet capacities."""
        cp = self.cp
        trace = self._array("trace", trace, 2)
        if trace.shape[0] != 3:
            raise ValueError("trace must have shape (3,nface)")
        nface = int(trace.shape[1])
        owner = self._array("owner", owner, 1, (nface,), integer=True)
        role = self._array("role", role, 1, (nface,), integer=True)
        roughness_parameter = self._array(
            "roughness_parameter", roughness_parameter, 1)
        ncell = int(roughness_parameter.size)
        kinematic_viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, 1, (ncell,))
        if not np.isfinite(float(dry_tolerance)) or float(dry_tolerance) < 0.0:
            raise ValueError("dry_tolerance must be finite and nonnegative")
        trace_role = (role == FACE_TRACE_INLET) | (role == FACE_TRACE_OUTLET)
        owner_ok = (owner >= 0) & (owner < ncell)
        safe_owner = cp.clip(owner, 0, max(ncell-1, 0))
        wet = trace_role & owner_ok & (trace[0] > np.float64(dry_tolerance))
        u, v, maximum_depth_status = self._characteristic_velocity(
            state=trace, wet=wet, maximum_waterdepth=maximum_waterdepth,
            support_offset=maximum_waterdepth_offset,
            support_depth=maximum_waterdepth_support_depth,
            support_active=maximum_waterdepth_support_active,
        )
        kwargs = {}
        if self.law == 2:
            mixture_d90 = self._array(
                "mixture_d90", mixture_d90, 1, (ncell,))
            kwargs["mixture_d90"] = cp.ascontiguousarray(mixture_d90[safe_owner])
        elif mixture_d90 is not None:
            raise ValueError(f"Law {self.law} does not consume mixture_d90")
        if hiding is not None:
            hiding = self._array(
                "hiding", hiding, 2, (self.nclass, ncell))
            kwargs["hiding"] = cp.ascontiguousarray(hiding[:, safe_owner])
        result = self._evaluate(
            depth=cp.ascontiguousarray(cp.where(wet, trace[0], 0.0)),
            u=u, v=v,
            roughness_parameter=cp.ascontiguousarray(
                roughness_parameter[safe_owner]),
            kinematic_viscosity=cp.ascontiguousarray(
                kinematic_viscosity[safe_owner]),
            active=cp.ascontiguousarray(wet.astype(cp.int32)), **kwargs,
        )
        role_invalid = ~cp.isin(role, cp.asarray(tuple(FACE_ROLES), dtype=cp.int32))
        malformed = role_invalid | (trace_role & ~owner_ok) | (
            trace_role & ((trace[0] < 0.0) |
                          ~cp.all(cp.isfinite(trace), axis=0)))
        status = cp.maximum(
            result.status,
            cp.maximum(maximum_depth_status, malformed.astype(cp.int32)))
        return ResidentScalarPoints(
            self.law, result.result, result.bed_mass, result.suspended_mass,
            cp.ascontiguousarray(status), result.active,
        )

    def faces(self, *, cells: ResidentScalarPoints,
              traces: ResidentScalarPoints, owner, neighbour, role, normal,
              availability, left_bed=None, right_bed=None
              ) -> ResidentScalarFaces:
        """Consume reconstructed cells and explicit boundary trace/owner roles."""
        cp = self.cp
        if cells.law != self.law or traces.law != self.law:
            raise ValueError("Cell/trace capacity law differs from runtime law")
        ncell = int(cells.bed_mass.shape[2])
        owner = self._array("owner", owner, 1, integer=True)
        nface = int(owner.size)
        neighbour = self._array(
            "neighbour", neighbour, 1, (nface,), integer=True)
        role = self._array("role", role, 1, (nface,), integer=True)
        normal = self._array("normal", normal, 2, (2, nface))
        availability = self._array(
            "availability", availability, 2, (self.nclass, ncell))
        if traces.bed_mass.shape != (2, self.nclass, nface):
            raise ValueError("Trace capacity count differs from face topology")
        safe_owner = cp.clip(owner, 0, max(ncell-1, 0))
        safe_neighbour = cp.clip(neighbour, 0, max(ncell-1, 0))
        if left_bed is None:
            left_bed = cp.ascontiguousarray(cells.bed_mass[:, :, safe_owner])
        else:
            left_bed = self._array(
                "left_bed", left_bed, 3, (2, self.nclass, nface))
        if right_bed is None:
            right_bed = cp.ascontiguousarray(cells.bed_mass[:, :, safe_neighbour])
        else:
            right_bed = self._array(
                "right_bed", right_bed, 3, (2, self.nclass, nface))
        flux = cp.zeros((self.nclass, nface), dtype=cp.float64)
        donor = cp.zeros((self.nclass, nface), dtype=cp.int32)
        status = cp.zeros(nface, dtype=cp.int32)
        with cp.cuda.Device(self.device_id):
            if nface:
                self.kernels["q3_scalar_face_dispatch"](
                    ((nface+255)//256,), (256,), (
                        np.int32(ncell), np.int32(nface), np.int32(self.nclass),
                        owner, neighbour, role, normal, left_bed, right_bed,
                        cells.bed_mass, traces.bed_mass, availability,
                        flux, donor, status,
                    ))
            left_status = cells.status[safe_owner]
            right_status = cp.where(
                neighbour >= 0, cells.status[safe_neighbour], 0)
            trace_status = cp.where(
                (role == FACE_TRACE_INLET) | (role == FACE_TRACE_OUTLET),
                traces.status, 0)
            topology_status = ((owner < 0) | (owner >= ncell) |
                               (neighbour < -1) | (neighbour >= ncell)).astype(cp.int32)
            status = cp.maximum(status, cp.maximum(
                topology_status, cp.maximum(
                    left_status, cp.maximum(right_status, trace_status))))
            flux = cp.where(status[None, :] == 0, flux, 0.0)
            donor = cp.where(status[None, :] == 0, donor, 0)
        return ResidentScalarFaces(
            cp.ascontiguousarray(flux), cp.ascontiguousarray(donor), role,
            cp.ascontiguousarray(status.astype(cp.int32, copy=False)),
            owner, neighbour,
        )

    def exner_candidate(self, *, old_mass, faces: ResidentScalarFaces,
                        offset, adjacency_faces, signs, face_length, area,
                        dt: float, morphological_factor: float
                        ) -> ResidentScalarExnerCandidate:
        """Build, but do not commit, one conservative accepted-state candidate."""
        cp = self.cp
        old_mass = self._array("old_mass", old_mass, 2)
        if old_mass.shape[0] != self.nclass:
            raise ValueError("old_mass class count differs")
        ncell = int(old_mass.shape[1])
        nface = int(faces.flux.shape[1])
        offset = self._array("offset", offset, 1, (ncell+1,), integer=True)
        adjacency_faces = self._array(
            "adjacency_faces", adjacency_faces, 1, integer=True)
        nadjacency = int(adjacency_faces.size)
        signs = self._array("signs", signs, 1, (nadjacency,), integer=True)
        face_length = self._array("face_length", face_length, 1, (nface,))
        area = self._array("area", area, 1, (ncell,))
        if not np.isfinite(float(dt)) or float(dt) < 0.0:
            raise ValueError("dt must be finite and nonnegative")
        if (not np.isfinite(float(morphological_factor)) or
                float(morphological_factor) < 0.0):
            raise ValueError("morphological_factor must be finite and nonnegative")
        candidate = cp.empty_like(old_mass)
        delta = cp.empty_like(old_mass)
        status = cp.zeros(ncell, dtype=cp.int32)
        with cp.cuda.Device(self.device_id):
            if ncell:
                self.kernels["q3_scalar_exner_candidate"](
                    ((ncell+255)//256,), (256,), (
                        np.int32(ncell), np.int32(nface),
                        np.int32(nadjacency), np.int32(self.nclass),
                        offset, adjacency_faces, signs, face_length, area,
                        faces.flux, faces.status, old_mass, np.float64(dt),
                        np.float64(morphological_factor), candidate, delta,
                        status,
                    ))
            # The conservative ledger is derived independently from the
            # candidate.  Internal faces cancel because both cells carry the
            # same shared face value with opposite adjacency signs.
            integrated_delta = cp.sum(delta*area[None, :], axis=1)
            # Boundary transfer is supplied by the adjacency imbalance: each
            # boundary face appears once, an internal face twice with signs
            # +1/-1.  bincount is deterministic for these compact diagnostics.
            multiplicity = cp.bincount(
                adjacency_faces, weights=signs.astype(cp.float64),
                minlength=nface)
            boundary_transfer = (np.float64(dt)*np.float64(morphological_factor)*
                                 cp.sum(faces.flux*face_length[None, :]*
                                        multiplicity[None, :], axis=1))
            residual = integrated_delta + boundary_transfer
        return ResidentScalarExnerCandidate(
            candidate, delta, status, boundary_transfer, residual)

    @staticmethod
    def diagnostic(points: ResidentScalarPoints) -> Mapping[str, object]:
        """Expose the exact arrays consumed downstream; no recomputation."""
        result = points.result
        return MappingProxyType({
            "bed_mass": points.bed_mass,
            "suspended_mass": points.suspended_mass,
            "absolute_shields": result.absolute_shields,
            "critical_shields": result.critical_shields,
            "stress_ratio": result.stress_ratio,
            "formula_auxiliary": result.formula_auxiliary,
            "status": points.status,
        })


__all__ = [
    "FACE_INTERNAL", "FACE_TRACE_INLET", "FACE_TRACE_OUTLET",
    "FACE_OWNER_OUTLET", "FACE_CLOSED", "SUPPORTED_LAWS", "ResidentScalarPoints",
    "ResidentScalarFaces", "ResidentScalarExnerCandidate",
    "ResidentScalarCapacityRuntime",
]
