"""Q3-H4 face/boundary composition for classic FLOW bed-slope methods.

This additive layer closes the insertion seam between the accepted scalar
capacity ABI and conservative Exner.  It selects the same raw face candidate
as ``braids_scalar_capacity_face`` without applying inventory availability,
executes exactly one authenticated FLOW slope correction, then chooses the
inventory donor from the *corrected* direction.  Open-boundary roles remain
explicit and no backend is selected or substituted automatically.

The resident implementation calls :class:`FlowBedSlopeCUDA`; the portable
implementation is a bounded CPU oracle.  Neither class is a live-controller
admission by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from braids_core.sediment.bed_slope_execution import (
    validate_bed_slope_execution_document,
)
from .flow_bed_slope_runtime import (
    FlowBedSlopeCUDA,
    STATUS_INVALID_INPUT,
    STATUS_INVALID_OUTPUT,
    STATUS_OK,
    STATUS_SINGULAR,
)
from .scalar_capacity_runtime import (
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_ROLES,
    FACE_TRACE_INLET,
    FACE_TRACE_OUTLET,
)


DONOR_NONE = 0
DONOR_OWNER = 1
DONOR_NEIGHBOUR = 2
DONOR_TRACE = 3


@dataclass(frozen=True)
class FlowBedSlopeFaceResult:
    """One slope-corrected shared face flux per class.

    ``raw_donor`` records only which reconstructed vector formed the raw
    candidate.  ``donor`` is the inventory owner selected from the corrected
    normal direction and is therefore the only donor used by availability.
    """

    backend: str
    method_id: str
    execution_fingerprint: str
    raw_transport: object
    corrected_transport: object
    flux: object
    raw_donor: object
    donor: object
    class_status: object
    status: object
    role: object
    owner: object
    neighbour: object


@dataclass(frozen=True)
class _PortableDirectionResult:
    transport_x: np.ndarray
    transport_y: np.ndarray
    status: np.ndarray


def _portable_direction(
    transport_x: np.ndarray,
    transport_y: np.ndarray,
    downhill_x: np.ndarray,
    downhill_y: np.ndarray,
    stress: np.ndarray,
    depth: np.ndarray,
    d50: np.ndarray,
    mean_diameter: np.ndarray,
    hiding_left: np.ndarray,
    hiding_right: np.ndarray,
    *,
    mode: int,
    controls: Mapping[str, float],
) -> _PortableDirectionResult:
    """Vectorised FP64 oracle for the already-qualified point primitive."""
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (
        transport_x, transport_y, downhill_x, downhill_y, stress, depth, d50,
        mean_diameter, hiding_left, hiding_right,
    ))
    if any(value.ndim != 1 for value in arrays):
        raise TypeError("portable FLOW bed-slope inputs must be 1-D FP64 arrays")
    count = arrays[0].size
    if any(value.shape != (count,) for value in arrays):
        raise ValueError("portable FLOW bed-slope input shapes differ")
    qx, qy, down_x, down_y, stress, depth, d50, mean_diameter, hide_l, hide_r = arrays
    corrected_x = qx.copy()
    corrected_y = qy.copy()
    status = np.full(count, STATUS_INVALID_INPUT, dtype=np.int32)

    valid = (np.isfinite(qx) & np.isfinite(qy) & np.isfinite(down_x) &
             np.isfinite(down_y) & np.isfinite(stress) & (stress >= 0.0) &
             np.isfinite(controls["alfabs"]))
    if mode == 2:
        valid &= np.isfinite(controls["alfabn"])
    elif mode == 3:
        valid &= (np.isfinite(depth) & np.isfinite(d50) &
                  np.isfinite(mean_diameter) & (depth > 0.0) & (d50 > 0.0) &
                  (mean_diameter > 0.0) & np.isfinite(controls["ashld"]) &
                  np.isfinite(controls["bshld"]) &
                  np.isfinite(controls["cshld"]) &
                  np.isfinite(controls["dshld"]))
    elif mode == 4:
        valid &= (np.isfinite(hide_l) & np.isfinite(hide_r) &
                  (hide_l >= 0.0) & (hide_r >= 0.0) &
                  np.isfinite(controls["alfpa"]) &
                  np.isfinite(controls["thcrpa"]) &
                  (controls["thcrpa"] >= 0.0))
    else:  # An authenticated identity can never reach this branch.
        raise ValueError("portable FLOW bed-slope mode must be 2, 3 or 4")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude = np.sqrt(qx*qx + qy*qy)
    valid &= np.isfinite(magnitude)
    status[valid] = STATUS_OK
    active = valid & (magnitude > 1.0e-6)
    if not np.any(active):
        return _PortableDirectionResult(corrected_x, corrected_y, status)

    # FLOW evaluates 30.0/180.0 and 0.9 in default real before promoting to
    # FP64.  Retain that source-level rounding in the portable oracle.
    tan_phi = np.tan(
        np.float64(np.float32(30.0) / np.float32(180.0)) * np.pi
    )
    cap = np.float64(np.float32(0.9)) * tan_phi
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dzds = down_x*qx/magnitude + down_y*qy/magnitude
        dzdn = -down_x*qy/magnitude + down_y*qx/magnitude
    dzds = np.minimum(cap, dzds)

    if mode == 2:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bagnold = tan_phi / (
                np.cos(np.arctan(dzds)) * (tan_phi-dzds)
            )
            along = np.maximum(
                0.0, 1.0 + controls["alfabs"]*(bagnold-1.0)
            )
            normal = controls["alfabn"]*dzdn
            above = stress >= 1.0
            normal[above] *= np.sqrt(1.0/stress[above])
            candidate_x = along*qx - along*qy*normal
            candidate_y = along*qy + along*qx*normal
        corrected_x[active] = candidate_x[active]
        corrected_y[active] = candidate_y[active]
    else:
        ftheta = np.zeros(count, dtype=np.float64)
        stressed = active & (stress != 0.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if mode == 3:
                ftheta[stressed] = (
                    controls["ashld"] * stress[stressed]**controls["bshld"] *
                    (d50[stressed]/depth[stressed])**controls["cshld"] *
                    (d50[stressed]/mean_diameter[stressed])**controls["dshld"]
                )
            else:
                local_hiding = 0.5*(hide_l+hide_r)
                denominator = np.maximum(stress*0.1,
                                         local_hiding*controls["thcrpa"])
                singular_denom = stressed & (
                    ~np.isfinite(denominator) | (denominator <= 0.0)
                )
                status[singular_denom] = STATUS_SINGULAR
                active &= ~singular_denom
                stressed &= ~singular_denom
                ftheta[stressed] = controls["alfpa"]*np.sqrt(
                    stress[stressed]/denominator[stressed]
                )
            has_slope = (down_x != 0.0) | (down_y != 0.0)
            direction_x = np.where(
                has_slope, ftheta*qx/magnitude+down_x, qx/magnitude,
            )
            direction_y = np.where(
                has_slope, ftheta*qy/magnitude+down_y, qy/magnitude,
            )
            norm = np.sqrt(direction_x*direction_x+direction_y*direction_y)
        singular = active & (~np.isfinite(norm) | (norm <= np.finfo(np.float64).tiny))
        status[singular] = STATUS_SINGULAR
        active &= ~singular
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            adjusted_magnitude = magnitude*(1.0+controls["alfabs"]*dzds)
            candidate_x = adjusted_magnitude*direction_x/norm
            candidate_y = adjusted_magnitude*direction_y/norm
        corrected_x[active] = candidate_x[active]
        corrected_y[active] = candidate_y[active]

    invalid_output = ((status == STATUS_OK) &
                      (~np.isfinite(corrected_x) | ~np.isfinite(corrected_y)))
    corrected_x[invalid_output] = qx[invalid_output]
    corrected_y[invalid_output] = qy[invalid_output]
    status[invalid_output] = STATUS_INVALID_OUTPUT
    return _PortableDirectionResult(corrected_x, corrected_y, status)


class _FaceComposerBase:
    def __init__(self, execution_identity: Mapping[str, object]):
        self.identity = validate_bed_slope_execution_document(execution_identity)
        self.method_id = str(self.identity["method_id"])
        self.mode = int(self.identity["runtime_mode"])
        self.nclass = len(self.identity["ordered_fraction_ids"])
        self.controls = dict(self.identity["parameters"])

    @staticmethod
    def _invalid_count(xp, status) -> int:
        value = xp.count_nonzero(status)
        return int(value.item() if hasattr(value, "item") else value)

    def _compose(self, xp, *, backend: str, evaluate, left_bed, right_bed,
                 owner_bed, trace_bed, owner, neighbour, role, normal,
                 availability, downhill, stress, depth, d50, mean_diameter,
                 hiding, reject_invalid: bool) -> FlowBedSlopeFaceResult:
        nface = int(owner.size)
        ncell = int(availability.shape[1])
        safe_neighbour = xp.clip(neighbour, 0, max(ncell-1, 0))

        internal = (role == FACE_INTERNAL) & (neighbour >= 0)
        closed = (role == FACE_INTERNAL) & (neighbour == -1)
        trace_in = role == FACE_TRACE_INLET
        trace_out = role == FACE_TRACE_OUTLET
        owner_out = role == FACE_OWNER_OUTLET

        left_normal = xp.sum(left_bed*normal[:, None, :], axis=0)
        right_normal = xp.sum(right_bed*normal[:, None, :], axis=0)
        raw_donor = xp.zeros((self.nclass, nface), dtype=xp.int32)
        choose_left = internal[None, :] & (left_normal > 0.0) & (right_normal > 0.0)
        choose_right = internal[None, :] & (left_normal < 0.0) & (right_normal < 0.0)
        central = internal[None, :] & ~choose_left & ~choose_right
        raw = xp.zeros_like(left_bed)
        raw = xp.where(choose_left[None, :, :], left_bed, raw)
        raw = xp.where(choose_right[None, :, :], right_bed, raw)
        raw = xp.where(central[None, :, :], 0.5*(left_bed+right_bed), raw)
        raw_donor = xp.where(choose_left, DONOR_OWNER, raw_donor)
        raw_donor = xp.where(choose_right, DONOR_NEIGHBOUR, raw_donor)
        central_normal = xp.sum(raw*normal[:, None, :], axis=0)
        raw_donor = xp.where(
            central & (central_normal < 0.0), DONOR_NEIGHBOUR, raw_donor,
        )
        raw_donor = xp.where(
            central & (central_normal >= 0.0), DONOR_OWNER, raw_donor,
        )
        trace_role = trace_in | trace_out
        raw = xp.where(trace_role[None, None, :], trace_bed, raw)
        raw = xp.where(owner_out[None, None, :], owner_bed, raw)
        raw_donor = xp.where(trace_role[None, :], DONOR_TRACE, raw_donor)
        raw_donor = xp.where(owner_out[None, :], DONOR_OWNER, raw_donor)

        owner_hiding = hiding[:, owner]
        neighbour_hiding = hiding[:, safe_neighbour]
        # FLOW averages the two internal cell factors.  Boundary traces do not
        # own an independent bed mixture in this ABI, so both inputs are the
        # explicitly mapped owner factor and their average remains the owner.
        hiding_left = owner_hiding
        hiding_right = xp.where(internal[None, :], neighbour_hiding, owner_hiding)
        d50_face = xp.broadcast_to(d50[:, None], (self.nclass, nface))

        # Delft executes the direction operator only when the selected raw
        # bed-load magnitude exceeds its epsilon.  Closed faces and genuinely
        # zero-capacity classes therefore must not be rejected merely because
        # they have no meaningful hydraulic depth or surface mean diameter.
        # Supply neutral finite values to the point primitive for those lanes;
        # participating nonzero lanes retain every authenticated input byte.
        participating = internal | trace_role | owner_out
        magnitude = xp.sqrt(xp.sum(raw*raw, axis=0))
        active_direction = participating[None, :] & (magnitude > 1.0e-6)
        safe_raw = xp.where(participating[None, None, :], raw, 0.0)
        safe_downhill = xp.where(
            active_direction[None, :, :],
            xp.broadcast_to(downhill[:, None, :], raw.shape), 0.0,
        )
        safe_stress = xp.where(active_direction, stress, 0.0)
        safe_depth = xp.where(active_direction, depth, 1.0)
        safe_mean = xp.where(active_direction, mean_diameter, 1.0)
        safe_hiding_left = xp.where(active_direction, hiding_left, 1.0)
        safe_hiding_right = xp.where(active_direction, hiding_right, 1.0)
        flattened = tuple(xp.ascontiguousarray(value.reshape(-1)) for value in (
            safe_raw[0], safe_raw[1], safe_downhill[0], safe_downhill[1],
            safe_stress, safe_depth, d50_face, safe_mean,
            safe_hiding_left, safe_hiding_right,
        ))
        direction = evaluate(*flattened)
        corrected = xp.ascontiguousarray(xp.stack((
            direction.transport_x.reshape(self.nclass, nface),
            direction.transport_y.reshape(self.nclass, nface),
        )))
        class_status = xp.ascontiguousarray(xp.where(
            participating[None, :],
            direction.status.reshape(self.nclass, nface), STATUS_OK,
        ).astype(xp.int32, copy=False))
        invalid_count = self._invalid_count(xp, class_status)
        if reject_invalid and invalid_count:
            raise ValueError(
                "FLOW face bed-slope composition rejected "
                f"{invalid_count} class-face state(s)"
            )
        # The accepted scalar/Exner ABI owns one status per shared face.  A
        # malformed class invalidates that entire shared face so no subset of
        # its class fluxes can enter a conservative candidate transaction.
        status = xp.ascontiguousarray(
            xp.max(class_status, axis=0).astype(xp.int32, copy=False)
        ) if nface else xp.empty(0, dtype=xp.int32)

        corrected_normal = xp.sum(corrected*normal[:, None, :], axis=0)
        donor = xp.zeros((self.nclass, nface), dtype=xp.int32)
        donor = xp.where(
            internal[None, :] & (corrected_normal >= 0.0), DONOR_OWNER, donor,
        )
        donor = xp.where(
            internal[None, :] & (corrected_normal < 0.0), DONOR_NEIGHBOUR, donor,
        )
        donor = xp.where(trace_role[None, :], DONOR_TRACE, donor)
        donor = xp.where(owner_out[None, :], DONOR_OWNER, donor)
        owner_availability = availability[:, owner]
        neighbour_availability = availability[:, safe_neighbour]
        selected_availability = xp.where(
            donor == DONOR_NEIGHBOUR, neighbour_availability, owner_availability,
        )
        flux = corrected_normal*selected_availability
        flux = xp.where(closed[None, :], 0.0, flux)
        flux = xp.where(trace_in[None, :], xp.minimum(0.0, flux), flux)
        flux = xp.where(
            (trace_out | owner_out)[None, :], xp.maximum(0.0, flux), flux,
        )
        flux = xp.where(status[None, :] == STATUS_OK, flux, 0.0)
        donor = xp.where(status[None, :] == STATUS_OK, donor, DONOR_NONE)
        return FlowBedSlopeFaceResult(
            backend=backend,
            method_id=self.method_id,
            execution_fingerprint=str(self.identity["fingerprint"]),
            raw_transport=xp.ascontiguousarray(raw),
            corrected_transport=corrected,
            flux=xp.ascontiguousarray(flux),
            raw_donor=xp.ascontiguousarray(raw_donor.astype(xp.int32, copy=False)),
            donor=xp.ascontiguousarray(donor.astype(xp.int32, copy=False)),
            class_status=class_status,
            status=status,
            role=role,
            owner=owner,
            neighbour=neighbour,
        )


class PortableFlowBedSlopeFaceComposer(_FaceComposerBase):
    """Strict contiguous-NumPy CPU oracle for the Q3-H4 component."""

    backend = "portable_cpu_oracle"

    @staticmethod
    def _array(name, value, ndim, shape=None, *, integer=False):
        dtype = np.int32 if integer else np.float64
        if (not isinstance(value, np.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a contiguous {ndim}-D {label} NumPy array"
            )
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        return value

    def compose(self, *, left_bed, right_bed, owner_bed, trace_bed, owner,
                neighbour, role, normal, availability, downhill, stress,
                depth, d50, mean_diameter, hiding,
                reject_invalid: bool = True) -> FlowBedSlopeFaceResult:
        values = self._validate_arrays(
            self._array, left_bed=left_bed, right_bed=right_bed,
            owner_bed=owner_bed, trace_bed=trace_bed, owner=owner,
            neighbour=neighbour, role=role, normal=normal,
            availability=availability, downhill=downhill, stress=stress,
            depth=depth, d50=d50, mean_diameter=mean_diameter, hiding=hiding,
            reject_invalid=reject_invalid,
        )

        def evaluate(*arrays):
            return _portable_direction(
                *arrays, mode=self.mode, controls=self.controls,
            )

        return self._compose(np, backend=self.backend, evaluate=evaluate, **values)

    def _validate_arrays(self, array, **values):
        return _validated_inputs(self, array, np, **values)


class ResidentFlowBedSlopeFaceComposer(_FaceComposerBase):
    """Resident actual-CUDA Q3-H4 component with no CPU fallback."""

    backend = "implicit_fwave_midpoint_gpu_component"

    def __init__(self, execution_identity: Mapping[str, object], *,
                 operator: FlowBedSlopeCUDA | None = None):
        super().__init__(execution_identity)
        self.operator = operator if operator is not None else FlowBedSlopeCUDA()
        self.cp = self.operator.cp
        self.device_id = self.operator.device_id

    def _array(self, name, value, ndim, shape=None, *, integer=False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        return value

    def compose(self, *, left_bed, right_bed, owner_bed, trace_bed, owner,
                neighbour, role, normal, availability, downhill, stress,
                depth, d50, mean_diameter, hiding,
                reject_invalid: bool = True) -> FlowBedSlopeFaceResult:
        values = _validated_inputs(
            self, self._array, self.cp, left_bed=left_bed,
            right_bed=right_bed, owner_bed=owner_bed, trace_bed=trace_bed,
            owner=owner, neighbour=neighbour, role=role, normal=normal,
            availability=availability, downhill=downhill, stress=stress,
            depth=depth, d50=d50, mean_diameter=mean_diameter, hiding=hiding,
            reject_invalid=reject_invalid,
        )

        def evaluate(*arrays):
            return self.operator.evaluate(
                *arrays, mode=self.mode, reject_invalid=False, **self.controls,
            )

        with self.cp.cuda.Device(self.device_id):
            return self._compose(
                self.cp, backend=self.backend, evaluate=evaluate, **values,
            )


def _validated_inputs(composer, array, xp, *, left_bed, right_bed, owner_bed,
                      trace_bed, owner, neighbour, role, normal, availability,
                      downhill, stress, depth, d50, mean_diameter, hiding,
                      reject_invalid):
    if not isinstance(reject_invalid, (bool, np.bool_)):
        raise TypeError("reject_invalid must be boolean")
    owner = array("owner", owner, 1, integer=True)
    nface = int(owner.size)
    neighbour = array("neighbour", neighbour, 1, (nface,), integer=True)
    role = array("role", role, 1, (nface,), integer=True)
    normal = array("normal", normal, 2, (2, nface))
    downhill = array("downhill", downhill, 2, (2, nface))
    d50 = array("d50", d50, 1, (composer.nclass,))
    availability = array("availability", availability, 2)
    if availability.shape[0] != composer.nclass or availability.shape[1] < 1:
        raise ValueError("availability must have shape (nclass,ncell), ncell >= 1")
    ncell = int(availability.shape[1])
    hiding = array("hiding", hiding, 2, (composer.nclass, ncell))
    vector_shape = (2, composer.nclass, nface)
    scalar_shape = (composer.nclass, nface)
    left_bed = array("left_bed", left_bed, 3, vector_shape)
    right_bed = array("right_bed", right_bed, 3, vector_shape)
    owner_bed = array("owner_bed", owner_bed, 3, vector_shape)
    trace_bed = array("trace_bed", trace_bed, 3, vector_shape)
    stress = array("stress", stress, 2, scalar_shape)
    depth = array("depth", depth, 2, scalar_shape)
    mean_diameter = array(
        "mean_diameter", mean_diameter, 2, scalar_shape,
    )

    finite_topology = (xp.all(xp.isfinite(normal)) &
                       xp.all(xp.isfinite(availability)) &
                       xp.all(availability >= 0.0))
    if not bool(finite_topology.item() if hasattr(finite_topology, "item")
                else finite_topology):
        raise ValueError("face normals and availability must be finite/nonnegative")
    valid_role = xp.isin(role, xp.asarray(tuple(FACE_ROLES), dtype=xp.int32))
    owner_ok = (owner >= 0) & (owner < ncell)
    neighbour_ok = (neighbour >= -1) & (neighbour < ncell)
    boundary_role = role != FACE_INTERNAL
    role_topology_ok = (~boundary_role) | (neighbour == -1)
    invalid_topology = ~valid_role | ~owner_ok | ~neighbour_ok | ~role_topology_ok
    if int(xp.count_nonzero(invalid_topology).item()):
        raise ValueError("face roles and owner/neighbour topology are inconsistent")

    return {
        "left_bed": left_bed, "right_bed": right_bed,
        "owner_bed": owner_bed, "trace_bed": trace_bed,
        "owner": owner, "neighbour": neighbour, "role": role,
        "normal": normal, "availability": availability,
        "downhill": downhill, "stress": stress, "depth": depth,
        "d50": d50, "mean_diameter": mean_diameter, "hiding": hiding,
        "reject_invalid": bool(reject_invalid),
    }


__all__ = [
    "DONOR_NONE", "DONOR_OWNER", "DONOR_NEIGHBOUR", "DONOR_TRACE",
    "FlowBedSlopeFaceResult", "PortableFlowBedSlopeFaceComposer",
    "ResidentFlowBedSlopeFaceComposer",
]
