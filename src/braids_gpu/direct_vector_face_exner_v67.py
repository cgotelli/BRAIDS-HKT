"""Law-neutral resident CUDA face, inventory, and Exner lifecycle.

This module is the downstream half of the Q3-H6 direct-vector family.  It
does not select or evaluate a transport formula.  Instead, an authenticated
producer supplies fraction-free bed-mass transport vectors and this consumer
owns, in order:

* the five explicit FLOW 2DH face roles;
* application of the accepted active-layer fraction at the face exactly once;
* one finite-inventory projection;
* a conservative, uncommitted Exner candidate; and
* commit, rollback, cancellation, and portable accepted-state restart.

The CUDA kernels are compiled directly from the already exercised
``scalar_capacity_runtime.cu`` face/Exner source and
``direct_vector_live_v66.cu`` digest/inventory source.  No constitutive law or
fake numeric formula identifier is instantiated.  All working arrays remain
on the caller-selected CUDA device; only explicit checkpoints cross to host.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from braids_core.sediment.classic_flow_taurat_v69 import (
    TAURAT_POSITIVE_ZERO,
)
from .flow_bed_slope_composition import ResidentFlowBedSlopeFaceComposer


SCHEMA = 1
SCOPE_ID = "braids-q3-h6-law-neutral-face-exner-v67-v1"
BACKEND = "implicit_fwave_midpoint_gpu"
INTEGRATOR = "implicit_fwave_midpoint"

FACE_INTERNAL = 0
FACE_TRACE_INLET = 1
FACE_TRACE_OUTLET = 2
FACE_OWNER_OUTLET = 3
FACE_CLOSED = 4
FACE_ROLES = frozenset((
    FACE_INTERNAL,
    FACE_TRACE_INLET,
    FACE_TRACE_OUTLET,
    FACE_OWNER_OUTLET,
    FACE_CLOSED,
))

CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAMES = (
    "q3_scalar_face_dispatch",
    "q3_scalar_exner_candidate",
    "direct_v66_digest_fp64",
    "direct_v66_inventory_fill",
)


class DirectVectorFaceExnerV67Error(ValueError):
    """The resident downstream contract or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorFaceExnerV67Error(
            "direct-vector v67 receipt is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DirectVectorFaceExnerV67Error(
            f"{name} must be a lowercase SHA-256 identity"
        )
    return value


def _finite(
    name: str,
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DirectVectorFaceExnerV67Error(f"{name} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        raise DirectVectorFaceExnerV67Error(
            f"{name} is outside its finite domain"
        )
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ResidentDirectVectorChannelsV67:
    """Authenticated fraction-free vectors from one constitutive producer."""

    execution_fingerprint: str
    producer_fingerprint: str
    point_role: str
    bed_mass_vector: object
    status: object
    surface_fraction_application_count: int
    channel_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorTopologyV67:
    """A semantically validated, reusable resident face topology receipt."""

    execution_fingerprint: str
    ncell: int
    nface: int
    owner: object
    neighbour: object
    role: object
    normal: object
    face_length: object
    cell_area: object
    offset: object
    adjacency_faces: object
    signs: object
    semantic_validation_count: int
    topology_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorFacesV67:
    """One shared conservative face flux for every class and face."""

    execution_fingerprint: str
    topology_sha256: str
    flux: object
    donor: object
    status: object
    face_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorSlopeFieldsV69:
    """Topology-bound exact-zero positive-TraFrm ISlope=2 fields."""

    extension_fingerprint: str
    slope_execution_fingerprint: str
    topology_sha256: str
    stress_semantics: str
    downhill: object
    stress: object
    depth: object
    d50: object
    mean_diameter: object
    hiding: object
    slope_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorStateV67:
    """Accepted morphodynamic state and cumulative boundary ledger."""

    execution_fingerprint: str
    bed_mass_kg_m2: object
    bulk_density_kg_m3: object
    active_fraction: object
    boundary_signed_kg: object
    boundary_absolute_kg: object
    accepted_step: int
    time_s: float
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorTrialV67:
    """An immutable candidate transaction; the parent state is untouched."""

    execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    cell_channel_sha256: str
    trace_channel_sha256: str
    dt_s: float
    morphological_factor: float
    provisional_faces: ResidentDirectVectorFacesV67
    faces: ResidentDirectVectorFacesV67
    inventory_fill: object
    accepted_face_availability: object
    cell_delta_kg_m2: object
    boundary_transfer_kg: object
    boundary_absolute_transfer_kg: object
    depletion_roundoff_transfer_kg: object
    conservation_residual_kg: object
    candidate: ResidentDirectVectorStateV67
    topology_semantic_validation_count: int
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str
    device_id: int
    slope_receipt_sha256: str | None = None


class ResidentDirectVectorFaceExnerV67:
    """Consume any authenticated direct-vector method without law dispatch."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(
        self,
        *,
        execution_fingerprint: str,
        bulk_density_kg_m3: object,
        accepted_step_limit: int = 20,
        face_exner_source_path: str | Path | None = None,
        inventory_source_path: str | Path | None = None,
        slope_composer: ResidentFlowBedSlopeFaceComposer | None = None,
        slope_extension_fingerprint: str | None = None,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution_fingerprint = _exact_sha256(
            "execution_fingerprint", execution_fingerprint
        )
        if slope_composer is None:
            if slope_extension_fingerprint is not None:
                raise DirectVectorFaceExnerV67Error(
                    "slope extension identity requires its resident composer"
                )
            self.slope_extension_fingerprint = None
        else:
            if not isinstance(slope_composer, ResidentFlowBedSlopeFaceComposer):
                raise DirectVectorFaceExnerV67Error(
                    "positive taurat slope requires the resident H4 composer"
                )
            if (slope_composer.device_id != self.device_id or
                    slope_composer.method_id !=
                    "bed_slope.delft_bagnold_ikeda" or
                    slope_composer.mode != 2):
                raise DirectVectorFaceExnerV67Error(
                    "positive taurat slope composer identity differs"
                )
            self.slope_extension_fingerprint = _exact_sha256(
                "slope_extension_fingerprint", slope_extension_fingerprint
            )
        self.slope_composer = slope_composer
        if (
            isinstance(accepted_step_limit, (bool, np.bool_))
            or type(accepted_step_limit) is not int
            or accepted_step_limit < 1
            or accepted_step_limit > 60
        ):
            raise DirectVectorFaceExnerV67Error(
                "accepted_step_limit must be an integer in [1, 60]"
            )
        self.accepted_step_limit = accepted_step_limit
        density = self._array(
            "bulk_density_kg_m3", bulk_density_kg_m3, 1, copy=True
        )
        if int(density.size) < 1 or not self._all(
            cp.isfinite(density) & (density > 0.0)
        ):
            raise DirectVectorFaceExnerV67Error(
                "bulk density must contain finite positive class values"
            )
        self.bulk_density = density
        self.nclass = int(density.size)

        face_path = Path(
            face_exner_source_path
            or Path(__file__).with_name("scalar_capacity_runtime.cu")
        ).resolve()
        inventory_path = Path(
            inventory_source_path
            or Path(__file__).with_name("direct_vector_live_v66.cu")
        ).resolve()
        face_source = face_path.read_text(encoding="utf-8")
        inventory_source = inventory_path.read_text(encoding="utf-8")
        self.source_receipt = MappingProxyType({
            "scope_id": SCOPE_ID,
            "face_exner_source": str(face_path),
            "face_exner_sha256": _sha256_file(face_path),
            "inventory_source": str(inventory_path),
            "inventory_sha256": _sha256_file(inventory_path),
            "kernels": KERNEL_NAMES,
        })
        # These source files own disjoint symbols and are compiled together so
        # the consumer cannot instantiate a constitutive dispatcher by accident.
        source = face_source + "\n" + inventory_source
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source,
                options=CUDA_OPTIONS,
                name_expressions=KERNEL_NAMES,
            )
            self.kernels = {
                name: module.get_function(name) for name in KERNEL_NAMES
            }

    def _array(
        self,
        name: str,
        value: object,
        ndim: int,
        shape: tuple[int, ...] | None = None,
        *,
        integer: bool = False,
        copy: bool = False,
    ):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != dtype
            or value.ndim != ndim
            or not value.flags.c_contiguous
        ):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} array"
            )
        if int(value.device.id) != self.device_id:
            raise DirectVectorFaceExnerV67Error(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise DirectVectorFaceExnerV67Error(
                f"{name} must have shape {shape}, got {tuple(value.shape)}"
            )
        return value.copy(order="C") if copy else value

    @staticmethod
    def _flag(value: object) -> bool:
        return bool(value.item() if hasattr(value, "item") else value)

    def _all(self, value: object) -> bool:
        return self._flag(self.cp.all(value))

    def _resident_digest(self, *arrays: object, label: str) -> str:
        cp = self.cp
        inventory: list[dict[str, object]] = []
        with cp.cuda.Device(self.device_id):
            digest = cp.zeros(2, dtype=cp.uint64)
            for ordinal, raw in enumerate(arrays):
                if (
                    not isinstance(raw, cp.ndarray)
                    or int(raw.device.id) != self.device_id
                ):
                    raise DirectVectorFaceExnerV67Error(
                        "resident receipt received a nonresident array"
                    )
                value = (
                    raw
                    if raw.dtype == cp.float64
                    else cp.ascontiguousarray(raw.astype(cp.float64))
                )
                value = cp.ascontiguousarray(value.reshape(-1))
                if int(value.size):
                    self.kernels["direct_v66_digest_fp64"](
                        ((int(value.size) + 255) // 256,),
                        (256,),
                        (
                            np.int64(value.size),
                            np.uint64(0x9E3779B97F4A7C15 ^ (ordinal + 1)),
                            value,
                            digest,
                        ),
                    )
                inventory.append({
                    "ordinal": ordinal,
                    "dtype": str(raw.dtype),
                    "shape": list(raw.shape),
                })
            words = (int(digest[0].item()), int(digest[1].item()))
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "label": label,
            "inventory": inventory,
            "resident_digest_u64": list(words),
        })

    def _channel_token(
        self,
        *,
        producer_fingerprint: str,
        point_role: str,
        vectors: object,
        status: object,
    ) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "producer_fingerprint": producer_fingerprint,
            "point_role": point_role,
            "surface_fraction_application_count": 0,
            "resident_arrays_sha256": self._resident_digest(
                vectors, status, label=f"{point_role}_fraction_free_channels"
            ),
        })

    def prepare_channels(
        self,
        *,
        point_role: str,
        bed_mass_vector: object,
        status: object,
        producer_fingerprint: str,
        surface_fraction_application_count: int = 0,
    ) -> ResidentDirectVectorChannelsV67:
        """Bind fraction-free producer vectors to this execution.

        A producer route that has already multiplied by surface fraction is
        rejected by receipt.  Callers must request a fraction-free route (or
        use unit fraction at the upstream composer) because accepted stock is
        owned here and is applied exactly once at the accepted face.
        """
        cp = self.cp
        if point_role not in ("cells", "traces"):
            raise DirectVectorFaceExnerV67Error(
                "point_role must be 'cells' or 'traces'"
            )
        if surface_fraction_application_count != 0:
            raise DirectVectorFaceExnerV67Error(
                "producer vectors must be fraction-free; accepted active "
                "fraction is owned by the face consumer"
            )
        producer = _exact_sha256("producer_fingerprint", producer_fingerprint)
        vectors = self._array("bed_mass_vector", bed_mass_vector, 3, copy=True)
        if vectors.shape[0] != 2 or vectors.shape[1] != self.nclass:
            raise DirectVectorFaceExnerV67Error(
                "bed_mass_vector must have shape (2,nclass,npoint)"
            )
        npoint = int(vectors.shape[2])
        accepted_status = self._array(
            "status", status, 1, (npoint,), integer=True, copy=True
        )
        if not self._all(cp.isfinite(vectors)) or not self._all(
            accepted_status >= 0
        ):
            raise DirectVectorFaceExnerV67Error(
                "producer vectors/status contain invalid values"
            )
        token = self._channel_token(
            producer_fingerprint=producer,
            point_role=point_role,
            vectors=vectors,
            status=accepted_status,
        )
        return ResidentDirectVectorChannelsV67(
            self.execution_fingerprint,
            producer,
            point_role,
            vectors,
            accepted_status,
            0,
            token,
            self.device_id,
        )

    def _validate_channels(
        self,
        channels: ResidentDirectVectorChannelsV67,
        *,
        point_role: str,
        npoint: int,
    ) -> None:
        if (
            not isinstance(channels, ResidentDirectVectorChannelsV67)
            or channels.execution_fingerprint != self.execution_fingerprint
            or channels.device_id != self.device_id
            or channels.point_role != point_role
            or channels.surface_fraction_application_count != 0
            or channels.bed_mass_vector.shape != (2, self.nclass, npoint)
            or channels.status.shape != (npoint,)
        ):
            raise DirectVectorFaceExnerV67Error(
                f"{point_role} channel receipt differs from execution"
            )
        expected = self._channel_token(
            producer_fingerprint=channels.producer_fingerprint,
            point_role=point_role,
            vectors=channels.bed_mass_vector,
            status=channels.status,
        )
        if expected != channels.channel_sha256:
            raise DirectVectorFaceExnerV67Error(
                f"{point_role} channel arrays are stale or mutated"
            )

    def _topology_token(self, *arrays: object, ncell: int, nface: int) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "ncell": ncell,
            "nface": nface,
            "face_roles": sorted(FACE_ROLES),
            "semantic_validation_count": 1,
            "resident_arrays_sha256": self._resident_digest(
                *arrays, label="validated_face_topology"
            ),
        })

    def prepare_topology(
        self,
        *,
        owner: object,
        neighbour: object,
        role: object,
        normal: object,
        face_length: object,
        cell_area: object,
        offset: object,
        adjacency_faces: object,
        signs: object,
    ) -> ResidentDirectVectorTopologyV67:
        """Validate all five roles and adjacency ownership once.

        The returned copied arrays are an authenticated topology receipt.
        Later steps only verify its resident content digest; they do not
        repeat the semantic topology walk.
        """
        cp = self.cp
        owner = self._array("owner", owner, 1, integer=True, copy=True)
        nface = int(owner.size)
        neighbour = self._array(
            "neighbour", neighbour, 1, (nface,), integer=True, copy=True
        )
        role = self._array("role", role, 1, (nface,), integer=True, copy=True)
        normal = self._array(
            "normal", normal, 2, (2, nface), copy=True
        )
        length = self._array(
            "face_length", face_length, 1, (nface,), copy=True
        )
        area = self._array("cell_area", cell_area, 1, copy=True)
        ncell = int(area.size)
        if ncell < 1:
            raise DirectVectorFaceExnerV67Error(
                "topology requires at least one cell"
            )
        offset = self._array(
            "offset", offset, 1, (ncell + 1,), integer=True, copy=True
        )
        adjacency = self._array(
            "adjacency_faces", adjacency_faces, 1, integer=True, copy=True
        )
        signs = self._array(
            "signs", signs, 1, (int(adjacency.size),), integer=True, copy=True
        )
        known_role = cp.isin(role, cp.asarray(sorted(FACE_ROLES), dtype=cp.int32))
        if (
            not self._all(known_role)
            or not self._all((role == FACE_INTERNAL) == (neighbour >= 0))
            or not self._all((owner >= 0) & (owner < ncell))
            or not self._all((neighbour >= -1) & (neighbour < ncell))
            or not self._all(
                (neighbour < 0) | (neighbour != owner)
            )
            or not self._all(cp.isfinite(normal))
            or not self._all(cp.isfinite(length) & (length >= 0.0))
            or not self._all(cp.isfinite(area) & (area > 0.0))
        ):
            raise DirectVectorFaceExnerV67Error(
                "face topology/role/geometry contract differs"
            )
        if (
            int(offset[0].item()) != 0
            or int(offset[-1].item()) != int(adjacency.size)
            or not self._all(offset[1:] >= offset[:-1])
            or not self._all((adjacency >= 0) & (adjacency < nface))
            or not self._all((signs == 1) | (signs == -1))
        ):
            raise DirectVectorFaceExnerV67Error(
                "face adjacency inventory differs"
            )
        cell_ids = cp.repeat(
            cp.arange(ncell, dtype=cp.int32), cp.diff(offset)
        )
        adjacent_owner = owner[adjacency]
        adjacent_neighbour = neighbour[adjacency]
        incidence_ok = (
            ((cell_ids == adjacent_owner) & (signs == 1))
            | ((cell_ids == adjacent_neighbour) & (signs == -1))
        )
        incidence_count = cp.bincount(adjacency, minlength=nface)
        expected_count = cp.where(role == FACE_INTERNAL, 2, 1)
        if not self._all(incidence_ok) or not self._all(
            incidence_count == expected_count
        ):
            raise DirectVectorFaceExnerV67Error(
                "face adjacency ownership/sign contract differs"
            )
        arrays = (
            owner,
            neighbour,
            role,
            normal,
            length,
            area,
            offset,
            adjacency,
            signs,
        )
        token = self._topology_token(*arrays, ncell=ncell, nface=nface)
        return ResidentDirectVectorTopologyV67(
            self.execution_fingerprint,
            ncell,
            nface,
            *arrays,
            1,
            token,
            self.device_id,
        )

    def _validate_topology_receipt(
        self, topology: ResidentDirectVectorTopologyV67
    ) -> None:
        if (
            not isinstance(topology, ResidentDirectVectorTopologyV67)
            or topology.execution_fingerprint != self.execution_fingerprint
            or topology.device_id != self.device_id
            or topology.semantic_validation_count != 1
        ):
            raise DirectVectorFaceExnerV67Error(
                "topology receipt differs from execution/device"
            )
        expected = self._topology_token(
            topology.owner,
            topology.neighbour,
            topology.role,
            topology.normal,
            topology.face_length,
            topology.cell_area,
            topology.offset,
            topology.adjacency_faces,
            topology.signs,
            ncell=topology.ncell,
            nface=topology.nface,
        )
        if expected != topology.topology_sha256:
            raise DirectVectorFaceExnerV67Error(
                "topology receipt arrays are stale or mutated"
            )

    def _state_token(
        self,
        mass: object,
        density: object,
        fraction: object,
        signed: object,
        absolute: object,
        step: int,
        time_s: float,
    ) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "accepted_step": step,
            "time_s": time_s,
            "resident_arrays_sha256": self._resident_digest(
                mass,
                density,
                fraction,
                signed,
                absolute,
                label="accepted_face_exner_state",
            ),
        })

    def resolve_state(
        self,
        *,
        bed_mass_kg_m2: object,
        active_fraction: object | None = None,
        boundary_signed_kg: object | None = None,
        boundary_absolute_kg: object | None = None,
        accepted_step: int = 0,
        time_s: float = 0.0,
    ) -> ResidentDirectVectorStateV67:
        cp = self.cp
        mass = self._array("bed_mass_kg_m2", bed_mass_kg_m2, 2, copy=True)
        if mass.shape[0] != self.nclass or mass.shape[1] < 1:
            raise DirectVectorFaceExnerV67Error(
                "bed mass class/cell layout differs from the runtime"
            )
        ncell = int(mass.shape[1])
        if not self._all(cp.isfinite(mass) & (mass >= 0.0)):
            raise DirectVectorFaceExnerV67Error(
                "accepted bed mass must be finite and nonnegative"
            )
        density = self.bulk_density.copy(order="C")
        volume = mass / density[:, None]
        total = cp.sum(volume, axis=0, keepdims=True)
        if not self._all(cp.isfinite(total) & (total >= 0.0)):
            raise DirectVectorFaceExnerV67Error(
                "accepted solid volume is invalid"
            )
        positive_stock = total[0] > 0.0
        derived = cp.ascontiguousarray(cp.where(
            positive_stock[None, :],
            volume / cp.where(total > 0.0, total, 1.0),
            0.0,
        ))
        if active_fraction is None:
            if self._flag(cp.any(~positive_stock)):
                raise DirectVectorFaceExnerV67Error(
                    "depleted cells require an explicit retained surface identity"
                )
            fraction = derived
        else:
            fraction = self._array(
                "active_fraction",
                active_fraction,
                2,
                (self.nclass, ncell),
                copy=True,
            )
            tolerance = 512.0 * np.finfo(np.float64).eps
            if not self._all(cp.isfinite(fraction) & (fraction >= 0.0)) or not self._flag(
                cp.allclose(cp.sum(fraction, axis=0), 1.0,
                            rtol=0.0, atol=tolerance*self.nclass)
            ) or not self._flag(
                cp.allclose(fraction[:, positive_stock],
                            derived[:, positive_stock],
                            rtol=tolerance, atol=tolerance)
            ):
                raise DirectVectorFaceExnerV67Error(
                    "accepted active fraction differs from positive finite "
                    "stock or is not a normalized retained identity"
                )
        signed = (
            cp.zeros(self.nclass, dtype=cp.float64)
            if boundary_signed_kg is None
            else self._array(
                "boundary_signed_kg",
                boundary_signed_kg,
                1,
                (self.nclass,),
                copy=True,
            )
        )
        absolute = (
            cp.zeros(self.nclass, dtype=cp.float64)
            if boundary_absolute_kg is None
            else self._array(
                "boundary_absolute_kg",
                boundary_absolute_kg,
                1,
                (self.nclass,),
                copy=True,
            )
        )
        if not self._all(cp.isfinite(signed)) or not self._all(
            cp.isfinite(absolute) & (absolute >= 0.0)
        ):
            raise DirectVectorFaceExnerV67Error(
                "accepted boundary ledger is invalid"
            )
        if (
            isinstance(accepted_step, (bool, np.bool_))
            or type(accepted_step) is not int
            or accepted_step < 0
            or accepted_step > self.accepted_step_limit
        ):
            raise DirectVectorFaceExnerV67Error("accepted step is invalid")
        accepted_time = _finite("accepted time", time_s, nonnegative=True)
        token = self._state_token(
            mass,
            density,
            fraction,
            signed,
            absolute,
            accepted_step,
            accepted_time,
        )
        return ResidentDirectVectorStateV67(
            self.execution_fingerprint,
            mass,
            density,
            fraction,
            signed,
            absolute,
            accepted_step,
            accepted_time,
            token,
            self.device_id,
        )

    def _validate_state(self, state: ResidentDirectVectorStateV67) -> None:
        if (
            not isinstance(state, ResidentDirectVectorStateV67)
            or state.execution_fingerprint != self.execution_fingerprint
            or state.device_id != self.device_id
        ):
            raise DirectVectorFaceExnerV67Error(
                "accepted state differs from execution/device"
            )
        expected = self._state_token(
            state.bed_mass_kg_m2,
            state.bulk_density_kg_m3,
            state.active_fraction,
            state.boundary_signed_kg,
            state.boundary_absolute_kg,
            state.accepted_step,
            state.time_s,
        )
        if expected != state.state_sha256:
            raise DirectVectorFaceExnerV67Error(
                "accepted state arrays are stale or mutated"
            )

    def _face_token(
        self,
        topology: ResidentDirectVectorTopologyV67,
        flux: object,
        donor: object,
        status: object,
        *,
        label: str,
    ) -> str:
        body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "topology_sha256": topology.topology_sha256,
            "label": label,
            "resident_arrays_sha256": self._resident_digest(
                flux, donor, status, label=label
            ),
        }
        if self.slope_extension_fingerprint is not None:
            body["slope_extension_fingerprint"] = (
                self.slope_extension_fingerprint
            )
        return _fingerprint(body)

    def _slope_token(
        self,
        *,
        topology: ResidentDirectVectorTopologyV67,
        downhill: object,
        stress: object,
        depth: object,
        d50: object,
        mean_diameter: object,
        hiding: object,
    ) -> str:
        assert self.slope_composer is not None
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "extension_fingerprint": self.slope_extension_fingerprint,
            "slope_execution_fingerprint":
                self.slope_composer.identity["fingerprint"],
            "topology_sha256": topology.topology_sha256,
            "stress_semantics": TAURAT_POSITIVE_ZERO,
            "resident_arrays_sha256": self._resident_digest(
                downhill, stress, depth, d50, mean_diameter, hiding,
                label="direct_vector_positive_taurat_slope_fields_v69",
            ),
        })

    def resolve_positive_taurat_slope_fields_v69(
        self,
        *,
        topology: ResidentDirectVectorTopologyV67,
        downhill: object,
        depth: object,
        d50: object,
        mean_diameter: object,
        hiding: object,
    ) -> ResidentDirectVectorSlopeFieldsV69:
        """Seal exact-zero stress and all topology-specific ISlope=2 fields."""
        cp = self.cp
        if self.slope_composer is None:
            raise DirectVectorFaceExnerV67Error(
                "historical slope-off consumer cannot resolve slope fields"
            )
        self._validate_topology_receipt(topology)
        down = self._array(
            "slope downhill", downhill, 2, (2, topology.nface), copy=True
        )
        water_depth = self._array(
            "slope depth", depth, 2,
            (self.nclass, topology.nface), copy=True,
        )
        diameter = self._array(
            "slope d50", d50, 1, (self.nclass,), copy=True
        )
        mean = self._array(
            "slope mean diameter", mean_diameter, 2,
            (self.nclass, topology.nface), copy=True,
        )
        hide = self._array(
            "slope hiding", hiding, 2,
            (self.nclass, topology.ncell), copy=True,
        )
        if (not self._all(cp.isfinite(down)) or
                not self._all(cp.isfinite(water_depth) &
                              (water_depth > 0.0)) or
                not self._all(cp.isfinite(diameter) & (diameter > 0.0)) or
                not self._all(cp.isfinite(mean) & (mean > 0.0)) or
                not self._all(cp.isfinite(hide) & (hide >= 0.0))):
            raise DirectVectorFaceExnerV67Error(
                "positive taurat slope fields are outside their domain"
            )
        stress = cp.zeros(
            (self.nclass, topology.nface), dtype=cp.float64
        )
        token = self._slope_token(
            topology=topology, downhill=down, stress=stress,
            depth=water_depth, d50=diameter, mean_diameter=mean,
            hiding=hide,
        )
        return ResidentDirectVectorSlopeFieldsV69(
            self.slope_extension_fingerprint,
            str(self.slope_composer.identity["fingerprint"]),
            topology.topology_sha256,
            TAURAT_POSITIVE_ZERO,
            down, stress, water_depth, diameter, mean, hide, token,
            self.device_id,
        )

    def _validate_slope_fields_v69(
        self,
        slope: ResidentDirectVectorSlopeFieldsV69 | None,
        topology: ResidentDirectVectorTopologyV67,
    ) -> None:
        cp = self.cp
        if self.slope_composer is None:
            if slope is not None:
                raise DirectVectorFaceExnerV67Error(
                    "historical slope-off consumer rejects slope fields"
                )
            return
        if (not isinstance(slope, ResidentDirectVectorSlopeFieldsV69) or
                slope.extension_fingerprint !=
                    self.slope_extension_fingerprint or
                slope.slope_execution_fingerprint !=
                    self.slope_composer.identity["fingerprint"] or
                slope.topology_sha256 != topology.topology_sha256 or
                slope.stress_semantics != TAURAT_POSITIVE_ZERO or
                slope.device_id != self.device_id):
            raise DirectVectorFaceExnerV67Error(
                "positive taurat slope receipt differs"
            )
        arrays = (
            ("downhill", slope.downhill, 2, (2, topology.nface)),
            ("stress", slope.stress, 2, (self.nclass, topology.nface)),
            ("depth", slope.depth, 2, (self.nclass, topology.nface)),
            ("d50", slope.d50, 1, (self.nclass,)),
            ("mean_diameter", slope.mean_diameter, 2,
             (self.nclass, topology.nface)),
            ("hiding", slope.hiding, 2, (self.nclass, topology.ncell)),
        )
        for name, value, ndim, shape in arrays:
            self._array(f"slope {name}", value, ndim, shape)
        if (not self._all(cp.isfinite(slope.downhill)) or
                not self._all(slope.stress == 0.0) or
                not self._all(cp.isfinite(slope.depth) &
                              (slope.depth > 0.0)) or
                not self._all(cp.isfinite(slope.d50) & (slope.d50 > 0.0)) or
                not self._all(cp.isfinite(slope.mean_diameter) &
                              (slope.mean_diameter > 0.0)) or
                not self._all(cp.isfinite(slope.hiding) &
                              (slope.hiding >= 0.0))):
            raise DirectVectorFaceExnerV67Error(
                "positive TraFrm ISlope=2 requires exact zero taurat"
            )
        expected = self._slope_token(
            topology=topology, downhill=slope.downhill,
            stress=slope.stress, depth=slope.depth, d50=slope.d50,
            mean_diameter=slope.mean_diameter, hiding=slope.hiding,
        )
        if expected != slope.slope_sha256:
            raise DirectVectorFaceExnerV67Error(
                "positive taurat slope arrays are stale or mutated"
            )

    def _faces(
        self,
        *,
        cells: ResidentDirectVectorChannelsV67,
        traces: ResidentDirectVectorChannelsV67,
        topology: ResidentDirectVectorTopologyV67,
        availability: object,
        label: str,
        left_bed: object | None = None,
        right_bed: object | None = None,
        slope: ResidentDirectVectorSlopeFieldsV69 | None = None,
    ) -> ResidentDirectVectorFacesV67:
        cp = self.cp
        ncell, nface = topology.ncell, topology.nface
        availability = self._array(
            "availability",
            availability,
            2,
            (self.nclass, ncell),
        )
        if not self._all(cp.isfinite(availability) & (availability >= 0.0)):
            raise DirectVectorFaceExnerV67Error(
                "accepted face availability is invalid"
            )
        safe_owner = cp.clip(topology.owner, 0, ncell - 1)
        safe_neighbour = cp.clip(topology.neighbour, 0, ncell - 1)
        if left_bed is None:
            left = cp.ascontiguousarray(
                cells.bed_mass_vector[:, :, safe_owner]
            )
        else:
            left = self._array(
                "left_bed", left_bed, 3, (2, self.nclass, nface)
            )
        if right_bed is None:
            right = cp.ascontiguousarray(
                cells.bed_mass_vector[:, :, safe_neighbour]
            )
        else:
            right = self._array(
                "right_bed", right_bed, 3, (2, self.nclass, nface)
            )
        if not self._all(cp.isfinite(left)) or not self._all(cp.isfinite(right)):
            raise DirectVectorFaceExnerV67Error(
                "reconstructed bed vector contains a nonfinite value"
            )
        self._validate_slope_fields_v69(slope, topology)
        flux = cp.zeros((self.nclass, nface), dtype=cp.float64)
        donor = cp.zeros((self.nclass, nface), dtype=cp.int32)
        status = cp.zeros(nface, dtype=cp.int32)
        with cp.cuda.Device(self.device_id):
            if nface and slope is None:
                self.kernels["q3_scalar_face_dispatch"](
                    ((nface + 255) // 256,),
                    (256,),
                    (
                        np.int32(ncell),
                        np.int32(nface),
                        np.int32(self.nclass),
                        topology.owner,
                        topology.neighbour,
                        topology.role,
                        topology.normal,
                        left,
                        right,
                        cells.bed_mass_vector,
                        traces.bed_mass_vector,
                        availability,
                        flux,
                        donor,
                        status,
                    ),
                )
            elif nface:
                assert self.slope_composer is not None
                h4_role = cp.ascontiguousarray(cp.where(
                    topology.role == FACE_CLOSED,
                    FACE_INTERNAL,
                    topology.role,
                ).astype(cp.int32, copy=False))
                owner_bed = cp.ascontiguousarray(
                    cells.bed_mass_vector[:, :, safe_owner]
                )
                result = self.slope_composer.compose(
                    left_bed=left, right_bed=right,
                    owner_bed=owner_bed,
                    trace_bed=traces.bed_mass_vector,
                    owner=topology.owner, neighbour=topology.neighbour,
                    role=h4_role, normal=topology.normal,
                    availability=availability,
                    downhill=slope.downhill, stress=slope.stress,
                    depth=slope.depth, d50=slope.d50,
                    mean_diameter=slope.mean_diameter,
                    hiding=slope.hiding, reject_invalid=True,
                )
                flux = cp.ascontiguousarray(result.flux)
                donor = cp.ascontiguousarray(result.donor)
                status = cp.ascontiguousarray(result.status)
            left_status = cells.status[safe_owner]
            right_status = cp.where(
                topology.neighbour >= 0,
                cells.status[safe_neighbour],
                0,
            )
            trace_status = cp.where(
                (topology.role == FACE_TRACE_INLET)
                | (topology.role == FACE_TRACE_OUTLET),
                traces.status,
                0,
            )
            status = cp.maximum(
                status,
                cp.maximum(left_status, cp.maximum(right_status, trace_status)),
            )
            flux = cp.ascontiguousarray(
                cp.where(status[None, :] == 0, flux, 0.0)
            )
            donor = cp.ascontiguousarray(
                cp.where(status[None, :] == 0, donor, 0).astype(
                    cp.int32, copy=False
                )
            )
            status = cp.ascontiguousarray(status.astype(cp.int32, copy=False))
        token = self._face_token(
            topology, flux, donor, status, label=label
        )
        return ResidentDirectVectorFacesV67(
            self.execution_fingerprint,
            topology.topology_sha256,
            flux,
            donor,
            status,
            token,
            self.device_id,
        )

    def _validate_faces(
        self,
        faces: ResidentDirectVectorFacesV67,
        topology: ResidentDirectVectorTopologyV67,
        *,
        label: str,
    ) -> None:
        if (
            not isinstance(faces, ResidentDirectVectorFacesV67)
            or faces.execution_fingerprint != self.execution_fingerprint
            or faces.topology_sha256 != topology.topology_sha256
            or faces.device_id != self.device_id
            or faces.flux.shape != (self.nclass, topology.nface)
        ):
            raise DirectVectorFaceExnerV67Error("face receipt differs")
        expected = self._face_token(
            topology, faces.flux, faces.donor, faces.status, label=label
        )
        if expected != faces.face_sha256:
            raise DirectVectorFaceExnerV67Error(
                "face receipt arrays are stale or mutated"
            )

    def begin_trial(
        self,
        *,
        state: ResidentDirectVectorStateV67,
        cells: ResidentDirectVectorChannelsV67,
        traces: ResidentDirectVectorChannelsV67,
        topology: ResidentDirectVectorTopologyV67,
        dt_s: float,
        morphological_factor: float = 1.0,
        left_bed: object | None = None,
        right_bed: object | None = None,
        slope: ResidentDirectVectorSlopeFieldsV69 | None = None,
        cancelled=lambda: False,
    ) -> ResidentDirectVectorTrialV67:
        """Build a conservative candidate without mutating accepted state."""
        cp = self.cp
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError(
                "direct-vector v67 transaction cancelled before launch"
            )
        self._validate_state(state)
        self._validate_topology_receipt(topology)
        self._validate_slope_fields_v69(slope, topology)
        if state.accepted_step >= self.accepted_step_limit:
            raise DirectVectorFaceExnerV67Error(
                "bounded accepted-step limit reached"
            )
        if state.bed_mass_kg_m2.shape[1] != topology.ncell:
            raise DirectVectorFaceExnerV67Error(
                "state cell count differs from prepared topology"
            )
        self._validate_channels(cells, point_role="cells", npoint=topology.ncell)
        self._validate_channels(
            traces, point_role="traces", npoint=topology.nface
        )
        dt = _finite("trial dt", dt_s, positive=True)
        morfac = _finite(
            "morphological factor", morphological_factor, nonnegative=True
        )

        # The preliminary evaluation estimates demand.  The accepted face
        # evaluation below is the sole application represented by the public
        # once-only counter; neither producer channel contains a fraction.
        provisional = self._faces(
            cells=cells,
            traces=traces,
            topology=topology,
            availability=state.active_fraction,
            left_bed=left_bed,
            right_bed=right_bed,
            slope=slope,
            label="provisional_inventory_demand",
        )
        if self._flag(cp.any(provisional.status != 0)):
            raise DirectVectorFaceExnerV67Error(
                "provisional face reconstruction rejected"
            )
        fill = cp.empty_like(state.active_fraction)
        availability = cp.empty_like(state.active_fraction)
        inventory_status = cp.zeros(topology.ncell, dtype=cp.int32)
        count = self.nclass * topology.ncell
        if count:
            self.kernels["direct_v66_inventory_fill"](
                ((count + 255) // 256,),
                (256,),
                (
                    np.int32(topology.ncell),
                    np.int32(topology.nface),
                    np.int32(topology.adjacency_faces.size),
                    np.int32(self.nclass),
                    topology.offset,
                    topology.adjacency_faces,
                    topology.owner,
                    topology.neighbour,
                    topology.role,
                    provisional.donor,
                    provisional.flux,
                    topology.face_length,
                    topology.cell_area,
                    state.bed_mass_kg_m2,
                    state.active_fraction,
                    np.float64(dt),
                    np.float64(morfac),
                    fill,
                    availability,
                    inventory_status,
                ),
            )
        if self._flag(cp.any(inventory_status != 0)):
            raise DirectVectorFaceExnerV67Error(
                "finite-inventory projection rejected"
            )
        fill = cp.ascontiguousarray(fill)
        availability = cp.ascontiguousarray(availability)
        faces = self._faces(
            cells=cells,
            traces=traces,
            topology=topology,
            availability=availability,
            left_bed=left_bed,
            right_bed=right_bed,
            slope=slope,
            label="accepted_inventory_limited_faces",
        )
        if self._flag(cp.any(faces.status != 0)):
            raise DirectVectorFaceExnerV67Error(
                "accepted face reconstruction rejected"
            )

        candidate_mass = cp.empty_like(state.bed_mass_kg_m2)
        delta = cp.empty_like(state.bed_mass_kg_m2)
        exner_status = cp.zeros(topology.ncell, dtype=cp.int32)
        if topology.ncell:
            self.kernels["q3_scalar_exner_candidate"](
                ((topology.ncell + 255) // 256,),
                (256,),
                (
                    np.int32(topology.ncell),
                    np.int32(topology.nface),
                    np.int32(topology.adjacency_faces.size),
                    np.int32(self.nclass),
                    topology.offset,
                    topology.adjacency_faces,
                    topology.signs,
                    topology.face_length,
                    topology.cell_area,
                    faces.flux,
                    faces.status,
                    state.bed_mass_kg_m2,
                    np.float64(dt),
                    np.float64(morfac),
                    candidate_mass,
                    delta,
                    exner_status,
                ),
            )
        if self._flag(cp.any(exner_status != 0)):
            raise DirectVectorFaceExnerV67Error(
                "conservative Exner candidate rejected"
            )
        candidate_mass = cp.ascontiguousarray(candidate_mass)
        clamped_candidate_mass = candidate_mass.copy()
        depletion_tolerance = (
            2048.0 * np.finfo(np.float64).eps
            * cp.maximum(1.0, cp.abs(state.bed_mass_kg_m2))
        )
        numerically_depleted = (
            cp.sum(clamped_candidate_mass, axis=0)
            <= cp.sum(depletion_tolerance, axis=0)
        )
        candidate_mass[:, numerically_depleted] = 0.0
        depletion_roundoff_transfer = cp.ascontiguousarray(cp.sum(
            (clamped_candidate_mass - candidate_mass)
            * topology.cell_area[None, :],
            axis=1,
        ))
        delta = cp.ascontiguousarray(delta)
        delta[:, numerically_depleted] = (
            candidate_mass[:, numerically_depleted]
            - state.bed_mass_kg_m2[:, numerically_depleted]
        )
        integrated_delta = cp.sum(
            delta * topology.cell_area[None, :], axis=1
        )
        multiplicity = cp.bincount(
            topology.adjacency_faces,
            weights=topology.signs.astype(cp.float64),
            minlength=topology.nface,
        )
        boundary_transfer = cp.ascontiguousarray(
            np.float64(dt)
            * np.float64(morfac)
            * cp.sum(
                faces.flux
                * topology.face_length[None, :]
                * multiplicity[None, :],
                axis=1,
            )
        )
        residual = cp.ascontiguousarray(
            integrated_delta + boundary_transfer
            + depletion_roundoff_transfer
        )
        tolerance = 4096.0 * np.finfo(np.float64).eps * cp.maximum(
            1.0,
            cp.sum(cp.abs(delta * topology.cell_area[None, :]), axis=1)
            + cp.abs(boundary_transfer) + depletion_roundoff_transfer,
        )
        if self._flag(cp.any(cp.abs(residual) > tolerance)):
            raise DirectVectorFaceExnerV67Error(
                "conservative Exner ledger does not close"
            )
        boundary_mask = (
            (topology.role == FACE_TRACE_INLET)
            | (topology.role == FACE_TRACE_OUTLET)
            | (topology.role == FACE_OWNER_OUTLET)
        )
        absolute_transfer = cp.ascontiguousarray(
            np.float64(dt)
            * np.float64(morfac)
            * cp.sum(
                cp.abs(faces.flux)
                * topology.face_length[None, :]
                * boundary_mask[None, :],
                axis=1,
            )
        )
        candidate_volume = candidate_mass / state.bulk_density_kg_m3[:, None]
        candidate_total = cp.sum(candidate_volume, axis=0, keepdims=True)
        candidate_positive = candidate_total[0] > 0.0
        candidate_fraction = cp.ascontiguousarray(cp.where(
            candidate_positive[None, :],
            candidate_volume / cp.where(candidate_total > 0.0,
                                        candidate_total, 1.0),
            state.active_fraction,
        ))
        candidate = self.resolve_state(
            bed_mass_kg_m2=candidate_mass,
            active_fraction=candidate_fraction,
            boundary_signed_kg=cp.ascontiguousarray(
                state.boundary_signed_kg + boundary_transfer
                + depletion_roundoff_transfer
            ),
            boundary_absolute_kg=cp.ascontiguousarray(
                state.boundary_absolute_kg + absolute_transfer
                + depletion_roundoff_transfer
            ),
            accepted_step=state.accepted_step + 1,
            time_s=state.time_s + dt,
        )
        arrays = (
            provisional.flux,
            provisional.donor,
            provisional.status,
            faces.flux,
            faces.donor,
            faces.status,
            fill,
            availability,
            delta,
            boundary_transfer,
            absolute_transfer,
            depletion_roundoff_transfer,
            residual,
        )
        token_body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "topology_sha256": topology.topology_sha256,
            "cell_channel_sha256": cells.channel_sha256,
            "trace_channel_sha256": traces.channel_sha256,
            "provisional_face_sha256": provisional.face_sha256,
            "accepted_face_sha256": faces.face_sha256,
            "dt_s": dt,
            "morphological_factor": morfac,
            "topology_semantic_validation_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "resident_arrays_sha256": self._resident_digest(
                *arrays, label="direct_vector_face_exner_trial_v67"
            ),
        }
        if slope is not None:
            token_body["slope_extension_fingerprint"] = (
                self.slope_extension_fingerprint
            )
            token_body["slope_receipt_sha256"] = slope.slope_sha256
        token = _fingerprint(token_body)
        trial = ResidentDirectVectorTrialV67(
            self.execution_fingerprint,
            state.state_sha256,
            topology.topology_sha256,
            cells.channel_sha256,
            traces.channel_sha256,
            dt,
            morfac,
            provisional,
            faces,
            fill,
            availability,
            delta,
            boundary_transfer,
            absolute_transfer,
            depletion_roundoff_transfer,
            residual,
            candidate,
            1,
            1,
            1,
            token,
            self.device_id,
            slope_receipt_sha256=(
                None if slope is None else slope.slope_sha256
            ),
        )
        if cancelled():
            raise InterruptedError(
                "direct-vector v67 transaction cancelled before publication"
            )
        return trial

    def _validate_trial(
        self,
        state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorTrialV67,
    ) -> None:
        self._validate_state(state)
        if (
            not isinstance(trial, ResidentDirectVectorTrialV67)
            or trial.execution_fingerprint != self.execution_fingerprint
            or trial.parent_state_sha256 != state.state_sha256
            or trial.device_id != self.device_id
            or trial.topology_semantic_validation_count != 1
            or trial.active_fraction_application_count != 1
            or trial.inventory_fill_application_count != 1
            or ((self.slope_composer is None) !=
                (trial.slope_receipt_sha256 is None))
        ):
            raise DirectVectorFaceExnerV67Error(
                "trial parent or once-only receipt differs"
            )
        for face_receipt in (trial.provisional_faces, trial.faces):
            if (
                not isinstance(face_receipt, ResidentDirectVectorFacesV67)
                or face_receipt.execution_fingerprint
                != self.execution_fingerprint
                or face_receipt.topology_sha256 != trial.topology_sha256
                or face_receipt.device_id != self.device_id
                or face_receipt.flux.shape[0] != self.nclass
            ):
                raise DirectVectorFaceExnerV67Error(
                    "trial face receipt differs"
                )
        arrays = (
            trial.provisional_faces.flux,
            trial.provisional_faces.donor,
            trial.provisional_faces.status,
            trial.faces.flux,
            trial.faces.donor,
            trial.faces.status,
            trial.inventory_fill,
            trial.accepted_face_availability,
            trial.cell_delta_kg_m2,
            trial.boundary_transfer_kg,
            trial.boundary_absolute_transfer_kg,
            trial.depletion_roundoff_transfer_kg,
            trial.conservation_residual_kg,
        )
        expected_body = {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution_fingerprint,
            "parent_state_sha256": trial.parent_state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "topology_sha256": trial.topology_sha256,
            "cell_channel_sha256": trial.cell_channel_sha256,
            "trace_channel_sha256": trial.trace_channel_sha256,
            "provisional_face_sha256": trial.provisional_faces.face_sha256,
            "accepted_face_sha256": trial.faces.face_sha256,
            "dt_s": trial.dt_s,
            "morphological_factor": trial.morphological_factor,
            "topology_semantic_validation_count": 1,
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
            "resident_arrays_sha256": self._resident_digest(
                *arrays, label="direct_vector_face_exner_trial_v67"
            ),
        }
        if trial.slope_receipt_sha256 is not None:
            _exact_sha256(
                "slope_receipt_sha256", trial.slope_receipt_sha256
            )
            expected_body["slope_extension_fingerprint"] = (
                self.slope_extension_fingerprint
            )
            expected_body["slope_receipt_sha256"] = (
                trial.slope_receipt_sha256
            )
        expected = _fingerprint(expected_body)
        if expected != trial.trial_sha256:
            raise DirectVectorFaceExnerV67Error(
                "trial arrays are stale or mutated"
            )
        self._validate_state(trial.candidate)

    def commit_trial(
        self,
        state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return trial.candidate

    def rollback_trial(
        self,
        state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return state

    def cancel_trial(
        self,
        state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return state

    def state_checkpoint_document(
        self, state: ResidentDirectVectorStateV67
    ) -> dict[str, object]:
        """Create a canonical portable checkpoint for one accepted state."""
        self._validate_state(state)
        cp = self.cp
        body: dict[str, object] = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID,
            "backend": BACKEND,
            "integrator": INTEGRATOR,
            "execution_fingerprint": self.execution_fingerprint,
            "nclass": self.nclass,
            "accepted_step_limit": self.accepted_step_limit,
            "accepted_step": state.accepted_step,
            "time_s": state.time_s,
            "bed_mass_kg_m2": cp.asnumpy(state.bed_mass_kg_m2).tolist(),
            "bulk_density_kg_m3": cp.asnumpy(
                state.bulk_density_kg_m3
            ).tolist(),
            "active_fraction": cp.asnumpy(state.active_fraction).tolist(),
            "boundary_signed_kg": cp.asnumpy(
                state.boundary_signed_kg
            ).tolist(),
            "boundary_absolute_kg": cp.asnumpy(
                state.boundary_absolute_kg
            ).tolist(),
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def load_state_checkpoint(
        self, value: Mapping[str, object]
    ) -> ResidentDirectVectorStateV67:
        """Validate and restore a portable checkpoint to this CUDA device."""
        if not isinstance(value, Mapping):
            raise DirectVectorFaceExnerV67Error(
                "state checkpoint must be an object"
            )
        document = dict(value)
        fingerprint = document.pop("fingerprint", None)
        if (
            fingerprint != _fingerprint(document)
            or document.get("schema") != SCHEMA
            or document.get("scope_id") != SCOPE_ID
            or document.get("backend") != BACKEND
            or document.get("integrator") != INTEGRATOR
            or document.get("execution_fingerprint")
            != self.execution_fingerprint
            or document.get("nclass") != self.nclass
            or document.get("accepted_step_limit") != self.accepted_step_limit
        ):
            raise DirectVectorFaceExnerV67Error(
                "state checkpoint header or fingerprint differs"
            )
        cp = self.cp
        density = cp.ascontiguousarray(
            cp.asarray(document.get("bulk_density_kg_m3"), dtype=cp.float64)
        )
        if density.shape != self.bulk_density.shape or not self._flag(
            cp.array_equal(density, self.bulk_density)
        ):
            raise DirectVectorFaceExnerV67Error(
                "state checkpoint bulk density differs"
            )
        state = self.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(
                cp.asarray(document.get("bed_mass_kg_m2"), dtype=cp.float64)
            ),
            active_fraction=cp.ascontiguousarray(
                cp.asarray(document.get("active_fraction"), dtype=cp.float64)
            ),
            boundary_signed_kg=cp.ascontiguousarray(
                cp.asarray(document.get("boundary_signed_kg"), dtype=cp.float64)
            ),
            boundary_absolute_kg=cp.ascontiguousarray(
                cp.asarray(
                    document.get("boundary_absolute_kg"), dtype=cp.float64
                )
            ),
            accepted_step=document.get("accepted_step"),  # type: ignore[arg-type]
            time_s=document.get("time_s"),  # type: ignore[arg-type]
        )
        if state.state_sha256 != document.get("state_sha256"):
            raise DirectVectorFaceExnerV67Error(
                "state checkpoint content token differs"
            )
        return state


__all__ = [
    "BACKEND",
    "CUDA_OPTIONS",
    "DirectVectorFaceExnerV67Error",
    "FACE_CLOSED",
    "FACE_INTERNAL",
    "FACE_OWNER_OUTLET",
    "FACE_ROLES",
    "FACE_TRACE_INLET",
    "FACE_TRACE_OUTLET",
    "INTEGRATOR",
    "KERNEL_NAMES",
    "ResidentDirectVectorChannelsV67",
    "ResidentDirectVectorFaceExnerV67",
    "ResidentDirectVectorFacesV67",
    "ResidentDirectVectorStateV67",
    "ResidentDirectVectorTopologyV67",
    "ResidentDirectVectorTrialV67",
    "SCHEMA",
    "SCOPE_ID",
]
