"""Authenticated resident CUDA lifecycle for the six Q3 direct-vector laws.

This is the public *execution* seam between the v67 family receipt and the
law-neutral face/Exner consumer.  The caller supplies hydraulic state and
physical settling/observation fields, never constitutive output channels.
Every fraction is evaluated by the method recorded in the execution receipt,
then the FLOW route is composed with an exact unit surface fraction.  The
real active-bed fraction is owned by :class:`ResidentDirectVectorFaceExnerV67`
and is therefore applied once, and only once, at accepted faces.

The catalogue and producer namespaces are deliberately distinct.  In
particular ``capacity.bijker`` is the catalogue entry while
``capacity.bijker_current`` is the current-only producer profile.  The
explicit, authenticated mapping below prevents a convenient alias from
becoming an unrecorded physical choice.
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

from braids_core.sediment.capacity_parameters import formula_contract
from braids_core.sediment.channel_composition import (
    TRA_BEDLOAD,
)
from braids_core.sediment.direct_current_parameters_v66 import (
    direct_current_contract_v66,
)
from braids_core.sediment.direct_vector_catalogue_bridge_v67 import (
    CATALOGUE_TO_PROFILE as CATALOGUE_TO_PRODUCER,
    PROFILE_TO_CATALOGUE as PRODUCER_TO_CATALOGUE,
    direct_vector_catalogue_bridge_document_v67,
    direct_vector_profile_method_ids_v67,
    load_direct_vector_catalogue_bridge_v67,
)
from braids_core.sediment.direct_vector_family_live_v67 import (
    METHOD_BIJKER_CURRENT,
    METHOD_SOULSBY_CURRENT,
    METHOD_SOULSBY_VAN_RIJN_CURRENT,
    METHOD_VAN_RIJN_1993_CURRENT,
    METHOD_VAN_THIEL_VAN_RIJN_2008_CURRENT,
    METHOD_XBEACH_SOULSBY_VAN_RIJN_CURRENT,
    DirectVectorFamilyLiveV67Error,
    direct_vector_family_state_checkpoint_v67,
    load_direct_vector_family_execution_v67,
    load_direct_vector_family_state_checkpoint_v67,
    resolve_direct_vector_family_state_v67,
    verify_direct_vector_family_sources_v67,
)
from braids_core.sediment.direct_vector_route_composer_v66 import (
    METHOD_SOULSBY,
    METHOD_SOULSBY_VAN_RIJN,
    ROUTE_TRA_BEDLOAD,
)
from braids_core.sediment.direct_vector_maximum_waterdepth_bridge_v68 import (
    load_direct_vector_maximum_waterdepth_execution_v68,
    verify_direct_vector_maximum_waterdepth_sources_v68,
)

from .delft_direct_current_v66 import FORMULA_FIELDS, INTEGER_FIELDS
from .direct_producer_adapter_v67 import DirectProducerAdapterV67CUDA
from .direct_vector_face_exner_v67 import (
    BACKEND,
    INTEGRATOR,
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
    FACE_TRACE_OUTLET,
    ResidentDirectVectorChannelsV67,
    ResidentDirectVectorFaceExnerV67,
    ResidentDirectVectorStateV67,
    ResidentDirectVectorTopologyV67,
    ResidentDirectVectorTrialV67,
)
from .direct_vector_route_composer_v66 import DirectVectorRouteComposerV66CUDA
from .direct_vector_maximum_waterdepth_bridge_v68 import (
    DirectVectorMaximumWaterdepthBridgeV68CUDA,
    ResidentDirectVectorHydraulicsV68,
)
from .hydraulic_roughness import HydraulicRoughnessCUDA
from .scalar_capacity_runtime import ResidentScalarCapacityRuntime


SCHEMA = 1
SCOPE_ID = "braids-q3-h6-gpu-direct-vector-family-live-v67-v1"
CHECKPOINT_SCOPE_ID = (
    "braids-q3-h6c-gpu-direct-vector-family-checkpoint-v68-v1"
)

_SCALAR_METHOD = MappingProxyType({
    METHOD_SOULSBY_CURRENT: (4, METHOD_SOULSBY),
    METHOD_SOULSBY_VAN_RIJN_CURRENT: (13, METHOD_SOULSBY_VAN_RIJN),
})

# The outer family receipt authenticates all constitutive/composer sources.
# These additional exact bytes are used only by this high-level seam and by
# its law-neutral consumer.  Source override arguments are intentionally not
# exposed by the class below.
_LOCAL_SOURCE_SHA256 = MappingProxyType({
    "braids_core/sediment/direct_vector_catalogue_bridge_v67.py":
        "f07e2073ea68797b565466543bd049165bf4c8853ef133b1fca0e2e326a907de",
    "braids_gpu/direct_vector_face_exner_v67.py":
        "1ba0a45b21112f36d8d1816872d180cddd303e5e55c511d30fb0b957863fd02a",
    "braids_gpu/direct_vector_live_v66.cu":
        "da18dc63a59d2a5daf1748de724a9cfe9494dd625aa33a5b5b38ad11922eec77",
    "braids_gpu/hydraulic_roughness.py":
        "f117822e49fbcd22c5a73728aa19e11122d82fb8298b888e721a9f91507daf6c",
    "braids_gpu/hydraulic_roughness.cu":
        "463c9892852ac07dc6df3c13beebd4c7b76006630642b5e9e65e51e84721e17b",
})


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorFamilyLiveV67Error(
            "GPU direct-vector family receipt is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(
    value: object, name: str, *, positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise DirectVectorFamilyLiveV67Error(f"{name} must be finite numeric data")
    result = float(value)
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise DirectVectorFamilyLiveV67Error(f"{name} is outside its domain")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ResidentDirectVectorProducerGroupV67:
    """Authenticated resident output of one method/route fraction group."""

    execution_fingerprint: str
    maximum_waterdepth_execution_fingerprint: str
    catalogue_entry_id: str
    producer_profile_method_id: str
    forcing_variant: str
    route: str
    fraction_indices: tuple[int, ...]
    producer_identity_fingerprints: tuple[str, ...]
    parameter_identity_fingerprints: tuple[str, ...]
    raw_values: object
    composed_values: object
    producer_status: object
    route_status: object
    surface_fraction_application_count: int
    post_producer_density_conversion_count: int
    hydraulic_preprocessor_application_count: int
    group_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorFamilyChannelsV67:
    """Six-method family result plus the sealed face-consumer channel."""

    execution_fingerprint: str
    maximum_waterdepth_execution_fingerprint: str
    catalogue_mapping_fingerprint: str
    point_role: str
    bed_mass_vector: object
    q4_equilibrium_mass_concentration: object
    class_status: object
    status: object
    producer_groups: tuple[ResidentDirectVectorProducerGroupV67, ...]
    face_channels: ResidentDirectVectorChannelsV67
    surface_fraction_application_count: int
    post_producer_density_conversion_count: int
    hydraulic_preprocessor_application_count: int
    channel_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorFamilyTopologyV67:
    """Reusable topology bound to the external topology dependency identity."""

    execution_fingerprint: str
    maximum_waterdepth_execution_fingerprint: str
    dependency_fingerprint: str
    face_topology: ResidentDirectVectorTopologyV67
    topology_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentDirectVectorFamilyTrialV67:
    """High-level trial binding producer channels to the Exner candidate."""

    execution_fingerprint: str
    maximum_waterdepth_execution_fingerprint: str
    parent_state_sha256: str
    topology_sha256: str
    cell_channel_sha256: str
    trace_channel_sha256: str
    face_trial: ResidentDirectVectorTrialV67
    trial_sha256: str
    device_id: int

    @property
    def candidate(self) -> ResidentDirectVectorStateV67:
        return self.face_trial.candidate


@dataclass(frozen=True)
class _ProducerGroup:
    method_id: str
    catalogue_entry_id: str
    route: str
    indices: tuple[int, ...]
    runtime: object


class ResidentDirectVectorFamilyLiveV67:
    """Run all six direct-vector methods through one accepted CUDA lifecycle."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(
        self, *, execution: Mapping[str, object],
        catalogue_entry_ids: tuple[str, ...] | list[str],
        maximum_waterdepth_execution: Mapping[str, object],
        cell_support_offset=None, cell_support_ids=None,
        trace_support_offset=None, trace_support_ids=None,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = load_direct_vector_family_execution_v67(execution)
        if (self.execution["backend"] != BACKEND or
                self.execution["integrator"] != INTEGRATOR or
                self.execution["automatic_backend_fallback"] is not False):
            raise DirectVectorFamilyLiveV67Error(
                "GPU family requires explicit implicit_fwave_midpoint_gpu "
                "execution with fallback disabled"
            )

        self.root = Path(__file__).resolve().parents[1]
        # Verification cannot be disabled: both the outer BRAIDS manifest and
        # the locked, unmodified Delft3D hashes are part of admission.
        self.outer_source_receipt = verify_direct_vector_family_sources_v67(
            self.execution, repository_root=self.root,
        )
        self._verify_local_sources()

        # The v68 receipt is mandatory even when MaximumWaterdepth is off.
        # That makes the source-ordered dwnvel/ERODED velocity semantics a
        # restart-relevant physical choice instead of an optional Boolean API.
        self.maximum_waterdepth_execution = (
            load_direct_vector_maximum_waterdepth_execution_v68(
                maximum_waterdepth_execution
            )
        )
        if (self.maximum_waterdepth_execution["parent_execution_fingerprint"]
                != self.execution["fingerprint"] or
                self.maximum_waterdepth_execution["parent_execution"]
                != self.execution or
                self.maximum_waterdepth_execution["topology_fingerprint"]
                != self.execution["dependencies"]["topology"]):
            raise DirectVectorFamilyLiveV67Error(
                "MaximumWaterdepth execution does not wrap this exact family "
                "execution and topology"
            )
        self.maximum_waterdepth_source_receipt = (
            verify_direct_vector_maximum_waterdepth_sources_v68(
                self.maximum_waterdepth_execution,
                repository_root=self.root,
            )
        )

        physical = self.execution["physical_profile"]
        self.physical = physical
        self.nclass = len(physical["fraction_ids"])
        self.bulk_density = cp.ascontiguousarray(cp.asarray(
            physical["bulk_density_kg_m3"], dtype=cp.float64,
        ))
        self.d50 = cp.ascontiguousarray(cp.asarray(
            physical["d50_m"], dtype=cp.float64,
        ))
        self.d90 = cp.ascontiguousarray(cp.asarray(
            physical["d90_m"], dtype=cp.float64,
        ))
        self.solid_density = cp.ascontiguousarray(cp.asarray(
            physical["solid_density_kg_m3"], dtype=cp.float64,
        ))
        self.transport_class = tuple(int(value) for value in
                                     physical["transport_classes"])
        self.routes = tuple(str(value) for value in physical["routes"])
        self.method_ids = tuple(
            str(row["method_id"]) for row in physical["fraction_methods"]
        )
        if any(method not in PRODUCER_TO_CATALOGUE for method in self.method_ids):
            raise DirectVectorFamilyLiveV67Error(
                "execution contains a producer profile without an explicit "
                "catalogue entry mapping"
            )
        # Public construction accepts live-catalogue entry IDs only.  The
        # shared bridge records (and authenticates) the current-only profile
        # mapping; passing an internal ``*_current`` ID is rejected there.
        bridge = direct_vector_catalogue_bridge_document_v67(
            catalogue_entry_ids, forcing_scope="current_only",
        )
        bridge = load_direct_vector_catalogue_bridge_v67(bridge)
        if direct_vector_profile_method_ids_v67(bridge) != self.method_ids:
            raise DirectVectorFamilyLiveV67Error(
                "public catalogue selection differs from the authenticated "
                "producer profiles in the execution receipt"
            )
        self.catalogue_entry_ids = tuple(
            row["catalogue_entry_id"] for row in bridge["fractions"]
        )
        self.catalogue_mapping = MappingProxyType(bridge)
        if (self.maximum_waterdepth_execution["catalogue_bridge_fingerprint"]
                != bridge["fingerprint"] or
                self.maximum_waterdepth_execution["catalogue_bridge"]
                != bridge):
            raise DirectVectorFamilyLiveV67Error(
                "MaximumWaterdepth catalogue binding differs from the public "
                "family selection"
            )

        self._parameters = tuple(
            self._parameter_values(row) for row in physical["fraction_methods"]
        )
        self.bed_factor = float(physical["bed_factor"])
        self.suspended_factor = float(physical["suspended_factor"])
        self.speed_epsilon = float(physical["speed_epsilon"])
        self.water_density = float(physical["water_density_kg_m3"])
        self.gravity = float(physical["gravity_m_s2"])
        self.minimum_work_depth = float(physical["minimum_work_depth_m"])
        self.roughness_method_id = str(physical["roughness_method_id"])
        self._roughness_mode = {
            "roughness.chezy": "chezy",
            "roughness.manning": "manning",
            "roughness.white_colebrook": "white_colebrook",
            "roughness.white_colebrook_waqua": "white_colebrook",
        }[self.roughness_method_id]
        self._waqua = self.roughness_method_id.endswith("_waqua")

        self.face_consumer = ResidentDirectVectorFaceExnerV67(
            execution_fingerprint=self.execution["fingerprint"],
            bulk_density_kg_m3=self.bulk_density,
            accepted_step_limit=int(self.execution["accepted_step_limit"]),
        )
        self._verify_consumer_receipt()
        self.maximum_waterdepth = DirectVectorMaximumWaterdepthBridgeV68CUDA(
            execution=self.maximum_waterdepth_execution,
            topology_fingerprint=self.execution["dependencies"]["topology"],
            cell_support_offset=cell_support_offset,
            cell_support_ids=cell_support_ids,
            trace_support_offset=trace_support_offset,
            trace_support_ids=trace_support_ids,
        )
        self._verify_maximum_waterdepth_receipt()
        self.route_composer = DirectVectorRouteComposerV66CUDA()
        self.roughness = HydraulicRoughnessCUDA()
        for component in (
                self.face_consumer, self.maximum_waterdepth,
                self.route_composer):
            if component.device_id != self.device_id:
                raise DirectVectorFamilyLiveV67Error(
                    "GPU family components selected different CUDA devices"
                )
        self.groups = self._build_groups()
        self.runtime_receipt = self._runtime_receipt()

    def _verify_local_sources(self) -> None:
        for relative, expected in _LOCAL_SOURCE_SHA256.items():
            path = (self.root / relative).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError as error:
                raise DirectVectorFamilyLiveV67Error(
                    "GPU family source path escaped the repository"
                ) from error
            actual = _sha256_file(path)
            if actual != expected:
                raise DirectVectorFamilyLiveV67Error(
                    f"GPU family source hash differs: {relative}"
                )

    def _verify_consumer_receipt(self) -> None:
        receipt = self.face_consumer.source_receipt
        face_path = (self.root / "braids_gpu/scalar_capacity_runtime.cu").resolve()
        inventory_path = (self.root / "braids_gpu/direct_vector_live_v66.cu").resolve()
        outer_rows = {
            str((self.root / row["path"]).resolve()): row["sha256"]
            for row in self.execution["implementation_sources"]
        }
        expected_face_hash = outer_rows.get(str(face_path))
        if expected_face_hash is None:
            raise DirectVectorFamilyLiveV67Error(
                "outer execution does not authenticate the face CUDA source"
            )
        if (
            Path(str(receipt["face_exner_source"])).resolve() != face_path
            or receipt["face_exner_sha256"] != expected_face_hash
            or Path(str(receipt["inventory_source"])).resolve()
                != inventory_path
            or receipt["inventory_sha256"]
                != _LOCAL_SOURCE_SHA256["braids_gpu/direct_vector_live_v66.cu"]
        ):
            raise DirectVectorFamilyLiveV67Error(
                "law-neutral face/Exner consumer source receipt differs"
            )

    def _verify_maximum_waterdepth_receipt(self) -> None:
        """Reject a bridge whose execution or authenticated bytes differ."""
        receipt = self.maximum_waterdepth.source_receipt
        if (self.maximum_waterdepth.execution
                != self.maximum_waterdepth_execution or
                receipt != self.maximum_waterdepth_source_receipt or
                self.maximum_waterdepth.device_id != self.device_id):
            raise DirectVectorFamilyLiveV67Error(
                "MaximumWaterdepth CUDA bridge receipt differs from the "
                "authenticated execution"
            )
        expected = {
            (category, row["path"]): row["sha256"]
            for category in ("reference_sources", "implementation_sources")
            for row in self.maximum_waterdepth_execution[category]
        }
        actual = {
            (row["category"], row["path"]): row["sha256"]
            for row in receipt["sources"]
            if row.get("verified") is True
        }
        if (actual != expected or
                receipt["execution_fingerprint"]
                != self.maximum_waterdepth_execution["fingerprint"] or
                receipt["source_manifest_fingerprint"]
                != self.maximum_waterdepth_execution[
                    "source_manifest_fingerprint"
                ]):
            raise DirectVectorFamilyLiveV67Error(
                "MaximumWaterdepth bridge source paths or hashes differ"
            )

    @staticmethod
    def _parameter_values(row: Mapping[str, object]) -> Mapping[str, float | int]:
        identity = row["parameter_identity"]
        native = identity["native_identity"]
        if native["schema_id"] == "capacity_parameters_v1":
            rows = native["parameters"]
        else:
            rows = native["document"]["parameters"]
        return MappingProxyType({item["keyword"]: item["value"] for item in rows})

    def _build_groups(self) -> tuple[_ProducerGroup, ...]:
        keys: list[tuple[str, str]] = []
        for method, route in zip(self.method_ids, self.routes, strict=True):
            key = (method, route)
            if key not in keys:
                keys.append(key)
        groups: list[_ProducerGroup] = []
        cp = self.cp
        for method, route in keys:
            indices = tuple(
                index for index, pair in enumerate(
                    zip(self.method_ids, self.routes, strict=True)
                ) if pair == (method, route)
            )
            if method in _SCALAR_METHOD:
                law, _ = _SCALAR_METHOD[method]
                contract = formula_contract(method)
                parameters = {}
                for specification in contract.parameters:
                    values = [
                        self._parameters[index][specification.keyword]
                        for index in indices
                    ]
                    dtype = (cp.int32 if specification.value_kind == "integer"
                             else cp.float64)
                    parameters[specification.keyword] = cp.ascontiguousarray(
                        cp.asarray(values, dtype=dtype)
                    )
                selected = cp.asarray(indices, dtype=cp.int32)
                runtime = ResidentScalarCapacityRuntime(
                    law,
                    d50=cp.ascontiguousarray(self.d50[selected]),
                    d90=cp.ascontiguousarray(self.d90[selected]),
                    solid_density=cp.ascontiguousarray(
                        self.solid_density[selected]
                    ),
                    parameters=parameters,
                    water_density=self.water_density,
                    gravity=self.gravity,
                    roughness_mode=self._roughness_mode,
                    minimum_work_depth=self.minimum_work_depth,
                    waqua_white_colebrook=self._waqua,
                )
            else:
                runtime = DirectProducerAdapterV67CUDA(method, route)
            if runtime.device_id != self.device_id:
                raise DirectVectorFamilyLiveV67Error(
                    "constitutive producer selected a different CUDA device"
                )
            groups.append(_ProducerGroup(
                method, PRODUCER_TO_CATALOGUE[method], route, indices, runtime,
            ))
        return tuple(groups)

    def _runtime_receipt(self) -> Mapping[str, object]:
        own_path = Path(__file__).resolve()
        body: dict[str, object] = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "backend": BACKEND,
            "integrator": INTEGRATOR,
            "automatic_backend_fallback": False,
            "device_id": self.device_id,
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "outer_source_receipt_fingerprint":
                self.outer_source_receipt["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "maximum_waterdepth_source_receipt_fingerprint":
                self.maximum_waterdepth_source_receipt["fingerprint"],
            "maximum_waterdepth":
                self.maximum_waterdepth_execution["maximum_waterdepth"],
            "hydraulic_preprocessor_application_count_per_point": 1,
            "high_level_source": {
                "path": str(own_path.relative_to(self.root)).replace("\\", "/"),
                "sha256": _sha256_file(own_path),
            },
            "local_source_sha256": dict(_LOCAL_SOURCE_SHA256),
            "fraction_free_producer_route": True,
            "active_fraction_owner": "face_exner_projection",
            "surface_fraction_application_count_before_face": 0,
            "post_producer_density_conversion_count": 0,
        }
        return MappingProxyType({**body, "fingerprint": _fingerprint(body)})

    def _array(
        self, name: str, value: object, ndim: int,
        shape: tuple[int, ...] | None = None, *, integer: bool = False,
    ):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} array"
            )
        if int(value.device.id) != self.device_id:
            raise DirectVectorFamilyLiveV67Error(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != shape:
            raise DirectVectorFamilyLiveV67Error(
                f"{name} must have shape {shape}, got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _flag(value: object) -> bool:
        return bool(value.item() if hasattr(value, "item") else value)

    def _digest(self, *arrays: object, label: str) -> str:
        return self.face_consumer._resident_digest(*arrays, label=label)

    def _roughness_state(self, depth, parameter):
        return self.roughness.state(
            depth, parameter, mode=self._roughness_mode,
            gravity=self.gravity,
            minimum_work_depth=self.minimum_work_depth,
            waqua_white_colebrook=self._waqua,
        )

    def _state_fields(
        self, *, state, roughness_parameter, kinematic_viscosity,
        active_mask, settling_velocity, point_role: str,
        support_depth=None, support_active=None,
    ) -> dict[str, object]:
        cp = self.cp
        npoint = int(state.shape[1])
        depth = cp.ascontiguousarray(state[0])
        safe_depth = cp.where(active_mask, depth, 1.0)
        transport_u = cp.ascontiguousarray(cp.where(
            active_mask, state[1], 0.0,
        ))
        transport_v = cp.ascontiguousarray(cp.where(
            active_mask, state[2], 0.0,
        ))
        hydraulic_u = cp.ascontiguousarray(cp.where(
            active_mask, state[1] / safe_depth, 0.0,
        ))
        hydraulic_v = cp.ascontiguousarray(cp.where(
            active_mask, state[2] / safe_depth, 0.0,
        ))
        # For the admitted current-only forcing, FLOW's
        # ustarc*Chezy/sqrt(g) target magnitude reduces to the hydraulic
        # speed.  The v68 bridge still performs dwnvel and EROSED's exact
        # direction reconstruction (including component zeroing and +eps)
        # once, before any fraction producer is selected.
        effective_speed = cp.ascontiguousarray(cp.hypot(
            hydraulic_u, hydraulic_v,
        ))
        chezy, z0, _, roughness_status = self._roughness_state(
            cp.ascontiguousarray(safe_depth), roughness_parameter,
        )
        if self._flag(cp.any(roughness_status != 0)):
            raise DirectVectorFamilyLiveV67Error(
                "hydraulic roughness rejected a producer point"
            )
        hydraulics = self.maximum_waterdepth.prepare(
            "cell" if point_role == "cells" else "trace",
            cell_depth=cp.ascontiguousarray(safe_depth),
            transport_u=transport_u, transport_v=transport_v,
            effective_speed=effective_speed,
            active=cp.ascontiguousarray(
                active_mask.astype(cp.int32, copy=False)
            ),
            support_depth=support_depth, support_active=support_active,
            direction_epsilon=1.0e-12,
            effective_direction_epsilon=self.speed_epsilon,
        )
        if (not isinstance(hydraulics, ResidentDirectVectorHydraulicsV68) or
                hydraulics.preprocessor_application_count != 1 or
                hydraulics.execution_fingerprint
                    != self.maximum_waterdepth_execution["fingerprint"] or
                hydraulics.support_kind
                    != ("cell" if point_role == "cells" else "trace")):
            raise DirectVectorFamilyLiveV67Error(
                "source-ordered hydraulic preprocessing receipt differs"
            )
        return {
            "npoint": npoint,
            "depth": cp.ascontiguousarray(safe_depth),
            "u": hydraulic_u,
            "v": hydraulic_v,
            "velocity": cp.ascontiguousarray(cp.stack(
                (hydraulic_u, hydraulic_v), axis=0,
            )),
            "roughness_parameter": roughness_parameter,
            "kinematic_viscosity": kinematic_viscosity,
            "settling_velocity": settling_velocity,
            "chezy": cp.ascontiguousarray(chezy),
            "roughness_z0": cp.ascontiguousarray(z0),
            "active_mask": active_mask,
            "hydraulics": hydraulics,
        }

    def _broadcast_class(self, values, npoint: int):
        cp = self.cp
        return cp.ascontiguousarray(
            cp.broadcast_to(values[:, None], (int(values.size), npoint))
        )

    def _broadcast_point(self, values, nclass: int):
        cp = self.cp
        return cp.ascontiguousarray(
            cp.broadcast_to(values[None, :], (nclass, int(values.size)))
        )

    def _vr1993_fields(self, group: _ProducerGroup, fields) -> dict[str, object]:
        cp = self.cp
        indices = cp.asarray(group.indices, dtype=cp.int32)
        npoint = int(fields["npoint"])
        d50 = cp.ascontiguousarray(self.d50[indices])
        d90 = cp.ascontiguousarray(self.d90[indices])
        rhos = cp.ascontiguousarray(self.solid_density[indices])
        rho = fields["depth"]*0.0 + np.float64(self.water_density)
        gravity = fields["depth"]*0.0 + np.float64(self.gravity)
        relative = (rhos[:, None]-rho[None, :])/rho[None, :]
        viscosity = fields["kinematic_viscosity"]
        dstar = d50[:, None]*(
            relative*gravity[None, :]/viscosity[None, :]**2
        )**np.float64(0.3333)
        shields = cp.where(
            dstar <= 4.0, 0.24/dstar,
            cp.where(
                dstar <= 10.0, 0.14/dstar**0.64,
                cp.where(
                    dstar <= 20.0, 0.04/dstar**0.10,
                    cp.where(dstar <= 150.0, 0.013*dstar**0.29, 0.055),
                ),
            ),
        )
        taucr = (rhos[:, None]-rho[None, :])*gravity[None, :]*d50[:, None]*shields
        params = [self._parameters[index] for index in group.indices]
        if any(int(row["iTauCr"]) != 1 for row in params):
            raise DirectVectorFamilyLiveV67Error(
                "Van Rijn 1993 iTauCr=2 requires the later typed critical-"
                "stress input route; it cannot be guessed"
            )
        iopkcw = tuple(int(row["IopKCW"]) for row in params)
        hydraulic_z0 = self._broadcast_point(fields["roughness_z0"], len(params))
        rdc = cp.asarray([float(row["RDC"]) for row in params], dtype=cp.float64)
        roughness_z0 = cp.ascontiguousarray(cp.where(
            cp.asarray(iopkcw, dtype=cp.int32)[:, None] == 1,
            hydraulic_z0,
            rdc[:, None]/30.0,
        ))
        return {
            "depth": fields["depth"],
            "velocity": fields["velocity"],
            "observation_height": fields["observation_height"],
            "water_density": cp.ascontiguousarray(rho),
            "gravity": cp.ascontiguousarray(gravity),
            "kinematic_viscosity": fields["kinematic_viscosity"],
            "additional_shear_stress": cp.zeros(npoint, dtype=cp.float64),
            "d50": d50,
            "d90": d90,
            "solid_density": rhos,
            "molecular_schmidt": cp.ones(len(params), dtype=cp.float64),
            "aks_factor": cp.asarray(
                [float(row["AksFac"]) for row in params], dtype=cp.float64,
            ),
            "maximum_volume_concentration": cp.full(
                len(params), 0.65, dtype=cp.float64,
            ),
            "minimum_depth": cp.full(
                len(params), self.minimum_work_depth, dtype=cp.float64,
            ),
            "settling_velocity": cp.ascontiguousarray(
                fields["settling_velocity"][indices]
            ),
            "roughness_z0": roughness_z0,
            "dstar": cp.ascontiguousarray(dstar),
            "critical_shear_stress": cp.ascontiguousarray(taucr),
        }

    def _direct_fields(self, group: _ProducerGroup, fields) -> dict[str, object]:
        cp = self.cp
        indices = cp.asarray(group.indices, dtype=cp.int32)
        nclass = len(group.indices)
        npoint = int(fields["npoint"])
        count = nclass*npoint

        def point(value):
            return cp.ascontiguousarray(
                self._broadcast_point(value, nclass).reshape(count)
            )

        def cls(value):
            return cp.ascontiguousarray(
                self._broadcast_class(value, npoint).reshape(count)
            )

        class_d50 = cp.ascontiguousarray(self.d50[indices])
        class_d90 = cp.ascontiguousarray(self.d90[indices])
        class_rhos = cp.ascontiguousarray(self.solid_density[indices])
        params = [self._parameters[index] for index in group.indices]
        contract = direct_current_contract_v66(group.method_id)
        integer_kernel_fields = INTEGER_FIELDS[contract.reference_selector]
        parameter_fields: dict[str, object] = {}
        for specification in contract.parameters:
            if specification.kernel_field is None:
                continue
            values = [row[specification.keyword] for row in params]
            # Parameter identity may be integer-valued while the historical
            # kernel ABI is real (Bijker's EpsSL is the important example).
            dtype = (cp.int32 if specification.kernel_field in integer_kernel_fields
                     else cp.float64)
            class_values = cp.asarray(values, dtype=dtype)
            parameter_fields[specification.kernel_field] = cp.ascontiguousarray(
                cp.broadcast_to(class_values[:, None], (nclass, npoint))
                .reshape(count)
            )
        zeros = cp.zeros(count, dtype=cp.float64)
        common = {
            "depth": point(fields["depth"]),
            "u": point(fields["u"]),
            "v": point(fields["v"]),
            "d50": cls(class_d50),
            "d90": cls(class_d90),
            "solid_density": cls(class_rhos),
            "water_density": cp.full(count, self.water_density, dtype=cp.float64),
            "gravity": cp.full(count, self.gravity, dtype=cp.float64),
            "chezy": point(fields["chezy"]),
            "wave_height": zeros,
            "wave_period": zeros.copy(),
            "wavelength": zeros.copy(),
            "wave_angle": zeros.copy(),
            "orbital_velocity": zeros.copy(),
            "breaker_turbulence": zeros.copy(),
            "bed_slope_x": zeros.copy(),
            "bed_slope_y": zeros.copy(),
        }
        if group.method_id == METHOD_BIJKER_CURRENT:
            common["settling_velocity"] = cp.ascontiguousarray(
                fields["settling_velocity"][indices].reshape(count)
            )
        else:
            common.update({
                "d15": zeros.copy(),
                "kinematic_viscosity": point(fields["kinematic_viscosity"]),
                "bed_change_rate": zeros.copy(),
            })
        common.update(parameter_fields)
        required = FORMULA_FIELDS[contract.reference_selector]
        missing = tuple(name for name in required if name not in common)
        extra = tuple(name for name in common if name not in required)
        if missing or extra:
            raise DirectVectorFamilyLiveV67Error(
                f"{group.method_id} generated fields differ; "
                f"missing={missing}, extra={extra}"
            )
        # Preserve exact integer ABI for the modern formula controls.
        for name in INTEGER_FIELDS[contract.reference_selector]:
            common[name] = cp.ascontiguousarray(
                common[name].astype(cp.int32, copy=False)
            )
        return common

    def _group_identity_rows(self, group: _ProducerGroup):
        rows = self.physical["fraction_methods"]
        return (
            tuple(rows[index]["producer_identity"]["fingerprint"]
                  for index in group.indices),
            tuple(rows[index]["parameter_identity"]["fingerprint"]
                  for index in group.indices),
        )

    def _group_token(
        self, group: _ProducerGroup, raw, composed, producer_status,
        route_status,
    ) -> str:
        producer_ids, parameter_ids = self._group_identity_rows(group)
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "catalogue_entry_id": group.catalogue_entry_id,
            "producer_profile_method_id": group.method_id,
            "forcing_variant": "current_only",
            "route": group.route,
            "fraction_indices": list(group.indices),
            "producer_identity_fingerprints": list(producer_ids),
            "parameter_identity_fingerprints": list(parameter_ids),
            "surface_fraction_application_count": 0,
            "post_producer_density_conversion_count": 0,
            "hydraulic_preprocessor_application_count": 1,
            "resident_arrays_sha256": self._digest(
                raw, composed, producer_status, route_status,
                label="direct_vector_family_producer_group_v67",
            ),
        })

    def _produce_group(
        self, group: _ProducerGroup, fields, *, point_role: str,
        owner=None, role=None,
    ) -> ResidentDirectVectorProducerGroupV67:
        cp = self.cp
        npoint = int(fields["npoint"])
        nclass = len(group.indices)
        count = nclass*npoint
        active_mask = fields["active_mask"]
        method_fields = self.maximum_waterdepth.producer_fields(
            group.method_id, fields["hydraulics"], fields,
        )
        indices = cp.asarray(group.indices, dtype=cp.int32)
        bed_factor = cp.full(
            nclass if group.method_id == METHOD_VAN_RIJN_1993_CURRENT else count,
            self.bed_factor, dtype=cp.float64,
        )
        suspended_factor = cp.full_like(bed_factor, self.suspended_factor)
        speed_epsilon = cp.full(
            (nclass, npoint) if group.method_id == METHOD_VAN_RIJN_1993_CURRENT
            else count,
            self.speed_epsilon, dtype=cp.float64,
        )

        if group.method_id in _SCALAR_METHOD:
            law, composer_method = _SCALAR_METHOD[group.method_id]
            producer_state = self.maximum_waterdepth.scalar_q(
                group.method_id, fields["hydraulics"],
            )
            if point_role == "cells":
                produced = group.runtime.cells(
                    q=producer_state,
                    roughness_parameter=method_fields["roughness_parameter"],
                    kinematic_viscosity=method_fields["kinematic_viscosity"],
                    active=cp.ascontiguousarray(active_mask.astype(cp.int32)),
                    dry_tolerance=0.0,
                )
            else:
                produced = group.runtime.traces(
                    trace=producer_state, owner=owner, role=role,
                    roughness_parameter=method_fields["roughness_parameter"],
                    kinematic_viscosity=method_fields["kinematic_viscosity"],
                    dry_tolerance=0.0,
                )
            raw_bed = cp.ascontiguousarray(produced.bed_mass.reshape(2, count))
            raw_suspended = cp.ascontiguousarray(
                produced.suspended_mass.reshape(2, count)
            )
            raw_concentration = cp.zeros(count, dtype=cp.float64)
            raw = cp.ascontiguousarray(cp.concatenate((
                raw_bed, raw_suspended, raw_concentration.reshape(1, count),
            ), axis=0))
            depth_points = cp.ascontiguousarray(
                self._broadcast_point(
                    method_fields["depth"], nclass,
                ).reshape(count)
            )
            method_velocity = cp.ascontiguousarray(cp.stack((
                method_fields["u"], method_fields["v"],
            ), axis=0))
            velocity_points = cp.ascontiguousarray(cp.broadcast_to(
                method_velocity[:, None, :], (2, nclass, npoint),
            ).reshape(2, count))
            composed_result = self.route_composer.compose(
                method=composer_method, route=group.route,
                depth=depth_points, velocity=velocity_points,
                raw_bed_mass=raw_bed,
                raw_suspended_mass=raw_suspended,
                raw_equilibrium_mass_concentration=raw_concentration,
                bed_factor=cp.full(count, self.bed_factor, dtype=cp.float64),
                suspended_factor=cp.full(
                    count, self.suspended_factor, dtype=cp.float64,
                ),
                active_fraction=cp.ones(count, dtype=cp.float64),
                speed_epsilon=cp.full(
                    count, self.speed_epsilon, dtype=cp.float64,
                ),
                reject_invalid=False,
            )
            composed = composed_result.values
            producer_status = cp.ascontiguousarray(cp.broadcast_to(
                produced.status[None, :], (nclass, npoint),
            ).reshape(count).astype(cp.int32, copy=False))
            route_status = composed_result.status
            if int(law) != int(produced.law):
                raise DirectVectorFamilyLiveV67Error(
                    "scalar producer law differs from its authenticated group"
                )
        else:
            if group.method_id == METHOD_VAN_RIJN_1993_CURRENT:
                producer_fields = self._vr1993_fields(group, method_fields)
            else:
                producer_fields = self._direct_fields(group, method_fields)
            adapted = group.runtime.evaluate_for_face_lifecycle(
                producer_fields=producer_fields,
                bed_factor=bed_factor,
                suspended_factor=suspended_factor,
                speed_epsilon=speed_epsilon,
                reject_invalid=False,
            )
            adapted.require_deferred_surface_fraction()
            raw = cp.ascontiguousarray(adapted.raw_values.reshape(5, count))
            composed = cp.ascontiguousarray(
                adapted.composed_values.reshape(3, count)
            )
            producer_status = cp.ascontiguousarray(
                adapted.producer_status.reshape(count)
            )
            route_status = cp.ascontiguousarray(
                adapted.route_status.reshape(count)
            )

        active_points = cp.ascontiguousarray(
            self._broadcast_point(active_mask.astype(cp.float64), nclass)
            .reshape(count) != 0.0
        )
        raw = cp.ascontiguousarray(cp.where(active_points[None, :], raw, 0.0))
        composed = cp.ascontiguousarray(cp.where(
            active_points[None, :], composed, 0.0,
        ))
        producer_status = cp.ascontiguousarray(cp.where(
            active_points, producer_status, 0,
        ).astype(cp.int32, copy=False))
        route_status = cp.ascontiguousarray(cp.where(
            active_points, route_status, 0,
        ).astype(cp.int32, copy=False))
        producer_ids, parameter_ids = self._group_identity_rows(group)
        token = self._group_token(
            group, raw, composed, producer_status, route_status,
        )
        return ResidentDirectVectorProducerGroupV67(
            self.execution["fingerprint"],
            self.maximum_waterdepth_execution["fingerprint"],
            group.catalogue_entry_id,
            group.method_id, "current_only", group.route, group.indices,
            producer_ids, parameter_ids, raw, composed, producer_status,
            route_status, 0, 0, 1, token, self.device_id,
        )

    def _produce(
        self, *, point_role: str, q_or_trace, roughness_parameter,
        kinematic_viscosity, active_mask, settling_velocity,
        support_depth=None, support_active=None, owner=None, role=None,
    ) -> ResidentDirectVectorFamilyChannelsV67:
        cp = self.cp
        npoint = int(q_or_trace.shape[1])
        fields = self._state_fields(
            state=q_or_trace, roughness_parameter=roughness_parameter,
            kinematic_viscosity=kinematic_viscosity,
            active_mask=active_mask, settling_velocity=settling_velocity,
            point_role=point_role, support_depth=support_depth,
            support_active=support_active,
        )
        bed = cp.zeros((2, self.nclass, npoint), dtype=cp.float64)
        q4 = cp.zeros((self.nclass, npoint), dtype=cp.float64)
        class_status = cp.zeros((self.nclass, npoint), dtype=cp.int32)
        receipts = []
        for group in self.groups:
            receipt = self._produce_group(
                group, fields, point_role=point_role, owner=owner, role=role,
            )
            receipts.append(receipt)
            reshaped = receipt.composed_values.reshape(3, len(group.indices), npoint)
            status = cp.maximum(
                receipt.producer_status, receipt.route_status,
            ).reshape(len(group.indices), npoint)
            for local, global_index in enumerate(group.indices):
                bed[:, global_index, :] = reshaped[0:2, local, :]
                q4[global_index, :] = reshaped[2, local, :]
                class_status[global_index, :] = status[local, :]
        bed = cp.ascontiguousarray(bed)
        q4 = cp.ascontiguousarray(q4)
        class_status = cp.ascontiguousarray(class_status)
        status = cp.ascontiguousarray(cp.max(class_status, axis=0))
        group_fingerprints = [receipt.group_sha256 for receipt in receipts]
        producer_fingerprint = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "point_role": point_role,
            "group_fingerprints": group_fingerprints,
            "surface_fraction_application_count": 0,
            "post_producer_density_conversion_count": 0,
            "hydraulic_preprocessor_application_count": 1,
            "resident_arrays_sha256": self._digest(
                bed, q4, class_status, status,
                label=f"direct_vector_family_{point_role}_v67",
            ),
        })
        face_channels = self.face_consumer.prepare_channels(
            point_role=point_role, bed_mass_vector=bed, status=status,
            producer_fingerprint=producer_fingerprint,
            surface_fraction_application_count=0,
        )
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "point_role": point_role,
            "producer_fingerprint": producer_fingerprint,
            "face_channel_sha256": face_channels.channel_sha256,
            "group_fingerprints": group_fingerprints,
            "surface_fraction_application_count": 0,
            "post_producer_density_conversion_count": 0,
            "hydraulic_preprocessor_application_count": 1,
            "resident_arrays_sha256": self._digest(
                bed, q4, class_status, status,
                label=f"direct_vector_family_{point_role}_v67",
            ),
        })
        return ResidentDirectVectorFamilyChannelsV67(
            self.execution["fingerprint"],
            self.maximum_waterdepth_execution["fingerprint"],
            self.catalogue_mapping["fingerprint"], point_role, bed, q4,
            class_status, status, tuple(receipts), face_channels, 0, 0, 1,
            token, self.device_id,
        )

    def cells(
        self, *, q: object, roughness_parameter: object,
        kinematic_viscosity: object, active: object, dry_tolerance: float,
        settling_velocity: object, support_depth=None, support_active=None,
    ) -> ResidentDirectVectorFamilyChannelsV67:
        """Produce authenticated fraction-free cell channels."""
        cp = self.cp
        q = self._array("q", q, 2)
        if q.shape[0] != 3 or q.shape[1] < 1:
            raise DirectVectorFamilyLiveV67Error("q must have shape (3,ncell)")
        ncell = int(q.shape[1])
        roughness = self._array(
            "roughness_parameter", roughness_parameter, 1, (ncell,),
        )
        viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, 1, (ncell,),
        )
        active = self._array("active", active, 1, (ncell,), integer=True)
        settling = self._array(
            "settling_velocity", settling_velocity, 2,
            (self.nclass, ncell),
        )
        dry = _finite(dry_tolerance, "dry tolerance", nonnegative=True)
        valid = (
            cp.all(cp.isfinite(q), axis=0) & (q[0] > dry) &
            (active == 1)
        )
        if (not self._flag(cp.all(valid)) or
                not self._flag(cp.all(cp.isfinite(viscosity) & (viscosity > 0.0))) or
                not self._flag(cp.all(cp.isfinite(settling) & (settling >= 0.0)))):
            raise DirectVectorFamilyLiveV67Error(
                "v67 family is currently admitted only for fully wet, active, "
                "finite cell states"
            )
        return self._produce(
            point_role="cells", q_or_trace=q,
            roughness_parameter=roughness,
            kinematic_viscosity=viscosity,
            active_mask=cp.ones(ncell, dtype=cp.bool_),
            settling_velocity=settling,
            support_depth=support_depth, support_active=support_active,
        )

    def traces(
        self, *, trace: object, owner: object, role: object,
        roughness_parameter: object, kinematic_viscosity: object,
        dry_tolerance: float, settling_velocity: object,
        support_depth=None, support_active=None,
    ) -> ResidentDirectVectorFamilyChannelsV67:
        """Produce explicit inlet/outlet trace channels; other roles are zero."""
        cp = self.cp
        trace = self._array("trace", trace, 2)
        if trace.shape[0] != 3:
            raise DirectVectorFamilyLiveV67Error("trace must have shape (3,nface)")
        nface = int(trace.shape[1])
        owner = self._array("owner", owner, 1, (nface,), integer=True)
        role = self._array("role", role, 1, (nface,), integer=True)
        roughness = self._array("roughness_parameter", roughness_parameter, 1)
        ncell = int(roughness.size)
        viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, 1, (ncell,),
        )
        settling = self._array(
            "settling_velocity", settling_velocity, 2,
            (self.nclass, nface),
        )
        dry = _finite(dry_tolerance, "dry tolerance", nonnegative=True)
        known = cp.isin(role, cp.asarray((
            FACE_INTERNAL, FACE_TRACE_INLET, FACE_TRACE_OUTLET,
            FACE_OWNER_OUTLET, FACE_CLOSED,
        ), dtype=cp.int32))
        owner_ok = (owner >= 0) & (owner < ncell)
        trace_role = (role == FACE_TRACE_INLET) | (role == FACE_TRACE_OUTLET)
        valid_trace = (
            (~trace_role) | (
                owner_ok & cp.all(cp.isfinite(trace), axis=0) &
                (trace[0] > dry)
            )
        )
        if (not self._flag(cp.all(known & valid_trace)) or
                not self._flag(cp.all(cp.isfinite(viscosity) & (viscosity > 0.0))) or
                not self._flag(cp.all(cp.isfinite(settling) & (settling >= 0.0)))):
            raise DirectVectorFamilyLiveV67Error(
                "open trace state/role/owner differs from the fully wet scope"
            )
        safe_owner = cp.clip(owner, 0, max(ncell-1, 0))
        point_roughness = cp.ascontiguousarray(roughness[safe_owner])
        point_viscosity = cp.ascontiguousarray(viscosity[safe_owner])
        # Non-trace faces carry safe inert state through constitutive kernels;
        # their output is zeroed before a channel receipt is formed.
        safe_trace = cp.ascontiguousarray(cp.stack((
            cp.where(trace_role, trace[0], 1.0),
            cp.where(trace_role, trace[1], 0.0),
            cp.where(trace_role, trace[2], 0.0),
        ), axis=0))
        return self._produce(
            point_role="traces", q_or_trace=safe_trace,
            roughness_parameter=point_roughness,
            kinematic_viscosity=point_viscosity,
            active_mask=cp.ascontiguousarray(trace_role),
            settling_velocity=settling, owner=owner, role=role,
            support_depth=support_depth, support_active=support_active,
        )

    def _validate_group_receipt(
        self, receipt: ResidentDirectVectorProducerGroupV67,
    ) -> None:
        key = (receipt.producer_profile_method_id, receipt.route,
               receipt.fraction_indices)
        group = next((value for value in self.groups if
                      (value.method_id, value.route, value.indices) == key), None)
        if (group is None or receipt.execution_fingerprint
                != self.execution["fingerprint"] or
                receipt.maximum_waterdepth_execution_fingerprint
                != self.maximum_waterdepth_execution["fingerprint"] or
                receipt.catalogue_entry_id != group.catalogue_entry_id or
                receipt.forcing_variant != "current_only" or
                receipt.device_id != self.device_id or
                receipt.surface_fraction_application_count != 0 or
                receipt.post_producer_density_conversion_count != 0 or
                receipt.hydraulic_preprocessor_application_count != 1):
            raise DirectVectorFamilyLiveV67Error(
                "producer-group receipt differs from execution"
            )
        expected = self._group_token(
            group, receipt.raw_values, receipt.composed_values,
            receipt.producer_status, receipt.route_status,
        )
        if expected != receipt.group_sha256:
            raise DirectVectorFamilyLiveV67Error(
                "producer-group resident arrays are stale or mutated"
            )

    def _validate_channels(
        self, channels: ResidentDirectVectorFamilyChannelsV67,
        *, point_role: str,
    ) -> None:
        if (not isinstance(channels, ResidentDirectVectorFamilyChannelsV67) or
                channels.execution_fingerprint != self.execution["fingerprint"] or
                channels.maximum_waterdepth_execution_fingerprint
                    != self.maximum_waterdepth_execution["fingerprint"] or
                channels.catalogue_mapping_fingerprint
                    != self.catalogue_mapping["fingerprint"] or
                channels.point_role != point_role or
                channels.device_id != self.device_id or
                channels.surface_fraction_application_count != 0 or
                channels.post_producer_density_conversion_count != 0 or
                channels.hydraulic_preprocessor_application_count != 1 or
                len(channels.producer_groups) != len(self.groups)):
            raise DirectVectorFamilyLiveV67Error(
                f"{point_role} family-channel receipt differs"
            )
        for receipt in channels.producer_groups:
            self._validate_group_receipt(receipt)
        group_fingerprints = [
            receipt.group_sha256 for receipt in channels.producer_groups
        ]
        producer_fingerprint = channels.face_channels.producer_fingerprint
        expected_producer = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "point_role": point_role,
            "group_fingerprints": group_fingerprints,
            "surface_fraction_application_count": 0,
            "post_producer_density_conversion_count": 0,
            "hydraulic_preprocessor_application_count": 1,
            "resident_arrays_sha256": self._digest(
                channels.bed_mass_vector,
                channels.q4_equilibrium_mass_concentration,
                channels.class_status, channels.status,
                label=f"direct_vector_family_{point_role}_v67",
            ),
        })
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "point_role": point_role,
            "producer_fingerprint": producer_fingerprint,
            "face_channel_sha256": channels.face_channels.channel_sha256,
            "group_fingerprints": group_fingerprints,
            "surface_fraction_application_count": 0,
            "post_producer_density_conversion_count": 0,
            "hydraulic_preprocessor_application_count": 1,
            "resident_arrays_sha256": self._digest(
                channels.bed_mass_vector,
                channels.q4_equilibrium_mass_concentration,
                channels.class_status, channels.status,
                label=f"direct_vector_family_{point_role}_v67",
            ),
        })
        if (producer_fingerprint != expected_producer or
                channels.channel_sha256 != expected):
            raise DirectVectorFamilyLiveV67Error(
                f"{point_role} family-channel arrays are stale or mutated"
            )

    def prepare_topology(self, **values) -> ResidentDirectVectorFamilyTopologyV67:
        face = self.face_consumer.prepare_topology(**values)
        dependency = self.execution["dependencies"]["topology"]
        token = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "dependency_fingerprint": dependency,
            "face_topology_sha256": face.topology_sha256,
            "semantic_validation_count": face.semantic_validation_count,
        })
        return ResidentDirectVectorFamilyTopologyV67(
            self.execution["fingerprint"],
            self.maximum_waterdepth_execution["fingerprint"], dependency,
            face, token, self.device_id,
        )

    def _validate_topology(
        self, topology: ResidentDirectVectorFamilyTopologyV67,
    ) -> None:
        if (not isinstance(topology, ResidentDirectVectorFamilyTopologyV67) or
                topology.execution_fingerprint != self.execution["fingerprint"] or
                topology.maximum_waterdepth_execution_fingerprint
                    != self.maximum_waterdepth_execution["fingerprint"] or
                topology.dependency_fingerprint
                    != self.execution["dependencies"]["topology"] or
                topology.device_id != self.device_id):
            raise DirectVectorFamilyLiveV67Error(
                "family topology receipt differs from execution"
            )
        self.face_consumer._validate_topology_receipt(topology.face_topology)
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "dependency_fingerprint": topology.dependency_fingerprint,
            "face_topology_sha256": topology.face_topology.topology_sha256,
            "semantic_validation_count":
                topology.face_topology.semantic_validation_count,
        })
        if expected != topology.topology_sha256:
            raise DirectVectorFamilyLiveV67Error(
                "family topology arrays are stale or mutated"
            )

    def resolve_state(self, **values) -> ResidentDirectVectorStateV67:
        if self.execution["accepted_state_admitted"] is not True:
            raise DirectVectorFamilyLiveV67Error(
                "TRA_COMBINE is diagnostic-only until Q4 owns suspended mass"
            )
        return self.face_consumer.resolve_state(**values)

    def _trial_token(
        self, state: ResidentDirectVectorStateV67,
        topology: ResidentDirectVectorFamilyTopologyV67,
        cells: ResidentDirectVectorFamilyChannelsV67,
        traces: ResidentDirectVectorFamilyChannelsV67,
        face_trial: ResidentDirectVectorTrialV67,
    ) -> str:
        return _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": face_trial.candidate.state_sha256,
            "topology_sha256": topology.topology_sha256,
            "cell_channel_sha256": cells.channel_sha256,
            "trace_channel_sha256": traces.channel_sha256,
            "face_trial_sha256": face_trial.trial_sha256,
            "surface_fraction_application_count_before_face": 0,
            "active_fraction_application_count":
                face_trial.active_fraction_application_count,
            "inventory_fill_application_count":
                face_trial.inventory_fill_application_count,
            "post_producer_density_conversion_count": 0,
        })

    def begin_trial(
        self, *, state: ResidentDirectVectorStateV67,
        cells: ResidentDirectVectorFamilyChannelsV67,
        traces: ResidentDirectVectorFamilyChannelsV67,
        topology: ResidentDirectVectorFamilyTopologyV67,
        dt_s: float, morphological_factor: float = 1.0,
        left_bed: object | None = None, right_bed: object | None = None,
        cancelled=lambda: False,
    ) -> ResidentDirectVectorFamilyTrialV67:
        """Create a class-1 candidate; class 3 remains a Q4 diagnostic."""
        if (self.execution["accepted_state_admitted"] is not True or
                any(value != TRA_BEDLOAD for value in self.transport_class)):
            raise DirectVectorFamilyLiveV67Error(
                "TRA_COMBINE is diagnostic-only until Q4 owns suspended mass"
            )
        self._validate_channels(cells, point_role="cells")
        self._validate_channels(traces, point_role="traces")
        self._validate_topology(topology)
        face_trial = self.face_consumer.begin_trial(
            state=state, cells=cells.face_channels,
            traces=traces.face_channels, topology=topology.face_topology,
            dt_s=dt_s, morphological_factor=morphological_factor,
            left_bed=left_bed, right_bed=right_bed, cancelled=cancelled,
        )
        token = self._trial_token(
            state, topology, cells, traces, face_trial,
        )
        return ResidentDirectVectorFamilyTrialV67(
            self.execution["fingerprint"],
            self.maximum_waterdepth_execution["fingerprint"],
            state.state_sha256,
            topology.topology_sha256, cells.channel_sha256,
            traces.channel_sha256, face_trial, token, self.device_id,
        )

    def _validate_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorFamilyTrialV67,
    ) -> None:
        if (not isinstance(trial, ResidentDirectVectorFamilyTrialV67) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.maximum_waterdepth_execution_fingerprint
                    != self.maximum_waterdepth_execution["fingerprint"] or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.device_id != self.device_id):
            raise DirectVectorFamilyLiveV67Error(
                "family trial parent or execution differs"
            )
        self.face_consumer._validate_trial(state, trial.face_trial)
        expected = _fingerprint({
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": trial.face_trial.candidate.state_sha256,
            "topology_sha256": trial.topology_sha256,
            "cell_channel_sha256": trial.cell_channel_sha256,
            "trace_channel_sha256": trial.trace_channel_sha256,
            "face_trial_sha256": trial.face_trial.trial_sha256,
            "surface_fraction_application_count_before_face": 0,
            "active_fraction_application_count":
                trial.face_trial.active_fraction_application_count,
            "inventory_fill_application_count":
                trial.face_trial.inventory_fill_application_count,
            "post_producer_density_conversion_count": 0,
        })
        if expected != trial.trial_sha256:
            raise DirectVectorFamilyLiveV67Error(
                "family trial receipt is stale or mutated"
            )

    def commit_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorFamilyTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.commit_trial(state, trial.face_trial)

    def rollback_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorFamilyTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.rollback_trial(state, trial.face_trial)

    def cancel_trial(
        self, state: ResidentDirectVectorStateV67,
        trial: ResidentDirectVectorFamilyTrialV67,
    ) -> ResidentDirectVectorStateV67:
        self._validate_trial(state, trial)
        return self.face_consumer.cancel_trial(state, trial.face_trial)

    def state_checkpoint_document(
        self, state: ResidentDirectVectorStateV67,
    ) -> dict[str, object]:
        """Return a v68-bound portable checkpoint, never trial arrays."""
        self.face_consumer._validate_state(state)
        cp = self.cp
        portable = resolve_direct_vector_family_state_v67(
            execution=self.execution,
            bed_mass_kg_m2=cp.asnumpy(state.bed_mass_kg_m2),
            active_fraction=cp.asnumpy(state.active_fraction),
            boundary_signed_kg=cp.asnumpy(state.boundary_signed_kg),
            boundary_absolute_kg=cp.asnumpy(state.boundary_absolute_kg),
            accepted_step=state.accepted_step, time_s=state.time_s,
        )
        family_checkpoint = direct_vector_family_state_checkpoint_v67(portable)
        body: dict[str, object] = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "maximum_waterdepth_execution_fingerprint":
                self.maximum_waterdepth_execution["fingerprint"],
            "maximum_waterdepth":
                self.maximum_waterdepth_execution["maximum_waterdepth"],
            "topology_fingerprint":
                self.maximum_waterdepth_execution["topology_fingerprint"],
            "catalogue_mapping_fingerprint":
                self.catalogue_mapping["fingerprint"],
            "family_state_checkpoint": family_checkpoint,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def load_state_checkpoint(
        self, value: Mapping[str, object],
    ) -> ResidentDirectVectorStateV67:
        if not isinstance(value, Mapping):
            raise DirectVectorFamilyLiveV67Error(
                "GPU family checkpoint must be an authenticated object"
            )
        expected_keys = {
            "schema", "scope_id", "execution_fingerprint",
            "maximum_waterdepth_execution_fingerprint",
            "maximum_waterdepth", "topology_fingerprint",
            "catalogue_mapping_fingerprint", "family_state_checkpoint",
            "fingerprint",
        }
        if set(value) != expected_keys:
            raise DirectVectorFamilyLiveV67Error(
                "GPU family checkpoint schema differs"
            )
        body = {key: value[key] for key in value if key != "fingerprint"}
        if (value["schema"] != SCHEMA or
                value["scope_id"] != CHECKPOINT_SCOPE_ID or
                value["execution_fingerprint"]
                    != self.execution["fingerprint"] or
                value["maximum_waterdepth_execution_fingerprint"]
                    != self.maximum_waterdepth_execution["fingerprint"] or
                value["maximum_waterdepth"]
                    is not self.maximum_waterdepth_execution[
                        "maximum_waterdepth"
                    ] or
                value["topology_fingerprint"]
                    != self.maximum_waterdepth_execution[
                        "topology_fingerprint"
                    ] or
                value["catalogue_mapping_fingerprint"]
                    != self.catalogue_mapping["fingerprint"] or
                value["fingerprint"] != _fingerprint(body)):
            raise DirectVectorFamilyLiveV67Error(
                "GPU family checkpoint switch, topology, or execution "
                "identity differs"
            )
        portable = load_direct_vector_family_state_checkpoint_v67(
            value["family_state_checkpoint"], execution=self.execution,
        )
        cp = self.cp
        return self.face_consumer.resolve_state(
            bed_mass_kg_m2=cp.ascontiguousarray(cp.asarray(
                portable.bed_mass_kg_m2, dtype=cp.float64,
            )),
            active_fraction=cp.ascontiguousarray(cp.asarray(
                portable.active_fraction, dtype=cp.float64,
            )),
            boundary_signed_kg=cp.ascontiguousarray(cp.asarray(
                portable.boundary_signed_kg, dtype=cp.float64,
            )),
            boundary_absolute_kg=cp.ascontiguousarray(cp.asarray(
                portable.boundary_absolute_kg, dtype=cp.float64,
            )),
            accepted_step=portable.accepted_step, time_s=portable.time_s,
        )


__all__ = [
    "BACKEND", "CATALOGUE_TO_PRODUCER", "INTEGRATOR",
    "PRODUCER_TO_CATALOGUE", "ResidentDirectVectorFamilyChannelsV67",
    "ResidentDirectVectorFamilyLiveV67", "ResidentDirectVectorFamilyTopologyV67",
    "ResidentDirectVectorFamilyTrialV67",
    "ResidentDirectVectorProducerGroupV67", "SCHEMA", "SCOPE_ID",
]
