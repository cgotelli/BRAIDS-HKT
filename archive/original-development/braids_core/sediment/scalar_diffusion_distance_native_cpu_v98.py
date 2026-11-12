"""Authenticated joined-interface distance adapter for native v98 diffusion.

The frozen combined diffusion kernel already accepts one centre-distance image.
This additive adapter changes only the two authenticated coarse/fine trace
entries before delegating exactly one launch to that kernel.  It does not own
another limiter, scalar update, clock, or physical-boundary ledger.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .coupled_amr_trace_cpu_v98 import (
    CoupledAMRInternalTraceCPUV98,
    validate_coupled_amr_internal_trace_structure_cpu_v98,
)
from .coupled_wetdry_owner_cpu_v97 import CoupledWetDryCPUExecutionV97
from .scalar_diffusion_trace_v98 import NativeScalarDiffusionTraceRuntimeV98
from .suspended_lifecycle_live_v76 import SuspendedLifecycleExecutionV76


ABI_ID = "braids-v98-native-scalar-diffusion-joined-distance-adapter-v1"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(
        list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class NativeScalarDiffusionDistanceRuntimeV98:
    """One-launch wrapper replacing only authenticated trace distances."""

    __slots__ = (
        "base_runtime", "topology", "fine_execution", "internal_trace",
        "receipt", "_coupled_distance")

    def __init__(
        self, base_runtime: NativeScalarDiffusionTraceRuntimeV98, *,
        topology: SuspendedLifecycleExecutionV76,
        fine_execution: CoupledWetDryCPUExecutionV97,
        internal_trace: CoupledAMRInternalTraceCPUV98,
    ) -> None:
        if (type(base_runtime) is not NativeScalarDiffusionTraceRuntimeV98 or
                type(topology) is not SuspendedLifecycleExecutionV76 or
                type(fine_execution) is not CoupledWetDryCPUExecutionV97 or
                internal_trace.joined_geometry_receipt_sha256 is None or
                internal_trace.physical_boundary_distance_reuse_count != 0):
            raise ValueError(
                "native v98 diffusion distance adapter requires qualified "
                "joined geometry")
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            internal_trace, fine_execution)
        distance = np.array(
            topology.centre_distance_m, dtype=np.float64, order="C", copy=True)
        faces = np.asarray(internal_trace.fine_faces, dtype=np.int64)
        distance[faces] = internal_trace.interface_centre_distance_m
        if np.any(distance <= 0.0) or not np.isfinite(distance).all():
            raise ValueError(
                "native v98 diffusion joined centre distance differs")
        distance.setflags(write=False)
        wrapper = Path(__file__).resolve()
        payload = {
            "abi": ABI_ID,
            "base_native_runtime_sha256":
                base_runtime.receipt["receipt_sha256"],
            "topology_fingerprint": topology.fingerprint,
            "fine_execution_fingerprint": fine_execution.fingerprint,
            "internal_trace_receipt_sha256": internal_trace.receipt_sha256,
            "joined_geometry_receipt_sha256":
                internal_trace.joined_geometry_receipt_sha256,
            "trace_faces": list(internal_trace.fine_faces),
            "coupled_distance_sha256": _digest(distance),
            "wrapper_sha256": _sha(wrapper),
            "native_diffusion_launch_count": 1,
            "distance_adapter_application_count": 1,
            "physical_boundary_distance_reuse_count": 0,
            "state_or_clock_publication_count": 0,
        }
        payload["receipt_sha256"] = _hash(payload)
        self.base_runtime = base_runtime
        self.topology = topology
        self.fine_execution = fine_execution
        self.internal_trace = internal_trace
        self._coupled_distance = distance
        self.receipt = payload

    def _validate_binding(
        self, topology: SuspendedLifecycleExecutionV76,
    ) -> None:
        validate_coupled_amr_internal_trace_structure_cpu_v98(
            self.internal_trace, self.fine_execution)
        payload = {
            key: value for key, value in self.receipt.items()
            if key != "receipt_sha256"
        }
        if (topology is not self.topology or
                type(topology) is not SuspendedLifecycleExecutionV76 or
                topology.fingerprint !=
                    self.receipt["topology_fingerprint"] or
                self.receipt["receipt_sha256"] != _hash(payload) or
                _sha(Path(__file__).resolve()) !=
                    self.receipt["wrapper_sha256"] or
                self.base_runtime.receipt["receipt_sha256"] !=
                    self.receipt["base_native_runtime_sha256"] or
                self.internal_trace.receipt_sha256 !=
                    self.receipt["internal_trace_receipt_sha256"] or
                _digest(self._coupled_distance) !=
                    self.receipt["coupled_distance_sha256"]):
            raise ValueError(
                "native v98 diffusion distance adapter identity differs")

    def evaluate(
        self, topology, dicuv_cell_m2_s, mass, depth, boundary,
        exchange_area, face_active, *, dt_s, trace_faces, trace_depth,
        trace_concentration, trace_dicuv, trace_wet_mask,
        trace_receipt_sha256, trace_source_sha256,
        trace_runtime_sha256, cap_receipt_sha256,
        trace_cap_receipt_sha256, evaluation_index, native_threads=4,
        cancelled: bool | Callable[[], bool] = False,
    ) -> dict:
        self._validate_binding(topology)
        faces = tuple(map(int, np.asarray(trace_faces).tolist()))
        if (faces != self.internal_trace.fine_faces or
                trace_source_sha256 !=
                    self.internal_trace.coarse_provisional_state_sha256):
            raise ValueError(
                "native v98 diffusion distance trace source differs")
        # `replace` is an ephemeral numerical view.  It is not a new v76
        # execution identity and is never published or admitted elsewhere.
        numerical_topology = replace(
            topology, centre_distance_m=self._coupled_distance)
        result = self.base_runtime.evaluate(
            numerical_topology, dicuv_cell_m2_s, mass, depth, boundary,
            exchange_area, face_active, dt_s=dt_s,
            trace_faces=trace_faces, trace_depth=trace_depth,
            trace_concentration=trace_concentration,
            trace_dicuv=trace_dicuv, trace_wet_mask=trace_wet_mask,
            trace_receipt_sha256=trace_receipt_sha256,
            trace_source_sha256=trace_source_sha256,
            trace_runtime_sha256=trace_runtime_sha256,
            cap_receipt_sha256=cap_receipt_sha256,
            trace_cap_receipt_sha256=trace_cap_receipt_sha256,
            evaluation_index=evaluation_index,
            native_threads=native_threads, cancelled=cancelled)
        result = dict(result)
        provenance = dict(result["trace_provenance"])
        provenance.update({
            "distance_adapter_receipt_sha256":
                self.receipt["receipt_sha256"],
            "joined_geometry_receipt_sha256":
                self.internal_trace.joined_geometry_receipt_sha256,
            "coupled_distance_sha256":
                self.receipt["coupled_distance_sha256"],
            "base_native_runtime_sha256":
                self.receipt["base_native_runtime_sha256"],
        })
        result["trace_provenance"] = provenance
        result["trace_inputs_sha256"] = _hash({
            "base_trace_inputs_sha256": result["trace_inputs_sha256"],
            "distance_adapter_receipt_sha256":
                self.receipt["receipt_sha256"],
            "coupled_distance_sha256":
                self.receipt["coupled_distance_sha256"],
        })
        result["native_runtime_sha256"] = \
            self.receipt["receipt_sha256"]
        return result

    def evaluate_arrays(self, *args, **kwargs):
        result = self.evaluate(*args, **kwargs)
        return tuple(result[key] for key in (
            "mass_kg_m2", "diffusive_face_mass_owner_out_kg",
            "diffusive_boundary_ledger_kg", "effective_diffusivity_m2_s"))


def bind_native_scalar_diffusion_distance_v98(
    base_runtime: NativeScalarDiffusionTraceRuntimeV98, *,
    topology: SuspendedLifecycleExecutionV76,
    fine_execution: CoupledWetDryCPUExecutionV97,
    internal_trace: CoupledAMRInternalTraceCPUV98,
) -> NativeScalarDiffusionDistanceRuntimeV98:
    return NativeScalarDiffusionDistanceRuntimeV98(
        base_runtime, topology=topology, fine_execution=fine_execution,
        internal_trace=internal_trace)


__all__ = [
    "ABI_ID", "NativeScalarDiffusionDistanceRuntimeV98",
    "bind_native_scalar_diffusion_distance_v98",
]
