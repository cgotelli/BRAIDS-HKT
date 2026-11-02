"""Resident-CUDA real-Q3 producer owner for the bounded v84 lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from braids_core.sediment.q3_producer_lifecycle_binding_v84 import (
    Q3AcceptedProducerRowsV84,
    Q3BoundRouteLifecycleTrialV84,
    Q3ProducerLifecycleV84Error,
    Q3RouteHandoffReceiptV84,
    _digest,
    _producer_token,
    _route_handoff_token,
    _trial_token,
    validate_binding_executions_v84,
    validate_producer_rows_v84,
    validate_route_handoff_receipt_v84,
)
from braids_core.sediment.route_lifecycle_composition_v81 import (
    RouteLifecycleExecutionV81,
    RouteLifecycleStateV81,
    validate_route_lifecycle_state_v81,
)
from braids_gpu.direct_vector_family_live_v67 import (
    ResidentDirectVectorFamilyLiveV67,
)
from braids_gpu.route_lifecycle_composition_v81 import (
    ResidentRouteHandoffV81,
    ResidentRouteLifecycleV81,
)


CUDA_SCOPE_ID = "braids-q4-real-q3-producer-lifecycle-binding-cuda-v84-v1"


@dataclass(frozen=True)
class ResidentQ3RouteHandoffV84:
    """Actual-CUDA producer and pure route bridge, with no scalar trial."""

    receipt: Q3RouteHandoffReceiptV84
    resident: ResidentRouteHandoffV81
    resident_sha256: str
    device_id: int


class ResidentQ3BoundRouteLifecycleV84:
    """Invoke real v67 CUDA producers and consume them in one v81 trial."""

    def __init__(
        self, *, producer_execution, catalogue_entry_ids,
        maximum_waterdepth_execution,
        lifecycle_execution: RouteLifecycleExecutionV81,
    ):
        self.execution = validate_binding_executions_v84(
            producer_execution, lifecycle_execution,
            backend="implicit_fwave_midpoint_gpu")
        if maximum_waterdepth_execution.get("maximum_waterdepth") is not False:
            raise Q3ProducerLifecycleV84Error(
                "v84 bounded CUDA owner requires MaximumWaterdepth off")
        self.producer = ResidentDirectVectorFamilyLiveV67(
            execution=producer_execution,
            catalogue_entry_ids=catalogue_entry_ids,
            maximum_waterdepth_execution=maximum_waterdepth_execution)
        self.lifecycle = ResidentRouteLifecycleV81(lifecycle_execution)
        self.lifecycle_execution = lifecycle_execution
        self.cp = self.producer.cp
        self.device_id = self.producer.device_id
        if self.lifecycle.device_id != self.device_id:
            raise Q3ProducerLifecycleV84Error(
                "v84 Q3 and Q4 CUDA owners selected different devices")
        self.runtime_receipt = {
            "scope_id": CUDA_SCOPE_ID,
            "execution_fingerprint": self.execution["fingerprint"],
            "producer_execution_fingerprint":
                self.producer.execution["fingerprint"],
            "producer_runtime_receipt_fingerprint": hashlib.sha256(
                repr(sorted(self.producer.runtime_receipt.items())).encode(
                    "utf-8")).hexdigest(),
            "lifecycle_execution_fingerprint": lifecycle_execution.fingerprint,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "actual_v67_cuda_producers": True,
            "actual_v81_cuda_lifecycle": True,
            "caller_supplied_raw_rows": False,
            "maximum_waterdepth": False,
            "automatic_cpu_fallback": False,
            "device_id": self.device_id,
        }

    def _device(self, value, shape, name, *, integer=False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.shape != shape or
                value.dtype != dtype or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise Q3ProducerLifecycleV84Error(
                f"v84 CUDA {name} must be resident contiguous {dtype} {shape}")
        return value

    def _produce(self, *, depth_after_m, endpoint_momentum_m2_s,
                 roughness_parameter, kinematic_viscosity_m2_s,
                 settling_velocity_m_s):
        cp = self.cp
        nc = self.lifecycle_execution.cell_count
        h = self._device(depth_after_m, (nc,), "depth")
        momentum = self._device(
            endpoint_momentum_m2_s, (2, nc), "endpoint momentum")
        roughness = self._device(
            roughness_parameter, (nc,), "Manning roughness")
        viscosity = self._device(
            kinematic_viscosity_m2_s, (nc,), "kinematic viscosity")
        settling = self._device(
            settling_velocity_m_s, (3, nc), "settling velocity")
        dry = (self.lifecycle_execution.lifecycle_execution
               .base_execution.dry_tolerance_m)
        threshold = (self.lifecycle_execution.lifecycle_execution
                     .base_execution.profile.sediment_threshold_m)
        wet = (h > dry) & (h > threshold)
        safe_h = cp.where(wet, h, 1.0)
        safe_q = cp.ascontiguousarray(cp.stack((
            safe_h, cp.where(wet, momentum[0], 0.0),
            cp.where(wet, momentum[1], 0.0)), axis=0))
        channels = self.producer.cells(
            q=safe_q, roughness_parameter=roughness,
            kinematic_viscosity=viscosity,
            active=cp.ones(nc, dtype=cp.int32), dry_tolerance=0.0,
            settling_velocity=settling)
        self.producer._validate_channels(channels, point_role="cells")
        raw = cp.zeros((5, 3, nc), dtype=cp.float64)
        for receipt in channels.producer_groups:
            values = receipt.raw_values.reshape(
                5, len(receipt.fraction_indices), nc)
            for local, global_index in enumerate(receipt.fraction_indices):
                raw[:, global_index, :] = values[:, local, :]
        raw = cp.ascontiguousarray(cp.where(wet[None, None, :], raw, 0.0))
        host_raw = np.ascontiguousarray(cp.asnumpy(raw))
        host_wet = np.ascontiguousarray(cp.asnumpy(wet).astype(np.int32))
        host_raw.setflags(write=False); host_wet.setflags(write=False)
        placeholder = Q3AcceptedProducerRowsV84(
            self.execution["fingerprint"], "actual_cuda", host_raw,
            host_wet, channels.channel_sha256,
            self.execution["producer_source_receipt_fingerprint"], 0, 0, "")
        evidence = Q3AcceptedProducerRowsV84(
            **{**placeholder.__dict__,
               "producer_sha256": _producer_token(placeholder)})
        validate_producer_rows_v84(evidence, self.execution["fingerprint"])
        return evidence, raw

    def begin_trial(
        self, parent: RouteLifecycleStateV81, *, roughness_parameter,
        kinematic_viscosity_m2_s, settling_velocity_m_s,
        cancelled: bool = False, **lifecycle_inputs,
    ) -> Q3BoundRouteLifecycleTrialV84:
        validate_route_lifecycle_state_v81(parent, self.lifecycle_execution)
        if cancelled:
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA cancelled before producer execution")
        forbidden = {key for key in lifecycle_inputs if key.startswith("raw_")}
        if forbidden:
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA public trial API rejects caller-supplied raw rows")
        evidence, raw = self._produce(
            depth_after_m=lifecycle_inputs["depth_after_m"],
            endpoint_momentum_m2_s=lifecycle_inputs["endpoint_momentum_m2_s"],
            roughness_parameter=roughness_parameter,
            kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
            settling_velocity_m_s=settling_velocity_m_s)
        try:
            inner = self.lifecycle.begin_trial(
                parent, raw_bed_x_kg_m_s=raw[0],
                raw_bed_y_kg_m_s=raw[1],
                raw_suspended_x_kg_m_s=raw[2],
                raw_suspended_y_kg_m_s=raw[3],
                raw_equilibrium_concentration_kg_m3=raw[4],
                **lifecycle_inputs)
        except Exception as error:
            raise Q3ProducerLifecycleV84Error(
                "v84 resident v67 output was rejected by v81") from error
        placeholder = Q3BoundRouteLifecycleTrialV84(
            self.execution["fingerprint"], parent.state_sha256,
            evidence, inner, "")
        return Q3BoundRouteLifecycleTrialV84(
            **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})

    def produce_route_handoff(
        self, *, depth_after_m, endpoint_momentum_m2_s,
        roughness_parameter, kinematic_viscosity_m2_s,
        settling_velocity_m_s, active_fraction,
    ) -> ResidentQ3RouteHandoffV84:
        """Run real v67 CUDA producers and v74 routing, but no v78 work."""
        cp = self.cp
        nc = self.lifecycle_execution.cell_count
        ng = self.lifecycle_execution.fraction_count
        depth = self._device(depth_after_m, (nc,), "pure handoff depth")
        momentum = self._device(
            endpoint_momentum_m2_s, (2, nc),
            "pure handoff endpoint momentum")
        fraction = self._device(
            active_fraction, (ng, nc), "pure handoff active fraction")
        if bool(cp.any((fraction < 0.0) | (fraction > 1.0)).item()):
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA pure active fraction is outside [0,1]")
        evidence, raw = self._produce(
            depth_after_m=depth,
            endpoint_momentum_m2_s=momentum,
            roughness_parameter=roughness_parameter,
            kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
            settling_velocity_m_s=settling_velocity_m_s)
        routed = self.lifecycle.compose_route_handoff(
            depth_after_m=depth, endpoint_momentum_m2_s=momentum,
            raw_bed_x_kg_m_s=raw[0], raw_bed_y_kg_m_s=raw[1],
            raw_suspended_x_kg_m_s=raw[2],
            raw_suspended_y_kg_m_s=raw[3],
            raw_equilibrium_concentration_kg_m3=raw[4],
            active_fraction=fraction)
        endpoint_sha = _digest(
            cp.asnumpy(depth), cp.asnumpy(momentum))
        fraction_sha = _digest(cp.asnumpy(fraction))
        placeholder = Q3RouteHandoffReceiptV84(
            self.execution["fingerprint"], "actual_cuda", evidence,
            routed.handoff, endpoint_sha, fraction_sha, 0, "")
        receipt = Q3RouteHandoffReceiptV84(
            **{**placeholder.__dict__,
               "receipt_sha256": _route_handoff_token(placeholder)})
        validate_route_handoff_receipt_v84(
            receipt, execution_fingerprint=self.execution["fingerprint"],
            route_execution=self.lifecycle_execution.route_execution,
            backend="actual_cuda",
            accepted_endpoint_sha256=endpoint_sha,
            active_fraction_sha256=fraction_sha)
        resident_sha = hashlib.sha256(
            (receipt.receipt_sha256+":"+routed.resident_sha256+":"+
             str(self.device_id)).encode("ascii")).hexdigest()
        return ResidentQ3RouteHandoffV84(
            receipt, routed, resident_sha, self.device_id)

    def validate_route_handoff(
        self, value: ResidentQ3RouteHandoffV84, *,
        accepted_endpoint_sha256: str, active_fraction_sha256: str,
    ) -> ResidentQ3RouteHandoffV84:
        if (not isinstance(value, ResidentQ3RouteHandoffV84) or
                value.device_id != self.device_id):
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA pure handoff device differs")
        validate_route_handoff_receipt_v84(
            value.receipt,
            execution_fingerprint=self.execution["fingerprint"],
            route_execution=self.lifecycle_execution.route_execution,
            backend="actual_cuda",
            accepted_endpoint_sha256=accepted_endpoint_sha256,
            active_fraction_sha256=active_fraction_sha256)
        self.lifecycle.validate_route_handoff(value.resident)
        expected = hashlib.sha256(
            (value.receipt.receipt_sha256+":"+
             value.resident.resident_sha256+":"+
             str(self.device_id)).encode("ascii")).hexdigest()
        if value.resident_sha256 != expected:
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA pure handoff resident receipt is stale")
        return value

    def _validate(self, parent, trial):
        validate_route_lifecycle_state_v81(parent, self.lifecycle_execution)
        if (not isinstance(trial, Q3BoundRouteLifecycleTrialV84) or
                trial.execution_fingerprint != self.execution["fingerprint"] or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.trial_sha256 != _trial_token(trial)):
            raise Q3ProducerLifecycleV84Error(
                "v84 CUDA trial is stale, mutated or has another parent")
        validate_producer_rows_v84(trial.producer, self.execution["fingerprint"])

    def commit(self, parent, trial):
        self._validate(parent, trial)
        return self.lifecycle.commit(parent, trial.lifecycle_trial)

    def rollback(self, parent, trial):
        self._validate(parent, trial)
        return self.lifecycle.rollback(parent, trial.lifecycle_trial)

    def cancel(self, parent, trial):
        self._validate(parent, trial)
        return self.lifecycle.cancel(parent, trial.lifecycle_trial)

    def checkpoint(self, state):
        return self.lifecycle.checkpoint(state)

    def restore_checkpoint(self, document):
        return self.lifecycle.restore_checkpoint(document)


__all__ = [
    "CUDA_SCOPE_ID", "ResidentQ3BoundRouteLifecycleV84",
    "ResidentQ3RouteHandoffV84",
]
