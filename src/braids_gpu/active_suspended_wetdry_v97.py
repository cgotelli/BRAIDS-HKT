"""Actual-CUDA v97 wet/dry endpoint through v94b/Q3/Exner once.

The moving wet/dry implicit owner is external to the historical v94 hydraulic
entry.  This lifecycle opens the same atomic v93/v94 restoration context,
consumes the typed v97 endpoint in the resident v92 scalar owner, and then
uses the existing active-IopSus provider and authoritative Exner owner.  It
does not launch a second hydraulic or shared-turbulence calculation.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy

import numpy as np

from braids_core.sediment.active_suspended_control_v94 import (
    MAX_ACCEPTED_UPDATES,
    ActiveSuspendedControlOwnerCountsV94,
    ActiveSuspendedControlTrialV94,
    ActiveSuspendedControlV94Error,
    _array_digest,
    _fingerprint,
    _trial_token,
    _wrap_state,
)
from braids_core.sediment.active_suspended_control_v94b import (
    checkpoint_active_suspended_control_state_v94b,
    execution_document_v94b,
    load_active_suspended_control_checkpoint_v94b,
    validate_active_suspended_control_state_v94b,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
    make_settling_diameter_receipt_v92,
)
from braids_core.sediment.q3_exner_adapter_v93 import _base_topology
from braids_core.sediment.recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
)
from braids_gpu.active_suspended_control_v94 import (
    ResidentActiveSuspendedControlLifecycleV94,
    ResidentQ3V83ActiveExnerAdapterV94,
    ResidentQ3V83ActiveFaceTrialV94,
)
from braids_gpu.suspended_exner_owner_v93 import (
    ResidentSuspendedExnerLifecycleV93,
    _cancelled,
)
from braids_gpu.transported_fluid_lifecycle_v92 import (
    ResidentTransportedFluidSettlingLifecycleV92,
)
from braids_gpu.transported_fluid_wetdry_v97 import (
    WetDryHydraulicEvidenceGPUV97,
    begin_transported_fluid_wetdry_gpu_trial_v97,
)


SCOPE_ID = "braids-q4-active-suspended-wetdry-actual-cuda-v97-v1"
Q3_WETDRY_SCOPE_ID = "braids-q4-q3-endpoint-wetdry-actual-cuda-v97-v1"
BACKEND = "actual_cuda"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveSuspendedControlV94Error(message)


class ResidentQ3V83ActiveWetDryExnerAdapterV97(
        ResidentQ3V83ActiveExnerAdapterV94):
    """Reviewed resident Q3 provider gated by the accepted endpoint wet set."""

    wetdry_scope_id = Q3_WETDRY_SCOPE_ID

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._endpoint_wet_mask_v97 = None
        self._endpoint_face_active_v97 = None
        self.runtime_receipt = {
            **self.runtime_receipt,
            "wetdry_scope_id": Q3_WETDRY_SCOPE_ID,
            "endpoint_dry_capacity_exact_zero": True,
            "endpoint_dry_external_shear_exact_zero": True,
            "dry_touching_q3_face_policy": "exact_closed_zero_flux",
            "automatic_cpu_fallback": False,
        }

    def _clear_wetdry_transaction(self) -> None:
        self._endpoint_wet_mask_v97 = None
        self._endpoint_face_active_v97 = None

    def arm_capacity(self, **kwargs) -> None:
        _require(self._endpoint_wet_mask_v97 is None and
                 self._endpoint_face_active_v97 is None,
                 "v97 CUDA Q3 provider retained a stale endpoint mask")
        super().arm_capacity(**kwargs)

    def resolve_armed_capacity(self, **kwargs):
        cp = self.cp
        q = self._device(
            kwargs["accepted_q"],
            (3, self.control_execution.cell_count),
            "accepted endpoint")
        dry = _base_topology(self.outer_execution).dry_tolerance_m
        wet = cp.ascontiguousarray(q[0] > dry)
        _require(not bool(cp.any(q[1:, ~wet] != 0.0).item()),
                 "v97 CUDA Q3 dry endpoint carries momentum")
        # Additional shear is a physical source on a wet water column, not
        # permission to create transport capacity in an accepted-dry cell.
        # Gate it before the sole v83 production so the producer receipt,
        # resident capacity, candidate DSS and later Q3 face work all share
        # the same endpoint mask; no receipt is reminted after computation.
        armed_inputs = self._armed["q3_inputs"]
        additional = self._device(
            armed_inputs["additional_shear_stress_pa"],
            (self.control_execution.cell_count,), "additional shear")
        masked_inputs = dict(armed_inputs)
        masked_inputs["additional_shear_stress_pa"] = cp.ascontiguousarray(
            cp.where(wet, additional, 0.0))
        self._armed["q3_inputs"] = masked_inputs
        try:
            result = super().resolve_armed_capacity(**kwargs)
        finally:
            self._armed["q3_inputs"] = armed_inputs
        _require(not bool(cp.any(
            result.resident_capacity[:, ~wet] != 0.0).item()) and
            not np.any(result.core_receipt.
                       suspended_capacity_kg_m_s[:, cp.asnumpy(~wet)] != 0.0),
            "v97 CUDA Q3 suspended capacity is nonzero on a dry endpoint")
        owner = self.binding.owner
        neighbour = self.binding.neighbour
        internal = neighbour >= 0
        safe_other = cp.maximum(neighbour, 0)
        face_active = cp.ascontiguousarray(
            wet[owner] & ((~internal) | wet[safe_other]))
        self._endpoint_wet_mask_v97 = wet
        self._endpoint_face_active_v97 = face_active
        return result

    def _validate_face_trial(self, value) -> None:
        super()._validate_face_trial(value)
        wet = self._endpoint_wet_mask_v97
        active = self._endpoint_face_active_v97
        _require(wet is not None and active is not None and
                 not bool(self.cp.any(
                     value.face_flux_kg_m_s[:, ~active] != 0.0).item()) and
                 not bool(self.cp.any(
                     value.cell_delta_kg_m2[:, ~wet] != 0.0).item()),
                 "v97 CUDA Q3 dry face flux or cell delta is nonzero")

    def disarm_capacity(self) -> None:
        try:
            super().disarm_capacity()
        finally:
            self._clear_wetdry_transaction()

    def publish_commit(self, value):
        try:
            return super().publish_commit(value)
        finally:
            self._clear_wetdry_transaction()

    def abort_closed_commit(self, value) -> None:
        try:
            super().abort_closed_commit(value)
        finally:
            self._clear_wetdry_transaction()

    def rollback(self, value):
        try:
            return super().rollback(value)
        finally:
            self._clear_wetdry_transaction()

    def restore_checkpoint(self, value):
        self._clear_wetdry_transaction()
        return super().restore_checkpoint(value)


class ResidentActiveSuspendedWetDryLifecycleV97(
        ResidentActiveSuspendedControlLifecycleV94):
    """One resident v97 hydraulic/scalar/Q3/Exner transaction.

    The class deliberately derives from the general v94 owner rather than
    the zero-vector-only convenience subclass.  A v94b execution document
    authenticates both the historical constructed-vector routes and the
    additive exact-zero SusCor=false route, so all four Delft control pairs
    use the same transaction owner.
    """

    is_v97_moving_wetdry_active_suspended_consumer = True

    def __init__(
        self, execution,
        inner_runtime: ResidentTransportedFluidSettlingLifecycleV92,
        q3_provider: ResidentQ3V83ActiveWetDryExnerAdapterV97,
    ):
        execution_document_v94b(execution)
        _require(type(q3_provider) is
                 ResidentQ3V83ActiveWetDryExnerAdapterV97,
                 "v97 CUDA lifecycle requires its endpoint-wet Q3 owner")
        super().__init__(execution, inner_runtime, q3_provider)
        self.runtime_receipt = {
            **self.runtime_receipt,
            "cuda_composition_id": SCOPE_ID,
            "moving_wetdry_hydraulic_owner": "typed_v97_actual_cuda",
            "preconverged_v92_scalar_owner_count": 1,
            "q3_authoritative_exner_owner_count": 1,
            "wetdry_cleanup_exner_owner_count": 1,
            "duplicate_hydraulic_owner_count": 0,
            "automatic_cpu_fallback": False,
        }

    def begin_wetdry_trial(
        self, parent, parent_hles, handoff, *,
        cap_receipt,
        hydraulic_evidence: WetDryHydraulicEvidenceGPUV97,
        reconstructed_left_fluid,
        reconstructed_right_fluid,
        prescribed_boundary_fluid,
        v92_scalar_inputs: Mapping[str, object],
        q3_inputs: Mapping[str, object],
        cancelled=False,
    ) -> ActiveSuspendedControlTrialV94:
        """Consume one authenticated v97 endpoint and open one v94 trial."""
        _require(self._control_parent is None and
                 self._control_trial is None,
                 "v97 CUDA previous outer transaction is unfinished")
        validate_active_suspended_control_state_v94b(
            parent, self.control_execution, self.q3_provider)
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v97 CUDA accepted update limit was reached")
        if _cancelled(cancelled):
            raise InterruptedError(
                "v97 CUDA cancelled before scalar/Q3 child work")
        if not isinstance(v92_scalar_inputs, Mapping) or not isinstance(
                q3_inputs, Mapping):
            raise ActiveSuspendedControlV94Error(
                "v97 CUDA scalar and Q3 inputs must be mappings")
        inputs = dict(v92_scalar_inputs)
        forbidden = {
            "settling_diameter_receipt", "suspended_capacity_resolver",
            "suspended_transport_capacity_kg_m_s", "accepted_water_flux_m2_s",
            "dt_s", "wetdry_hydraulic_evidence",
        }
        overlap = forbidden.intersection(inputs)
        _require(not overlap,
                 "v97 CUDA caller duplicates owned scalar inputs: " +
                 ", ".join(sorted(overlap)))
        supplied_surface = inputs.get("surface_fraction")
        if supplied_surface is not None:
            surface = self.q3_provider._device(
                supplied_surface,
                (self.control_execution.fraction_count,
                 self.control_execution.cell_count),
                "caller surface fraction")
            _require(bool(self.cp.array_equal(
                surface,
                provider_surface_fraction_v114(self.q3_provider)).item()),
                "v97 CUDA caller surface fraction differs from bed owner")
        inputs["surface_fraction"] = provider_surface_fraction_v114(
            self.q3_provider)
        inputs["suspended_transport_capacity_kg_m_s"] = self.cp.zeros(
            (self.control_execution.fraction_count,
             self.control_execution.cell_count), dtype=self.cp.float64)
        if np.any(self.control_execution.iopsus_by_fraction == 1):
            settling = make_settling_diameter_receipt_v92(
                self.control_execution.base_execution.base_execution,
                parent.base_state.base_state,
                suspended_diameter_m=parent.suspended_diameter_m,
                source_kind=SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
                source_state_sha256=parent.state_sha256)
        else:
            settling = make_settling_diameter_receipt_v92(
                self.control_execution.base_execution.base_execution,
                parent.base_state.base_state,
                suspended_diameter_m=parent.suspended_diameter_m)

        try:
            # Snapshot both committed children before either resident scalar
            # or Q3 state can change.  No hydraulic child is launched here.
            ResidentSuspendedExnerLifecycleV93.\
                begin_preconverged_hydraulic_context(
                    self, parent.base_state, parent_hles, handoff,
                    cancelled=cancelled)
            self._control_parent = parent
            self.q3_provider.arm_capacity(
                control_parent=parent, q3_inputs=q3_inputs,
                cancelled=lambda: _cancelled(cancelled))
            v92_trial = begin_transported_fluid_wetdry_gpu_trial_v97(
                self.inner, parent.base_state.base_state,
                parent_hles, handoff, cap_receipt=cap_receipt,
                hydraulic_evidence=hydraulic_evidence,
                reconstructed_left_fluid=reconstructed_left_fluid,
                reconstructed_right_fluid=reconstructed_right_fluid,
                prescribed_boundary_fluid=prescribed_boundary_fluid,
                settling_diameter_receipt=settling,
                suspended_capacity_resolver=
                    self.q3_provider.resolve_armed_capacity,
                cancelled=cancelled, **inputs)
            base_trial = ResidentSuspendedExnerLifecycleV93.\
                begin_preconverged_scalar_trial(
                    self, v92_trial, q3_inputs={}, cancelled=cancelled)
            face = base_trial.q3_open_trial.face_trial
            _require(type(face) is ResidentQ3V83ActiveFaceTrialV94,
                     "v97 CUDA provider returned a different face trial")
            face_dss = np.ascontiguousarray(
                self.cp.asnumpy(face.candidate_suspended_diameter_m))
            live = np.ascontiguousarray(self.cp.asnumpy(face.live_taurat))
            stage = np.ascontiguousarray(
                self.cp.asnumpy(face.transport_stage))
            chain = _fingerprint({
                "previous": parent.accepted_iopsus_chain_sha256,
                "base_v93_trial": base_trial.trial_sha256,
                "parent_DSS_receipt": settling.receipt_sha256,
                "v83_producer": face.producer_receipt.receipt_sha256,
                "candidate_DSS_sha256": _array_digest(face_dss),
                "iopsus_by_fraction":
                    self.control_execution.iopsus_by_fraction.tolist(),
            })
            candidate = _wrap_state(
                self.control_execution, base_trial.candidate,
                suspended_diameter_m=face_dss,
                accepted_iopsus_chain_sha256=chain)
            ng, nc = (self.control_execution.fraction_count,
                      self.control_execution.cell_count)
            ones = np.ones((ng, nc), dtype=np.int32)
            zeros = np.zeros((ng, nc), dtype=np.int32)
            host_factcr = np.ascontiguousarray(
                self.cp.asnumpy(face.factcr_application_count))
            host_iopsus = np.ascontiguousarray(
                self.cp.asnumpy(face.iopsus_application_count))
            host_correction = np.ascontiguousarray(self.cp.asnumpy(
                face.suscor_correction_application_count))
            host_vectors = np.ascontiguousarray(self.cp.asnumpy(
                face.suspended_vector_construction_owner_count))
            for value in (ones, zeros, host_factcr, host_iopsus,
                          host_correction, host_vectors, live, stage,
                          face_dss):
                value.setflags(write=False)
            counts = ActiveSuspendedControlOwnerCountsV94(
                ones, ones, host_factcr, ones, zeros, ones,
                host_iopsus, host_correction, host_vectors)
            placeholder = ActiveSuspendedControlTrialV94(
                self.control_execution.fingerprint, parent.state_sha256,
                base_trial, settling,
                base_trial.base_v92_trial.suspended_capacity_receipt,
                face.producer_receipt, live, stage, face_dss, counts,
                candidate, BACKEND, "")
            trial = ActiveSuspendedControlTrialV94(
                **{**placeholder.__dict__,
                   "trial_sha256": _trial_token(placeholder)})
            self._validate_resident_control_trial(parent, trial)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v97 CUDA cancelled before active trial publication")
        except Exception:
            try:
                if self._parent is not None:
                    self.abort_hydraulic_trial()
                if (self.q3_provider._armed is not None and
                      self.q3_provider._open is None and
                      self.q3_provider._prepared is None and
                      self.q3_provider._closed is None):
                    self.q3_provider.disarm_capacity()
            finally:
                self._clear_control()
            raise
        self._control_trial = trial
        return trial

    def checkpoint(self, state) -> dict[str, object]:
        _require(self._control_parent is None,
                 "v97 CUDA checkpoint requires a committed point")
        document = checkpoint_active_suspended_control_state_v94b(
            self.control_execution, state, self.q3_provider)
        resident = self.inner.checkpoint(state.base_state.base_state)
        _require(resident == document["base_v93_checkpoint"][
                     "base_v92_checkpoint"],
                 "v97 CUDA resident/core v92 checkpoints differ")
        return document

    def restore_checkpoint(self, document):
        _require(self._control_parent is None,
                 "v97 CUDA restart cannot replace an open trial")
        cp = self.cp
        old_arrays = tuple(value.copy() for value in (
            self.inner.combined.accepted_last,
            self.inner.combined.accepted_clock,
            self.inner.combined.trial_last,
            self.inner.combined.trial_clock))
        old_q3 = self.q3_provider.checkpoint()
        old_receipts = (
            copy.deepcopy(self.inner.last_accepted_receipt),
            copy.deepcopy(self.inner.base.last_accepted_receipt),
            copy.deepcopy(self.last_accepted_receipt))
        try:
            state = load_active_suspended_control_checkpoint_v94b(
                self.control_execution, document, self.q3_provider)
            resident = self.inner.restore_checkpoint(
                document["base_v93_checkpoint"]["base_v92_checkpoint"])
            _require(resident.state_sha256 ==
                     state.base_state.base_state.state_sha256,
                     "v97 CUDA restored v92 state differs")
            self.inner.last_accepted_receipt = None
            self.inner.base.last_accepted_receipt = None
            self.last_accepted_receipt = None
            return state
        except Exception:
            for target, saved in zip((
                    self.inner.combined.accepted_last,
                    self.inner.combined.accepted_clock,
                    self.inner.combined.trial_last,
                    self.inner.combined.trial_clock), old_arrays):
                cp.copyto(target, saved)
            self.q3_provider.restore_checkpoint(old_q3)
            (self.inner.last_accepted_receipt,
             self.inner.base.last_accepted_receipt,
             self.last_accepted_receipt) = old_receipts
            raise


__all__ = [
    "BACKEND", "SCOPE_ID", "Q3_WETDRY_SCOPE_ID",
    "ResidentActiveSuspendedWetDryLifecycleV97",
    "ResidentQ3V83ActiveWetDryExnerAdapterV97",
]
