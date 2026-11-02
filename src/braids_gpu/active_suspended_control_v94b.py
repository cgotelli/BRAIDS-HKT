"""Actual-CUDA admission for the Q4/v94b exact-zero suspended vector.

The numerical transaction remains the reviewed resident v94 transaction.
Only the Delft FLOW ``SusCor=false``/``NeglectEntrainment=true`` selector is
added: v92 continues to evolve the water-column scalar and exchange, while
v93 materializes an exact-zero suspended bed contribution without launching
the unused topology-divergence operator.  Q3 remains the single nonzero bed
contribution on that route.
"""
from __future__ import annotations

import copy
from typing import Mapping

from braids_core.sediment.active_suspended_control_v94b import (
    ActiveSuspendedControlV94bError,
    checkpoint_active_suspended_control_state_v94b,
    execution_document_v94b,
    load_active_suspended_control_checkpoint_v94b,
    method_identity_v94b,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    SUSPENDED_VECTOR_EXACT_ZERO_V93,
)
from braids_gpu.active_suspended_control_v94 import (
    ResidentActiveSuspendedControlLifecycleV94,
    ResidentQ3V83ActiveExnerAdapterV94,
)
from braids_gpu.transported_fluid_lifecycle_v92 import (
    ResidentTransportedFluidSettlingLifecycleV92,
)


CUDA_COMPOSITION_ID_V94B = (
    "actual_cuda_v94_active_iopsus_suscor_false_exact_zero_vector_v94b_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveSuspendedControlV94bError(message)


class ResidentActiveSuspendedControlLifecycleV94b(
        ResidentActiveSuspendedControlLifecycleV94):
    """Bounded resident v94 transaction with the additive zero-vector mode."""

    is_v94b_exact_zero_suspended_vector_consumer = True

    def __init__(
        self, execution,
        inner_runtime: ResidentTransportedFluidSettlingLifecycleV92,
        q3_provider: ResidentQ3V83ActiveExnerAdapterV94,
    ):
        execution_document_v94b(execution)
        _require(
            execution.base_execution.suspended_vector_mode ==
                SUSPENDED_VECTOR_EXACT_ZERO_V93,
            "v94b CUDA lifecycle requires the exact-zero suspended-vector "
            "execution")
        super().__init__(execution, inner_runtime, q3_provider)
        self.runtime_receipt = {
            **self.runtime_receipt,
            "cuda_composition_id": CUDA_COMPOSITION_ID_V94B,
            "actual_cuda_v94b_zero_vector": True,
            "raw_v92_scalar_exchange_retained": True,
            "v86_topology_divergence_skipped": True,
            "resident_exact_zero_selected_suspended_exner": True,
            "q3_bed_delta_applied_once": True,
            "automatic_cpu_fallback": False,
        }

    def checkpoint(self, state) -> dict[str, object]:
        _require(self._control_parent is None,
                 "v94b CUDA checkpoint requires a committed point")
        document = checkpoint_active_suspended_control_state_v94b(
            self.control_execution, state, self.q3_provider)
        inner = self.inner.checkpoint(state.base_state.base_state)
        _require(
            inner == document["base_v93_checkpoint"]["base_v92_checkpoint"],
            "v94b CUDA resident/core v92 checkpoints differ")
        return document

    def restore_checkpoint(self, document: Mapping[str, object]):
        """Restore v94b atomically, including all four resident work arrays."""
        _require(self._control_parent is None,
                 "v94b CUDA restart cannot replace an open trial")
        cp = self.cp
        old_last = self.inner.combined.accepted_last.copy()
        old_clock = self.inner.combined.accepted_clock.copy()
        old_trial_last = self.inner.combined.trial_last.copy()
        old_trial_clock = self.inner.combined.trial_clock.copy()
        old_q3 = self.q3_provider.checkpoint()
        old_inner_receipt = copy.deepcopy(self.inner.last_accepted_receipt)
        old_base_receipt = copy.deepcopy(
            self.inner.base.last_accepted_receipt)
        old_outer_receipt = copy.deepcopy(self.last_accepted_receipt)
        try:
            state = load_active_suspended_control_checkpoint_v94b(
                self.control_execution, document, self.q3_provider)
            resident = self.inner.restore_checkpoint(
                document["base_v93_checkpoint"]["base_v92_checkpoint"])
            _require(
                resident.state_sha256 ==
                    state.base_state.base_state.state_sha256,
                "v94b CUDA restored v92 state differs")
            self.inner.last_accepted_receipt = None
            self.inner.base.last_accepted_receipt = None
            self.last_accepted_receipt = None
            return state
        except Exception:
            cp.copyto(self.inner.combined.accepted_last, old_last)
            cp.copyto(self.inner.combined.accepted_clock, old_clock)
            cp.copyto(self.inner.combined.trial_last, old_trial_last)
            cp.copyto(self.inner.combined.trial_clock, old_trial_clock)
            self.q3_provider.restore_checkpoint(old_q3)
            self.inner.last_accepted_receipt = old_inner_receipt
            self.inner.base.last_accepted_receipt = old_base_receipt
            self.last_accepted_receipt = old_outer_receipt
            raise


def method_identity_v94b_cuda() -> dict[str, object]:
    """Machine-readable bounded CUDA identity; not product admission."""
    return {
        **method_identity_v94b(),
        "cuda_composition_id": CUDA_COMPOSITION_ID_V94B,
        "backend": "actual_cuda",
        "integrator": "implicit_fwave_midpoint",
        "resident_water_scalar_exchange": True,
        "topology_divergence_kernel_count": 0,
        "resident_zero_vector_materialization_count": 1,
        "q3_authoritative_bed_owner_count": 1,
        "host_numerical_fallback": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID_V94B",
    "ResidentActiveSuspendedControlLifecycleV94b",
    "method_identity_v94b_cuda",
]
