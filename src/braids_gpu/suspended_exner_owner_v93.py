"""Actual-CUDA Q4/v93 authoritative suspended/Q3 Exner transaction.

The resident v92 lifecycle remains the only scalar/water-column owner, the
reviewed v84 provider remains the only Q3 constitutive/face owner, and the
outer transaction publishes one authoritative bed.  There is no CPU
numerical fallback: suspended selection uses the canonical v86 CUDA kernels
and the two accepted bed terms are combined on the selected CUDA device.
"""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    MAX_ACCEPTED_UPDATES,
    SUSPENDED_VECTOR_EXACT_ZERO_V93,
    SuspendedExnerExecutionV93,
    SuspendedExnerStateV93,
    SuspendedExnerTrialV93,
    SuspendedExnerV93Error,
    _build_trial_from_live_provider,
    _moving_wetdry,
    _validate_provider_identity,
    _validate_trial,
    _wrap_suspended_selection_v93,
    load_suspended_exner_checkpoint_v93,
    suspended_exner_checkpoint_v93,
    validate_suspended_exner_state_v93,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    TransportedFluidTrialV92,
)
from braids_gpu.q3_exner_adapter_v93 import (
    ResidentQ3V84MixedExnerAdapterV93,
)
from braids_gpu.suspended_controls_v86 import (
    ResidentSuspendedControlOperatorsV86,
)
from braids_gpu.transported_fluid_lifecycle_v92 import (
    ResidentTransportedFluidSettlingLifecycleV92,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_COMPOSITION_ID = "canonical_v86_selector_plus_v84_q3_single_exner_v93_v1"


def _cancelled(value) -> bool:
    if callable(value):
        return bool(value())
    if type(value) is bool:
        return value
    raise TypeError("cancelled must be Boolean or callable")


class ResidentSuspendedExnerLifecycleV93:
    """Bounded, fail-closed actual-CUDA v93 outer transaction."""

    is_v91_shared_turbulence_consumer = True
    is_v92_transported_fluid_consumer = True
    is_v93_suspended_exner_consumer = True

    def __init__(
        self, execution: SuspendedExnerExecutionV93,
        inner_runtime: ResidentTransportedFluidSettlingLifecycleV92,
        q3_provider: ResidentQ3V84MixedExnerAdapterV93,
    ):
        # v94 is an additive, exact concrete provider admitted by the shared
        # v93 owner.  The local import avoids a module-import cycle while
        # retaining exact-type (not protocol/string) admission.
        from braids_gpu.active_suspended_control_v94 import (
            ResidentQ3V83ActiveExnerAdapterV94,
        )
        try:
            from braids_gpu.active_suspended_wetdry_v97 import (
                ResidentQ3V83ActiveWetDryExnerAdapterV97,
            )
            active_types = (
                ResidentQ3V83ActiveExnerAdapterV94,
                ResidentQ3V83ActiveWetDryExnerAdapterV97)
        except ImportError:  # module can be mid-import during class loading
            active_types = (ResidentQ3V83ActiveExnerAdapterV94,)
        active_v94 = type(q3_provider) in active_types
        if (not isinstance(execution, SuspendedExnerExecutionV93) or
                execution.backend != "implicit_fwave_midpoint_gpu" or
                type(inner_runtime) is not
                    ResidentTransportedFluidSettlingLifecycleV92 or
                inner_runtime.execution.fingerprint !=
                    execution.base_execution.fingerprint or
                (type(q3_provider) is not
                     ResidentQ3V84MixedExnerAdapterV93 and not active_v94)):
            raise SuspendedExnerV93Error(
                "v93 CUDA lifecycle requires exact accepted resident owners")
        _validate_provider_identity(execution, None, q3_provider)
        self.execution = execution
        self.inner = inner_runtime
        self.q3_provider = q3_provider
        self.selector = ResidentSuspendedControlOperatorsV86()
        if (self.inner.device_id != self.q3_provider.device_id or
                self.inner.device_id != self.selector.device_id):
            raise SuspendedExnerV93Error(
                "v93 CUDA child owners are on different devices")
        self.cp = self.inner.cp
        self.device_id = self.inner.device_id
        self._parent = None
        self._parent_hles = None
        self._handoff = None
        self._v92_trial = None
        self._trial = None
        self._parent_v92_checkpoint = None
        self._parent_q3_checkpoint = None
        self._parent_inner_receipt = None
        self._parent_base_receipt = None
        self._selector_dispatch_count = 0
        self._topology_divergence_dispatch_count = 0
        self._zero_vector_materialization_count = 0
        self._resident_bed_delta_v101 = None
        self.last_accepted_receipt = None
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        sources = (
            Path(__file__).resolve(),
            Path(__import__("braids_gpu.suspended_controls_v86",
                            fromlist=["__file__"]).__file__).resolve(),
            Path(__import__("braids_gpu.suspended_controls_v86",
                            fromlist=["__file__"]).__file__).with_suffix(
                                ".cu").resolve(),
            Path(__import__("braids_gpu.q3_exner_adapter_v93",
                            fromlist=["__file__"]).__file__).resolve(),
            Path(__import__("braids_gpu.q3_exner_adapter_v93",
                            fromlist=["__file__"]).__file__).with_suffix(
                                ".cu").resolve(),
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "integrator": "implicit_fwave_midpoint",
            "device_id": self.device_id,
            "device_name": device_name,
            "canonical_v86_selector": True,
            "real_v84_q3_provider": True,
            "one_authoritative_exner_owner": True,
            "additional_scalar_lifecycle_count": 0,
            "morphological_factor": 1.0,
            "accepted_update_limit": MAX_ACCEPTED_UPDATES,
            "fraction_count_scope": 3,
            "fully_wet_fixed_grid": True,
            "downhill_scope": "exact_zero_only",
            "boundary_scope": "closed_or_owner_free_export_only",
            "authoritative_inventory": "bed_mass_not_elevation",
            "custom_v93_cuda_kernel": True,
            "custom_outer_v93_exner_addition_kernel": False,
            "q3_deterministic_gather_kernels": True,
            "q3_deterministic_source_sha256":
                self.q3_provider.deterministic_source_sha256,
            "q3_v84_runtime_receipt_fingerprint": hashlib.sha256(
                repr(sorted((
                    self.q3_provider.runtime.runtime_receipt
                    if not active_v94 else
                    self.q3_provider.runtime_receipt).items())).encode(
                        "utf-8")).hexdigest(),
            "resident_cupy_exner_addition": True,
            "product_admission": False,
            "automatic_cpu_fallback": False,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        }

    @property
    def kernel_launch_counts(self):
        result = {
            "v92": self.inner.kernel_launch_counts,
            "q3_deterministic_gathers":
                self.q3_provider.deterministic_kernel_launch_counts,
            "v93_canonical_selector_dispatches":
                self._selector_dispatch_count,
        }
        if self.execution.suspended_vector_mode == \
                SUSPENDED_VECTOR_EXACT_ZERO_V93:
            result.update({
                "v93_topology_divergence_kernel_dispatches":
                    self._topology_divergence_dispatch_count,
                "v93_zero_vector_materializations":
                    self._zero_vector_materialization_count,
            })
        return result

    def bind_controller(self, controller) -> None:
        self.inner.bind_controller(controller)

    def _clear(self) -> None:
        self._parent = self._parent_hles = self._handoff = None
        self._v92_trial = self._trial = None
        self._parent_v92_checkpoint = self._parent_q3_checkpoint = None
        self._parent_inner_receipt = self._parent_base_receipt = None
        self._resident_bed_delta_v101 = None

    def _restore_parent_children(self) -> None:
        """Return both resident children to the saved committed point."""
        errors = []
        provider = self.q3_provider
        try:
            if provider._closed is not None:
                provider.abort_closed_commit(provider._closed)
            elif provider._open is not None:
                provider.rollback(provider._open)
            if (self._parent_q3_checkpoint is not None and
                    provider.state_sha256 !=
                    self._parent.q3_provider_state_sha256):
                provider.restore_checkpoint(self._parent_q3_checkpoint)
        except Exception as error:  # pragma: no cover - injected failure gate
            errors.append(error)
        try:
            if (self.inner._parent is not None or
                    self.inner.base._pending is not None or
                    self.inner.combined._open):
                self.inner.abort_hydraulic_trial()
            if self._parent_v92_checkpoint is not None:
                self.inner.restore_checkpoint(self._parent_v92_checkpoint)
            self.inner.last_accepted_receipt = copy.deepcopy(
                self._parent_inner_receipt)
            self.inner.base.last_accepted_receipt = copy.deepcopy(
                self._parent_base_receipt)
        except Exception as error:  # pragma: no cover - injected failure gate
            errors.append(error)
        if errors:
            raise RuntimeError(
                "v93 CUDA failed to restore an atomic parent checkpoint") \
                from errors[0]

    def begin_hydraulic_trial(self, parent, parent_hles, handoff, *,
                              dt_s: float, time_s: float,
                              cancelled=False) -> None:
        if self._parent is not None or self._trial is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA previous transaction is unfinished")
        if _cancelled(cancelled):
            raise SuspendedExnerV93Error(
                "v93 CUDA transaction cancelled before child work")
        validate_suspended_exner_state_v93(
            parent, self.execution, self.q3_provider)
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SuspendedExnerV93Error(
                "v93 accepted update limit was reached")
        # Snapshots precede any mutable resident child work.
        v92_checkpoint = self.inner.checkpoint(parent.base_state)
        q3_checkpoint = self.q3_provider.checkpoint()
        # Install the complete restoration point before invoking a mutable
        # child.  This also covers a child that opens or writes then raises.
        self._parent = parent
        self._parent_hles = parent_hles
        self._handoff = handoff
        self._parent_v92_checkpoint = v92_checkpoint
        self._parent_q3_checkpoint = q3_checkpoint
        self._parent_inner_receipt = copy.deepcopy(
            self.inner.last_accepted_receipt)
        self._parent_base_receipt = copy.deepcopy(
            self.inner.base.last_accepted_receipt)
        try:
            self.inner.begin_hydraulic_trial(
                parent.base_state, parent_hles, handoff,
                dt_s=dt_s, time_s=time_s, cancelled=cancelled)
        except Exception:
            try:
                self._restore_parent_children()
            finally:
                self._clear()
            raise

    def begin_preconverged_hydraulic_context(
        self, parent, parent_hles, handoff, *, cancelled=False,
    ) -> None:
        """Open only the v93 restoration context for an external hydraulic.

        V97 owns the moving wet/dry implicit solve and then hands its exact
        resident endpoint to v92.  Calling the historical hydraulic child
        here would run a second f-wave/v91 owner.  This entry therefore
        snapshots the same two committed children as ``begin_hydraulic_trial``
        without launching either one.  The subsequent typed v97 handoff must
        open ``self.inner`` before :meth:`begin_preconverged_scalar_trial`.
        """
        if self._parent is not None or self._trial is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA previous transaction is unfinished")
        if _cancelled(cancelled):
            raise InterruptedError(
                "v93 CUDA preconverged transaction cancelled before child work")
        validate_suspended_exner_state_v93(
            parent, self.execution, self.q3_provider)
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SuspendedExnerV93Error(
                "v93 accepted update limit was reached")
        v92_checkpoint = self.inner.checkpoint(parent.base_state)
        q3_checkpoint = self.q3_provider.checkpoint()
        self._parent = parent
        self._parent_hles = parent_hles
        self._handoff = handoff
        self._parent_v92_checkpoint = v92_checkpoint
        self._parent_q3_checkpoint = q3_checkpoint
        self._parent_inner_receipt = copy.deepcopy(
            self.inner.last_accepted_receipt)
        self._parent_base_receipt = copy.deepcopy(
            self.inner.base.last_accepted_receipt)

    def require_open_for_controller(self, dt_s, time_s, q) -> None:
        self.inner.require_open_for_controller(dt_s, time_s, q)

    def add_momentum_rate(self, q, full_rate) -> None:
        self.inner.add_momentum_rate(q, full_rate)

    def finish_hydraulic_trial(self, midpoint_q, accepted_q,
                               accepted_water_flux, *, dt_s: float,
                               time_s: float):
        return self.inner.finish_hydraulic_trial(
            midpoint_q, accepted_q, accepted_water_flux,
            dt_s=dt_s, time_s=time_s)

    def abort_hydraulic_trial(self) -> None:
        if self._parent is None:
            self.inner.abort_hydraulic_trial()
            return
        try:
            self._restore_parent_children()
        finally:
            self._clear()

    def _select_suspended(self, trial: TransportedFluidTrialV92):
        cp = self.cp
        evidence = trial.suspended_trial.base_trial
        if (np.any(np.asarray(
                evidence.wetdry_transfer_to_bed_kg_m2) != 0.0) and
                not _moving_wetdry(self.execution)):
            raise SuspendedExnerV93Error(
                "v93 moving wet/dry transfer is deferred to v97")
        inactive = np.asarray(self.execution.routes) != ROUTE_COMBINE
        for value in (
                evidence.transferred_to_suspension_kg_m2,
                evidence.advective_face_flux_kg_m_s,
                evidence.diffusive_face_mass_owner_out_kg,
                self._parent.base_state.suspended_state.
                    suspended_mass_kg_m2,
                trial.candidate.suspended_state.suspended_mass_kg_m2):
            if np.any(np.asarray(value)[inactive] != 0.0):
                raise SuspendedExnerV93Error(
                    "v93 OFF/TOTAL route acquired suspended scalar work")
        base = (self.execution.base_execution.base_execution.base_execution.
                base_execution)
        if self.execution.suspended_vector_mode == \
                SUSPENDED_VECTOR_EXACT_ZERO_V93:
            # Locked FLOW initializes ssuu/ssvv to exact zero and does not
            # construct them when SusCor=false.  Keep the real v92 scalar
            # exchange as authenticated evidence, but do not launch the v86
            # topology-divergence path that is physically unused here.
            exchange = cp.ascontiguousarray(-cp.asarray(
                evidence.transferred_to_suspension_kg_m2,
                dtype=cp.float64))
            flux = cp.zeros(
                (self.execution.fraction_count, self.execution.cell_count),
                dtype=cp.float64)
            selected = cp.zeros_like(flux)
            boundary = cp.zeros(
                (self.execution.fraction_count, base.boundary_count),
                dtype=cp.float64)
            self._zero_vector_materialization_count += 1
        else:
            exchange, flux, selected, boundary = \
                self.selector.bed_balance_topology(
                    base_execution=base, routes=self.execution.routes,
                    neglect_entrainment=self.execution.neglect_entrainment,
                    transferred_to_suspension_kg_m2=
                        evidence.transferred_to_suspension_kg_m2,
                    advective_face_flux_kg_m_s=
                        evidence.advective_face_flux_kg_m_s,
                    diffusive_face_mass_owner_out_kg=
                        evidence.diffusive_face_mass_owner_out_kg,
                    dt_s=float(trial.hydraulic_receipt.dt_s))
            self._topology_divergence_dispatch_count += 1
        if self.execution.neglect_entrainment:
            selected_exchange = cp.zeros(
                self.execution.fraction_count, dtype=cp.float64)
            selected_boundary = boundary
        else:
            selected_exchange = cp.sum(
                exchange*cp.asarray(base.cell_area_m2)[None, :], axis=1)
            selected_boundary = cp.zeros_like(boundary)
        cp.cuda.get_current_stream().synchronize()
        self._selector_dispatch_count += 1
        receipt = _wrap_suspended_selection_v93(
            self.execution, self._parent, trial,
            exchange_delta=cp.asnumpy(exchange),
            flux_delta=cp.asnumpy(flux),
            selected_delta=cp.asnumpy(selected),
            selected_exchange_ledger=cp.asnumpy(selected_exchange),
            selected_boundary_ledger=cp.asnumpy(selected_boundary),
            backend="actual_cuda")
        return receipt, cp.ascontiguousarray(selected)

    def _compose_preconverged_v92_trial(
        self, v92_trial: TransportedFluidTrialV92, *,
        q3_inputs: Mapping[str, object], cancelled=False,
    ) -> SuspendedExnerTrialV93:
        """Compose one already-open v92 scalar trial with Q3/Exner once."""
        if self.inner._trial is not v92_trial:
            raise SuspendedExnerV93Error(
                "v93 CUDA preconverged v92 trial is not resident/open")
        selection, selected_device = self._select_suspended(v92_trial)
        if _cancelled(cancelled):
            raise InterruptedError(
                "v93 CUDA scalar trial cancelled after selection launch")
        open_q3 = self.q3_provider.begin(
            outer_parent=self._parent,
            parent_hles=self._parent_hles, handoff=self._handoff,
            v92_trial=v92_trial, cancelled=lambda: _cancelled(cancelled),
            **dict(q3_inputs))
        if _cancelled(cancelled):
            raise InterruptedError(
                "v93 CUDA scalar trial cancelled after Q3 launch")
        q3_delta = open_q3.face_trial.cell_delta_kg_m2
        cleanup = self.cp.ascontiguousarray(self.cp.asarray(
            v92_trial.suspended_trial.base_trial.
                wetdry_transfer_to_bed_kg_m2,
            dtype=self.cp.float64))
        total = self.cp.ascontiguousarray(
            selected_device+q3_delta+cleanup)
        # A downstream authoritative layer owner consumes this exact device
        # increment, without uploading its host audit image or recomputing Q3.
        self._resident_bed_delta_v101 = total.copy()
        parent_bed = self.cp.ascontiguousarray(self.cp.asarray(
            self._parent.bed_mass_kg_m2, dtype=self.cp.float64))
        candidate_bed = self.cp.ascontiguousarray(parent_bed+total)
        self.cp.cuda.get_current_stream().synchronize()
        return _build_trial_from_live_provider(
            self.execution, self._parent, v92_trial, selection,
            self.q3_provider, self._parent_q3_checkpoint, open_q3,
            backend="actual_cuda",
            authoritative_bed_delta_kg_m2=self.cp.asnumpy(total),
            candidate_bed_mass_kg_m2=self.cp.asnumpy(candidate_bed))

    def begin_preconverged_scalar_trial(
        self, v92_trial: TransportedFluidTrialV92, *,
        q3_inputs: Mapping[str, object], cancelled=False,
    ) -> SuspendedExnerTrialV93:
        """Finish v93 from the exact typed v97->v92 resident handoff."""
        if self._parent is None or self._v92_trial is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA preconverged scalar owner requires one context")
        if _cancelled(cancelled):
            self.abort_hydraulic_trial()
            raise InterruptedError(
                "v93 CUDA preconverged scalar trial cancelled before launch")
        try:
            trial = self._compose_preconverged_v92_trial(
                v92_trial, q3_inputs=q3_inputs, cancelled=cancelled)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v93 CUDA cancelled before trial publication")
        except Exception:
            try:
                self._restore_parent_children()
            finally:
                self._clear()
            raise
        self._v92_trial = v92_trial
        self._trial = trial
        return trial

    def begin_scalar_trial(self, *, q3_inputs: Mapping[str, object],
                           cancelled=False, **scalar_inputs):
        if self._parent is None or self._v92_trial is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA scalar owner requires one hydraulic candidate")
        if _cancelled(cancelled):
            self.abort_hydraulic_trial()
            raise SuspendedExnerV93Error(
                "v93 CUDA scalar trial cancelled before launch")
        try:
            v92_trial = self.inner.begin_scalar_trial(
                cancelled=cancelled, **scalar_inputs)
            if _cancelled(cancelled):
                raise SuspendedExnerV93Error(
                    "v93 CUDA scalar trial cancelled after v92 launch")
            trial = self._compose_preconverged_v92_trial(
                v92_trial, q3_inputs=q3_inputs, cancelled=cancelled)
        except Exception:
            try:
                self._restore_parent_children()
            finally:
                self._clear()
            raise
        self._v92_trial = v92_trial
        self._trial = trial
        return trial

    def commit(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise SuspendedExnerV93Error(
                "v93 CUDA trial is not the resident open transaction")
        _validate_trial(
            self.execution, parent, self._parent_hles,
            self._handoff, trial)
        prepared = trial.q3_prepared_commit
        provider = self.q3_provider
        try:
            rebound = provider.close_commit(prepared)
            if rebound.state_sha256 != \
                    trial.candidate.q3_provider_state_sha256:
                raise SuspendedExnerV93Error(
                    "v93 CUDA closed Q3 child differs")
            source = self.inner.commit(parent.base_state,
                                       trial.base_v92_trial)
            if source.state_sha256 != \
                    trial.v92_bed_projection.source_state_sha256:
                raise SuspendedExnerV93Error(
                    "v93 CUDA committed v92 source differs")
            # Final publication contains no validation or numerical work.
            provider.publish_commit(prepared)
        except Exception:
            try:
                self._restore_parent_children()
            finally:
                self._clear()
            raise
        accepted = trial.candidate
        self.last_accepted_receipt = {
            "outer_transaction_status": "accepted",
            "execution_fingerprint": self.execution.fingerprint,
            "trial_sha256": trial.trial_sha256,
            "accepted_state_sha256": accepted.state_sha256,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
        }
        self._clear()
        return accepted

    def rollback(self, parent, trial):
        if parent is not self._parent or trial is not self._trial:
            raise SuspendedExnerV93Error(
                "v93 CUDA trial is not the resident open transaction")
        _validate_trial(
            self.execution, parent, self._parent_hles,
            self._handoff, trial)
        try:
            self._restore_parent_children()
            validate_suspended_exner_state_v93(
                parent, self.execution, self.q3_provider)
            return parent
        finally:
            self._clear()

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state) -> dict:
        if self._parent is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA checkpoint requires a committed point")
        inner = self.inner.checkpoint(state.base_state)
        document = suspended_exner_checkpoint_v93(
            self.execution, state, self.q3_provider)
        if inner != document["base_v92_checkpoint"]:
            raise SuspendedExnerV93Error(
                "v93 CUDA resident/core v92 checkpoints differ")
        return document

    def restore_checkpoint(self, document):
        if self._parent is not None:
            raise SuspendedExnerV93Error(
                "v93 CUDA restart cannot replace an open trial")
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
            state = load_suspended_exner_checkpoint_v93(
                self.execution, document, self.q3_provider)
            resident = self.inner.restore_checkpoint(
                document["base_v92_checkpoint"])
            if resident.state_sha256 != state.base_state.state_sha256:
                raise SuspendedExnerV93Error(
                    "v93 CUDA restored v92 state differs")
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


def method_identity_v93_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "canonical_v86_selector": True,
        "real_v84_q3_provider": True,
        "one_authoritative_exner_owner": True,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
        "morphological_factor": 1.0,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "fraction_count_scope": 3,
        "fully_wet_fixed_grid": True,
        "downhill_scope": "exact_zero_only",
        "boundary_scope": "closed_or_owner_free_export_only",
        "authoritative_inventory": "bed_mass_not_elevation",
        "product_admission": False,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentSuspendedExnerLifecycleV93",
    "method_identity_v93_cuda",
]
