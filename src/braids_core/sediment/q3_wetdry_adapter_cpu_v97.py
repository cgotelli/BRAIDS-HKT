"""Endpoint-wet Q3/Exner adapter for the connected CPU v97 owner.

The v94 provider deliberately authenticates a fully-wet *topological*
binding.  That binding remains the immutable mesh identity here, while the
accepted v97 hydraulic endpoint supplies a transaction-local wet mask.  The
pure v83 producer is still evaluated once on the accepted endpoint.  Its
suspended capacity must already be exact zero in dry cells; its bed vector is
masked locally and every face touching a dry cell is closed before the one
availability/face/Exner composition.

This is additive v97 behaviour.  The historical v94 provider and its saved
identity are unchanged.
"""
from __future__ import annotations

import numpy as np

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    ActiveSuspendedControlV94Error,
    CPUQ3V83ActiveExnerAdapterV94,
    CPUQ3V83ActiveFaceTrialV94,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from .factcr_native_cpu_v97 import (
    NativeFactCrEvidenceV97,
    NativeFactCrRuntimeV97,
)
from .q3_face_inventory_native_cpu_v97 import (
    NativeQ3FaceInventoryResultV97,
    NativeQ3FaceInventoryRuntimeV97,
    _digest as _native_q3_digest,
)
from .recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
    provider_surface_slope_fields_v114,
)


SCOPE_ID = "braids-q4-q3-endpoint-wetdry-cpu-v97-v1"
PROVIDER_KIND = _v94.CPU_PROVIDER_KIND


class Q3WetDryAdapterV97Error(ActiveSuspendedControlV94Error):
    """The endpoint wet mask or masked Q3 transaction is invalid."""


def _endpoint_masks(provider, v92_trial):
    q = np.asarray(v92_trial.candidate.q)
    dry_tolerance = _v94._base_topology(
        provider.outer_execution).dry_tolerance_m
    wet = np.ascontiguousarray(q[0] > dry_tolerance)
    topology = provider.binding.topology
    internal = topology.neighbour >= 0
    safe_other = np.maximum(topology.neighbour, 0)
    face_active = np.ascontiguousarray(
        wet[topology.owner] &
        (~internal | wet[safe_other]))
    wet.setflags(write=False)
    face_active.setflags(write=False)
    return wet, face_active


class CPUQ3V83ActiveWetDryExnerAdapterV97(
        CPUQ3V83ActiveExnerAdapterV94):
    """Exact reviewed CPU v97 provider with endpoint-wet face gating."""

    provider_kind = PROVIDER_KIND
    wetdry_scope_id = SCOPE_ID

    def __init__(self, *, native_factcr_runtime: NativeFactCrRuntimeV97,
                 native_q3_face_runtime: NativeQ3FaceInventoryRuntimeV97,
                 native_threads: int = 4, **kwargs):
        if (type(native_factcr_runtime) is not NativeFactCrRuntimeV97 or
                type(native_q3_face_runtime) is not
                    NativeQ3FaceInventoryRuntimeV97 or
                type(native_threads) is not int or
                not 1 <= native_threads <= 256):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native FactCr/face runtime or threads differ")
        super().__init__(**kwargs)
        self.native_factcr_runtime = native_factcr_runtime
        self.native_q3_face_runtime = native_q3_face_runtime
        self.native_factcr_threads = native_threads
        self._endpoint_wet_mask_v97: np.ndarray | None = None
        self._endpoint_face_active_v97: np.ndarray | None = None
        self._native_factcr_evidence_v97: NativeFactCrEvidenceV97 | None = None
        self._native_q3_face_evidence_v97: \
            NativeQ3FaceInventoryResultV97 | None = None

    def _clear_wetdry_transaction(self) -> None:
        self._endpoint_wet_mask_v97 = None
        self._endpoint_face_active_v97 = None
        self._native_factcr_evidence_v97 = None
        self._native_q3_face_evidence_v97 = None

    def arm_capacity(self, **kwargs) -> None:
        if (self._endpoint_wet_mask_v97 is not None or
                self._endpoint_face_active_v97 is not None):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 wet/dry provider retained a stale endpoint mask")
        super().arm_capacity(**kwargs)

    @property
    def native_factcr_runtime_sha256(self) -> str:
        return str(self.native_factcr_runtime.receipt["receipt_sha256"])

    @property
    def native_q3_face_runtime_sha256(self) -> str:
        return str(self.native_q3_face_runtime.receipt["receipt_sha256"])

    @property
    def native_factcr_evidence_v97(self) -> NativeFactCrEvidenceV97:
        evidence = self._native_factcr_evidence_v97
        if evidence is None:
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native FactCr evidence is unavailable")
        return evidence

    @property
    def native_q3_face_evidence_v97(self) -> NativeQ3FaceInventoryResultV97:
        evidence = self._native_q3_face_evidence_v97
        if evidence is None:
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native face/inventory evidence is unavailable")
        return evidence

    def resolve_armed_capacity(
        self, *, execution, parent, accepted_q,
        settling_velocity_m_s, settling_diameter_receipt,
        surface_fraction, shear_velocity_m_s,
    ):
        if self._armed is None or self._producer_receipt is not None:
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native capacity provider is not singly armed")
        armed = self._armed
        control_parent = armed["control_parent"]
        if (execution is not self.outer_execution.base_execution or
                parent is not control_parent.base_state.base_state):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native capacity parent or execution differs")
        if armed["cancelled"]():
            raise InterruptedError(
                "v97 Q3 cancelled before native v83 production")
        active_iopsus = bool(np.any(
            self.control_execution.iopsus_by_fraction == 1))
        expected_source = (
            _v94.SETTLING_DIAMETER_ACTIVE_IOPSUS_V94 if active_iopsus else
            _v94.SETTLING_DIAMETER_STATIC_PROFILE)
        expected_source_state = (
            control_parent.state_sha256 if active_iopsus else
            execution.fingerprint)
        if (not isinstance(settling_diameter_receipt,
                           _v94.SettlingDiameterReceiptV92) or
                settling_diameter_receipt.source_kind != expected_source or
                settling_diameter_receipt.source_state_sha256 !=
                    expected_source_state):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 parent DSS receipt differs")
        _v94.validate_settling_diameter_receipt_v92(
            settling_diameter_receipt, execution, parent)
        fraction = _v94._readonly(
            surface_fraction,
            (self.control_execution.fraction_count,
             self.control_execution.cell_count),
            "surface fraction", nonnegative=True)
        if not np.array_equal(
                fraction, provider_surface_fraction_v114(self)):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 v92 and bed surface owners differ")
        _v94._readonly(
            shear_velocity_m_s, (self.control_execution.cell_count,),
            "accepted shear velocity", nonnegative=True)
        q = _v94._readonly(
            accepted_q, (3, self.control_execution.cell_count),
            "accepted hydraulic endpoint")
        v92_parent = control_parent.base_state.base_state
        profile = _v94._v78_execution(
            self.control_execution.base_execution).base_execution.profile
        inputs = armed["q3_inputs"]
        frozen = self.control_execution.frozen_uniform_threshold
        receipt, native_evidence = self.native_factcr_runtime.produce(
            self.control_execution.factcr_execution,
            depth_after_m=q[0], endpoint_momentum_m2_s=q[1:],
            observation_height_m=inputs["observation_height_m"],
            water_density_kg_m3=v92_parent.water_density_cell_kg_m3,
            gravity_m_s2=np.full(
                self.control_execution.cell_count, profile.gravity_m_s2),
            kinematic_viscosity_m2_s=np.full(
                self.control_execution.cell_count,
                frozen.initializer_kinematic_viscosity_m2_s),
            additional_shear_stress_pa=
                inputs["additional_shear_stress_pa"],
            sediment_d50_m=profile.sediment_d50_m,
            sediment_d90_m=inputs["sediment_d90_m"],
            sediment_density_kg_m3=profile.sediment_density_kg_m3,
            molecular_schmidt=inputs["molecular_schmidt"],
            aks_factor=inputs["aks_factor"],
            maximum_volume_concentration=
                inputs["maximum_volume_concentration"],
            minimum_depth_m=inputs["minimum_depth_m"],
            bed_factor=inputs["bed_factor"],
            suspended_factor=inputs["suspended_factor"],
            settling_velocity_m_s=settling_velocity_m_s,
            roughness_z0_m=inputs["roughness_z0_m"],
            surface_fraction=fraction,
            frozen_uniform_threshold=(
                frozen.dstar, frozen.critical_shields,
                frozen.unfactored_critical_shear_pa,
                frozen.effective_critical_shear_pa),
            native_threads=self.native_factcr_threads,
            cancelled=armed["cancelled"])
        _v94.validate_factcr_producer_receipt_v83(
            receipt, self.control_execution.factcr_execution)
        if (native_evidence.runtime_sha256 !=
                self.native_factcr_runtime_sha256 or
                native_evidence.execution_fingerprint !=
                    self.control_execution.factcr_execution.fingerprint or
                native_evidence.v83_receipt_sha256 !=
                    receipt.receipt_sha256 or
                native_evidence.native_threads !=
                    self.native_factcr_threads or
                native_evidence.native_producer_batch_owner_count != 1 or
                native_evidence.python_numerical_cell_face_loop_count != 0 or
                native_evidence.scalar_or_clock_publication_count != 0):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native FactCr evidence differs")
        produced_arrays = (
            receipt.dstar, receipt.critical_shields,
            receipt.unfactored_critical_shear_pa,
            receipt.effective_critical_shear_pa)
        frozen_arrays = (
            frozen.dstar, frozen.critical_shields,
            frozen.unfactored_critical_shear_pa,
            frozen.effective_critical_shear_pa)
        if not all(np.array_equal(
                produced,
                np.broadcast_to(expected[:, None], produced.shape))
                for produced, expected in zip(
                    produced_arrays, frozen_arrays)):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native producer did not consume frozen thresholds")
        expected_background_viscosity = np.full(
            self.control_execution.cell_count,
            frozen.initializer_kinematic_viscosity_m2_s)
        if receipt.background_kinematic_viscosity_sha256 != \
                _v94._v83_digest(expected_background_viscosity):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 native equilibrium diffusion viscosity differs")
        capacity = _v94.make_suspended_capacity_receipt_v92(
            execution, parent, accepted_q=q,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_kg_m_s=receipt.lifecycle_capacity_kg_m_s,
            source_kind=_v94.SUSPENDED_CAPACITY_ACTIVE_V83_V94,
            source_receipt_sha256=receipt.receipt_sha256)
        _v94.validate_suspended_capacity_receipt_v92(
            capacity, execution, parent, accepted_q=q,
            settling_diameter_receipt=settling_diameter_receipt)
        if armed["cancelled"]():
            raise InterruptedError(
                "v97 Q3 cancelled after native v83 production")
        self._producer_receipt = receipt
        self._capacity_receipt = capacity
        self._native_factcr_evidence_v97 = native_evidence
        q = np.asarray(accepted_q)
        if q.shape != (3, self.control_execution.cell_count):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 endpoint hydraulic shape differs")
        # Build masks from the same accepted q that v83 and the v92 capacity
        # receipt authenticate.  A dry endpoint must carry no momentum.
        dry_tolerance = _v94._base_topology(
            self.outer_execution).dry_tolerance_m
        wet = np.ascontiguousarray(q[0] > dry_tolerance)
        if np.any(q[1:, ~wet] != 0.0):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 dry endpoint carries hydraulic momentum")
        if np.any(capacity.suspended_capacity_kg_m_s[:, ~wet] != 0.0):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 suspended capacity is nonzero on a dry endpoint")
        topology = self.binding.topology
        internal = topology.neighbour >= 0
        safe_other = np.maximum(topology.neighbour, 0)
        face_active = np.ascontiguousarray(
            wet[topology.owner] &
            (~internal | wet[safe_other]))
        wet.setflags(write=False)
        face_active.setflags(write=False)
        self._endpoint_wet_mask_v97 = wet
        self._endpoint_face_active_v97 = face_active
        return capacity

    def _face_trial(
        self, *, producer, v92_trial, stage, candidate_dss,
    ) -> CPUQ3V83ActiveFaceTrialV94:
        """Run the v94 face owner once on the accepted endpoint wet set."""
        wet = self._endpoint_wet_mask_v97
        face_active = self._endpoint_face_active_v97
        if wet is None or face_active is None:
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 face work lacks the authenticated endpoint mask")
        expected_wet, expected_face = _endpoint_masks(self, v92_trial)
        if (not np.array_equal(wet, expected_wet) or
                not np.array_equal(face_active, expected_face)):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 endpoint mask differs from the accepted hydraulics")

        ng = self.control_execution.fraction_count
        nc, nf = self.binding.topology.ncell, self.binding.topology.nface
        cell_vector = np.array(
            producer.bed_capacity_kg_m_s, dtype=np.float64,
            order="C", copy=True)
        cell_vector[:, :, ~wet] = 0.0
        mapping = np.asarray(self.binding.family_boundary_faces)
        hydraulic_boundary = np.ascontiguousarray(
            v92_trial.hydraulic_face_receipt.
            accepted_water_flux_m2_s[mapping])
        role, _ = self._dynamic_roles(
            hydraulic_boundary=hydraulic_boundary,
            cell_vector=cell_vector)
        role = np.array(role, dtype=np.int32, order="C", copy=True)
        # The additive native v97 composer admits FACE_CLOSED with the real
        # internal neighbour intact.  This is an authenticated sediment-face
        # role, not a falsified hydraulic topology, and guarantees exact-zero
        # reconstruction/flux on every face touching an endpoint-dry cell.
        role[~face_active] = FACE_CLOSED
        topology = self.binding.topology

        taurat = np.ascontiguousarray(producer.producer_diagnostics[2])
        stress = np.zeros((ng, nf), dtype=np.float64)
        internal = role == FACE_INTERNAL
        stress[:, internal] = 0.5*(
            taurat[:, topology.owner[internal]]+
            taurat[:, topology.neighbour[internal]])
        outlet = role == FACE_OWNER_OUTLET
        stress[:, outlet] = taurat[:, topology.owner[outlet]]
        neutral = np.ones((ng, nf), dtype=np.float64)
        hiding, surface_mean = provider_surface_slope_fields_v114(
            self, topology, np)
        d50 = np.ascontiguousarray(np.asarray(
            _v94._base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=np.float64))
        dt = _v94._finite(
            v92_trial.hydraulic_receipt.dt_s, "accepted dt", positive=True)
        base_topology = _v94._base_topology(self.outer_execution)
        native = self.native_q3_face_runtime.evaluate(
            base_topology, self.binding.bed_slope_execution,
            cell_vector, self._state.authoritative_bed_mass_kg_m2,
            np.ascontiguousarray(role, dtype=np.int32),
            downhill=np.ascontiguousarray(self.binding.downhill),
            stress=stress, depth=neutral, d50=d50,
            mean_diameter=surface_mean, hiding=hiding, dt_s=dt,
            native_threads=self.native_factcr_threads,
            cancelled=self._armed["cancelled"])
        if (type(native) is not NativeQ3FaceInventoryResultV97 or
                native.runtime_sha256 != self.native_q3_face_runtime_sha256 or
                native.method_fingerprint !=
                    self.binding.bed_slope_execution["fingerprint"] or
                native.native_threads != self.native_factcr_threads or
                native.native_q3_face_batch_owner_count != 1 or
                native.python_numerical_cell_face_loop_count != 0 or
                native.accepted_bed_or_clock_publication_count != 0):
            raise Q3WetDryAdapterV97Error(
                "v97 native Q3 face/inventory evidence differs")
        self._native_q3_face_evidence_v97 = native
        flux = np.ascontiguousarray(native.face_flux_kg_m_s)
        if np.any(flux[:, ~face_active] != 0.0):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 inactive wet/dry face acquired bed flux")
        delta = np.ascontiguousarray(native.cell_delta_kg_m2)
        fill = np.ascontiguousarray(native.availability)
        named = np.ascontiguousarray(native.named_boundary_transfer_kg)
        absolute = np.ascontiguousarray(native.named_boundary_absolute_kg)
        depletion = np.ascontiguousarray(
            native.depletion_roundoff_positive_in_kg)
        residual = np.ascontiguousarray(native.classwise_residual_kg)
        if (np.any(self._state.authoritative_bed_mass_kg_m2+delta < 0.0) or
                np.any(delta[:, ~wet] != 0.0)):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 dry-cell or depletion update differs")

        ones = np.ones(ng, dtype=np.int32)
        zeros = np.zeros(ng, dtype=np.int32)
        iopsus_count = np.broadcast_to(
            self.control_execution.iopsus_by_fraction[:, None],
            (ng, nc)).copy()
        vector_count = np.full(
            (ng, nc), int(self.control_execution.suscor), dtype=np.int32)
        correction_count = np.zeros((ng, nc), dtype=np.int32)
        values = (
            flux, delta, fill, fill, named, absolute, depletion, residual,
            ones, ones, ones, zeros, ones, ones, ones, zeros,
            taurat, stage, candidate_dss, iopsus_count,
            producer.factcr_application_count, correction_count,
            vector_count,
        )
        arrays = []
        for value in values:
            array = np.ascontiguousarray(value)
            array.setflags(write=False)
            arrays.append(array)
        placeholder = CPUQ3V83ActiveFaceTrialV94(
            self._state.state_sha256, producer,
            _v94._v84_digest(
                v92_trial.candidate.q[0], v92_trial.candidate.q[1:]),
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            dt, _v94.MORPHOLOGICAL_FACTOR, *arrays, "")
        return CPUQ3V83ActiveFaceTrialV94(
            **{**placeholder.__dict__,
               "trial_sha256": _v94._face_token(placeholder)})

    def _validate_face_trial(self, value) -> None:
        super()._validate_face_trial(value)
        wet = self._endpoint_wet_mask_v97
        active = self._endpoint_face_active_v97
        native = self._native_q3_face_evidence_v97
        if wet is None or active is None:
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 face validation lost the endpoint mask")
        if (type(native) is not NativeQ3FaceInventoryResultV97 or
                native.runtime_sha256 != self.native_q3_face_runtime_sha256 or
                native.output_sha256 != _native_q3_digest(
                    native.face_flux_kg_m_s, native.cell_delta_kg_m2,
                    native.availability,
                    native.named_boundary_transfer_kg,
                    native.named_boundary_absolute_kg,
                    native.depletion_roundoff_positive_in_kg,
                    native.classwise_residual_kg) or
                not np.array_equal(value.face_flux_kg_m_s,
                                   native.face_flux_kg_m_s) or
                not np.array_equal(value.cell_delta_kg_m2,
                                   native.cell_delta_kg_m2) or
                not np.array_equal(value.inventory_fill,
                                   native.availability) or
                not np.array_equal(value.named_boundary_transfer_kg,
                                   native.named_boundary_transfer_kg) or
                not np.array_equal(value.named_boundary_absolute_kg,
                                   native.named_boundary_absolute_kg) or
                not np.array_equal(
                    value.depletion_roundoff_positive_in_kg,
                    native.depletion_roundoff_positive_in_kg) or
                not np.array_equal(value.conservation_residual_kg,
                                   native.classwise_residual_kg) or
                np.any(value.producer_receipt.
                   lifecycle_capacity_kg_m_s[:, ~wet] != 0.0) or
                np.any(value.face_flux_kg_m_s[:, ~active] != 0.0) or
                np.any(value.cell_delta_kg_m2[:, ~wet] != 0.0)):
            raise Q3WetDryAdapterV97Error(
                "v97 Q3 dry capacity, face flux, or cell delta is nonzero")

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

    def rollback(self, value):
        try:
            return super().rollback(value)
        finally:
            self._clear_wetdry_transaction()

    def restore_checkpoint(self, value):
        self._clear_wetdry_transaction()
        return super().restore_checkpoint(value)


__all__ = [
    "CPUQ3V83ActiveWetDryExnerAdapterV97", "PROVIDER_KIND",
    "Q3WetDryAdapterV97Error", "SCOPE_ID",
]
