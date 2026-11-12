"""Exact CPU Q3 provider for one v98 fine internal-interface substep."""
from __future__ import annotations

import numpy as np

from . import active_suspended_control_v94 as _v94
from .active_suspended_control_v94 import (
    CPUQ3V83ActiveExnerAdapterV94,
    CPUQ3V83ActiveFaceTrialV94,
)
from .coupled_amr_q3_trace_cpu_v98 import (
    CoupledAMRQ3TraceCPUV98,
)
from .direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from .q3_face_interface_native_cpu_v98 import (
    NativeQ3FaceInterfaceResultV98,
    NativeQ3FaceInterfaceRuntimeV98,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
    Q3WetDryAdapterV97Error,
    _endpoint_masks,
)


SCOPE_ID = "braids-q4-q3-amr-interface-cpu-v98-v1"
PROVIDER_KIND = _v94.CPU_PROVIDER_KIND


class Q3AMRAdapterV98Error(Q3WetDryAdapterV97Error):
    """The v98 Q3 trace/provider/runtime identity differs."""


class CPUQ3V83ActiveAMRExnerAdapterV98(
        CPUQ3V83ActiveWetDryExnerAdapterV97):
    """V97 endpoint-wet provider with one authoritative v98 trace face pass."""

    provider_kind = PROVIDER_KIND
    amr_scope_id = SCOPE_ID

    def __init__(self, *,
                 native_q3_interface_runtime:
                     NativeQ3FaceInterfaceRuntimeV98,
                 q3_trace: CoupledAMRQ3TraceCPUV98,
                 substep_index: int, **kwargs):
        if (type(native_q3_interface_runtime) is not
                NativeQ3FaceInterfaceRuntimeV98 or
                type(q3_trace) is not CoupledAMRQ3TraceCPUV98 or
                type(substep_index) is not int or substep_index not in (0, 1)):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 interface runtime, trace or substep differs")
        super().__init__(**kwargs)
        if (q3_trace.fine_execution_fingerprint !=
                self.control_execution.fingerprint):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 trace belongs to another fine execution")
        self.native_q3_interface_runtime = native_q3_interface_runtime
        self.q3_trace = q3_trace
        self.substep_index = substep_index
        self._native_q3_interface_evidence_v98: \
            NativeQ3FaceInterfaceResultV98 | None = None

    @property
    def native_q3_interface_runtime_sha256(self) -> str:
        return self.native_q3_interface_runtime.runtime_sha256

    @property
    def native_q3_interface_evidence_v98(
            self) -> NativeQ3FaceInterfaceResultV98:
        value = self._native_q3_interface_evidence_v98
        if value is None:
            raise Q3AMRAdapterV98Error(
                "v98 Q3 interface evidence is unavailable")
        return value

    def _clear_wetdry_transaction(self) -> None:
        super()._clear_wetdry_transaction()
        self._native_q3_interface_evidence_v98 = None

    def _face_trial(
        self, *, producer, v92_trial, stage, candidate_dss,
    ) -> CPUQ3V83ActiveFaceTrialV94:
        wet = self._endpoint_wet_mask_v97
        face_active = self._endpoint_face_active_v97
        if wet is None or face_active is None:
            raise Q3AMRAdapterV98Error(
                "v98 Q3 interface work lacks the endpoint mask")
        expected_wet, expected_face = _endpoint_masks(self, v92_trial)
        if (not np.array_equal(wet, expected_wet) or
                not np.array_equal(face_active, expected_face)):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 endpoint mask differs from accepted hydraulics")

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
        role[~face_active] = FACE_CLOSED
        selected = np.asarray(self.q3_trace.interface_faces, dtype=np.int32)
        if (not np.array_equal(
                selected, np.asarray(self.q3_trace.interface_faces)) or
                np.any(~face_active[selected])):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 interface is inactive at the hydraulic endpoint")
        # The selected negative-neighbour rows remain topologically closed;
        # only the typed v98 trace runtime is allowed to supply their exterior.
        role[selected] = FACE_CLOSED
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
        hiding = np.ones((ng, nc), dtype=np.float64)
        d50 = np.ascontiguousarray(np.asarray(
            _v94._base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=np.float64))
        dt = _v94._finite(
            v92_trial.hydraulic_receipt.dt_s, "accepted dt", positive=True)
        base_topology = _v94._base_topology(self.outer_execution)
        native = self.native_q3_interface_runtime.evaluate(
            base_topology, self.binding.bed_slope_execution,
            cell_vector, self._state.authoritative_bed_mass_kg_m2, role,
            downhill=np.ascontiguousarray(self.binding.downhill),
            stress=stress, depth=neutral, d50=d50,
            mean_diameter=neutral, hiding=hiding,
            interface_faces=selected,
            interface_vector_kg_m_s=
                self.q3_trace.interface_vector_kg_m_s[
                    self.substep_index],
            interface_hiding=self.q3_trace.interface_hiding,
            interface_availability=self.q3_trace.interface_availability,
            dt_s=dt, native_threads=self.native_factcr_threads,
            cancelled=self._armed["cancelled"])
        if (type(native) is not NativeQ3FaceInterfaceResultV98 or
                native.runtime_sha256 !=
                    self.native_q3_interface_runtime_sha256 or
                native.method_fingerprint !=
                    self.binding.bed_slope_execution["fingerprint"] or
                native.native_threads != self.native_factcr_threads or
                native.native_q3_face_batch_owner_count != 1 or
                native.internal_interface_owner_count != 1 or
                native.physical_boundary_surrogate_count != 0 or
                native.accepted_bed_or_clock_publication_count != 0 or
                not np.array_equal(native.interface_faces, selected)):
            raise Q3AMRAdapterV98Error(
                "v98 native Q3 interface evidence differs")
        self._native_q3_interface_evidence_v98 = native
        flux = np.ascontiguousarray(native.face_flux_kg_m_s)
        delta = np.ascontiguousarray(native.cell_delta_kg_m2)
        fill = np.ascontiguousarray(native.availability)
        named = np.ascontiguousarray(native.named_boundary_transfer_kg)
        absolute = np.ascontiguousarray(native.named_boundary_absolute_kg)
        depletion = np.ascontiguousarray(
            native.depletion_roundoff_positive_in_kg)
        residual = np.ascontiguousarray(native.classwise_residual_kg)
        if (np.any(self._state.authoritative_bed_mass_kg_m2+delta < 0.0) or
                np.any(delta[:, ~wet] != 0.0) or
                np.any(flux[:, ~face_active &
                            ~np.isin(np.arange(nf), selected)] != 0.0)):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 dry/inactive/depletion update differs")

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
        # The frozen v94 validator authenticates every common field.  The v97
        # subclass's native-v97 evidence check is intentionally bypassed: the
        # exact v98 evidence below is authoritative for this provider type.
        CPUQ3V83ActiveExnerAdapterV94._validate_face_trial(self, value)
        wet = self._endpoint_wet_mask_v97
        active = self._endpoint_face_active_v97
        native = self._native_q3_interface_evidence_v98
        selected = np.asarray(self.q3_trace.interface_faces, dtype=np.int32)
        if (wet is None or active is None or
                type(native) is not NativeQ3FaceInterfaceResultV98 or
                native.runtime_sha256 !=
                    self.native_q3_interface_runtime_sha256 or
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
                not np.array_equal(native.interface_faces, selected) or
                np.any(value.producer_receipt.
                    lifecycle_capacity_kg_m_s[:, ~wet] != 0.0) or
                np.any(value.cell_delta_kg_m2[:, ~wet] != 0.0)):
            raise Q3AMRAdapterV98Error(
                "v98 Q3 interface face trial differs")


__all__ = [
    "CPUQ3V83ActiveAMRExnerAdapterV98", "PROVIDER_KIND",
    "Q3AMRAdapterV98Error", "SCOPE_ID",
]
