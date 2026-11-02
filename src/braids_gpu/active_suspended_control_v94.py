"""Actual-CUDA Q4/v94 active-IopSus transaction.

This module is the resident peer of the bounded CPU oracle.  Parent ``Dss``
is consumed by v92's existing CUDA settling/exchange owner; one resident v83
Van Rijn 1993 producer is then shared by the v92 capacity consumer and the
v93 face/Exner owner.  Its candidate ``Dss`` is published only by the outer
atomic commit.  Host readback constructs immutable receipts only and never
provides a numerical fallback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.active_suspended_control_v94 import (
    FORMULA_CODE,
    MAX_ACCEPTED_UPDATES,
    MORPHOLOGICAL_FACTOR,
    ActiveSuspendedControlOwnerCountsV94,
    ActiveSuspendedControlExecutionV94,
    ActiveSuspendedControlStateV94,
    ActiveSuspendedControlTrialV94,
    ActiveSuspendedControlV94Error,
    GPU_PROVIDER_KIND,
    _trial_token,
    _candidate_diameter_v94,
    _validate_trial as _validate_control_trial,
    _v78_execution,
    _wrap_state,
    checkpoint_active_suspended_control_state_v94,
    execution_document_v94,
    load_active_suspended_control_checkpoint_v94,
    validate_active_suspended_control_state_v94,
    validate_frozen_threshold_receipt_v94,
)
from braids_core.sediment.bedload_boundary_live_v70 import (
    SOURCE_EQUILIBRIUM_TRACE,
    SOURCE_NONE,
    SOURCE_OWNER_FREE_EXPORT,
    SOURCE_PRESCRIBED_EXTERNAL,
)
from braids_core.sediment.direct_vector_wetdry_amr_live_v71 import (
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
)
from braids_core.sediment.factcr_lifecycle_v83 import (
    FactCrProducerReceiptV83,
    SPEED_EPSILON,
    _digest as _v83_digest,
    make_factcr_producer_receipt_v83,
    validate_factcr_producer_receipt_v83,
)
from braids_core.sediment.multifraction_suspended_lifecycle_v78 import (
    ROLE_CUDA,
)
from braids_core.sediment.q3_exner_adapter_v93 import (
    Q3ExnerReceiptV93,
    _array_digest,
    _base_topology,
    _fingerprint,
    _host_array,
    _receipt_token,
    _shared_topology_sha256,
    _v92_physical_sha256,
    _validate_outer_state,
    validate_q3_v75_exner_receipt_v93,
)
from braids_core.sediment.q3_mixed_route_exner_adapter_v93 import (
    Q3V84MixedExnerAdapterV93Error,
)
from braids_core.sediment.q3_producer_lifecycle_binding_v84 import (
    _digest as _v84_array_digest,
)
from braids_core.sediment.route_lifecycle_composition_v81 import (
    ROUTE_COMBINE as CORE_ROUTE_COMBINE,
)
from braids_core.sediment.recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
    provider_surface_slope_fields_v114,
)
from braids_core.sediment.suspended_exner_owner_v93 import (
    _build_trial_from_live_provider,
    _validate_trial as _validate_v93_trial,
    _wrap_suspended_selection_v93,
    validate_suspended_exner_state_v93,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
    SETTLING_DIAMETER_STATIC_PROFILE,
    SUSPENDED_CAPACITY_ACTIVE_V83_V94,
    SettlingDiameterReceiptV92,
    SuspendedCapacityReceiptV92,
    TransportedFluidTrialV92,
    _validate_trial as _validate_v92_trial,
    make_settling_diameter_receipt_v92,
    make_suspended_capacity_receipt_v92,
    validate_settling_diameter_receipt_v92,
    validate_suspended_capacity_receipt_v92,
)
from braids_gpu.q3_exner_adapter_v93 import (
    CUDA_OPTIONS as V93_CUDA_OPTIONS,
    DETERMINISTIC_KERNEL_NAMES,
    ResidentQ3V84OpenTrialV93,
    ResidentQ3V84MixedBindingV93,
    ResidentQ3V84MixedExnerAdapterV93,
    ResidentQ3V84PreparedCommitV93,
    ResidentQ3V84MixedStateV93,
    _open_token,
    _resident_digest,
    validate_resident_q3_v84_mixed_state_v93,
)
from braids_gpu.flow_bed_slope_composition import (
    DONOR_NEIGHBOUR,
    DONOR_NONE,
    DONOR_OWNER,
    DONOR_TRACE,
)
from braids_gpu.suspended_exner_owner_v93 import (
    ResidentSuspendedExnerLifecycleV93,
    _cancelled,
)
from braids_gpu.suspended_controls_v86 import (
    ResidentSuspendedControlOperatorsV86,
)
from braids_gpu.transported_fluid_lifecycle_v92 import (
    ResidentSuspendedCapacityResolutionV94,
    ResidentTransportedFluidSettlingLifecycleV92,
)
from braids_gpu.van_rijn_1993_route_v66 import (
    ROUTE_COMBINE,
    ResidentVanRijn1993RouteV66,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_COMPOSITION_ID = "v83_v73_v92_v93_active_iopsus_actual_cuda_v94_v1"
BACKEND = "actual_cuda"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveSuspendedControlV94Error(message)


@dataclass(frozen=True)
class ResidentFactCrProducerRowsV94:
    """One resident v83 result and its immutable host provenance."""

    receipt: FactCrProducerReceiptV83
    dstar: object
    critical_shields: object
    unfactored_critical_shear_pa: object
    effective_critical_shear_pa: object
    raw_bed_capacity_kg_m_s: object
    raw_equilibrium_concentration_kg_m3: object
    bed_capacity_kg_m_s: object
    lifecycle_capacity_kg_m_s: object
    equilibrium_target_kg_m3: object
    producer_diagnostics: object
    accepted_endpoint: object
    active_fraction: object
    incoming_settling_velocity_m_s: object
    device_id: int


@dataclass(frozen=True)
class ResidentQ3V83ActiveFaceTrialV94:
    """Resident face/Exner candidate plus immutable v94 owner evidence."""

    parent_state_sha256: str
    producer_receipt: FactCrProducerReceiptV83
    accepted_endpoint_sha256: str
    hydraulic_face_receipt_sha256: str
    dt_s: float
    morphological_factor: float
    face_flux_kg_m_s: object
    cell_delta_kg_m2: object
    inventory_fill: object
    accepted_cell_availability: object
    named_boundary_transfer_kg: object
    named_boundary_absolute_kg: object
    depletion_roundoff_positive_in_kg: object
    conservation_residual_kg: object
    constitutive_producer_count: object
    route_handoff_count: object
    face_reconstruction_count: object
    bed_slope_count: object
    boundary_owner_count: object
    inventory_fill_count: object
    face_exner_count: object
    extra_scalar_lifecycle_count: object
    live_taurat: object
    transport_stage: object
    candidate_suspended_diameter_m: object
    iopsus_application_count: object
    factcr_application_count: object
    suscor_correction_application_count: object
    suspended_vector_construction_owner_count: object
    trial_sha256: str
    device_id: int


def _resident_face_token(value: ResidentQ3V83ActiveFaceTrialV94, cp) -> str:
    return _fingerprint({
        "scope_id": "braids-q4-flow2dh-iopsus-active-v94-v1:cuda-face",
        "parent_state_sha256": value.parent_state_sha256,
        "producer_receipt_sha256": value.producer_receipt.receipt_sha256,
        "accepted_endpoint_sha256": value.accepted_endpoint_sha256,
        "hydraulic_face_receipt_sha256":
            value.hydraulic_face_receipt_sha256,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "resident_arrays_sha256": _resident_digest(
            cp, value.face_flux_kg_m_s, value.cell_delta_kg_m2,
            value.inventory_fill, value.accepted_cell_availability,
            value.named_boundary_transfer_kg,
            value.named_boundary_absolute_kg,
            value.depletion_roundoff_positive_in_kg,
            value.conservation_residual_kg,
            value.constitutive_producer_count,
            value.route_handoff_count, value.face_reconstruction_count,
            value.bed_slope_count, value.boundary_owner_count,
            value.inventory_fill_count, value.face_exner_count,
            value.extra_scalar_lifecycle_count, value.live_taurat,
            value.transport_stage, value.candidate_suspended_diameter_m,
            value.iopsus_application_count, value.factcr_application_count,
            value.suscor_correction_application_count,
            value.suspended_vector_construction_owner_count),
        "single_v83_producer": True,
        "single_outer_face_owner": True,
        "child_scalar_lifecycle": False,
        "device_id": value.device_id,
    })


class _ResidentFactCrPureProducerV94:
    """Pure threshold + v66 producer; it never starts a v78 lifecycle."""

    def __init__(self, execution):
        _require(execution.role == ROLE_CUDA,
                 "v94 CUDA v83 producer requires the actual-CUDA role")
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - host dependent
            raise ActiveSuspendedControlV94Error(
                "v94 actual CUDA requires CuPy") from error
        try:
            _require(cp.cuda.runtime.getDeviceCount() >= 1,
                     "v94 actual CUDA requires a CUDA device")
        except ActiveSuspendedControlV94Error:
            raise
        except Exception as error:  # pragma: no cover - host dependent
            raise ActiveSuspendedControlV94Error(
                "v94 CUDA device discovery failed") from error
        self.cp = cp
        self.execution = execution
        self.device_id = int(cp.cuda.Device().id)
        try:
            self.producer = ResidentVanRijn1993RouteV66(ROUTE_COMBINE)
        except Exception as error:
            raise ActiveSuspendedControlV94Error(
                "v94 CUDA v66 producer kernels failed to compile") from error
        _require(self.producer.device_id == self.device_id,
                 "v94 CUDA v83/v66 producers selected different devices")
        self.threshold_launch_count = 0
        self.frozen_threshold_materialization_count = 0
        self.producer_launch_count = 0

    def _device(self, value, shape, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == cp.float64 and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v94 CUDA {name} must be resident contiguous float64 "
                 f"{shape}")
        return value

    def produce(
        self, *, accepted_endpoint, observation_height_m,
        water_density_kg_m3, gravity_m_s2, kinematic_viscosity_m2_s,
        additional_shear_stress_pa, sediment_d50_m, sediment_d90_m,
        sediment_density_kg_m3, molecular_schmidt, aks_factor,
        maximum_volume_concentration, minimum_depth_m, bed_factor,
        suspended_factor, settling_velocity_m_s, roughness_z0_m,
        active_fraction, frozen_uniform_threshold,
    ) -> ResidentFactCrProducerRowsV94:
        cp = self.cp
        ng = self.execution.fraction_count
        nc = self.execution.cell_count
        q = self._device(accepted_endpoint, (3, nc), "accepted endpoint")
        zvel = self._device(
            observation_height_m, (nc,), "observation height")
        rho = self._device(water_density_kg_m3, (nc,), "water density")
        gravity = self._device(gravity_m_s2, (nc,), "gravity")
        viscosity = self._device(
            kinematic_viscosity_m2_s, (nc,), "kinematic viscosity")
        tauadd = self._device(
            additional_shear_stress_pa, (nc,), "additional shear")
        d50 = self._device(sediment_d50_m, (ng,), "D50")
        d90 = self._device(sediment_d90_m, (ng,), "D90")
        rho_s = self._device(
            sediment_density_kg_m3, (ng,), "solid density")
        schmidt = self._device(
            molecular_schmidt, (ng,), "molecular Schmidt")
        aks = self._device(aks_factor, (ng,), "Aks factor")
        camax = self._device(
            maximum_volume_concentration, (ng,), "maximum concentration")
        hmin = self._device(minimum_depth_m, (ng,), "minimum depth")
        bedfac = self._device(bed_factor, (ng,), "bed factor")
        susfac = self._device(suspended_factor, (ng,), "suspended factor")
        ws = self._device(
            settling_velocity_m_s, (ng, nc), "incoming settling velocity")
        z0 = self._device(roughness_z0_m, (ng, nc), "roughness z0")
        fraction = self._device(
            active_fraction, (ng, nc), "active surface fraction")
        floating = (
            q, zvel, rho, gravity, viscosity, tauadd, d50, d90, rho_s,
            schmidt, aks, camax, hmin, bedfac, susfac, ws, z0, fraction)
        _require(all(bool(cp.all(cp.isfinite(value)).item())
                     for value in floating),
                 "v94 CUDA v83 producer inputs must be finite")
        _require(bool(cp.all(q[0] >= 0.0).item()) and
                 bool(cp.all(zvel > 0.0).item()) and
                 bool(cp.all(rho > 0.0).item()) and
                 bool(cp.all(gravity > 0.0).item()) and
                 bool(cp.all(viscosity > 0.0).item()) and
                 bool(cp.all(tauadd >= 0.0).item()) and
                 bool(cp.all(d50 > 0.0).item()) and
                 bool(cp.all(d90 >= d50).item()) and
                 bool(cp.all(rho_s[:, None] > rho[None, :]).item()) and
                 bool(cp.all(schmidt > 0.0).item()) and
                 bool(cp.all(aks > 0.0).item()) and
                 bool(cp.all((camax > 0.0) & (camax <= 1.0)).item()) and
                 bool(cp.all(hmin >= 0.0).item()) and
                 bool(cp.all(bedfac >= 0.0).item()) and
                 bool(cp.all(susfac >= 0.0).item()) and
                 bool(cp.all(ws >= 0.0).item()) and
                 bool(cp.all(z0 > 0.0).item()) and
                 bool(cp.all((fraction >= 0.0) &
                             (fraction <= 1.0)).item()) and
                 bool(cp.all(cp.abs(cp.sum(fraction, axis=0)-1.0) <=
                             32.0*np.finfo(np.float64).eps).item()),
                 "v94 CUDA v83 producer input bounds differ")
        profile = self.execution.lifecycle_execution.base_execution.profile
        _require(bool(cp.array_equal(
            d50, cp.asarray(profile.sediment_d50_m,
                            dtype=cp.float64)).item()) and
                 bool(cp.array_equal(
            rho_s, cp.asarray(profile.sediment_density_kg_m3,
                              dtype=cp.float64)).item()) and
                 bool(cp.all(gravity == profile.gravity_m_s2).item()),
                 "v94 CUDA v83 producer physical profile differs")

        # Uniform/fixed D50 follows Delft's initsedtra ordering: Dstar,
        # tetacr and taucr0 are frozen once from initializer rho/nu.  Current
        # v92 rho and viscosity continue into current shear/v66 transport,
        # but must not silently recompute these thresholds each update.
        frozen = frozen_uniform_threshold
        validate_frozen_threshold_receipt_v94(frozen, self.execution)
        _require(frozen.factcr_execution_fingerprint ==
                 self.execution.fingerprint,
                 "v94 CUDA frozen threshold receipt differs")
        shape = (ng, nc)
        dstar = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(frozen.dstar, dtype=cp.float64)[:, None], shape))
        theta = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(frozen.critical_shields,
                       dtype=cp.float64)[:, None], shape))
        base_taucr = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(frozen.unfactored_critical_shear_pa,
                       dtype=cp.float64)[:, None], shape))
        effective_taucr = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(frozen.effective_critical_shear_pa,
                       dtype=cp.float64)[:, None], shape))
        self.frozen_threshold_materialization_count += 1
        wet = q[0] > self.execution.lifecycle_execution.base_execution.dry_tolerance_m
        velocity = cp.ascontiguousarray(cp.where(
            wet[None, :], q[1:]/cp.where(wet, q[0], 1.0)[None, :], 0.0))
        produced = self.producer.evaluate(
            depth=q[0], velocity=velocity, observation_height=zvel,
            water_density=rho, gravity=gravity,
            kinematic_viscosity=viscosity,
            additional_shear_stress=tauadd, d50=d50, d90=d90,
            solid_density=rho_s, molecular_schmidt=schmidt,
            aks_factor=aks, maximum_volume_concentration=camax,
            minimum_depth=hmin, bed_factor=bedfac,
            suspended_factor=susfac, settling_velocity=ws,
            roughness_z0=z0, dstar=dstar,
            critical_shear_stress=effective_taucr,
            surface_fraction=fraction)
        self.producer_launch_count += 1
        sediment_active = wet & (
            q[0] > profile.sediment_threshold_m)
        exchange_depth = cp.where(
            sediment_active,
            cp.maximum(profile.reference_depth_floor_m, q[0]), 1.0)
        speed = cp.hypot(velocity[0], velocity[1])
        capacity = cp.ascontiguousarray(
            susfac[:, None]*produced.raw_conc2d*
            (speed[None, :]+SPEED_EPSILON)*exchange_depth[None, :])
        capacity[:, ~sediment_active] = 0.0
        target = cp.ascontiguousarray(produced.equilibrium_concentration)
        target[:, ~sediment_active] = 0.0
        resident = (
            dstar, theta, base_taucr, effective_taucr,
            cp.ascontiguousarray(produced.raw_bed_mass),
            cp.ascontiguousarray(produced.raw_conc2d),
            cp.ascontiguousarray(produced.bed_mass), capacity, target,
            cp.ascontiguousarray(produced.diagnostics))
        host = tuple(np.ascontiguousarray(cp.asnumpy(value))
                     for value in resident)
        q_host = np.ascontiguousarray(cp.asnumpy(q))
        fraction_host = np.ascontiguousarray(cp.asnumpy(fraction))
        ws_host = np.ascontiguousarray(cp.asnumpy(ws))
        host_background_viscosity = np.ascontiguousarray(
            cp.asnumpy(viscosity))
        receipt = make_factcr_producer_receipt_v83(
            self.execution, accepted_endpoint=q_host,
            active_fraction=fraction_host,
            incoming_settling_velocity_m_s=ws_host,
            background_kinematic_viscosity_m2_s=
                host_background_viscosity,
            dstar=host[0], critical_shields=host[1],
            unfactored_critical_shear_pa=host[2],
            effective_critical_shear_pa=host[3],
            raw_bed_capacity_kg_m_s=host[4],
            raw_equilibrium_concentration_kg_m3=host[5],
            bed_capacity_kg_m_s=host[6],
            lifecycle_capacity_kg_m_s=host[7],
            equilibrium_target_kg_m3=host[8],
            producer_diagnostics=host[9], backend=ROLE_CUDA)
        validate_factcr_producer_receipt_v83(receipt, self.execution)
        return ResidentFactCrProducerRowsV94(
            receipt, *resident, q, fraction, ws, self.device_id)


class ResidentQ3V83ActiveExnerAdapterV94(
        ResidentQ3V84MixedExnerAdapterV93):
    """Identity-held v83 capacity/face provider on one CUDA device."""

    backend = BACKEND
    provider_kind = GPU_PROVIDER_KIND

    def __init__(
        self, *, outer_execution: ActiveSuspendedControlExecutionV94,
        binding: ResidentQ3V84MixedBindingV93,
        state: ResidentQ3V84MixedStateV93,
    ):
        execution_document_v94(outer_execution)
        _require(outer_execution.backend == "implicit_fwave_midpoint_gpu" and
                 type(binding) is ResidentQ3V84MixedBindingV93 and
                 type(state) is ResidentQ3V84MixedStateV93,
                 "v94 CUDA provider requires exact resident v94 objects")
        base = outer_execution.base_execution
        _require(binding.topology_id == base.q3_topology_id and
                 binding.binding_sha256 == base.q3_binding_sha256 and
                 binding.boundary_names == tuple(base.q3_boundary_names),
                 "v94 CUDA provider topology, binding, or boundary differs")
        _require(outer_execution.live_taurat_execution[
                     "bed_slope_execution_fingerprint"] ==
                 binding.core.bed_slope_execution["fingerprint"],
                 "v94 CUDA live-taurat and bound slope executions differ")
        _require(np.array_equal(
            state.core.bulk_density_kg_m3,
            np.asarray(binding.core.boundary_execution[
                "bulk_density_kg_m3"], dtype=np.float64)),
            "v94 CUDA bulk density differs from bound v70 physics")
        self.control_execution = outer_execution
        self.outer_execution = base
        self.binding = binding
        self.boundary_names = binding.boundary_names
        self.cp = binding.composer.cp
        self.device_id = binding.device_id
        self._producer = _ResidentFactCrPureProducerV94(
            outer_execution.factcr_execution)
        self._selector = ResidentSuspendedControlOperatorsV86()
        _require(self._producer.device_id == self.device_id ==
                 self._selector.device_id,
                 "v94 CUDA child operators selected different devices")
        source_path = Path(__import__(
            "braids_gpu.q3_exner_adapter_v93",
            fromlist=["__file__"]).__file__).with_suffix(".cu").resolve()
        source = source_path.read_text(encoding="utf-8")
        try:
            module = self.cp.RawModule(
                code=source, options=V93_CUDA_OPTIONS,
                name_expressions=DETERMINISTIC_KERNEL_NAMES)
            self._deterministic_kernels = {
                name: module.get_function(name)
                for name in DETERMINISTIC_KERNEL_NAMES
            }
        except Exception as error:
            raise ActiveSuspendedControlV94Error(
                "v94 CUDA deterministic Exner kernels failed to compile") \
                from error
        self.deterministic_source_sha256 = hashlib.sha256(
            source.encode("utf-8")).hexdigest()
        self._deterministic_kernel_launch_counts = {
            name: 0 for name in DETERMINISTIC_KERNEL_NAMES
        }
        self._validate_topology_and_profile()
        validate_resident_q3_v84_mixed_state_v93(
            state, execution_fingerprint=self.execution_fingerprint,
            binding=binding)
        self._state = state
        self._open = None
        self._prepared = None
        self._closed = None
        self._armed = None
        self._producer_rows: ResidentFactCrProducerRowsV94 | None = None
        self._capacity_receipt: SuspendedCapacityReceiptV92 | None = None
        self._iopsus_production_launch_count = 0
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        raw_name = properties.get("name", b"unknown")
        device_name = (raw_name.decode("utf-8", errors="replace")
                       if isinstance(raw_name, bytes) else str(raw_name))
        paths = (
            Path(__file__).resolve(),
            Path(__file__).with_name("van_rijn_1993_route_v66.cu").resolve(),
            Path(__file__).with_name("suspended_controls_v86.cu").resolve(),
            Path(__file__).with_name(
                "transported_fluid_settling_v92.cu").resolve(),
            source_path,
        )
        frozen = outer_execution.frozen_uniform_threshold
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": outer_execution.fingerprint,
            "factcr_execution_fingerprint": self.execution_fingerprint,
            "backend": BACKEND,
            "integrator": "implicit_fwave_midpoint",
            "device_id": self.device_id,
            "device_name": device_name,
            "frozen_initializer_v83_threshold_receipt": True,
            "frozen_threshold_receipt_sha256": frozen.receipt_sha256,
            "initializer_water_density_kg_m3":
                frozen.initializer_water_density_kg_m3,
            "initializer_background_vicmol_m2_s":
                frozen.initializer_kinematic_viscosity_m2_s,
            "v66_bakdif_uses_initializer_vicmol": True,
            "v92_settlefrm2_uses_dynamic_temperature_viscosity": True,
            "current_cell_water_density_drives_shear": True,
            "per_update_threshold_recompute": False,
            "actual_cuda_v66_current_only_producer": True,
            "actual_cuda_v73_live_taurat_face_channel": True,
            "actual_cuda_v92_parent_dss_and_capacity_consumer": True,
            "actual_cuda_v93_single_exner_owner": True,
            "accepted_parent_dss_drives_current_settling": True,
            "candidate_dss_visible_after_outer_accept_only": True,
            "one_shared_constitutive_producer": True,
            "additional_scalar_lifecycle_count": 0,
            "automatic_cpu_fallback": False,
            "host_readback_scope":
                "immutable receipt and checkpoint evidence only",
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }

    @property
    def execution_fingerprint(self) -> str:
        return self.control_execution.factcr_execution.fingerprint

    @property
    def kernel_launch_counts(self) -> dict[str, object]:
        return {
            "v83_threshold_recompute": self._producer.threshold_launch_count,
            "v83_frozen_threshold_materializations":
                self._producer.frozen_threshold_materialization_count,
            "v66_producer": self._producer.producer_launch_count,
            "v86_iopsus_production":
                self._iopsus_production_launch_count,
            "v93_deterministic_gathers":
                self.deterministic_kernel_launch_counts,
        }

    def _validate_topology_and_profile(self) -> None:
        base = _base_topology(self.outer_execution)
        core = self.binding.core
        mapping = np.asarray(core.family_boundary_faces)
        local = np.asarray(core.boundary_topology.boundary_index)
        expected = np.flatnonzero(base.boundary_index >= 0)
        _require(all((
            core.topology.ncell == base.cell_count,
            core.topology.nface == base.face_count,
            np.array_equal(core.topology.owner, base.owner),
            np.array_equal(core.topology.neighbour, base.neighbour),
            np.array_equal(core.topology.cell_area_m2, base.cell_area_m2),
            np.array_equal(core.normal, base.normal),
            np.array_equal(core.face_length_m, base.face_length_m),
            np.array_equal(core.cell_face_offset, base.cell_face_offset),
            np.array_equal(core.cell_faces, base.cell_faces),
            np.array_equal(core.signs, base.cell_face_sign),
            base.boundary_count == len(self.boundary_names),
            np.array_equal(mapping, expected),
            np.array_equal(local, base.boundary_index[mapping]),
            bool(np.all(core.topology.wet)),
            bool(np.all(core.downhill == 0.0)),
        )), "v94 CUDA topology/boundary/exact-zero slope differs from v92")
        profile = base.profile
        physical = self.control_execution.live_taurat_execution[
            "parent_execution"]["physical_profile"]
        _require(tuple(physical["fraction_ids"]) ==
                 self.ordered_fraction_ids and
                 tuple(core.bed_slope_execution["ordered_fraction_ids"]) ==
                 self.ordered_fraction_ids and
                 np.array_equal(np.asarray(physical["d50_m"]),
                                np.asarray(profile.sediment_d50_m)) and
                 np.array_equal(
                     np.asarray(physical["solid_density_kg_m3"]),
                     np.asarray(profile.sediment_density_kg_m3)),
                 "v94 CUDA physical profile differs from v92")

    def _device(self, value, shape, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == cp.float64 and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v94 CUDA {name} must be resident contiguous float64 "
                 f"{shape}")
        return value

    def arm_capacity(
        self, *, control_parent: ActiveSuspendedControlStateV94,
        q3_inputs: Mapping[str, object],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        validate_active_suspended_control_state_v94(
            control_parent, self.control_execution, self)
        _require(self._armed is None and self._producer_rows is None and
                 self._capacity_receipt is None and self._open is None and
                 self._prepared is None and self._closed is None,
                 "v94 CUDA provider already has a transaction")
        _require(isinstance(q3_inputs, Mapping) and callable(cancelled),
                 "v94 CUDA producer inputs or cancellation callback differ")
        required = {
            "observation_height_m", "additional_shear_stress_pa",
            "sediment_d90_m", "molecular_schmidt", "aks_factor",
            "maximum_volume_concentration", "minimum_depth_m",
            "bed_factor", "suspended_factor", "roughness_z0_m",
        }
        _require(set(q3_inputs) == required,
                 "v94 CUDA producer input inventory differs")
        self._armed = {
            "control_parent": control_parent,
            "q3_inputs": dict(q3_inputs),
            "cancelled": cancelled,
        }

    def disarm_capacity(self) -> None:
        _require(self._open is None and self._prepared is None and
                 self._closed is None,
                 "v94 CUDA cannot disarm an open transaction")
        self._armed = None
        self._producer_rows = None
        self._capacity_receipt = None

    def resolve_armed_capacity(
        self, *, execution, parent, accepted_q,
        settling_velocity_m_s, settling_diameter_receipt,
        surface_fraction, shear_velocity_m_s,
    ) -> ResidentSuspendedCapacityResolutionV94:
        _require(self._armed is not None and self._producer_rows is None,
                 "v94 CUDA capacity provider is not singly armed")
        armed = self._armed
        control_parent = armed["control_parent"]
        _require(execution is self.outer_execution.base_execution and
                 parent is control_parent.base_state.base_state,
                 "v94 CUDA capacity parent or execution differs")
        if armed["cancelled"]():
            raise InterruptedError("v94 CUDA cancelled before v83 production")
        _require(isinstance(settling_diameter_receipt,
                            SettlingDiameterReceiptV92) and
                 settling_diameter_receipt.source_kind == (
                     SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                     if np.any(self.control_execution.
                               iopsus_by_fraction == 1)
                     else SETTLING_DIAMETER_STATIC_PROFILE) and
                 (settling_diameter_receipt.source_state_sha256 ==
                      control_parent.state_sha256
                  if np.any(self.control_execution.
                            iopsus_by_fraction == 1)
                  else settling_diameter_receipt.
                       source_state_sha256 == execution.fingerprint),
                 "v94 CUDA parent DSS receipt differs")
        validate_settling_diameter_receipt_v92(
            settling_diameter_receipt, execution, parent)
        ng, nc = (self.control_execution.fraction_count,
                  self.control_execution.cell_count)
        fraction = self._device(
            surface_fraction, (ng, nc), "surface fraction")
        _require(bool(self.cp.array_equal(
            fraction, provider_surface_fraction_v114(self)).item()),
            "v94 CUDA v92 and Q3 surface owners differ")
        self._device(
            shear_velocity_m_s, (nc,), "accepted shear velocity")
        q = self._device(accepted_q, (3, nc), "accepted endpoint")
        ws = self._device(
            settling_velocity_m_s, (ng, nc), "incoming settling velocity")
        v92_parent = control_parent.base_state.base_state
        profile = _v78_execution(
            self.control_execution.base_execution).base_execution.profile
        inputs = armed["q3_inputs"]
        cp = self.cp
        rows = self._producer.produce(
            accepted_endpoint=q,
            observation_height_m=self._device(
                inputs["observation_height_m"], (nc,),
                "observation height"),
            water_density_kg_m3=cp.ascontiguousarray(cp.asarray(
                v92_parent.water_density_cell_kg_m3, dtype=cp.float64)),
            gravity_m_s2=cp.full(
                nc, profile.gravity_m_s2, dtype=cp.float64),
            # FLOW's equilibrium background diffusion keeps the initializer
            # scalar vicmol.  Dynamic T/S viscosity remains owned by v92
            # settling; it is not a tram1/eqtran input for uniform D50.
            kinematic_viscosity_m2_s=cp.full(
                nc, self.control_execution.frozen_uniform_threshold.
                initializer_kinematic_viscosity_m2_s,
                dtype=cp.float64),
            additional_shear_stress_pa=self._device(
                inputs["additional_shear_stress_pa"], (nc,),
                "additional shear"),
            sediment_d50_m=cp.ascontiguousarray(cp.asarray(
                profile.sediment_d50_m, dtype=cp.float64)),
            sediment_d90_m=self._device(
                inputs["sediment_d90_m"], (ng,), "D90"),
            sediment_density_kg_m3=cp.ascontiguousarray(cp.asarray(
                profile.sediment_density_kg_m3, dtype=cp.float64)),
            molecular_schmidt=self._device(
                inputs["molecular_schmidt"], (ng,), "molecular Schmidt"),
            aks_factor=self._device(
                inputs["aks_factor"], (ng,), "Aks factor"),
            maximum_volume_concentration=self._device(
                inputs["maximum_volume_concentration"], (ng,),
                "maximum concentration"),
            minimum_depth_m=self._device(
                inputs["minimum_depth_m"], (ng,), "minimum depth"),
            bed_factor=self._device(
                inputs["bed_factor"], (ng,), "bed factor"),
            suspended_factor=self._device(
                inputs["suspended_factor"], (ng,), "suspended factor"),
            settling_velocity_m_s=ws,
            roughness_z0_m=self._device(
                inputs["roughness_z0_m"], (ng, nc), "roughness z0"),
            active_fraction=fraction,
            frozen_uniform_threshold=
                self.control_execution.frozen_uniform_threshold)
        receipt = rows.receipt
        q_host = np.ascontiguousarray(cp.asnumpy(q))
        capacity = make_suspended_capacity_receipt_v92(
            execution, parent, accepted_q=q_host,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_kg_m_s=
                receipt.lifecycle_capacity_kg_m_s,
            source_kind=SUSPENDED_CAPACITY_ACTIVE_V83_V94,
            source_receipt_sha256=receipt.receipt_sha256)
        validate_suspended_capacity_receipt_v92(
            capacity, execution, parent, accepted_q=q_host,
            settling_diameter_receipt=settling_diameter_receipt)
        if armed["cancelled"]():
            raise InterruptedError("v94 CUDA cancelled after v83 production")
        self._producer_rows = rows
        self._capacity_receipt = capacity
        return ResidentSuspendedCapacityResolutionV94(
            capacity, rows.lifecycle_capacity_kg_m_s)

    def _candidate_diameter(self, parent: ActiveSuspendedControlStateV94):
        _require(self._producer_rows is not None,
                 "v94 CUDA candidate DSS requires its shared v83 producer")
        cp = self.cp
        ng, nc = (self.control_execution.fraction_count,
                  self.control_execution.cell_count)
        stage = cp.ascontiguousarray(cp.maximum(
            0.0, self._producer_rows.producer_diagnostics[2]-1.0))
        parent_dss = cp.ascontiguousarray(cp.asarray(
            parent.suspended_diameter_m, dtype=cp.float64))
        profile = _v78_execution(
            self.control_execution.base_execution).base_execution.profile
        d50 = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(profile.sediment_d50_m,
                       dtype=cp.float64)[:, None], (ng, nc)))
        active_rows = np.flatnonzero(
            self.control_execution.iopsus_by_fraction == 1)
        candidate = parent_dss.copy()
        if active_rows.size:
            rows = cp.asarray(active_rows, dtype=cp.int32)
            active_candidate = self._selector.iopsus_diameter(
                formula_code=FORMULA_CODE, iopsus=1,
                fixed_dss_m=cp.ascontiguousarray(parent_dss[rows]),
                d50_m=cp.ascontiguousarray(d50[rows]),
                transport_stage=cp.ascontiguousarray(stage[rows]))
            candidate[rows] = active_candidate
            self._iopsus_production_launch_count += 1
        candidate = cp.ascontiguousarray(candidate)
        mask = cp.asarray(
            self.control_execution.iopsus_by_fraction,
            dtype=cp.int32)[:, None] == 1
        _require(bool(cp.array_equal(
            candidate[~mask[:, 0]], parent_dss[~mask[:, 0]]).item()),
            "v94 CUDA inactive IopSus fraction changed DSS")
        return stage, candidate

    def _face_trial(
        self, *, producer: ResidentFactCrProducerRowsV94,
        v92_trial: TransportedFluidTrialV92, stage, candidate_dss,
    ) -> ResidentQ3V83ActiveFaceTrialV94:
        """Run the exact v73/H4/v93 face and Exner path on CUDA."""
        cp = self.cp
        ng = self.control_execution.fraction_count
        nc = self.control_execution.cell_count
        nf = self.binding.core.topology.nface
        cell_vector = cp.ascontiguousarray(producer.bed_capacity_kg_m_s)
        endpoint_wet = getattr(self, "_endpoint_wet_mask_v97", None)
        endpoint_face_active = getattr(
            self, "_endpoint_face_active_v97", None)
        if (endpoint_wet is None) != (endpoint_face_active is None):
            raise ActiveSuspendedControlV94Error(
                "v97 CUDA Q3 endpoint mask inventory differs")
        if endpoint_wet is not None:
            _require(
                isinstance(endpoint_wet, cp.ndarray) and
                endpoint_wet.shape == (nc,) and
                endpoint_wet.dtype == cp.bool_ and
                isinstance(endpoint_face_active, cp.ndarray) and
                endpoint_face_active.shape == (nf,) and
                endpoint_face_active.dtype == cp.bool_,
                "v97 CUDA Q3 endpoint mask residency differs")
            cell_vector = cell_vector.copy()
            cell_vector[:, :, ~endpoint_wet] = 0.0
        water = cp.ascontiguousarray(cp.asarray(
            v92_trial.hydraulic_face_receipt.accepted_water_flux_m2_s,
            dtype=cp.float64))
        mapping = self.binding.family_boundary_faces
        role, trace = self._dynamic_roles(
            hydraulic_boundary=cp.ascontiguousarray(water[mapping]),
            cell_vector=cell_vector)
        if endpoint_face_active is not None:
            boundary_face = self.binding.neighbour < 0
            role = role.copy()
            role[(~endpoint_face_active) & boundary_face] = FACE_CLOSED
        owner = self.binding.owner
        neighbour = self.binding.neighbour
        safe_other = cp.clip(neighbour, 0, nc-1)
        left = cp.ascontiguousarray(cell_vector[:, :, owner])
        right = cp.ascontiguousarray(cell_vector[:, :, safe_other])
        if endpoint_face_active is not None:
            left[:, :, ~endpoint_face_active] = 0.0
            right[:, :, ~endpoint_face_active] = 0.0

        # Exact v73 face projection.  The admitted downhill vector is zero,
        # but live taurat remains authenticated and is not replaced by a
        # constant/host surrogate.
        taurat = cp.ascontiguousarray(producer.producer_diagnostics[2])
        stress = cp.zeros((ng, nf), dtype=cp.float64)
        internal = role == FACE_INTERNAL
        outlet = role == FACE_OWNER_OUTLET
        stress[:, internal] = 0.5*(
            taurat[:, owner[internal]]+taurat[:, neighbour[internal]])
        stress[:, outlet] = taurat[:, owner[outlet]]
        stress[:, role == FACE_CLOSED] = 0.0
        if endpoint_face_active is not None:
            stress[:, ~endpoint_face_active] = 0.0
        neutral = cp.ones((ng, nf), dtype=cp.float64)
        hiding, surface_mean = provider_surface_slope_fields_v114(
            self, self.binding.core.topology, cp)
        d50 = cp.ascontiguousarray(cp.asarray(
            _base_topology(self.outer_execution).profile.sediment_d50_m,
            dtype=cp.float64))
        common = dict(
            left_bed=left, right_bed=right, owner_bed=left,
            trace_bed=trace, owner=owner, neighbour=neighbour, role=role,
            normal=self.binding.normal, downhill=self.binding.downhill,
            stress=stress, depth=neutral, d50=d50,
            mean_diameter=surface_mean, hiding=hiding)
        provisional = self.binding.composer.compose(
            availability=cp.ones((ng, nc), dtype=cp.float64), **common)
        dt = float(v92_trial.hydraulic_receipt.dt_s)
        _require(np.isfinite(dt) and dt > 0.0,
                 "v94 CUDA accepted dt must be positive finite")
        outgoing = cp.zeros((ng, nc), dtype=cp.float64)
        item_count = ng*nc
        threads = 128
        kernel_name = "q3_v93_outgoing_cell_gather"
        self._deterministic_kernels[kernel_name](
            ((item_count+threads-1)//threads,), (threads,), (
                np.int32(ng), np.int32(nc), np.int32(nf),
                self.binding.cell_face_offset, self.binding.cell_faces,
                provisional.donor, owner, neighbour, role,
                provisional.flux, self.binding.face_length_m,
                np.float64(dt), self.binding.cell_area_m2,
                np.int32(DONOR_OWNER), np.int32(DONOR_NEIGHBOUR),
                np.int32(DONOR_TRACE), np.int32(FACE_INTERNAL),
                np.int32(FACE_OWNER_OUTLET), outgoing))
        self._deterministic_kernel_launch_counts[kernel_name] += 1
        fill = cp.where(self._state.bed_mass_kg_m2 > 0.0, 1.0, 0.0)
        fill = cp.where(
            outgoing > 0.0,
            cp.minimum(1.0, self._state.bed_mass_kg_m2 /
                       cp.where(outgoing > 0.0, outgoing, 1.0)), fill)
        fill = cp.ascontiguousarray(fill)
        accepted = self.binding.composer.compose(
            availability=fill, **common)
        flux = cp.ascontiguousarray(accepted.flux)
        if endpoint_face_active is not None:
            _require(not bool(cp.any(
                flux[:, ~endpoint_face_active] != 0.0).item()),
                "v97 CUDA Q3 dry-touching face acquired bed flux")
        amount = flux*self.binding.face_length_m[None, :]*dt
        delta = cp.zeros((ng, nc), dtype=cp.float64)
        kernel_name = "q3_v93_delta_cell_gather"
        self._deterministic_kernels[kernel_name](
            ((item_count+threads-1)//threads,), (threads,), (
                np.int32(ng), np.int32(nc), np.int32(nf),
                self.binding.cell_face_offset, self.binding.cell_faces,
                self.binding.cell_face_sign, amount,
                self.binding.cell_area_m2, delta))
        self._deterministic_kernel_launch_counts[kernel_name] += 1
        nb = len(self.boundary_names)
        named = cp.zeros((ng, nb), dtype=cp.float64)
        absolute = cp.zeros_like(named)
        if ng*nb:
            kernel_name = "q3_v93_named_boundary_gather"
            self._deterministic_kernels[kernel_name](
                ((ng*nb+threads-1)//threads,), (threads,), (
                    np.int32(ng), np.int32(nb),
                    np.int32(self.binding.boundary_topology.nface),
                    np.int32(nf), mapping,
                    self.binding.boundary_topology.boundary_index,
                    amount, named, absolute))
            self._deterministic_kernel_launch_counts[kernel_name] += 1
        bed = self._state.bed_mass_kg_m2
        candidate = bed+delta
        tolerance = (4096.0*np.finfo(np.float64).eps *
                     cp.maximum(1.0, cp.abs(bed)))
        _require(not bool(cp.any(candidate < -tolerance).item()),
                 "v94 CUDA inventory limiter produced a negative bed")
        before = delta.copy()
        delta = cp.ascontiguousarray(cp.where(candidate < 0.0, -bed, delta))
        if endpoint_wet is not None:
            _require(not bool(cp.any(delta[:, ~endpoint_wet] != 0.0).item()),
                     "v97 CUDA Q3 dry cell acquired a bed delta")
        depletion = cp.ascontiguousarray(cp.sum(
            (delta-before)*self.binding.cell_area_m2[None, :], axis=1))
        residual = cp.ascontiguousarray(
            cp.sum(delta*self.binding.cell_area_m2[None, :], axis=1)+
            cp.sum(named, axis=1)-depletion)
        scale = cp.maximum(
            1.0, cp.sum(cp.abs(delta*self.binding.cell_area_m2[None, :]),
                        axis=1)+cp.sum(absolute, axis=1)+depletion)
        _require(not bool(cp.any(
            cp.abs(residual) >
            8192.0*np.finfo(np.float64).eps*scale).item()),
            "v94 CUDA face/Exner ledger does not close")

        ones = cp.ones(ng, dtype=cp.int32)
        zeros = cp.zeros(ng, dtype=cp.int32)
        iopsus = cp.ascontiguousarray(cp.broadcast_to(
            cp.asarray(self.control_execution.iopsus_by_fraction,
                       dtype=cp.int32)[:, None], (ng, nc)))
        factcr = cp.ascontiguousarray(cp.asarray(
            producer.receipt.factcr_application_count, dtype=cp.int32))
        correction = cp.zeros((ng, nc), dtype=cp.int32)
        vectors = cp.full(
            (ng, nc), int(self.control_execution.suscor), dtype=cp.int32)
        endpoint = _v84_array_digest(
            np.ascontiguousarray(v92_trial.candidate.q[0]),
            np.ascontiguousarray(v92_trial.candidate.q[1:]))
        placeholder = ResidentQ3V83ActiveFaceTrialV94(
            self._state.state_sha256, producer.receipt, endpoint,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            dt, MORPHOLOGICAL_FACTOR, flux, delta, fill, fill,
            named, absolute, depletion, residual,
            ones, ones, ones, zeros, ones, ones, ones, zeros,
            taurat, stage, candidate_dss, iopsus, factcr, correction,
            vectors, "", self.device_id)
        return ResidentQ3V83ActiveFaceTrialV94(
            **{**placeholder.__dict__,
               "trial_sha256": _resident_face_token(placeholder, cp)})

    def _validate_face_trial(
        self, value: ResidentQ3V83ActiveFaceTrialV94,
    ) -> None:
        _require(type(value) is ResidentQ3V83ActiveFaceTrialV94 and
                 value.device_id == self.device_id and
                 value.parent_state_sha256 == self._state.state_sha256 and
                 value.dt_s > 0.0 and
                 value.morphological_factor == MORPHOLOGICAL_FACTOR and
                 value.trial_sha256 == _resident_face_token(value, self.cp)
                 and value.producer_receipt is
                 (self._producer_rows.receipt
                  if self._producer_rows is not None else None),
                 "v94 CUDA face trial is stale or cross-device")
        validate_factcr_producer_receipt_v83(
            value.producer_receipt,
            self.control_execution.factcr_execution)
        cp = self.cp
        _require(bool(cp.array_equal(
            value.factcr_application_count,
            cp.asarray(value.producer_receipt.factcr_application_count,
                       dtype=cp.int32)).item()) and
            bool(cp.array_equal(
                value.iopsus_application_count,
                cp.broadcast_to(cp.asarray(
                    self.control_execution.iopsus_by_fraction,
                    dtype=cp.int32)[:, None],
                    value.iopsus_application_count.shape)).item()) and
            not bool(cp.any(
                value.extra_scalar_lifecycle_count != 0).item()) and
            not bool(cp.any(
                value.suscor_correction_application_count != 0).item()) and
            not bool(cp.any(
                value.suspended_vector_construction_owner_count !=
                int(self.control_execution.suscor)).item()),
            "v94 CUDA producer/control owner counts differ")
        _require(self._armed is not None,
                 "v94 CUDA face trial lost its parent control owner")
        expected_stage, expected_dss = _candidate_diameter_v94(
            self.control_execution, self._armed["control_parent"],
            value.producer_receipt)
        _require(bool(cp.array_equal(
            value.transport_stage,
            cp.asarray(expected_stage, dtype=cp.float64)).item()) and
            bool(cp.array_equal(
                value.candidate_suspended_diameter_m,
                cp.asarray(expected_dss, dtype=cp.float64)).item()) and
            bool(cp.array_equal(
                value.live_taurat,
                self._producer_rows.producer_diagnostics[2]).item()),
            "v94 CUDA transport stage or candidate DSS differs")

    def _make_receipt(self, *, outer_parent, v92_trial,
                      trial: ResidentQ3V83ActiveFaceTrialV94):
        cp = self.cp
        ng = self.outer_execution.fraction_count
        delta = _host_array(
            cp.asnumpy(trial.cell_delta_kg_m2),
            (ng, self.outer_execution.cell_count), "v94 CUDA bed delta")
        positive = _host_array(
            -cp.asnumpy(trial.named_boundary_transfer_kg),
            (ng, len(self.boundary_names)), "v94 CUDA named positive-in")
        absolute = _host_array(
            cp.asnumpy(trial.named_boundary_absolute_kg), positive.shape,
            "v94 CUDA named absolute", nonnegative=True)
        depletion = _host_array(
            cp.asnumpy(trial.depletion_roundoff_positive_in_kg), (ng,),
            "v94 CUDA depletion", nonnegative=True)
        boundary = _host_array(
            np.sum(positive, axis=1)+depletion, (ng,),
            "v94 CUDA global boundary")
        routes = _host_array(
            self.outer_execution.routes, (ng,), "v94 CUDA routes",
            integer=True)
        owner = _host_array(
            np.ones(ng, dtype=np.int32), (ng,),
            "v94 CUDA owner count", integer=True)
        zero = _host_array(
            np.zeros(ng, dtype=np.int32), (ng,),
            "v94 CUDA extra scalar count", integer=True)
        placeholder = Q3ExnerReceiptV93(
            self.outer_execution.fingerprint, outer_parent.state_sha256,
            v92_trial.trial_sha256,
            v92_trial.hydraulic_receipt.receipt_sha256,
            v92_trial.hydraulic_face_receipt.receipt_sha256,
            self.execution_fingerprint, self.binding.binding_sha256,
            self.binding.topology_id, self._state.state_sha256,
            trial.trial_sha256, trial.trial_sha256,
            _resident_digest(cp, trial.face_flux_kg_m_s),
            _shared_topology_sha256(_base_topology(self.outer_execution)),
            self.ordered_fraction_ids,
            _v92_physical_sha256(self.outer_execution),
            trial.dt_s, MORPHOLOGICAL_FACTOR, self.boundary_names,
            routes, delta, positive, absolute, depletion, boundary, owner,
            self.backend, self.provider_kind, owner, owner, zero, "")
        return Q3ExnerReceiptV93(
            **{**placeholder.__dict__,
               "receipt_sha256": _receipt_token(placeholder)})

    def begin(
        self, *, outer_parent, parent_hles, handoff,
        v92_trial: TransportedFluidTrialV92,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> ResidentQ3V84OpenTrialV93:
        _require(callable(cancelled), "cancelled must be callable")
        _require(self._armed is not None and
                 self._producer_rows is not None and
                 self._capacity_receipt is not None,
                 "v94 CUDA provider capacity was not resolved once")
        control_parent = self._armed["control_parent"]
        _require(outer_parent is control_parent.base_state and
                 self._open is None and self._prepared is None and
                 self._closed is None,
                 "v94 CUDA provider parent or transaction differs")
        _validate_outer_state(outer_parent, self.outer_execution)
        _validate_v92_trial(
            self.outer_execution.base_execution,
            outer_parent.base_state, parent_hles, handoff, v92_trial)
        self.validate_current_bed_clock(outer_parent)
        _require(v92_trial.backend == BACKEND and
                 v92_trial.suspended_capacity_receipt is
                 self._capacity_receipt and
                 v92_trial.suspended_capacity_receipt.
                 source_receipt_sha256 ==
                 self._producer_rows.receipt.receipt_sha256 and
                 v92_trial.settling_diameter_receipt.source_kind == (
                     SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                     if np.any(self.control_execution.
                               iopsus_by_fraction == 1)
                     else SETTLING_DIAMETER_STATIC_PROFILE) and
                 (v92_trial.settling_diameter_receipt.
                      source_state_sha256 == control_parent.state_sha256
                  if np.any(self.control_execution.
                            iopsus_by_fraction == 1)
                  else v92_trial.settling_diameter_receipt.
                       source_state_sha256 ==
                       self.outer_execution.base_execution.fingerprint),
                 "v94 CUDA v92 did not consume the armed producer/parent DSS")
        producer = self._producer_rows.receipt
        _require(producer.accepted_endpoint_sha256 ==
                 _v83_digest(v92_trial.candidate.q),
                 "v94 CUDA v83 producer differs from exact v92 endpoint")
        _require(producer.incoming_settling_velocity_sha256 ==
                 _v83_digest(v92_trial.suspended_trial.base_trial.
                             settling_velocity_m_s),
                 "v94 CUDA v83 producer differs from exact v92 settling")
        _require(producer.active_fraction_sha256 == _v83_digest(
                     self.cp.asnumpy(provider_surface_fraction_v114(self))),
                 "v94 CUDA v83 producer differs from resident surface owner")
        if cancelled():
            raise InterruptedError("v94 CUDA cancelled before Q3 face work")
        try:
            stage, candidate = self._candidate_diameter(control_parent)
            face = self._face_trial(
                producer=self._producer_rows, v92_trial=v92_trial,
                stage=stage, candidate_dss=candidate)
            self._validate_face_trial(face)
            receipt = self._make_receipt(
                outer_parent=outer_parent, v92_trial=v92_trial,
                trial=face)
            handle = ResidentQ3V84OpenTrialV93(
                self._state, face, receipt, "")
            handle = ResidentQ3V84OpenTrialV93(
                handle.parent_state, handle.face_trial, handle.receipt,
                _open_token(handle))
            if cancelled():
                raise InterruptedError(
                    "v94 CUDA cancelled before Q3 trial publication")
            self._open = handle
            return handle
        except Exception:
            self._open = None
            self._prepared = None
            self._armed = None
            self._producer_rows = None
            self._capacity_receipt = None
            raise

    def publish_commit(self, value):
        result = super().publish_commit(value)
        self._armed = None
        self._producer_rows = None
        self._capacity_receipt = None
        return result

    def abort_closed_commit(self, value) -> None:
        """Abort a closed child and clear every v94 capacity transient.

        V93 closes Q3 before publishing the v92 child.  If that publication
        fails, the inherited abort clears only the closed Q3 handle.  The
        active-IopSus provider additionally owns its armed parent, resident
        producer rows, and capacity receipt, all of which must be discarded
        before an exact retry can start.
        """
        try:
            super().abort_closed_commit(value)
        finally:
            self._armed = None
            self._producer_rows = None
            self._capacity_receipt = None

    def rollback(self, value):
        result = super().rollback(value)
        self._armed = None
        self._producer_rows = None
        self._capacity_receipt = None
        return result

    def cancel(self, value):
        return self.rollback(value)

    def checkpoint(self) -> dict[str, object]:
        _require(self._armed is None,
                 "v94 CUDA checkpoint requires a committed control point")
        return super().checkpoint()

    def restore_checkpoint(self, value: Mapping[str, object]):
        self._armed = None
        self._producer_rows = None
        self._capacity_receipt = None
        return super().restore_checkpoint(value)


class ResidentActiveSuspendedControlLifecycleV94(
        ResidentSuspendedExnerLifecycleV93):
    """Atomic actual-CUDA v94 lifecycle with accepted-parent DSS lag."""

    is_v94_active_suspended_control_consumer = True

    def __init__(
        self, execution: ActiveSuspendedControlExecutionV94,
        inner_runtime: ResidentTransportedFluidSettlingLifecycleV92,
        q3_provider: ResidentQ3V83ActiveExnerAdapterV94,
    ):
        execution_document_v94(execution)
        try:
            from braids_gpu.active_suspended_wetdry_v97 import (
                ResidentQ3V83ActiveWetDryExnerAdapterV97,
            )
            accepted_provider_types = (
                ResidentQ3V83ActiveExnerAdapterV94,
                ResidentQ3V83ActiveWetDryExnerAdapterV97)
        except ImportError:  # additive v97 module can be mid-import
            accepted_provider_types = (
                ResidentQ3V83ActiveExnerAdapterV94,)
        _require(execution.backend == "implicit_fwave_midpoint_gpu" and
                 type(inner_runtime) is
                 ResidentTransportedFluidSettlingLifecycleV92 and
                 type(q3_provider) in accepted_provider_types and
                 q3_provider.control_execution is execution,
                 "v94 CUDA lifecycle requires its exact reviewed owners")
        super().__init__(execution.base_execution, inner_runtime, q3_provider)
        self.control_execution = execution
        self._control_parent: ActiveSuspendedControlStateV94 | None = None
        self._control_trial: ActiveSuspendedControlTrialV94 | None = None
        self.runtime_receipt = {
            **self.runtime_receipt,
            **q3_provider.runtime_receipt,
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "active_iopsus_by_fraction":
                execution.iopsus_by_fraction.tolist(),
            "parent_dss_current_settling": True,
            "current_taurat_candidate_dss": True,
            "candidate_dss_publication": "outer_accept_only",
            "actual_cuda_v94_outer_transaction": True,
            "automatic_cpu_fallback": False,
        }

    @property
    def kernel_launch_counts(self):
        result = super().kernel_launch_counts
        result["v94_provider"] = self.q3_provider.kernel_launch_counts
        return result

    def _clear_control(self) -> None:
        self._control_parent = None
        self._control_trial = None

    def _validate_resident_control_trial(
        self, parent: ActiveSuspendedControlStateV94,
        trial: ActiveSuspendedControlTrialV94,
    ) -> None:
        """Bind public host evidence back to the still-open CUDA owner."""
        try:
            _validate_control_trial(
                self.control_execution, parent, self._parent_hles,
                self._handoff, trial)
        except ActiveSuspendedControlV94Error as error:
            expected_stage, expected_dss = _candidate_diameter_v94(
                self.control_execution, parent, trial.producer_receipt)
            shape = (self.control_execution.fraction_count,
                     self.control_execution.cell_count)
            expected_iopsus = np.broadcast_to(
                self.control_execution.iopsus_by_fraction[:, None], shape)
            checks = {
                "live": np.array_equal(
                    trial.live_taurat,
                    trial.producer_receipt.producer_diagnostics[2]),
                "stage": np.array_equal(
                    trial.transport_stage, expected_stage),
                "dss": np.array_equal(
                    trial.candidate_suspended_diameter_m, expected_dss),
                "factcr": np.array_equal(
                    trial.owner_counts.factcr_application_count,
                    trial.producer_receipt.factcr_application_count),
                "iopsus": np.array_equal(
                    trial.owner_counts.iopsus_application_count,
                    expected_iopsus),
            }
            raise ActiveSuspendedControlV94Error(
                f"v94 CUDA shared validation failed; field checks={checks}") \
                from error
        face = trial.base_v93_trial.q3_open_trial.face_trial
        _require(type(face) is ResidentQ3V83ActiveFaceTrialV94 and
                 trial.base_v93_trial.q3_open_trial is
                 self.q3_provider._open and
                 face.producer_receipt is trial.producer_receipt,
                 "v94 CUDA outer trial lost its resident face owner")
        cp = self.cp
        pairs = (
            (face.live_taurat, trial.live_taurat),
            (face.transport_stage, trial.transport_stage),
            (face.candidate_suspended_diameter_m,
             trial.candidate_suspended_diameter_m),
            (face.factcr_application_count,
             trial.owner_counts.factcr_application_count),
            (face.iopsus_application_count,
             trial.owner_counts.iopsus_application_count),
            (face.suscor_correction_application_count,
             trial.owner_counts.suscor_correction_application_count),
            (face.suspended_vector_construction_owner_count,
             trial.owner_counts.suspended_vector_construction_owner_count),
        )
        _require(all(bool(cp.array_equal(
            device, cp.asarray(host, dtype=device.dtype)).item())
                     for device, host in pairs),
                 "v94 CUDA public evidence differs from its resident face")

    def begin_hydraulic_trial(
        self, parent: ActiveSuspendedControlStateV94,
        parent_hles, handoff, *, dt_s: float, time_s: float,
        cancelled=False,
    ) -> None:
        _require(self._control_parent is None and
                 self._control_trial is None,
                 "v94 CUDA previous outer transaction is unfinished")
        validate_active_suspended_control_state_v94(
            parent, self.control_execution, self.q3_provider)
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v94 accepted update limit was reached")
        if _cancelled(cancelled):
            raise InterruptedError(
                "v94 CUDA cancelled before child work")
        try:
            super().begin_hydraulic_trial(
                parent.base_state, parent_hles, handoff,
                dt_s=dt_s, time_s=time_s, cancelled=cancelled)
        except Exception:
            self._clear_control()
            raise
        self._control_parent = parent

    def abort_hydraulic_trial(self) -> None:
        try:
            super().abort_hydraulic_trial()
        finally:
            self._clear_control()

    def begin_scalar_trial(
        self, *, q3_inputs: Mapping[str, object], cancelled=False,
        **scalar_inputs,
    ) -> ActiveSuspendedControlTrialV94:
        parent = self._control_parent
        _require(parent is not None and self._control_trial is None,
                 "v94 CUDA scalar owner requires one hydraulic candidate")
        if _cancelled(cancelled):
            self.abort_hydraulic_trial()
            raise InterruptedError(
                "v94 CUDA scalar trial cancelled before launch")
        inputs = dict(scalar_inputs)
        _require("settling_diameter_receipt" not in inputs and
                 "suspended_capacity_resolver" not in inputs,
                 "v94 CUDA owns the v92 settling/capacity seams")
        # The low-level ABI retains this legacy required keyword, but the
        # authenticated resolver is the sole numerical producer.  A resident
        # zero placeholder is passed and is deliberately not consumed when
        # ``suspended_capacity_resolver`` is present.
        inputs["suspended_transport_capacity_kg_m_s"] = self.cp.zeros(
            (self.control_execution.fraction_count,
             self.control_execution.cell_count), dtype=self.cp.float64)
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
                "v94 CUDA caller surface fraction differs from bed owner")
        inputs["surface_fraction"] = provider_surface_fraction_v114(
            self.q3_provider)
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
        self.q3_provider.arm_capacity(
            control_parent=parent, q3_inputs=q3_inputs,
            cancelled=lambda: _cancelled(cancelled))
        try:
            base_trial = super().begin_scalar_trial(
                q3_inputs={}, cancelled=cancelled,
                **inputs,
                settling_diameter_receipt=settling,
                suspended_capacity_resolver=
                    self.q3_provider.resolve_armed_capacity)
            if _cancelled(cancelled):
                raise InterruptedError(
                    "v94 CUDA cancelled after resident child launch")
            face = base_trial.q3_open_trial.face_trial
            _require(type(face) is ResidentQ3V83ActiveFaceTrialV94,
                     "v94 CUDA provider returned a different face trial")
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
            host_correction = np.ascontiguousarray(
                self.cp.asnumpy(
                    face.suscor_correction_application_count))
            host_vectors = np.ascontiguousarray(
                self.cp.asnumpy(
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
                    "v94 CUDA cancelled before trial publication")
        except Exception:
            # The inherited owner restores both children and clears its
            # transaction.  A failure before it opens still leaves an armed
            # provider, which is explicitly disarmed here.
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

    def commit(self, parent, trial):
        _require(parent is self._control_parent and
                 trial is self._control_trial,
                 "v94 CUDA trial is not the resident open transaction")
        self._validate_resident_control_trial(parent, trial)
        _require(trial.base_v93_trial.candidate.state_sha256 ==
                 trial.candidate.base_state.state_sha256,
                 "v94 CUDA candidate v93 state differs before publication")
        accepted = trial.candidate
        accepted_receipt = {
            "outer_transaction_status": "accepted",
            "execution_fingerprint": self.control_execution.fingerprint,
            "trial_sha256": trial.trial_sha256,
            "accepted_state_sha256": accepted.state_sha256,
            "actual_cuda_kernels": True,
            "parent_dss_lag_enforced": True,
            "automatic_cpu_fallback": False,
        }
        try:
            # The inherited atomic owner validates, closes both children and
            # publishes the already-authenticated exact candidate.  No
            # fallible validation or numerical work follows publication.
            super().commit(parent.base_state, trial.base_v93_trial)
        except Exception:
            self._clear_control()
            raise
        self.last_accepted_receipt = accepted_receipt
        self._clear_control()
        return accepted

    def rollback(self, parent, trial):
        _require(parent is self._control_parent and
                 trial is self._control_trial,
                 "v94 CUDA trial is not the resident open transaction")
        self._validate_resident_control_trial(parent, trial)
        try:
            base = super().rollback(
                parent.base_state, trial.base_v93_trial)
            _require(base is parent.base_state,
                     "v94 CUDA rollback changed the parent v93 state")
            return parent
        finally:
            self._clear_control()

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state) -> dict[str, object]:
        _require(self._control_parent is None,
                 "v94 CUDA checkpoint requires a committed point")
        document = checkpoint_active_suspended_control_state_v94(
            self.control_execution, state, self.q3_provider)
        inner = self.inner.checkpoint(state.base_state.base_state)
        _require(inner == document["base_v93_checkpoint"][
                     "base_v92_checkpoint"],
                 "v94 CUDA resident/core v92 checkpoints differ")
        return document

    def restore_checkpoint(self, document):
        _require(self._control_parent is None,
                 "v94 CUDA restart cannot replace an open trial")
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
            state = load_active_suspended_control_checkpoint_v94(
                self.control_execution, document, self.q3_provider)
            resident = self.inner.restore_checkpoint(
                document["base_v93_checkpoint"]["base_v92_checkpoint"])
            _require(resident.state_sha256 ==
                     state.base_state.base_state.state_sha256,
                     "v94 CUDA restored v92 state differs")
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


def method_identity_v94_cuda() -> dict[str, object]:
    """Machine-readable bounded identity; this is not product admission."""
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "backend": BACKEND,
        "integrator": "implicit_fwave_midpoint",
        "active_iopsus": "per_fraction_exact_0_or_1_all_eight_profiles",
        "lag_order": "parent_DSS_current_ws_current_taurat_next_DSS",
        "transport_stage": "max(0,taurat-1)",
        "actual_cuda_v83_v66_v92_v93": True,
        "host_numerical_fallback": False,
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "suscor_kmax1_correction_application_count": 0,
        "downhill": "exact_zero_only",
        "product_admission": False,
    }


__all__ = [
    "BACKEND", "CUDA_COMPOSITION_ID", "GPU_PROVIDER_KIND",
    "ResidentActiveSuspendedControlLifecycleV94",
    "ResidentFactCrProducerRowsV94",
    "ResidentQ3V83ActiveExnerAdapterV94",
    "ResidentQ3V83ActiveFaceTrialV94",
    "method_identity_v94_cuda",
]
