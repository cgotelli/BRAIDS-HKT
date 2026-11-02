"""Resident scalar settling/exchange stage for a private v98 fine update.

This additive owner consumes the resident output of the one combined
ordinary+trace diffusion limiter.  It then executes FLOW's parent-lagged
SettleFrm=2 preparation, the current v83 capacity producer, equilibrium
exchange and endpoint dry cleanup on CUDA.  It deliberately does not create
or commit a legacy v92 state: a nonzero coarse/fine transfer belongs to the
outer v98 internal-interface ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from braids_core.sediment.amr_suspended_bed_selection_v100 import (
    wrap_amr_bed_selection_v100,
)
from braids_gpu.amr_bed_selector_v100 import ResidentAMRBedSelectorV100

from braids_core.sediment.active_suspended_control_v94 import (
    _candidate_diameter_v94,
    validate_factcr_producer_receipt_v83,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
)
from braids_core.sediment.transported_fluid_settling_v92 import (
    FLUID_SALINITY,
    FLUID_TEMPERATURE,
    SETTLING_DIAMETER_ACTIVE_IOPSUS_V94,
    SETTLING_DIAMETER_STATIC_PROFILE,
    FluidScalarTransportResultV92,
    _fluid_result_token,
    make_settling_diameter_receipt_v92,
    validate_settling_diameter_receipt_v92,
    validate_suspended_capacity_receipt_v92,
)
from braids_core.sediment.recursive_surface_feedback_v114 import (
    provider_surface_fraction_v114,
)
from braids_gpu.active_suspended_wetdry_v97 import (
    ResidentActiveSuspendedWetDryLifecycleV97,
    ResidentQ3V83ActiveWetDryExnerAdapterV97,
)
from braids_gpu.coupled_amr_fwave_v98 import (
    WetDryAMRInterfaceGPUTrialV98,
)
from braids_gpu.coupled_amr_geometry_v98 import (
    CoupledAMRGeometryBindingGPUV98,
    validate_coupled_amr_geometry_binding_gpu_v98,
)
from braids_gpu.coupled_amr_scalar_fct_v98 import (
    AMRScalarFCTResultGPUV98,
    ResidentAMRScalarFCTGPUV98,
    validate_amr_scalar_fct_result_gpu_v98,
)
from braids_gpu.coupled_amr_scalar_trace_v98 import (
    CoupledAMRScalarTraceGPUV98,
    validate_coupled_amr_scalar_trace_gpu_v98,
)
from braids_gpu.coupled_amr_trace_v98 import (
    CoupledAMRInternalTraceGPUV98,
)
from braids_gpu.coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    _grid,
    _v92_execution,
    _v92_state,
)
from braids_gpu.scalar_diffusion_trace_v98 import (
    ResidentScalarDiffusionTraceV98,
)
from braids_gpu.transported_fluid_settling_v92 import (
    CUDA_THREADS_PER_BLOCK,
    TEMPERATURE_FLOOR_C,
    ResidentParentTimedDynamicExchangeV92,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.scalar_exchange.actual_cuda.v98"
BACKEND = "actual_cuda"


class CoupledAMRScalarExchangeGPUV98Error(ValueError):
    """The resident private scalar/exchange stage differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRScalarExchangeGPUV98Error(
            "v98 CUDA scalar/exchange identity is not canonical JSON") \
            from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _poll(value: bool | Callable[[], bool]) -> Callable[[], bool]:
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return lambda: bool(value())
    raise CoupledAMRScalarExchangeGPUV98Error(
        "v98 CUDA scalar/exchange cancellation must be bool or callable")


def _frozen(value: object, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


class RetainedScalarDiffusionTraceV98(ResidentScalarDiffusionTraceV98):
    """Backward-compatible evidence call retaining its real CUDA outputs.

    The frozen parent implementation still performs all validation and host
    evidence publication.  This subclass only retains the exact device
    dictionary returned by that same launch so settling/exchange does not
    round-trip numerical arrays through the host.
    """

    def __init__(self, topology, *, backend):
        super().__init__(topology, backend=backend)
        self.base_runtime_sha256 = self.receipt["receipt_sha256"]
        self.resident_result = None
        self.geometry_binding = None
        self.interface_distance_m = None
        body = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID+":retained-diffusion",
            "base_runtime_sha256": self.base_runtime_sha256,
            "wrapper_source_sha256": _sha(Path(__file__)),
            "same_launch_retention_only": True,
            "additional_numerical_kernel_count": 0,
            "host_to_device_numerical_copy_count": 0,
            "backend": BACKEND,
        }
        self.retained_receipt = {
            **body, "receipt_sha256": _hash(body)}

    def clear_private(self) -> None:
        self.resident_result = None
        self.geometry_binding = None
        self.interface_distance_m = None

    def launch_resident(self, inputs, trace_inputs, *, dt_s):
        if self.geometry_binding is not None:
            cp = self.cp
            distance = cp.array(inputs[7], copy=True, order="C")
            faces = cp.asarray(
                self.geometry_binding.joined_geometry.fine_faces,
                dtype=cp.int32)
            distance[faces] = cp.asarray(
                self.interface_distance_m, dtype=cp.float64)
            inputs = (*inputs[:7], distance)
        result = super().launch_resident(
            inputs, trace_inputs, dt_s=dt_s)
        self.resident_result = result
        return result

    def evaluate(self, *args, geometry_binding=None, **kwargs):
        self.clear_private()
        try:
            if geometry_binding is not None:
                if type(geometry_binding) is not \
                        CoupledAMRGeometryBindingGPUV98:
                    raise CoupledAMRScalarExchangeGPUV98Error(
                        "v98 CUDA diffusion geometry type differs")
                validate_coupled_amr_geometry_binding_gpu_v98(
                    geometry_binding)
                trace_faces = tuple(map(int, kwargs.get("trace_faces", ())))
                if (geometry_binding.fine_grid.fingerprint !=
                        self.topology.fingerprint or
                        trace_faces !=
                        tuple(geometry_binding.joined_geometry.fine_faces)):
                    raise CoupledAMRScalarExchangeGPUV98Error(
                        "v98 CUDA diffusion geometry and trace differ")
                self.geometry_binding = geometry_binding
                self.interface_distance_m = np.ascontiguousarray(
                    geometry_binding.joined_geometry.
                        normal_centre_distance_m)
            result = super().evaluate(*args, **kwargs)
            if self.resident_result is None:
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA diffusion launch was not retained")
            return {
                **result,
                "joined_geometry_receipt_sha256": (
                    None if self.geometry_binding is None else
                    self.geometry_binding.receipt_sha256),
                "joined_cell_centre_distance":
                    self.geometry_binding is not None,
            }
        except Exception:
            self.clear_private()
            raise


@dataclass(frozen=True)
class AMRScalarExchangeResultGPUV98:
    execution_fingerprint: str
    parent_state_sha256: str
    hydraulic_trial_sha256: str
    scalar_trace_receipt_sha256: str
    substep_index: int
    dt_s: float
    fct_result_sha256: str
    diffusion_runtime_sha256: str
    diffusion_output_sha256: str
    diffusion_geometry_receipt_sha256: str | None
    settling_diameter_receipt: object
    suspended_capacity_receipt: object
    producer_receipt: object
    transport_stage: np.ndarray
    candidate_suspended_diameter_m: np.ndarray
    final_suspended_mass_kg_m2: np.ndarray
    final_bed_mass_kg_m2: np.ndarray
    settling_velocity_m_s: np.ndarray
    equilibrium_concentration_kg_m3: np.ndarray
    transferred_to_suspension_kg_m2: np.ndarray
    wetdry_transfer_to_bed_kg_m2: np.ndarray
    water_exchange_ledger_increment_kg: np.ndarray
    bed_exchange_ledger_increment_kg: np.ndarray
    wetdry_water_ledger_increment_kg: np.ndarray
    wetdry_bed_ledger_increment_kg: np.ndarray
    fluid_result: FluidScalarTransportResultV92
    endpoint_water_density_kg_m3: np.ndarray
    endpoint_kinematic_viscosity_m2_s: np.ndarray
    suspended_advective_face_flux_kg_m_s: np.ndarray
    suspended_diffusive_face_mass_owner_out_kg: np.ndarray
    suspended_advective_boundary_ledger_increment_kg: np.ndarray
    suspended_diffusive_boundary_ledger_increment_kg: np.ndarray
    boundary_last_outflow_concentration_kg_m3: np.ndarray
    boundary_return_clock_s: np.ndarray
    advective_internal_interface_ledger_kg: np.ndarray
    diffusive_internal_interface_ledger_kg: np.ndarray
    interface_faces: np.ndarray
    settling_launch_count: int
    capacity_producer_count: int
    exchange_launch_count: int
    endpoint_settling_wet_gate_count: int
    fluid_unpack_launch_count: int
    endpoint_property_launch_count: int
    state_or_clock_publication_count: int
    host_to_device_numerical_copy_count: int
    backend: str
    result_sha256: str
    suspended_bed_selection: object = None


def _result_payload(value: AMRScalarExchangeResultGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":result",
        "suspended_bed_selection": value.suspended_bed_selection.receipt_sha256,
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "hydraulic": value.hydraulic_trial_sha256,
        "scalar_trace": value.scalar_trace_receipt_sha256,
        "substep_index": value.substep_index,
        "dt_s": value.dt_s,
        "fct_result": value.fct_result_sha256,
        "diffusion_runtime": value.diffusion_runtime_sha256,
        "diffusion_output": value.diffusion_output_sha256,
        "diffusion_geometry": value.diffusion_geometry_receipt_sha256,
        "settling_diameter":
            value.settling_diameter_receipt.receipt_sha256,
        "capacity": value.suspended_capacity_receipt.receipt_sha256,
        "producer": value.producer_receipt.receipt_sha256,
        "fluid_result": value.fluid_result.result_sha256,
        "arrays_sha256": _digest(
            value.transport_stage,
            value.candidate_suspended_diameter_m,
            value.final_suspended_mass_kg_m2,
            value.final_bed_mass_kg_m2,
            value.settling_velocity_m_s,
            value.equilibrium_concentration_kg_m3,
            value.transferred_to_suspension_kg_m2,
            value.wetdry_transfer_to_bed_kg_m2,
            value.water_exchange_ledger_increment_kg,
            value.bed_exchange_ledger_increment_kg,
            value.wetdry_water_ledger_increment_kg,
            value.wetdry_bed_ledger_increment_kg,
            value.endpoint_water_density_kg_m3,
            value.endpoint_kinematic_viscosity_m2_s,
            value.suspended_advective_face_flux_kg_m_s,
            value.suspended_diffusive_face_mass_owner_out_kg,
            value.suspended_advective_boundary_ledger_increment_kg,
            value.suspended_diffusive_boundary_ledger_increment_kg,
            value.boundary_last_outflow_concentration_kg_m3,
            value.boundary_return_clock_s,
            value.advective_internal_interface_ledger_kg,
            value.diffusive_internal_interface_ledger_kg,
            value.interface_faces),
        "counts": [
            value.settling_launch_count,
            value.capacity_producer_count,
            value.exchange_launch_count,
            value.endpoint_settling_wet_gate_count,
            value.fluid_unpack_launch_count,
            value.endpoint_property_launch_count,
            value.state_or_clock_publication_count,
            value.host_to_device_numerical_copy_count,
        ],
        "backend": value.backend,
        "actual_cuda_kernels": True,
        "automatic_cpu_fallback": False,
    }


class ResidentAMRScalarExchangeGPUV98:
    """One open private settling/exchange context for fine Q3."""

    def __init__(
        self, execution: CoupledWetDryGPUExecutionV97,
        active_runtime: ResidentActiveSuspendedWetDryLifecycleV97,
    ) -> None:
        if (type(execution) is not CoupledWetDryGPUExecutionV97 or
                type(active_runtime) is not
                    ResidentActiveSuspendedWetDryLifecycleV97 or
                active_runtime.control_execution is not
                    execution.active_execution):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange execution/runtime differs")
        self.execution = execution
        self.active_runtime = active_runtime
        self.provider = active_runtime.q3_provider
        if type(self.provider) is not ResidentQ3V83ActiveWetDryExnerAdapterV97:
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange requires endpoint-wet Q3 owner")
        self.combined = active_runtime.inner.combined
        self.cp = self.combined.cp
        self.bed_selector = ResidentAMRBedSelectorV100(
            execution.active_execution.base_execution, self.cp)
        self.v92_execution = _v92_execution(execution.active_execution)
        self.grid = _grid(execution.active_execution)
        self.exchange = ResidentParentTimedDynamicExchangeV92(
            self.combined.base, self.v92_execution)
        self._open = False
        self._result = None
        self._parent = None
        self._accepted_q_host = None
        self._diffusion_geometry_receipt = None
        self.resident_final_suspended = None
        self.resident_final_bed = None
        self.runtime_receipt = {
            "schema": SCHEMA,
            "scope_id": SCOPE_ID+":runtime",
            "execution_fingerprint": execution.fingerprint,
            "active_execution_fingerprint":
                execution.active_execution.fingerprint,
            "provider_runtime_sha256": _hash(self.provider.runtime_receipt),
            "v92_combined_runtime_sha256":
                _hash(self.active_runtime.inner.runtime_receipt),
            "wrapper_source_sha256": _sha(Path(__file__)),
            "parent_lagged_settling": True,
            "current_capacity_then_exchange": True,
            "legacy_v92_state_publication_count": 0,
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "production_admission": False,
        }
        self.runtime_receipt["receipt_sha256"] = _hash(
            self.runtime_receipt)

    def _launch(self, name: str, count: int, args) -> None:
        if count:
            self.combined.fluid_kernels[name](
                ((count+CUDA_THREADS_PER_BLOCK-1)//
                 CUDA_THREADS_PER_BLOCK,),
                (CUDA_THREADS_PER_BLOCK,), args)

    def _clear(self) -> None:
        self._open = False
        self._result = None
        self._parent = None
        self._accepted_q_host = None
        self._diffusion_geometry_receipt = None
        self.resident_final_suspended = None
        self.resident_final_bed = None

    def abort(self) -> None:
        try:
            if (self.provider._armed is not None and
                    self.provider._open is None and
                    self.provider._prepared is None and
                    self.provider._closed is None):
                self.provider.disarm_capacity()
        finally:
            self.exchange._prepared = False
            self.exchange.parent_settling_diameter = None
            self._clear()

    def finish(self, result: AMRScalarExchangeResultGPUV98) -> None:
        if not self._open or result is not self._result:
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange result is not open")
        try:
            validate_amr_scalar_exchange_result_gpu_v98(
                result, self)
            self.provider.disarm_capacity()
        finally:
            self._clear()

    def begin(
        self, parent: CoupledWetDryGPUStateV97,
        hydraulic: WetDryAMRInterfaceGPUTrialV98,
        fct_runtime: ResidentAMRScalarFCTGPUV98,
        fct_result: AMRScalarFCTResultGPUV98,
        diffusion_runtime: RetainedScalarDiffusionTraceV98,
        diffusion_result: Mapping[str, object], *,
        q3_inputs: Mapping[str, object], shear_velocity_m_s,
        combined_boundary_concentration,
        substep_index: int,
        cancelled: bool | Callable[[], bool] = False,
    ) -> AMRScalarExchangeResultGPUV98:
        if self._open:
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange transaction is already open")
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError(
                "v98 CUDA scalar/exchange cancelled before work")
        if (type(parent) is not CoupledWetDryGPUStateV97 or
                parent.execution_fingerprint != self.execution.fingerprint or
                type(hydraulic) is not WetDryAMRInterfaceGPUTrialV98 or
                hydraulic.execution_fingerprint !=
                    self.execution.hydraulic_execution.fingerprint or
                type(fct_runtime) is not ResidentAMRScalarFCTGPUV98 or
                type(diffusion_runtime) is not
                    RetainedScalarDiffusionTraceV98 or
                type(substep_index) is not int or substep_index not in (0, 1)):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange child identity differs")
        validate_amr_scalar_fct_result_gpu_v98(fct_result, fct_runtime)
        validate_coupled_amr_scalar_trace_gpu_v98(
            fct_runtime.scalar_trace, fct_runtime.internal_trace,
            self.execution)
        if (fct_result.substep_index != substep_index or
                diffusion_runtime.resident_result is None or
                diffusion_result.get("runtime_sha256") !=
                    diffusion_runtime.base_runtime_sha256 or
                diffusion_result.get("scalar_diffusion_owner_count") != 1 or
                diffusion_result.get("inventory_limiter_owner_count") != 1 or
                diffusion_result.get("turbulence_cap_evaluation_count") != 0 or
                diffusion_result.get("state_or_clock_publication_count") != 0):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange diffusion evidence differs")
        geometry_receipt = diffusion_result.get(
            "joined_geometry_receipt_sha256")
        expected_geometry_receipt = (
            None if diffusion_runtime.geometry_binding is None else
            diffusion_runtime.geometry_binding.receipt_sha256)
        if (geometry_receipt != expected_geometry_receipt or
                bool(diffusion_result.get(
                    "joined_cell_centre_distance", False)) !=
                (expected_geometry_receipt is not None)):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange diffusion geometry differs")
        cp = self.cp
        resident = diffusion_runtime.resident_result
        if (not isinstance(resident.get("mass"), cp.ndarray) or
                not bool(cp.array_equal(
                    resident["mass"], cp.asarray(
                        diffusion_result["mass_kg_m2"])).item())):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange resident diffusion differs")
        trial_h = hydraulic.wetdry_trial
        dt = float(trial_h.dt_s)
        if (not math.isfinite(dt) or dt <= 0.0 or
                fct_result.dt_s != dt):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange timestep differs")
        v92_parent = _v92_state(parent.active_state)
        active_parent = parent.active_state
        ng, nc, nf = (self.execution.fraction_count,
                      self.execution.cell_count,
                      self.execution.face_count)
        active_fluid = tuple(fct_runtime.scalar_trace.active_fluid_lanes)
        nl = ng+len(active_fluid)
        boundary = np.asarray(combined_boundary_concentration)
        if (boundary.shape != (nl, nf) or not np.isfinite(boundary).all() or
                np.any(boundary < 0.0)):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange boundary concentration differs")
        water_host = np.asarray(
            trial_h.accepted_limited_face_water_flux_m2_s)
        # Physical-boundary flux remains owned by the selected v97 hydraulic
        # solve and is consumed unchanged.  Only the two selected AMR faces
        # are diverted into the additive internal-interface ledger.
        h0 = cp.ascontiguousarray(cp.asarray(parent.q[0]))
        h1 = cp.ascontiguousarray(cp.asarray(trial_h.endpoint_q[0]))
        momentum = cp.ascontiguousarray(cp.asarray(trial_h.endpoint_q[1:]))
        water = cp.ascontiguousarray(cp.asarray(water_host))
        parent_suspended = cp.ascontiguousarray(cp.asarray(
            v92_parent.suspended_state.suspended_mass_kg_m2))
        parent_bed = cp.ascontiguousarray(cp.asarray(
            v92_parent.suspended_state.bed_mass_kg_m2))
        parent_temperature = cp.ascontiguousarray(cp.asarray(
            v92_parent.temperature_cell_c))
        parent_density = cp.ascontiguousarray(cp.asarray(
            v92_parent.water_density_cell_kg_m3))
        shear = cp.ascontiguousarray(cp.asarray(
            shear_velocity_m_s, dtype=cp.float64))
        if (shear.shape != (nc,) or
                not bool(cp.all(cp.isfinite(shear) & (shear >= 0.0)).item())):
            raise CoupledAMRScalarExchangeGPUV98Error(
                "v98 CUDA scalar/exchange shear differs")
        active_iopsus = bool(np.any(
            self.execution.active_execution.iopsus_by_fraction == 1))
        settling_receipt = make_settling_diameter_receipt_v92(
            self.v92_execution, v92_parent,
            suspended_diameter_m=active_parent.suspended_diameter_m,
            source_kind=(SETTLING_DIAMETER_ACTIVE_IOPSUS_V94
                         if active_iopsus else
                         SETTLING_DIAMETER_STATIC_PROFILE),
            source_state_sha256=(active_parent.state_sha256
                                 if active_iopsus else None))
        validate_settling_diameter_receipt_v92(
            settling_receipt, self.v92_execution, v92_parent)
        settling_diameter = cp.ascontiguousarray(cp.asarray(
            settling_receipt.suspended_diameter_m))
        self.provider.arm_capacity(
            control_parent=active_parent, q3_inputs=q3_inputs,
            cancelled=poll)
        try:
            self.exchange.prepare_parent(
                depth_before=h0, suspended_mass=parent_suspended,
                temperature=parent_temperature,
                water_density=parent_density,
                settling_diameter_m=settling_diameter)
            endpoint_dry = h1 <= self.grid.dry_tolerance_m
            if (self.v92_execution.base_execution.state_domain_mode ==
                    STATE_DOMAIN_MOVING_WETDRY_V97):
                self.exchange.settling_velocity[:, endpoint_dry] = 0.0
                wet_gate_count = 1
            else:
                wet_gate_count = 0
            if poll():
                raise InterruptedError(
                    "v98 CUDA scalar/exchange cancelled after settling")
            accepted_q = cp.ascontiguousarray(cp.vstack((h1, momentum)))
            capacity_resolution = self.provider.resolve_armed_capacity(
                execution=self.v92_execution, parent=v92_parent,
                accepted_q=accepted_q,
                settling_velocity_m_s=self.exchange.settling_velocity,
                settling_diameter_receipt=settling_receipt,
                surface_fraction=provider_surface_fraction_v114(self.provider),
                shear_velocity_m_s=shear)
            capacity = capacity_resolution.resident_capacity
            capacity_receipt = capacity_resolution.core_receipt
            validate_suspended_capacity_receipt_v92(
                capacity_receipt, self.v92_execution, v92_parent,
                accepted_q=cp.asnumpy(accepted_q),
                settling_diameter_receipt=settling_receipt)
            stage_resident, candidate_resident = \
                self.provider._candidate_diameter(active_parent)
            producer_rows = self.provider._producer_rows
            if producer_rows is None:
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange producer is absent")
            producer = producer_rows.receipt
            validate_factcr_producer_receipt_v83(
                producer,
                self.execution.active_execution.factcr_execution)
            wet = cp.ascontiguousarray((h1 > self.grid.dry_tolerance_m).
                                       astype(cp.int32))
            speed = cp.where(
                wet != 0,
                cp.hypot(momentum[0], momentum[1])/
                cp.where(wet != 0, h1, 1.0), 0.0)
            sediment_active = ((wet != 0) &
                               (h1 > self.grid.profile.
                                sediment_threshold_m))
            safe_depth = cp.where(
                sediment_active,
                cp.maximum(self.grid.profile.reference_depth_floor_m, h1),
                1.0)
            equilibrium_capacity = cp.ascontiguousarray(
                capacity/(speed[None, :]+1.0e-6)/safe_depth[None, :])
            total_shear = cp.ascontiguousarray(
                self.grid.profile.water_density_kg_m3*shear*shear)
            diffused = resident["mass"]
            exchanged = self.exchange.step(
                h0, h1, momentum, diffused[:ng], parent_bed,
                equilibrium_capacity, provider_surface_fraction_v114(
                    self.provider),
                total_shear, wet, dt)
            if bool(cp.any(exchanged["status"] != 0).item()):
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange kernel rejected the stage")
            cleanup = cp.where(
                wet[None, :] != 0, 0.0, exchanged["suspended"])
            final_suspended = cp.ascontiguousarray(
                exchanged["suspended"]-cleanup)
            bed_selection_numeric = self.bed_selector.launch(
                parent_bed, parent_suspended, cp.ascontiguousarray(diffused[:ng]),
                exchanged["delta"], cleanup)
            final_bed = bed_selection_numeric[3]

            # Reuse the frozen v92 CUDA affine inverse and EOS kernels.  The
            # resident diffusion output is consumed directly.
            constants = (
                self.v92_execution.equilibrium_temperature_c-
                TEMPERATURE_FLOOR_C,
                self.v92_execution.equilibrium_salinity_ppt)
            self.combined.full_final_fluid[0] = h1*constants[0]
            self.combined.full_final_fluid[1] = h1*constants[1]
            for local, lane in enumerate(active_fluid):
                self.combined.full_final_fluid[lane] = diffused[ng+local]
            self._launch("v92_unpack_affine_fluid", nc, (
                np.int32(nc), np.float64(self.grid.dry_tolerance_m),
                np.float64(TEMPERATURE_FLOOR_C),
                np.float64(self.v92_execution.equilibrium_temperature_c),
                np.float64(self.v92_execution.equilibrium_salinity_ppt),
                h1, self.combined.full_final_fluid,
                self.combined.h_temperature, self.combined.h_salinity,
                self.combined.temperature, self.combined.salinity,
                self.combined.wetdry_fluid_removed,
                self.combined.fluid_unpack_status))
            if bool(cp.any(self.combined.fluid_unpack_status != 0).item()):
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange fluid inverse failed")
            if FLUID_TEMPERATURE not in active_fluid:
                cp.multiply(
                    h1,
                    np.float64(
                        self.v92_execution.equilibrium_temperature_c),
                    out=self.combined.h_temperature)
                self.combined.temperature.fill(
                    self.v92_execution.equilibrium_temperature_c)
                self.combined.wetdry_fluid_removed[
                    FLUID_TEMPERATURE].fill(0.0)
            if FLUID_SALINITY not in active_fluid:
                cp.multiply(
                    h1,
                    np.float64(self.v92_execution.equilibrium_salinity_ppt),
                    out=self.combined.h_salinity)
                self.combined.salinity.fill(
                    self.v92_execution.equilibrium_salinity_ppt)
                self.combined.wetdry_fluid_removed[
                    FLUID_SALINITY].fill(0.0)
            if FLUID_TEMPERATURE in active_fluid:
                self._launch("v92_water_continuity", nc, (
                    np.int32(nc), np.float64(dt), h0, h1, water,
                    self.combined.length, self.combined.area,
                    self.combined.offset, self.combined.faces,
                    self.combined.signs, self.combined.water_residual,
                    self.combined.water_tolerance,
                    self.combined.water_status))
                if bool(cp.any(self.combined.water_status != 0).item()):
                    raise CoupledAMRScalarExchangeGPUV98Error(
                        "v98 CUDA scalar/exchange water continuity failed")
            else:
                self.combined.water_residual.fill(0.0)
            eos_code = {"eckart": 0, "unesco": 1, "nacl": 2}[
                self.v92_execution.density_eos]
            self._launch("v92_fluid_properties", nc, (
                np.int32(nc), self.combined.temperature,
                self.combined.salinity, np.int32(eos_code),
                np.int32(bool(len(active_fluid))),
                np.float64(self.grid.profile.water_density_kg_m3),
                self.combined.endpoint_density,
                self.combined.endpoint_viscosity,
                self.combined.endpoint_property_status))
            if bool(cp.any(
                    self.combined.endpoint_property_status != 0).item()):
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange endpoint EOS failed")
            self._launch("v92_boundary_water_ledger", self.combined.nb, (
                np.int32(self.combined.nb), np.float64(dt),
                self.combined.boundary_offset,
                self.combined.boundary_faces, water,
                self.combined.length, self.combined.water_ledger,
                self.combined.water_ledger_status))
            if bool(cp.any(
                    self.combined.water_ledger_status != 0).item()):
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange water ledger failed")
            if poll():
                raise InterruptedError(
                    "v98 CUDA scalar/exchange cancelled after exchange")

            # One post-kernel evidence readback.  No readback is uploaded
            # again for numerical work.
            arrays = {
                "final_suspended": _frozen(cp.asnumpy(final_suspended)),
                "final_bed": _frozen(cp.asnumpy(final_bed)),
                "settling": _frozen(cp.asnumpy(
                    exchanged["settling_velocity"])),
                "equilibrium": _frozen(cp.asnumpy(
                    exchanged["equilibrium_concentration"])),
                "delta": _frozen(cp.asnumpy(exchanged["delta"])),
                "cleanup": _frozen(cp.asnumpy(cleanup)),
                "stage": _frozen(cp.asnumpy(stage_resident)),
                "candidate_dss": _frozen(cp.asnumpy(candidate_resident)),
                "ht": _frozen(cp.asnumpy(self.combined.h_temperature)),
                "hs": _frozen(cp.asnumpy(self.combined.h_salinity)),
                "temperature": _frozen(cp.asnumpy(
                    self.combined.temperature)),
                "salinity": _frozen(cp.asnumpy(self.combined.salinity)),
                "fluid_cleanup": _frozen(cp.asnumpy(
                    self.combined.wetdry_fluid_removed)),
                "water_residual": _frozen(cp.asnumpy(
                    self.combined.water_residual)),
                "endpoint_density": _frozen(cp.asnumpy(
                    self.combined.endpoint_density)),
                "endpoint_viscosity": _frozen(cp.asnumpy(
                    self.combined.endpoint_viscosity)),
            }
            area = np.asarray(self.grid.cell_area_m2)[None, :]
            diffused_host = np.asarray(diffusion_result["mass_kg_m2"])
            water_exchange = _frozen(np.sum(
                (arrays["final_suspended"]+arrays["cleanup"]-
                 diffused_host[:ng])*area, axis=1))
            bed_exchange = _frozen(np.sum(
                cp.asnumpy(bed_selection_numeric[2])*area, axis=1)
                if not self.bed_selector.execution.neglect_entrainment
                else np.zeros(ng))
            wetdry_water = _frozen(-np.sum(arrays["cleanup"]*area, axis=1))
            wetdry_bed = _frozen(-wetdry_water)
            advective_flux = np.asarray(
                fct_result.advective_face_flux_kg_m_s)
            diffusive_amount = np.asarray(
                diffusion_result["diffusive_face_mass_owner_out_kg"])
            advective_boundary = np.asarray(
                fct_result.advective_boundary_ledger_kg)
            diffusive_boundary = np.asarray(
                diffusion_result["diffusive_boundary_ledger_kg"])
            effective = np.asarray(
                diffusion_result["effective_diffusivity_m2_s"])
            faces = np.asarray(fct_result.internal_interface_faces)
            adv_internal = _frozen(
                fct_result.internal_interface_owner_ledger_kg)
            diff_internal = _frozen(
                -diffusive_amount[:, faces])
            if not np.array_equal(
                    np.sum(diff_internal, axis=1),
                    diffusion_result["internal_interface_mass_in_kg"]):
                raise CoupledAMRScalarExchangeGPUV98Error(
                    "v98 CUDA scalar/exchange diffusion ledger differs")

            fluid_boundary = np.vstack((
                np.full(nf,
                        self.v92_execution.equilibrium_temperature_c),
                np.full(nf,
                        self.v92_execution.equilibrium_salinity_ppt)))
            fluid_flux = np.zeros((2, nf))
            fluid_diffusion = np.zeros((2, nf))
            fluid_advective_ledger = np.zeros(
                (2, self.v92_execution.boundary_count))
            fluid_diffusive_ledger = np.zeros_like(
                fluid_advective_ledger)
            lane_count = np.zeros(2, dtype=np.int32)
            fluid_wetdry = np.zeros(2)
            for local, lane in enumerate(active_fluid):
                row = ng+local
                fluid_boundary[lane] = boundary[row]
                fluid_flux[lane] = advective_flux[row]
                fluid_diffusion[lane] = diffusive_amount[row]
                fluid_advective_ledger[lane] = advective_boundary[row]
                fluid_diffusive_ledger[lane] = diffusive_boundary[row]
                fluid_wetdry[lane] = -np.sum(
                    arrays["fluid_cleanup"][lane]*area[0])
                lane_count[lane] = 1
                if lane == FLUID_TEMPERATURE:
                    fluid_boundary[lane] += TEMPERATURE_FLOOR_C
                    fluid_flux[lane] += TEMPERATURE_FLOOR_C*water_host
                    fluid_advective_ledger[lane] += (
                        TEMPERATURE_FLOOR_C*cp.asnumpy(
                            self.combined.water_ledger))
            fluid_values = [
                arrays["ht"], arrays["hs"], arrays["temperature"],
                arrays["salinity"], _frozen(fluid_boundary),
                _frozen(v92_parent.boundary_last_outflow_fluid),
                _frozen(v92_parent.boundary_return_clock_s),
                _frozen(fluid_flux), _frozen(fluid_diffusion),
                _frozen(fluid_advective_ledger),
                _frozen(fluid_diffusive_ledger), _frozen(fluid_wetdry),
                _frozen(effective), arrays["water_residual"],
                _frozen(lane_count, np.int32),
            ]
            fluid_placeholder = FluidScalarTransportResultV92(
                *fluid_values, "")
            fluid_result = FluidScalarTransportResultV92(
                **{**fluid_placeholder.__dict__,
                   "result_sha256":
                       _fluid_result_token(fluid_placeholder)})
            diffusion_sha = _digest(
                diffusion_result["mass_kg_m2"],
                diffusion_result["diffusive_face_mass_owner_out_kg"],
                diffusion_result["diffusive_boundary_ledger_kg"],
                diffusion_result["internal_interface_mass_in_kg"],
                diffusion_result["effective_diffusivity_m2_s"])
            selection = wrap_amr_bed_selection_v100(
                self.bed_selector.execution,
                [cp.asnumpy(a) for a in bed_selection_numeric],
                area=area[0], advective_boundary=advective_boundary[:ng],
                diffusive_boundary=diffusive_boundary[:ng],
                internal_ledger=adv_internal+diff_internal,
                backend=BACKEND,
                runtime_sha256=self.bed_selector.runtime_sha256,
                native_threads=0)
            placeholder = AMRScalarExchangeResultGPUV98(
                self.execution.fingerprint, parent.state_sha256,
                hydraulic.trial_sha256,
                fct_runtime.scalar_trace.receipt_sha256,
                substep_index, dt, fct_result.result_sha256,
                diffusion_runtime.retained_receipt["receipt_sha256"],
                diffusion_sha,
                diffusion_result.get("joined_geometry_receipt_sha256"),
                settling_receipt, capacity_receipt,
                producer, arrays["stage"], arrays["candidate_dss"],
                arrays["final_suspended"], arrays["final_bed"],
                arrays["settling"], arrays["equilibrium"],
                arrays["delta"], arrays["cleanup"], water_exchange,
                bed_exchange, wetdry_water, wetdry_bed, fluid_result,
                arrays["endpoint_density"], arrays["endpoint_viscosity"],
                _frozen(advective_flux[:ng]),
                _frozen(diffusive_amount[:ng]),
                _frozen(advective_boundary[:ng]),
                _frozen(diffusive_boundary[:ng]),
                _frozen(v92_parent.suspended_state.lifecycle.
                        boundary_last_outflow_concentration_kg_m3),
                _frozen(v92_parent.suspended_state.lifecycle.
                        boundary_return_clock_s),
                adv_internal, diff_internal,
                _frozen(faces, np.int32), 1, 1, 1, wet_gate_count,
                1, 1, 0, 0, BACKEND, "", selection)
            result = AMRScalarExchangeResultGPUV98(
                **{**placeholder.__dict__,
                   "result_sha256": _hash(_result_payload(placeholder))})
            self.resident_final_suspended = final_suspended.copy()
            self.resident_final_bed = final_bed.copy()
            self._parent = parent
            self._accepted_q_host = _frozen(cp.asnumpy(accepted_q))
            self._diffusion_geometry_receipt = geometry_receipt
            self._open = True
            self._result = result
            validate_amr_scalar_exchange_result_gpu_v98(result, self)
            return result
        except Exception:
            self.abort()
            raise


def validate_amr_scalar_exchange_result_gpu_v98(
    value: AMRScalarExchangeResultGPUV98,
    runtime: ResidentAMRScalarExchangeGPUV98,
) -> None:
    if (type(value) is not AMRScalarExchangeResultGPUV98 or
            type(runtime) is not ResidentAMRScalarExchangeGPUV98):
        raise CoupledAMRScalarExchangeGPUV98Error(
            "v98 CUDA scalar/exchange result/runtime type differs")
    ng, nc, nf = (runtime.execution.fraction_count,
                  runtime.execution.cell_count,
                  runtime.execution.face_count)
    nb = runtime.v92_execution.boundary_count
    nl = value.advective_internal_interface_ledger_kg.shape[0]
    shapes = (
        (value.transport_stage, (ng, nc)),
        (value.candidate_suspended_diameter_m, (ng, nc)),
        (value.final_suspended_mass_kg_m2, (ng, nc)),
        (value.final_bed_mass_kg_m2, (ng, nc)),
        (value.settling_velocity_m_s, (ng, nc)),
        (value.equilibrium_concentration_kg_m3, (ng, nc)),
        (value.transferred_to_suspension_kg_m2, (ng, nc)),
        (value.wetdry_transfer_to_bed_kg_m2, (ng, nc)),
        (value.water_exchange_ledger_increment_kg, (ng,)),
        (value.bed_exchange_ledger_increment_kg, (ng,)),
        (value.wetdry_water_ledger_increment_kg, (ng,)),
        (value.wetdry_bed_ledger_increment_kg, (ng,)),
        (value.endpoint_water_density_kg_m3, (nc,)),
        (value.endpoint_kinematic_viscosity_m2_s, (nc,)),
        (value.suspended_advective_face_flux_kg_m_s, (ng, nf)),
        (value.suspended_diffusive_face_mass_owner_out_kg, (ng, nf)),
        (value.suspended_advective_boundary_ledger_increment_kg,
         (ng, nb)),
        (value.suspended_diffusive_boundary_ledger_increment_kg,
         (ng, nb)),
        (value.boundary_last_outflow_concentration_kg_m3, (ng, nf)),
        (value.boundary_return_clock_s, (ng, nf)),
        (value.advective_internal_interface_ledger_kg, (nl, 2)),
        (value.diffusive_internal_interface_ledger_kg, (nl, 2)),
        (value.interface_faces, (2,)),
    )
    if runtime._parent is None or runtime._accepted_q_host is None:
        raise CoupledAMRScalarExchangeGPUV98Error(
            "v98 CUDA scalar/exchange validation context is absent")
    parent_v92 = _v92_state(runtime._parent.active_state)
    validate_settling_diameter_receipt_v92(
        value.settling_diameter_receipt, runtime.v92_execution, parent_v92)
    validate_suspended_capacity_receipt_v92(
        value.suspended_capacity_receipt, runtime.v92_execution, parent_v92,
        accepted_q=runtime._accepted_q_host,
        settling_diameter_receipt=value.settling_diameter_receipt)
    validate_factcr_producer_receipt_v83(
        value.producer_receipt,
        runtime.execution.active_execution.factcr_execution)
    expected_stage, expected_dss = _candidate_diameter_v94(
        runtime.execution.active_execution,
        runtime.provider._armed["control_parent"],
        value.producer_receipt)
    if (value.execution_fingerprint != runtime.execution.fingerprint or
            value.backend != BACKEND or
            value.diffusion_geometry_receipt_sha256 !=
                runtime._diffusion_geometry_receipt or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            any(np.asarray(array).shape != shape or
                np.asarray(array).flags.writeable or
                not np.isfinite(array).all()
                for array, shape in shapes) or
            np.any(value.final_suspended_mass_kg_m2 < 0.0) or
            np.any(value.final_bed_mass_kg_m2 < 0.0) or
            not np.array_equal(value.transport_stage, expected_stage) or
            not np.array_equal(
                value.candidate_suspended_diameter_m, expected_dss) or
            not np.array_equal(
                value.wetdry_bed_ledger_increment_kg,
                -value.wetdry_water_ledger_increment_kg) or
            any(count != 1 for count in (
                value.settling_launch_count,
                value.capacity_producer_count,
                value.exchange_launch_count,
                value.fluid_unpack_launch_count,
                value.endpoint_property_launch_count)) or
            value.endpoint_settling_wet_gate_count != 1 or
            value.state_or_clock_publication_count != 0 or
            value.host_to_device_numerical_copy_count != 0 or
            value.result_sha256 != _hash(_result_payload(value))):
        raise CoupledAMRScalarExchangeGPUV98Error(
            "v98 CUDA scalar/exchange result identity differs")


__all__ = [
    "AMRScalarExchangeResultGPUV98", "BACKEND",
    "CoupledAMRScalarExchangeGPUV98Error",
    "ResidentAMRScalarExchangeGPUV98",
    "RetainedScalarDiffusionTraceV98", "SCOPE_ID",
    "validate_amr_scalar_exchange_result_gpu_v98",
]
