"""Native combined scalar phase for one real v98 AMR fine substep.

This module is an additive transcription of the frozen v92 scalar phase.  It
keeps boundary memory, settling, exchange, ledgers and lifecycle publication
in their accepted order, but replaces the two numerical cell/face operators
with one combined sediment+active-fluid native OpenMP FCT launch followed by
one combined native OpenMP diffusion launch.  It owns no hydraulic solve and
does not advance a second clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping

import numpy as np

from .amr_suspended_bed_selection_v100 import (
    native_amr_bed_selection_v100, wrap_amr_bed_selection_v100,
)

from . import transported_fluid_settling_v92 as _v92
from .scalar_diffusion_distance_native_cpu_v98 import (
    NativeScalarDiffusionDistanceRuntimeV98,
)
from .coupled_amr_scalar_trace_cpu_v98 import CoupledAMRScalarTraceCPUV98
from .coupled_amr_trace_cpu_v98 import CoupledAMRInternalTraceCPUV98
from .coupled_amr_turbulence_trace_cpu_v98 import (
    CoupledAMRTurbulenceTraceCPUV98,
)
from .scalar_fct_interface_native_cpu_v98 import (
    NativeScalarFCTInterfaceRuntimeV98,
)
from .shared_turbulence_consumers_v91 import (
    ChkvicUsedTotalReceiptV91,
    HydraulicTurbulenceReceiptV91,
)
from .suspended_support_closures_v82 import SuspendedSupportExecutionV82
from .transported_fluid_settling_v92 import (
    AcceptedScalarHydraulicFaceReceiptV92,
    SettlingDiameterReceiptV92,
    TransportedFluidExecutionV92,
    TransportedFluidStateV92,
)


SCOPE_ID = "braids-q4-native-combined-scalar-amr-cpu-v98-v1"


@dataclass(frozen=True)
class NativeCombinedScalarInterfaceEvidenceV98:
    """Authoritative AMR-interface evidence around a v92-compatible trial."""

    compatibility_evidence: NativeCombinedScalarEvidenceV96
    scalar_trace_receipt_sha256: str
    substep_index: int
    interface_faces: tuple[int, ...]
    internal_interface_owner_ledger_kg: np.ndarray
    interface_fct_result_sha256: str
    interface_fct_runtime_sha256: str
    interface_owner_count: int
    physical_boundary_surrogate_count: int
    accepted_clock_increment_count: int
    evidence_sha256: str


@dataclass(frozen=True)
class AMRScalarStageResultV98:
    """Unpublished scalar/exchange images with an internal-interface ledger."""

    final_suspended_mass_kg_m2: np.ndarray
    final_bed_mass_kg_m2: np.ndarray
    boundary_last_outflow_concentration_kg_m3: np.ndarray
    boundary_return_clock_s: np.ndarray
    advective_boundary_ledger_increment_kg: np.ndarray
    diffusive_boundary_ledger_increment_kg: np.ndarray
    water_exchange_ledger_increment_kg: np.ndarray
    bed_exchange_ledger_increment_kg: np.ndarray
    wetdry_water_ledger_increment_kg: np.ndarray
    wetdry_bed_ledger_increment_kg: np.ndarray
    advective_face_flux_kg_m_s: np.ndarray
    diffusive_face_mass_owner_out_kg: np.ndarray
    settling_velocity_m_s: np.ndarray
    equilibrium_concentration_kg_m3: np.ndarray
    transferred_to_suspension_kg_m2: np.ndarray
    wetdry_transfer_to_bed_kg_m2: np.ndarray
    effective_diffusivity_m2_s: np.ndarray
    fluid_result: object
    parent_total_suspended_concentration_kg_m3: np.ndarray
    settling_water_density_kg_m3: np.ndarray
    settling_molecular_viscosity_m2_s: np.ndarray
    suspended_capacity_receipt: object
    scalar_trace_receipt_sha256: str
    substep_index: int
    internal_interface_owner_ledger_kg: np.ndarray
    interface_faces: tuple[int, ...]
    fct_runtime_sha256: str
    fct_result_sha256: str
    diffusion_runtime_sha256: str
    settling_exchange_runtime_sha256: str | None
    scalar_boundary_runtime_sha256: str | None
    native_threads: int
    scalar_owner_count: int
    internal_interface_owner_count: int
    physical_boundary_surrogate_count: int
    accepted_state_or_clock_publication_count: int
    result_sha256: str
    suspended_bed_selection: object = None


def _raw_result_token(value: AMRScalarStageResultV98) -> str:
    arrays = (
        value.final_suspended_mass_kg_m2, value.final_bed_mass_kg_m2,
        value.boundary_last_outflow_concentration_kg_m3,
        value.boundary_return_clock_s,
        value.advective_boundary_ledger_increment_kg,
        value.diffusive_boundary_ledger_increment_kg,
        value.water_exchange_ledger_increment_kg,
        value.bed_exchange_ledger_increment_kg,
        value.wetdry_water_ledger_increment_kg,
        value.wetdry_bed_ledger_increment_kg,
        value.advective_face_flux_kg_m_s,
        value.diffusive_face_mass_owner_out_kg,
        value.settling_velocity_m_s,
        value.equilibrium_concentration_kg_m3,
        value.transferred_to_suspension_kg_m2,
        value.wetdry_transfer_to_bed_kg_m2,
        value.effective_diffusivity_m2_s,
        value.fluid_result.h_temperature_c_m,
        value.fluid_result.h_salinity_ppt_m,
        value.internal_interface_owner_ledger_kg,
    )
    return _hash({
        "scope_id": SCOPE_ID + ":raw-stage",
        "suspended_bed_selection": value.suspended_bed_selection.receipt_sha256,
        "arrays_sha256": _digest(*arrays),
        "fluid_result": value.fluid_result.result_sha256,
        "capacity": value.suspended_capacity_receipt.receipt_sha256,
        "scalar_trace": value.scalar_trace_receipt_sha256,
        "substep_index": value.substep_index,
        "interface_faces": list(value.interface_faces),
        "fct_runtime": value.fct_runtime_sha256,
        "fct_result": value.fct_result_sha256,
        "diffusion_runtime": value.diffusion_runtime_sha256,
        "settling_exchange_runtime":
            value.settling_exchange_runtime_sha256,
        "scalar_boundary_runtime": value.scalar_boundary_runtime_sha256,
        "native_threads": value.native_threads,
        "counts": [
            value.scalar_owner_count,
            value.internal_interface_owner_count,
            value.physical_boundary_surrogate_count,
            value.accepted_state_or_clock_publication_count,
        ],
    })


def validate_amr_scalar_stage_result_v98(
    value: AMRScalarStageResultV98, *,
    execution: TransportedFluidExecutionV92,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    diffusion_runtime: NativeScalarDiffusionDistanceRuntimeV98,
    substep_index: int,
    native_threads: int,
) -> None:
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    lanes = scalar_trace.substep_concentration.shape[1]
    shapes = (
        (value.final_suspended_mass_kg_m2, (ng, nc)),
        (value.final_bed_mass_kg_m2, (ng, nc)),
        (value.boundary_last_outflow_concentration_kg_m3, (ng, nf)),
        (value.boundary_return_clock_s, (ng, nf)),
        (value.advective_boundary_ledger_increment_kg, (ng, nb)),
        (value.diffusive_boundary_ledger_increment_kg, (ng, nb)),
        (value.water_exchange_ledger_increment_kg, (ng,)),
        (value.bed_exchange_ledger_increment_kg, (ng,)),
        (value.wetdry_water_ledger_increment_kg, (ng,)),
        (value.wetdry_bed_ledger_increment_kg, (ng,)),
        (value.advective_face_flux_kg_m_s, (ng, nf)),
        (value.diffusive_face_mass_owner_out_kg, (ng, nf)),
        (value.settling_velocity_m_s, (ng, nc)),
        (value.equilibrium_concentration_kg_m3, (ng, nc)),
        (value.transferred_to_suspension_kg_m2, (ng, nc)),
        (value.wetdry_transfer_to_bed_kg_m2, (ng, nc)),
        (value.effective_diffusivity_m2_s, (nf,)),
        (value.parent_total_suspended_concentration_kg_m3, (nc,)),
        (value.settling_water_density_kg_m3, (nc,)),
        (value.settling_molecular_viscosity_m2_s, (nc,)),
        (value.internal_interface_owner_ledger_kg,
         (lanes, len(scalar_trace.interface_faces))),
    )
    if (type(value) is not AMRScalarStageResultV98 or
            type(scalar_trace) is not CoupledAMRScalarTraceCPUV98 or
            type(fct_runtime) is not NativeScalarFCTInterfaceRuntimeV98 or
            type(diffusion_runtime) is not
                NativeScalarDiffusionDistanceRuntimeV98 or
            value.scalar_trace_receipt_sha256 !=
                scalar_trace.receipt_sha256 or
            value.substep_index != substep_index or
            value.interface_faces != scalar_trace.interface_faces or
            value.fct_runtime_sha256 != fct_runtime.runtime_sha256 or
            value.diffusion_runtime_sha256 !=
                diffusion_runtime.receipt["receipt_sha256"] or
            value.native_threads != native_threads or
            value.scalar_owner_count != 1 or
            value.internal_interface_owner_count != 1 or
            value.physical_boundary_surrogate_count != 0 or
            value.accepted_state_or_clock_publication_count != 0 or
            value.result_sha256 != _raw_result_token(value) or
            any(array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()
                for array, shape in shapes) or
            np.any(value.final_suspended_mass_kg_m2 < 0.0) or
            np.any(value.final_bed_mass_kg_m2 < 0.0)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR scalar raw-stage evidence differs")


def _interface_evidence_token(
    value: NativeCombinedScalarInterfaceEvidenceV98,
) -> str:
    return _hash({
        "scope_id": SCOPE_ID + ":interface-evidence",
        "compatibility": value.compatibility_evidence.evidence_sha256,
        "scalar_trace": value.scalar_trace_receipt_sha256,
        "substep_index": value.substep_index,
        "interface_faces": list(value.interface_faces),
        "interface_ledger_sha256": _digest(
            value.internal_interface_owner_ledger_kg),
        "interface_fct_result": value.interface_fct_result_sha256,
        "interface_fct_runtime": value.interface_fct_runtime_sha256,
        "interface_owner_count": value.interface_owner_count,
        "physical_boundary_surrogate_count":
            value.physical_boundary_surrogate_count,
        "accepted_clock_increment_count":
            value.accepted_clock_increment_count,
    })


def validate_native_combined_scalar_interface_evidence_v98(
    value: NativeCombinedScalarInterfaceEvidenceV98, *,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    substep_index: int,
    suspended_trial_sha256: str,
    fluid_result_sha256: str,
) -> None:
    if (type(value) is not NativeCombinedScalarInterfaceEvidenceV98 or
            type(scalar_trace) is not CoupledAMRScalarTraceCPUV98 or
            type(fct_runtime) is not NativeScalarFCTInterfaceRuntimeV98 or
            type(value.compatibility_evidence) is not
                NativeCombinedScalarEvidenceV96 or
            type(substep_index) is not int or substep_index not in (0, 1) or
            value.scalar_trace_receipt_sha256 !=
                scalar_trace.receipt_sha256 or
            value.substep_index != substep_index or
            value.interface_faces != scalar_trace.interface_faces or
            value.internal_interface_owner_ledger_kg.shape !=
                (scalar_trace.substep_concentration.shape[1],
                 len(scalar_trace.interface_faces)) or
            value.internal_interface_owner_ledger_kg.flags.writeable or
            not np.isfinite(
                value.internal_interface_owner_ledger_kg).all() or
            value.interface_fct_runtime_sha256 !=
                fct_runtime.runtime_sha256 or
            value.compatibility_evidence.fct_runtime_sha256 !=
                fct_runtime.runtime_sha256 or
            value.compatibility_evidence.suspended_trial_sha256 !=
                suspended_trial_sha256 or
            value.compatibility_evidence.fluid_result_sha256 !=
                fluid_result_sha256 or
            value.interface_owner_count != 1 or
            value.physical_boundary_surrogate_count != 0 or
            value.accepted_clock_increment_count != 0 or
            value.evidence_sha256 != _interface_evidence_token(value)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR scalar interface evidence differs")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeCombinedScalarEvidenceV96:
    fct_runtime_sha256: str
    diffusion_runtime_sha256: str
    fct_result_sha256: str
    diffusion_result_sha256: str
    suspended_trial_sha256: str
    fluid_result_sha256: str
    combined_lane_count: int
    native_threads: int
    fct_launch_count: int
    diffusion_launch_count: int
    python_numerical_cell_face_loop_count: int
    state_or_clock_publication_count: int
    evidence_sha256: str
    settling_exchange_runtime_sha256: str | None = None
    settling_exchange_result_sha256: str | None = None
    settling_launch_count: int = 0
    exchange_dry_transfer_launch_count: int = 0
    scalar_boundary_runtime_sha256: str | None = None
    scalar_boundary_result_sha256: str | None = None
    scalar_boundary_launch_count: int = 0


def _evidence_token(value: NativeCombinedScalarEvidenceV96) -> str:
    payload = {
        "scope_id": SCOPE_ID,
        "fct_runtime_sha256": value.fct_runtime_sha256,
        "diffusion_runtime_sha256": value.diffusion_runtime_sha256,
        "fct_result_sha256": value.fct_result_sha256,
        "diffusion_result_sha256": value.diffusion_result_sha256,
        "suspended_trial_sha256": value.suspended_trial_sha256,
        "fluid_result_sha256": value.fluid_result_sha256,
        "combined_lane_count": value.combined_lane_count,
        "native_threads": value.native_threads,
        "fct_launch_count": value.fct_launch_count,
        "diffusion_launch_count": value.diffusion_launch_count,
        "python_numerical_cell_face_loop_count":
            value.python_numerical_cell_face_loop_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
    }
    if value.settling_exchange_runtime_sha256 is not None:
        payload.update({
            "settling_exchange_runtime_sha256":
                value.settling_exchange_runtime_sha256,
            "settling_exchange_result_sha256":
                value.settling_exchange_result_sha256,
            "settling_launch_count": value.settling_launch_count,
            "exchange_dry_transfer_launch_count":
                value.exchange_dry_transfer_launch_count,
        })
    if value.scalar_boundary_runtime_sha256 is not None:
        payload.update({
            "scalar_boundary_runtime_sha256":
                value.scalar_boundary_runtime_sha256,
            "scalar_boundary_result_sha256":
                value.scalar_boundary_result_sha256,
            "scalar_boundary_launch_count":
                value.scalar_boundary_launch_count,
        })
    return _hash(payload)


def validate_native_combined_scalar_evidence_v96(
    value: NativeCombinedScalarEvidenceV96, *,
    fct_runtime: NativeScalarFCTRuntimeV96,
    diffusion_runtime: NativeScalarDiffusionDistanceRuntimeV98,
    combined_lane_count: int,
    native_threads: int,
    settling_exchange_runtime=None,
    scalar_boundary_runtime=None,
) -> None:
    if type(value) is not NativeCombinedScalarEvidenceV96:
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native combined scalar evidence type differs")
    native_exchange = value.settling_exchange_runtime_sha256 is not None
    if settling_exchange_runtime is None:
        exchange_valid = (
            not native_exchange and
            value.settling_exchange_result_sha256 is None and
            value.settling_launch_count == 0 and
            value.exchange_dry_transfer_launch_count == 0)
    else:
        from .settling_exchange_native_cpu_v97 import (
            NativeSettlingExchangeRuntimeV97,
        )
        exchange_valid = (
            type(settling_exchange_runtime) is
                NativeSettlingExchangeRuntimeV97 and native_exchange and
            value.settling_exchange_runtime_sha256 ==
                settling_exchange_runtime.receipt.get("receipt_sha256") and
            isinstance(value.settling_exchange_result_sha256, str) and
            len(value.settling_exchange_result_sha256) == 64 and
            value.settling_launch_count == 1 and
            value.exchange_dry_transfer_launch_count == 1)
    native_boundary = value.scalar_boundary_runtime_sha256 is not None
    if scalar_boundary_runtime is None:
        boundary_valid = (
            not native_boundary and
            value.scalar_boundary_result_sha256 is None and
            value.scalar_boundary_launch_count == 0)
    else:
        from .scalar_boundary_native_cpu_v97 import (
            NativeScalarBoundaryRuntimeV97,
        )
        from .scalar_boundary_interface_native_cpu_v98 import (
            NativeScalarBoundaryInterfaceRuntimeV98,
        )
        boundary_valid = (
            type(scalar_boundary_runtime) in (
                NativeScalarBoundaryRuntimeV97,
                NativeScalarBoundaryInterfaceRuntimeV98) and
            native_boundary and
            value.scalar_boundary_runtime_sha256 ==
                scalar_boundary_runtime.receipt.get("receipt_sha256") and
            isinstance(value.scalar_boundary_result_sha256, str) and
            len(value.scalar_boundary_result_sha256) == 64 and
            value.scalar_boundary_launch_count == 1)
    if (value.fct_runtime_sha256 != fct_runtime.runtime_sha256 or
            value.diffusion_runtime_sha256 !=
                diffusion_runtime.receipt.get("receipt_sha256") or
            value.combined_lane_count != combined_lane_count or
            value.native_threads != native_threads or
            value.fct_launch_count != 1 or
            value.diffusion_launch_count != 1 or
            value.python_numerical_cell_face_loop_count != 0 or
            value.state_or_clock_publication_count != 0 or
            not exchange_valid or not boundary_valid or
            value.evidence_sha256 != _evidence_token(value)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native combined scalar evidence differs")


def _diffusion_result_token(
    runtime_sha256: str, native_threads: int, *,
    input_arrays: tuple[np.ndarray, ...],
    output_arrays: tuple[np.ndarray, ...],
) -> str:
    return _hash({
        "scope_id": SCOPE_ID + ":diffusion-result",
        "native_runtime_sha256": runtime_sha256,
        "native_threads": native_threads,
        "input_sha256": _digest(*input_arrays),
        "output_sha256": _digest(*output_arrays),
    })


def suspended_dynamic_trial_amr_scalar_cpu_v98(
    execution: TransportedFluidExecutionV92,
    parent: TransportedFluidStateV92,
    scalar_execution: SuspendedSupportExecutionV82,
    hydraulic: HydraulicTurbulenceReceiptV91, *,
    hydraulic_face_receipt: AcceptedScalarHydraulicFaceReceiptV92,
    settling_diameter_receipt: SettlingDiameterReceiptV92,
    depth_before_m: np.ndarray,
    depth_after_m: np.ndarray,
    endpoint_momentum_m2_s: np.ndarray,
    reconstructed_left_fluid: object,
    reconstructed_right_fluid: object,
    prescribed_boundary_fluid: object,
    dt_s: float,
    scalar_inputs: Mapping[str, object],
    fct_runtime: NativeScalarFCTInterfaceRuntimeV98,
    diffusion_runtime: NativeScalarDiffusionRuntimeV96,
    native_threads: int,
    settling_exchange_runtime=None,
    scalar_boundary_runtime=None,
    scalar_trace: CoupledAMRScalarTraceCPUV98,
    internal_trace: CoupledAMRInternalTraceCPUV98,
    turbulence_trace: CoupledAMRTurbulenceTraceCPUV98,
    fine_cap_receipt: ChkvicUsedTotalReceiptV91,
    substep_index: int,
    suspended_capacity_resolver: Callable[..., object] | None = None,
    suspended_bed_execution=None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple:
    """Run one native combined scalar pass, then the accepted v92 exchange."""
    if (type(fct_runtime) is not NativeScalarFCTInterfaceRuntimeV98 or
            type(diffusion_runtime) is not
                NativeScalarDiffusionDistanceRuntimeV98 or
            type(scalar_trace) is not CoupledAMRScalarTraceCPUV98 or
            type(internal_trace) is not CoupledAMRInternalTraceCPUV98 or
            type(turbulence_trace) is not
                CoupledAMRTurbulenceTraceCPUV98 or
            not isinstance(fine_cap_receipt, ChkvicUsedTotalReceiptV91) or
            type(substep_index) is not int or substep_index not in (0, 1) or
            type(native_threads) is not int or not 1 <= native_threads <= 256):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR scalar runtime, trace, substep or threads differ")
    if settling_exchange_runtime is not None:
        from .settling_exchange_native_cpu_v97 import (
            NativeSettlingExchangeRuntimeV97,
        )
        if type(settling_exchange_runtime) is not \
                NativeSettlingExchangeRuntimeV97:
            raise _v92.TransportedFluidSettlingV92Error(
                "v97 native settling/exchange runtime type differs")
    if scalar_boundary_runtime is not None:
        from .scalar_boundary_native_cpu_v97 import (
            NativeScalarBoundaryRuntimeV97,
        )
        from .scalar_boundary_interface_native_cpu_v98 import (
            NativeScalarBoundaryInterfaceRuntimeV98,
        )
        if type(scalar_boundary_runtime) not in (
                NativeScalarBoundaryRuntimeV97,
                NativeScalarBoundaryInterfaceRuntimeV98):
            raise _v92.TransportedFluidSettlingV92Error(
                "v98 native scalar-boundary runtime type differs")
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise InterruptedError(
            "v96 native scalar cancelled before scalar consumers")

    base_v78 = scalar_execution.base_execution
    base = base_v78.base_execution
    parent_v78 = parent.base_state.suspended_state
    parent_v82 = _v92._wrap_v82_state(scalar_execution, parent_v78)
    _v92.validate_implicit_fwave_midpoint_step_receipt_v78(
        hydraulic.base_step_receipt,
        hydraulic_execution_sha256=base_v78.hydraulic_execution_sha256,
        parent_accepted_updates=parent.accepted_updates,
        parent_time_s=parent.time_s, dt_s=dt_s)
    ng, nc, nf, nb = (execution.fraction_count, execution.cell_count,
                      execution.face_count, execution.boundary_count)
    required = {
        "reconstructed_left_concentration_kg_m3",
        "reconstructed_right_concentration_kg_m3",
        "prescribed_boundary_concentration_kg_m3",
        "accepted_face_exchange_area_m2", "transport_face_active",
        "shear_velocity_m_s", "surface_fraction", "hydro_constant",
    }
    if suspended_capacity_resolver is None:
        required.add("suspended_transport_capacity_kg_m_s")
    missing = required-set(scalar_inputs)
    if missing:
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native suspended inputs are absent: " +
            ", ".join(sorted(missing)))
    _v92.validate_accepted_scalar_hydraulic_face_receipt_v92(
        hydraulic_face_receipt, execution, parent, hydraulic,
        accepted_q=np.vstack((depth_after_m, endpoint_momentum_m2_s)))
    _v92.validate_settling_diameter_receipt_v92(
        settling_diameter_receipt, execution, parent)
    water = np.array(
        hydraulic_face_receipt.accepted_water_flux_m2_s, copy=True)
    left = _v92._mutable_array(
        scalar_inputs["reconstructed_left_concentration_kg_m3"], (ng, nf),
        "left suspended reconstruction", nonnegative=True)
    right = _v92._mutable_array(
        scalar_inputs["reconstructed_right_concentration_kg_m3"], (ng, nf),
        "right suspended reconstruction", nonnegative=True)
    prescribed = _v92._mutable_array(
        scalar_inputs["prescribed_boundary_concentration_kg_m3"], (ng, nb),
        "prescribed suspended boundary", nonnegative=True)
    face_area = _v92._mutable_array(
        scalar_inputs["accepted_face_exchange_area_m2"], (nf,),
        "face exchange area", nonnegative=True)
    face_active = _v92._mutable_array(
        scalar_inputs["transport_face_active"], (nf,),
        "transport face active", boolean=True)
    shear = _v92._mutable_array(
        scalar_inputs["shear_velocity_m_s"], (nc,), "shear velocity",
        nonnegative=True)
    capacity = (None if suspended_capacity_resolver is not None else
                _v92._mutable_array(
                    scalar_inputs["suspended_transport_capacity_kg_m_s"],
                    (ng, nc), "suspended capacity", nonnegative=True))
    surface = _v92._mutable_array(
        scalar_inputs["surface_fraction"], (ng, nc), "surface fraction",
        nonnegative=True)
    if np.any(surface > 1.0):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native surface fraction exceeds one")
    constant = _v92._mutable_array(
        scalar_inputs["hydro_constant"], (nc,), "hydro constant",
        boolean=True)
    h0 = np.asarray(depth_before_m)
    h1 = np.asarray(depth_after_m)
    dt = float(dt_s)
    old = parent_v78.lifecycle

    # Prepare compact active fluid lanes.  Temperature retains v92's affine
    # non-negative representation; publication below converts it back to hT.
    fluid_left_all = _v92._mutable_array(
        reconstructed_left_fluid, (2, nf), "left fluid reconstruction")
    fluid_right_all = _v92._mutable_array(
        reconstructed_right_fluid, (2, nf), "right fluid reconstruction")
    fluid_prescribed_all = _v92._mutable_array(
        prescribed_boundary_fluid, (2, nb), "prescribed boundary fluid")
    if (np.any(fluid_left_all[0] <= _v92.TEMPERATURE_FLOOR_C) or
            np.any(fluid_right_all[0] <= _v92.TEMPERATURE_FLOOR_C) or
            (nb and np.any(
                fluid_prescribed_all[0] <= _v92.TEMPERATURE_FLOOR_C)) or
            np.any(fluid_left_all[1] < 0.0) or
            np.any(fluid_right_all[1] < 0.0) or
            (nb and np.any(fluid_prescribed_all[1] < 0.0))):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native reconstructed fluid fields are outside bounds")
    active = np.asarray(execution.transported_mask)
    active_lanes = np.flatnonzero(active)
    lane_position = {int(lane): position
                     for position, lane in enumerate(active_lanes)}
    ht0 = np.asarray(parent.h_temperature_c_m)
    hs0 = np.asarray(parent.h_salinity_ppt_m)
    affine_temperature = ht0-_v92.TEMPERATURE_FLOOR_C*h0
    expected_affine = h0*(
        parent.temperature_cell_c-_v92.TEMPERATURE_FLOOR_C)
    relation_tolerance = 8192.0*np.finfo(float).eps*np.maximum(
        1.0, np.abs(expected_affine))
    if (np.any(np.abs(affine_temperature-expected_affine) >
               relation_tolerance) or
            np.any(affine_temperature < -relation_tolerance) or
            np.any(np.abs(hs0-h0*parent.salinity_cell_ppt) >
                   relation_tolerance)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native parent hT/hS relation differs")
    fluid_mass_all = np.vstack((
        np.maximum(affine_temperature, 0.0), hs0))
    fluid_left_all[0] -= _v92.TEMPERATURE_FLOOR_C
    fluid_right_all[0] -= _v92.TEMPERATURE_FLOOR_C
    fluid_prescribed_all[0] -= _v92.TEMPERATURE_FLOOR_C

    # The additive v97 boundary batch also owns the CSR water-divergence
    # bookkeeping.  Preserve the historical Python oracle only when that
    # optional runtime is not supplied.
    water_divergence = None
    continuity = None
    if scalar_boundary_runtime is None:
        water_divergence = np.zeros(nc)
        for cell in range(nc):
            for slot in range(base.cell_face_offset[cell],
                              base.cell_face_offset[cell+1]):
                face = int(base.cell_faces[slot])
                water_divergence[cell] += (
                    base.cell_face_sign[slot]*base.face_length_m[face]*
                    water[face]/base.cell_area_m2[cell])
        continuity = h1-h0+dt*water_divergence

    sediment_interior = np.zeros((ng, nc))
    wet0 = h0 > base.dry_tolerance_m
    sediment_interior[:, wet0] = (
        old.suspended_mass_kg_m2[:, wet0]/h0[wet0])
    sediment_boundary = np.zeros((ng, nf))
    next_sediment_last = np.array(
        old.boundary_last_outflow_concentration_kg_m3, copy=True)
    next_sediment_clock = np.array(old.boundary_return_clock_s, copy=True)
    if scalar_boundary_runtime is None:
        for face in range(nf):
            boundary_id = int(base.boundary_index[face])
            if boundary_id < 0:
                continue
            owner = int(base.owner[face])
            inward_q = -water[face]*base.face_length_m[face]
            resolved = _v92.resolve_flow2dh_open_boundary(
                prescribed[:, boundary_id], sediment_interior[:, owner],
                np.full(ng, inward_q), dt_s=dt,
                neumann=np.asarray(
                    base.profile.neumann_boundary_by_fraction, dtype=bool),
                return_time_s=base.boundary_return_time_s[:, boundary_id],
                previous_memory=_v92.Flow2DHBoundaryMemory(
                    old.boundary_last_outflow_concentration_kg_m3[:, face],
                    old.boundary_return_clock_s[:, face]))
            sediment_boundary[:, face] = \
                resolved.boundary_concentration_kg_m3
            next_sediment_last[:, face] = \
                resolved.memory.last_outflow_concentration_kg_m3
            next_sediment_clock[:, face] = resolved.memory.return_clock_s

    fluid_count = active_lanes.size
    fluid_mass = fluid_mass_all[active_lanes]
    fluid_left = fluid_left_all[active_lanes]
    fluid_right = fluid_right_all[active_lanes]
    fluid_prescribed = fluid_prescribed_all[active_lanes]
    fluid_interior = np.zeros((fluid_count, nc))
    if fluid_count:
        fluid_interior[:, wet0] = fluid_mass[:, wet0]/h0[wet0]
    fluid_boundary = np.zeros((fluid_count, nf))
    next_fluid_last = np.array(parent.boundary_last_outflow_fluid, copy=True)
    next_fluid_clock = np.array(parent.boundary_return_clock_s, copy=True)
    _v92._validate_boundary_memory(
        execution, next_fluid_last, next_fluid_clock)
    if scalar_boundary_runtime is None:
        for face in range(nf):
            boundary_id = int(base.boundary_index[face])
            if boundary_id < 0 or not fluid_count:
                continue
            owner = int(base.owner[face])
            inward_q = -water[face]*base.face_length_m[face]
            previous_affine_all = np.array(
                parent.boundary_last_outflow_fluid[:, face], copy=True)
            previous_affine_all[_v92.FLUID_TEMPERATURE] -= \
                _v92.TEMPERATURE_FLOOR_C
            resolved = _v92.resolve_flow2dh_open_boundary(
                fluid_prescribed[:, boundary_id], fluid_interior[:, owner],
                np.full(fluid_count, inward_q), dt_s=dt,
                neumann=np.zeros(fluid_count, dtype=bool),
                return_time_s=execution.fluid_boundary_return_time_s[
                    active_lanes, boundary_id],
                previous_memory=_v92.Flow2DHBoundaryMemory(
                    previous_affine_all[active_lanes],
                    parent.boundary_return_clock_s[active_lanes, face]))
            fluid_boundary[:, face] = resolved.boundary_concentration_kg_m3
            for compact, lane in enumerate(active_lanes):
                next_fluid_last[lane, face] = \
                    resolved.memory.last_outflow_concentration_kg_m3[compact]
                if lane == _v92.FLUID_TEMPERATURE:
                    next_fluid_last[lane, face] += \
                        _v92.TEMPERATURE_FLOOR_C
                next_fluid_clock[lane, face] = \
                    resolved.memory.return_clock_s[compact]
    native_boundary_result = None
    if scalar_boundary_runtime is not None:
        exterior = base.boundary_index >= 0
        fluid_last = np.array(
            parent.boundary_last_outflow_fluid[active_lanes], copy=True)
        if _v92.FLUID_TEMPERATURE in lane_position:
            position = lane_position[_v92.FLUID_TEMPERATURE]
            fluid_last[position, exterior] -= _v92.TEMPERATURE_FLOOR_C
        combined_mass_for_boundary = np.concatenate(
            (old.suspended_mass_kg_m2, fluid_mass))
        combined_prescribed = np.concatenate(
            (prescribed, fluid_prescribed), axis=0)
        combined_return_time = np.concatenate((
            base.boundary_return_time_s,
            execution.fluid_boundary_return_time_s[active_lanes]), axis=0)
        combined_last = np.concatenate((
            old.boundary_last_outflow_concentration_kg_m3,
            fluid_last), axis=0)
        combined_clock = np.concatenate((
            old.boundary_return_clock_s,
            parent.boundary_return_clock_s[active_lanes]), axis=0)
        combined_neumann = np.concatenate((
            np.asarray(base.profile.neumann_boundary_by_fraction,
                       dtype=np.int32),
            np.zeros(fluid_count, dtype=np.int32)))
        from .scalar_boundary_interface_native_cpu_v98 import (
            NativeScalarBoundaryInterfaceRuntimeV98,
        )
        boundary_kwargs = dict(
            dt_s=dt, native_threads=native_threads, cancelled=cancelled)
        if type(scalar_boundary_runtime) is \
                NativeScalarBoundaryInterfaceRuntimeV98:
            boundary_kwargs.update(
                trace_faces=np.asarray(
                    internal_trace.fine_faces, dtype=np.int32),
                trace_receipt_sha256=internal_trace.receipt_sha256,
                trace_source_sha256=
                    internal_trace.coarse_provisional_state_sha256,
                trace_runtime_sha256=
                    scalar_boundary_runtime.receipt["receipt_sha256"],
                substep_index=substep_index)
        native_boundary_result = scalar_boundary_runtime.evaluate(
            base, combined_mass_for_boundary, h0, h1, water,
            combined_prescribed, combined_return_time, combined_last,
            combined_clock, combined_neumann, **boundary_kwargs)
        sediment_boundary = np.asarray(
            native_boundary_result.boundary_concentration[:ng])
        fluid_boundary = np.asarray(
            native_boundary_result.boundary_concentration[ng:])
        next_sediment_last = np.asarray(
            native_boundary_result.next_last_outflow[:ng])
        next_sediment_clock = np.asarray(
            native_boundary_result.next_return_clock_s[:ng])
        next_fluid_last = np.zeros((2, nf), dtype=np.float64)
        next_fluid_clock = np.zeros((2, nf), dtype=np.float64)
        next_fluid_last[active_lanes] = \
            native_boundary_result.next_last_outflow[ng:]
        next_fluid_clock[active_lanes] = \
            native_boundary_result.next_return_clock_s[ng:]
        if _v92.FLUID_TEMPERATURE in lane_position:
            next_fluid_last[_v92.FLUID_TEMPERATURE, exterior] += \
                _v92.TEMPERATURE_FLOOR_C
        water_divergence = np.asarray(
            native_boundary_result.water_divergence_m_s)
        continuity = np.asarray(
            native_boundary_result.continuity_residual_m)
    next_fluid_last[~active] = 0.0
    next_fluid_clock[~active] = 0.0
    assert continuity is not None and water_divergence is not None
    continuity_scale = np.maximum(1.0, np.maximum(np.abs(h0), np.abs(h1)))
    if active[_v92.FLUID_TEMPERATURE] and np.any(
            np.abs(continuity) >
            32768.0*np.finfo(float).eps*continuity_scale):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native affine temperature continuity differs")

    # Source-faithful order: settling consumes the accepted parent r0 and
    # parent T/rho before constituent transport starts.
    parent_concentration = np.zeros((nc, ng))
    parent_concentration[wet0] = (
        old.suspended_mass_kg_m2[:, wet0]/h0[wet0]).T
    parent_ctot = np.ascontiguousarray(np.sum(parent_concentration, axis=1))
    # In the additive moving-wet/dry route FLOW's post-hydraulic fallve
    # stage consumes the parent concentration image but is gated by the
    # accepted endpoint wet set.  Historical fixed-domain routes retain the
    # frozen parent-wet gate exactly.
    settling_wet = (h1 > base.dry_tolerance_m if
        execution.base_execution.state_domain_mode ==
            _v92.STATE_DOMAIN_MOVING_WETDRY_V97 else wet0)
    settling_kwargs = dict(
        settling_diameter_m=
            settling_diameter_receipt.suspended_diameter_m.T,
        settling_reference_concentration_kg_m3=
            base.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base.profile.gravity_m_s2, wet_mask=settling_wet)
    if settling_exchange_runtime is None:
        settling = _v92.settlefrm2_noncohesive(
            parent_concentration, base.profile.sediment_d50_m,
            base.profile.settling_fac_dss,
            base.profile.sediment_density_kg_m3,
            parent.water_density_cell_kg_m3,
            parent.temperature_cell_c, **settling_kwargs)
    else:
        settling = settling_exchange_runtime.settle(
            parent_concentration, base.profile.sediment_d50_m,
            base.profile.settling_fac_dss,
            base.profile.sediment_density_kg_m3,
            parent.water_density_cell_kg_m3,
            parent.temperature_cell_c, native_threads=native_threads,
            cancelled=cancelled, **settling_kwargs)
    accepted_q = np.ascontiguousarray(np.vstack((h1, endpoint_momentum_m2_s)))
    if suspended_capacity_resolver is not None:
        if not callable(suspended_capacity_resolver):
            raise _v92.TransportedFluidSettlingV92Error(
                "v96 native capacity resolver must be callable")
        if cancelled():
            raise InterruptedError(
                "v96 native scalar cancelled before capacity production")
        try:
            resolved_capacity = suspended_capacity_resolver(
                execution=execution, parent=parent,
                accepted_q=accepted_q,
                settling_velocity_m_s=np.ascontiguousarray(
                    settling.settling_velocity_m_s.T),
                settling_diameter_receipt=settling_diameter_receipt,
                surface_fraction=np.ascontiguousarray(surface),
                shear_velocity_m_s=np.ascontiguousarray(shear))
        except InterruptedError:
            raise
        except Exception as error:
            raise _v92.TransportedFluidSettlingV92Error(
                "v96 native capacity resolver failed") from error
        if not isinstance(
                resolved_capacity, _v92.SuspendedCapacityReceiptV92):
            raise _v92.TransportedFluidSettlingV92Error(
                "v96 native capacity resolver returned another type")
        _v92.validate_suspended_capacity_receipt_v92(
            resolved_capacity, execution, parent, accepted_q=accepted_q,
            settling_diameter_receipt=settling_diameter_receipt)
        capacity_receipt = resolved_capacity
        capacity = _v92._mutable_array(
            resolved_capacity.suspended_capacity_kg_m_s, (ng, nc),
            "resolved suspended capacity", nonnegative=True)
        if cancelled():
            raise InterruptedError(
                "v96 native scalar cancelled after capacity production")
    else:
        assert capacity is not None
        capacity_receipt = _v92.make_suspended_capacity_receipt_v92(
            execution, parent, accepted_q=accepted_q,
            settling_diameter_receipt=settling_diameter_receipt,
            suspended_capacity_kg_m_s=capacity,
            source_kind=_v92.SUSPENDED_CAPACITY_LEGACY_INPUT,
            source_receipt_sha256=_v92._array_digest(capacity))
    assert capacity is not None
    if not np.array_equal(
            settling.molecular_viscosity_m2_s,
            parent.molecular_viscosity_cell_m2_s):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native settling viscosity differs from parent")

    combined_mass = np.concatenate((old.suspended_mass_kg_m2, fluid_mass))
    combined_left = np.concatenate((left, fluid_left))
    combined_right = np.concatenate((right, fluid_right))
    combined_boundary = np.concatenate((sediment_boundary, fluid_boundary))
    if (scalar_trace.fraction_count != ng or
            scalar_trace.active_fluid_lanes != tuple(map(int, active_lanes)) or
            scalar_trace.substep_concentration.shape !=
                (2, ng+fluid_count, len(scalar_trace.interface_faces))):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR combined scalar trace dimensions differ")
    fct_result = fct_runtime.evaluate(
        base, combined_mass, h0, h1, water, combined_left,
        combined_right, combined_boundary, constant,
        interface_faces=np.asarray(scalar_trace.interface_faces),
        interface_concentration=
            scalar_trace.substep_concentration[substep_index],
        dt_s=dt, native_threads=native_threads, cancelled=cancelled)
    fct_runtime.validate_result(
        fct_result, base, combined_mass, h0, h1, water, combined_left,
        combined_right, combined_boundary, constant,
        interface_faces=np.asarray(scalar_trace.interface_faces),
        interface_concentration=
            scalar_trace.substep_concentration[substep_index],
        dt_s=dt, native_threads=native_threads)
    combined_advected = np.asarray(fct_result["mass_kg_m2"])
    combined_flux = np.asarray(
        fct_result["advective_face_flux_kg_m_s"])
    combined_advective_ledger = np.asarray(
        fct_result["advective_boundary_ledger_kg"])
    if (native_boundary_result is not None and not np.allclose(
            combined_advective_ledger,
            native_boundary_result.expected_advective_boundary_ledger,
            rtol=32.0*np.finfo(float).eps, atol=2.0e-18)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v97 native boundary and FCT advective ledgers differ")

    if (scalar_trace.internal_trace_receipt_sha256 !=
            internal_trace.receipt_sha256 or
            turbulence_trace.base_trace_receipt_sha256 !=
                internal_trace.receipt_sha256 or
            turbulence_trace.fine_execution_fingerprint !=
                scalar_trace.fine_execution_fingerprint or
            tuple(internal_trace.fine_faces) !=
                tuple(scalar_trace.interface_faces) or
            tuple(internal_trace.fine_faces) !=
                tuple(turbulence_trace.fine_faces)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR scalar/diffusion trace identity differs")
    diffusion_result = diffusion_runtime.evaluate(
        base, scalar_execution.dicuv_cell_m2_s, combined_advected, h1,
        combined_boundary, face_area, face_active,
        dt_s=dt,
        trace_faces=np.asarray(internal_trace.fine_faces, dtype=np.int32),
        trace_depth=internal_trace.trace_q[substep_index, 0],
        trace_concentration=
            scalar_trace.substep_concentration[substep_index],
        trace_dicuv=
            turbulence_trace.trace_capped_dicuv_m2_s[substep_index],
        trace_wet_mask=internal_trace.trace_wet_mask[substep_index],
        trace_receipt_sha256=scalar_trace.receipt_sha256,
        trace_source_sha256=
            internal_trace.coarse_provisional_state_sha256,
        trace_runtime_sha256=
            diffusion_runtime.receipt["receipt_sha256"],
        cap_receipt_sha256=fine_cap_receipt.receipt_sha256,
        trace_cap_receipt_sha256=
            turbulence_trace.coarse_cap_receipt_sha256,
        evaluation_index=1, native_threads=native_threads,
        cancelled=cancelled)
    diffusion_expected_keys = {
        "mass_kg_m2", "diffusive_face_mass_owner_out_kg",
        "diffusive_boundary_ledger_kg", "effective_diffusivity_m2_s",
        "internal_interface_mass_in_kg", "trace_receipt_sha256",
        "trace_provenance", "trace_inputs_sha256",
        "physical_boundary_surrogate_count",
        "turbulence_cap_evaluation_count",
        "native_runtime_sha256", "native_threads",
        "scalar_diffusion_owner_count",
        "python_numerical_cell_face_loop_count",
        "state_or_clock_publication_count", "native_diffusion_launch_count",
    }
    if (not isinstance(diffusion_result, Mapping) or
            set(diffusion_result) != diffusion_expected_keys or
            diffusion_result["native_runtime_sha256"] !=
                diffusion_runtime.receipt.get("receipt_sha256") or
            diffusion_result["native_threads"] != native_threads or
            diffusion_result["trace_receipt_sha256"] !=
                scalar_trace.receipt_sha256 or
            diffusion_result["physical_boundary_surrogate_count"] != 0 or
            diffusion_result["turbulence_cap_evaluation_count"] != 0 or
            diffusion_result["scalar_diffusion_owner_count"] != 1 or
            diffusion_result["python_numerical_cell_face_loop_count"] != 0 or
            diffusion_result["state_or_clock_publication_count"] != 0 or
            diffusion_result["native_diffusion_launch_count"] != 1):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native diffusion result identity differs")
    combined_diffused = np.asarray(diffusion_result["mass_kg_m2"])
    combined_face_mass = np.asarray(
        diffusion_result["diffusive_face_mass_owner_out_kg"])
    combined_diffusive_ledger = np.asarray(
        diffusion_result["diffusive_boundary_ledger_kg"])
    effective = np.asarray(diffusion_result["effective_diffusivity_m2_s"])
    expected_shapes = (
        (combined_diffused, (ng+fluid_count, nc)),
        (combined_face_mass, (ng+fluid_count, nf)),
        (combined_diffusive_ledger, (ng+fluid_count, nb)),
        (effective, (nf,)),
    )
    if any(array.shape != shape or not np.isfinite(array).all()
           for array, shape in expected_shapes):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native diffusion output shape differs")

    diffused = combined_diffused[:ng]
    advective_flux = combined_flux[:ng]
    face_mass = combined_face_mass[:ng]
    advective_ledger = combined_advective_ledger[:ng]
    diffusive_ledger = combined_diffusive_ledger[:ng]
    fluid_diffused = combined_diffused[ng:]
    fluid_flux = combined_flux[ng:]
    fluid_face_mass = combined_face_mass[ng:]
    fluid_advective = combined_advective_ledger[ng:]
    fluid_diffusive = combined_diffusive_ledger[ng:]

    wet1 = h1 > base.dry_tolerance_m
    speed = np.zeros(nc)
    speed[wet1] = np.hypot(
        endpoint_momentum_m2_s[0, wet1],
        endpoint_momentum_m2_s[1, wet1])/h1[wet1]
    exchange_kwargs = dict(
        dt_s=dt, fac_tsd=base.profile.fac_tsd,
        sediment_threshold_m=base.profile.sediment_threshold_m,
        minimum_reference_depth_m=base.profile.reference_depth_floor_m,
        flow_wet_mask=wet1)
    native_exchange_result = None
    if settling_exchange_runtime is None:
        exchange = _v92.advance_flow2dh_equilibrium_exchange(
            diffused, old.bed_mass_kg_m2, capacity, surface, speed, shear,
            h0, h1, settling.settling_velocity_m_s.T, base.cell_area_m2,
            **exchange_kwargs)
        cleanup = np.where(
            wet1[None, :], 0.0, exchange.suspended_mass_kg_m2)
        final_suspended = exchange.suspended_mass_kg_m2-cleanup
        final_bed = exchange.bed_mass_kg_m2+cleanup
        cleanup_water = -np.sum(
            cleanup*base.cell_area_m2[None, :], axis=1)
        cleanup_bed = -cleanup_water
    else:
        native_exchange_result = settling_exchange_runtime.evaluate(
            diffused, old.bed_mass_kg_m2, capacity, surface, speed, shear,
            h0, h1, settling.settling_velocity_m_s.T, base.cell_area_m2,
            native_threads=native_threads, cancelled=cancelled,
            **exchange_kwargs)
        exchange = native_exchange_result.exchange
        cleanup = native_exchange_result.wetdry_transfer_to_bed_kg_m2
        final_suspended = native_exchange_result.final_suspended_mass_kg_m2
        final_bed = native_exchange_result.final_bed_mass_kg_m2
        cleanup_water = native_exchange_result.wetdry_water_ledger_kg
        cleanup_bed = native_exchange_result.wetdry_bed_ledger_kg
    # A nonzero coarse/fine interface transfer is not a named physical
    # boundary ledger and therefore cannot be published through the frozen
    # v76/v78 state schema.  Stop at unpublished numerical images here.  The
    # additive v98 fine owner applies this stage once, records its internal
    # ledger, and rebases a private numerical parent with a fresh local legacy
    # reference window; it never relabels this result as an accepted v92 trial.

    fluid_cleanup = np.where(wet1[None, :], 0.0, fluid_diffused)
    final_fluid = fluid_diffused-fluid_cleanup
    ht1 = h1*execution.equilibrium_temperature_c
    hs1 = h1*execution.equilibrium_salinity_ppt
    temperature = np.full(nc, execution.equilibrium_temperature_c)
    salinity = np.full(nc, execution.equilibrium_salinity_ppt)
    if active[_v92.FLUID_TEMPERATURE]:
        position = lane_position[_v92.FLUID_TEMPERATURE]
        ht1 = final_fluid[position]+_v92.TEMPERATURE_FLOOR_C*h1
        temperature[wet1] = ht1[wet1]/h1[wet1]
    if active[_v92.FLUID_SALINITY]:
        position = lane_position[_v92.FLUID_SALINITY]
        hs1 = final_fluid[position]
        salinity[wet1] = hs1[wet1]/h1[wet1]
    # The affine-temperature cleanup ledger below removes both transported
    # affine mass and the T_floor*h offset.  Publish exact-zero hT/hS on the
    # accepted dry set so v97 has one physical scalar state rather than a
    # residual dry-cell shadow inventory.
    ht1 = np.where(wet1, ht1, 0.0)
    hs1 = np.where(wet1, hs1, 0.0)
    if (np.any(temperature[wet1] <= _v92.TEMPERATURE_FLOOR_C) or
            np.any(salinity[wet1] < -8192.0*np.finfo(float).eps)):
        raise _v92.TransportedFluidSettlingV92Error(
            "v96 native scalar pass produced invalid endpoint T/S")
    salinity = np.maximum(salinity, 0.0)
    physical_flux = np.zeros((2, nf))
    physical_face_mass = np.zeros((2, nf))
    physical_advective = np.zeros((2, nb))
    physical_diffusive = np.zeros((2, nb))
    physical_cleanup = np.zeros(2)
    boundary_physical = np.vstack((
        np.full(nf, execution.equilibrium_temperature_c),
        np.full(nf, execution.equilibrium_salinity_ppt)))
    lane_application_count = np.zeros(2, dtype=np.int32)
    if native_boundary_result is None:
        water_boundary = np.zeros(nb)
        for face in range(nf):
            boundary_id = int(base.boundary_index[face])
            if boundary_id >= 0:
                water_boundary[boundary_id] -= (
                    dt*base.face_length_m[face]*water[face])
    else:
        water_boundary = np.asarray(
            native_boundary_result.water_boundary_ledger_m3)
    for compact, lane in enumerate(active_lanes):
        physical_flux[lane] = fluid_flux[compact]
        physical_face_mass[lane] = fluid_face_mass[compact]
        physical_advective[lane] = fluid_advective[compact]
        physical_diffusive[lane] = fluid_diffusive[compact]
        physical_cleanup[lane] = -np.sum(
            fluid_cleanup[compact]*base.cell_area_m2)
        boundary_physical[lane] = fluid_boundary[compact]
        lane_application_count[lane] = 1
        if lane == _v92.FLUID_TEMPERATURE:
            physical_flux[lane] += _v92.TEMPERATURE_FLOOR_C*water
            physical_advective[lane] += \
                _v92.TEMPERATURE_FLOOR_C*water_boundary
            boundary_physical[lane] += _v92.TEMPERATURE_FLOOR_C
            physical_cleanup[lane] -= np.sum(
                _v92.TEMPERATURE_FLOOR_C*np.where(wet1, 0.0, h1)*
                base.cell_area_m2)
    fluid_arrays = [np.ascontiguousarray(item) for item in (
        ht1, hs1, temperature, salinity, boundary_physical,
        next_fluid_last, next_fluid_clock, physical_flux,
        physical_face_mass, physical_advective, physical_diffusive,
        physical_cleanup, effective, continuity, lane_application_count)]
    for item in fluid_arrays:
        item.setflags(write=False)
    fluid_placeholder = _v92.FluidScalarTransportResultV92(*fluid_arrays, "")
    fluid_result = _v92.FluidScalarTransportResultV92(
        **{**fluid_placeholder.__dict__,
           "result_sha256": _v92._fluid_result_token(fluid_placeholder)})

    diffusion_runtime_sha = str(
        diffusion_runtime.receipt["receipt_sha256"])
    diffusion_result_sha = _diffusion_result_token(
        diffusion_runtime_sha, native_threads,
        input_arrays=(
            np.asarray(scalar_execution.dicuv_cell_m2_s),
            np.asarray(combined_advected), h1,
            np.asarray(combined_boundary), face_area,
            np.asarray(face_active, dtype=np.int32)),
        output_arrays=(
            combined_diffused, combined_face_mass,
            combined_diffusive_ledger, effective))
    selected_interface_faces = np.asarray(
        scalar_trace.interface_faces, dtype=np.int64)
    interface_ledger = np.ascontiguousarray(
        fct_result["internal_interface_owner_ledger_kg"]-
        combined_face_mass[:, selected_interface_faces])
    if not np.allclose(
            np.sum(interface_ledger, axis=1),
            np.sum(fct_result["internal_interface_owner_ledger_kg"],
                   axis=1)+
            np.asarray(diffusion_result["internal_interface_mass_in_kg"]),
            rtol=0.0, atol=8.0*np.finfo(float).eps):
        raise _v92.TransportedFluidSettlingV92Error(
            "v98 AMR scalar interface advection/diffusion ledger differs")
    if suspended_bed_execution is None:
        raise _v92.TransportedFluidSettlingV92Error(
            "v100 fine AMR requires the selected suspended-bed execution")
    selected_numerical, selector_runtime_sha = native_amr_bed_selection_v100(
        suspended_bed_execution, old.bed_mass_kg_m2,
        old.suspended_mass_kg_m2, diffused,
        exchange.transferred_to_suspension_kg_m2, cleanup,
        native_threads=native_threads)
    selection = wrap_amr_bed_selection_v100(
        suspended_bed_execution, selected_numerical, area=base.cell_area_m2,
        advective_boundary=advective_ledger, diffusive_boundary=diffusive_ledger,
        internal_ledger=interface_ledger, backend="native_cpu_openmp",
        runtime_sha256=selector_runtime_sha, native_threads=native_threads)
    final_bed = selection.selected_bed_mass_before_q3_kg_m2
    raw_arrays = [np.ascontiguousarray(value) for value in (
        final_suspended, final_bed, next_sediment_last,
        next_sediment_clock, advective_ledger, diffusive_ledger,
        exchange.suspended_exchange_ledger_kg,
        selection.selected_exchange_ledger_kg, cleanup_water, cleanup_bed,
        advective_flux, face_mass, settling.settling_velocity_m_s.T,
        exchange.equilibrium_concentration_kg_m3,
        exchange.transferred_to_suspension_kg_m2, cleanup, effective,
        parent_ctot, parent.water_density_cell_kg_m3,
        parent.molecular_viscosity_cell_m2_s, interface_ledger)]
    for array in raw_arrays:
        array.setflags(write=False)
    exchange_runtime_sha = (
        None if native_exchange_result is None else
        native_exchange_result.native_runtime_sha256)
    boundary_runtime_sha = (
        None if native_boundary_result is None else
        native_boundary_result.runtime_sha256)
    placeholder = AMRScalarStageResultV98(
        *raw_arrays[:17], fluid_result, *raw_arrays[17:20],
        capacity_receipt, scalar_trace.receipt_sha256, substep_index,
        raw_arrays[20],
        tuple(map(int, fct_result["internal_interface_faces"])),
        fct_runtime.runtime_sha256, str(fct_result["result_sha256"]),
        diffusion_runtime_sha, exchange_runtime_sha,
        boundary_runtime_sha, native_threads, 1,
        int(fct_result["internal_interface_owner_count"]),
        int(fct_result["physical_boundary_surrogate_count"]), 0, "", selection)
    result = AMRScalarStageResultV98(
        **{**placeholder.__dict__,
           "result_sha256": _raw_result_token(placeholder)})
    if cancelled():
        raise InterruptedError(
            "v98 AMR scalar cancelled before unpublished result return")
    return result


__all__ = [
    "AMRScalarStageResultV98", "SCOPE_ID",
    "suspended_dynamic_trial_amr_scalar_cpu_v98",
    "validate_amr_scalar_stage_result_v98",
]
