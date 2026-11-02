"""Actual-CUDA moving-wet/dry midpoint owner for the bounded Q4/v97 gate.

This is an additive controller mode.  It does not change the historical
fully-wet :mod:`implicit_midpoint_fwave` controller.  The accepted-parent wet
set is frozen only for one inner nonlinear solve, then rebuilt from the
candidate depth; a changed set restarts the inner solve.  The existing
augmented f-wave face pipeline and its donor-draining limiter remain the sole
hydraulic flux owner.

The object is intentionally a provisional hydraulic owner: it publishes no
clock or model state.  A later v97 joint owner consumes its authenticated
endpoint and face flux before the v92/v94b scalar/Exner transaction commits.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import types
from typing import Callable, Mapping

import numpy as np

from braids_core.sediment.densin_fwave_v85 import (
    DensInExecutionV85,
    execution_document_v85,
    verify_locked_sources_v85,
)
from braids_core.sediment.density_active_projection_v97 import (
    POLICY as DENSITY_PROJECTION_POLICY_V97,
    RANK_TOL,
    RESIDUAL_TOL,
    resolve_density_projection_geometry_v97,
)
from braids_core.sediment.shared_turbulence_consumers_v91 import (
    STATE_DOMAIN_MOVING_WETDRY_V97,
    ChkvicUsedTotalReceiptV91,
    _cap_token,
    execution_document_v91,
)
from .density_active_projection_v97 import ResidentDensityActiveProjectionV97
from .implicit_midpoint_fwave import (
    CUDA_OPTIONS,
    ImplicitMidpointController,
    ImplicitStepCancelled,
    ImplicitStepRejected,
)


ROOT = Path(__file__).resolve().parents[1]
DENSIN_SOURCE = Path(__file__).with_name("densin_fwave_v85.cu")
CONTROLLER_SOURCE = Path(__file__).with_name("implicit_midpoint_fwave.cu")
SCOPE_ID = "braids-q4-moving-wetdry-actual-cuda-v97-v1"
SEMANTIC_ID = SCOPE_ID + ":active-set-draining-augmented-fwave"
MODE_WATER_EOS = "water_eos"
MODE_WATER_PLUS_SEDIMENT = "water_plus_sediment"
MAX_ACTIVE_SET_RESTARTS = 6
MAX_INNER_ITERATIONS = 80


class WetDryFwaveGPUV97Error(ValueError):
    """The v97 execution, resident runtime, or provisional trial differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA identity is not finite canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_name(path: str | Path) -> str:
    """Return stable repo-relative provenance, or an explicit external path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if type(value) is bool:
        return value
    if callable(value):
        return bool(value())
    raise WetDryFwaveGPUV97Error(
        "v97 CUDA cancellation must be bool or callable")


def _frozen(value: object, shape: tuple[int, ...], name: str, *,
            integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiub" or
            not np.isfinite(raw).all()):
        raise WetDryFwaveGPUV97Error(
            f"v97 CUDA {name} shape/type/finite identity differs")
    if integer and (np.any(raw != np.floor(raw)) or
                    np.any(np.abs(raw) > np.iinfo(np.int32).max)):
        raise WetDryFwaveGPUV97Error(f"v97 CUDA {name} must be int32")
    result = np.array(raw, dtype=np.int32 if integer else np.float64,
                      order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WetDryFwaveGPUExecutionV97:
    density_execution: DensInExecutionV85
    domain_active_mask: np.ndarray
    density_mode: str
    shared_turbulence_enabled: bool
    shared_turbulence_execution_fingerprint: str | None
    controller_source_sha256: str
    max_active_set_restarts: int
    max_inner_iterations: int
    nonlinear_tolerance: float
    positivity_fraction: float
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.density_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.density_execution.face_count


def _execution_payload(value: WetDryFwaveGPUExecutionV97) -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": "actual_cuda",
        "integrator": "implicit_fwave_midpoint",
        "density_execution_fingerprint": value.density_execution.fingerprint,
        "domain_active_sha256": _digest(value.domain_active_mask),
        "density_mode": value.density_mode,
        "shared_turbulence_enabled": value.shared_turbulence_enabled,
        "shared_turbulence_execution_fingerprint":
            value.shared_turbulence_execution_fingerprint,
        "controller_source_sha256": value.controller_source_sha256,
        "max_active_set_restarts": value.max_active_set_restarts,
        "max_inner_iterations": value.max_inner_iterations,
        "nonlinear_tolerance": value.nonlinear_tolerance,
        "positivity_fraction": value.positivity_fraction,
        "wet_mask": "domain_active_and_depth_strictly_above_dry_tolerance",
        "inner_mask_policy": "frozen_per_active_set_pass",
        "spatial_flux": "existing_augmented_fwave_face_pipeline",
        "draining": "existing_resident_donor_limiter",
        "density_projection": DENSITY_PROJECTION_POLICY_V97,
        "nonlinear_solver": "damped_picard_true_residual",
        "limiter_derivative_evaluation_count": 0,
        "mask_derivative_evaluation_count": 0,
        "automatic_explicit_fallback": False,
        "automatic_cpu_fallback": False,
        "state_or_clock_publication_count": 0,
        "production_admission": False,
    }


def resolve_wetdry_fwave_gpu_execution_v97(
    controller: ImplicitMidpointController,
    density_execution: DensInExecutionV85, *,
    domain_active_mask: object,
    density_mode: str = MODE_WATER_PLUS_SEDIMENT,
    shared_turbulence_execution=None,
    max_active_set_restarts: int = MAX_ACTIVE_SET_RESTARTS,
    max_inner_iterations: int = MAX_INNER_ITERATIONS,
    nonlinear_tolerance: float = 2.0e-10,
    positivity_fraction: float = 0.9,
) -> WetDryFwaveGPUExecutionV97:
    if type(controller) is not ImplicitMidpointController:
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA requires the exact historical midpoint controller")
    execution_document_v85(density_execution)
    if density_execution.role != "implicit_fwave_midpoint_gpu":
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA density execution backend differs")
    nc = density_execution.cell_count
    active = _frozen(domain_active_mask, (nc,), "domain-active mask",
                     integer=True)
    if np.any((active != 0) & (active != 1)) or not np.any(active):
        raise WetDryFwaveGPUV97Error("v97 CUDA domain-active mask is invalid")
    # The first bounded fixed-grid owner has no inactive holes.  Keeping this
    # separate authenticated field prevents conflating it with the wet mask.
    if np.any(active == 0):
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA bounded topology does not yet admit inactive holes")
    if density_mode not in (MODE_WATER_EOS, MODE_WATER_PLUS_SEDIMENT):
        raise WetDryFwaveGPUV97Error("v97 CUDA density mode differs")
    if shared_turbulence_execution is None:
        shared_turbulence_enabled = False
        shared_turbulence_fingerprint = None
    else:
        document = execution_document_v91(shared_turbulence_execution)
        shared_turbulence_enabled = True
        shared_turbulence_fingerprint = str(document["fingerprint"])
        grid = shared_turbulence_execution.base_execution.base_execution
        if (shared_turbulence_execution.state_domain_mode !=
                STATE_DOMAIN_MOVING_WETDRY_V97 or
                shared_turbulence_execution.backend != "implicit_fwave_midpoint_gpu" or
                grid.cell_count != density_execution.cell_count or
                grid.face_count != density_execution.face_count or
                not np.array_equal(grid.owner, density_execution.owner) or
                not np.array_equal(grid.neighbour, density_execution.neighbour) or
                not np.array_equal(grid.normal, density_execution.normal) or
                not np.array_equal(grid.face_length_m,
                                   density_execution.face_length_m) or
                not np.array_equal(grid.centre_distance_m,
                                   density_execution.face_distance_m) or
                not np.array_equal(grid.cell_area_m2,
                                   density_execution.cell_area_m2) or
                not np.array_equal(grid.cell_face_offset,
                                   density_execution.cell_face_offset) or
                not np.array_equal(grid.cell_faces,
                                   density_execution.cell_faces) or
                not np.array_equal(grid.cell_face_sign,
                                   density_execution.cell_face_sign)):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA shared-turbulence topology/backend differs")
    if (type(max_active_set_restarts) is not int or
            not 0 <= max_active_set_restarts <= 16 or
            type(max_inner_iterations) is not int or
            not 1 <= max_inner_iterations <= 200):
        raise WetDryFwaveGPUV97Error("v97 CUDA iteration bounds differ")
    if (isinstance(nonlinear_tolerance, bool) or
            not math.isfinite(float(nonlinear_tolerance)) or
            not 64*np.finfo(float).eps <= float(nonlinear_tolerance) <= 1e-5 or
            isinstance(positivity_fraction, bool) or
            not math.isfinite(float(positivity_fraction)) or
            not .5 <= float(positivity_fraction) < 1.0):
        raise WetDryFwaveGPUV97Error("v97 CUDA tolerance/damping differs")
    if (controller.nc != nc or controller.nf != density_execution.face_count or
            _file_sha(controller.source_path) != controller.source_sha256 or
            controller.source_sha256 != _file_sha(CONTROLLER_SOURCE)):
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA controller size/source identity differs")
    placeholder = WetDryFwaveGPUExecutionV97(
        density_execution, active, density_mode, shared_turbulence_enabled,
        shared_turbulence_fingerprint, controller.source_sha256,
        max_active_set_restarts,
        max_inner_iterations, float(nonlinear_tolerance),
        float(positivity_fraction), "")
    return WetDryFwaveGPUExecutionV97(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_execution_payload(placeholder))})


def execution_document_wetdry_fwave_gpu_v97(
    value: WetDryFwaveGPUExecutionV97,
) -> dict[str, object]:
    if type(value) is not WetDryFwaveGPUExecutionV97:
        raise WetDryFwaveGPUV97Error("v97 CUDA execution type differs")
    execution_document_v85(value.density_execution)
    payload = _execution_payload(value)
    if (value.domain_active_mask.shape != (value.cell_count,) or
            value.domain_active_mask.dtype != np.int32 or
            value.domain_active_mask.flags.writeable or
            np.any(value.domain_active_mask != 1) or
            value.shared_turbulence_enabled !=
                (value.shared_turbulence_execution_fingerprint is not None) or
            (value.shared_turbulence_execution_fingerprint is not None and
             (len(value.shared_turbulence_execution_fingerprint) != 64 or
              any(character not in "0123456789abcdef" for character in
                  value.shared_turbulence_execution_fingerprint))) or
            value.fingerprint != _hash(payload)):
        raise WetDryFwaveGPUV97Error("v97 CUDA execution identity differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class WetDryFwaveGPUTrialV97:
    execution_fingerprint: str
    parent_q_sha256: str
    accepted_bed_sha256: str
    accepted_water_density_sha256: str
    accepted_mixture_density_sha256: str
    dt_s: float
    evaluation_time_s: float
    parent_q: np.ndarray
    midpoint_q: np.ndarray
    endpoint_q: np.ndarray
    frozen_wet_mask: np.ndarray
    accepted_wet_mask: np.ndarray
    active_wet_wet_face_mask: np.ndarray
    accepted_limited_face_water_flux_m2_s: np.ndarray
    accepted_draining_theta: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    shared_turbulence_rate: np.ndarray
    shared_turbulence_face_flux: np.ndarray
    accepted_total_rate: np.ndarray
    dry_momentum_cleanup_rate_m2_s2: np.ndarray
    density_numerical_rank: np.ndarray
    density_projection_residual: np.ndarray
    shared_turbulence_cap_receipt_sha256: str | None
    active_set_restart_count: int
    inner_iteration_count: int
    true_residual_evaluation_count: int
    damping_application_count: int
    draining_limiter_application_count: int
    augmented_riemann_call_count: int
    density_face_evaluation_count: int
    density_projection_application_count: int
    shared_turbulence_application_count: int
    limiter_derivative_evaluation_count: int
    mask_derivative_evaluation_count: int
    dry_momentum_cleanup_cell_count: int
    dry_momentum_cleanup_linf_m2_s2: float
    projected_endpoint_identity_linf: float
    state_or_clock_publication_count: int
    water_volume_residual_m3: float
    nonlinear_residual_linf: float
    backend: str
    trial_sha256: str


def _trial_payload(value: WetDryFwaveGPUTrialV97) -> dict:
    arrays = (
        value.parent_q, value.midpoint_q, value.endpoint_q,
        value.frozen_wet_mask, value.accepted_wet_mask,
        value.active_wet_wet_face_mask,
        value.accepted_limited_face_water_flux_m2_s,
        value.accepted_draining_theta, value.base_fwave_rate,
        value.density_momentum_rate, value.shared_turbulence_rate,
        value.shared_turbulence_face_flux,
        value.accepted_total_rate,
        value.dry_momentum_cleanup_rate_m2_s2,
        value.density_numerical_rank,
        value.density_projection_residual,
    )
    return {
        "scope_id": SCOPE_ID + ":trial",
        "execution_fingerprint": value.execution_fingerprint,
        "parent_q_sha256": value.parent_q_sha256,
        "accepted_bed_sha256": value.accepted_bed_sha256,
        "accepted_water_density_sha256":
            value.accepted_water_density_sha256,
        "accepted_mixture_density_sha256":
            value.accepted_mixture_density_sha256,
        "dt_s": value.dt_s,
        "evaluation_time_s": value.evaluation_time_s,
        "arrays_sha256": _digest(*arrays),
        "shared_turbulence_cap_receipt_sha256":
            value.shared_turbulence_cap_receipt_sha256,
        "active_set_restart_count": value.active_set_restart_count,
        "inner_iteration_count": value.inner_iteration_count,
        "true_residual_evaluation_count":
            value.true_residual_evaluation_count,
        "damping_application_count": value.damping_application_count,
        "draining_limiter_application_count":
            value.draining_limiter_application_count,
        "augmented_riemann_call_count": value.augmented_riemann_call_count,
        "density_face_evaluation_count": value.density_face_evaluation_count,
        "density_projection_application_count":
            value.density_projection_application_count,
        "shared_turbulence_application_count":
            value.shared_turbulence_application_count,
        "limiter_derivative_evaluation_count":
            value.limiter_derivative_evaluation_count,
        "mask_derivative_evaluation_count":
            value.mask_derivative_evaluation_count,
        "dry_momentum_cleanup_cell_count":
            value.dry_momentum_cleanup_cell_count,
        "dry_momentum_cleanup_linf_m2_s2":
            value.dry_momentum_cleanup_linf_m2_s2,
        "projected_endpoint_identity_linf":
            value.projected_endpoint_identity_linf,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "backend": value.backend,
    }


class ResidentWetDryFwaveV97:
    """Provisional actual-CUDA active-set midpoint solve, no publication."""

    def __init__(
        self, execution: WetDryFwaveGPUExecutionV97,
        controller: ImplicitMidpointController, *,
        density_projection: ResidentDensityActiveProjectionV97,
        shared_turbulence=None,
        shared_turbulence_cap_receipt: ChkvicUsedTotalReceiptV91 | None = None,
    ) -> None:
        execution_document_wetdry_fwave_gpu_v97(execution)
        if type(controller) is not ImplicitMidpointController:
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA controller concrete type differs")
        if (controller.source_sha256 != execution.controller_source_sha256 or
                controller.nc != execution.cell_count or
                controller.nf != execution.face_count):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA controller binding differs")
        if (controller.shared_turbulence_v91 is not None or
                controller.densin_residual_v95 is not None):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA requires sole composite residual ownership")
        if (type(density_projection) is not ResidentDensityActiveProjectionV97 or
                density_projection.geometry.cell_count != execution.cell_count or
                density_projection.geometry.face_count != execution.face_count):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA density projection binding differs")
        if execution.shared_turbulence_enabled != (shared_turbulence is not None):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA shared-turbulence binding differs")
        if (shared_turbulence is not None and
                getattr(getattr(shared_turbulence, "execution", None),
                        "fingerprint", None) !=
                execution.shared_turbulence_execution_fingerprint):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA shared-turbulence execution differs")
        if execution.shared_turbulence_enabled:
            if (type(shared_turbulence_cap_receipt) is not
                    ChkvicUsedTotalReceiptV91 or
                    shared_turbulence_cap_receipt.execution_fingerprint !=
                        execution.shared_turbulence_execution_fingerprint or
                    shared_turbulence_cap_receipt.receipt_sha256 !=
                        _cap_token(shared_turbulence_cap_receipt)):
                raise WetDryFwaveGPUV97Error(
                    "v97 CUDA shared-turbulence cap receipt differs")
        elif shared_turbulence_cap_receipt is not None:
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA unselected turbulence received a cap receipt")
        import cupy as cp
        self.cp, self.execution, self.controller = cp, execution, controller
        self.projection, self.turbulence = density_projection, shared_turbulence
        self.turbulence_cap = shared_turbulence_cap_receipt
        self.device_id = int(cp.cuda.Device().id)
        self.nc, self.nf = execution.cell_count, execution.face_count
        base = execution.density_execution
        host_geometry = (
            base.owner, base.neighbour, base.normal, base.face_length_m,
            base.face_distance_m, base.cell_area_m2,
            base.cell_face_offset, base.cell_faces, base.cell_face_sign,
        )
        actual_geometry = (
            cp.asnumpy(controller.owner), cp.asnumpy(controller.neighbour),
            cp.asnumpy(controller.normal), cp.asnumpy(controller.length),
            base.face_distance_m, cp.asnumpy(controller.area),
            cp.asnumpy(controller.offset), cp.asnumpy(controller.faces),
            cp.asnumpy(controller.signs),
        )
        if any(not np.array_equal(a, b)
               for a, b in zip(actual_geometry, host_geometry)):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA hydraulic/density topology differs")
        self.domain_active = cp.asarray(execution.domain_active_mask)
        self.owner = cp.asarray(base.owner)
        self.neighbour = cp.asarray(base.neighbour)
        self.distance = cp.asarray(base.face_distance_m)
        self.masked_q = cp.empty((3, self.nc), dtype=cp.float64)
        self.wet = cp.empty(self.nc, dtype=cp.int32)
        self.face_active = cp.empty(self.nf, dtype=cp.int32)
        self.face_acceleration = cp.empty(self.nf, dtype=cp.float64)
        self.face_status = cp.empty(self.nf, dtype=cp.int32)
        self.density_cell_acceleration = cp.empty((2, self.nc), cp.float64)
        self.density_rate = cp.empty((3, self.nc), cp.float64)
        self.density_rank = cp.empty(self.nc, cp.int32)
        self.density_residual = cp.empty(self.nc, cp.float64)
        self.density_status = cp.empty(self.nc, cp.int32)
        self.base_rate = cp.empty((3, self.nc), cp.float64)
        self.total_rate = cp.empty((3, self.nc), cp.float64)
        self.turbulence_rate = cp.zeros((3, self.nc), cp.float64)
        self.turbulence_face_flux = cp.zeros((2, self.nf), cp.float64)
        self.capped_vicuv = (cp.zeros(self.nc, cp.float64)
            if shared_turbulence_cap_receipt is None else
            cp.ascontiguousarray(cp.asarray(
                shared_turbulence_cap_receipt.capped_total_vicuv_m2_s,
                dtype=cp.float64)))
        self.final_water_flux = cp.empty(self.nf, cp.float64)
        self.final_theta = cp.empty(self.nc, cp.float64)
        # Additive v98 AMR evidence.  These buffers retain the exact left and
        # right augmented-f-wave side images from the final true residual.
        # They are not an AMR register and never participate in the v97 state
        # token.  A v98 extractor may consume them only while their bound
        # trial SHA below matches the accepted provisional v97 trial.
        self.v98_final_augmented_left_flux = cp.zeros(
            (3, self.nf), cp.float64)
        self.v98_final_augmented_right_flux = cp.zeros(
            (3, self.nf), cp.float64)
        self._v98_hydraulic_trial_sha256 = None
        self._v98_hydraulic_side_evaluation_count = 0
        # Optional additive v98 internal coarse/fine trace owner.  The
        # historical v97 path keeps this exact None and therefore retains
        # its original numerical route and trial identity.
        self._amr_interface_v98 = None
        source = DENSIN_SOURCE.read_text(encoding="utf-8")
        self.densin_module = cp.RawModule(
            code=source, options=CUDA_OPTIONS,
            name_expressions=("densin_face_v85",))
        self.densin_face_kernel = self.densin_module.get_function(
            "densin_face_v85")
        sources = {
            _source_name(path): _file_sha(path)
            for path in (
                Path(__file__), DENSIN_SOURCE,
                Path(controller.source_path),
                Path(__file__).with_name("density_active_projection_v97.cu"),
                Path(__file__).with_name("density_active_projection_v97.py"),
            )
        }
        self.runtime_receipt = {
            "scope_id": SCOPE_ID + ":runtime",
            "execution_fingerprint": execution.fingerprint,
            "backend": "actual_cuda",
            "actual_cuda_kernels": True,
            "automatic_cpu_fallback": False,
            "resident_nonlinear_arrays": True,
            "state_or_clock_publication_count": 0,
            "source_sha256": sources,
            "locked_delft_source_sha256": verify_locked_sources_v85(ROOT),
            "density_projection_runtime_sha256":
                density_projection.receipt["receipt_sha256"],
            "shared_turbulence_runtime_sha256": (
                None if shared_turbulence is None else
                shared_turbulence.receipt["receipt_sha256"]),
            "device_id": self.device_id,
            "production_admission": False,
        }
        self.runtime_receipt["receipt_sha256"] = _hash(self.runtime_receipt)
        self.last_trial = None

    def bind_shared_turbulence_cap(
        self, receipt: ChkvicUsedTotalReceiptV91,
    ) -> None:
        """Bind the accepted-parent cap for the next provisional solve.

        The v90/HLES state advances only after the complete outer v97
        transaction commits.  Consequently each new hydraulic trial has a
        new authenticated cap even though geometry and CUDA kernels remain
        resident.  Rebinding performs no numerical work and is forbidden for
        executions without the v91 branch.
        """
        if (not self.execution.shared_turbulence_enabled or
                type(receipt) is not ChkvicUsedTotalReceiptV91 or
                receipt.execution_fingerprint !=
                    self.execution.shared_turbulence_execution_fingerprint or
                receipt.receipt_sha256 != _cap_token(receipt)):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA shared-turbulence cap rebind differs")
        values = self.cp.ascontiguousarray(self.cp.asarray(
            receipt.capped_total_vicuv_m2_s, dtype=self.cp.float64))
        if (values.shape != (self.nc,) or
                not bool(self.cp.isfinite(values).all().item()) or
                bool(self.cp.any(values < 0.0).item())):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA rebound capped vicuv differs")
        self.turbulence_cap = receipt
        self.capped_vicuv = values

    def invalidate_v98_hydraulic_side_evidence(self) -> None:
        """Clear additive AMR evidence without touching accepted v97 state."""
        self._v98_hydraulic_trial_sha256 = None
        self._v98_hydraulic_side_evaluation_count = 0
        self.v98_final_augmented_left_flux.fill(0.0)
        self.v98_final_augmented_right_flux.fill(0.0)

    def _bind_amr_interface_v98(self, runtime) -> None:
        """Bind one exact additive v98 trace owner for the next solve."""
        from .coupled_amr_fwave_v98 import ResidentAMRInterfaceFwaveGPUV98

        if (type(runtime) is not ResidentAMRInterfaceFwaveGPUV98 or
                runtime.hydraulic_runtime is not self or
                self._amr_interface_v98 is not None):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA v98 interface owner binding differs")
        self._amr_interface_v98 = runtime

    def _unbind_amr_interface_v98(self, runtime) -> None:
        """Remove the exact additive v98 owner without publishing state."""
        if self._amr_interface_v98 is not runtime:
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA v98 interface owner unbind differs")
        self._amr_interface_v98 = None

    def _validate_runtime_identity(self) -> None:
        execution_document_wetdry_fwave_gpu_v97(self.execution)
        if (int(self.cp.cuda.Device().id) != self.device_id or
                self.runtime_receipt.get("receipt_sha256") != _hash({
                    key: value for key, value in self.runtime_receipt.items()
                    if key != "receipt_sha256"}) or
                self.runtime_receipt.get("source_sha256") != {
                    _source_name(path): _file_sha(path)
                    for path in (
                        Path(__file__), DENSIN_SOURCE,
                        Path(self.controller.source_path),
                        Path(__file__).with_name(
                            "density_active_projection_v97.cu"),
                        Path(__file__).with_name(
                            "density_active_projection_v97.py"),
                    )} or
                self.controller.source_sha256 != _file_sha(
                    self.controller.source_path) or
                self.controller.source_sha256 != _file_sha(
                    CONTROLLER_SOURCE)):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA runtime/source identity differs")
        base = self.execution.density_execution
        actual = (
            self.cp.asnumpy(self.controller.owner),
            self.cp.asnumpy(self.controller.neighbour),
            self.cp.asnumpy(self.controller.normal),
            self.cp.asnumpy(self.controller.length),
            self.cp.asnumpy(self.controller.area),
            self.cp.asnumpy(self.controller.offset),
            self.cp.asnumpy(self.controller.faces),
            self.cp.asnumpy(self.controller.signs),
        )
        expected = (
            base.owner, base.neighbour, base.normal, base.face_length_m,
            base.cell_area_m2, base.cell_face_offset, base.cell_faces,
            base.cell_face_sign,
        )
        if any(not np.array_equal(left, right)
               for left, right in zip(actual, expected)):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA resident controller geometry differs")
        if self.turbulence_cap is not None and (
                self.turbulence_cap.receipt_sha256 !=
                    _cap_token(self.turbulence_cap) or
                not bool(self.cp.array_equal(
                    self.capped_vicuv,
                    self.cp.asarray(self.turbulence_cap.
                                    capped_total_vicuv_m2_s)).item())):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA resident turbulence cap differs")

    def _launch_face_density(self, depth, selected_density) -> None:
        self.face_status.fill(0)
        self.densin_face_kernel(
            ((self.nf+127)//128,), (128,), (
                depth, selected_density, self.owner, self.neighbour,
                self.distance, np.int64(self.nf),
                np.float64(self.execution.density_execution.gravity_m_s2),
                np.float64(self.execution.density_execution.
                           hydraulic_reference_density_kg_m3),
                self.face_acceleration, self.face_status))
        if bool(self.cp.any(self.face_status != 0).item()):
            raise ImplicitStepRejected(
                "v97 CUDA face-density kernel failed")

    def _launch_projection(self) -> None:
        runtime = self.projection
        if runtime.receipt["receipt_sha256"] != _hash({
                key: value for key, value in runtime.receipt.items()
                if key != "receipt_sha256"}):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA density projection receipt differs")
        normal, length, offset, faces, signs = runtime.geometry_device
        runtime.kernel(((self.nc+127)//128,), (128,), (
            np.int64(self.nc), np.int64(self.nf), self.masked_q[0],
            self.face_acceleration, normal, length, offset, faces, signs,
            self.wet, self.face_active, np.float64(RANK_TOL),
            np.float64(RESIDUAL_TOL), self.density_cell_acceleration,
            self.density_rate[1:], self.density_rank,
            self.density_residual, self.density_status))
        self.density_rate[0].fill(0)
        if bool(self.cp.any(self.density_status != 0).item()):
            raise ImplicitStepRejected(
                "v97 CUDA density projection rejected rank/residual")

    def _rate(self, midpoint, wet, selected_density,
              dt: float, time: float, poll) -> tuple[float, int]:
        if poll():
            raise ImplicitStepCancelled("v97 CUDA residual cancelled")
        cp, controller, solver = self.cp, self.controller, self.controller.solver
        self.wet[...] = wet
        self.masked_q[...] = cp.where(wet[None, :] != 0, midpoint, 0.0)
        # The production boundary wrappers intentionally bind the canonical
        # stage allocation by object identity.  Evaluate the masked active-set
        # state in that same allocation instead of handing the live boundary a
        # look-alike buffer (which would bypass its sole-owner guarantee).
        q_eval = controller.midpoint
        cp.copyto(q_eval, self.masked_q)
        right = cp.maximum(self.neighbour, 0)
        self.face_active[...] = ((self.neighbour >= 0) &
            (wet[self.owner] != 0) & (wet[right] != 0)).astype(cp.int32)
        solver.boundary.begin_trial()
        bed = controller._current_bed()
        reconstructed = solver.reconstruction.reconstruct(q_eval)
        solver.boundary.apply(q_eval, reconstructed["trace"])
        secondary = solver.secondary.reconstruct(
            q_eval, bed,
            controller.secondary_base if controller.component_composition
            else solver.state["secondary"])
        solver.flux.inputs["q"] = q_eval
        old_target = solver.flux.controls["cfl_target"]
        solver.flux.controls["dt"] = np.float64(dt)
        solver.flux.controls["cfl_target"] = np.float64(1.e100)

        def postassemble(buffers):
            solver.boundary.apply_pressure(
                q_eval, bed, reconstructed["trace"],
                buffers["fl"], buffers["fr"], buffers["face_status"])
            if self._amr_interface_v98 is not None:
                self._amr_interface_v98.replace_reflective_faces(
                    q_eval=q_eval, bed_elevation_m=bed,
                    reconstructed_trace=reconstructed["trace"],
                    selected_density_kg_m3=selected_density,
                    flux_buffers=buffers, cancelled=poll)
        try:
            flux, stress = solver.flux.step(
                solver.secondary, postassemble=postassemble)
        finally:
            solver.flux.controls["cfl_target"] = old_target
        statuses = (
            reconstructed["primitive_status"],
            reconstructed["reconstruction_status"], solver.boundary.status,
            solver.boundary.face_status, flux["augmented_status"],
            flux["face_status"], flux["cell_status"], secondary["status"],
            stress)
        if not controller._all_status_clear(statuses):
            raise ImplicitStepRejected(
                "v97 CUDA augmented f-wave residual status is nonzero")
        if self._amr_interface_v98 is not None:
            self._amr_interface_v98.capture_limited_faces(
                flux_buffers=flux, cancelled=poll)
        controller._launch("implicit_full_rate_from_faces", self.nc, (
                q_eval, controller.area, controller.length,
            controller.normal, flux["fl"], flux["fr"], controller.offset,
            controller.faces, controller.signs, self.domain_active,
            np.int64(self.nc), np.int64(self.nf), controller.grav,
            self.base_rate))
        self._launch_face_density(q_eval[0], selected_density)
        self._launch_projection()
        self.turbulence_rate.fill(0)
        self.turbulence_face_flux.fill(0)
        turbulence_count = 0
        if self.turbulence is not None:
            result = self.turbulence.evaluate(
                q_eval, self.capped_vicuv, self.wet, self.face_active,
                cancelled=poll)
            cp.copyto(self.turbulence_rate,
                      result["momentum_rate_m2_s2"])
            cp.copyto(self.turbulence_face_flux,
                      result["face_flux_m4_s2"])
            turbulence_count = 1
        self.total_rate[...] = self.base_rate
        self.total_rate += self.density_rate
        self.total_rate += self.turbulence_rate
        cp.copyto(self.final_water_flux, flux["fl"][0])
        cp.copyto(self.final_theta, flux["theta"])
        # Copy, do not recompute: fl/fr are the sole augmented-f-wave owner's
        # actual side fluxes for this nonlinear residual.  Repeated residual
        # images are provisional and overwritten; the converged call remains.
        cp.copyto(self.v98_final_augmented_left_flux, flux["fl"][:3])
        cp.copyto(self.v98_final_augmented_right_flux, flux["fr"][:3])
        controller.fwave_evaluations += 1
        return float(cp.max(flux["cfl_ratio"]).item()), turbulence_count

    def _snapshot_boundary(self) -> tuple[list[tuple[object, object]], dict[str, object]]:
        """Snapshot every resident controller/boundary scratch allocation.

        The active-set solve is provisional.  In particular, using the
        canonical stage allocation for the sole live boundary must not leak a
        changed predictor, trace cache, boundary record, or stage cursor to the
        enclosing v97 transaction on success, cancellation, or rejection.
        """
        cp = self.cp
        arrays: dict[int, object] = {}

        def visit(value, depth=0):
            if isinstance(value, cp.ndarray):
                arrays.setdefault(id(value), value)
                return
            if depth >= 5 or isinstance(value, types.ModuleType) or callable(value):
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    visit(item, depth+1)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    visit(item, depth+1)
                return
            members = getattr(value, "__dict__", None)
            if isinstance(members, Mapping):
                for item in members.values():
                    visit(item, depth+1)

        boundary = self.controller.solver.boundary
        visit(boundary)
        # Controller scratch is deliberately not accepted state, but it is
        # restart-critical (predictor and residual workspaces included).
        for value in self.controller.__dict__.values():
            if isinstance(value, cp.ndarray):
                arrays.setdefault(id(value), value)
        flux = self.controller.solver.flux
        metadata = {
            "fwave_evaluations": self.controller.fwave_evaluations,
            "linear_matvecs": self.controller.linear_matvecs,
            "have_predictor": self.controller.have_predictor,
            "last_diagnostics": copy.deepcopy(
                self.controller.last_diagnostics),
            "boundary_next_stage": getattr(
                boundary, "_g1_next_stage", None),
            "flux_controls": dict(flux.controls),
            "flux_had_q": "q" in flux.inputs,
            "flux_input_q": flux.inputs.get("q"),
        }
        return ([(value, value.copy()) for value in arrays.values()],
                metadata)

    def _restore_boundary(self, saved, inventory) -> None:
        for target, value in saved:
            self.cp.copyto(target, value)
        self.controller.fwave_evaluations = inventory["fwave_evaluations"]
        self.controller.linear_matvecs = inventory["linear_matvecs"]
        self.controller.have_predictor = inventory["have_predictor"]
        self.controller.last_diagnostics = copy.deepcopy(
            inventory["last_diagnostics"])
        boundary = self.controller.solver.boundary
        if hasattr(boundary, "_g1_next_stage"):
            boundary._g1_next_stage = inventory["boundary_next_stage"]
        flux = self.controller.solver.flux
        flux.controls.clear(); flux.controls.update(inventory["flux_controls"])
        if inventory["flux_had_q"]:
            flux.inputs["q"] = inventory["flux_input_q"]
        else:
            flux.inputs.pop("q", None)

    def solve(
        self, parent_q, *, accepted_water_density_kg_m3,
        accepted_mixture_density_kg_m3, bed_elevation_m,
        dt_s: float, evaluation_time_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> WetDryFwaveGPUTrialV97:
        """Return one authenticated provisional endpoint; publish nothing."""
        self._validate_runtime_identity()
        poll = (cancelled if callable(cancelled)
                else lambda: _cancelled(cancelled))
        if _cancelled(cancelled):
            raise InterruptedError("v97 CUDA cancelled before child work")
        cp, execution = self.cp, self.execution
        q0 = cp.ascontiguousarray(cp.asarray(parent_q, dtype=cp.float64))
        if (q0.shape != (3, self.nc) or
                not bool(cp.isfinite(q0).all().item()) or
                bool(cp.any(q0[0] < 0).item())):
            raise WetDryFwaveGPUV97Error("v97 CUDA parent q differs")
        water = cp.ascontiguousarray(cp.asarray(
            accepted_water_density_kg_m3, dtype=cp.float64))
        mixture = cp.ascontiguousarray(cp.asarray(
            accepted_mixture_density_kg_m3, dtype=cp.float64))
        bed = cp.ascontiguousarray(cp.asarray(bed_elevation_m,
                                              dtype=cp.float64))
        if (water.shape != (self.nc,) or mixture.shape != (self.nc,) or
                bed.shape != (self.nc,) or
                not bool(cp.isfinite(water).all().item()) or
                not bool(cp.isfinite(mixture).all().item()) or
                not bool(cp.isfinite(bed).all().item()) or
                bool(cp.any(water <= 0).item()) or
                bool(cp.any(mixture < water).item())):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA accepted bed/density arrays differ")
        if self.controller._v97_provisional_bed is not None:
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA controller already has a provisional bed owner")
        dt = float(dt_s); evaluation_time = float(evaluation_time_s)
        if (isinstance(dt_s, bool) or not math.isfinite(dt) or dt <= 0 or
                isinstance(evaluation_time_s, bool) or
                not math.isfinite(evaluation_time) or evaluation_time < 0):
            raise WetDryFwaveGPUV97Error("v97 CUDA trial clock differs")
        dry = float(execution.density_execution.dry_tolerance_m)
        parent_wet = ((self.domain_active != 0) & (q0[0] > dry))
        if bool(cp.any((~parent_wet) & (q0[1:] != 0)).item()):
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA dry parent momentum must be exact zero")
        selected_density = (mixture if execution.density_mode ==
                            MODE_WATER_PLUS_SEDIMENT else water)
        if execution.shared_turbulence_enabled:
            if (self.turbulence_cap is None or
                    float(dt_s) != self.turbulence_cap.dt_s or
                    self.turbulence_cap.receipt_sha256 !=
                        _cap_token(self.turbulence_cap)):
                raise WetDryFwaveGPUV97Error(
                    "v97 CUDA selected turbulence cap/clock differs")
            if (self.capped_vicuv.shape != (self.nc,) or
                    not bool(cp.isfinite(self.capped_vicuv).all().item()) or
                    bool(cp.any(self.capped_vicuv < 0).item())):
                raise WetDryFwaveGPUV97Error(
                    "v97 CUDA accepted vicuv differs")
        else:
            if self.turbulence_cap is not None:
                raise WetDryFwaveGPUV97Error(
                    "v97 CUDA unselected turbulence retained cap")
        self.controller._v97_provisional_bed = bed
        if not bool(cp.array_equal(
                bed, self.controller._current_bed()).item()):
            self.controller._v97_provisional_bed = None
            raise WetDryFwaveGPUV97Error(
                "v97 CUDA accepted bed differs from hydraulic owner")
        saved, inventory = self._snapshot_boundary()
        initial_fwave = self.controller.fwave_evaluations
        # A failed or cancelled new solve invalidates any prior v98 evidence.
        self.invalidate_v98_hydraulic_side_evidence()
        completed_v98_evidence = False
        try:
            tau = .5*dt
            wet = parent_wet.astype(cp.int32)
            midpoint = q0.copy()
            seen: set[str] = set()
            total_iterations = total_residuals = damping_count = 0
            total_limiter = total_riemann = total_turbulence = 0
            residual_norm = math.inf
            endpoint = None
            final_wet = None
            for restart in range(execution.max_active_set_restarts+1):
                key = _digest(cp.asnumpy(wet))
                if key in seen:
                    raise ImplicitStepRejected(
                        "v97 CUDA wet active set oscillated")
                seen.add(key)
                if restart:
                    midpoint = .5*(q0+endpoint)
                midpoint[1:, wet == 0] = 0.0
                converged = False
                for _ in range(execution.max_inner_iterations+1):
                    if poll():
                        raise ImplicitStepCancelled(
                            "v97 CUDA midpoint solve cancelled")
                    _, turbulence_count = self._rate(
                        midpoint, wet, selected_density, dt,
                        evaluation_time, poll)
                    residual = midpoint-q0-tau*self.total_rate
                    residual_norm = float(cp.max(cp.abs(residual)).item())
                    scale = max(
                        1.0, float(cp.max(cp.abs(midpoint)).item()),
                        float(cp.max(cp.abs(q0)).item()),
                        float(cp.max(cp.abs(tau*self.total_rate)).item()))
                    total_residuals += 1
                    total_riemann += self.nf
                    total_limiter += int(cp.count_nonzero(
                        self.final_theta < 1.0).item())
                    total_turbulence += turbulence_count
                    if residual_norm <= execution.nonlinear_tolerance*scale:
                        converged = True
                        break
                    if total_iterations >= ((restart+1)*
                                             execution.max_inner_iterations):
                        break
                    delta = -residual
                    falling = delta[0] < 0.0
                    alpha = 1.0
                    if bool(cp.any(falling).item()):
                        floor = cp.where(wet != 0,
                                         np.finfo(np.float64).tiny, 0.0)
                        available = midpoint[0, falling]-floor[falling]
                        if bool(cp.any(available <= 0).item()):
                            raise ImplicitStepRejected(
                                "v97 CUDA damping has no positive depth")
                        alpha = min(alpha, execution.positivity_fraction*float(
                            cp.min(available/(-delta[0, falling])).item()))
                    if not math.isfinite(alpha) or alpha <= 0:
                        raise ImplicitStepRejected(
                            "v97 CUDA damping factor differs")
                    damping_count += int(alpha < 1.0)
                    midpoint += alpha*delta
                    if (not bool(cp.isfinite(midpoint).all().item()) or
                            bool(cp.any(midpoint[0] < 0).item())):
                        raise ImplicitStepRejected(
                            "v97 CUDA damped midpoint left envelope")
                    total_iterations += 1
                if not converged:
                    raise ImplicitStepRejected(
                        "v97 CUDA midpoint did not converge")
                endpoint = q0+dt*self.total_rate
                if (not bool(cp.isfinite(endpoint).all().item()) or
                        bool(cp.any(endpoint[0] < 0).item())):
                    raise ImplicitStepRejected(
                        "v97 CUDA limited endpoint is invalid")
                final_wet = ((self.domain_active != 0) &
                             (endpoint[0] > dry)).astype(cp.int32)
                # A growing set must be resolved again because newly wetted
                # cells enter the augmented f-wave, density and v91
                # operators.  A shrinking set must *not* be resolved with
                # the drying cell removed: doing so removes its donor flux,
                # restores the parent depth and creates an A/B oscillation.
                # Retain the converged hydraulic depth/face ledger and defer
                # only the newly-dry momentum projection to the accepted
                # endpoint.  The scalar/Q3 owners consume ``final_wet``.
                newly_wet = (final_wet != 0) & (wet == 0)
                newly_dry = (wet != 0) & (final_wet == 0)
                if (bool(cp.any(newly_wet).item()) and
                        bool(cp.any(newly_dry).item())):
                    raise ImplicitStepRejected(
                        "v97 CUDA simultaneous wetting/drying is outside "
                        "the bounded monotone active-set route")
                if not bool(cp.any(newly_wet).item()):
                    break
                wet = ((wet != 0) | (final_wet != 0)).astype(cp.int32)
                midpoint = .5*(q0+endpoint)
                midpoint[1:, wet == 0] = 0.0
            else:
                raise ImplicitStepRejected(
                    "v97 CUDA active-set restart bound exhausted")
            if bool(cp.any((final_wet != 0) & (wet == 0)).item()):
                raise ImplicitStepRejected(
                    "v97 CUDA accepted wet mask has an unresolved growth")
            accepted_dry = final_wet == 0
            cleanup_cell = accepted_dry & cp.any(
                endpoint[1:] != 0.0, axis=0)
            accepted_total_rate = self.total_rate.copy()
            accepted_total_rate[1:, accepted_dry] = (
                -q0[1:, accepted_dry]/dt)
            dry_cleanup = accepted_total_rate-self.total_rate
            endpoint = q0+dt*accepted_total_rate
            endpoint[1:, accepted_dry] = 0.0
            dry_cleanup_count = int(cp.count_nonzero(cleanup_cell).item())
            dry_cleanup_linf = float(cp.max(cp.abs(dry_cleanup)).item())
            projected_identity_linf = float(cp.max(cp.abs(
                endpoint-(q0+dt*accepted_total_rate))).item())
            volume_change = float(cp.sum(
                self.controller.area*(endpoint[0]-q0[0])).item())
            boundary = self.neighbour < 0
            boundary_ledger = -dt*float(cp.sum(
                self.final_water_flux[boundary]*
                self.controller.length[boundary]).item())
            water_residual = volume_change-boundary_ledger
            volume_scale = max(1.0, abs(volume_change), float(cp.sum(
                self.controller.area*q0[0]).item()))
            if abs(water_residual) > 2e-11*volume_scale:
                raise ImplicitStepRejected(
                    "v97 CUDA water ledger differs")
            host = tuple(np.ascontiguousarray(cp.asnumpy(value)) for value in (
                q0, midpoint, endpoint, wet, final_wet, self.face_active,
                self.final_water_flux, self.final_theta, self.base_rate,
                self.density_rate, self.turbulence_rate,
                self.turbulence_face_flux, accepted_total_rate,
                dry_cleanup,
                self.density_rank, self.density_residual))
            for value in host:
                value.setflags(write=False)
            placeholder = WetDryFwaveGPUTrialV97(
                execution.fingerprint, _digest(host[0]), _digest(cp.asnumpy(bed)),
                _digest(cp.asnumpy(water)), _digest(cp.asnumpy(mixture)), dt,
                evaluation_time, *host,
                (None if self.turbulence_cap is None else
                 self.turbulence_cap.receipt_sha256),
                restart, total_iterations,
                total_residuals, damping_count, total_limiter, total_riemann,
                total_residuals, total_residuals, total_turbulence, 0, 0,
                dry_cleanup_count, dry_cleanup_linf,
                projected_identity_linf, 0,
                water_residual, residual_norm, "actual_cuda", "")
            trial = WetDryFwaveGPUTrialV97(
                **{**placeholder.__dict__,
                   "trial_sha256": _hash(_trial_payload(placeholder))})
            self.last_trial = trial
            self._v98_hydraulic_trial_sha256 = trial.trial_sha256
            self._v98_hydraulic_side_evaluation_count = total_residuals
            completed_v98_evidence = True
            return trial
        except ImplicitStepCancelled as error:
            raise InterruptedError(str(error)) from error
        finally:
            if not completed_v98_evidence:
                self.invalidate_v98_hydraulic_side_evidence()
            self._restore_boundary(saved, inventory)
            self.controller._v97_provisional_bed = None
            # Every residual was provisional; no controller clock/count is
            # allowed to escape even on successful readback.
            self.controller.fwave_evaluations = initial_fwave


def validate_wetdry_fwave_gpu_trial_v97(
    value: WetDryFwaveGPUTrialV97,
    execution: WetDryFwaveGPUExecutionV97,
) -> None:
    if type(value) is not WetDryFwaveGPUTrialV97:
        raise WetDryFwaveGPUV97Error("v97 CUDA trial type differs")
    execution_document_wetdry_fwave_gpu_v97(execution)
    nc, nf = execution.cell_count, execution.face_count
    arrays_shapes = (
        (value.parent_q, (3, nc)), (value.midpoint_q, (3, nc)),
        (value.endpoint_q, (3, nc)),
        (value.frozen_wet_mask, (nc,)),
        (value.accepted_wet_mask, (nc,)),
        (value.active_wet_wet_face_mask, (nf,)),
        (value.accepted_limited_face_water_flux_m2_s, (nf,)),
        (value.accepted_draining_theta, (nc,)),
        (value.base_fwave_rate, (3, nc)),
        (value.density_momentum_rate, (3, nc)),
        (value.shared_turbulence_rate, (3, nc)),
        (value.shared_turbulence_face_flux, (2, nf)),
        (value.accepted_total_rate, (3, nc)),
        (value.dry_momentum_cleanup_rate_m2_s2, (3, nc)),
        (value.density_numerical_rank, (nc,)),
        (value.density_projection_residual, (nc,)),
    )
    if (value.execution_fingerprint != execution.fingerprint or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            not math.isfinite(value.evaluation_time_s) or
            value.evaluation_time_s < 0.0 or
            any(array.shape != shape or array.flags.writeable or
                not np.isfinite(array).all()
                for array, shape in arrays_shapes) or
            np.any((value.frozen_wet_mask != 0) &
                   (value.frozen_wet_mask != 1)) or
            np.any((value.accepted_wet_mask != 0) &
                   (value.accepted_wet_mask != 1)) or
            np.any((value.active_wet_wet_face_mask != 0) &
                   (value.active_wet_wet_face_mask != 1)) or
            np.any(value.accepted_draining_theta < 0) or
            np.any(value.accepted_draining_theta > 1) or
            value.density_face_evaluation_count !=
                value.true_residual_evaluation_count or
            value.density_projection_application_count !=
                value.true_residual_evaluation_count or
            value.shared_turbulence_application_count != (
                value.true_residual_evaluation_count
                if execution.shared_turbulence_enabled else 0) or
            (not execution.shared_turbulence_enabled and
             (np.any(value.shared_turbulence_rate != 0.0) or
              np.any(value.shared_turbulence_face_flux != 0.0))) or
            (value.shared_turbulence_cap_receipt_sha256 is not None) !=
                execution.shared_turbulence_enabled or
            value.limiter_derivative_evaluation_count != 0 or
            value.mask_derivative_evaluation_count != 0 or
            value.dry_momentum_cleanup_cell_count < 0 or
            not math.isfinite(value.dry_momentum_cleanup_linf_m2_s2) or
            value.dry_momentum_cleanup_linf_m2_s2 < 0.0 or
            not math.isfinite(value.projected_endpoint_identity_linf) or
            value.projected_endpoint_identity_linf < 0.0 or
            value.state_or_clock_publication_count != 0 or
            value.backend != "actual_cuda" or
            value.trial_sha256 != _hash(_trial_payload(value))):
        raise WetDryFwaveGPUV97Error("v97 CUDA trial identity differs")
    owner, neighbour = (execution.density_execution.owner,
                        execution.density_execution.neighbour)
    expected = ((neighbour >= 0) &
                (value.frozen_wet_mask[owner] != 0) &
                (value.frozen_wet_mask[np.maximum(neighbour, 0)] != 0))
    if not np.array_equal(value.active_wet_wet_face_mask != 0, expected):
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA trial wet-wet face mask differs")
    if (value.parent_q_sha256 != _digest(value.parent_q) or
            any(len(item) != 64 or any(character not in
                "0123456789abcdef" for character in item) for item in (
                    value.accepted_bed_sha256,
                    value.accepted_water_density_sha256,
                    value.accepted_mixture_density_sha256))):
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA accepted parent/physical input identity differs")
    expected_wet = ((execution.domain_active_mask != 0) &
                    (value.endpoint_q[0] >
                     execution.density_execution.dry_tolerance_m))
    frozen_wet = value.frozen_wet_mask != 0
    accepted_wet = value.accepted_wet_mask != 0
    accepted_dry = ~accepted_wet
    parent_wet = ((execution.domain_active_mask != 0) &
                  (value.parent_q[0] >
                   execution.density_execution.dry_tolerance_m))
    expected_evaluation_wet = parent_wet | accepted_wet
    physical_total_rate = (
        value.base_fwave_rate+value.density_momentum_rate+
        value.shared_turbulence_rate)
    midpoint_defect = (value.midpoint_q-value.parent_q-
                       .5*value.dt_s*physical_total_rate)
    residual = float(np.max(np.abs(midpoint_defect)))
    scale = max(1.0, float(np.max(np.abs(value.midpoint_q))),
                float(np.max(np.abs(value.parent_q))),
                float(np.max(np.abs(
                    .5*value.dt_s*physical_total_rate))))
    raw_endpoint = value.parent_q+value.dt_s*physical_total_rate
    accepted_total_expected = np.array(physical_total_rate, copy=True)
    accepted_total_expected[1:, accepted_dry] = (
        -value.parent_q[1:, accepted_dry]/value.dt_s)
    cleanup_expected = accepted_total_expected-physical_total_rate
    endpoint_expected = value.parent_q+(
        value.dt_s*accepted_total_expected)
    endpoint_error = float(np.max(np.abs(
        value.endpoint_q-endpoint_expected)))
    cleanup_linf = float(np.max(np.abs(
        value.dry_momentum_cleanup_rate_m2_s2)))
    endpoint_scale = max(1.0, float(np.max(np.abs(value.endpoint_q))),
                         float(np.max(np.abs(endpoint_expected))))
    area = execution.density_execution.cell_area_m2
    boundary = execution.density_execution.neighbour < 0
    volume_change = float(np.sum(
        area*(value.endpoint_q[0]-value.parent_q[0])))
    boundary_ledger = -value.dt_s*float(np.sum(
        value.accepted_limited_face_water_flux_m2_s[boundary]*
        execution.density_execution.face_length_m[boundary]))
    water_residual = volume_change-boundary_ledger
    volume_scale = max(1.0, abs(volume_change), float(np.sum(
        area*value.parent_q[0])))
    if (not np.array_equal(accepted_wet, expected_wet) or
            not np.array_equal(frozen_wet, expected_evaluation_wet) or
            not np.array_equal(
                value.dry_momentum_cleanup_rate_m2_s2,
                cleanup_expected) or
            not np.array_equal(
                value.accepted_total_rate, accepted_total_expected) or
            value.dry_momentum_cleanup_cell_count !=
                int(np.count_nonzero(
                    accepted_dry & np.any(
                        raw_endpoint[1:] != 0.0, axis=0))) or
            value.dry_momentum_cleanup_linf_m2_s2 != cleanup_linf or
            value.projected_endpoint_identity_linf != endpoint_error or
            np.any(value.dry_momentum_cleanup_rate_m2_s2[0] != 0.0) or
            np.any(value.endpoint_q[0] < 0.0) or
            np.any(value.parent_q[0] < 0.0) or
            np.any(value.endpoint_q[1:, ~accepted_wet] != 0.0) or
            np.any(value.density_momentum_rate[:, ~frozen_wet] != 0.0) or
            abs(residual-value.nonlinear_residual_linf) >
                16*np.finfo(float).eps*scale or
            residual > execution.nonlinear_tolerance*scale or
            endpoint_error > 16*np.finfo(float).eps*endpoint_scale or
            abs(water_residual-value.water_volume_residual_m3) >
                64*np.finfo(float).eps*volume_scale or
            abs(water_residual) > 2e-11*volume_scale or
            value.augmented_riemann_call_count !=
                nf*value.true_residual_evaluation_count or
            value.true_residual_evaluation_count < 1 or
            value.inner_iteration_count < 0 or
            value.active_set_restart_count < 0 or
            value.active_set_restart_count >
                execution.max_active_set_restarts or
            value.inner_iteration_count >
                (value.active_set_restart_count+1)*
                execution.max_inner_iterations):
        raise WetDryFwaveGPUV97Error(
            "v97 CUDA trial equations, wet set, counts, or ledgers differ")


def method_identity_wetdry_fwave_gpu_v97() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "integrator": "implicit_fwave_midpoint",
        "actual_cuda_kernels": True,
        "augmented_fwave_preserved": True,
        "moving_wet_dry_active_set": True,
        "draining_limiter_inside_true_residual": True,
        "drying_policy":
            "retain_converged_depth_flux_then_project_dry_momentum_once",
        "wetting_policy": "monotone_active_set_restart",
        "rank_aware_density_projection": DENSITY_PROJECTION_POLICY_V97,
        "automatic_cpu_fallback": False,
        "state_or_clock_publication_count": 0,
        "production_admission": False,
    }


__all__ = [
    "MAX_ACTIVE_SET_RESTARTS", "MAX_INNER_ITERATIONS", "MODE_WATER_EOS",
    "MODE_WATER_PLUS_SEDIMENT", "ResidentWetDryFwaveV97", "SCOPE_ID",
    "WetDryFwaveGPUExecutionV97", "WetDryFwaveGPUTrialV97",
    "WetDryFwaveGPUV97Error", "execution_document_wetdry_fwave_gpu_v97",
    "method_identity_wetdry_fwave_gpu_v97",
    "resolve_wetdry_fwave_gpu_execution_v97",
    "validate_wetdry_fwave_gpu_trial_v97",
]
