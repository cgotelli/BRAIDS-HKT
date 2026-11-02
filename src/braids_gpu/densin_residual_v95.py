"""Actual-CUDA DensIn feedback inside the live implicit f-wave residual.

This is a bounded validation composition, not product admission.  The accepted
v78 concentration and its water density are frozen for one hydraulic trial.
The current nonlinear midpoint depth is nevertheless consumed on every
authoritative residual evaluation.  The resulting density momentum source is
added to the resident rate before the implicit residual is formed; it is never
applied after convergence.

The numerical path is CUDA-only.  Host equation oracles are used after the
trial solely to authenticate the resident work and are never used to advance
the state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np

from braids_core.sediment.densin_fwave_v85 import (
    AcceptedConcentrationReceiptV85,
    DensInExecutionV85,
    LOCKED_UPSTREAM_COMMIT,
    delft_kmax1_face_acceleration_v85,
    execution_document_v85,
    project_face_acceleration_v85,
    validate_accepted_concentration_receipt_v85,
    verify_locked_sources_v85,
)
from .densin_fwave_v85 import ResidentDensInFwaveV85


SCHEMA = 1
SCOPE_ID = "braids-q4-densin-residual-gpu-v95-v1"
SEMANTIC_ID = SCOPE_ID + ":validation-live-fwave-residual-hook"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
CUDA_COMPOSITION_ID = "resident-densin-true-implicit-residual-v95-v1"
MAX_ACCEPTED_UPDATES = 4
DENSIN_SEDIMENT_INCREMENT = "sediment_increment"
BACKEND = "actual_cuda"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = Path(__file__).resolve().parents[1]


class DensInResidualGPUV95Error(ValueError):
    """The bounded resident v95 composition or transaction is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise DensInResidualGPUV95Error(
            "v95 CUDA value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DensInResidualGPUV95Error(
            f"v95 CUDA {name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise DensInResidualGPUV95Error(
            f"v95 CUDA {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise DensInResidualGPUV95Error(
            f"v95 CUDA {name} must be positive")
    return result


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise DensInResidualGPUV95Error(
            f"v95 CUDA {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise DensInResidualGPUV95Error(f"v95 CUDA {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise DensInResidualGPUV95Error(f"v95 CUDA {name} must be positive")
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not callable(value):
        raise DensInResidualGPUV95Error(
            "v95 CUDA cancellation must be boolean or a zero-argument callable")
    result = value()
    if not isinstance(result, (bool, np.bool_)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA cancellation callback must return boolean data")
    return bool(result)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DensInResidualGPUV95Error(message)


def _manning_half_step_oracle(
    q: np.ndarray, roughness: np.ndarray, half_dt_s: float, *,
    gravity_m_s2: float, dry_tolerance_m: float,
    chezy_minimum_depth_m: float,
) -> np.ndarray:
    """Host authentication of the unchanged ``ti_manning`` split map.

    This is acceptance evidence only.  The candidate is advanced by the
    existing CUDA kernel owned by the implicit controller.
    """
    result = np.array(q, dtype=np.float64, copy=True, order="C")
    h, hu, hv = result
    dry = h <= dry_tolerance_m
    wet = ~dry
    result[1:, dry] = 0.0
    if np.any(wet):
        hw = h[wet]
        xw = hu[wet]
        yw = hv[wet]
        nw = roughness[wet]
        speed = np.sqrt(xw*xw+yw*yw)/hw
        denominator = np.power(hw, 4.0/3.0)
        if chezy_minimum_depth_m > 0.0:
            floored = hw < chezy_minimum_depth_m
            denominator[floored] = (
                hw[floored]*np.power(chezy_minimum_depth_m, 1.0/3.0))
        factor = (1.0+half_dt_s*gravity_m_s2*(nw*nw)*speed/
                  denominator)
        result[1, wet] = xw/factor
        result[2, wet] = yw/factor
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DensInResidualGPUExecutionV95:
    base_execution: DensInExecutionV85
    densin_mode: str
    manning_roughness: np.ndarray
    manning_chezy_minimum_depth_m: float
    nonlinear_rtol: float
    nonlinear_atol: float
    maximum_dt_s: float
    maximum_residual_evaluations: int
    locked_delft_commit: str
    locked_source_sha256: tuple[tuple[str, str], ...]
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count


def _execution_payload(value: DensInResidualGPUExecutionV95) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "base_v85_execution_fingerprint": value.base_execution.fingerprint,
        "densin_mode": value.densin_mode,
        "split_friction": "authenticated_manning_half_steps",
        "manning_roughness_sha256": _digest(value.manning_roughness),
        "manning_chezy_minimum_depth_m":
            value.manning_chezy_minimum_depth_m,
        "densin_off_identity": "hook_absent_from_controller",
        "accepted_concentration_source": "typed_v78_via_v85_only",
        "water_eos_mode": "uniform_accepted_water_density",
        "spatial_water_density": "rejected_deferred",
        "integrator": "implicit_fwave_midpoint",
        "spatial_flux": "unchanged_live_augmented_geoclaw_fwave",
        "reconstruction": "live_controller_selected_scheme",
        "grid_scope": "fully_wet_cartesian_fixed_grid",
        "density_timing": (
            "accepted_parent_mixture_frozen;current_midpoint_depth_"
            "reevaluated_each_full_rate;no_post_convergence_apply"),
        "approximate_jacobian_density_derivative": "omitted_frozen_acoustic",
        "nonlinear_rtol": value.nonlinear_rtol,
        "nonlinear_atol": value.nonlinear_atol,
        "maximum_dt_s": value.maximum_dt_s,
        "maximum_residual_evaluations": value.maximum_residual_evaluations,
        "locked_delft_commit": value.locked_delft_commit,
        "locked_delft_source_sha256": dict(value.locked_source_sha256),
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "production_admission": False,
    }


def resolve_densin_residual_gpu_execution_v95(
    base_execution: DensInExecutionV85, *, densin_enabled: bool,
    manning_roughness: object | None = None,
    manning_chezy_minimum_depth_m: float = 0.0,
    nonlinear_rtol: float = 5.0e-2,
    nonlinear_atol: float = 1.0e-14,
    maximum_dt_s: float = 0.02,
    maximum_residual_evaluations: int = 81,
) -> DensInResidualGPUExecutionV95:
    if type(base_execution) is not DensInExecutionV85:
        raise DensInResidualGPUV95Error(
            "v95 CUDA requires an exact v85 execution")
    execution_document_v85(base_execution)
    if base_execution.role != "implicit_fwave_midpoint_gpu":
        raise DensInResidualGPUV95Error(
            "v95 CUDA requires the selected implicit GPU role")
    if type(densin_enabled) is not bool or not densin_enabled:
        raise DensInResidualGPUV95Error(
            "v95 CUDA DensIn-off is the byte-identical unbound controller path")
    if np.ptp(base_execution.water_density_kg_m3) != 0.0:
        raise DensInResidualGPUV95Error(
            "v95 CUDA rejects spatial accepted water density")
    normals = base_execution.normal
    if not np.all((np.abs(normals[0]) <= 2e-13) |
                  (np.abs(normals[1]) <= 2e-13)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA bounded proof requires Cartesian faces")
    maximum_dt = _finite(maximum_dt_s, "maximum timestep", positive=True)
    roughness = _array(
        np.zeros(base_execution.cell_count, dtype=np.float64)
        if manning_roughness is None else manning_roughness,
        (base_execution.cell_count,), "Manning roughness")
    if np.any(roughness < 0.0):
        raise DensInResidualGPUV95Error(
            "v95 CUDA Manning roughness must be nonnegative")
    chezy_min = _finite(
        manning_chezy_minimum_depth_m, "Manning minimum work depth")
    if chezy_min < 0.0:
        raise DensInResidualGPUV95Error(
            "v95 CUDA Manning minimum work depth is negative")
    rtol = _finite(nonlinear_rtol, "nonlinear relative tolerance", positive=True)
    atol = _finite(nonlinear_atol, "nonlinear absolute tolerance", positive=True)
    if not (rtol < 1.0 and atol < 1.0):
        raise DensInResidualGPUV95Error(
            "v95 CUDA nonlinear tolerances are outside scope")
    if (type(maximum_residual_evaluations) is not int or not
            1 <= maximum_residual_evaluations <= 81):
        raise DensInResidualGPUV95Error(
            "v95 CUDA residual evaluation bound is invalid")
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    placeholder = DensInResidualGPUExecutionV95(
        base_execution, DENSIN_SEDIMENT_INCREMENT, roughness, chezy_min,
        rtol, atol, maximum_dt,
        maximum_residual_evaluations, LOCKED_UPSTREAM_COMMIT, locked, "")
    return DensInResidualGPUExecutionV95(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v95(
    value: DensInResidualGPUExecutionV95,
) -> dict:
    if type(value) is not DensInResidualGPUExecutionV95:
        raise DensInResidualGPUV95Error("v95 CUDA execution type differs")
    execution_document_v85(value.base_execution)
    locked = tuple(sorted(verify_locked_sources_v85(_ROOT).items()))
    roughness = _array(
        value.manning_roughness, (value.cell_count,), "Manning roughness")
    payload = _execution_payload(value)
    if (value.densin_mode != DENSIN_SEDIMENT_INCREMENT or
            value.locked_source_sha256 != locked or
            value.locked_delft_commit != LOCKED_UPSTREAM_COMMIT or
            np.any(roughness < 0.0) or
            value.manning_chezy_minimum_depth_m < 0.0 or
            value.fingerprint != _fingerprint(payload)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class DensInResidualGPUStateV95:
    execution_fingerprint: str
    q: np.ndarray
    accepted_updates: int
    time_s: float
    accepted_chain_sha256: str
    state_sha256: str


def _state_token(value: DensInResidualGPUStateV95) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "q_sha256": _digest(value.q),
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "accepted_chain_sha256": value.accepted_chain_sha256,
    })


def _wrap_state(execution: DensInResidualGPUExecutionV95, *, q: object,
                accepted_updates: int, time_s: float,
                accepted_chain_sha256: str) -> DensInResidualGPUStateV95:
    execution_document_v95(execution)
    if (type(accepted_updates) is not int or not
            0 <= accepted_updates <= MAX_ACCEPTED_UPDATES):
        raise DensInResidualGPUV95Error(
            "v95 CUDA state update count is outside scope")
    time = _finite(time_s, "state time")
    if time < 0.0:
        raise DensInResidualGPUV95Error("v95 CUDA state time is negative")
    state_q = _array(q, (3, execution.cell_count), "hydraulic state")
    if np.any(state_q[0] <= execution.base_execution.dry_tolerance_m):
        raise DensInResidualGPUV95Error(
            "v95 CUDA state left the fully wet scope")
    placeholder = DensInResidualGPUStateV95(
        execution.fingerprint, state_q, accepted_updates, time,
        _sha(accepted_chain_sha256, "accepted chain"), "")
    return DensInResidualGPUStateV95(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_densin_residual_gpu_state_v95(
    execution: DensInResidualGPUExecutionV95, *, q: object,
) -> DensInResidualGPUStateV95:
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_q_sha256": _digest(np.asarray(q, dtype=np.float64)),
    })
    return _wrap_state(
        execution, q=q, accepted_updates=0, time_s=0.0,
        accepted_chain_sha256=chain)


def validate_densin_residual_gpu_state_v95(
    state: DensInResidualGPUStateV95,
    execution: DensInResidualGPUExecutionV95,
) -> None:
    if (type(state) is not DensInResidualGPUStateV95 or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA state is stale or mismatched")
    _wrap_state(
        execution, q=state.q, accepted_updates=state.accepted_updates,
        time_s=state.time_s,
        accepted_chain_sha256=state.accepted_chain_sha256)


@dataclass(frozen=True)
class DensInResidualGPUTrialV95:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_concentration: AcceptedConcentrationReceiptV85
    dt_s: float
    candidate: DensInResidualGPUStateV95
    residual_base_q: np.ndarray
    residual_endpoint_q: np.ndarray
    midpoint_q: np.ndarray
    mixture_density_kg_m3: np.ndarray
    base_fwave_rate: np.ndarray
    density_momentum_rate: np.ndarray
    full_rate: np.ndarray
    final_residual: np.ndarray
    face_normal_acceleration_m_s2: np.ndarray
    cell_acceleration_m_s2: np.ndarray
    face_water_flux_m2_s: np.ndarray
    midpoint_depth_sha256_by_evaluation: tuple[str, ...]
    residual_evaluation_count: int
    density_source_evaluation_count: int
    augmented_riemann_call_count: int
    approximate_jacobian_evaluation_count: int
    density_jacobian_derivative_evaluation_count: int
    post_convergence_density_application_count: int
    accepted_parent_density_mutation_count: int
    nonlinear_iterations: int
    nonlinear_residual_linf: float
    water_volume_residual_m3: float
    frozen_suspended_inventory_residual_kg: float
    backend: str
    trial_sha256: str


def _trial_token(value: DensInResidualGPUTrialV95) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_concentration_receipt_sha256":
            value.accepted_concentration.receipt_sha256,
        "dt_s": value.dt_s,
        "candidate_state_sha256": value.candidate.state_sha256,
        "arrays_sha256": _digest(
            value.residual_base_q, value.residual_endpoint_q,
            value.midpoint_q, value.mixture_density_kg_m3,
            value.base_fwave_rate, value.density_momentum_rate,
            value.full_rate, value.final_residual,
            value.face_normal_acceleration_m_s2,
            value.cell_acceleration_m_s2, value.face_water_flux_m2_s),
        "midpoint_depth_sha256_by_evaluation":
            list(value.midpoint_depth_sha256_by_evaluation),
        "residual_evaluation_count": value.residual_evaluation_count,
        "density_source_evaluation_count":
            value.density_source_evaluation_count,
        "augmented_riemann_call_count": value.augmented_riemann_call_count,
        "approximate_jacobian_evaluation_count":
            value.approximate_jacobian_evaluation_count,
        "density_jacobian_derivative_evaluation_count":
            value.density_jacobian_derivative_evaluation_count,
        "post_convergence_density_application_count":
            value.post_convergence_density_application_count,
        "accepted_parent_density_mutation_count":
            value.accepted_parent_density_mutation_count,
        "nonlinear_iterations": value.nonlinear_iterations,
        "nonlinear_residual_linf": value.nonlinear_residual_linf,
        "water_volume_residual_m3": value.water_volume_residual_m3,
        "frozen_suspended_inventory_residual_kg":
            value.frozen_suspended_inventory_residual_kg,
        "backend": value.backend,
    })


def _validate_trial(
    execution: DensInResidualGPUExecutionV95,
    parent: DensInResidualGPUStateV95,
    trial: DensInResidualGPUTrialV95,
) -> None:
    validate_densin_residual_gpu_state_v95(parent, execution)
    if (type(trial) is not DensInResidualGPUTrialV95 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != BACKEND or
            trial.trial_sha256 != _trial_token(trial)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA trial is stale or mismatched")
    try:
        validate_accepted_concentration_receipt_v85(
            trial.accepted_concentration, execution.base_execution,
            accepted_update=parent.accepted_updates, time_s=parent.time_s,
            require_typed_v78=True)
    except (TypeError, ValueError) as error:
        raise DensInResidualGPUV95Error(
            "v95 CUDA trial concentration receipt differs") from error
    validate_densin_residual_gpu_state_v95(trial.candidate, execution)
    count = trial.residual_evaluation_count
    dt = _finite(trial.dt_s, "trial timestep", positive=True)
    arrays = (
        (trial.residual_base_q, (3, execution.cell_count),
         "trial residual base"),
        (trial.residual_endpoint_q, (3, execution.cell_count),
         "trial residual endpoint"),
        (trial.midpoint_q, (3, execution.cell_count), "trial midpoint"),
        (trial.mixture_density_kg_m3, (execution.cell_count,),
         "trial mixture"),
        (trial.base_fwave_rate, (3, execution.cell_count), "trial base rate"),
        (trial.density_momentum_rate, (2, execution.cell_count),
         "trial density rate"),
        (trial.full_rate, (3, execution.cell_count), "trial full rate"),
        (trial.final_residual, (3, execution.cell_count), "trial residual"),
        (trial.face_normal_acceleration_m_s2, (execution.face_count,),
         "trial face acceleration"),
        (trial.cell_acceleration_m_s2, (2, execution.cell_count),
         "trial cell acceleration"),
        (trial.face_water_flux_m2_s, (execution.face_count,),
         "trial water flux"),
    )
    for value, shape, name in arrays:
        _array(value, shape, name)
    if (dt > execution.maximum_dt_s or
            trial.candidate.accepted_updates != parent.accepted_updates+1 or
            trial.candidate.time_s != parent.time_s+trial.dt_s or
            not 1 <= count <= execution.maximum_residual_evaluations or
            trial.density_source_evaluation_count != count or
            trial.augmented_riemann_call_count != execution.face_count*count or
            trial.approximate_jacobian_evaluation_count < 0 or
            trial.density_jacobian_derivative_evaluation_count != 0 or
            trial.post_convergence_density_application_count != 0 or
            trial.accepted_parent_density_mutation_count != 0 or
            trial.nonlinear_iterations != count-1 or
            len(trial.midpoint_depth_sha256_by_evaluation) != count or
            trial.frozen_suspended_inventory_residual_kg != 0.0):
        raise DensInResidualGPUV95Error(
            "v95 CUDA trial counts or ledgers differ")
    if (any(_SHA256.fullmatch(value) is None for value in
            trial.midpoint_depth_sha256_by_evaluation) or
            trial.midpoint_depth_sha256_by_evaluation[-1] !=
            _digest(trial.midpoint_q[0])):
        raise DensInResidualGPUV95Error(
            "v95 CUDA midpoint depth-history identity differs")
    if (not np.array_equal(trial.accepted_concentration.depth_m, parent.q[0]) or
            not np.allclose(
                trial.mixture_density_kg_m3,
                trial.accepted_concentration.mixture_density_kg_m3,
                rtol=2*np.finfo(float).eps, atol=0.0) or
            not np.array_equal(trial.full_rate[0],
                               trial.base_fwave_rate[0])):
        raise DensInResidualGPUV95Error(
            "v95 CUDA accepted density or mass-rate ownership differs")
    expected_full = np.array(trial.base_fwave_rate, copy=True)
    expected_full[1:] += trial.density_momentum_rate
    if not np.array_equal(expected_full, trial.full_rate):
        raise DensInResidualGPUV95Error(
            "v95 CUDA full-rate composition differs")
    expected_residual = (
        trial.midpoint_q-trial.residual_base_q-
        0.5*trial.dt_s*trial.full_rate)
    residual_scale = max(1.0, float(np.max(np.abs(expected_residual))))
    if (not np.allclose(expected_residual, trial.final_residual,
                        rtol=16*np.finfo(float).eps,
                        atol=16*np.finfo(float).eps*residual_scale) or
            trial.nonlinear_residual_linf !=
                float(np.max(np.abs(trial.final_residual)))):
        raise DensInResidualGPUV95Error(
            "v95 CUDA nonlinear residual evidence differs")
    l2_residual = float(np.linalg.norm(trial.final_residual))
    update_norm = float(np.linalg.norm(
        trial.midpoint_q-trial.residual_base_q))
    forcing_norm = float(np.linalg.norm(0.5*trial.dt_s*trial.full_rate))
    tolerance = max(
        execution.nonlinear_atol*math.sqrt(3.0*execution.cell_count),
        execution.nonlinear_rtol*max(update_norm, forcing_norm))
    if not math.isfinite(l2_residual) or l2_residual > tolerance:
        raise DensInResidualGPUV95Error(
            "v95 CUDA nonlinear residual did not converge")
    endpoint = (2.0*trial.midpoint_q-trial.residual_base_q-
                2.0*trial.final_residual)
    endpoint_scale = max(1.0, float(np.max(np.abs(endpoint))))
    if not np.allclose(
            endpoint, trial.residual_endpoint_q,
            rtol=128*np.finfo(float).eps,
            atol=128*np.finfo(float).eps*endpoint_scale):
        raise DensInResidualGPUV95Error(
            "v95 CUDA implicit endpoint identity differs")
    if (not np.array_equal(trial.residual_base_q[0], parent.q[0]) or
            not np.array_equal(trial.candidate.q[0],
                               trial.residual_endpoint_q[0])):
        raise DensInResidualGPUV95Error(
            "v95 CUDA split friction changed water depth")
    expected_base = _manning_half_step_oracle(
        parent.q, execution.manning_roughness, 0.5*dt,
        gravity_m_s2=execution.base_execution.gravity_m_s2,
        dry_tolerance_m=execution.base_execution.dry_tolerance_m,
        chezy_minimum_depth_m=execution.manning_chezy_minimum_depth_m)
    expected_candidate = _manning_half_step_oracle(
        trial.residual_endpoint_q, execution.manning_roughness, 0.5*dt,
        gravity_m_s2=execution.base_execution.gravity_m_s2,
        dry_tolerance_m=execution.base_execution.dry_tolerance_m,
        chezy_minimum_depth_m=execution.manning_chezy_minimum_depth_m)
    friction_scale = max(
        1.0, float(np.max(np.abs(expected_base))),
        float(np.max(np.abs(expected_candidate))))
    if (not np.allclose(
            trial.residual_base_q, expected_base,
            rtol=64*np.finfo(float).eps,
            atol=64*np.finfo(float).eps*friction_scale) or
            not np.allclose(
                trial.candidate.q, expected_candidate,
                rtol=64*np.finfo(float).eps,
                atol=64*np.finfo(float).eps*friction_scale)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA authenticated Manning split differs")
    face = delft_kmax1_face_acceleration_v85(
        trial.midpoint_q[0], trial.mixture_density_kg_m3,
        execution.base_execution.owner, execution.base_execution.neighbour,
        execution.base_execution.face_distance_m,
        gravity_m_s2=execution.base_execution.gravity_m_s2,
        hydraulic_reference_density_kg_m3=
            execution.base_execution.hydraulic_reference_density_kg_m3)
    cell = project_face_acceleration_v85(
        face, execution.base_execution.normal,
        execution.base_execution.face_length_m,
        execution.base_execution.cell_face_offset,
        execution.base_execution.cell_faces,
        execution.base_execution.cell_face_sign)
    density = trial.midpoint_q[0][None, :]*cell
    scale = max(1.0, float(np.max(np.abs(density))))
    if (not np.allclose(face, trial.face_normal_acceleration_m_s2,
                        rtol=8*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(cell, trial.cell_acceleration_m_s2,
                            rtol=16*np.finfo(float).eps, atol=1e-18) or
            not np.allclose(density, trial.density_momentum_rate,
                            rtol=16*np.finfo(float).eps,
                            atol=16*np.finfo(float).eps*scale)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA source differs from the locked v85 equation oracle")
    water_scale = max(1.0, float(np.sum(
        execution.base_execution.cell_area_m2*parent.q[0])))
    boundary = execution.base_execution.neighbour < 0
    reconstructed_water_residual = float(np.sum(
        execution.base_execution.cell_area_m2*
        (trial.candidate.q[0]-parent.q[0])) + trial.dt_s*np.sum(
            execution.base_execution.face_length_m[boundary]*
            trial.face_water_flux_m2_s[boundary]))
    if (abs(trial.water_volume_residual_m3-reconstructed_water_residual) >
            128*np.finfo(float).eps*water_scale or
            abs(trial.water_volume_residual_m3) > max(
                1.e-12, 1.e-11*water_scale)):
        raise DensInResidualGPUV95Error(
            "v95 CUDA authenticated water budget differs")


class ResidentDensInResidualV95:
    """One explicitly selected resident DensIn residual owner."""

    is_v95_densin_residual_consumer = True

    def __init__(self, execution: DensInResidualGPUExecutionV95):
        execution_document_v95(execution)
        self.execution = execution
        self.operator = ResidentDensInFwaveV85(execution.base_execution)
        self.cp = self.operator.cp
        self.device_id = self.operator.device_id
        self.nc, self.nf = execution.cell_count, execution.face_count
        self.accepted_q = self.cp.empty((3, self.nc), dtype=self.cp.float64)
        self.residual_base = self.cp.empty_like(self.accepted_q)
        self.residual_endpoint = self.cp.empty_like(self.accepted_q)
        self.midpoint_last = self.cp.empty_like(self.accepted_q)
        self.base_rate_last = self.cp.empty_like(self.accepted_q)
        self.full_rate_last = self.cp.empty_like(self.accepted_q)
        self.depth_history = self.cp.empty(
            (execution.maximum_residual_evaluations, self.nc),
            dtype=self.cp.float64)
        self._controller = None
        self._parent = None
        self._accepted_concentration = None
        self._frozen_mixture_host = None
        self._dt = self._time = None
        self._initial_fwave_evaluations = None
        self._initial_linear_matvecs = None
        self._initial_kernel_counts = None
        self._residual_evaluations = 0
        self._residual_endpoint_captured = False
        self._open = False
        self._pending = None
        self._cancelled_callback = False
        self._resident_parent_sha256 = None
        self.last_accepted_receipt = None
        self.kernel_launch_counts = {
            "v85_mixture_production": 0,
            "v85_face_residual": 0,
            "v85_projection_residual": 0,
            "resident_rate_add": 0,
            "host_numerical_fallback": 0,
        }
        self._v85_runtime_receipt_sha256 = _fingerprint(
            self.operator.runtime_receipt)
        sources = (
            _ROOT/"braids_core/sediment/densin_fwave_v85.py",
            _ROOT/"braids_gpu/densin_fwave_v85.py",
            _ROOT/"braids_gpu/densin_fwave_v85.cu",
            Path(__file__).resolve(),
            _ROOT/"braids_gpu/implicit_midpoint_fwave.py",
        )
        self._source_paths = sources
        properties = self.cp.cuda.runtime.getDeviceProperties(self.device_id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        self.runtime_receipt = {
            "scope_id": SCOPE_ID + ":runtime",
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "backend": BACKEND,
            "device_id": self.device_id,
            "device_name": str(name),
            "actual_cuda_kernels": True,
            "density_inside_authoritative_residual": True,
            "accepted_parent_density_frozen": True,
            "midpoint_depth_reevaluated": True,
            "post_convergence_density_application_count": 0,
            "approximate_jacobian_density_derivative": "omitted",
            "uniform_accepted_water_density_only": True,
            "automatic_cpu_fallback": False,
            "locked_delft_source_sha256":
                dict(execution.locked_source_sha256),
            "locked_delft_commit": execution.locked_delft_commit,
            "nested_v85_runtime_receipt_sha256":
                self._v85_runtime_receipt_sha256,
            "source_sha256": {
                str(path.relative_to(_ROOT)).replace("\\", "/"):
                    _file_sha256(path) for path in sources
            },
        }
        self.runtime_receipt["receipt_sha256"] = _fingerprint(
            self.runtime_receipt)

    def _device(self, value, shape: tuple[int, ...], name: str):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and value.shape == shape and
                 value.dtype == cp.float64 and value.flags.c_contiguous and
                 int(value.device.id) == self.device_id,
                 f"v95 CUDA {name} must be resident contiguous float64 {shape}")
        return value

    def _readback(self, value):
        """Single injectable acceptance-readback boundary."""
        return self.cp.asnumpy(value)

    def _validate_runtime_identity(self, *, require_controller: bool) -> None:
        _require(_fingerprint(self.operator.runtime_receipt) ==
                 self._v85_runtime_receipt_sha256,
                 "v95 CUDA nested v85 runtime identity differs")
        expected = {
            str(path.relative_to(_ROOT)).replace("\\", "/"):
                _file_sha256(path) for path in self._source_paths
        }
        _require(self.runtime_receipt.get("source_sha256") == expected,
                 "v95 CUDA runtime source identity differs")
        if require_controller:
            _require(self._controller is not None and
                     self.runtime_receipt.get("controller_cuda_source_sha256")
                     == self._controller.source_sha256 and
                     _file_sha256(Path(self._controller.source_path)) ==
                     self._controller.source_sha256,
                     "v95 CUDA controller runtime identity differs")
        _require(self.runtime_receipt.get("receipt_sha256") == _fingerprint({
            key: value for key, value in self.runtime_receipt.items()
            if key != "receipt_sha256"
        }), "v95 CUDA runtime receipt differs")

    def _validate_bound_controller_state(self) -> None:
        """Reauthenticate every mutable controller input before a trial.

        ``bind_controller`` proves provenance at attachment time, but CuPy
        arrays remain mutable.  The execution claims exact fixed geometry and
        a fully-wet domain, so a later mutation must be rejected before any
        numerical launch rather than becoming self-consistent trial evidence.
        """
        controller = self._controller
        _require(controller is not None,
                 "v95 CUDA owner must be bound to the live controller")
        cp = self.cp
        _require(getattr(controller, "nc", None) == self.nc and
                 getattr(controller, "nf", None) == self.nf and
                 getattr(controller, "cp", None) is cp,
                 "v95 CUDA controller dimensions/backend differ")
        _require(bool(cp.array_equal(
                     controller.active,
                     cp.ones(self.nc, dtype=cp.int32)).item()),
                 "v95 CUDA owner requires a fully wet fixed grid")
        for actual, expected, name in (
            (controller.owner, self.execution.base_execution.owner, "owner"),
            (controller.neighbour, self.execution.base_execution.neighbour,
             "neighbour"),
            (controller.normal, self.execution.base_execution.normal,
             "normal"),
            (controller.length,
             self.execution.base_execution.face_length_m, "face length"),
            (controller.area, self.execution.base_execution.cell_area_m2,
             "cell area"),
            (controller.offset,
             self.execution.base_execution.cell_face_offset,
             "cell-face offset"),
            (controller.faces, self.execution.base_execution.cell_faces,
             "cell faces"),
            (controller.signs,
             self.execution.base_execution.cell_face_sign,
             "cell-face signs"),
        ):
            _require(bool(cp.array_equal(
                         actual, cp.asarray(expected)).item()),
                     "v95 CUDA controller geometry differs: " + name)
        manning = controller.solver.boundary.arrays.get("manning")
        _require(controller.roughness_mode == "manning" and
                 manning is not None and
                 bool(cp.array_equal(
                     manning,
                     cp.asarray(self.execution.manning_roughness)).item()) and
                 float(controller.solver.stepper.hydro.chezy_min) ==
                     self.execution.manning_chezy_minimum_depth_m and
                 getattr(controller.solver, "diffusion", None) is None,
                 "v95 CUDA bounded proof friction/viscosity differs")
        _require(float(controller.grav) ==
                 self.execution.base_execution.gravity_m_s2 and
                 float(controller.dry) ==
                 self.execution.base_execution.dry_tolerance_m,
                 "v95 CUDA controller gravity/dry threshold differs")
        _require(controller.nonlinear_maxiter+1 <=
                 self.execution.maximum_residual_evaluations and
                 controller.nonlinear_rtol == self.execution.nonlinear_rtol and
                 controller.nonlinear_atol == self.execution.nonlinear_atol,
                 "v95 CUDA controller nonlinear contract differs")
        _require(not controller.component_composition and
                 not controller.prognostic_secondary_composition and
                 not controller.suspended_composition and
                 controller.shared_turbulence_v91 is None,
                 "v95 CUDA bounded proof excludes other residual/scalar owners")
        controller_source = Path(controller.source_path).resolve()
        _require(controller_source.is_file() and
                 not controller_source.is_symlink() and
                 _file_sha256(controller_source) == controller.source_sha256 ==
                 self.runtime_receipt.get("controller_cuda_source_sha256"),
                 "v95 CUDA controller runtime identity differs")

    def bind_controller(self, controller) -> None:
        _require(self._controller is None or self._controller is controller,
                 "v95 CUDA owner is already bound to another controller")
        cp = self.cp
        _require(getattr(controller, "nc", None) == self.nc and
                 getattr(controller, "nf", None) == self.nf and
                 getattr(controller, "cp", None) is cp,
                 "v95 CUDA controller dimensions/backend differ")
        _require(bool(cp.all(controller.active == 1).item()),
                 "v95 CUDA owner requires a fully wet fixed grid")
        for actual, expected, name in (
            (controller.owner, self.execution.base_execution.owner, "owner"),
            (controller.neighbour, self.execution.base_execution.neighbour,
             "neighbour"),
            (controller.normal, self.execution.base_execution.normal, "normal"),
            (controller.length, self.execution.base_execution.face_length_m,
             "face length"),
            (controller.area, self.execution.base_execution.cell_area_m2,
             "cell area"),
            (controller.offset, self.execution.base_execution.cell_face_offset,
             "cell-face offset"),
            (controller.faces, self.execution.base_execution.cell_faces,
             "cell faces"),
            (controller.signs, self.execution.base_execution.cell_face_sign,
             "cell-face signs"),
        ):
            expected_device = cp.asarray(expected)
            _require(bool(cp.array_equal(actual, expected_device).item()),
                     "v95 CUDA controller geometry differs: " + name)
        _require(controller.nonlinear_maxiter+1 <=
                 self.execution.maximum_residual_evaluations,
                 "v95 CUDA controller nonlinear bound exceeds execution")
        _require(controller.nonlinear_rtol == self.execution.nonlinear_rtol and
                 controller.nonlinear_atol == self.execution.nonlinear_atol,
                 "v95 CUDA controller nonlinear tolerances differ")
        _require(not controller.component_composition and
                 not controller.prognostic_secondary_composition and
                 not controller.suspended_composition and
                 controller.shared_turbulence_v91 is None,
                 "v95 CUDA bounded proof excludes other residual/scalar owners")
        manning = controller.solver.boundary.arrays.get("manning")
        controller_chezy_min = float(
            controller.solver.stepper.hydro.chezy_min)
        _require(controller.roughness_mode == "manning" and
                 manning is not None and
                 bool(cp.array_equal(
                     manning,
                     cp.asarray(self.execution.manning_roughness)).item()) and
                 controller_chezy_min ==
                     self.execution.manning_chezy_minimum_depth_m and
                 getattr(controller.solver, "diffusion", None) is None,
                 "v95 CUDA bounded proof friction/viscosity differs")
        _require(float(controller.grav) ==
                 self.execution.base_execution.gravity_m_s2 and
                 float(controller.dry) ==
                 self.execution.base_execution.dry_tolerance_m,
                 "v95 CUDA controller gravity/dry threshold differs")
        _require(hasattr(controller, "source_path") and
                 hasattr(controller, "source_sha256"),
                 "v95 CUDA controller source identity is unavailable")
        controller_source = Path(controller.source_path).resolve()
        default_controller_source = (
            _ROOT/"braids_gpu/implicit_midpoint_fwave.cu").resolve()
        expected_controller_sha = _file_sha256(default_controller_source)
        _require(controller_source.is_file() and
                 not controller_source.is_symlink() and
                 _file_sha256(controller_source) == controller.source_sha256 ==
                     expected_controller_sha,
                 "v95 CUDA rejects an unauthenticated/custom controller source")
        try:
            recorded_controller_source = str(
                controller_source.relative_to(_ROOT)).replace("\\", "/")
        except ValueError:
            recorded_controller_source = str(controller_source)
        self.runtime_receipt["controller_cuda_source"] = \
            recorded_controller_source
        self.runtime_receipt["controller_cuda_source_sha256"] = \
            _file_sha256(controller_source)
        self.runtime_receipt["receipt_sha256"] = _fingerprint({
            key: value for key, value in self.runtime_receipt.items()
            if key != "receipt_sha256"
        })
        self._controller = controller
        self._validate_runtime_identity(require_controller=True)
        self._validate_bound_controller_state()

    def _sync_parent(self, parent: DensInResidualGPUStateV95) -> None:
        if self._resident_parent_sha256 != parent.state_sha256:
            self.accepted_q.set(parent.q)
            self._resident_parent_sha256 = parent.state_sha256

    def _after_face_density(self, midpoint_q) -> None:
        """Additive v96 seam between face production and one projection.

        Historical v95 has no exterior-density override.  The v96 subclass
        uses this exact point to overwrite authenticated physical-open faces
        before the unchanged single face-to-cell projection.
        """
        return None

    def begin_hydraulic_trial(
        self, parent: DensInResidualGPUStateV95, *,
        accepted_concentration: AcceptedConcentrationReceiptV85,
        dt_s: float, time_s: float,
        cancelled: bool | Callable[[], bool] = False,
    ) -> None:
        _require(self._controller is not None,
                 "v95 CUDA owner must be bound to the live controller")
        _require(not self._open and self._pending is None,
                 "v95 CUDA previous trial is unfinished")
        validate_densin_residual_gpu_state_v95(parent, self.execution)
        self._validate_runtime_identity(require_controller=True)
        self._validate_bound_controller_state()
        _require(not _cancelled(cancelled),
                 "v95 CUDA trial cancelled before any launch")
        _require(parent.accepted_updates < MAX_ACCEPTED_UPDATES,
                 "v95 CUDA accepted update limit was reached")
        dt = _finite(dt_s, "timestep", positive=True)
        _require(dt <= self.execution.maximum_dt_s,
                 "v95 CUDA timestep exceeds the bounded limit")
        _require(float(time_s) == parent.time_s,
                 "v95 CUDA trial clock differs from its parent")
        try:
            validate_accepted_concentration_receipt_v85(
                accepted_concentration, self.execution.base_execution,
                accepted_update=parent.accepted_updates,
                time_s=parent.time_s, require_typed_v78=True)
        except (TypeError, ValueError) as error:
            raise DensInResidualGPUV95Error(
                "v95 CUDA requires the current typed v85 concentration handoff"
            ) from error
        _require(np.array_equal(parent.q[0], accepted_concentration.depth_m),
                 "v95 CUDA concentration depth differs from parent hydraulics")
        self._sync_parent(parent)
        self._cancelled_callback = cancelled
        self._initial_kernel_counts = dict(self.kernel_launch_counts)
        try:
            self.operator.concentration.set(
                accepted_concentration.concentration_kg_m3)
            self.operator.cell_status.fill(0)
            self.kernel_launch_counts["v85_mixture_production"] += 1
            self.operator._launch("densin_mixture_v85", self.nc, (
                self.operator.concentration, self.operator.water,
                self.operator.solid, np.int64(self.operator.ng),
                np.int64(self.nc), self.operator.mixture,
                self.operator.cell_status))
            _require(not bool(self.cp.any(
                         self.operator.cell_status != 0).item()),
                     "v95 CUDA mixture-density kernel failed")
            mixture = np.ascontiguousarray(
                self._readback(self.operator.mixture))
            _require(np.allclose(
                         mixture,
                         accepted_concentration.mixture_density_kg_m3,
                         rtol=2*np.finfo(float).eps, atol=0.0),
                     "v95 CUDA accepted mixture density differs")
            _require(not _cancelled(self._cancelled_callback),
                     "v95 CUDA trial cancelled after mixture launch")
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._parent = parent
        self._accepted_concentration = accepted_concentration
        self._frozen_mixture_host = mixture
        self._dt, self._time = dt, float(time_s)
        self._initial_fwave_evaluations = int(
            self._controller.fwave_evaluations)
        self._initial_linear_matvecs = int(self._controller.linear_matvecs)
        self._residual_evaluations = 0
        self._open = True

    def require_open_for_controller(self, dt_s: float, time_s: float, q) -> None:
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise DensInResidualGPUV95Error(
                "v95 CUDA trial cancelled before controller launch")
        _require(self._open and self._pending is None,
                 "v95 CUDA controller entered without an open trial")
        _require(float(dt_s) == self._dt and float(time_s) == self._time,
                 "v95 CUDA controller clock differs from its handoff")
        resident = self._device(q, (3, self.nc), "controller accepted state")
        _require(np.array_equal(self._readback(resident), self._parent.q),
                 "v95 CUDA controller accepted state differs from parent")

    def add_momentum_rate(self, q, full_rate) -> None:
        """Add DensIn once in every authoritative controller residual."""
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise DensInResidualGPUV95Error(
                "v95 CUDA trial cancelled before density residual launch")
        _require(self._open and self._pending is None,
                 "v95 CUDA density consumer has no open trial")
        q = self._device(q, (3, self.nc), "midpoint state")
        full_rate = self._device(full_rate, (3, self.nc), "full residual rate")
        row = self._residual_evaluations
        _require(row < self.execution.maximum_residual_evaluations,
                 "v95 CUDA residual evaluation bound was exceeded")
        try:
            self.cp.copyto(self.depth_history[row], q[0])
            self.cp.copyto(self.midpoint_last, q)
            self.cp.copyto(self.base_rate_last, full_rate)
            self.operator.face_status.fill(0)
            self.kernel_launch_counts["v85_face_residual"] += 1
            self.operator._launch("densin_face_v85", self.nf, (
                q[0], self.operator.mixture, self.operator.owner,
                self.operator.neighbour, self.operator.distance,
                np.int64(self.nf),
                np.float64(self.execution.base_execution.gravity_m_s2),
                np.float64(self.execution.base_execution.
                           hydraulic_reference_density_kg_m3),
                self.operator.face_acceleration,
                self.operator.face_status))
            _require(not bool(self.cp.any(
                         self.operator.face_status != 0).item()),
                     "v95 CUDA face-density kernel failed")
            self._after_face_density(q)
            _require(not bool(self.cp.any(
                         self.operator.face_status != 0).item()),
                     "v95 CUDA exterior face-density kernel failed")
            self.operator.cell_status.fill(0)
            self.kernel_launch_counts["v85_projection_residual"] += 1
            self.operator._launch("densin_project_stride_v85", self.nc, (
                self.operator.face_acceleration, self.operator.normal,
                self.operator.length, self.operator.offset,
                self.operator.faces, self.operator.signs, q[0],
                np.int64(self.nc), np.int64(self.nf),
                self.operator.cell_acceleration,
                self.operator.density_rate, self.operator.cell_status))
            _require(not bool(self.cp.any(
                         self.operator.cell_status != 0).item()),
                     "v95 CUDA face-to-cell projection failed")
            self.kernel_launch_counts["resident_rate_add"] += 1
            full_rate[1:] += self.operator.density_rate
            self.cp.copyto(self.full_rate_last, full_rate)
            _require(bool(self.cp.isfinite(full_rate).all().item()),
                     "v95 CUDA density residual produced non-finite rate")
            _require(not _cancelled(self._cancelled_callback),
                     "v95 CUDA trial cancelled after density residual launch")
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._residual_evaluations += 1

    def capture_converged_residual_endpoint(self, residual_base_q,
                                             residual_endpoint_q) -> None:
        """Bind the pre/post-residual states before second split friction."""
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise DensInResidualGPUV95Error(
                "v95 CUDA trial cancelled before endpoint handoff")
        _require(self._open and self._pending is None and
                 not self._residual_endpoint_captured,
                 "v95 CUDA residual endpoint handoff differs")
        base = self._device(
            residual_base_q, (3, self.nc), "residual base state")
        endpoint = self._device(
            residual_endpoint_q, (3, self.nc), "residual endpoint state")
        try:
            self.cp.copyto(self.residual_base, base)
            self.cp.copyto(self.residual_endpoint, endpoint)
            _require(bool(self.cp.isfinite(self.residual_base).all().item()) and
                     bool(self.cp.isfinite(
                         self.residual_endpoint).all().item()),
                     "v95 CUDA residual endpoint handoff is non-finite")
            _require(not _cancelled(self._cancelled_callback),
                     "v95 CUDA trial cancelled after endpoint handoff")
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._residual_endpoint_captured = True

    def finish_hydraulic_trial(
        self, midpoint_q, accepted_q, accepted_water_flux, *,
        dt_s: float, time_s: float,
    ) -> dict:
        if _cancelled(self._cancelled_callback):
            self.abort_hydraulic_trial()
            raise DensInResidualGPUV95Error(
                "v95 CUDA trial cancelled before hydraulic finish")
        _require(self._open and self._pending is None,
                 "v95 CUDA hydraulic finish has no open trial")
        _require(self._residual_endpoint_captured,
                 "v95 CUDA residual endpoint was not captured")
        _require(float(dt_s) == self._dt and float(time_s) == self._time,
                 "v95 CUDA hydraulic finish clock differs")
        midpoint = self._device(
            midpoint_q, (3, self.nc), "accepted midpoint")
        endpoint = self._device(
            accepted_q, (3, self.nc), "accepted endpoint")
        water = self._device(
            accepted_water_flux, (self.nf,), "accepted midpoint water flux")
        try:
            _require(bool(self.cp.array_equal(
                         water, self._controller.midpoint_water).item()),
                     "v95 CUDA accepted water flux differs from the "
                     "authoritative controller midpoint flux")
            count = self._residual_evaluations
            _require(count >= 1 and
                     int(self._controller.fwave_evaluations) -
                     self._initial_fwave_evaluations == count,
                     "v95 CUDA source/f-wave residual counts differ")
            launch_delta = {
                name: self.kernel_launch_counts[name]-
                    self._initial_kernel_counts[name]
                for name in self.kernel_launch_counts
            }
            _require(launch_delta == {
                         "v85_mixture_production": 1,
                         "v85_face_residual": count,
                         "v85_projection_residual": count,
                         "resident_rate_add": count,
                         "host_numerical_fallback": 0,
                     }, "v95 CUDA per-trial kernel ownership differs")
            midpoint_h = np.ascontiguousarray(self._readback(midpoint))
            endpoint_h = np.ascontiguousarray(self._readback(endpoint))
            residual_base_h = np.ascontiguousarray(
                self._readback(self.residual_base))
            residual_endpoint_h = np.ascontiguousarray(
                self._readback(self.residual_endpoint))
            water_h = np.ascontiguousarray(self._readback(water))
            recorded_midpoint = np.ascontiguousarray(
                self._readback(self.midpoint_last))
            base_rate = np.ascontiguousarray(
                self._readback(self.base_rate_last))
            density_rate = np.ascontiguousarray(
                self._readback(self.operator.density_rate))
            full_rate = np.ascontiguousarray(
                self._readback(self.full_rate_last))
            face = np.ascontiguousarray(
                self._readback(self.operator.face_acceleration))
            cell = np.ascontiguousarray(
                self._readback(self.operator.cell_acceleration))
            mixture = np.ascontiguousarray(
                self._readback(self.operator.mixture))
            concentration = np.ascontiguousarray(
                self._readback(self.operator.concentration))
            history = np.ascontiguousarray(
                self._readback(self.depth_history[:count]))
            final_residual = np.ascontiguousarray(
                self._readback(self._controller.residual))
            _require(np.array_equal(midpoint_h, recorded_midpoint),
                     "v95 CUDA accepted midpoint differs from final source input")
            _require(np.array_equal(mixture, self._frozen_mixture_host) and
                     np.array_equal(
                         concentration,
                         self._accepted_concentration.concentration_kg_m3),
                     "v95 CUDA accepted parent density mutated")
            _require(not _cancelled(self._cancelled_callback),
                     "v95 CUDA trial cancelled after hydraulic readback")
            depth_digests = tuple(_digest(row) for row in history)
            _require(all(_SHA256.fullmatch(value) is not None
                         for value in depth_digests) and
                     depth_digests[-1] == _digest(midpoint_h[0]),
                     "v95 CUDA midpoint depth history differs")
            chain = _fingerprint({
                "previous": self._parent.accepted_chain_sha256,
                "accepted_concentration_receipt":
                    self._accepted_concentration.receipt_sha256,
                "runtime_receipt": self.runtime_receipt["receipt_sha256"],
                "dt_s": self._dt,
                "midpoint_sha256": _digest(midpoint_h),
                "residual_base_sha256": _digest(residual_base_h),
                "residual_endpoint_sha256": _digest(residual_endpoint_h),
                "full_rate_sha256": _digest(full_rate),
                "residual_evaluation_count": count,
                "density_source_evaluation_count": count,
                "post_convergence_density_application_count": 0,
            })
            candidate = _wrap_state(
                self.execution, q=endpoint_h,
                accepted_updates=self._parent.accepted_updates+1,
                time_s=self._parent.time_s+self._dt,
                accepted_chain_sha256=chain)
            mass_delta = float(np.sum(
                self.execution.base_execution.cell_area_m2*
                (endpoint_h[0]-self._parent.q[0])))
            boundary_delta = float(self.cp.sum(
                self._controller.water_ledger).item())
            water_residual = mass_delta-boundary_delta
            nonlinear_linf = float(np.max(np.abs(final_residual)))
            approximate_jacobian_evaluations = int(
                self._controller.linear_matvecs) - self._initial_linear_matvecs
            _require(approximate_jacobian_evaluations >= 0,
                     "v95 CUDA approximate-Jacobian count differs")
            placeholder = DensInResidualGPUTrialV95(
                self.execution.fingerprint, self._parent.state_sha256,
                self._accepted_concentration, self._dt, candidate,
                _array(residual_base_h, (3, self.nc),
                       "trial residual base"),
                _array(residual_endpoint_h, (3, self.nc),
                       "trial residual endpoint"),
                _array(midpoint_h, (3, self.nc), "trial midpoint"),
                _array(mixture, (self.nc,), "trial mixture", positive=True),
                _array(base_rate, (3, self.nc), "trial base rate"),
                _array(density_rate, (2, self.nc), "trial density rate"),
                _array(full_rate, (3, self.nc), "trial full rate"),
                _array(final_residual, (3, self.nc), "trial residual"),
                _array(face, (self.nf,), "trial face acceleration"),
                _array(cell, (2, self.nc), "trial cell acceleration"),
                _array(water_h, (self.nf,), "trial water flux"),
                depth_digests, count, count, self.nf*count,
                approximate_jacobian_evaluations, 0, 0, 0,
                count-1, nonlinear_linf, water_residual, 0.0, BACKEND, "")
            trial = DensInResidualGPUTrialV95(
                **{**placeholder.__dict__,
                   "trial_sha256": _trial_token(placeholder)})
            _validate_trial(self.execution, self._parent, trial)
        except Exception:
            self.abort_hydraulic_trial()
            raise
        self._pending = trial
        self._open = False
        return self.hydraulic_receipt_document(trial)

    @staticmethod
    def hydraulic_receipt_document(trial: DensInResidualGPUTrialV95) -> dict:
        return {
            "scope_id": SCOPE_ID + ":hydraulic-trial-receipt",
            "trial_sha256": trial.trial_sha256,
            "accepted_concentration_receipt_sha256":
                trial.accepted_concentration.receipt_sha256,
            "residual_evaluation_count": trial.residual_evaluation_count,
            "density_source_evaluation_count":
                trial.density_source_evaluation_count,
            "augmented_riemann_call_count":
                trial.augmented_riemann_call_count,
            "approximate_jacobian_evaluation_count":
                trial.approximate_jacobian_evaluation_count,
            "density_jacobian_derivative_evaluation_count": 0,
            "post_convergence_density_application_count": 0,
            "accepted_parent_density_mutation_count": 0,
            "backend": BACKEND,
            "outer_transaction_status": "provisional",
        }

    @property
    def pending_trial(self) -> DensInResidualGPUTrialV95 | None:
        return self._pending

    def commit(
        self, parent: DensInResidualGPUStateV95,
        trial: DensInResidualGPUTrialV95,
    ) -> DensInResidualGPUStateV95:
        _require(self._pending is trial,
                 "v95 CUDA trial is not the resident pending transaction")
        _validate_trial(self.execution, parent, trial)
        try:
            self.accepted_q.set(trial.candidate.q)
        except Exception:
            self.accepted_q.set(parent.q)
            self._resident_parent_sha256 = parent.state_sha256
            raise
        accepted = self.hydraulic_receipt_document(trial)
        accepted.update(
            outer_transaction_status="accepted",
            accepted_state_sha256=trial.candidate.state_sha256)
        self.last_accepted_receipt = accepted
        self._resident_parent_sha256 = trial.candidate.state_sha256
        self._clear()
        return trial.candidate

    def rollback(
        self, parent: DensInResidualGPUStateV95,
        trial: DensInResidualGPUTrialV95,
    ) -> DensInResidualGPUStateV95:
        _require(self._pending is trial,
                 "v95 CUDA trial is not the resident pending transaction")
        _validate_trial(self.execution, parent, trial)
        self._sync_parent(parent)
        self._clear()
        return parent

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def abort_hydraulic_trial(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._parent = None
        self._accepted_concentration = None
        self._frozen_mixture_host = None
        self._dt = self._time = None
        self._initial_fwave_evaluations = None
        self._initial_linear_matvecs = None
        self._initial_kernel_counts = None
        self._residual_evaluations = 0
        self._residual_endpoint_captured = False
        self._open = False
        self._pending = None
        self._cancelled_callback = False

    def checkpoint(self, state: DensInResidualGPUStateV95) -> dict:
        _require(not self._open and self._pending is None,
                 "v95 CUDA checkpoint requires a committed point")
        validate_densin_residual_gpu_state_v95(state, self.execution)
        self._validate_runtime_identity(require_controller=True)
        self._sync_parent(state)
        resident = np.ascontiguousarray(self._readback(self.accepted_q))
        _require(np.array_equal(resident, state.q),
                 "v95 CUDA resident state differs from checkpoint state")
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "semantic_id": SEMANTIC_ID,
            "execution_fingerprint": self.execution.fingerprint,
            "runtime_receipt_sha256": self.runtime_receipt["receipt_sha256"],
            "q": state.q.tolist(),
            "accepted_updates": state.accepted_updates,
            "time_s": state.time_s,
            "accepted_chain_sha256": state.accepted_chain_sha256,
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(
        self, document: Mapping[str, object],
    ) -> DensInResidualGPUStateV95:
        _require(not self._open and self._pending is None,
                 "v95 CUDA restart cannot replace an open trial")
        self._validate_runtime_identity(require_controller=True)
        expected = {
            "schema", "scope_id", "semantic_id", "execution_fingerprint",
            "runtime_receipt_sha256", "q", "accepted_updates", "time_s",
            "accepted_chain_sha256", "state_sha256", "fingerprint",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise DensInResidualGPUV95Error(
                "v95 CUDA checkpoint inventory differs")
        body = {key: document[key] for key in document if key != "fingerprint"}
        if (document["fingerprint"] != _fingerprint(body) or
                document["schema"] != SCHEMA or
                document["scope_id"] != CHECKPOINT_SCOPE_ID or
                document["semantic_id"] != SEMANTIC_ID or
                document["execution_fingerprint"] != self.execution.fingerprint or
                document["runtime_receipt_sha256"] !=
                    self.runtime_receipt["receipt_sha256"]):
            raise DensInResidualGPUV95Error(
                "v95 CUDA checkpoint identity differs")
        state = _wrap_state(
            self.execution, q=document["q"],
            accepted_updates=document["accepted_updates"],
            time_s=document["time_s"],
            accepted_chain_sha256=document["accepted_chain_sha256"])
        if state.state_sha256 != document["state_sha256"]:
            raise DensInResidualGPUV95Error(
                "v95 CUDA checkpoint state digest differs")
        previous = self.cp.asnumpy(self.accepted_q)
        previous_sha = self._resident_parent_sha256
        previous_receipt = self.last_accepted_receipt
        try:
            self.accepted_q.set(state.q)
            resident = np.ascontiguousarray(self._readback(self.accepted_q))
            _require(np.array_equal(resident, state.q),
                     "v95 CUDA restored resident state differs")
        except Exception:
            self.accepted_q.set(previous)
            self._resident_parent_sha256 = previous_sha
            self.last_accepted_receipt = previous_receipt
            raise
        self._resident_parent_sha256 = state.state_sha256
        self.last_accepted_receipt = None
        return state


def method_identity_v95_cuda() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "spatial_flux": "unchanged_live_augmented_geoclaw_fwave",
        "density_equation": "v85_locked_delft3d_kmax1_sediment_increment",
        "density_in_nonlinear_residual": True,
        "post_convergence_density_application_count": 0,
        "accepted_parent_density_frozen": True,
        "midpoint_depth_recomputed": True,
        "approximate_jacobian_density_derivative": "omitted",
        "water_eos_spatial_gradient": "rejected_deferred",
        "densin_off_identity": "unbound_historical_controller",
        "actual_cuda_face_cell_source": True,
        "automatic_cpu_fallback": False,
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "production_admission": False,
        "not_claimed": [
            "AMR", "wet_dry", "spatial_water_EOS", "v96_atomic_owner",
            "plugin_support", "long_trajectory_qualification",
            "Delft_staggered_grid_identity",
        ],
    }


__all__ = [
    "BACKEND", "CUDA_COMPOSITION_ID", "DENSIN_SEDIMENT_INCREMENT",
    "DensInResidualGPUExecutionV95", "DensInResidualGPUStateV95",
    "DensInResidualGPUTrialV95", "DensInResidualGPUV95Error",
    "MAX_ACCEPTED_UPDATES", "ResidentDensInResidualV95",
    "execution_document_v95", "method_identity_v95_cuda",
    "resolve_densin_residual_gpu_execution_v95",
    "resolve_densin_residual_gpu_state_v95",
    "validate_densin_residual_gpu_state_v95",
]
