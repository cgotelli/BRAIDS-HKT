"""Resident actual-CUDA adapter for the bounded Q4/v77 GLM control.

The CUDA backend is explicit and has no CPU numerical fallback.  One raw
kernel performs a bit-preserving ``uint64`` copy of every concentration value;
the common host publisher then rejects any byte difference.  A second raw
kernel is exposed only for the locked-source equation probe that demonstrates
the exact wave/roller forcing predicate.  Runtime wave execution remains
rejected by the core v77 resolver.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from braids_core.sediment.euleriso_glm_live_v77 import (
    BACKEND_CUDA,
    CUDA_THREADS_PER_BLOCK,
    ExactWaveRollerForcingV77,
    EulerisoGLMStateV77,
    EulerisoGLMTrialV77,
    EulerisoGLMV77Error,
    _publish_trial_v77,
    cancel_trial_v77,
    checkpoint_document_v77,
    commit_trial_v77,
    evaluate_rolcor_constituent_flux_probe_v77,
    load_checkpoint_v77,
    load_execution_v77,
    rollback_trial_v77,
    validate_exact_wave_roller_forcing_v77,
    validate_state_v77,
)


CUDA_IDENTITY_KERNEL = "braids_euleriso_glm_identity_v77"
CUDA_SOURCE_GATE_KERNEL = "braids_euleriso_glm_source_gate_v77"

_CUDA_SOURCE = r'''
extern "C" __global__
void braids_euleriso_glm_identity_v77(
    const unsigned long long* input,
    unsigned long long* output,
    const unsigned long long count)
{
    const unsigned long long i =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count) output[i] = input[i];
}

extern "C" __global__
void braids_euleriso_glm_source_gate_v77(
    const double* grmasu, const double* grmasv,
    const double* grfacu, const double* grfacv,
    const double* guu, const double* gvv,
    const double* tpu, const double* tpv,
    const int* kfu, const int* kcu, const int* kfv, const int* kcv,
    double* dqx, double* dqy, const unsigned long long count,
    const int requested, const int wave_enabled, const int roller_enabled,
    const int cyclic_solver, const int constituent_count,
    const double layer_thickness)
{
    const unsigned long long i =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    double out_qx = 0.0;
    double out_qy = 0.0;
    if (requested && wave_enabled && roller_enabled && cyclic_solver &&
        constituent_count > 0) {
        if (kfu[i] == 1 && kcu[i] <= 2 && tpu[i] > 0.5)
            out_qx = -(grmasu[i] + grfacu[i]) * guu[i] * layer_thickness;
        if (kfv[i] == 1 && kcv[i] <= 2 && tpv[i] > 0.5)
            out_qy = -(grmasv[i] + grfacv[i]) * gvv[i] * layer_thickness;
    }
    dqx[i] = out_qx;
    dqy[i] = out_qy;
}
'''


def _error(message: str, category: str) -> None:
    raise EulerisoGLMV77Error(message, category=category)


def _cupy():
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - environment gate
        raise EulerisoGLMV77Error(
            "v77 CUDA backend requires CuPy",
            category="cuda_unavailable") from error
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _error("v77 CUDA backend requires an actual device",
                   "cuda_unavailable")
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise EulerisoGLMV77Error(
            "v77 CUDA device/runtime is unavailable",
            category="cuda_unavailable") from error
    return cp


def _device_name(cp) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(
        cp.cuda.runtime.getDevice())
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name)


class ResidentEulerisoGLMV77:
    """Explicit resident CUDA identity owner for one v77 execution."""

    def __init__(self, execution: Mapping[str, object]):
        cp = _cupy()
        resolved = load_execution_v77(execution)
        if resolved["backend"] != BACKEND_CUDA:
            _error("resident v77 CUDA owner requires a CUDA receipt",
                   "backend_identity_mismatch")
        self.cp = cp
        self.execution = resolved
        self.identity_kernel = cp.RawKernel(
            _CUDA_SOURCE, CUDA_IDENTITY_KERNEL,
            options=("--std=c++11",))
        self.runtime_receipt = {
            "backend": BACKEND_CUDA,
            "actual_cuda_kernel": True,
            "automatic_cpu_fallback": False,
            "identity_kernel": CUDA_IDENTITY_KERNEL,
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "device_id": int(cp.cuda.runtime.getDevice()),
            "device_name": _device_name(cp),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        }

    def begin_trial(
        self, parent: EulerisoGLMStateV77, concentration_kg_m3=None, *,
        cancelled: bool = False,
    ) -> EulerisoGLMTrialV77:
        cp = self.cp
        validate_state_v77(parent, execution=self.execution)
        if type(cancelled) is not bool:
            _error("v77 cancelled must be an exact Boolean",
                   "invalid_control")
        if cancelled:
            _error("v77 update cancelled before CUDA candidate publication",
                   "cancelled")
        if concentration_kg_m3 is None:
            source = cp.ascontiguousarray(
                cp.asarray(parent.concentration_kg_m3, dtype=cp.float64))
        else:
            source = concentration_kg_m3
        if (not isinstance(source, cp.ndarray) or
                source.dtype != cp.float64 or
                source.shape != parent.concentration_kg_m3.shape or
                not source.flags.c_contiguous):
            _error(
                "v77 CUDA concentration must be contiguous device float64 "
                f"{parent.concentration_kg_m3.shape}",
                "invalid_device_input")
        if bool(cp.any(~cp.isfinite(source)).item()) or \
                bool(cp.any(source < 0.0).item()):
            _error("v77 CUDA concentration must be finite and nonnegative",
                   "invalid_device_input")
        parent_device = cp.ascontiguousarray(cp.asarray(
            parent.concentration_kg_m3, dtype=cp.float64))
        if not bool(cp.all(
                source.view(cp.uint64) == parent_device.view(cp.uint64)).item()):
            _error("v77 CUDA identity input differs from its accepted parent",
                   "parent_concentration_mismatch")
        output = cp.empty_like(source)
        count = int(source.size)
        blocks = (count + CUDA_THREADS_PER_BLOCK - 1) // \
            CUDA_THREADS_PER_BLOCK
        self.identity_kernel(
            (blocks,), (CUDA_THREADS_PER_BLOCK,),
            (source.view(cp.uint64), output.view(cp.uint64), np.uint64(count)))
        cp.cuda.Stream.null.synchronize()
        return _publish_trial_v77(
            execution=self.execution, parent=parent,
            input_concentration_kg_m3=cp.asnumpy(source),
            output_concentration_kg_m3=cp.asnumpy(output),
            backend=BACKEND_CUDA,
            observed_parallel_workers=CUDA_THREADS_PER_BLOCK,
            kernel_launches=1)

    def commit(self, parent: EulerisoGLMStateV77,
               trial: EulerisoGLMTrialV77) -> EulerisoGLMStateV77:
        return commit_trial_v77(
            parent, trial, execution=self.execution)

    def rollback(self, parent: EulerisoGLMStateV77,
                 trial: EulerisoGLMTrialV77) -> EulerisoGLMStateV77:
        return rollback_trial_v77(
            parent, trial, execution=self.execution)

    def cancel(self, parent: EulerisoGLMStateV77,
               trial: EulerisoGLMTrialV77) -> EulerisoGLMStateV77:
        return cancel_trial_v77(
            parent, trial, execution=self.execution)

    def checkpoint(self, value: EulerisoGLMStateV77) -> dict[str, object]:
        return checkpoint_document_v77(value, execution=self.execution)

    def restore_checkpoint(
        self, value: Mapping[str, object],
    ) -> EulerisoGLMStateV77:
        return load_checkpoint_v77(value, execution=self.execution)


def evaluate_rolcor_constituent_flux_probe_cuda_v77(
    *, euleriso_glm: bool, wave_enabled: bool, roller_enabled: bool,
    transport_solver: str, constituent_count: int,
    forcing: ExactWaveRollerForcingV77 | None,
) -> dict[str, object]:
    """Run the 2DH ``tritra``/``rolcor`` flux predicate on actual CUDA.

    This is not a wave runtime route.  The common CPU result authenticates and
    interprets the forcing, while a raw CUDA kernel independently evaluates
    the locked 2DH equations and must match exactly within floating arithmetic.
    """
    cp = _cupy()
    expected = evaluate_rolcor_constituent_flux_probe_v77(
        euleriso_glm=euleriso_glm, wave_enabled=wave_enabled,
        roller_enabled=roller_enabled, transport_solver=transport_solver,
        constituent_count=constituent_count, forcing=forcing)
    if forcing is None:
        zeros = np.zeros(1, dtype=np.float64)
        integer_zeros = np.zeros(1, dtype=np.int32)
        forcing_values = (
            zeros, zeros, zeros, zeros, np.ones(1), np.ones(1),
            zeros, zeros, integer_zeros, integer_zeros,
            integer_zeros, integer_zeros)
        thickness = 1.0
    else:
        forcing = validate_exact_wave_roller_forcing_v77(forcing)
        forcing_values = (
            forcing.grmasu_m2_s, forcing.grmasv_m2_s,
            forcing.grfacu_m2_s, forcing.grfacv_m2_s,
            forcing.guu_inverse_m, forcing.gvv_inverse_m,
            forcing.u_wave_period_s, forcing.v_wave_period_s,
            forcing.kfu, forcing.kcu, forcing.kfv, forcing.kcv,
        )
        thickness = forcing.layer_thickness
    device = [cp.ascontiguousarray(cp.asarray(item))
              for item in forcing_values]
    count = int(device[0].size)
    dqx = cp.zeros(count, dtype=cp.float64)
    dqy = cp.zeros(count, dtype=cp.float64)
    kernel = cp.RawKernel(
        _CUDA_SOURCE, CUDA_SOURCE_GATE_KERNEL, options=("--std=c++11",))
    blocks = (count + CUDA_THREADS_PER_BLOCK - 1) // CUDA_THREADS_PER_BLOCK
    kernel((blocks,), (CUDA_THREADS_PER_BLOCK,), (
        *device, dqx, dqy, np.uint64(count), np.int32(bool(euleriso_glm)),
        np.int32(bool(wave_enabled)), np.int32(bool(roller_enabled)),
        np.int32(transport_solver == "cyclic-method"),
        np.int32(constituent_count), np.float64(thickness)))
    cp.cuda.Stream.null.synchronize()
    host_dqx = cp.asnumpy(dqx)
    host_dqy = cp.asnumpy(dqy)
    if not (np.array_equal(host_dqx, expected.u_flux_delta) and
            np.array_equal(host_dqy, expected.v_flux_delta)):
        _error("v77 CUDA source gate differs from locked CPU equation probe",
               "cuda_parity_failure")
    return {
        "actual_cuda_kernel": True,
        "automatic_cpu_fallback": False,
        "kernel": CUDA_SOURCE_GATE_KERNEL,
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "device_name": _device_name(cp),
        "active": expected.active,
        "active_face_count": expected.active_face_count,
        "forcing_sha256": expected.forcing_sha256,
        "result_sha256": expected.result_sha256,
        "u_flux_delta": host_dqx,
        "v_flux_delta": host_dqy,
    }


evaluate_source_gate_cuda_v77 = evaluate_rolcor_constituent_flux_probe_cuda_v77


__all__ = [
    "CUDA_IDENTITY_KERNEL", "CUDA_SOURCE_GATE_KERNEL",
    "ResidentEulerisoGLMV77",
    "evaluate_rolcor_constituent_flux_probe_cuda_v77",
    "evaluate_source_gate_cuda_v77",
]
