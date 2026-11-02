"""Backend-bound geometry certificate for the actual-CUDA v98 seam.

The shared certificate proves the global Cartesian 2:1 join and derives the
coarse-centre to fine-centre normal distance.  This wrapper binds that proof
to the two coupled execution identities consumed by the CUDA trace owners.
It owns no numerical operation, state, or clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from braids_core.sediment.coupled_amr_geometry_v98 import (
    CartesianAMRPatchGeometryV98,
    CoupledAMRJoinedGeometryV98,
    validate_joined_amr_geometry_v98,
)
from braids_core.sediment.suspended_lifecycle_live_v76 import (
    SuspendedLifecycleExecutionV76,
)


SCHEMA = 1
SCOPE_ID = "braids.q4.coupled_amr.geometry_binding.actual_cuda.v98"
BACKEND = "actual_cuda"


class CoupledAMRGeometryGPUV98Error(ValueError):
    """The CUDA hierarchy geometry proof is absent, stale, or cross-bound."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRGeometryGPUV98Error(
            "v98 CUDA geometry identity is not canonical JSON") from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CoupledAMRGeometryBindingGPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    coarse_grid: SuspendedLifecycleExecutionV76
    fine_grid: SuspendedLifecycleExecutionV76
    coarse_patch: CartesianAMRPatchGeometryV98
    fine_patch: CartesianAMRPatchGeometryV98
    joined_geometry: CoupledAMRJoinedGeometryV98
    backend: str
    receipt_sha256: str


def _payload(value: CoupledAMRGeometryBindingGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_grid": value.coarse_grid.fingerprint,
        "fine_grid": value.fine_grid.fingerprint,
        "coarse_patch": value.coarse_patch.receipt_sha256,
        "fine_patch": value.fine_patch.receipt_sha256,
        "joined_geometry": value.joined_geometry.receipt_sha256,
        "donor_reconstruction": "cell_centre_piecewise_constant",
        "distance_consumer": [
            "augmented_fwave_wetdry_geometry",
            "density_trace_projection",
            "momentum_trace",
            "combined_scalar_diffusion",
        ],
        "backend": value.backend,
        "physics_evaluation_count": 0,
        "state_or_clock_publication_count": 0,
    }


def resolve_coupled_amr_geometry_binding_gpu_v98(
    *, coarse_execution_fingerprint: str,
    fine_execution_fingerprint: str,
    coarse_grid: SuspendedLifecycleExecutionV76,
    fine_grid: SuspendedLifecycleExecutionV76,
    coarse_patch: CartesianAMRPatchGeometryV98,
    fine_patch: CartesianAMRPatchGeometryV98,
    joined_geometry: CoupledAMRJoinedGeometryV98,
) -> CoupledAMRGeometryBindingGPUV98:
    validate_joined_amr_geometry_v98(
        joined_geometry, coarse_grid, fine_grid, coarse_patch, fine_patch)
    for name, value in (
            ("coarse execution", coarse_execution_fingerprint),
            ("fine execution", fine_execution_fingerprint)):
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef"
                    for character in value)):
            raise CoupledAMRGeometryGPUV98Error(
                f"v98 CUDA {name} fingerprint differs")
    placeholder = CoupledAMRGeometryBindingGPUV98(
        coarse_execution_fingerprint, fine_execution_fingerprint,
        coarse_grid, fine_grid, coarse_patch, fine_patch,
        joined_geometry, BACKEND, "")
    return CoupledAMRGeometryBindingGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})


def validate_coupled_amr_geometry_binding_gpu_v98(
    value: CoupledAMRGeometryBindingGPUV98,
) -> None:
    if (type(value) is not CoupledAMRGeometryBindingGPUV98 or
            value.backend != BACKEND or
            value.receipt_sha256 != _hash(_payload(value)) or
            value.coarse_grid.fingerprint !=
                value.coarse_patch.execution_fingerprint or
            value.fine_grid.fingerprint !=
                value.fine_patch.execution_fingerprint):
        raise CoupledAMRGeometryGPUV98Error(
            "v98 CUDA geometry binding identity differs")
    try:
        validate_joined_amr_geometry_v98(
            value.joined_geometry, value.coarse_grid, value.fine_grid,
            value.coarse_patch, value.fine_patch)
    except ValueError as error:
        raise CoupledAMRGeometryGPUV98Error(
            "v98 CUDA joined geometry rebuild differs") from error


__all__ = [
    "BACKEND", "SCOPE_ID", "CoupledAMRGeometryBindingGPUV98",
    "CoupledAMRGeometryGPUV98Error",
    "resolve_coupled_amr_geometry_binding_gpu_v98",
    "validate_coupled_amr_geometry_binding_gpu_v98",
]
