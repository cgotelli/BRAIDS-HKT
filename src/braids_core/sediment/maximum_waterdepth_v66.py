"""FLOW ``MaximumWaterdepth`` characteristic-velocity operator.

Classic FLOW applies this switch in ``dwnvel`` after assembling the two
depth-integrated directional velocity numerators and before division by the
zeta-point depth.  When enabled, the divisor is the maximum of the local water
depth and every active surrounding velocity-point depth.  The parsed
``MaximumWaterdepthFraction`` is not consumed by classic FLOW and is therefore
deliberately absent from this operator.

The NumPy implementation is the independent host oracle.  Production CPU and
CUDA implementations live in the matching Fortran and CUDA v66 modules.  The
CSR support table lets the same contract cover Cartesian and cut-cell/AMR
topologies without inventing a four-face restriction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids-q3-maximum-waterdepth-v66-v1"
# ``dwnvel.f90`` writes the unsuffixed default-real literal ``0.368``.  The
# locked build therefore promotes binary32 0.368 to fp, rather than evaluating
# the decimal directly in binary64.
FLOW_DWNVEL_OBSERVATION_FACTOR = float(np.float32(0.368))
LOCKED_SOURCES = MappingProxyType({
    "dwnvel": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/dwnvel.f90"
        ),
        "sha256": "cb26d6c037804ce2be3bcc975dab81cce21b78571f75e33fa302bcc14bf17ccd",
        "lines": "223-264",
    },
    "morphology_defaults": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_data/src/morphology_data_module.f90"
        ),
        "sha256": "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac",
        "lines": "333-336,1666-1667",
    },
    "rdmor": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_io/src/rdmor.f90"
        ),
        "sha256": "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
        "lines": "826-830",
    },
})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def semantic_document() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "locked_sources": dict(LOCKED_SOURCES),
        "input": {
            "cell_depth": "positive physical zeta-point water depth",
            "transport_numerator": (
                "directional depth-integrated characteristic-velocity numerator "
                "assembled before dwnvel division"
            ),
            "support": "CSR active velocity-point depths surrounding each cell",
        },
        "enabled": (
            "observation_depth=max(cell_depth,active_support_depths); "
            "u=transport_u/observation_depth; "
            "v=transport_v/observation_depth"
        ),
        "disabled": "observation_depth=cell_depth",
        "observation_height_2dh": {
            "factor": FLOW_DWNVEL_OBSERVATION_FACTOR,
            "source_semantics": "default_real_binary32_promoted_to_fp",
        },
        "maximum_waterdepth_fraction": (
            "parsed_but_unused_in_classic_FLOW_not_an_operator_input"
        ),
        "topology": "CSR_generalization_preserves_source_maximum_operation",
        "backend": ["native_openmp_cpu", "resident_cuda"],
        "product_admitted": False,
    }


SEMANTIC_ID = hashlib.sha256(_canonical(semantic_document())).hexdigest()


@dataclass(frozen=True)
class MaximumWaterdepthResultV66:
    observation_depth: np.ndarray
    characteristic_u: np.ndarray
    characteristic_v: np.ndarray
    characteristic_speed: np.ndarray
    observation_height: np.ndarray
    status: np.ndarray


def _fp64_vector(name: str, value: object, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.float64 or result.ndim != 1 or not result.flags.c_contiguous:
        raise TypeError(f"{name} must be a contiguous one-dimensional FP64 array")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _i64_vector(name: str, value: object, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.int64 or result.ndim != 1 or not result.flags.c_contiguous:
        raise TypeError(f"{name} must be a contiguous one-dimensional int64 array")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _i32_vector(name: str, value: object, size: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.int32 or result.ndim != 1 or not result.flags.c_contiguous:
        raise TypeError(f"{name} must be a contiguous one-dimensional int32 array")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def maximum_waterdepth_characteristic_velocity_v66(
    *,
    cell_depth: object,
    transport_u: object,
    transport_v: object,
    active: object,
    support_offset: object,
    support_depth: object,
    support_active: object,
    enabled: bool,
    direction_epsilon: float = 1.0e-12,
) -> MaximumWaterdepthResultV66:
    """Apply the source-ordered 2DH divisor limiter without mutating inputs."""
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be Boolean")
    epsilon = float(direction_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("direction_epsilon must be finite and positive")

    depth = _fp64_vector("cell_depth", cell_depth)
    ncell = int(depth.size)
    numerator_u = _fp64_vector("transport_u", transport_u, ncell)
    numerator_v = _fp64_vector("transport_v", transport_v, ncell)
    cell_active = _i32_vector("active", active, ncell)
    offset = _i64_vector("support_offset", support_offset, ncell + 1)
    links = _fp64_vector("support_depth", support_depth)
    link_active = _i32_vector("support_active", support_active, links.size)

    if (offset[0] != 0 or offset[-1] != links.size or
            np.any(offset[1:] < offset[:-1])):
        raise ValueError("support_offset must be a canonical CSR offset table")
    if np.any((cell_active != 0) & (cell_active != 1)):
        raise ValueError("active must contain only zero or one")
    if np.any((link_active != 0) & (link_active != 1)):
        raise ValueError("support_active must contain only zero or one")
    if (not np.isfinite(depth).all() or not np.isfinite(numerator_u).all() or
            not np.isfinite(numerator_v).all() or not np.isfinite(links).all()):
        raise ValueError("MaximumWaterdepth inputs must be finite")
    if np.any(depth < 0.0) or np.any(links < 0.0):
        raise ValueError("MaximumWaterdepth depths must be nonnegative")
    if np.any((cell_active == 1) & (depth <= 0.0)):
        raise ValueError("active cells require positive water depth")

    observation = np.zeros(ncell, dtype=np.float64)
    u = np.zeros(ncell, dtype=np.float64)
    v = np.zeros(ncell, dtype=np.float64)
    status = np.zeros(ncell, dtype=np.int32)
    for cell in range(ncell):
        if cell_active[cell] == 0:
            continue
        work_depth = float(depth[cell])
        if enabled:
            start, stop = int(offset[cell]), int(offset[cell + 1])
            selected = links[start:stop][link_active[start:stop] == 1]
            if selected.size:
                work_depth = max(work_depth, float(np.max(selected)))
        uu = float(numerator_u[cell]) / work_depth
        vv = float(numerator_v[cell]) / work_depth
        if abs(uu) < epsilon:
            uu = 0.0
        if abs(vv) < epsilon:
            vv = 0.0
        observation[cell] = work_depth
        u[cell] = uu
        v[cell] = vv

    return MaximumWaterdepthResultV66(
        observation,
        u,
        v,
        np.hypot(u, v),
        np.float64(FLOW_DWNVEL_OBSERVATION_FACTOR) * observation,
        status,
    )


__all__ = [
    "LOCKED_SOURCES",
    "FLOW_DWNVEL_OBSERVATION_FACTOR",
    "MaximumWaterdepthResultV66",
    "SCHEMA",
    "SCOPE_ID",
    "SEMANTIC_ID",
    "maximum_waterdepth_characteristic_velocity_v66",
    "semantic_document",
]
