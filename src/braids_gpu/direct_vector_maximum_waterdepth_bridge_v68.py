"""Authenticated resident-CUDA v68 direct-vector hydraulic bridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.direct_vector_family_live_v67 import (
    METHOD_SOULSBY_CURRENT,
    METHOD_SOULSBY_VAN_RIJN_CURRENT,
    METHOD_VAN_RIJN_1993_CURRENT,
)
from braids_core.sediment.direct_vector_maximum_waterdepth_bridge_v68 import (
    FLOW_EPS,
    METHOD_EFFECTS,
    DirectVectorMaximumWaterdepthV68Error,
    load_direct_vector_maximum_waterdepth_execution_v68,
    verify_direct_vector_maximum_waterdepth_sources_v68,
)
from braids_core.sediment.maximum_waterdepth_live_v67 import (
    verify_support_descriptor_v67,
)


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAME = "direct_vector_maximum_waterdepth_v68"
BACKEND = "implicit_fwave_midpoint_gpu"
INTEGRATOR = "implicit_fwave_midpoint"


@dataclass(frozen=True)
class ResidentDirectVectorHydraulicsV68:
    physical_depth: object
    observation_depth: object
    characteristic_u: object
    characteristic_v: object
    characteristic_speed: object
    observation_height: object
    effective_u: object
    effective_v: object
    effective_speed: object
    active: object
    status: object
    maximum_waterdepth: bool
    support_kind: str
    execution_fingerprint: str
    preprocessor_application_count: int = 1

    def producer_velocity(self, method_id: str):
        if method_id not in METHOD_EFFECTS:
            raise DirectVectorMaximumWaterdepthV68Error(
                f"Q3-H6c/v68 unsupported direct-vector method: {method_id!r}"
            )
        import cupy as cp
        if method_id == METHOD_VAN_RIJN_1993_CURRENT:
            return cp.ascontiguousarray(cp.stack(
                (self.characteristic_u, self.characteristic_v), axis=0
            ))
        return cp.ascontiguousarray(cp.stack(
            (self.effective_u, self.effective_v), axis=0
        ))


class DirectVectorMaximumWaterdepthBridgeV68CUDA:
    """One-device bridge; no host numerical or automatic backend fallback."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(
        self, *, execution: Mapping[str, object], topology_fingerprint: str,
        cell_support_offset=None, cell_support_ids=None,
        trace_support_offset=None, trace_support_ids=None,
        verify_sources: bool = True, source_path: str | Path | None = None,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        resolved = load_direct_vector_maximum_waterdepth_execution_v68(execution)
        if (resolved["backend"] != BACKEND or
                resolved["integrator"] != INTEGRATOR or
                resolved["automatic_backend_fallback"] is not False):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 CUDA backend/integrator differs"
            )
        if topology_fingerprint != resolved["topology_fingerprint"]:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 runtime mesh topology differs"
            )
        enabled = bool(resolved["maximum_waterdepth"])
        arrays = (cell_support_offset, cell_support_ids,
                  trace_support_offset, trace_support_ids)
        if enabled and any(value is None for value in arrays):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 enabled CUDA bridge requires complete support arrays"
            )
        if not enabled and any(value is not None for value in arrays):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 MaximumWaterdepth-off rejects support arrays"
            )
        if enabled:
            cell_offset = self._array(
                "cell support offset", cell_support_offset, cp.int64
            )
            cell_ids = self._array("cell support IDs", cell_support_ids, cp.int64)
            trace_offset = self._array(
                "trace support offset", trace_support_offset, cp.int64
            )
            trace_ids = self._array(
                "trace support IDs", trace_support_ids, cp.int64
            )
            # The immutable, compact topology descriptor is the sole setup
            # crossing. Dynamic depths and every producer input remain resident.
            verify_support_descriptor_v67(
                resolved["cell_support"],
                support_offset=np.ascontiguousarray(cp.asnumpy(cell_offset)),
                support_ids=np.ascontiguousarray(cp.asnumpy(cell_ids)),
            )
            verify_support_descriptor_v67(
                resolved["trace_support"],
                support_offset=np.ascontiguousarray(cp.asnumpy(trace_offset)),
                support_ids=np.ascontiguousarray(cp.asnumpy(trace_ids)),
            )
            self._offset = {"cell": cell_offset, "trace": trace_offset}
            self._ids = {"cell": cell_ids, "trace": trace_ids}
            self._link_count = {
                "cell": int(cell_ids.size), "trace": int(trace_ids.size),
            }
        else:
            self._offset = {"cell": None, "trace": None}
            self._ids = {"cell": None, "trace": None}
            self._link_count = {"cell": 0, "trace": 0}
        self.execution = resolved
        self.enabled = enabled
        self.source_receipt = (
            verify_direct_vector_maximum_waterdepth_sources_v68(resolved)
            if verify_sources else None
        )
        source = Path(source_path or Path(__file__).with_suffix(".cu")).read_text(
            encoding="utf-8"
        )
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source, options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = module.get_function(KERNEL_NAME)

    def _array(self, name: str, value, dtype, size: int | None = None):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != 1 or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id or
                (size is not None and int(value.size) != size)):
            label = "FP64" if dtype == cp.float64 else str(dtype)
            suffix = "" if size is None else f" containing {size} values"
            raise DirectVectorMaximumWaterdepthV68Error(
                f"Q3-H6c/v68 {name} must be resident contiguous {label}{suffix}"
            )
        return value

    def prepare(
        self, kind: str, *, cell_depth, transport_u, transport_v,
        effective_speed, active, support_depth=None, support_active=None,
        direction_epsilon: float = FLOW_EPS,
        effective_direction_epsilon: float = FLOW_EPS,
        reject_invalid: bool = True,
    ) -> ResidentDirectVectorHydraulicsV68:
        cp = self.cp
        if kind not in {"cell", "trace"}:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 support kind must be cell or trace"
            )
        depth = self._array("cell depth", cell_depth, cp.float64)
        count = int(depth.size)
        numerator_u = self._array(
            "transport-u numerator", transport_u, cp.float64, count
        )
        numerator_v = self._array(
            "transport-v numerator", transport_v, cp.float64, count
        )
        target_speed = self._array(
            "effective speed", effective_speed, cp.float64, count
        )
        flags = self._array("active flags", active, cp.int32, count)
        try:
            direction_eps = float(direction_epsilon)
            effective_eps = float(effective_direction_epsilon)
        except (TypeError, ValueError) as error:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 direction epsilons must be numeric"
            ) from error
        if (not np.isfinite(direction_eps) or direction_eps <= 0.0 or
                not np.isfinite(effective_eps) or effective_eps <= 0.0):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 direction epsilons must be finite and positive"
            )
        if self.enabled:
            links = self._array(
                f"{kind} support depth", support_depth, cp.float64,
                self._link_count[kind],
            )
            link_flags = self._array(
                f"{kind} support activity", support_active, cp.int32,
                self._link_count[kind],
            )
            offset = self._offset[kind]
            assert offset is not None
            if int(offset.size) != count+1:
                raise DirectVectorMaximumWaterdepthV68Error(
                    f"Q3-H6c/v68 {kind} point count differs from support receipt"
                )
        else:
            if support_depth is not None or support_active is not None:
                raise DirectVectorMaximumWaterdepthV68Error(
                    "Q3-H6c/v68 MaximumWaterdepth-off rejects support data"
                )
            offset = cp.zeros(count+1, dtype=cp.int64)
            links = cp.empty(0, dtype=cp.float64)
            link_flags = cp.empty(0, dtype=cp.int32)

        outputs = [cp.empty(count, dtype=cp.float64) for _ in range(8)]
        status = cp.empty(count, dtype=cp.int32)
        with cp.cuda.Device(self.device_id):
            if count:
                self.kernel(
                    ((count+255)//256,), (256,), (
                        np.int64(count), np.int64(int(links.size)),
                        np.int32(int(self.enabled)), depth, numerator_u,
                        numerator_v, target_speed, flags, offset, links,
                        link_flags, np.float64(direction_eps),
                        np.float64(effective_eps), *outputs, status,
                    )
                )
            if reject_invalid and int(cp.count_nonzero(status).item()):
                raise DirectVectorMaximumWaterdepthV68Error(
                    "Q3-H6c/v68 CUDA hydraulic preprocessing rejected state(s)"
                )
        return ResidentDirectVectorHydraulicsV68(
            depth, *outputs, flags, status, self.enabled, kind,
            self.execution["fingerprint"], 1,
        )

    def producer_fields(
        self, method_id: str, hydraulics: ResidentDirectVectorHydraulicsV68,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        if method_id not in METHOD_EFFECTS:
            raise DirectVectorMaximumWaterdepthV68Error(
                f"Q3-H6c/v68 unsupported direct-vector method: {method_id!r}"
            )
        if (not isinstance(hydraulics, ResidentDirectVectorHydraulicsV68) or
                hydraulics.execution_fingerprint != self.execution["fingerprint"] or
                hydraulics.preprocessor_application_count != 1):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 resident preprocessing receipt differs"
            )
        if not isinstance(fields, Mapping) or any(type(key) is not str for key in fields):
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 producer fields must be a string-keyed mapping"
            )
        result = dict(fields)
        result["depth"] = hydraulics.physical_depth
        if method_id == METHOD_VAN_RIJN_1993_CURRENT:
            result["velocity"] = hydraulics.producer_velocity(method_id)
            result["observation_height"] = hydraulics.observation_height
            result.pop("u", None)
            result.pop("v", None)
        else:
            result["u"] = hydraulics.effective_u
            result["v"] = hydraulics.effective_v
            result.pop("velocity", None)
            result.pop("observation_height", None)
        return result

    def scalar_q(self, method_id: str, hydraulics: ResidentDirectVectorHydraulicsV68):
        """Return the physical-depth state consumed by laws 4/13 once."""
        cp = self.cp
        fields = self.producer_fields(method_id, hydraulics, {})
        if method_id not in {
                METHOD_SOULSBY_CURRENT, METHOD_SOULSBY_VAN_RIJN_CURRENT}:
            raise DirectVectorMaximumWaterdepthV68Error(
                "Q3-H6c/v68 scalar q is restricted to laws 4 and 13"
            )
        return cp.ascontiguousarray(cp.stack((
            fields["depth"], fields["depth"]*fields["u"],
            fields["depth"]*fields["v"],
        ), axis=0))


__all__ = [
    "BACKEND", "CUDA_OPTIONS", "INTEGRATOR", "KERNEL_NAME",
    "DirectVectorMaximumWaterdepthBridgeV68CUDA",
    "ResidentDirectVectorHydraulicsV68",
]
