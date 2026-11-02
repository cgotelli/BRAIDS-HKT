"""Typed Delft3D4 FLOW suspended-control composition (Q4/v86).

The locked FLOW source gives the three controls in this module very different
2DH meanings:

* ``IopSus`` changes the suspended grain diameter only in the private
  ``TraFrm=-1/-2/-4`` producers.  It is therefore equation-qualified here, but
  is rejected on the accepted v84 Bijker/19/20 lifecycle instead of being
  silently ignored.
* ``NeglectEntrainment`` does not change the transported scalar.  It changes
  the *morphological bed-balance owner* from erosion/deposition exchange to
  divergence of the accepted suspended transport fluxes.
* ``SusCor`` is parsed in 2DH, but its correction is guarded by ``kmax>1``.
  Its on/off identity is retained while its 2DH application count is exactly
  zero.

V86 wraps exactly one accepted v84 -> v81 -> v78 transaction.  The v78
water-column/active-bed conservation ledger is left unchanged.  A separate
morphological-bed ledger records the Delft ``NeglectEntrainment`` choice for a
later Exner owner; this avoids inventing a second exchange or capacity owner.
The bounded owner accepts at most four updates and is not product admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .q3_producer_lifecycle_binding_v84 import (
    MAX_ACCEPTED_UPDATES,
    PROFILE_METHODS,
    PUBLIC_METHODS,
    Q3BoundRouteLifecycleTrialV84,
)
from .route_lifecycle_composition_v81 import (
    ROUTE_COMBINE,
    ROUTE_OFF,
    ROUTE_TOTAL,
    RouteLifecycleExecutionV81,
    RouteLifecycleStateV81,
    validate_route_lifecycle_state_v81,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-suspended-controls-v86-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
LOCKED_UPSTREAM_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"
INTEGRATOR = "implicit_fwave_midpoint"
IOPSUS_FIXED = "fixed_suspended_diameter_iopsus_0"
IOPSUS_FLOW_DEPENDENT = "flow_dependent_suspended_diameter_iopsus_1"
BED_BALANCE_EXCHANGE = "erosion_deposition_exchange"
BED_BALANCE_FLUX_DIVERGENCE = "suspended_flux_divergence"
SUSCOR_ON_2DH_NOOP = "enabled_parsed_2dh_structural_noop"
SUSCOR_OFF_2DH_NOOP = "disabled_2dh_structural_noop"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


LOCKED_SOURCES = {
    "morphology_defaults": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_data/src/morphology_data_module.f90",
        "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac",
    ),
    "morphology_reader": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdmor.f90",
        "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
    ),
    "sediment_reader": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdsed.f90",
        "b3de2a67842935bee704eecee329e0e0a71fdd0bb95704a15f9d2ef567514fd2",
    ),
    "transport_formula_reader": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdtrafrm.f90",
        "0a2542980bc9e9e03e46b2abb8b06eda2fea3e28a67c78c4e61b619572154ea7",
    ),
    "flow_sediment_transport_reader": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_io/src/input/rdsedmortra.f90",
        "d20803ae2b7646973b8b1aa8e8252ac2c00c6dc2a690b665b557ba236ab5854f",
    ),
    "tram1": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tram1.f90",
        "33f5f60439c8b98facc7d5d1431e8f4a06fc37fe7b14d531a8fa9d9be6f74246",
    ),
    "tram2": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tram2.f90",
        "527f2c01065e1da343100b10e2d7b791d32dfcbd801de70a4e9b71429fb3133a",
    ),
    "bedbc1993": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/bedbc1993.f90",
        "7fd7b619a111e3ee99ad6625097882a0bf3baa0a0f6f6d39f939671390fd73d5",
    ),
    "bedbc2004": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/bedbc2004.f90",
        "25f227bd47dc716c806df1e29d70106092e6dc0d4ed8012a4dde7fb241068654",
    ),
    "flow_bott3d": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/bott3d.f90",
        "5abfee8230f66ba42b3eb87ff89948f65a0bfccc915a015c25f8c1412f4e47b1",
    ),
}


LEGAL_ROUTE_MATRIX = {
    "TraFrm=-1": {
        "iopsus": "0_or_1_source_active; equation-qualified_v86; not_v84",
        "neglect_entrainment": "combine_water_route_only",
        "suscor_2dh": "parsed_noop_kmax_guard",
    },
    "TraFrm=-2": {
        "iopsus": "0_or_1_source_active; equation-qualified_only",
        "neglect_entrainment": "combine_water_route_only",
        "suscor_2dh": "parsed_noop_kmax_guard",
    },
    "TraFrm=-4": {
        "iopsus": "0_or_1_source_active; equation-qualified_only",
        "neglect_entrainment": "combine_water_route_only",
        "suscor_2dh": "parsed_noop_kmax_guard",
    },
    "v84_Bijker_19_20": {
        "iopsus": "must_be_0; source_does_not_consume_IopSus",
        "neglect_entrainment": "supported_on_TRA_COMBINE_exactly_once",
        "suscor_2dh": "on_or_off_identity; zero_applications",
    },
}


class SuspendedControlsV86Error(ValueError):
    """Fail-closed v86 identity, route, transaction, or ledger error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise SuspendedControlsV86Error(
            "v86 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False,
           dtype=np.float64) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    if raw.shape != shape:
        raise SuspendedControlsV86Error(
            f"v86 {name} must have shape {shape}")
    if expected.kind in "iu":
        if raw.dtype.kind not in "iu":
            raise SuspendedControlsV86Error(f"v86 {name} must be integer")
    elif raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise SuspendedControlsV86Error(f"v86 {name} must be finite numeric")
    result = np.ascontiguousarray(raw, dtype=expected)
    if positive and np.any(result <= 0.0):
        raise SuspendedControlsV86Error(f"v86 {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise SuspendedControlsV86Error(f"v86 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SuspendedControlsV86Error(f"v86 {name} must be Boolean")
    return value


def verify_locked_sources_v86(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result: dict[str, str] = {}
    for key, (relative, expected) in LOCKED_SOURCES.items():
        path = root/relative
        if not path.is_file():
            raise SuspendedControlsV86Error(
                "v86 locked source is absent: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SuspendedControlsV86Error(
                "v86 locked source hash differs: " + relative)
        result[key] = actual
    return result


def iopsus_diameter_v86(
    *, formula_code: int, iopsus: int, fixed_dss_m: object,
    d50_m: object, transport_stage: object | None = None,
    d10_m: object | None = None, mobility_psi: object | None = None,
    silt_diameter_m: float = 32.0e-6,
) -> np.ndarray:
    """Exact locked ``bedbc1993/2004`` suspended-diameter branch.

    ``fixed_dss_m`` is the already initialized ``D50*FacDSS`` value used when
    ``IopSus=0``.  Only integer values 0 and 1 are admitted even though the
    Fortran comparison would treat other integers as the off branch.
    """
    if type(formula_code) is not int or formula_code not in (-4, -2, -1):
        raise SuspendedControlsV86Error(
            "v86 IopSus is defined only for TraFrm -1/-2/-4")
    if type(iopsus) is not int or iopsus not in (0, 1):
        raise SuspendedControlsV86Error("v86 IopSus must be integer 0 or 1")
    d50_raw = np.asarray(d50_m)
    if d50_raw.ndim < 1:
        raise SuspendedControlsV86Error("v86 D50 must be an array")
    shape = d50_raw.shape
    d50 = _array(d50_raw, shape, "D50", positive=True)
    fixed = _array(fixed_dss_m, shape, "fixed DSS", positive=True)
    if iopsus == 0:
        return fixed
    if formula_code == -1:
        if transport_stage is None:
            raise SuspendedControlsV86Error(
                "v86 Van Rijn 1993 IopSus requires transport stage")
        stage = _array(transport_stage, shape, "transport stage",
                       nonnegative=True)
        result = np.where(
            stage <= 1.0, 0.64*d50,
            np.where(stage >= 25.0, d50,
                     d50*(1.0+0.015*(stage-25.0))))
    else:
        if d10_m is None or mobility_psi is None:
            raise SuspendedControlsV86Error(
                "v86 TRANSPOR2004 IopSus requires D10 and mobility psi")
        d10 = _array(d10_m, shape, "D10", positive=True)
        psi = _array(mobility_psi, shape, "mobility psi", nonnegative=True)
        if np.any(d10 > d50):
            raise SuspendedControlsV86Error("v86 D10 cannot exceed D50")
        if (isinstance(silt_diameter_m, bool) or
                not isinstance(silt_diameter_m, (int, float)) or
                not math.isfinite(float(silt_diameter_m)) or
                float(silt_diameter_m) <= 0.0):
            raise SuspendedControlsV86Error(
                "v86 sand/silt diameter bounds must be positive")
        result = np.where(
            psi < 550.0,
            (1.0+0.0006*(d50/d10-1.0)*(psi-550.0))*d50,
            d50)
        result = np.maximum(d10, result)
        result = np.where(d50 < float(silt_diameter_m), d50, result)
        result = np.maximum(result, 0.5*float(silt_diameter_m))
    result = np.ascontiguousarray(result)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise SuspendedControlsV86Error(
            "v86 IopSus equation produced invalid DSS")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SuspendedControlsExecutionV86:
    binding_execution: Mapping[str, object]
    lifecycle_execution: RouteLifecycleExecutionV81
    backend: str
    iopsus: int
    neglect_entrainment: bool
    suscor: bool
    semantic_id: str
    fingerprint: str

    @property
    def fraction_count(self) -> int:
        return self.lifecycle_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.lifecycle_execution.cell_count

    @property
    def face_count(self) -> int:
        return self.lifecycle_execution.lifecycle_execution.face_count

    @property
    def boundary_count(self) -> int:
        return self.lifecycle_execution.lifecycle_execution.boundary_count


def _execution_payload(value: SuspendedControlsExecutionV86) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "locked_upstream_commit": LOCKED_UPSTREAM_COMMIT,
        "binding_execution_fingerprint": value.binding_execution["fingerprint"],
        "lifecycle_execution_fingerprint": value.lifecycle_execution.fingerprint,
        "backend": value.backend,
        "integrator": INTEGRATOR,
        "method_ids": list(PUBLIC_METHODS),
        "producer_method_ids": list(PROFILE_METHODS),
        "iopsus": value.iopsus,
        "iopsus_method_id": (IOPSUS_FIXED if value.iopsus == 0 else
                              IOPSUS_FLOW_DEPENDENT),
        "iopsus_lifecycle_disposition": (
            "valid_fixed_DSS" if value.iopsus == 0 else
            "illegal_for_v84_methods_fail_closed"),
        "neglect_entrainment": value.neglect_entrainment,
        "bed_balance_method_id": (
            BED_BALANCE_FLUX_DIVERGENCE if value.neglect_entrainment else
            BED_BALANCE_EXCHANGE),
        "suscor": value.suscor,
        "suscor_method_id": (SUSCOR_ON_2DH_NOOP if value.suscor else
                             SUSCOR_OFF_2DH_NOOP),
        "suscor_2dh_application_count": 0,
        "suscor_source_guard": "sus_nonzero_and_l_suscor_then_kmax_gt_1",
        "suspended_vector_construction_count": (
            1 if value.suscor else 0),
        "suscor_neglect_entrainment_compatibility": (
            "required_for_flux_divergence" if value.neglect_entrainment else
            "optional_for_exchange_bed_balance"),
        "scalar_lifecycle_changed_by_neglect_entrainment": False,
        "inner_v78_bed_state_source_equivalent_under_switch": False,
        "morphological_ledger_applied_to_production_exner": False,
        "accepted_inner_transaction_count": 1,
        "capacity_owner_count": 1,
        "exchange_owner_count": 1,
        "morphological_bed_balance_owner_count": 1,
        "automatic_cpu_fallback": False,
        "fixed_grid": True,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "legal_route_matrix": LEGAL_ROUTE_MATRIX,
        "product_admission": False,
    }


def resolve_suspended_controls_execution_v86(
    binding_execution: Mapping[str, object],
    lifecycle_execution: RouteLifecycleExecutionV81, *, backend: str,
    iopsus: int = 0, neglect_entrainment: bool = False,
    suscor: bool = True,
) -> SuspendedControlsExecutionV86:
    if backend not in ("cpu", "implicit_fwave_midpoint_gpu"):
        raise SuspendedControlsV86Error("v86 backend must be explicit")
    if not isinstance(binding_execution, Mapping):
        raise SuspendedControlsV86Error("v86 v84 binding execution is absent")
    required = {
        "fingerprint", "backend", "methods", "routes",
        "lifecycle_execution_fingerprint", "automatic_backend_fallback",
    }
    if not required.issubset(binding_execution):
        raise SuspendedControlsV86Error("v86 v84 binding inventory differs")
    if (not isinstance(binding_execution["fingerprint"], str) or
            _SHA256.fullmatch(str(binding_execution["fingerprint"])) is None or
            binding_execution["backend"] != backend or
            tuple(binding_execution["methods"]) != PUBLIC_METHODS or
            binding_execution["lifecycle_execution_fingerprint"] !=
            lifecycle_execution.fingerprint or
            binding_execution["automatic_backend_fallback"] is not False):
        raise SuspendedControlsV86Error(
            "v86 requires the exact no-fallback v84 binding")
    if type(iopsus) is not int or iopsus not in (0, 1):
        raise SuspendedControlsV86Error("v86 IopSus must be integer 0 or 1")
    # Locked bedbc1993/2004 consume IopSus only for -1/-2/-4.  Treating it as
    # a generic settling switch for Bijker/19/20 would be a false method.
    if iopsus == 1:
        raise SuspendedControlsV86Error(
            "v86 IopSus=1 is incompatible with the v84 Bijker/19/20 profile")
    neglect = _boolean(neglect_entrainment, "NeglectEntrainment")
    correction = _boolean(suscor, "SusCor")
    if neglect and not correction:
        raise SuspendedControlsV86Error(
            "v86 SusCor=false plus NeglectEntrainment=true is a source-legal "
            "zero-suspended-vector route not implemented by this bounded "
            "lifecycle")
    placeholder = SuspendedControlsExecutionV86(
        dict(binding_execution), lifecycle_execution, backend, iopsus,
        neglect, correction, SEMANTIC_ID, "")
    return SuspendedControlsExecutionV86(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v86(
    value: SuspendedControlsExecutionV86,
) -> dict[str, object]:
    if not isinstance(value, SuspendedControlsExecutionV86):
        raise SuspendedControlsV86Error("v86 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise SuspendedControlsV86Error("v86 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


def suspended_flux_divergence_v86(
    execution: SuspendedControlsExecutionV86, *,
    advective_face_flux_kg_m_s: object,
    diffusive_face_mass_owner_out_kg: object, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-area bed change and logical-boundary mass-in ledger.

    Positive values are deposition, matching FLOW ``dbodsd``.  The numerical
    face fluxes are BRAIDS v78 accepted FCT/diffusion fluxes; constitutive
    ownership follows FLOW while the spatial discretisation deliberately does
    not claim Delft bit identity.
    """
    base = execution.lifecycle_execution.lifecycle_execution.base_execution
    routes = np.asarray(
        execution.lifecycle_execution.route_execution["routes"],
        dtype=np.int32)
    return suspended_flux_divergence_topology_v86(
        base_execution=base, routes=routes,
        advective_face_flux_kg_m_s=advective_face_flux_kg_m_s,
        diffusive_face_mass_owner_out_kg=diffusive_face_mass_owner_out_kg,
        dt_s=dt_s)


TOPOLOGY_SELECTOR_ABI_ID = hashlib.sha256(
    b"braids-v86-suspended-flux-divergence-topology-abi-v1"
).hexdigest()


def suspended_flux_divergence_topology_v86(
    *, base_execution: object, routes: object,
    advective_face_flux_kg_m_s: object,
    diffusive_face_mass_owner_out_kg: object, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical v86 selector ABI for an authenticated v76 topology.

    This is the numerical body used by the accepted v86 owner.  Later
    composition layers may bind an exact accepted scalar trial to this ABI;
    they must not maintain a second transcription of the face divergence.
    """
    base = base_execution
    required = (
        "fraction_count", "cell_count", "face_count", "boundary_count",
        "face_length_m", "cell_face_offset", "cell_faces",
        "cell_face_sign", "cell_area_m2", "boundary_index",
    )
    if any(not hasattr(base, name) for name in required):
        raise SuspendedControlsV86Error(
            "v86 topology selector requires an accepted v76 base execution")
    ng, nc, nf, nb = (int(base.fraction_count), int(base.cell_count),
                      int(base.face_count), int(base.boundary_count))
    adv = _array(advective_face_flux_kg_m_s, (ng, nf), "advective face flux")
    diff = _array(diffusive_face_mass_owner_out_kg, (ng, nf),
                  "diffusive face mass")
    if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
            not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0):
        raise SuspendedControlsV86Error("v86 timestep must be positive")
    integral = adv*base.face_length_m[None, :]*float(dt_s)+diff
    delta = np.zeros((ng, nc), dtype=np.float64)
    for cell in range(nc):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell+1]):
            face = base.cell_faces[slot]
            delta[:, cell] -= base.cell_face_sign[slot]*integral[:, face]
        delta[:, cell] /= base.cell_area_m2[cell]
    boundary = np.zeros((ng, nb), dtype=np.float64)
    for face in range(nf):
        bid = base.boundary_index[face]
        if bid >= 0:
            boundary[:, bid] -= integral[:, face]
    route = _array(routes, (ng,), "route selector", dtype=np.int32)
    if np.any(~np.isin(route, (ROUTE_OFF, ROUTE_TOTAL, ROUTE_COMBINE))):
        raise SuspendedControlsV86Error(
            "v86 topology selector route is not OFF, TOTAL, or COMBINE")
    inactive = route != ROUTE_COMBINE
    delta[inactive] = 0.0
    boundary[inactive] = 0.0
    scale = np.maximum(1.0, np.maximum(
        np.abs(np.sum(delta*base.cell_area_m2[None, :], axis=1)),
        np.abs(np.sum(boundary, axis=1))))
    if np.any(np.abs(np.sum(delta*base.cell_area_m2[None, :], axis=1)-
                     np.sum(boundary, axis=1)) >
              32768.0*np.finfo(float).eps*scale):
        raise SuspendedControlsV86Error(
            "v86 suspended-flux divergence ledger did not close")
    return _array(delta, (ng, nc), "flux-divergence delta"), _array(
        boundary, (ng, nb), "flux-divergence boundary ledger")


@dataclass(frozen=True)
class SuspendedControlsStateV86:
    execution_fingerprint: str
    inner: RouteLifecycleStateV81
    morphological_bed_change_kg_m2: np.ndarray
    morphological_exchange_ledger_kg: np.ndarray
    morphological_boundary_ledger_kg: np.ndarray
    accepted_control_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.inner.accepted_updates

    @property
    def time_s(self) -> float:
        return self.inner.time_s


def _state_token(value: SuspendedControlsStateV86) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "inner_state_sha256": value.inner.state_sha256,
        "accepted_control_chain_sha256": value.accepted_control_chain_sha256,
        "arrays_sha256": _digest(
            value.morphological_bed_change_kg_m2,
            value.morphological_exchange_ledger_kg,
            value.morphological_boundary_ledger_kg),
    })


def _wrap_state(
    execution: SuspendedControlsExecutionV86,
    inner: RouteLifecycleStateV81, *, bed_change: object,
    exchange_ledger: object, boundary_ledger: object, chain: str,
) -> SuspendedControlsStateV86:
    execution_document_v86(execution)
    validate_route_lifecycle_state_v81(inner, execution.lifecycle_execution)
    ng, nc, nb = (execution.fraction_count, execution.cell_count,
                  execution.boundary_count)
    bed = _array(bed_change, (ng, nc), "morphological bed change")
    exchange = _array(exchange_ledger, (ng,), "morphological exchange ledger")
    boundary = _array(boundary_ledger, (ng, nb),
                      "morphological boundary ledger")
    if not isinstance(chain, str) or _SHA256.fullmatch(chain) is None:
        raise SuspendedControlsV86Error("v86 control chain digest differs")
    area = execution.lifecycle_execution.lifecycle_execution.base_execution.cell_area_m2
    current = np.sum(bed*area[None, :], axis=1)
    expected = (np.sum(boundary, axis=1) if execution.neglect_entrainment
                else exchange)
    scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(expected)))
    if np.any(np.abs(current-expected) >
              65536.0*np.finfo(float).eps*scale):
        raise SuspendedControlsV86Error(
            "v86 morphological bed-balance ledger did not close")
    if (execution.neglect_entrainment and np.any(exchange != 0.0)) or \
            (not execution.neglect_entrainment and np.any(boundary != 0.0)):
        raise SuspendedControlsV86Error(
            "v86 inactive morphological ledger owner is nonzero")
    placeholder = SuspendedControlsStateV86(
        execution.fingerprint, inner, bed, exchange, boundary, chain, "")
    return SuspendedControlsStateV86(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_suspended_controls_state_v86(
    execution: SuspendedControlsExecutionV86,
    inner: RouteLifecycleStateV81,
) -> SuspendedControlsStateV86:
    if inner.accepted_updates != 0:
        raise SuspendedControlsV86Error("v86 fresh state must start at update zero")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_inner_state_sha256": inner.state_sha256,
    })
    return _wrap_state(
        execution, inner,
        bed_change=np.zeros((execution.fraction_count, execution.cell_count)),
        exchange_ledger=np.zeros(execution.fraction_count),
        boundary_ledger=np.zeros((execution.fraction_count,
                                  execution.boundary_count)),
        chain=chain)


def validate_suspended_controls_state_v86(
    state: SuspendedControlsStateV86,
    execution: SuspendedControlsExecutionV86,
) -> None:
    if (not isinstance(state, SuspendedControlsStateV86) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise SuspendedControlsV86Error(
            "v86 state is stale, mutated, or execution-mismatched")
    _wrap_state(
        execution, state.inner,
        bed_change=state.morphological_bed_change_kg_m2,
        exchange_ledger=state.morphological_exchange_ledger_kg,
        boundary_ledger=state.morphological_boundary_ledger_kg,
        chain=state.accepted_control_chain_sha256)


@dataclass(frozen=True)
class SuspendedControlsTrialV86:
    execution_fingerprint: str
    parent_state_sha256: str
    inner_trial: Q3BoundRouteLifecycleTrialV84
    candidate: SuspendedControlsStateV86
    exchange_bed_delta_kg_m2: np.ndarray
    flux_divergence_bed_delta_kg_m2: np.ndarray
    selected_bed_delta_kg_m2: np.ndarray
    step_boundary_ledger_kg: np.ndarray
    exchange_owner_count: np.ndarray
    flux_divergence_owner_count: np.ndarray
    iopsus_application_count: np.ndarray
    suscor_application_count: np.ndarray
    duplicate_owner_count: np.ndarray
    backend: str
    trial_sha256: str


def _trial_token(value: SuspendedControlsTrialV86) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "inner_trial_sha256": value.inner_trial.trial_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "backend": value.backend,
        "arrays_sha256": _digest(
            value.exchange_bed_delta_kg_m2,
            value.flux_divergence_bed_delta_kg_m2,
            value.selected_bed_delta_kg_m2,
            value.step_boundary_ledger_kg,
            value.exchange_owner_count,
            value.flux_divergence_owner_count,
            value.iopsus_application_count,
            value.suscor_application_count,
            value.duplicate_owner_count),
    })


def build_suspended_controls_trial_v86(
    execution: SuspendedControlsExecutionV86,
    parent: SuspendedControlsStateV86,
    inner_trial: Q3BoundRouteLifecycleTrialV84, *,
    flux_delta: object, step_boundary_ledger: object, backend: str,
) -> SuspendedControlsTrialV86:
    validate_suspended_controls_state_v86(parent, execution)
    if (not isinstance(inner_trial, Q3BoundRouteLifecycleTrialV84) or
            inner_trial.parent_state_sha256 != parent.inner.state_sha256):
        raise SuspendedControlsV86Error("v86 inner v84 trial differs")
    if inner_trial.candidate.accepted_updates != parent.accepted_updates+1:
        raise SuspendedControlsV86Error("v86 inner update count differs")
    lifecycle_trial = inner_trial.lifecycle_trial.lifecycle_trial
    ng, nc, nb = (execution.fraction_count, execution.cell_count,
                  execution.boundary_count)
    exchange = _array(-lifecycle_trial.transferred_to_suspension_kg_m2,
                      (ng, nc), "exchange bed delta")
    flux = _array(flux_delta, (ng, nc), "flux-divergence bed delta")
    boundary = _array(step_boundary_ledger, (ng, nb), "step boundary ledger")
    routes = np.asarray(
        execution.lifecycle_execution.route_execution["routes"],
        dtype=np.int32)
    inactive = routes != ROUTE_COMBINE
    if (np.any(exchange[inactive] != 0.0) or np.any(flux[inactive] != 0.0) or
            np.any(boundary[inactive] != 0.0)):
        raise SuspendedControlsV86Error(
            "v86 non-combine route acquired a suspended bed balance")
    exchange_owner = np.ascontiguousarray(
        ((routes == ROUTE_COMBINE) &
         (not execution.neglect_entrainment)).astype(np.int32))
    flux_owner = np.ascontiguousarray(
        ((routes == ROUTE_COMBINE) &
         execution.neglect_entrainment).astype(np.int32))
    selected = flux if execution.neglect_entrainment else exchange
    area = execution.lifecycle_execution.lifecycle_execution.base_execution.cell_area_m2
    exchange_step = np.sum(exchange*area[None, :], axis=1)
    next_exchange = (parent.morphological_exchange_ledger_kg+
                     (np.zeros(ng) if execution.neglect_entrainment else
                      exchange_step))
    next_boundary = (parent.morphological_boundary_ledger_kg+
                     (boundary if execution.neglect_entrainment else
                      np.zeros_like(boundary)))
    chain = _fingerprint({
        "previous": parent.accepted_control_chain_sha256,
        "inner_trial_sha256": inner_trial.trial_sha256,
        "bed_balance_method_id": (
            BED_BALANCE_FLUX_DIVERGENCE if execution.neglect_entrainment else
            BED_BALANCE_EXCHANGE),
        "selected_delta_sha256": _digest(selected),
    })
    candidate = _wrap_state(
        execution, inner_trial.candidate,
        bed_change=parent.morphological_bed_change_kg_m2+selected,
        exchange_ledger=next_exchange, boundary_ledger=next_boundary,
        chain=chain)
    zeros = np.zeros(ng, dtype=np.int32)
    duplicate = exchange_owner*flux_owner
    arrays = tuple(_array(value, value.shape, name, dtype=value.dtype) for
                   value, name in (
                       (exchange, "exchange bed delta"),
                       (flux, "flux-divergence bed delta"),
                       (selected, "selected bed delta"),
                       (boundary, "step boundary ledger"),
                       (exchange_owner, "exchange owner count"),
                       (flux_owner, "flux owner count"),
                       (zeros, "IopSus application count"),
                       (zeros.copy(), "SusCor application count"),
                       (duplicate, "duplicate owner count")))
    if np.any(arrays[-1] != 0) or np.any(arrays[6] != 0) or \
            np.any(arrays[7] != 0):
        raise SuspendedControlsV86Error(
            "v86 owner/application counts differ")
    placeholder = SuspendedControlsTrialV86(
        execution.fingerprint, parent.state_sha256, inner_trial, candidate,
        *arrays, backend, "")
    return SuspendedControlsTrialV86(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


class SuspendedControlsLifecycleV86:
    """CPU outer owner around exactly one accepted native v84 transaction."""

    def __init__(self, execution: SuspendedControlsExecutionV86, inner_runtime):
        if execution.backend != "cpu":
            raise SuspendedControlsV86Error(
                "v86 CPU owner requires the explicit CPU execution")
        if (not hasattr(inner_runtime, "execution") or
                inner_runtime.execution["fingerprint"] !=
                execution.binding_execution["fingerprint"]):
            raise SuspendedControlsV86Error("v86 inner v84 runtime differs")
        self.execution = execution
        self.inner = inner_runtime
        self._open_parent: str | None = None
        self._open_trial: str | None = None

    def _operator(self, inner_trial, dt_s):
        lifecycle = inner_trial.lifecycle_trial.lifecycle_trial
        return suspended_flux_divergence_v86(
            self.execution,
            advective_face_flux_kg_m_s=lifecycle.advective_face_flux_kg_m_s,
            diffusive_face_mass_owner_out_kg=
                lifecycle.diffusive_face_mass_owner_out_kg,
            dt_s=dt_s)

    def begin_trial(self, parent: SuspendedControlsStateV86, *,
                    cancelled: bool = False, **inputs) -> SuspendedControlsTrialV86:
        validate_suspended_controls_state_v86(parent, self.execution)
        if self._open_trial is not None:
            raise SuspendedControlsV86Error("v86 previous trial is unfinished")
        if cancelled:
            raise SuspendedControlsV86Error(
                "v86 cancelled before the inner transaction")
        if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
            raise SuspendedControlsV86Error(
                "v86 accepted update limit was reached")
        inner_trial = None
        try:
            inner_trial = self.inner.begin_trial(
                parent.inner, cancelled=False, **inputs)
            flux, boundary = self._operator(inner_trial, inputs["dt_s"])
            trial = build_suspended_controls_trial_v86(
                self.execution, parent, inner_trial, flux_delta=flux,
                step_boundary_ledger=boundary, backend="isolated_cpu_oracle")
        except Exception as error:
            if inner_trial is not None:
                try:
                    self.inner.rollback(parent.inner, inner_trial)
                except Exception as rollback_error:
                    raise SuspendedControlsV86Error(
                        "v86 candidate failed and inner rollback failed") \
                        from rollback_error
            raise SuspendedControlsV86Error(
                "v86 CPU control transaction failed") from error
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _validate(self, parent, trial):
        validate_suspended_controls_state_v86(parent, self.execution)
        if (not isinstance(trial, SuspendedControlsTrialV86) or
                trial.execution_fingerprint != self.execution.fingerprint or
                trial.parent_state_sha256 != parent.state_sha256 or
                trial.trial_sha256 != _trial_token(trial) or
                self._open_parent != parent.state_sha256 or
                self._open_trial != trial.trial_sha256):
            raise SuspendedControlsV86Error(
                "v86 trial is stale, mutated, or not the open transaction")
        validate_suspended_controls_state_v86(trial.candidate, self.execution)

    def _finish(self, parent, trial, *, commit):
        self._validate(parent, trial)
        inner = (self.inner.commit(parent.inner, trial.inner_trial) if commit
                 else self.inner.rollback(parent.inner, trial.inner_trial))
        result = trial.candidate if commit else parent
        if commit and inner.state_sha256 != result.inner.state_sha256:
            raise SuspendedControlsV86Error(
                "v86 committed inner state differs from the candidate")
        if not commit and inner is not parent.inner:
            raise SuspendedControlsV86Error("v86 rollback changed inner state")
        self._open_parent = self._open_trial = None
        return result

    def commit(self, parent, trial):
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial):
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial):
        return self.rollback(parent, trial)

    def checkpoint(self, state: SuspendedControlsStateV86) -> dict[str, object]:
        validate_suspended_controls_state_v86(state, self.execution)
        if self._open_trial is not None:
            raise SuspendedControlsV86Error(
                "v86 checkpoint requires a committed point")
        body = {
            "schema": SCHEMA,
            "scope_id": CHECKPOINT_SCOPE_ID,
            "semantic_id": SEMANTIC_ID,
            "execution_fingerprint": self.execution.fingerprint,
            "controls": {
                "iopsus": self.execution.iopsus,
                "neglect_entrainment": self.execution.neglect_entrainment,
                "suscor": self.execution.suscor,
            },
            "morphological_bed_change_kg_m2":
                state.morphological_bed_change_kg_m2.tolist(),
            "morphological_exchange_ledger_kg":
                state.morphological_exchange_ledger_kg.tolist(),
            "morphological_boundary_ledger_kg":
                state.morphological_boundary_ledger_kg.tolist(),
            "accepted_control_chain_sha256":
                state.accepted_control_chain_sha256,
            "inner_checkpoint_v84": self.inner.checkpoint(state.inner),
            "state_sha256": state.state_sha256,
        }
        return {**body, "fingerprint": _fingerprint(body)}

    def restore_checkpoint(self, document: Mapping[str, object]):
        if self._open_trial is not None:
            raise SuspendedControlsV86Error(
                "v86 restart cannot replace an open transaction")
        expected = {
            "schema", "scope_id", "semantic_id", "execution_fingerprint",
            "controls", "morphological_bed_change_kg_m2",
            "morphological_exchange_ledger_kg",
            "morphological_boundary_ledger_kg",
            "accepted_control_chain_sha256", "inner_checkpoint_v84",
            "state_sha256", "fingerprint",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise SuspendedControlsV86Error("v86 checkpoint inventory differs")
        body = {key: document[key] for key in document if key != "fingerprint"}
        if (document["fingerprint"] != _fingerprint(body) or
                document["schema"] != SCHEMA or
                document["scope_id"] != CHECKPOINT_SCOPE_ID or
                document["semantic_id"] != SEMANTIC_ID or
                document["execution_fingerprint"] != self.execution.fingerprint or
                document["controls"] != {
                    "iopsus": self.execution.iopsus,
                    "neglect_entrainment": self.execution.neglect_entrainment,
                    "suscor": self.execution.suscor,
                }):
            raise SuspendedControlsV86Error("v86 checkpoint identity differs")
        inner = self.inner.restore_checkpoint(document["inner_checkpoint_v84"])
        state = _wrap_state(
            self.execution, inner,
            bed_change=document["morphological_bed_change_kg_m2"],
            exchange_ledger=document["morphological_exchange_ledger_kg"],
            boundary_ledger=document["morphological_boundary_ledger_kg"],
            chain=str(document["accepted_control_chain_sha256"]))
        if state.state_sha256 != document["state_sha256"]:
            raise SuspendedControlsV86Error("v86 checkpoint state differs")
        return state


def method_identity_v86() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "locked_upstream_commit": LOCKED_UPSTREAM_COMMIT,
        "integrator": INTEGRATOR,
        "iopsus": {
            "typed_values": {0: IOPSUS_FIXED, 1: IOPSUS_FLOW_DEPENDENT},
            "source_routes": (-1, -2, -4),
            "v84_active_value": 0,
            "active_equations_qualified_cpu_cuda": True,
            "active_v84_lifecycle": False,
        },
        "neglect_entrainment": {
            False: BED_BALANCE_EXCHANGE,
            True: BED_BALANCE_FLUX_DIVERGENCE,
            "scalar_lifecycle_unchanged": True,
            "inner_v78_bed_state_source_equivalent_under_switch": False,
            "morphological_ledger_applied_to_production_exner": False,
        },
        "suscor": {
            False: SUSCOR_OFF_2DH_NOOP,
            True: SUSCOR_ON_2DH_NOOP,
            "2dh_application_count": 0,
            "guard": "kmax>1",
            "outer_vector_gate": (
                "SusCor=false suppresses bott3d ssuu/ssvv construction"),
            "neglect_entrainment_true_requires": True,
        },
        "legal_route_matrix": LEGAL_ROUTE_MATRIX,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "automatic_cpu_fallback": False,
        "long_run_claimed": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "BED_BALANCE_EXCHANGE", "BED_BALANCE_FLUX_DIVERGENCE",
    "CHECKPOINT_SCOPE_ID", "INTEGRATOR", "IOPSUS_FIXED",
    "IOPSUS_FLOW_DEPENDENT", "LEGAL_ROUTE_MATRIX", "LOCKED_SOURCES",
    "LOCKED_UPSTREAM_COMMIT",
    "MAX_ACCEPTED_UPDATES", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "SUSCOR_OFF_2DH_NOOP", "SUSCOR_ON_2DH_NOOP",
    "SuspendedControlsExecutionV86", "SuspendedControlsLifecycleV86",
    "SuspendedControlsStateV86", "SuspendedControlsTrialV86",
    "SuspendedControlsV86Error", "build_suspended_controls_trial_v86",
    "execution_document_v86", "iopsus_diameter_v86", "method_identity_v86",
    "resolve_suspended_controls_execution_v86",
    "resolve_suspended_controls_state_v86",
    "TOPOLOGY_SELECTOR_ABI_ID", "suspended_flux_divergence_topology_v86",
    "suspended_flux_divergence_v86", "validate_suspended_controls_state_v86",
    "verify_locked_sources_v86",
]
