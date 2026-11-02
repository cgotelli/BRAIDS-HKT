"""Bounded Q4 classic-FLOW ``EulerisoGLM`` disposition (v77).

This additive control authenticates the locked Delft3D4 FLOW 2DH source and
then makes the current-only/no-wave result explicit: ``EulerisoGLM`` is an
identity operation because ``euler.f90`` copies the input velocities when
``wave`` is false and ``tritra.f90`` calls the roller flux correction only
when the roller process is active.  In the classic source, the default false
value retains GLM suspended transport when wave forcing exists, while the bed
and reference-concentration velocity remain Eulerian.  The control never
changes concentration in this bounded no-wave gate.

Wave/roller execution is intentionally *not* admitted here.  A separate
source-equation probe can authenticate the exact ``grmas*``/``grfac*`` forcing
and show when the classic 2DH velocity-frame correction would be active, but
an execution receipt requesting that scope is rejected before the first
update.  ``GLMisoEuler`` is retained in receipts and logs as a classic-FLOW
parsed/printed value with no classic computational consumer; its active
``jatranspvel=0`` meaning belongs to D-Flow FM.

Only immutable accepted states own the update count.  Trials may be committed,
rolled back, or cancelled, and no execution can accept more than four updates.
No Delft3D source is copied, generated, or modified by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SCHEMA = 1
SCOPE_ID = "braids-q4-classic-flow-euleriso-glm-v77-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
MAX_ACCEPTED_UPDATES = 4
BACKEND_CPU = "cpu"
BACKEND_CUDA = "implicit_fwave_midpoint_gpu"
BACKENDS = frozenset((BACKEND_CPU, BACKEND_CUDA))
ENGINE_CLASSIC_FLOW_2DH = "delft3d4_flow_2dh"
ENGINE_DFLOW_FM = "dflowfm"
EULERISO_DEFAULT = False
GLMISO_DEFAULT = False
NO_WAVE_DISPOSITION = "authenticated_identity_current_only_no_wave"
GLMISO_CLASSIC_DISPOSITION = "parsed_logged_unused_classic_flow"
GLMISO_OTHER_ENGINE_DISPOSITION = "dflowfm_only_active_jatranspvel_0"
CUDA_THREADS_PER_BLOCK = 128
_SHA = re.compile(r"[0-9a-f]{64}\Z")


# Every file is a locked, read-only reference.  The line ranges are evidence
# metadata only; the whole-file digest is authenticated before resolution.
LOCKED_SOURCES = {
    "morphology_data": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_data/src/morphology_data_module.f90"),
        "sha256": "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac",
        "lines": "551-552,1756-1757",
        "role": "control declarations and false defaults",
    },
    "rdmor": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_io/src/rdmor.f90"),
        "sha256": "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
        "lines": "568-574,1750-1765",
        "role": "both parsers and classic-FLOW logging",
    },
    "rdsedmortra": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_io/src/input/rdsedmortra.f90"),
        "sha256": "d20803ae2b7646973b8b1aa8e8252ac2c00c6dc2a690b665b557ba236ab5854f",
        "lines": "141-156,225-237",
        "role": "classic morphology reader and echo call chain",
    },
    "properties": {
        "path": (
            "resources/Delft3D-main/src/utils_lgpl/deltares_common/packages/"
            "deltares_common/src/properties.f90"),
        "sha256": "5bb730072df9591781c7dd9be27963e0c59a531bd4421bbfa40469b65f27920e",
        "lines": "2320-2362",
        "role": "case-insensitive property lookup and parse-status behavior",
    },
    "string_module": {
        "path": (
            "resources/Delft3D-main/src/utils_lgpl/deltares_common/packages/"
            "deltares_common/src/string_module.f90"),
        "sha256": "91a7d1e44fe7cec048d4845bb5ca26fd60e46ff6e90e20cb53d17cd5f07ff357",
        "lines": "1126-1144",
        "role": "classic accepted logical literals and numeric conversion",
    },
    "trisol": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/main/trisol.f90"),
        "sha256": "4b3e7ce63aed6c0cf95fe20ea3b9aa80fee9b3cd6fdf8101f0fbf7af5ca689e0",
        "lines": "1998-2003,2190-2205,3071-3076,3264-3279",
        "role": "EulerisoGLM classic-FLOW consumers and tritra forwarding",
    },
    "euler": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute/euler.f90"),
        "sha256": "47484fa30c54a76cfedf204bdaeee9a865458f63090456402cb0a9d75525084e",
        "lines": "128-149,360-390",
        "role": "wave gate, exact 2DH mass-flux correction, no-wave copy",
    },
    "tritra": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute/tritra.f90"),
        "sha256": "e76ed0399b829009a77b13d810be3e0a834915a120b03ccd4600ea908bc5d676",
        "lines": "279-291,339-346,464-476,519-531",
        "role": "roller-only calls carrying EulerisoGLM into rolcor",
    },
    "rolcor": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_roller/rolcor.f90"),
        "sha256": "999484b627579b285ba8af55b44b4f156bd219435c4997b4771966ab3e5ec909",
        "lines": "37-40,126-240",
        "role": "EulerisoGLM roller constituent-flux correction",
    },
    "massfl": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_roller/massfl.f90"),
        "sha256": "d7953d54a34f3ebde688e258783eaab5cbf6b252fd83f88985f24c40c7da91c5",
        "lines": "86-117",
        "role": "2DH wave/roller mass-flux producer for grmasu/grmasv",
    },
    "bott3d": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/bott3d.f90"),
        "sha256": "5abfee8230f66ba42b3eb87ff89948f65a0bfccc915a015c25f8c1412f4e47b1",
        "lines": "360-383,466,548,569-610",
        "role": "selected velocity use is 3D correction; 2DH uses DIFU flux",
    },
    "erosed": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/erosed.f90"),
        "sha256": "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
        "lines": "303-304,660-667,856-870,1341-1358",
        "role": "classic bedload/reference concentration remains Eulerian",
    },
    "z_trisol": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/main/z_trisol.f90"),
        "sha256": "28f8eb1eca9eb49cf987e811b45f3eb3921249b8c8a6c96ab19fd9f5607bb900",
        "lines": "1673-1722,1818-1831,2478-2537,2756-2769",
        "role": "negative control: neither keyword controls z-model path",
    },
    "z_trisol_nhfull": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/main/z_trisol_nhfull.f90"),
        "sha256": "0975606b2ab6d8d2f48f665e6cff094d9d4d4ec79cc391d5972a58b079754b2e",
        "lines": "runtime-wide negative consumer search",
        "role": "negative control: neither keyword controls z-model NH path",
    },
    "fm_sedmorinit": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/dflowfm/packages/"
            "dflowfm_kernel/src/dflowfm_kernel/prepost/flow_sedmorinit.f90"),
        "sha256": "a1195b40a5bc2c5dceb10319af906491b11b26a514274289367a40d9b2acf546",
        "lines": "171-184",
        "role": "separate D-Flow FM active GLMisoEuler consumer",
    },
}


class EulerisoGLMV77Error(ValueError):
    """Fail-closed v77 error carrying a stable diagnostic category."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


def _fail(message: str, category: str) -> None:
    raise EulerisoGLMV77Error(message, category=category)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise EulerisoGLMV77Error(
            "v77 data are not finite canonical JSON",
            category="invalid_serialization",
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        item = np.ascontiguousarray(value)
        digest.update(item.dtype.str.encode("ascii"))
        digest.update(_canonical(list(item.shape)))
        digest.update(item.tobytes(order="C"))
    return digest.hexdigest()


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"v77 {name} must be an exact Boolean", "invalid_control")
    return value


def _array(value: object, shape: tuple[int, ...], name: str, *,
           boolean: bool = False, positive: bool = False,
           nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        _fail(f"v77 {name} must have shape {shape}", "invalid_array")
    if boolean:
        if raw.dtype.kind != "b":
            _fail(f"v77 {name} must be Boolean", "invalid_array")
        result = np.ascontiguousarray(raw, dtype=np.bool_)
    else:
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            _fail(f"v77 {name} must be finite numeric data", "invalid_array")
        result = np.ascontiguousarray(raw, dtype=np.float64)
        if positive and np.any(result <= 0.0):
            _fail(f"v77 {name} must be positive", "invalid_array")
        if nonnegative and np.any(result < 0.0):
            _fail(f"v77 {name} must be nonnegative", "invalid_array")
    result.setflags(write=False)
    return result


def _integer_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "iu":
        _fail(f"v77 {name} must be an integer array with shape {shape}",
              "invalid_array")
    result = np.ascontiguousarray(raw, dtype=np.int32)
    result.setflags(write=False)
    return result


def verify_locked_sources_v77(
    root: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Hash every locked source anchor; this routine performs no writes."""
    repository = (Path(root).resolve() if root is not None
                  else Path(__file__).resolve().parents[2])
    rows: list[dict[str, object]] = []
    for symbol, expected in LOCKED_SOURCES.items():
        relative = expected["path"]
        path = repository / relative
        if not path.is_file():
            _fail(f"locked Delft3D source is missing: {relative}",
                  "source_integrity")
        actual = _file_sha256(path)
        if actual != expected["sha256"]:
            _fail(f"locked Delft3D source hash differs: {relative}",
                  "source_integrity")
        rows.append({
            "symbol": symbol,
            "path": relative,
            "lines": expected["lines"],
            "role": expected["role"],
            "sha256": actual,
            "read_only_reference": True,
        })
    return tuple(rows)


def semantic_document_v77() -> dict[str, object]:
    """Return the source-audited, backend-independent v77 contract."""
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "controls": {
            "EulerisoGLM": {
                "default": EULERISO_DEFAULT,
                "classic_flow_classification": "active_conditional",
                "classic_default_velocity_disposition": (
                    "GLM suspended transport under wave forcing; Eulerian "
                    "bed and reference-concentration velocity"),
                "euler_velocity_probe_condition": (
                    "requested_true AND wave_enabled AND exact typed 2DH "
                    "wave/roller fixture fields present"),
                "rolcor_constituent_flux_condition": (
                    "requested_true AND wave_enabled AND roller_enabled AND "
                    "cyclic-method AND lstsci>0 AND exact typed fields; then "
                    "face codes and wave period gate each flux"),
                "current_only_no_wave_effect": "bitwise_identity",
            },
            "GLMisoEuler": {
                "default": GLMISO_DEFAULT,
                "classic_flow_classification": "parsed_logged_unused",
                "classic_flow_computational_consumers": 0,
                "classic_velocity_disposition": (
                    "no effect: default GLM suspended plus Eulerian bed and "
                    "reference-concentration velocity remains unchanged"),
                "other_engine": GLMISO_OTHER_ENGINE_DISPOSITION,
                "both_true_classic_result": (
                    "EulerisoGLM behavior applies; GLMisoEuler remains inert"),
            },
        },
        "exact_2dh_source_equations": {
            "euleriso_glm_rolcor_flux": (
                "dqx=-sw*(grmasu+grfacu)*guu*thick(1); "
                "dqy=-sw*(grmasv+grfacv)*gvv*thick(1)"),
            "route_gate": (
                "EulerisoGLM AND wave AND roller AND cyclic-method AND "
                "lstsci>0; per-face kfu/kfv==1, kcu/kcv<=2, period>0.5s"),
            "no_wave": "rolcor is not called; concentration operation is identity",
            "constituent_flux_gate": (
                "tritra calls rolcor only for cyclic suspended constituents "
                "with roller; EulerisoGLM selects the exact 2DH branch above"),
        },
        "runtime_scope": {
            "engine": ENGINE_CLASSIC_FLOW_2DH,
            "forcing": "current_only_no_wave_no_roller",
            "wave_execution_supported": False,
            "classic_z_model_path_supported": False,
            "concentration_operation": "identity_only",
            "accepted_update_limit": MAX_ACCEPTED_UPDATES,
            "automatic_backend_fallback": False,
            "normal_product_admission": False,
        },
        "claims_not_made": [
            "wave or roller execution support",
            "D-Flow FM execution support",
            "long-term morphodynamic equivalence",
            "a new suspended-transport discretisation",
            "external-run provenance for the source-equation fixture",
        ],
        "parser_hardening": (
            "v77 accepts exact Booleans only; it does not reproduce the "
            "classic parser's invalid-text-retains-current-value behavior"),
    }


SEMANTIC_ID = _fingerprint(semantic_document_v77())


@dataclass(frozen=True)
class ExactWaveRollerForcingV77:
    """Content-authenticated classic-FLOW 2DH wave/roller fixture fields."""

    hu_m: np.ndarray
    hv_m: np.ndarray
    grmasu_m2_s: np.ndarray
    grmasv_m2_s: np.ndarray
    grfacu_m2_s: np.ndarray
    grfacv_m2_s: np.ndarray
    guu_inverse_m: np.ndarray
    gvv_inverse_m: np.ndarray
    u_wave_period_s: np.ndarray
    v_wave_period_s: np.ndarray
    kfu: np.ndarray
    kcu: np.ndarray
    kfv: np.ndarray
    kcv: np.ndarray
    layer_thickness: float
    roller_correction_mode: int
    source_anchor_sha256: str
    forcing_sha256: str


def _forcing_token(value: ExactWaveRollerForcingV77) -> str:
    return _fingerprint({
        "schema": SCHEMA,
        "producer": "locked_classic_flow_2dh_tritra_rolcor_fixture",
        "source_anchor_sha256": value.source_anchor_sha256,
        "layer_thickness": value.layer_thickness,
        "roller_correction_mode": value.roller_correction_mode,
        "arrays_sha256": _array_digest(
            value.hu_m, value.hv_m,
            value.grmasu_m2_s, value.grmasv_m2_s,
            value.grfacu_m2_s, value.grfacv_m2_s,
            value.guu_inverse_m, value.gvv_inverse_m,
            value.u_wave_period_s, value.v_wave_period_s,
            value.kfu, value.kcu, value.kfv, value.kcv,
        ),
    })


ROL_COR_SOURCE_ANCHOR_SHA256 = _fingerprint({
    symbol: LOCKED_SOURCES[symbol]["sha256"]
    for symbol in ("tritra", "rolcor", "massfl")
})


def resolve_exact_wave_roller_forcing_v77(
    *, hu_m: object, hv_m: object,
    grmasu_m2_s: object, grmasv_m2_s: object,
    grfacu_m2_s: object, grfacv_m2_s: object,
    guu_inverse_m: object, gvv_inverse_m: object,
    u_wave_period_s: object, v_wave_period_s: object,
    kfu: object, kcu: object, kfv: object, kcv: object,
    layer_thickness: float = 1.0,
    roller_correction_mode: int = 2,
) -> ExactWaveRollerForcingV77:
    """Bind exact fixture fields consumed by 2DH ``euler`` and ``rolcor``."""
    raw = np.asarray(grmasu_m2_s)
    if raw.ndim != 1 or raw.size < 1:
        _fail("v77 forcing must contain a nonempty face vector",
              "invalid_forcing")
    shape = (int(raw.size),)
    if type(layer_thickness) is not float or not np.isfinite(layer_thickness) \
            or layer_thickness <= 0.0:
        _fail("v77 2DH layer thickness must be a positive float",
              "invalid_forcing")
    if type(roller_correction_mode) is not int or \
            roller_correction_mode not in (0, 1, 2):
        _fail("v77 roller correction mode must be exact 0, 1, or 2",
              "invalid_forcing")
    values = (
        _array(hu_m, shape, "hu", positive=True),
        _array(hv_m, shape, "hv", positive=True),
        _array(raw, shape, "grmasu"),
        _array(grmasv_m2_s, shape, "grmasv"),
        _array(grfacu_m2_s, shape, "grfacu"),
        _array(grfacv_m2_s, shape, "grfacv"),
        _array(guu_inverse_m, shape, "guu", positive=True),
        _array(gvv_inverse_m, shape, "gvv", positive=True),
        _array(u_wave_period_s, shape, "u wave period", nonnegative=True),
        _array(v_wave_period_s, shape, "v wave period", nonnegative=True),
        _integer_array(kfu, shape, "kfu"),
        _integer_array(kcu, shape, "kcu"),
        _integer_array(kfv, shape, "kfv"),
        _integer_array(kcv, shape, "kcv"),
    )
    placeholder = ExactWaveRollerForcingV77(
        *values, layer_thickness, roller_correction_mode,
        ROL_COR_SOURCE_ANCHOR_SHA256, "")
    return ExactWaveRollerForcingV77(
        *values, layer_thickness, roller_correction_mode,
        ROL_COR_SOURCE_ANCHOR_SHA256,
        _forcing_token(placeholder))


def validate_exact_wave_roller_forcing_v77(
    value: ExactWaveRollerForcingV77,
) -> ExactWaveRollerForcingV77:
    if not isinstance(value, ExactWaveRollerForcingV77):
        _fail("v77 wave/roller forcing lacks a typed content receipt",
              "invalid_forcing")
    shape = value.hu_m.shape
    float_fields = (
        value.hu_m, value.hv_m, value.grmasu_m2_s, value.grmasv_m2_s,
        value.grfacu_m2_s, value.grfacv_m2_s, value.guu_inverse_m,
        value.gvv_inverse_m, value.u_wave_period_s, value.v_wave_period_s)
    integer_fields = (value.kfu, value.kcu, value.kfv, value.kcv)
    if (len(shape) != 1 or shape[0] < 1 or
            any(item.shape != shape or item.dtype != np.float64 or
                item.flags.writeable or not np.isfinite(item).all()
                for item in float_fields) or
            any(item.shape != shape or item.dtype != np.int32 or
                item.flags.writeable for item in integer_fields) or
            np.any(value.hu_m <= 0.0) or np.any(value.hv_m <= 0.0) or
            np.any(value.guu_inverse_m <= 0.0) or
            np.any(value.gvv_inverse_m <= 0.0) or
            np.any(value.u_wave_period_s < 0.0) or
            np.any(value.v_wave_period_s < 0.0) or
            value.source_anchor_sha256 != ROL_COR_SOURCE_ANCHOR_SHA256 or
            type(value.layer_thickness) is not float or
            not np.isfinite(value.layer_thickness) or
            value.layer_thickness <= 0.0 or
            type(value.roller_correction_mode) is not int or
            value.roller_correction_mode not in (0, 1, 2) or
            value.forcing_sha256 != _forcing_token(value)):
        _fail("v77 wave/roller forcing is stale or mutated",
              "invalid_forcing")
    return value


@dataclass(frozen=True)
class SourceGateResultV77:
    active: bool
    u_flux_delta: np.ndarray
    v_flux_delta: np.ndarray
    active_face_count: int
    forcing_sha256: str | None
    result_sha256: str


@dataclass(frozen=True)
class EulerVelocityProbeResultV77:
    """Audit-only result for the distinct 2DH ``euler.f90`` branch."""

    active: bool
    u_velocity_delta_m_s: np.ndarray
    v_velocity_delta_m_s: np.ndarray
    active_face_count: int
    forcing_sha256: str | None
    result_sha256: str


def evaluate_euler_velocity_probe_v77(
    *, euleriso_glm: bool, wave_enabled: bool,
    forcing: ExactWaveRollerForcingV77 | None,
) -> EulerVelocityProbeResultV77:
    """Evaluate only ``euler.f90`` velocity conversion; never a runtime route."""
    requested = _strict_bool(euleriso_glm, "EulerisoGLM")
    wave = _strict_bool(wave_enabled, "wave_enabled")
    route_ready = requested and wave
    if route_ready and forcing is None:
        _fail("EulerisoGLM euler velocity activity requires exact fixture fields",
              "missing_exact_forcing")
    if forcing is not None:
        forcing = validate_exact_wave_roller_forcing_v77(forcing)
        shape = forcing.hu_m.shape
    else:
        shape = (1,)
    du = np.zeros(shape, dtype=np.float64)
    dv = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.bool_)
    if route_ready and forcing is not None:
        roller_u = (forcing.grmasu_m2_s if
                    forcing.roller_correction_mode > 0 else 0.0)
        roller_v = (forcing.grmasv_m2_s if
                    forcing.roller_correction_mode > 0 else 0.0)
        u_mask = forcing.kfu == 1
        v_mask = forcing.kfv == 1
        du[u_mask] = -(np.asarray(roller_u) + forcing.grfacu_m2_s)[u_mask] \
            / forcing.hu_m[u_mask]
        dv[v_mask] = -(np.asarray(roller_v) + forcing.grfacv_m2_s)[v_mask] \
            / forcing.hv_m[v_mask]
        mask = u_mask | v_mask
    du.setflags(write=False)
    dv.setflags(write=False)
    count = int(np.count_nonzero(mask))
    active = bool(route_ready and count > 0)
    body = {
        "probe": "euler_2dh_velocity",
        "active": active,
        "active_face_count": count,
        "forcing_sha256": None if forcing is None else forcing.forcing_sha256,
        "arrays_sha256": _array_digest(du, dv),
    }
    return EulerVelocityProbeResultV77(
        active, du, dv, count, body["forcing_sha256"], _fingerprint(body))


def evaluate_rolcor_constituent_flux_probe_v77(
    *, euleriso_glm: bool, wave_enabled: bool, roller_enabled: bool,
    transport_solver: str, constituent_count: int,
    forcing: ExactWaveRollerForcingV77 | None,
) -> SourceGateResultV77:
    """Evaluate the locked 2DH ``tritra``/``rolcor`` flux predicate.

    This is an equation/audit probe only.  Runtime receipts remain current-only
    and reject ``wave_enabled=True`` in :func:`resolve_execution_v77`.
    """
    requested = _strict_bool(euleriso_glm, "EulerisoGLM")
    wave = _strict_bool(wave_enabled, "wave_enabled")
    roller = _strict_bool(roller_enabled, "roller_enabled")
    if type(transport_solver) is not str:
        _fail("v77 transport_solver must be a string", "invalid_control")
    if type(constituent_count) is not int or constituent_count < 0:
        _fail("v77 constituent_count must be a nonnegative integer",
              "invalid_control")
    route_ready = (requested and wave and roller and
                   transport_solver == "cyclic-method" and
                   constituent_count > 0)
    if route_ready and forcing is None:
        _fail("EulerisoGLM roller flux activity requires exact fixture fields",
              "missing_exact_forcing")
    if forcing is not None:
        forcing = validate_exact_wave_roller_forcing_v77(forcing)
        shape = forcing.grmasu_m2_s.shape
    else:
        shape = (1,)
    dqx = np.zeros(shape, dtype=np.float64)
    dqy = np.zeros(shape, dtype=np.float64)
    active_mask = np.zeros(shape, dtype=np.bool_)
    if route_ready and forcing is not None:
        u_mask = ((forcing.kfu == 1) & (forcing.kcu <= 2) &
                  (forcing.u_wave_period_s > 0.5))
        v_mask = ((forcing.kfv == 1) & (forcing.kcv <= 2) &
                  (forcing.v_wave_period_s > 0.5))
        dqx[u_mask] = -(forcing.grmasu_m2_s[u_mask] +
                        forcing.grfacu_m2_s[u_mask]) * \
            forcing.guu_inverse_m[u_mask] * forcing.layer_thickness
        dqy[v_mask] = -(forcing.grmasv_m2_s[v_mask] +
                        forcing.grfacv_m2_s[v_mask]) * \
            forcing.gvv_inverse_m[v_mask] * forcing.layer_thickness
        active_mask = u_mask | v_mask
    dqx.setflags(write=False)
    dqy.setflags(write=False)
    count = int(np.count_nonzero(active_mask))
    active = bool(route_ready and count > 0)
    body = {
        "active": active,
        "active_face_count": count,
        "forcing_sha256": None if forcing is None else forcing.forcing_sha256,
        "arrays_sha256": _array_digest(dqx, dqy),
    }
    return SourceGateResultV77(
        active, dqx, dqy, count, body["forcing_sha256"], _fingerprint(body))


# Compatibility alias inside the additive v77 surface.  New evidence uses the
# explicit name so the rolcor flux result cannot be confused with euler.f90.
evaluate_source_gate_v77 = evaluate_rolcor_constituent_flux_probe_v77


def resolve_execution_v77(
    *, backend: str, euleriso_glm: bool = EULERISO_DEFAULT,
    glmiso_euler: bool = GLMISO_DEFAULT,
    engine: str = ENGINE_CLASSIC_FLOW_2DH,
    current_only: bool = True, wave_enabled: bool = False,
    roller_enabled: bool = False,
    exact_wave_roller_forcing: ExactWaveRollerForcingV77 | None = None,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Resolve one fail-closed current-only execution identity."""
    if type(backend) is not str or backend not in BACKENDS:
        _fail("v77 backend must be explicit CPU or CUDA",
              "unsupported_backend")
    if type(engine) is not str or engine != ENGINE_CLASSIC_FLOW_2DH:
        if engine == ENGINE_DFLOW_FM:
            _fail("D-Flow FM GLMisoEuler is a separate unsupported engine route",
                  "unsupported_engine")
        _fail("v77 requires the classic Delft3D4 FLOW 2DH engine identity",
              "unsupported_engine")
    euler = _strict_bool(euleriso_glm, "EulerisoGLM")
    glm = _strict_bool(glmiso_euler, "GLMisoEuler")
    current = _strict_bool(current_only, "current_only")
    wave = _strict_bool(wave_enabled, "wave_enabled")
    roller = _strict_bool(roller_enabled, "roller_enabled")
    if not current:
        _fail("v77 admits only the current-only scope", "unsupported_scope")
    if exact_wave_roller_forcing is not None:
        exact_wave_roller_forcing = validate_exact_wave_roller_forcing_v77(
            exact_wave_roller_forcing)
    if exact_wave_roller_forcing is not None and not (wave and roller):
        _fail("exact wave/roller forcing requires both process flags",
              "orphaned_forcing")
    if roller and not wave:
        _fail("roller execution requires the wave process and exact forcing",
              "incompatible_physics")
    if wave and roller and euler and exact_wave_roller_forcing is None:
        _fail("EulerisoGLM wave execution lacks exact mass-flux forcing",
              "missing_exact_forcing")
    if wave or roller:
        # The forcing is validated first so no vague "wave=true" shortcut can
        # masquerade as qualification.  Actual wave execution remains outside
        # this bounded control even when the exact fields are present.
        _fail("v77 does not support wave/roller execution",
              "unsupported_wave_execution")
    sources = verify_locked_sources_v77(root)
    controls = [
        {
            "keyword": "EulerisoGLM", "requested": euler,
            "default": EULERISO_DEFAULT, "effective": False,
            "disposition": NO_WAVE_DISPOSITION,
            "reason": "locked euler.f90 wave=false copy branch",
        },
        {
            "keyword": "GLMisoEuler", "requested": glm,
            "default": GLMISO_DEFAULT, "effective": False,
            "disposition": GLMISO_CLASSIC_DISPOSITION,
            "reason": (
                "classic FLOW has data/parser/log references but no "
                "computational consumer; active meaning is D-Flow FM-only"),
        },
    ]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": backend,
        "engine": engine,
        "current_only": current,
        "wave_enabled": wave,
        "roller_enabled": roller,
        "euleriso_glm": euler,
        "glmiso_euler": glm,
        "euleriso_glm_effective": False,
        "glmiso_euler_effective": False,
        "control_log": controls,
        "locked_sources": list(sources),
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "concentration_operation": "bitwise_identity",
        "wave_execution_supported": False,
        "automatic_backend_fallback": False,
        "gpu_integrator": (
            "implicit_fwave_midpoint" if backend == BACKEND_CUDA else None),
        "normal_product_admission": False,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_execution_v77(
    value: Mapping[str, object], *, root: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail("v77 execution must be an object", "invalid_execution")
    expected = {
        "schema", "scope_id", "semantic_id", "backend", "engine",
        "current_only", "wave_enabled", "roller_enabled", "euleriso_glm",
        "glmiso_euler", "euleriso_glm_effective",
        "glmiso_euler_effective", "control_log", "locked_sources",
        "accepted_update_limit", "concentration_operation",
        "wave_execution_supported", "automatic_backend_fallback",
        "gpu_integrator", "normal_product_admission", "fingerprint",
    }
    if set(value) != expected:
        _fail("v77 execution field inventory differs", "invalid_execution")
    rebuilt = resolve_execution_v77(
        backend=value["backend"], euleriso_glm=value["euleriso_glm"],
        glmiso_euler=value["glmiso_euler"], engine=value["engine"],
        current_only=value["current_only"],
        wave_enabled=value["wave_enabled"],
        roller_enabled=value["roller_enabled"], root=root,
    )
    if rebuilt != dict(value):
        _fail("v77 execution identity or source receipt differs",
              "invalid_execution")
    return rebuilt


@dataclass(frozen=True)
class EulerisoGLMStateV77:
    execution_fingerprint: str
    concentration_kg_m3: np.ndarray
    accepted_updates: int
    identity_checks: int
    euleriso_active_updates: int
    glmiso_logged_updates: int
    state_sha256: str


def _state_token(value: EulerisoGLMStateV77) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "accepted_updates": value.accepted_updates,
        "identity_checks": value.identity_checks,
        "euleriso_active_updates": value.euleriso_active_updates,
        "glmiso_logged_updates": value.glmiso_logged_updates,
        "concentration_sha256": _array_digest(value.concentration_kg_m3),
    })


def resolve_state_v77(
    *, execution: Mapping[str, object], concentration_kg_m3: object,
    accepted_updates: int = 0, identity_checks: int = 0,
    euleriso_active_updates: int = 0, glmiso_logged_updates: int = 0,
) -> EulerisoGLMStateV77:
    resolved = load_execution_v77(execution)
    raw = np.asarray(concentration_kg_m3)
    if raw.ndim != 2 or 0 in raw.shape:
        _fail("v77 concentration must be nonempty (fraction,cell) data",
              "invalid_concentration")
    concentration = _array(
        raw, tuple(raw.shape), "concentration", nonnegative=True)
    counts = (accepted_updates, identity_checks, euleriso_active_updates,
              glmiso_logged_updates)
    if any(type(item) is not int or item < 0 for item in counts) or \
            accepted_updates > MAX_ACCEPTED_UPDATES or \
            identity_checks != accepted_updates or \
            euleriso_active_updates != 0 or \
            glmiso_logged_updates != accepted_updates:
        _fail("v77 accepted-state counters are inconsistent",
              "invalid_state")
    placeholder = EulerisoGLMStateV77(
        resolved["fingerprint"], concentration, accepted_updates,
        identity_checks, euleriso_active_updates, glmiso_logged_updates, "")
    return EulerisoGLMStateV77(
        placeholder.execution_fingerprint, placeholder.concentration_kg_m3,
        placeholder.accepted_updates, placeholder.identity_checks,
        placeholder.euleriso_active_updates,
        placeholder.glmiso_logged_updates, _state_token(placeholder))


def validate_state_v77(
    value: EulerisoGLMStateV77, *, execution: Mapping[str, object],
) -> EulerisoGLMStateV77:
    resolved = load_execution_v77(execution)
    structurally_valid = False
    if isinstance(value, EulerisoGLMStateV77):
        counts = (value.accepted_updates, value.identity_checks,
                  value.euleriso_active_updates, value.glmiso_logged_updates)
        concentration = value.concentration_kg_m3
        structurally_valid = (
            isinstance(concentration, np.ndarray) and
            concentration.dtype == np.float64 and concentration.ndim == 2 and
            0 not in concentration.shape and not concentration.flags.writeable and
            np.isfinite(concentration).all() and np.all(concentration >= 0.0) and
            all(type(item) is int and item >= 0 for item in counts) and
            value.accepted_updates <= MAX_ACCEPTED_UPDATES and
            value.identity_checks == value.accepted_updates and
            value.euleriso_active_updates == 0 and
            value.glmiso_logged_updates == value.accepted_updates)
    if (not isinstance(value, EulerisoGLMStateV77) or not structurally_valid or
            value.execution_fingerprint != resolved["fingerprint"] or
            value.state_sha256 != _state_token(value)):
        _fail("v77 state is stale, mutated, or execution-mismatched",
              "invalid_state")
    return value


@dataclass(frozen=True)
class EulerisoGLMTrialV77:
    execution_fingerprint: str
    parent_state_sha256: str
    candidate: EulerisoGLMStateV77
    input_concentration_kg_m3: np.ndarray
    output_concentration_kg_m3: np.ndarray
    control_delta_kg_m3: np.ndarray
    input_sha256: str
    output_sha256: str
    identity_exact: bool
    euleriso_active: bool
    glmiso_disposition: str
    backend: str
    observed_parallel_workers: int
    kernel_launches: int
    trial_sha256: str


def _trial_token(value: EulerisoGLMTrialV77) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "arrays_sha256": _array_digest(
            value.input_concentration_kg_m3,
            value.output_concentration_kg_m3,
            value.control_delta_kg_m3),
        "input_sha256": value.input_sha256,
        "output_sha256": value.output_sha256,
        "identity_exact": value.identity_exact,
        "euleriso_active": value.euleriso_active,
        "glmiso_disposition": value.glmiso_disposition,
        "backend": value.backend,
        "observed_parallel_workers": value.observed_parallel_workers,
        "kernel_launches": value.kernel_launches,
    })


def _publish_trial_v77(
    *, execution: Mapping[str, object], parent: EulerisoGLMStateV77,
    input_concentration_kg_m3: object, output_concentration_kg_m3: object,
    backend: str, observed_parallel_workers: int, kernel_launches: int,
) -> EulerisoGLMTrialV77:
    resolved = load_execution_v77(execution)
    validate_state_v77(parent, execution=resolved)
    if parent.accepted_updates >= MAX_ACCEPTED_UPDATES:
        _fail("v77 accepted update limit was reached", "update_limit")
    if backend != resolved["backend"]:
        _fail("v77 trial backend differs from execution identity",
              "backend_identity_mismatch")
    if (type(observed_parallel_workers) is not int or
            observed_parallel_workers < 1 or type(kernel_launches) is not int or
            kernel_launches < 0):
        _fail("v77 parallel execution receipt is invalid",
              "invalid_runtime_receipt")
    shape = parent.concentration_kg_m3.shape
    input_value = _array(
        input_concentration_kg_m3, shape, "input concentration",
        nonnegative=True)
    output_value = _array(
        output_concentration_kg_m3, shape, "output concentration",
        nonnegative=True)
    input_digest = _array_digest(input_value)
    output_digest = _array_digest(output_value)
    parent_digest = _array_digest(parent.concentration_kg_m3)
    if input_digest != parent_digest or not np.array_equal(
            input_value.view(np.uint64),
            parent.concentration_kg_m3.view(np.uint64)):
        _fail("v77 identity gate input differs from its accepted parent state",
              "parent_concentration_mismatch")
    if input_digest != output_digest or not np.array_equal(
            input_value.view(np.uint64), output_value.view(np.uint64)):
        _fail("v77 identity control attempted to alter concentration",
              "concentration_mutation")
    delta = np.zeros(shape, dtype=np.float64)
    delta.setflags(write=False)
    candidate = resolve_state_v77(
        execution=resolved, concentration_kg_m3=output_value,
        accepted_updates=parent.accepted_updates + 1,
        identity_checks=parent.identity_checks + 1,
        euleriso_active_updates=parent.euleriso_active_updates,
        glmiso_logged_updates=parent.glmiso_logged_updates + 1,
    )
    placeholder = EulerisoGLMTrialV77(
        resolved["fingerprint"], parent.state_sha256, candidate,
        input_value, output_value, delta, input_digest, output_digest, True,
        False, GLMISO_CLASSIC_DISPOSITION, backend,
        observed_parallel_workers, kernel_launches, "")
    return EulerisoGLMTrialV77(
        placeholder.execution_fingerprint, placeholder.parent_state_sha256,
        placeholder.candidate, placeholder.input_concentration_kg_m3,
        placeholder.output_concentration_kg_m3,
        placeholder.control_delta_kg_m3, placeholder.input_sha256,
        placeholder.output_sha256, placeholder.identity_exact,
        placeholder.euleriso_active, placeholder.glmiso_disposition,
        placeholder.backend, placeholder.observed_parallel_workers,
        placeholder.kernel_launches, _trial_token(placeholder))


def begin_trial_v77(
    *, execution: Mapping[str, object], parent: EulerisoGLMStateV77,
    concentration_kg_m3: object | None = None, cancelled: bool = False,
) -> EulerisoGLMTrialV77:
    """Publish one CPU identity candidate bound bitwise to its parent."""
    resolved = load_execution_v77(execution)
    if resolved["backend"] != BACKEND_CPU:
        _fail("CPU begin requires a CPU execution receipt",
              "backend_identity_mismatch")
    validate_state_v77(parent, execution=resolved)
    if _strict_bool(cancelled, "cancelled"):
        _fail("v77 update cancelled before candidate publication",
              "cancelled")
    source = (parent.concentration_kg_m3 if concentration_kg_m3 is None
              else concentration_kg_m3)
    # The explicit copy mirrors the locked no-wave branch.  Publication checks
    # exact float64 bytes, so a future arithmetic change cannot be silent.
    output = np.array(source, dtype=np.float64, order="C", copy=True)
    return _publish_trial_v77(
        execution=resolved, parent=parent,
        input_concentration_kg_m3=source,
        output_concentration_kg_m3=output, backend=BACKEND_CPU,
        observed_parallel_workers=1, kernel_launches=0)


def validate_trial_v77(
    parent: EulerisoGLMStateV77, trial: EulerisoGLMTrialV77, *,
    execution: Mapping[str, object],
) -> EulerisoGLMTrialV77:
    resolved = load_execution_v77(execution)
    validate_state_v77(parent, execution=resolved)
    exact_arrays = False
    if isinstance(trial, EulerisoGLMTrialV77):
        arrays = (trial.input_concentration_kg_m3,
                  trial.output_concentration_kg_m3,
                  trial.candidate.concentration_kg_m3)
        exact_arrays = all(
            item.shape == parent.concentration_kg_m3.shape and
            not item.flags.writeable and np.array_equal(
                item.view(np.uint64), parent.concentration_kg_m3.view(np.uint64))
            for item in arrays)
        zero_bits = np.zeros(
            parent.concentration_kg_m3.shape, dtype=np.float64).view(np.uint64)
        exact_delta = (
            trial.control_delta_kg_m3.shape == parent.concentration_kg_m3.shape and
            not trial.control_delta_kg_m3.flags.writeable and
            np.array_equal(trial.control_delta_kg_m3.view(np.uint64), zero_bits))
    else:
        exact_delta = False
    expected_runtime = (
        (trial.backend == BACKEND_CPU and
         trial.observed_parallel_workers == 1 and trial.kernel_launches == 0)
        if isinstance(trial, EulerisoGLMTrialV77) else False)
    if isinstance(trial, EulerisoGLMTrialV77) and trial.backend == BACKEND_CUDA:
        expected_runtime = (
            trial.observed_parallel_workers == CUDA_THREADS_PER_BLOCK and
            trial.kernel_launches == 1)
    if (not isinstance(trial, EulerisoGLMTrialV77) or
            trial.execution_fingerprint != resolved["fingerprint"] or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.backend != resolved["backend"] or
            trial.trial_sha256 != _trial_token(trial) or
            trial.candidate.accepted_updates != parent.accepted_updates + 1 or
            trial.candidate.identity_checks != parent.identity_checks + 1 or
            trial.candidate.euleriso_active_updates !=
                parent.euleriso_active_updates or
            trial.candidate.glmiso_logged_updates !=
                parent.glmiso_logged_updates + 1 or
            not trial.identity_exact or trial.euleriso_active or
            trial.glmiso_disposition != GLMISO_CLASSIC_DISPOSITION or
            trial.input_sha256 != trial.output_sha256 or
            trial.input_sha256 != _array_digest(
                trial.input_concentration_kg_m3) or
            trial.output_sha256 != _array_digest(
                trial.output_concentration_kg_m3) or
            not exact_arrays or not exact_delta or not expected_runtime):
        _fail("v77 trial is stale, mutated, nonidentity, or mismatched",
              "invalid_trial")
    validate_state_v77(trial.candidate, execution=resolved)
    return trial


def commit_trial_v77(
    parent: EulerisoGLMStateV77, trial: EulerisoGLMTrialV77, *,
    execution: Mapping[str, object],
) -> EulerisoGLMStateV77:
    return validate_trial_v77(
        parent, trial, execution=execution).candidate


def rollback_trial_v77(
    parent: EulerisoGLMStateV77, trial: EulerisoGLMTrialV77, *,
    execution: Mapping[str, object],
) -> EulerisoGLMStateV77:
    validate_trial_v77(parent, trial, execution=execution)
    return parent


def cancel_trial_v77(
    parent: EulerisoGLMStateV77, trial: EulerisoGLMTrialV77, *,
    execution: Mapping[str, object],
) -> EulerisoGLMStateV77:
    return rollback_trial_v77(parent, trial, execution=execution)


def checkpoint_document_v77(
    value: EulerisoGLMStateV77, *, execution: Mapping[str, object],
) -> dict[str, object]:
    resolved = load_execution_v77(execution)
    validate_state_v77(value, execution=resolved)
    body: dict[str, object] = {
        "schema": SCHEMA,
        "checkpoint_scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "state": {
            "concentration_kg_m3": value.concentration_kg_m3.tolist(),
            "accepted_updates": value.accepted_updates,
            "identity_checks": value.identity_checks,
            "euleriso_active_updates": value.euleriso_active_updates,
            "glmiso_logged_updates": value.glmiso_logged_updates,
            "state_sha256": value.state_sha256,
        },
    }
    return {**body, "checkpoint_sha256": _fingerprint(body)}


def load_checkpoint_v77(
    value: Mapping[str, object], *, execution: Mapping[str, object],
) -> EulerisoGLMStateV77:
    resolved = load_execution_v77(execution)
    if not isinstance(value, Mapping) or set(value) != {
            "schema", "checkpoint_scope_id", "semantic_id",
            "execution_fingerprint", "state", "checkpoint_sha256"}:
        _fail("v77 checkpoint field inventory differs", "invalid_checkpoint")
    body = {key: value[key] for key in value if key != "checkpoint_sha256"}
    if (value["schema"] != SCHEMA or
            value["checkpoint_scope_id"] != CHECKPOINT_SCOPE_ID or
            value["semantic_id"] != SEMANTIC_ID or
            value["execution_fingerprint"] != resolved["fingerprint"] or
            value["checkpoint_sha256"] != _fingerprint(body)):
        _fail("v77 checkpoint identity or digest differs",
              "invalid_checkpoint")
    state = value["state"]
    if not isinstance(state, Mapping) or set(state) != {
            "concentration_kg_m3", "accepted_updates", "identity_checks",
            "euleriso_active_updates", "glmiso_logged_updates",
            "state_sha256"}:
        _fail("v77 checkpoint state inventory differs", "invalid_checkpoint")
    restored = resolve_state_v77(
        execution=resolved,
        concentration_kg_m3=state["concentration_kg_m3"],
        accepted_updates=state["accepted_updates"],
        identity_checks=state["identity_checks"],
        euleriso_active_updates=state["euleriso_active_updates"],
        glmiso_logged_updates=state["glmiso_logged_updates"],
    )
    if restored.state_sha256 != state["state_sha256"]:
        _fail("v77 checkpoint state digest differs", "invalid_checkpoint")
    return restored


__all__ = [
    "BACKEND_CPU", "BACKEND_CUDA", "BACKENDS", "CHECKPOINT_SCOPE_ID",
    "CUDA_THREADS_PER_BLOCK", "ENGINE_CLASSIC_FLOW_2DH", "ENGINE_DFLOW_FM",
    "EULERISO_DEFAULT", "EulerVelocityProbeResultV77",
    "ExactWaveRollerForcingV77",
    "EulerisoGLMStateV77", "EulerisoGLMTrialV77", "EulerisoGLMV77Error",
    "GLMISO_CLASSIC_DISPOSITION", "GLMISO_DEFAULT",
    "GLMISO_OTHER_ENGINE_DISPOSITION", "LOCKED_SOURCES",
    "MAX_ACCEPTED_UPDATES", "NO_WAVE_DISPOSITION", "SCHEMA", "SEMANTIC_ID",
    "SCOPE_ID", "SourceGateResultV77", "begin_trial_v77",
    "cancel_trial_v77", "checkpoint_document_v77", "commit_trial_v77",
    "evaluate_euler_velocity_probe_v77",
    "evaluate_rolcor_constituent_flux_probe_v77", "evaluate_source_gate_v77",
    "load_checkpoint_v77", "load_execution_v77",
    "resolve_exact_wave_roller_forcing_v77", "resolve_execution_v77",
    "resolve_state_v77", "rollback_trial_v77", "semantic_document_v77",
    "validate_exact_wave_roller_forcing_v77", "validate_state_v77",
    "validate_trial_v77", "verify_locked_sources_v77",
]
