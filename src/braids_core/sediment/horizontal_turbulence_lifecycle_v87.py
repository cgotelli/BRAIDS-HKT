"""Bounded FLOW 2DH horizontal-turbulence lifecycle composition (Q4/v87).

The locked classic-FLOW call order is material here.  ``turclo`` first forms
the layer-one coefficients from the background field and the *previously
accepted* HLES slot.  Later in the hydraulic update ``lpfluc -> protke ->
detvic`` replaces the HLES slot, and scalar transport consumes the already
formed layer-one ``dicuv``.  Consequently a newly evaluated HLES/Elder field
is used by the next accepted sediment update.  V87 preserves that one-update
lag as restartable transaction state; rejected and cancelled trials cannot
advance it.

The actual suspended lifecycle remains the accepted v82/v78 implementation.
For each trial v87 constructs one ephemeral v82 execution with
``background_dicuv + parent.accepted_dynamic_dicuv``.  The frozen v78 base is
required to have zero diffusion, so there is exactly one horizontal-diffusion
owner and the shared KMAX=1 ``dicuv/0.7`` branch remains constituent
independent.

This module implements the ``detvic`` current-only provider, not the upstream
``lpfluc``/``protke`` operators.  Their accepted TKE-production field is bound
to the hydraulic receipt.  The provider returns both ``vicuv`` and ``dicuv``;
this Q4 gate consumes ``dicuv`` in sediment transport and exposes ``vicuv``
for the separate hydraulic momentum-diffusion owner.
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

from .multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    resolve_multifraction_suspended_state_v78,
    validate_multifraction_suspended_state_v78,
)
from .suspended_support_closures_v82 import (
    SuspendedSupportExecutionV82,
    SuspendedSupportStateV82,
    SuspendedSupportTrialV82,
    _wrap_state as _wrap_state_v82,
    begin_suspended_support_cpu_trial_v82,
    commit_suspended_support_trial_v82,
    resolve_suspended_support_execution_v82,
    rollback_suspended_support_trial_v82,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-horizontal-turbulence-lifecycle-v87-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
INPUT_RECEIPT_SCOPE_ID = SCOPE_ID + ":accepted-detvic-input"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
CALL_ORDER = (
    "turclo(background+previous_accepted_hles)->hydraulics->"
    "lpfluc->protke->detvic(candidate_hles)->tritra(previous_dicuv)"
)
PROVIDER_ID = "locked_delft3d4_flow2dh_detvic_current_only_v1"
BACKGROUND_MODES = ("constant", "spatial", "filedy_freeformatted")
MAX_UPDATES = MAX_ACCEPTED_UPDATES
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HorizontalTurbulenceV87Error(ValueError):
    """Fail-closed v87 configuration, receipt or transaction error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise HorizontalTurbulenceV87Error(
            "v87 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must be positive")
    if nonnegative and result < 0.0:
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must be nonnegative")
    return result


def _frozen_array(value: object, shape: tuple[int, ...], name: str, *,
                  nonnegative: bool = False, integer_mask: bool = False
                  ) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu b":
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must be a finite numeric array with shape {shape}")
    result = np.ascontiguousarray(
        raw, dtype=np.int32 if integer_mask else np.float64)
    if not np.isfinite(result).all():
        raise HorizontalTurbulenceV87Error(f"v87 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must be nonnegative")
    if integer_mask and np.any((result != 0) & (result != 1)):
        raise HorizontalTurbulenceV87Error(
            f"v87 {name} must contain only 0/1")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TurbulenceBackgroundV87:
    mode: str
    vicuv_m2_s: np.ndarray
    dicuv_m2_s: np.ndarray
    source_name: str
    source_sha256: str | None
    mapping_sha256: str | None
    fingerprint: str


def _background_payload(value: TurbulenceBackgroundV87) -> dict:
    return {
        "schema": SCHEMA,
        "mode": value.mode,
        "vicuv_m2_s": value.vicuv_m2_s.tolist(),
        "dicuv_m2_s": value.dicuv_m2_s.tolist(),
        "source_name": value.source_name,
        "source_sha256": value.source_sha256,
        "mapping_sha256": value.mapping_sha256,
    }


def _wrap_background(*, mode: str, vicuv: object, dicuv: object,
                     cell_count: int, source_name: str,
                     source_sha256: str | None = None,
                     mapping_sha256: str | None = None,
                     ) -> TurbulenceBackgroundV87:
    if mode not in BACKGROUND_MODES:
        raise HorizontalTurbulenceV87Error(
            "v87 background mode is unsupported")
    if not isinstance(source_name, str) or not source_name.strip():
        raise HorizontalTurbulenceV87Error(
            "v87 background source name is required")
    arrays = (
        _frozen_array(vicuv, (cell_count,), "background vicuv",
                      nonnegative=True),
        _frozen_array(dicuv, (cell_count,), "background dicuv",
                      nonnegative=True),
    )
    placeholder = TurbulenceBackgroundV87(
        mode, arrays[0], arrays[1], source_name.strip(), source_sha256,
        mapping_sha256, "")
    return TurbulenceBackgroundV87(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_background_payload(placeholder))})


def constant_turbulence_background_v87(
    cell_count: int, *, vicouv_m2_s: float, dicouv_m2_s: float,
) -> TurbulenceBackgroundV87:
    """Resolve classic-FLOW uniform ``Vicouv``/``Dicouv`` background."""
    if isinstance(cell_count, bool) or type(cell_count) is not int or cell_count < 1:
        raise HorizontalTurbulenceV87Error(
            "v87 cell count must be a positive integer")
    vic = _finite(vicouv_m2_s, "Vicouv", nonnegative=True)
    dic = _finite(dicouv_m2_s, "Dicouv", nonnegative=True)
    return _wrap_background(
        mode="constant", vicuv=np.full(cell_count, vic),
        dicuv=np.full(cell_count, dic), cell_count=cell_count,
        source_name="classic_flow_uniform_Vicouv_Dicouv")


def spatial_turbulence_background_v87(
    *, vicuv_m2_s: object, dicuv_m2_s: object, source_name: str,
) -> TurbulenceBackgroundV87:
    """Resolve an authenticated prepared spatial background field."""
    vic = np.asarray(vicuv_m2_s)
    if vic.ndim != 1:
        raise HorizontalTurbulenceV87Error(
            "v87 spatial background must be one-dimensional")
    return _wrap_background(
        mode="spatial", vicuv=vic, dicuv=dicuv_m2_s,
        cell_count=vic.size, source_name=source_name)


def filedy_turbulence_background_v87(
    content: str | bytes, *, nmax: int, mmax: int,
    structured_flat_index_by_cell: object, source_name: str,
    fmtedy: str = "FR",
) -> TurbulenceBackgroundV87:
    """Read the two freeformatted FileDy blocks and map them to BRAIDS cells.

    Locked ``edyfil`` reads ``n`` rows with ``m`` varying fastest: first the
    viscosity block and then, when constituents are active, diffusivity.  The
    caller supplies the already audited structured-cell-to-BRAIDS mapping.
    Binary/unformatted FileDy is deliberately rejected rather than guessed.
    """
    if fmtedy.strip().upper() not in ("FR", "FREEFORMATTED"):
        raise HorizontalTurbulenceV87Error(
            "v87 supports only source-equivalent freeformatted FileDy")
    if (isinstance(nmax, bool) or type(nmax) is not int or nmax < 1 or
            isinstance(mmax, bool) or type(mmax) is not int or mmax < 1):
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy dimensions must be positive integers")
    payload = content if isinstance(content, bytes) else str(content).encode("utf-8")
    text = payload.decode("utf-8")
    # ``edyfil`` calls ``skipstarlines`` exactly once before the first
    # free-formatted block.  That helper tests column one (not lstrip) and
    # skips only the consecutive leading records.  Preserve that precise
    # contract: a later star record is data to the Fortran list-directed read
    # and therefore makes the file invalid rather than becoming a permissive
    # BRAIDS-only comment extension.
    lines = text.splitlines()
    first_data = 0
    while first_data < len(lines) and lines[first_data].startswith("*"):
        first_data += 1
    lines = lines[first_data:]
    tokens = " ".join(lines).replace(",", " ").split()
    expected = 2*nmax*mmax
    if len(tokens) != expected:
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy must contain exactly one vicuv and one dicuv block")
    try:
        values = np.array(
            [float(token.replace("D", "E").replace("d", "e"))
             for token in tokens], dtype=np.float64)
    except ValueError as error:
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy contains a nonnumeric value") from error
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy coefficients must be finite and nonnegative")
    mapping = np.asarray(structured_flat_index_by_cell)
    if mapping.ndim != 1 or mapping.dtype.kind not in "iu":
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy mapping must be a one-dimensional integer array")
    mapping = np.ascontiguousarray(mapping, dtype=np.int64)
    count = nmax*mmax
    if np.any(mapping < 0) or np.any(mapping >= count):
        raise HorizontalTurbulenceV87Error(
            "v87 FileDy mapping index is outside the structured grid")
    return _wrap_background(
        mode="filedy_freeformatted", vicuv=values[:count][mapping],
        dicuv=values[count:][mapping], cell_count=mapping.size,
        source_name=source_name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        mapping_sha256=_array_digest(mapping))


def background_document_v87(value: TurbulenceBackgroundV87) -> dict:
    if (not isinstance(value, TurbulenceBackgroundV87) or
            value.fingerprint != _fingerprint(_background_payload(value))):
        raise HorizontalTurbulenceV87Error(
            "v87 background field is stale or mutated")
    return {**_background_payload(value), "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class HorizontalTurbulenceExecutionV87:
    base_execution: MultifractionSuspendedExecutionV78
    background: TurbulenceBackgroundV87
    tpsnumber_by_fraction: np.ndarray
    hles_enabled: bool
    elder_enabled: bool
    alpha: float
    nd: int
    sigmat: float
    flp: float
    reltim_s: float
    vicmol_m2_s: float
    dicmol_m2_s: float
    von_karman: float
    gravity_m_s2: float
    semantic_id: str
    fingerprint: str

    @property
    def role(self) -> str:
        return self.base_execution.role

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def gamma(self) -> float:
        return 0.844*math.sqrt(
            (1.0-self.alpha**-2)/(2.0*float(self.nd)))


NON_2DH_DISPOSITIONS = {
    "Tkemod": "inactive_and_rejected_for_KMAX_1; vertical turbulence only",
    "Vicoww": "must_be_zero_for_KMAX_1; vertical eddy viscosity only",
    "Dicoww": "must_be_zero_for_KMAX_1; vertical eddy diffusivity only",
    "Xlo": "must_be_zero_for_KMAX_1; vertical turbulence length scale",
    "diffusionCal": (
        "rejected; D-Flow_FM provenance, not locked classic FLOW 2DH detvic"),
    "diffusionScaling": (
        "rejected; D-Flow_FM provenance, not locked classic FLOW 2DH detvic"),
}


def _execution_payload(value: HorizontalTurbulenceExecutionV87) -> dict:
    from .multifraction_suspended_lifecycle_v78 import execution_document_v78
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "integrator": INTEGRATOR,
        "base_v78_execution": execution_document_v78(value.base_execution),
        "background": background_document_v87(value.background),
        "tpsnumber_by_fraction": value.tpsnumber_by_fraction.tolist(),
        "hles_enabled": value.hles_enabled,
        "elder_enabled": value.elder_enabled,
        "alpha": value.alpha,
        "nd": value.nd,
        "sigmat": value.sigmat,
        "flp": value.flp,
        "reltim_s": value.reltim_s,
        "gamma": value.gamma,
        "vicmol_m2_s": value.vicmol_m2_s,
        "dicmol_m2_s": value.dicmol_m2_s,
        "von_karman": value.von_karman,
        "gravity_m_s2": value.gravity_m_s2,
        "kmax": 1,
        "provider": PROVIDER_ID,
        "provider_input": "accepted_lpfluc_protke_output",
        "call_order": CALL_ORDER,
        "dynamic_field_lag_accepted_updates": 1,
        "horizontal_schmidt": 0.7,
        "per_fraction_vertical_schmidt_horizontal_effect": False,
        "non_2dh_dispositions": NON_2DH_DISPOSITIONS,
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "locked_delft3d_source_read_only": True,
    }


def resolve_horizontal_turbulence_execution_v87(
    base_execution: MultifractionSuspendedExecutionV78, *,
    background: TurbulenceBackgroundV87,
    tpsnumber_by_fraction: object | None = None,
    hles_enabled: bool = True, elder_enabled: bool = True,
    alpha: float = 5.0/3.0, nd: int = 2, sigmat: float = 0.7,
    flp: float = 1.0/3.0, reltim_s: float = -1.0,
    vicmol_m2_s: float = 0.0, dicmol_m2_s: float = 0.0,
    von_karman: float = 0.41, gravity_m_s2: float = 9.81,
    kmax: int = 1, tkemod: str | None = None,
    vicoww_m2_s: float = 0.0, dicoww_m2_s: float = 0.0,
    xlo_m: float = 0.0, diffusion_cal: object | None = None,
    diffusion_scaling: object | None = None,
) -> HorizontalTurbulenceExecutionV87:
    """Resolve the bounded KMAX=1 closure and reject 3D/FM controls."""
    from .multifraction_suspended_lifecycle_v78 import execution_document_v78
    execution_document_v78(base_execution)
    if base_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA):
        raise HorizontalTurbulenceV87Error("v87 base role differs")
    if base_execution.base_execution.profile.horizontal_diffusivity_m2_s != 0.0:
        raise HorizontalTurbulenceV87Error(
            "v87 requires a zero-dicuv v78 base to prevent duplicate diffusion")
    document = background_document_v87(background)
    if len(document["dicuv_m2_s"]) != base_execution.cell_count:
        raise HorizontalTurbulenceV87Error(
            "v87 background cell count differs from the lifecycle")
    if type(kmax) is not int or kmax != 1:
        raise HorizontalTurbulenceV87Error(
            "v87 admits KMAX=1 only; 3D turbulence controls are unsupported")
    if tkemod not in (None, "", "inactive_kmax1"):
        raise HorizontalTurbulenceV87Error(
            "v87 Tkemod is a KMAX>1 control and is rejected")
    for raw, name in ((vicoww_m2_s, "Vicoww"),
                      (dicoww_m2_s, "Dicoww"), (xlo_m, "Xlo")):
        if _finite(raw, name) != 0.0:
            raise HorizontalTurbulenceV87Error(
                f"v87 {name} is KMAX>1-only and must remain zero")
    if diffusion_cal is not None or diffusion_scaling is not None:
        raise HorizontalTurbulenceV87Error(
            "v87 diffusionCal/diffusionScaling are FM-only and rejected")
    if type(hles_enabled) is not bool or type(elder_enabled) is not bool:
        raise HorizontalTurbulenceV87Error(
            "v87 HLES/Elder switches must be Boolean")
    if elder_enabled and not hles_enabled:
        raise HorizontalTurbulenceV87Error(
            "v87 Elder belongs to the active HLES detvic branch")
    if isinstance(nd, bool) or type(nd) is not int or nd < 1:
        raise HorizontalTurbulenceV87Error("v87 Hturnd must be positive")
    alpha_f = _finite(alpha, "Htural", positive=True)
    if alpha_f <= 1.0:
        raise HorizontalTurbulenceV87Error("v87 Htural must exceed one")
    reltim = _finite(reltim_s, "Hturrt")
    if hles_enabled and not (reltim > 0.0 or reltim == -1.0):
        raise HorizontalTurbulenceV87Error(
            "v87 Hturrt must be positive or the locked -1 sentinel")
    tps = (np.ones(base_execution.fraction_count, dtype=np.float64)
           if tpsnumber_by_fraction is None else tpsnumber_by_fraction)
    tps_a = _frozen_array(
        tps, (base_execution.fraction_count,), "tpsnumber",
        nonnegative=True)
    if np.any(tps_a <= 0.0):
        raise HorizontalTurbulenceV87Error(
            "v87 vertical Schmidt provenance values must be positive")
    placeholder = HorizontalTurbulenceExecutionV87(
        base_execution=base_execution, background=background,
        tpsnumber_by_fraction=tps_a, hles_enabled=hles_enabled,
        elder_enabled=elder_enabled, alpha=alpha_f, nd=nd,
        sigmat=_finite(sigmat, "Hturst", positive=True),
        flp=_finite(flp, "Hturlp", positive=True), reltim_s=reltim,
        vicmol_m2_s=_finite(vicmol_m2_s, "vicmol", nonnegative=True),
        dicmol_m2_s=_finite(dicmol_m2_s, "Hturdm", nonnegative=True),
        von_karman=_finite(von_karman, "von Karman", positive=True),
        gravity_m_s2=_finite(gravity_m_s2, "gravity", positive=True),
        semantic_id=SEMANTIC_ID, fingerprint="")
    return HorizontalTurbulenceExecutionV87(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v87(value: HorizontalTurbulenceExecutionV87) -> dict:
    if not isinstance(value, HorizontalTurbulenceExecutionV87):
        raise HorizontalTurbulenceV87Error("v87 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise HorizontalTurbulenceV87Error(
            "v87 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class DetvicFieldV87:
    hles_vicuv_m2_s: np.ndarray
    elder_vicuv_m2_s: np.ndarray
    dynamic_vicuv_m2_s: np.ndarray
    dynamic_dicuv_m2_s: np.ndarray


def evaluate_detvic_field_v87(
    execution: HorizontalTurbulenceExecutionV87, *,
    depth_m: object, mean_velocity_m_s: object, chezy_mhalf_s: object,
    ptke_s2: object, active_cell: object,
) -> DetvicFieldV87:
    """Evaluate locked ``detvic`` algebra on accepted cell-centred inputs."""
    execution_document_v87(execution)
    nc = execution.cell_count
    depth = _frozen_array(depth_m, (nc,), "closure depth", nonnegative=True)
    velocity = _frozen_array(
        mean_velocity_m_s, (2, nc), "closure mean velocity")
    chezy = _frozen_array(
        chezy_mhalf_s, (nc,), "closure Chezy", nonnegative=True)
    ptke = _frozen_array(ptke_s2, (nc,), "closure ptke", nonnegative=True)
    active = _frozen_array(
        active_cell, (nc,), "closure active-cell mask", integer_mask=True)
    hles = np.zeros(nc, dtype=np.float64)
    elder = np.zeros(nc, dtype=np.float64)
    dynamic_vic = np.zeros(nc, dtype=np.float64)
    dynamic_dic = np.zeros(nc, dtype=np.float64)
    if execution.hles_enabled:
        wet = active != 0
        area = execution.base_execution.base_execution.cell_area_m2
        utot = np.hypot(velocity[0], velocity[1])
        safe_chezy = np.maximum(1.0, chezy)
        safe_depth = np.maximum(
            execution.base_execution.base_execution.dry_tolerance_m, depth)
        b = (0.75*execution.gravity_m_s2*utot/
             (safe_depth*safe_chezy*safe_chezy))
        ks = math.pi*execution.flp/np.sqrt(area)
        hles[wet] = (
            np.sqrt(
                execution.gamma**2*execution.sigmat**2*ptke[wet]
                + b[wet]*b[wet])-b[wet])/(ks[wet]*ks[wet])
        if execution.elder_enabled:
            ustar = utot*np.sqrt(execution.gravity_m_s2)/safe_chezy
            elder[wet] = (
                execution.von_karman*ustar[wet]*safe_depth[wet]/6.0)
        dynamic_vic[wet] = (
            hles[wet]+elder[wet]+execution.vicmol_m2_s)
        # detvic copies the turbulence/Elder viscosity before adding the
        # distinct molecular diffusivity.
        dynamic_dic[wet] = (
            hles[wet]+elder[wet]+execution.dicmol_m2_s)
    for array in (hles, elder, dynamic_vic, dynamic_dic):
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise HorizontalTurbulenceV87Error(
                "v87 detvic produced an invalid coefficient")
        array.setflags(write=False)
    return DetvicFieldV87(hles, elder, dynamic_vic, dynamic_dic)


@dataclass(frozen=True)
class HorizontalTurbulenceStateV87:
    execution_fingerprint: str
    base_state: MultifractionSuspendedStateV78
    accepted_dynamic_vicuv_m2_s: np.ndarray
    accepted_dynamic_dicuv_m2_s: np.ndarray
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.bed_mass_kg_m2


def _state_token(value: HorizontalTurbulenceStateV87) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "accepted_dynamic_vicuv_sha256":
            _array_digest(value.accepted_dynamic_vicuv_m2_s),
        "accepted_dynamic_dicuv_sha256":
            _array_digest(value.accepted_dynamic_dicuv_m2_s),
    })


def _wrap_state(
    execution: HorizontalTurbulenceExecutionV87,
    base_state: MultifractionSuspendedStateV78,
    dynamic_vicuv: object, dynamic_dicuv: object,
) -> HorizontalTurbulenceStateV87:
    execution_document_v87(execution)
    validate_multifraction_suspended_state_v78(
        base_state, execution.base_execution)
    vic = _frozen_array(
        dynamic_vicuv, (execution.cell_count,), "accepted dynamic vicuv",
        nonnegative=True)
    dic = _frozen_array(
        dynamic_dicuv, (execution.cell_count,), "accepted dynamic dicuv",
        nonnegative=True)
    placeholder = HorizontalTurbulenceStateV87(
        execution.fingerprint, base_state, vic, dic, "")
    return HorizontalTurbulenceStateV87(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_horizontal_turbulence_state_v87(
    execution: HorizontalTurbulenceExecutionV87, **state_fields,
) -> HorizontalTurbulenceStateV87:
    """Create a cold state; its first update uses background diffusion only."""
    try:
        base = resolve_multifraction_suspended_state_v78(
            execution.base_execution, **state_fields)
    except (TypeError, ValueError) as error:
        raise HorizontalTurbulenceV87Error(
            "v87 initial suspended state is invalid") from error
    zeros = np.zeros(execution.cell_count, dtype=np.float64)
    return _wrap_state(execution, base, zeros, zeros)


def validate_horizontal_turbulence_state_v87(
    state: HorizontalTurbulenceStateV87,
    execution: HorizontalTurbulenceExecutionV87,
) -> None:
    if (not isinstance(state, HorizontalTurbulenceStateV87) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise HorizontalTurbulenceV87Error(
            "v87 state is stale, mutated, or execution-mismatched")
    validate_multifraction_suspended_state_v78(
        state.base_state, execution.base_execution)


def _closure_inputs(
    execution: HorizontalTurbulenceExecutionV87, *, depth_m: object,
    mean_velocity_m_s: object, chezy_mhalf_s: object, ptke_s2: object,
    active_cell: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nc = execution.cell_count
    return (
        _frozen_array(depth_m, (nc,), "closure depth", nonnegative=True),
        _frozen_array(mean_velocity_m_s, (2, nc), "closure mean velocity"),
        _frozen_array(chezy_mhalf_s, (nc,), "closure Chezy", nonnegative=True),
        _frozen_array(ptke_s2, (nc,), "closure ptke", nonnegative=True),
        _frozen_array(active_cell, (nc,), "closure active cell",
                      integer_mask=True),
    )


def make_detvic_input_receipt_v87(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87, *,
    accepted_step_receipt: Mapping[str, object], depth_m: object,
    mean_velocity_m_s: object, chezy_mhalf_s: object, ptke_s2: object,
    active_cell: object,
) -> dict:
    """Bind accepted ``lpfluc/protke`` inputs to one hydraulic step."""
    validate_horizontal_turbulence_state_v87(parent, execution)
    arrays = _closure_inputs(
        execution, depth_m=depth_m, mean_velocity_m_s=mean_velocity_m_s,
        chezy_mhalf_s=chezy_mhalf_s, ptke_s2=ptke_s2,
        active_cell=active_cell)
    body = {
        "schema": SCHEMA,
        "scope_id": INPUT_RECEIPT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "parent_state_sha256": parent.state_sha256,
        "parent_accepted_updates": parent.accepted_updates,
        "parent_time_s": parent.time_s,
        "accepted_step_receipt_sha256": _fingerprint(accepted_step_receipt),
        "depth_sha256": _array_digest(arrays[0]),
        "mean_velocity_sha256": _array_digest(arrays[1]),
        "chezy_sha256": _array_digest(arrays[2]),
        "ptke_sha256": _array_digest(arrays[3]),
        "active_cell_sha256": _array_digest(arrays[4]),
        "provider": PROVIDER_ID,
        "call_order": CALL_ORDER,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def _validate_detvic_receipt(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    receipt: Mapping[str, object], accepted_step_receipt: Mapping[str, object],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    keys = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "parent_state_sha256", "parent_accepted_updates", "parent_time_s",
        "accepted_step_receipt_sha256", "depth_sha256",
        "mean_velocity_sha256", "chezy_sha256", "ptke_sha256",
        "active_cell_sha256", "provider", "call_order", "fingerprint",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != keys:
        raise HorizontalTurbulenceV87Error(
            "v87 detvic input receipt inventory differs")
    body = {key: receipt[key] for key in receipt if key != "fingerprint"}
    expected = (
        _array_digest(arrays[0]), _array_digest(arrays[1]),
        _array_digest(arrays[2]), _array_digest(arrays[3]),
        _array_digest(arrays[4]))
    if (receipt["fingerprint"] != _fingerprint(body) or
            receipt["schema"] != SCHEMA or
            receipt["scope_id"] != INPUT_RECEIPT_SCOPE_ID or
            receipt["semantic_id"] != SEMANTIC_ID or
            receipt["execution_fingerprint"] != execution.fingerprint or
            receipt["parent_state_sha256"] != parent.state_sha256 or
            receipt["parent_accepted_updates"] != parent.accepted_updates or
            receipt["parent_time_s"] != parent.time_s or
            receipt["accepted_step_receipt_sha256"] !=
                _fingerprint(accepted_step_receipt) or
            receipt["provider"] != PROVIDER_ID or
            receipt["call_order"] != CALL_ORDER or
            tuple(receipt[key] for key in (
                "depth_sha256", "mean_velocity_sha256", "chezy_sha256",
                "ptke_sha256", "active_cell_sha256")) != expected):
        raise HorizontalTurbulenceV87Error(
            "v87 detvic input receipt differs from accepted inputs")


def _v82_context(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
) -> tuple[SuspendedSupportExecutionV82, SuspendedSupportStateV82,
           np.ndarray, np.ndarray]:
    used_vic = np.asarray(
        execution.background.vicuv_m2_s+parent.accepted_dynamic_vicuv_m2_s)
    used_dic = np.asarray(
        execution.background.dicuv_m2_s+parent.accepted_dynamic_dicuv_m2_s)
    used_vic.setflags(write=False)
    used_dic.setflags(write=False)
    v82_execution = resolve_suspended_support_execution_v82(
        execution.base_execution, dicuv_cell_m2_s=used_dic,
        tpsnumber_by_fraction=execution.tpsnumber_by_fraction)
    v82_parent = _wrap_state_v82(v82_execution, parent.base_state)
    return v82_execution, v82_parent, used_vic, used_dic


@dataclass(frozen=True)
class HorizontalTurbulenceTrialV87:
    execution_fingerprint: str
    parent_state_sha256: str
    v82_execution_fingerprint: str
    base_trial: SuspendedSupportTrialV82
    candidate: HorizontalTurbulenceStateV87
    used_total_vicuv_m2_s: np.ndarray
    used_total_dicuv_m2_s: np.ndarray
    candidate_hles_vicuv_m2_s: np.ndarray
    candidate_elder_vicuv_m2_s: np.ndarray
    detvic_input_receipt_fingerprint: str
    trial_sha256: str


def _trial_token(value: HorizontalTurbulenceTrialV87) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "v82_execution_fingerprint": value.v82_execution_fingerprint,
        "base_trial_sha256": value.base_trial.trial_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "used_total_vicuv_sha256": _array_digest(value.used_total_vicuv_m2_s),
        "used_total_dicuv_sha256": _array_digest(value.used_total_dicuv_m2_s),
        "candidate_hles_sha256":
            _array_digest(value.candidate_hles_vicuv_m2_s),
        "candidate_elder_sha256":
            _array_digest(value.candidate_elder_vicuv_m2_s),
        "detvic_input_receipt_fingerprint":
            value.detvic_input_receipt_fingerprint,
    })


def _wrap_trial(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    v82_execution: SuspendedSupportExecutionV82,
    base_trial: SuspendedSupportTrialV82,
    used_vic: object, used_dic: object, field: DetvicFieldV87,
    receipt_fingerprint: str,
) -> HorizontalTurbulenceTrialV87:
    candidate = _wrap_state(
        execution, base_trial.candidate.base_state,
        field.dynamic_vicuv_m2_s, field.dynamic_dicuv_m2_s)
    arrays = (
        _frozen_array(used_vic, (execution.cell_count,), "used total vicuv",
                      nonnegative=True),
        _frozen_array(used_dic, (execution.cell_count,), "used total dicuv",
                      nonnegative=True),
        _frozen_array(field.hles_vicuv_m2_s, (execution.cell_count,),
                      "candidate HLES vicuv", nonnegative=True),
        _frozen_array(field.elder_vicuv_m2_s, (execution.cell_count,),
                      "candidate Elder vicuv", nonnegative=True),
    )
    if not isinstance(receipt_fingerprint, str) or not _SHA256.fullmatch(
            receipt_fingerprint):
        raise HorizontalTurbulenceV87Error(
            "v87 detvic receipt fingerprint is invalid")
    placeholder = HorizontalTurbulenceTrialV87(
        execution.fingerprint, parent.state_sha256,
        v82_execution.fingerprint, base_trial, candidate,
        arrays[0], arrays[1], arrays[2], arrays[3], receipt_fingerprint, "")
    return HorizontalTurbulenceTrialV87(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def begin_horizontal_turbulence_cpu_trial_v87(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87, *,
    accepted_step_receipt: Mapping[str, object],
    detvic_input_receipt: Mapping[str, object],
    closure_depth_m: object, closure_mean_velocity_m_s: object,
    closure_chezy_mhalf_s: object, closure_ptke_s2: object,
    closure_active_cell: object, cancelled: bool = False, **inputs,
) -> HorizontalTurbulenceTrialV87:
    if execution.role != ROLE_CPU_ORACLE:
        raise HorizontalTurbulenceV87Error(
            "v87 CPU trial requires isolated_cpu_oracle role")
    validate_horizontal_turbulence_state_v87(parent, execution)
    if cancelled:
        raise HorizontalTurbulenceV87Error(
            "v87 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_UPDATES:
        raise HorizontalTurbulenceV87Error(
            "v87 accepted update limit was reached")
    arrays = _closure_inputs(
        execution, depth_m=closure_depth_m,
        mean_velocity_m_s=closure_mean_velocity_m_s,
        chezy_mhalf_s=closure_chezy_mhalf_s, ptke_s2=closure_ptke_s2,
        active_cell=closure_active_cell)
    _validate_detvic_receipt(
        execution, parent, detvic_input_receipt, accepted_step_receipt, arrays)
    v82_execution, v82_parent, used_vic, used_dic = _v82_context(
        execution, parent)
    try:
        base_trial = begin_suspended_support_cpu_trial_v82(
            v82_execution, v82_parent,
            accepted_step_receipt=accepted_step_receipt, **inputs)
    except (TypeError, ValueError) as error:
        raise HorizontalTurbulenceV87Error(
            "v87 accepted v82 CPU lifecycle rejected the trial") from error
    field = evaluate_detvic_field_v87(
        execution, depth_m=arrays[0], mean_velocity_m_s=arrays[1],
        chezy_mhalf_s=arrays[2], ptke_s2=arrays[3], active_cell=arrays[4])
    return _wrap_trial(
        execution, parent, v82_execution, base_trial, used_vic, used_dic,
        field, str(detvic_input_receipt["fingerprint"]))


def _validate_trial(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    trial: HorizontalTurbulenceTrialV87,
) -> tuple[SuspendedSupportExecutionV82, SuspendedSupportStateV82]:
    validate_horizontal_turbulence_state_v87(parent, execution)
    if (not isinstance(trial, HorizontalTurbulenceTrialV87) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise HorizontalTurbulenceV87Error(
            "v87 trial is stale, mutated, or belongs to another parent")
    validate_horizontal_turbulence_state_v87(trial.candidate, execution)
    v82_execution, v82_parent, used_vic, used_dic = _v82_context(
        execution, parent)
    if (trial.v82_execution_fingerprint != v82_execution.fingerprint or
            not np.array_equal(trial.used_total_vicuv_m2_s, used_vic) or
            not np.array_equal(trial.used_total_dicuv_m2_s, used_dic)):
        raise HorizontalTurbulenceV87Error(
            "v87 trial used a different accepted closure field")
    return v82_execution, v82_parent


def commit_horizontal_turbulence_trial_v87(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    trial: HorizontalTurbulenceTrialV87,
) -> HorizontalTurbulenceStateV87:
    v82_execution, v82_parent = _validate_trial(execution, parent, trial)
    committed = commit_suspended_support_trial_v82(
        v82_execution, v82_parent, trial.base_trial)
    if committed.base_state.state_sha256 != trial.candidate.base_state.state_sha256:
        raise HorizontalTurbulenceV87Error(
            "v87 committed v82 candidate differs")
    return trial.candidate


def rollback_horizontal_turbulence_trial_v87(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    trial: HorizontalTurbulenceTrialV87,
) -> HorizontalTurbulenceStateV87:
    v82_execution, v82_parent = _validate_trial(execution, parent, trial)
    result = rollback_suspended_support_trial_v82(
        v82_execution, v82_parent, trial.base_trial)
    if result is not v82_parent:
        raise HorizontalTurbulenceV87Error("v87 v82 rollback differs")
    return parent


def cancel_horizontal_turbulence_trial_v87(
    execution: HorizontalTurbulenceExecutionV87,
    parent: HorizontalTurbulenceStateV87,
    trial: HorizontalTurbulenceTrialV87,
) -> HorizontalTurbulenceStateV87:
    return rollback_horizontal_turbulence_trial_v87(execution, parent, trial)


def horizontal_turbulence_checkpoint_v87(
    execution: HorizontalTurbulenceExecutionV87,
    state: HorizontalTurbulenceStateV87,
) -> dict:
    validate_horizontal_turbulence_state_v87(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "call_order": CALL_ORDER,
        "base_v78_checkpoint": multifraction_checkpoint_v78(
            execution.base_execution, state.base_state),
        "accepted_dynamic_vicuv_m2_s":
            state.accepted_dynamic_vicuv_m2_s.tolist(),
        "accepted_dynamic_dicuv_m2_s":
            state.accepted_dynamic_dicuv_m2_s.tolist(),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_horizontal_turbulence_checkpoint_v87(
    execution: HorizontalTurbulenceExecutionV87,
    document: Mapping[str, object],
) -> HorizontalTurbulenceStateV87:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "integrator", "call_order", "base_v78_checkpoint",
        "accepted_dynamic_vicuv_m2_s", "accepted_dynamic_dicuv_m2_s",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise HorizontalTurbulenceV87Error(
            "v87 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR or
            document["call_order"] != CALL_ORDER):
        raise HorizontalTurbulenceV87Error(
            "v87 checkpoint identity differs")
    try:
        base = load_multifraction_checkpoint_v78(
            execution.base_execution, document["base_v78_checkpoint"])
        state = _wrap_state(
            execution, base, document["accepted_dynamic_vicuv_m2_s"],
            document["accepted_dynamic_dicuv_m2_s"])
    except (TypeError, ValueError) as error:
        raise HorizontalTurbulenceV87Error(
            "v87 checkpoint payload is invalid") from error
    if state.state_sha256 != document["state_sha256"]:
        raise HorizontalTurbulenceV87Error(
            "v87 checkpoint state digest differs")
    return state


def method_identity_v87() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on": [
            "braids-q4-flow2dh-suspended-support-closures-v82-v1",
            "braids-q4-multifraction-suspended-lifecycle-v78-v1",
        ],
        "integrator": INTEGRATOR,
        "provider": PROVIDER_ID,
        "background_modes": list(BACKGROUND_MODES),
        "call_order": CALL_ORDER,
        "dynamic_field_lag_accepted_updates": 1,
        "kmax": 1,
        "horizontal_schmidt": 0.7,
        "per_fraction_vertical_schmidt_horizontal_effect": False,
        "non_2dh_dispositions": NON_2DH_DISPOSITIONS,
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "transaction": (
            "immutable_commit_rollback_cancel_exact_cold_restart_v1"),
        "not_claimed": [
            "lpfluc/protke generation", "KMAX>1 vertical turbulence",
            "FM diffusionCal or diffusionScaling",
            "hydraulic momentum consumption of exposed vicuv",
            "waves", "general AMR", "long-run equivalence",
        ],
        "delft3d_source_modified": False,
    }


FROZEN_PREDECESSOR_SHA256 = {
    "braids_core/sediment/suspended_support_closures_v82.py":
        "aad9844d3ee66a7c98f2995e9f187229dbf9a3d62498056261aba113cde8fd24",
    "braids_gpu/suspended_support_closures_v82.py":
        "41656dfcf196a1018c8e8cf1e43adbf4276f002bba8b020c32ef98e6586f5271",
    "braids_core/sediment/multifraction_suspended_lifecycle_v78.py":
        "eaba4861d38369eaf740803c43a722e10a0ee70bb9a2e1dcf56bddba2a7e10dd",
    "braids_gpu/multifraction_suspended_lifecycle_v78.py":
        "e4fdd4aa5244d0ceff7f7b5479cdfbe4bc2cd9c5344f731cecf5351180facc6b",
    "braids_gpu/flow2dh_boundary_diffusion.cu":
        "b61c2ffe2c4a4bc411ffc4c0ce0fc07ca9edd1d86a23d622d903cb5aa15e71bf",
}


def verify_frozen_predecessors_v87(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in FROZEN_PREDECESSOR_SHA256.items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise HorizontalTurbulenceV87Error(
                "v87 frozen predecessor hash differs: " + relative)
        result[relative] = actual
    return result


def verify_sources_v87(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = (
        "braids_core/sediment/horizontal_turbulence_lifecycle_v87.py",
        "braids_gpu/horizontal_turbulence_lifecycle_v87.py",
        "braids_gpu/horizontal_turbulence_lifecycle_v87.cu",
        "tests/test_q4_horizontal_turbulence_lifecycle_v87.py",
        "tests/test_q4_horizontal_turbulence_lifecycle_cuda_v87.py",
    )
    result = {}
    for relative in paths:
        path = root/relative
        if not path.is_file():
            raise HorizontalTurbulenceV87Error(
                "v87 additive source is absent: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


__all__ = [
    "BACKGROUND_MODES", "CALL_ORDER", "CHECKPOINT_SCOPE_ID", "INTEGRATOR",
    "INPUT_RECEIPT_SCOPE_ID", "MAX_UPDATES", "NON_2DH_DISPOSITIONS",
    "PROVIDER_ID", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID", "DetvicFieldV87",
    "HorizontalTurbulenceExecutionV87", "HorizontalTurbulenceStateV87",
    "HorizontalTurbulenceTrialV87", "HorizontalTurbulenceV87Error",
    "TurbulenceBackgroundV87", "background_document_v87",
    "begin_horizontal_turbulence_cpu_trial_v87",
    "cancel_horizontal_turbulence_trial_v87",
    "commit_horizontal_turbulence_trial_v87",
    "constant_turbulence_background_v87", "evaluate_detvic_field_v87",
    "execution_document_v87", "filedy_turbulence_background_v87",
    "horizontal_turbulence_checkpoint_v87",
    "load_horizontal_turbulence_checkpoint_v87",
    "make_detvic_input_receipt_v87", "method_identity_v87",
    "resolve_horizontal_turbulence_execution_v87",
    "resolve_horizontal_turbulence_state_v87",
    "rollback_horizontal_turbulence_trial_v87",
    "spatial_turbulence_background_v87",
    "validate_horizontal_turbulence_state_v87",
    "verify_frozen_predecessors_v87", "verify_sources_v87",
]
