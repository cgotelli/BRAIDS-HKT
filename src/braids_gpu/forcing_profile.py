"""Authenticated Delft3D C/M/W roughness and sediment-shear selection.

The native deck historically names its per-cell roughness column
``manning_cell``. That storage name is not a physical type. This module binds
the bytes to a canonical C/n/ks identity and exposes a distinct
``roughness_parameter`` alias. Consumers must never reinterpret those bytes as
Manning merely because of the legacy storage name.

Unsupported dependent physics fails closed before any CUDA module is created;
no formula or backend substitution is permitted.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

import numpy as np


SCHEMA = 2
BINDING_FIELD = "delft_forcing_binding"
ROUGHNESS = {
    "chezy": dict(native=1, method="roughness.chezy",
                  parameter="chezy_C", units="m^0.5/s"),
    "manning": dict(native=2, method="roughness.manning",
                    parameter="manning_n", units="s/m^(1/3)"),
    "white_colebrook": dict(native=3, method="roughness.white_colebrook",
                            parameter="nikuradse_ks", units="m"),
}
SHEAR = {
    "total": (0, "shear.total_roughness"),
    "soulsby_clarke_skin": (1, "shear.soulsby_clarke_skin"),
}
_SKIN_DEFAULTS = dict(
    kinematic_viscosity_m2_s=1.0e-6,
    sand_roughness_m=0.003,
    silt_roughness_m=0.0003,
    critical_mud_fraction_low=0.2,
    critical_mud_fraction_high=0.8,
    fixed_fluff_fraction=0.0,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _finite(value, *, positive=False, nonnegative=False):
    return (not isinstance(value, (bool, np.bool_)) and
            isinstance(value, (int, float, np.integer, np.floating)) and
            np.isfinite(value) and (not positive or value > 0.0) and
            (not nonnegative or value >= 0.0))


def _parameter_array(value):
    array = np.asarray(value)
    _require(array.ndim == 1 and array.dtype == np.float64 and
             array.flags.c_contiguous and np.isfinite(array).all(),
             "Roughness parameter must be a contiguous finite FP64 vector")
    return array


def _array_identity(value):
    array = _parameter_array(value)
    return dict(shape=[int(item) for item in array.shape],
                dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def forcing_binding_from_metadata(metadata: Mapping[str, object],
                                  roughness_parameter, *,
                                  native_skin_controls: Mapping[str, object]) -> dict:
    """Build a canonical declaration including the actual parameter bytes."""
    _require(isinstance(metadata, Mapping),
             "Forcing case metadata must be a mapping")
    rough = metadata.get("hydraulic_roughness")
    shear = metadata.get("sediment_shear")
    _require(isinstance(rough, Mapping) and isinstance(shear, Mapping),
             "Case metadata lacks typed roughness/shear declarations")
    mode, shear_mode = rough.get("mode"), shear.get("mode")
    _require(mode in ROUGHNESS and shear_mode in SHEAR,
             "Unknown Delft3D roughness or sediment-shear declaration")
    specification = ROUGHNESS[mode]
    _require(rough.get("parameter_units") == specification["units"],
             "Roughness parameter units differ from the selected method")
    default_parameter = rough.get("default_parameter")
    _require(_finite(default_parameter, positive=(mode != "manning"),
                    nonnegative=(mode == "manning")),
             "Default roughness parameter is invalid")
    parameter = _parameter_array(roughness_parameter)
    _require(parameter.size > 0 and
             (np.all(parameter > 0.0) if mode != "manning" else
              np.all(parameter >= 0.0)),
             "Per-cell roughness parameter is invalid for its selected method")
    minimum, maximum = (rough.get("minimum_parameter"),
                        rough.get("maximum_parameter"))
    _require(_finite(minimum, nonnegative=True) and
             _finite(maximum, nonnegative=True) and
             float(minimum) == float(np.min(parameter)) and
             float(maximum) == float(np.max(parameter)),
             "Metadata/per-cell roughness range differs")
    chzk2d = rough.get("white_colebrook_chzk2d")
    _require(type(chzk2d) is bool and
             (not chzk2d or mode == "white_colebrook"),
             "ChzK2D requires White-Colebrook roughness")
    method_id = ("roughness.white_colebrook_waqua" if chzk2d else
                 specification["method"])
    work = rough.get("minimum_work_depth_m")
    _require(_finite(work, nonnegative=True) and
             (mode == "manning" or float(work) > 0.0),
             "Roughness minimum work depth is invalid")
    manning = metadata.get("manning", {})
    _require(isinstance(manning, Mapping), "Manning metadata must be a mapping")
    friction_floor = manning.get("friction_chezy_minimum_depth_m", 0.0)
    _require(_finite(friction_floor, nonnegative=True),
             "Hydraulic friction work-depth floor is invalid")
    _require(mode != "manning" or float(work) <= float(friction_floor),
             "Manning roughness work depth exceeds the authenticated friction floor")

    _require(isinstance(native_skin_controls, Mapping) and
             set(native_skin_controls) == set(_SKIN_DEFAULTS),
             "Native skin-control inventory differs")
    skin = {key: float(native_skin_controls[key]) for key in _SKIN_DEFAULTS}
    if shear_mode == "soulsby_clarke_skin":
        _require(all(shear.get(key) == skin[key] for key in skin),
                 "Metadata/native Soulsby-Clarke controls differ")
        _require(all(_finite(skin[key], positive=True) for key in
                     ("kinematic_viscosity_m2_s", "sand_roughness_m",
                      "silt_roughness_m")) and
                 all(_finite(skin[key], nonnegative=True) for key in
                     ("critical_mud_fraction_low",
                      "critical_mud_fraction_high", "fixed_fluff_fraction")) and
                 0.0 <= skin["critical_mud_fraction_low"] <
                 skin["critical_mud_fraction_high"] <= 1.0 and
                 skin["fixed_fluff_fraction"] <= 1.0,
                 "Soulsby/Clarke skin-shear controls are invalid")
    body = dict(
        schema=SCHEMA, roughness_method_id=method_id,
        hydraulic_roughness_mode=mode,
        native_hydraulic_roughness_mode=int(specification["native"]),
        white_colebrook_chzk2d=chzk2d,
        roughness_parameter_name=specification["parameter"],
        roughness_parameter_units=specification["units"],
        roughness_storage_field="manning_cell",
        roughness_default_parameter=float(default_parameter),
        roughness_parameter_minimum=float(minimum),
        roughness_parameter_maximum=float(maximum),
        roughness_parameter_array=_array_identity(parameter),
        roughness_minimum_work_depth_m=float(work),
        friction_minimum_work_depth_m=float(friction_floor),
        effective_minimum_work_depth_m=float(max(work, friction_floor)),
        sediment_shear_method_id=SHEAR[shear_mode][1],
        sediment_shear_mode=shear_mode,
        native_sediment_shear_mode=SHEAR[shear_mode][0],
        skin_shear_controls={key: float(value) for key, value in skin.items()},
        skin_shear_controls_active=(shear_mode == "soulsby_clarke_skin"),
        hydraulic_friction_unchanged_by_skin=True,
    )
    return {**body, "fingerprint": hashlib.sha256(_canonical(body)).hexdigest()}


def _native_scalar(data, key, kind):
    value = np.asarray(data.get(key))
    _require(value.shape == () and value.dtype.kind in kind,
             "Native forcing record is absent or has wrong type: " + key)
    return value.item()


def _native_ascii(data, key):
    value = np.asarray(data.get(key))
    _require(value.shape == (64,) and value.dtype.kind in "iu" and
             np.all((value >= 32) & (value <= 126)),
             "Native Q3-C identity record is absent or malformed: " + key)
    return bytes(int(item) for item in value).decode("ascii").rstrip()


def _validated_native_q3c_foundation(data, binding, law):
    expected_method = {2: "capacity.mpm", 8: "capacity.engelund_hansen"}.get(law)
    _require(expected_method is not None and
             int(_native_scalar(data, "native_q3c_forcing_verified", "iu")) == 1 and
             int(_native_scalar(data, "native_q3c_forcing_envelope", "iu")) == 54 and
             int(_native_scalar(data, "native_q3c_formula_code", "iu")) == law and
             _native_ascii(data, "native_q3c_method_id_ascii") == expected_method,
             "Non-Manning law 2/8 requires a natively verified Q3-C v54 route")
    execution = _native_ascii(data, "native_q3c_execution_fingerprint_ascii")
    profile = _native_ascii(data, "native_q3c_profile_fingerprint_ascii")
    forcing = _native_ascii(data, "native_q3c_forcing_fingerprint_ascii")
    raw = _native_ascii(data, "native_q3c_raw_roughness_sha256_ascii")
    _require(re.fullmatch(r"[0-9a-f]{64}", execution) is not None and
             re.fullmatch(r"[0-9a-f]{64}", profile) is not None and
             forcing == binding["fingerprint"] and
             raw == binding["roughness_parameter_array"]["sha256"],
             "Native Q3-C v54 forcing/profile fingerprints differ")


def _expected_fields():
    return {
        "schema", "roughness_method_id", "hydraulic_roughness_mode",
        "native_hydraulic_roughness_mode", "white_colebrook_chzk2d",
        "roughness_parameter_name", "roughness_parameter_units",
        "roughness_storage_field", "roughness_default_parameter",
        "roughness_parameter_minimum", "roughness_parameter_maximum",
        "roughness_parameter_array", "roughness_minimum_work_depth_m",
        "friction_minimum_work_depth_m", "effective_minimum_work_depth_m",
        "sediment_shear_method_id", "sediment_shear_mode",
        "native_sediment_shear_mode", "skin_shear_controls",
        "skin_shear_controls_active",
        "hydraulic_friction_unchanged_by_skin", "fingerprint",
    }


def _verify_binding(binding):
    _require(isinstance(binding, Mapping) and
             set(binding) == _expected_fields() and
             binding.get("schema") == SCHEMA,
             "Forcing binding fields differ")
    body = {key: binding[key] for key in binding if key != "fingerprint"}
    _require(binding["fingerprint"] ==
             hashlib.sha256(_canonical(body)).hexdigest(),
             "Forcing binding fingerprint differs")
    mode = binding["hydraulic_roughness_mode"]
    specification = ROUGHNESS.get(mode)
    chzk2d = binding["white_colebrook_chzk2d"]
    expected_roughness_method = (
        "roughness.white_colebrook_waqua" if chzk2d else
        None if specification is None else specification["method"])
    shear_mode = binding["sediment_shear_mode"]
    expected_shear = SHEAR.get(shear_mode)
    _require(specification is not None and type(chzk2d) is bool and
             (not chzk2d or mode == "white_colebrook") and
             binding["native_hydraulic_roughness_mode"] == specification["native"] and
             binding["roughness_method_id"] == expected_roughness_method and
             binding["roughness_parameter_name"] == ROUGHNESS[mode]["parameter"] and
             binding["roughness_parameter_units"] == ROUGHNESS[mode]["units"] and
             binding["roughness_storage_field"] == "manning_cell" and
             expected_shear is not None and
             binding["native_sediment_shear_mode"] == expected_shear[0] and
             binding["sediment_shear_method_id"] == expected_shear[1] and
             binding["skin_shear_controls_active"] is
             (shear_mode == "soulsby_clarke_skin") and
             binding["hydraulic_friction_unchanged_by_skin"] is True and
             (mode != "manning" or
              binding["roughness_minimum_work_depth_m"] <=
              binding["friction_minimum_work_depth_m"]) and
             binding["effective_minimum_work_depth_m"] == max(
                 binding["roughness_minimum_work_depth_m"],
                 binding["friction_minimum_work_depth_m"]),
             "Canonical roughness/shear semantic tuple differs")


def rebind_forcing_parameter(binding: Mapping[str, object],
                             roughness_parameter) -> dict:
    """Bind the same physical method to a topology-local raw parameter vector.

    AMR and other topology owners must transfer the selected *raw* C, n or ks
    values.  They may not retain the parent array hash, convert the values to
    an equivalent Manning coefficient, or alter any constitutive control.
    This helper changes only the range/byte identity fields that necessarily
    describe the new vector, then signs the complete canonical schema-2 body.
    """
    _verify_binding(binding)
    raw = _parameter_array(roughness_parameter)
    _require(raw.size > 0,
             "Rebound roughness parameter vector cannot be empty")
    mode = binding["hydraulic_roughness_mode"]
    _require(np.all(raw > 0.0) if mode != "manning" else
             np.all(raw >= 0.0),
             "Rebound roughness parameter is invalid for its selected method")
    body = {key: (dict(value) if isinstance(value, Mapping) else value)
            for key, value in binding.items() if key != "fingerprint"}
    body.update(
        roughness_parameter_minimum=float(np.min(raw)),
        roughness_parameter_maximum=float(np.max(raw)),
        roughness_parameter_array=_array_identity(raw),
    )
    rebound = {
        **body,
        "fingerprint": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    _verify_binding(rebound)
    _require(all(rebound[key] == binding[key] for key in rebound
                 if key not in {
                     "fingerprint", "roughness_parameter_minimum",
                     "roughness_parameter_maximum",
                     "roughness_parameter_array"}),
             "Topology-local rebind changed constitutive forcing semantics")
    return rebound


def _validate_consumer_scope(data, binding):
    """Reject every still-Manning-only dependent path before CUDA work."""
    mode, shear = (binding["hydraulic_roughness_mode"],
                   binding["sediment_shear_mode"])
    law = int(_native_scalar(data, "bedlaw", "iu"))
    exchange = int(_native_scalar(data, "exchange_mode", "iu"))
    secondary = int(_native_scalar(data, "secondary_flow_model", "iu"))
    modes = np.asarray(data.get("mode", np.zeros(0, np.int32)))
    _require(modes.ndim == 1 and modes.dtype.kind in "iu",
             "Boundary mode vector differs")
    if mode != "manning" and np.any(modes == 9):
        controls = np.asarray(data.get("section_control", ()))
        _require(controls.shape == modes.shape and
                 controls.dtype.kind in "iu" and
                 np.all(controls[modes == 9] == 1),
                 "Non-Manning mode 9 requires surveyed normal-depth control 1")
    # The composed viscous-inlet operator authenticates its complete dependency
    # bundle (mode 6, fixed grid, viscosity, geometry and implicit integrator)
    # in ``viscous_inlet_boundary.validate_viscous_inlet_scope``.  Do not
    # duplicate that validation here or reject its qualified C/W roughness
    # branches merely because this binding is non-Manning.
    _require(not (mode != "manning" and
                  bool(np.any(data.get("bedform_roughness", 0)))),
             "Non-Manning bedform feedback is not qualified")
    _require(not (mode != "manning" and secondary != 0),
             "Typed roughness secondary-flow consumers are not yet qualified")
    if mode != "manning" and law in (2, 8):
        _validated_native_q3c_foundation(data, binding, law)
    else:
        _require(mode == "manning" or law in (0, 5, 6, 7, 11),
                 "Selected capacity law still has a Manning-only GPU consumer")
    if law == 11:
        _require(shear == "total",
                 "Law 11 does not consume Soulsby/Clarke RP_TAUB")
        _require(exchange in (0, 4),
                 "Typed law-11 suspended exchange requires FLOW 2DH mode 4")
        _require(not (exchange == 4 and mode != "manning"),
                 "Typed non-Manning suspended exchange is not yet qualified")
    elif shear == "soulsby_clarke_skin":
        _require(law in (5, 6, 7),
                 "Soulsby/Clarke skin shear has no active GPU RP_TAUB consumer")
        _require(exchange == 0,
                 "Wilcock/Gaeuman typed GPU slice is bedload-only")
    elif mode != "manning":
        _require(exchange == 0,
                 "Selected typed GPU capacity has no qualified exchange path")
    _require(not (law == 9 and shear != "total"),
             "Delft3D BsSkin cannot replace TELEMAC-GAIA skin correction")


def apply_forcing_binding(data: dict, binding: Mapping[str, object]):
    """Verify native observations and install the resident forcing contract."""
    _require(isinstance(data, dict),
             "Forcing binding must target a mutable operator mapping")
    _verify_binding(binding)
    native = dict(
        native_hydraulic_roughness_mode=int(_native_scalar(
            data, "native_hydraulic_roughness_mode", "iu")),
        white_colebrook_chzk2d=bool(int(_native_scalar(
            data, "native_white_colebrook_chzk2d", "iu"))),
        native_sediment_shear_mode=int(_native_scalar(
            data, "native_sediment_shear_mode", "iu")),
        roughness_minimum_work_depth_m=float(_native_scalar(
            data, "native_roughness_minimum_work_depth_m", "f")),
        friction_minimum_work_depth_m=float(_native_scalar(
            data, "manning_friction_chezy_minimum_depth", "f")),
    )
    _require(native == {key: binding[key] for key in native},
             "Native parser roughness/shear state differs from metadata")
    raw = _parameter_array(data.get("manning_cell"))
    _require(_array_identity(raw) == binding["roughness_parameter_array"] and
             float(np.min(raw)) == binding["roughness_parameter_minimum"] and
             float(np.max(raw)) == binding["roughness_parameter_maximum"],
             "Native roughness parameter bytes differ from metadata binding")
    if "manning" in data:
        _require(np.asarray(data["manning"]).dtype == np.float64 and
                 np.array_equal(np.asarray(data["manning"]), raw),
                 "Boundary roughness alias differs from native cell parameter")
    controls = binding["skin_shear_controls"]
    native_skin = dict(
        kinematic_viscosity_m2_s=float(_native_scalar(
            data, "native_skin_kinematic_viscosity_m2_s", "f")),
        sand_roughness_m=float(_native_scalar(
            data, "native_skin_sand_roughness_m", "f")),
        silt_roughness_m=float(_native_scalar(
            data, "native_skin_silt_roughness_m", "f")),
        critical_mud_fraction_low=float(_native_scalar(
            data, "native_skin_critical_mud_fraction_low", "f")),
        critical_mud_fraction_high=float(_native_scalar(
            data, "native_skin_critical_mud_fraction_high", "f")),
        fixed_fluff_fraction=float(_native_scalar(
            data, "native_skin_fluff_fraction", "f")),
    )
    _require(native_skin == controls,
             "Native skin-control bytes differ from the forcing binding")
    _validate_consumer_scope(data, binding)

    data["roughness_parameter"] = raw  # same object, never an equivalent n
    data.update(
        delft_forcing_binding=dict(binding),
        delft_forcing_fingerprint=binding["fingerprint"],
        roughness_parameter_name=binding["roughness_parameter_name"],
        roughness_parameter_units=binding["roughness_parameter_units"],
        capacity_roughness_method_id=binding["roughness_method_id"],
        capacity_hydraulic_roughness_mode=binding["hydraulic_roughness_mode"],
        capacity_white_colebrook_chzk2d=np.int32(
            binding["white_colebrook_chzk2d"]),
        capacity_roughness_work_depth_m=np.float64(
            binding["effective_minimum_work_depth_m"]),
        capacity_sediment_shear_method_id=binding["sediment_shear_method_id"],
        capacity_sediment_shear_mode=binding["sediment_shear_mode"],
    )
    _require(data["roughness_parameter"] is data["manning_cell"] and
             np.array_equal(data["roughness_parameter"], data["manning_cell"]),
             "Canonical roughness alias differs from native parameter bytes")
    for key, value in controls.items():
        data["capacity_skin_" + key] = np.float64(value)
    return data


def forcing_identity(data: Mapping[str, object]) -> dict:
    binding = data.get(BINDING_FIELD)
    _verify_binding(binding)
    raw = _parameter_array(data.get("roughness_parameter"))
    _require(_array_identity(raw) == binding["roughness_parameter_array"] and
             np.array_equal(raw, np.asarray(data.get("manning_cell"))),
             "Prepared roughness parameter identity differs")
    return dict(
        delft_forcing_schema=SCHEMA,
        delft_forcing_fingerprint=binding["fingerprint"],
        hydraulic_roughness_method_id=binding["roughness_method_id"],
        roughness_parameter_name=binding["roughness_parameter_name"],
        roughness_parameter_units=binding["roughness_parameter_units"],
        roughness_parameter_array=dict(binding["roughness_parameter_array"]),
        roughness_default_parameter=binding["roughness_default_parameter"],
        roughness_minimum_work_depth_m=binding[
            "effective_minimum_work_depth_m"],
        sediment_shear_method_id=binding["sediment_shear_method_id"],
        white_colebrook_chzk2d=binding["white_colebrook_chzk2d"],
        automatic_roughness_substitution=False,
        automatic_shear_substitution=False,
        automatic_CPU_fallback=False,
    )


__all__ = [
    "SCHEMA", "BINDING_FIELD", "ROUGHNESS", "SHEAR",
    "forcing_binding_from_metadata", "apply_forcing_binding",
    "forcing_identity", "rebind_forcing_parameter",
]
