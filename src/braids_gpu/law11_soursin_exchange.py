"""Resident CUDA FLOW 2DH exchange for the law-11 suspended channel.

This is a distinct operator from the historical ``exchange_mode == 2``
Van Rijn 1993 closure.  A law-11 capacity evaluation supplies ``ssus`` and
this operator applies the locked Delft3D4 FLOW ``eqtran``/``soursin_2d``
conversion and source split.  It updates only private suspended/active-bed
trial arrays; the enclosing implicit transaction owns commit and rollback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.channel_composition import (
    TRA_COMBINE,
    ChannelCompositionError,
    load_channel_composition_profile,
)
from braids_core.sediment.flow2dh_exchange import (
    FLOW2DH_EXCHANGE_ID,
    FLOW2DH_EXCHANGE_SEMANTIC_ID,
)
from braids_core.sediment.flow2dh_settling import (
    FLOW2DH_SETTLING_FORM,
    FLOW2DH_SETTLING_ID,
    FLOW2DH_SETTLING_SEMANTIC_ID,
    settlefrm2_noncohesive,
)
from braids_core.sediment.suspension_profile import (
    FRACTION_BASIS_CODES,
    REFERENCE_DEPTH_FLOOR_M,
    SEDIMENT_ACTIVITY_CONTRACT,
    TEMPERATURE_MODE_CODES,
    WET_MASK_PROVIDER,
    SuspensionProfileError,
    load_suspension_profile,
    suspension_profile_semantic_id,
)


HERE = Path(__file__).resolve().parent
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
           "--prec-sqrt=true", "--ftz=false")
EXCHANGE_MODE = 4
CAPACITY_ROUTE = "general_capacity_galappatti_v1"
KERNEL = "law11_soursin_2d_exchange"
SUSPENSION_PROFILE_KEY = "suspension_runtime_profile"
SUSPENSION_FINGERPRINT_KEY = "suspension_profile_fingerprint"
SETTLING_CLEAR_WATER_KEY = "settling_clear_water_velocity_m_s"
CHANNEL_PROFILE_KEY = "channel_composition_profile"
CHANNEL_FINGERPRINT_KEY = "channel_composition_fingerprint"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _controls(data: Mapping[str, object]):
    try:
        profile = load_suspension_profile(data.get(SUSPENSION_PROFILE_KEY))
    except SuspensionProfileError as error:
        raise ValueError(
            "FLOW 2DH suspension-profile authentication failed: " + str(error)
        ) from error
    _require(data.get(SUSPENSION_FINGERPRINT_KEY) == profile.fingerprint,
             "FLOW 2DH prepared/profile suspension fingerprint differs")
    basis = data.get("capacity_fraction_basis")
    _require(basis in FRACTION_BASIS_CODES,
             "FLOW 2DH law-11 exchange requires an explicit IFractions basis")
    code = np.asarray(data.get("ifractions_code"))
    _require(code.shape == () and code.dtype.kind in "iu" and
             int(code) == FRACTION_BASIS_CODES[basis],
             "FLOW IFractions code and capacity fraction basis differ")
    factsd = np.asarray(data.get("factsd"))
    _require(factsd.shape == () and factsd.dtype.kind in "fiu" and
             np.isfinite(factsd) and float(factsd) > 0.0,
             "FLOW 2DH law-11 exchange requires positive finite FacTsd")
    thresholds = np.asarray(data.get("sedthr_fraction"))
    _require(thresholds.shape == (1,) and thresholds.dtype.kind in "fiu" and
             np.isfinite(thresholds).all() and np.all(thresholds >= 0.0),
             "FLOW 2DH law-11 exchange requires one finite nonnegative SedThr")
    _require(profile.fraction_basis == basis and
             profile.fraction_basis_code == int(code) and
             profile.fac_tsd == float(factsd) and
             profile.sediment_threshold_m == float(thresholds[0]) and
             profile.reference_depth_floor_m == REFERENCE_DEPTH_FLOOR_M,
             "FLOW 2DH suspension profile and prepared controls differ")
    _require(profile.temperature_mode == "equilibrium_constant" and
             profile.temperature_mode_code ==
             TEMPERATURE_MODE_CODES["equilibrium_constant"],
             "FLOW 2DH GPU first slice requires equilibrium-constant temperature")

    def scalar(key, kind="fiu"):
        value = np.asarray(data.get(key))
        _require(value.shape == () and value.dtype.kind in kind and
                 (value.dtype.kind not in "f" or np.isfinite(value)),
                 "FLOW 2DH prepared settling control differs: " + key)
        return value.item()

    settlefrm = int(scalar("settlefrm_code", "iu"))
    fac_dss = float(scalar("settling_fac_dss"))
    temperature_code = int(scalar("settling_temperature_mode_code", "iu"))
    temperature = float(scalar("settling_temperature_c"))
    reference_concentration = float(scalar(
        "settling_reference_concentration_kg_m3"))
    calibration = float(scalar("suspended_calibration_factor"))
    activity_code = int(scalar("sediment_activity_contract_code", "iu"))
    wet_provider_code = int(scalar("wet_mask_provider_code", "iu"))
    water_density = float(scalar("rho"))
    gravity = float(scalar("grav"))
    d50 = np.asarray(data.get("d50_fraction"))
    density = np.asarray(data.get("rhos_fraction"))
    _require(d50.shape == density.shape == (1,) and
             d50.dtype.kind in "fiu" and density.dtype.kind in "fiu" and
             np.isfinite(d50).all() and np.isfinite(density).all(),
             "FLOW 2DH prepared SedD50/RhoSol arrays differ")
    _require(
        settlefrm == profile.settling_form == FLOW2DH_SETTLING_FORM and
        fac_dss == profile.settling_fac_dss[0] and
        float(d50[0]) == profile.sediment_d50_m[0] and
        float(density[0]) == profile.sediment_density_kg_m3[0] and
        reference_concentration ==
        profile.settling_reference_concentration_kg_m3 and
        water_density == profile.water_density_kg_m3 and
        temperature_code == profile.temperature_mode_code and
        temperature == profile.temperature_value_c and
        calibration == profile.suspended_calibration_factor and
        gravity == profile.gravity_m_s2 and
        activity_code == 1 and wet_provider_code == 1 and
        profile.sediment_activity_contract == SEDIMENT_ACTIVITY_CONTRACT and
        profile.wet_mask_provider == WET_MASK_PROVIDER,
        "FLOW 2DH suspension profile and prepared settling controls differ")
    _require(calibration > 0.0,
             "FLOW 2DH law-11 exchange requires positive global Sus calibration")
    try:
        channel = load_channel_composition_profile(
            data.get(CHANNEL_PROFILE_KEY))
    except ChannelCompositionError as error:
        raise ValueError(
            "FLOW 2DH channel-composition profile authentication failed: " +
            str(error)
        ) from error
    _require(data.get(CHANNEL_FINGERPRINT_KEY) == channel.fingerprint,
             "FLOW 2DH prepared/channel-composition fingerprint differs")
    _require(len(channel.fractions) == 1 and
             channel.fractions[0].transport_class == TRA_COMBINE and
             channel.suspended_factor == calibration,
             "FLOW 2DH exchange requires the authenticated class-3 Sus owner")
    bed_factor = np.asarray(data.get("transport_calibration"))
    _require(bed_factor.shape == () and bed_factor.dtype.kind in "fiu" and
             np.isfinite(bed_factor) and float(bed_factor) >= 0.0 and
             channel.bed_factor == float(bed_factor),
             "FLOW 2DH global Bed factor differs from channel composition")
    old_cref = np.asarray(data.get("hindered_settling_reference_concentration"))
    _require(old_cref.shape == () and old_cref.dtype.kind in "fiu" and
             float(old_cref) == reference_concentration,
             "FLOW 2DH legacy and canonical Cref controls differ")
    clear = settlefrm2_noncohesive(
        np.zeros((1, 1), dtype=float), d50, [fac_dss], density,
        water_density, temperature,
        settling_reference_concentration_kg_m3=reference_concentration,
        gravity_m_s2=gravity,
    ).clear_water_settling_velocity_m_s[0]
    recorded_clear = np.asarray(data.get(SETTLING_CLEAR_WATER_KEY))
    legacy_clear = np.asarray(data.get("ws_fraction"))
    _require(recorded_clear.shape == legacy_clear.shape == (1,) and
             recorded_clear.dtype.kind in "fiu" and
             legacy_clear.dtype.kind in "fiu" and
             np.isfinite(recorded_clear).all() and
             np.isfinite(legacy_clear).all() and
             np.array_equal(recorded_clear.astype(float, copy=False), clear) and
             np.array_equal(legacy_clear.astype(float, copy=False), clear),
             "FLOW 2DH resolved clear-water settling velocity differs")
    return dict(
        profile=profile, basis=str(basis), basis_code=int(code),
        fac_tsd=float(factsd), sediment_thresholds=tuple(
            float(v) for v in thresholds),
        settling_fac_dss=(fac_dss,), sediment_d50_m=(float(d50[0]),),
        sediment_density_kg_m3=(float(density[0]),),
        reference_concentration=reference_concentration,
        water_density=water_density, temperature_c=temperature,
        calibration=calibration, gravity=gravity, channel=channel,
        clear_water_velocity=tuple(float(value) for value in clear),
    )


def method_identity(data: Mapping[str, object]) -> dict:
    """Stable physical/numerical identity recorded in run and restart hashes."""
    controls = _controls(data)
    profile = controls["profile"]
    return dict(
        suspended_exchange_mode=EXCHANGE_MODE,
        suspended_exchange_method=FLOW2DH_EXCHANGE_ID,
        suspended_exchange_semantic_id=FLOW2DH_EXCHANGE_SEMANTIC_ID,
        suspension_profile_semantic_id=suspension_profile_semantic_id(),
        suspension_profile_fingerprint=profile.fingerprint,
        settling_operator_id=FLOW2DH_SETTLING_ID,
        settling_semantic_id=FLOW2DH_SETTLING_SEMANTIC_ID,
        settling_form=FLOW2DH_SETTLING_FORM,
        settling_implementation="resident_cuda_settlefrm2_noncohesive_v1",
        settling_fac_dss=list(controls["settling_fac_dss"]),
        sediment_d50_m=list(controls["sediment_d50_m"]),
        sediment_density_kg_m3=list(controls["sediment_density_kg_m3"]),
        settling_reference_concentration_kg_m3=
            controls["reference_concentration"],
        water_density_kg_m3=controls["water_density"],
        settling_temperature_mode=profile.temperature_mode,
        settling_temperature_mode_code=profile.temperature_mode_code,
        settling_temperature_c=controls["temperature_c"],
        resolved_clear_water_settling_velocity_m_s=list(
            controls["clear_water_velocity"]),
        settling_hindering=(
            "max(0,1-max(0,total_concentration)/Cref)^5"),
        gravity_m_s2=controls["gravity"],
        capacity_suspended_route=CAPACITY_ROUTE,
        suspended_calibration_factor=controls["calibration"],
        channel_composition_fingerprint=controls["channel"].fingerprint,
        channel_composition_semantic_id=controls["channel"].semantic_id,
        channel_transport_class=TRA_COMBINE,
        capacity_to_concentration=(
            "precomposed_signed_equilibrium_concentration*"
            "configured_surface_fraction"),
        suspended_factor_owner="authenticated_channel_composer_once",
        capacity_fraction_basis=controls["basis"],
        ifractions_code=controls["basis_code"],
        adaptation_factor_keyword="FacTsd",
        factsd=controls["fac_tsd"],
        sediment_activity=SEDIMENT_ACTIVITY_CONTRACT,
        wet_mask_provider=WET_MASK_PROVIDER,
        sedthr_m=list(controls["sediment_thresholds"]),
        reference_depth_floor_m=REFERENCE_DEPTH_FLOOR_M,
        explicit_source_reference_depth="max(0.01,h0_actual)",
        assembled_mass_numerator_depth="h0_actual",
        equilibrium_and_sink_reference_depth="max(0.01,h1_actual)",
        source_treatment="mass_form_explicit_source_implicit_sink_backward_euler",
        settling_floor_m_s=1.0e-3,
        finite_inventory_limiter=(
            "braids_nonnegative_available_mass_v1_not_FLOW_red_soursin"),
        exchange_ledger="signed_suspended_gain_and_bed_gain_kg",
        automatic_CPU_fallback=False,
    )


def validate_scope(data: dict) -> None:
    """Fail closed before importing CuPy or compiling a CUDA module."""
    _require(int(np.asarray(data.get("bedlaw", -1))) == 11 and
             int(np.asarray(data.get("exchange_mode", -1))) == EXCHANGE_MODE,
             "FLOW 2DH law-11 exchange requires bedlaw 11 and exchange mode 4")
    _require(data.get("capacity_suspended_route") == CAPACITY_ROUTE,
             "FLOW 2DH law-11 suspended route identity differs")
    _require(int(np.asarray(data.get("settling_mode", -1))) == 1,
             "FLOW 2DH SettleFrm=2 exchange requires hindered settling mode 1")
    _require(int(np.asarray(data.get("scheme", -1))) == 6 and
             not bool(np.any(data.get("amr_enabled", 0))),
             "FLOW 2DH law-11 exchange requires fixed-grid scheme 6")
    q = np.asarray(data.get("q"))
    layers = np.asarray(data.get("layer_mass"))
    _require(q.ndim == 2 and q.shape[0] == 3 and layers.ndim == 3 and
             layers.shape[2] == q.shape[1] and layers.shape[1] >= 1,
             "FLOW 2DH law-11 exchange state dimensions differ")
    ng = int(layers.shape[1])
    _require(ng == 1 and int(np.asarray(data.get("nfrac", -1))) == 1,
             "FLOW 2DH law-11 exchange is initially limited to one fraction")
    _controls(data)
    kinds = np.asarray(data.get("sediment_type_fraction"))
    density = np.asarray(data.get("rhos_fraction"))
    _require(kinds.shape == (ng,) and kinds.dtype.kind in "iu" and
             not np.any(kinds),
             "FLOW 2DH law-11 exchange currently supports noncohesive fractions only")
    _require(density.shape == (ng,) and np.isfinite(density).all() and
             np.all(density > float(data["rho"])),
             "FLOW 2DH law-11 solid densities are invalid")
    _require(np.isfinite(float(data.get("rho", np.nan))) and
             float(data["rho"]) > 0.0 and
             np.isfinite(float(data.get("dry_tolerance", np.nan))) and
             float(data["dry_tolerance"]) >= 0.0,
             "FLOW 2DH law-11 fluid/dry controls are invalid")


class ResidentLaw11SoursinExchange:
    """One-cell-per-thread bounded FLOW 2DH water/bed exchange."""

    def __init__(self, data: dict, *, source_path=None):
        validate_scope(data)
        import cupy as cp

        self.cp = cp
        self.nc = int(np.asarray(data["q"]).shape[1])
        self.ng = int(np.asarray(data["layer_mass"]).shape[1])
        controls = _controls(data)
        self.rho = np.float64(controls["water_density"])
        self.dry = np.float64(data["dry_tolerance"])
        self.factsd = np.float64(controls["fac_tsd"])
        self.temperature = np.float64(controls["temperature_c"])
        self.reference_concentration = np.float64(
            controls["reference_concentration"])
        self.suspended_calibration = np.float64(controls["calibration"])
        self.capacity_speed_epsilon = np.float64(
            controls["channel"].epsilon)
        self.gravity = np.float64(controls["gravity"])
        self.reference_depth_floor = np.float64(REFERENCE_DEPTH_FLOOR_M)
        self.sediment_d50 = cp.asarray(np.ascontiguousarray(
            controls["sediment_d50_m"], dtype=np.float64))
        self.settling_fac_dss = cp.asarray(np.ascontiguousarray(
            controls["settling_fac_dss"], dtype=np.float64))
        self.sediment_density = cp.asarray(np.ascontiguousarray(
            controls["sediment_density_kg_m3"], dtype=np.float64))
        self.sediment_threshold = cp.asarray(
            np.ascontiguousarray(data["sedthr_fraction"], dtype=np.float64))
        path = Path(source_path or HERE / "law11_soursin_exchange.cu")
        module = cp.RawModule(code=path.read_text(encoding="utf-8"),
                              options=OPTIONS, name_expressions=(KERNEL,))
        self.kernel = module.get_function(KERNEL)
        shape = (self.ng, self.nc)
        self.next_suspended = cp.empty(shape, dtype=cp.float64)
        self.next_active = cp.empty(shape, dtype=cp.float64)
        self.equilibrium_concentration = cp.empty(shape, dtype=cp.float64)
        self.adaptation_velocity = cp.empty(shape, dtype=cp.float64)
        self.dimensionless_timescale = cp.empty(shape, dtype=cp.float64)
        self.clear_water_settling_velocity = cp.empty(shape, dtype=cp.float64)
        self.settling_velocity = cp.empty(shape, dtype=cp.float64)
        self.hindered_factor = cp.empty(self.nc, dtype=cp.float64)
        self.delta = cp.empty(shape, dtype=cp.float64)
        self.limiter_correction = cp.empty(shape, dtype=cp.float64)
        self.sediment_active = cp.empty(self.nc, dtype=cp.int32)
        self.status = cp.empty(self.nc, dtype=cp.int32)

    def _array(self, value, shape, name):
        _require(isinstance(value, self.cp.ndarray) and value.shape == shape and
                 value.dtype == self.cp.float64 and value.flags.c_contiguous,
                 f"{name} must be contiguous device float64 {shape}")

    def _mask(self, value, name):
        _require(isinstance(value, self.cp.ndarray) and
                 value.shape == (self.nc,) and value.dtype == self.cp.int32 and
                 value.flags.c_contiguous,
                 f"{name} must be contiguous device int32 ({self.nc},)")

    def step(self, depth_before, depth_after, endpoint_momentum, suspended,
             active_bed, equilibrium_capacity, surface_fraction, total_shear,
             wet_mask, dt):
        """Advance one private trial using the backend-neutral FLOW identity."""
        cp = self.cp
        self._array(depth_before, (self.nc,), "depth_before")
        self._array(depth_after, (self.nc,), "depth_after")
        self._array(endpoint_momentum, (2, self.nc), "endpoint_momentum")
        self._array(suspended, (self.ng, self.nc), "suspended")
        self._array(active_bed, (self.ng, self.nc), "active_bed")
        self._array(equilibrium_capacity, (self.ng, self.nc),
                    "equilibrium_capacity")
        self._array(surface_fraction, (self.ng, self.nc), "surface_fraction")
        self._array(total_shear, (self.nc,), "total_shear")
        self._mask(wet_mask, "wet_mask")
        _require(np.isfinite(dt) and dt > 0.0,
                 "FLOW 2DH law-11 exchange requires a positive finite timestep")
        self.kernel(((self.nc + 127)//128,), (128,), (
            np.int32(self.nc), np.int32(self.ng), depth_before, depth_after,
            endpoint_momentum, suspended, active_bed, equilibrium_capacity,
            surface_fraction, total_shear, wet_mask, self.sediment_d50,
            self.settling_fac_dss, self.sediment_density,
            self.sediment_threshold, self.rho, self.temperature,
            self.reference_concentration, self.capacity_speed_epsilon,
            self.gravity, self.dry, self.factsd,
            self.reference_depth_floor, np.float64(dt),
            self.next_suspended, self.next_active,
            self.equilibrium_concentration, self.adaptation_velocity,
            self.dimensionless_timescale,
            self.clear_water_settling_velocity, self.settling_velocity,
            self.hindered_factor, self.delta,
            self.limiter_correction, self.sediment_active, self.status))
        return dict(
            suspended=self.next_suspended,
            active=self.next_active,
            equilibrium_concentration=self.equilibrium_concentration,
            adaptation_velocity=self.adaptation_velocity,
            dimensionless_timescale=self.dimensionless_timescale,
            clear_water_settling_velocity=
                self.clear_water_settling_velocity,
            settling_velocity=self.settling_velocity,
            hindered_factor=self.hindered_factor,
            delta=self.delta,
            limiter_correction=self.limiter_correction,
            sediment_active=self.sediment_active,
            status=self.status,
            method_id=FLOW2DH_EXCHANGE_ID,
            semantic_id=FLOW2DH_EXCHANGE_SEMANTIC_ID,
        )


__all__ = [
    "EXCHANGE_MODE", "CAPACITY_ROUTE", "REFERENCE_DEPTH_FLOOR_M",
    "FRACTION_BASIS_CODES", "SUSPENSION_PROFILE_KEY",
    "SUSPENSION_FINGERPRINT_KEY", "SETTLING_CLEAR_WATER_KEY",
    "CHANNEL_PROFILE_KEY", "CHANNEL_FINGERPRINT_KEY",
    "FLOW2DH_EXCHANGE_ID",
    "FLOW2DH_EXCHANGE_SEMANTIC_ID", "method_identity", "validate_scope",
    "ResidentLaw11SoursinExchange",
]
