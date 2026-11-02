"""Fixed-grid implicit bridge for Delft RP_TAUB laws 5/6/7.

The bridge replaces only cell capacity preparation. Existing reconstructed
face transport, conservative Exner update, donor limiting and stratigraphy
remain authoritative. Open sediment boundaries are deliberately disabled in
this first bounded slice so no MPM boundary formula can be substituted.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .forcing_profile import forcing_identity


SCHEMA = 1
GENERATION = "implicit_rp_taub_laws567_v1"
PARAMETERS = {
    5: {"ACal": None},
    6: {"ThetaC0": 0.021, "Alpha0": 0.33},
    7: {"ThetaC0": 0.03, "Alpha0": 0.30},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _scalar(data, key):
    value = np.asarray(data.get(key))
    _require(value.shape == (), "RP_TAUB input lacks scalar " + key)
    return value.item()


def validate_rp_taub_prepared(data: Mapping[str, object]):
    _require(isinstance(data, Mapping), "RP_TAUB prepared input must be a mapping")
    law = int(_scalar(data, "bedlaw"))
    _require(law in PARAMETERS, "RP_TAUB bridge requires bed law 5, 6 or 7")
    forcing = forcing_identity(data)
    _require(forcing["sediment_shear_method_id"] in
             ("shear.total_roughness", "shear.soulsby_clarke_skin"),
             "RP_TAUB bridge lacks an authenticated shear choice")
    _require(int(_scalar(data, "exchange_mode")) == 0 and
             int(_scalar(data, "bed_slope")) == 0 and
             int(_scalar(data, "hiding_exposure")) == 0 and
             int(_scalar(data, "secondary_flow_model")) == 0,
             "RP_TAUB first slice requires bedload-only slope/hiding/secondary OFF")
    _require(int(_scalar(data, "nfrac")) >= 2 and
             int(_scalar(data, "nbed_layers")) >= 2,
             "Wilcock/Gaeuman requires a graded active layer and substrate")
    kinds = np.asarray(data.get("sediment_type_fraction"))
    _require(kinds.shape == (int(_scalar(data, "nfrac")),) and
             kinds.dtype.kind in "iu" and not np.any(kinds),
             "RP_TAUB first slice is noncohesive bedload only")
    bed_modes = np.asarray(data.get("bed_boundary_mode"))
    _require(bed_modes.shape == (int(_scalar(data, "nb")),) and
             bed_modes.dtype.kind in "iu" and not np.any(bed_modes),
             "RP_TAUB first slice excludes open sediment capacity/feed boundaries")
    for key in ("bc_bed_input", "fraction_bed_input", "series_bed_input",
                "fraction_series_bed_input"):
        _require(not np.any(data.get(key, 0)),
                 "RP_TAUB first slice excludes prescribed feed: " + key)
    calibration = float(_scalar(data, "transport_calibration"))
    _require(np.isfinite(calibration) and calibration >= 0.0,
             "RP_TAUB transport calibration is invalid")
    return law, forcing


def method_identity(data: Mapping[str, object]) -> dict:
    law, forcing = validate_rp_taub_prepared(data)
    parameters = dict(PARAMETERS[law])
    if law == 5:
        parameters["ACal"] = float(_scalar(data, "transport_calibration"))
    return dict(
        rp_taub_runtime_schema=SCHEMA,
        rp_taub_generation=GENERATION,
        capacity_formula_code=law,
        capacity_formula={5: "capacity.wilcock_crowe",
                          6: "capacity.wilcock_crowe_modified",
                          7: "capacity.gaeuman"}[law],
        capacity_parameters=parameters,
        sediment_shear_method_id=forcing["sediment_shear_method_id"],
        forcing_fingerprint=forcing["delft_forcing_fingerprint"],
        bed_capacity_channel="resident_reconstructed_face_exner_v1",
        open_sediment_boundary="closed_zero_flux_first_slice",
        automatic_CPU_fallback=False,
    )


def make_rp_taub_transport(Base):
    class ResidentRPTaubTransport(Base):
        def __init__(self, data, *args, **kwargs):
            law, forcing_identity_row = validate_rp_taub_prepared(data)
            source_value = kwargs.get("source_path")
            _require(source_value is not None,
                     "RP_TAUB bridge requires an authenticated source archive")
            folder = Path(source_value).resolve().parent
            required = ("m1_capacity_dispatch.cu", "hydraulic_roughness.cu",
                        "sediment_skin_shear.cu")
            _require(all((folder/name).is_file() and
                         not (folder/name).is_symlink() for name in required),
                     "RP_TAUB archived CUDA source inventory differs")
            shadow = dict(data, bedlaw=np.int32(2), bed_slope=np.int32(0),
                          exchange_mode=np.int32(0), settling_mode=np.int32(0),
                          hiding_exposure=np.int32(0),
                          secondary_flow_model=np.int32(0),
                          secondary_flow=np.int32(0),
                          secondary_flow_transport_direction=np.int32(0),
                          mpm_friction_mode=np.int32(0))
            super().__init__(shadow, *args, **kwargs)
            from .m1_capacity_dispatch import ResidentM1CapacityDispatch
            self.actual_law = law
            self.forcing_identity_row = forcing_identity_row
            # Persist the already authenticated receipt.  The constructor's
            # ``data`` name is local and must never be consulted later when a
            # runtime receipt is requested (or after a restart restores the
            # resident operator).
            self.rp_taub_identity = method_identity(data)
            self.coupled_mpm = self.coupled_gaia = self.coupled_bedload = False
            self.coupled_kind = 0
            self.dispatch = ResidentM1CapacityDispatch(
                source_path=folder/"m1_capacity_dispatch.cu",
                roughness_source_path=folder/"hydraulic_roughness.cu",
                skin_source_path=folder/"sediment_skin_shear.cu")
            cp = self.cp
            values = dict(PARAMETERS[law])
            if law == 5:
                values["ACal"] = float(data["transport_calibration"])
            self.capacity_parameters = MappingProxyType({
                key: cp.full(self.ng, value, dtype=cp.float64)
                for key, value in values.items()})
            binding = data["delft_forcing_binding"]
            self.roughness_mode = binding["hydraulic_roughness_mode"]
            self.roughness_waqua = bool(binding["white_colebrook_chzk2d"])
            self.work_depth = float(binding["effective_minimum_work_depth_m"])
            self.kinematic_viscosity = cp.full(
                self.nc, binding["skin_shear_controls"][
                    "kinematic_viscosity_m2_s"], dtype=cp.float64)
            self.sediment_type = cp.asarray(
                np.asarray(data["sediment_type_fraction"], dtype=np.int32))
            self.skin_fields = None
            if binding["sediment_shear_mode"] == "soulsby_clarke_skin":
                controls = binding["skin_shear_controls"]
                self.skin_fields = MappingProxyType(dict(
                    ks_sand=cp.full(self.nc, controls["sand_roughness_m"]),
                    ks_silt=cp.full(self.nc, controls["silt_roughness_m"]),
                    mud_factor=cp.zeros(self.nc, dtype=cp.float64),
                    critical_mud_1=cp.full(
                        self.nc, controls["critical_mud_fraction_low"]),
                    critical_mud_2=cp.full(
                        self.nc, controls["critical_mud_fraction_high"]),
                    fluff_fraction=cp.full(
                        self.nc, controls["fixed_fluff_fraction"])))
            b = self.buffers
            b["capacity_absolute_shields"] = cp.zeros(
                (self.ng, self.nc), dtype=cp.float64)
            b["capacity_critical_shields"] = cp.zeros_like(
                b["capacity_absolute_shields"])
            b["capacity_stress_ratio"] = cp.zeros_like(
                b["capacity_absolute_shields"])
            b["capacity_total_shear"] = cp.zeros(self.nc, dtype=cp.float64)
            b["capacity_sediment_shear"] = cp.zeros(self.nc, dtype=cp.float64)
            self.last_capacity_result = None

        def prepare(self, q, bed, layers, secondary_mass, active):
            cp, c, a, b = self.cp, self.config, self.geometry, self.buffers
            self._array(q, (3, self.nc)); self._array(bed, (self.nc,))
            self._array(layers, (self.nl, self.ng, self.nc))
            self._array(secondary_mass, (self.nc,))
            self._array(active, (self.nc,), cp.int32)
            surface_volume = layers[0]/a["bulk"][:, None]
            volume = cp.sum(surface_volume, axis=0)
            fill = cp.minimum(1.0, volume/np.float64(c["target"]))
            b["availability"][...] = cp.where(
                volume[None, :] > 0.0,
                surface_volume/cp.where(volume[None, :] > 0.0,
                                        volume[None, :], 1.0)*fill[None, :],
                0.0)
            present = surface_volume > 0.0
            local_d90 = cp.max(cp.where(present, a["d90"][:, None], 0.0), axis=0)
            b["mixture_d90"][...] = cp.where(
                local_d90 > 0.0, local_d90, cp.max(a["d90"]))
            wet = (q[0] > np.float64(c["dry"])) & (active != 0)
            denominator = cp.where(wet, q[0], 1.0)
            u = cp.ascontiguousarray(cp.where(wet, q[1]/denominator, 0.0))
            v = cp.ascontiguousarray(cp.where(wet, q[2]/denominator, 0.0))
            result = self.dispatch.evaluate(
                self.actual_law, depth=q[0], u=u, v=v,
                roughness_parameter=a["manning_cell"],
                kinematic_viscosity=self.kinematic_viscosity,
                d50=a["d50"], d90=a["d90"], solid_density=a["rhos"],
                parameters=self.capacity_parameters,
                roughness_mode=self.roughness_mode,
                waqua_white_colebrook=self.roughness_waqua,
                water_density=float(c["rho"]), gravity=float(c["grav"]),
                minimum_work_depth=self.work_depth,
                bulk_density=a["bulk"], sediment_type=self.sediment_type,
                surface_mass=layers[0], skin_fields=self.skin_fields,
                reject_invalid=False)
            cp.copyto(b["capacity"], result.bed_mass)
            cp.copyto(b["capacity_absolute_shields"], result.absolute_shields)
            cp.copyto(b["capacity_critical_shields"], result.critical_shields)
            cp.copyto(b["capacity_stress_ratio"], result.stress_ratio)
            cp.copyto(b["capacity_total_shear"], result.total_shear)
            cp.copyto(b["capacity_sediment_shear"], result.sediment_shear)
            cp.copyto(b["cell_status"], result.status)
            b["unrotated_status"].fill(0)
            self.last_capacity_result = result
            self._launch("st_gradients", self.nc, (
                np.int32(self.nc), np.int32(self.nf), np.int32(self.na),
                np.int32(self.ng), b["capacity"], bed, a["bed_ls_matrix"],
                a["bed_ls_delta"], a["bed_ls_weight"], a["xy"], a["mid"],
                a["owner"], a["neighbour"], a["offset"], a["faces"],
                a["signs"], active, b["gradient"], b["downslope"]))
            return b["unrotated_status"]

        def capacity_diagnostic(self):
            b = self.buffers
            return dict(schema=SCHEMA, formula_code=self.actual_law,
                        total_shear=b["capacity_total_shear"],
                        sediment_shear=b["capacity_sediment_shear"],
                        absolute_shields=b["capacity_absolute_shields"],
                        critical_shields=b["capacity_critical_shields"],
                        stress_ratio=b["capacity_stress_ratio"],
                        status=b["cell_status"])

        def profile_receipt(self):
            return dict(self.rp_taub_identity)

    ResidentRPTaubTransport.__name__ = "ResidentRPTaubTransport"
    return ResidentRPTaubTransport


__all__ = ["SCHEMA", "GENERATION", "PARAMETERS",
           "validate_rp_taub_prepared", "method_identity",
           "make_rp_taub_transport"]
