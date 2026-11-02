"""Resident CUDA wiring for locked FLOW 2DH boundary/diffusion physics.

The physical concentration selector and ``dicuv/0.7`` face law match the
locked Delft3D4 FLOW 2DH source.  BRAIDS deliberately retains its accepted
water-flux FCT advection and uses an explicit conservative donor-limited
diffusion update; those numerical choices have separate identities and are
never presented as the FLOW ADI solver.  The admitted integration obtains its
face area only from BRAIDS' accepted reconstructed hydraulic trace adapter.
The low-level device operator receives that resolved area explicitly.  There
is no CPU fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.flow2dh_boundary_diffusion import (
    FLOW2DH_HORIZONTAL_DIFFUSION_ID,
    FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
    FLOW2DH_HORIZONTAL_SCHMIDT,
    FLOW2DH_OPEN_BOUNDARY_ID,
    FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
)
from braids_core.sediment.suspension_profile import (
    BOUNDARY_CONCENTRATION_PROVIDER,
    DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER,
    DIFFUSION_FACE_ACTIVITY_CONTRACT,
    DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER,
    HORIZONTAL_DIFFUSIVITY_PROVIDER,
    load_suspension_profile,
)


HERE = Path(__file__).resolve().parent
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
           "--prec-sqrt=true", "--ftz=false")
KERNELS = (
    "m2b_resolve_boundary", "m2b_expected_advective_ledger",
    "m2b_sc_gradients",
    "m2b_patch_boundary_faces", "m2b_sa_face_low",
    "m2b_sa_cell_bounds", "m2b_face_area_from_trace",
    "m2b_diffusion_faces", "m2b_diffusion_theta",
    "m2b_diffusion_limit", "m2b_diffusion_update",
    "m2b_diffusion_ledger",
)
BOUNDARY_NUMERICAL_ID = "braids_accepted_flux_fct_flow2dh_boundary_v1"
DIFFUSION_NUMERICAL_ID = (
    "braids_explicit_donor_limited_shared_face_flow2dh_diffusion_v1"
)
FACE_AREA_PROVIDER = DIFFUSION_FACE_EXCHANGE_AREA_PROVIDER
BOUNDARY_EXTERIOR_DEPTH_PROVIDER = DIFFUSION_BOUNDARY_EXTERIOR_DEPTH_PROVIDER


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _host_array(data: Mapping[str, object], key: str, shape, *, integer=False,
                boolean=False):
    value = np.asarray(data.get(key))
    _require(value.shape == shape, f"FLOW 2DH GPU {key} shape differs")
    if boolean:
        _require(value.dtype.kind == "b", f"FLOW 2DH GPU {key} must be Boolean")
        return np.ascontiguousarray(value, dtype=np.int32)
    if integer:
        _require(value.dtype.kind in "iu", f"FLOW 2DH GPU {key} must be integer")
        return np.ascontiguousarray(value, dtype=np.int32)
    _require(value.dtype.kind in "fiu" and np.isfinite(value).all(),
             f"FLOW 2DH GPU {key} must be finite numeric")
    return np.ascontiguousarray(value, dtype=np.float64)


def method_identity(data: Mapping[str, object] | None = None) -> dict:
    result = dict(
        suspended_open_boundary_method=FLOW2DH_OPEN_BOUNDARY_ID,
        suspended_open_boundary_semantic_id=FLOW2DH_OPEN_BOUNDARY_SEMANTIC_ID,
        suspended_open_boundary_numerical_mapping=BOUNDARY_NUMERICAL_ID,
        boundary_concentration_provider=BOUNDARY_CONCENTRATION_PROVIDER,
        suspended_boundary_memory=(
            "accepted_per_face_per_fraction_last_outflow_concentration_and_return_clock"),
        horizontal_suspended_diffusion_method=FLOW2DH_HORIZONTAL_DIFFUSION_ID,
        horizontal_suspended_diffusion_semantic_id=
            FLOW2DH_HORIZONTAL_DIFFUSION_SEMANTIC_ID,
        horizontal_suspended_diffusion_numerical_mapping=DIFFUSION_NUMERICAL_ID,
        horizontal_suspended_schmidt=FLOW2DH_HORIZONTAL_SCHMIDT,
        diffusion_face_exchange_area_provider=FACE_AREA_PROVIDER,
        diffusion_boundary_exterior_depth_provider=
            BOUNDARY_EXTERIOR_DEPTH_PROVIDER,
        diffusion_face_activity_contract=DIFFUSION_FACE_ACTIVITY_CONTRACT,
        diffusion_temporal_method=(
            "explicit_post_FCT_conservative_donor_limited_v1"),
        boundary_ledger_sign="positive_into_domain",
        diffusion_ledger_sign="positive_into_domain",
        automatic_CPU_fallback=False,
    )
    if data is not None:
        profile = load_suspension_profile(data.get("suspension_runtime_profile"))
        _require(data.get("suspension_profile_fingerprint") ==
                 profile.fingerprint,
                 "M2-B identity/profile suspension fingerprint differs")
        return_time = np.asarray(data.get("boundary_return_time_s"))
        ng = int(data.get("nfrac", 0))
        nb = int(data.get("nb", 0))
        nf = int(data.get("nf", 0))
        valid_shapes = ((nb,), (ng, nb))
        _require(ng == 1 and nb > 0 and nf > 0 and
                 return_time.dtype.kind in "fiu" and
                 return_time.shape in valid_shapes and
                 np.isfinite(return_time).all(),
                 "M2-B identity requires finite logical-boundary return times")
        result.update(
            boundary_prescription_method_id=
                profile.boundary_prescription_method_id,
            boundary_return_method_id=profile.boundary_return_method_id,
            boundary_return_time_provider=
                profile.boundary_return_time_provider,
            boundary_schedule_identity_contract=
                profile.boundary_schedule_identity_contract,
            thatcher_harleman_memory_contract=
                profile.thatcher_harleman_memory_contract,
            thatcher_harleman_clock_contract=
                profile.thatcher_harleman_clock_contract,
            sediment_fraction_classes=list(profile.sediment_fraction_classes),
            neubc_sand=profile.neubc_sand,
            neubc_mud=profile.neubc_mud,
            neumann_boundary_by_fraction=list(
                profile.neumann_boundary_by_fraction),
            neubc_resolution_contract=profile.neubc_resolution_contract,
            boundary_concentration_unit=profile.boundary_concentration_unit,
            horizontal_diffusivity_provider=
                profile.horizontal_diffusivity_provider,
            horizontal_diffusivity_mode=profile.horizontal_diffusivity_mode,
            horizontal_diffusivity_m2_s=
                profile.horizontal_diffusivity_m2_s,
            horizontal_schmidt_branch_id=
                profile.horizontal_schmidt_branch_id,
            diffusion_dry_threshold_m=profile.diffusion_dry_threshold_m,
            boundary_return_time_s=return_time.astype(float).tolist(),
        )
    return result


class ResidentFlow2DHBoundaryDiffusion:
    """One-sand fixed-grid boundary memory and conservative diffusion path.

    The admitted case input is uniform within each logical boundary ID.  The
    resolved value can still differ face by face after the return-time rule
    because every face owns its flow direction and memory.  Arbitrary FLOW
    INCBCC along-boundary interpolation is not claimed by this slice.
    """

    def __init__(self, data: Mapping[str, object], suspended_faces,
                 suspended_advection, *, source_path=None):
        import cupy as cp

        self.cp = cp
        self.faces = suspended_faces
        self.advection = suspended_advection
        self.nc, self.nf = suspended_advection.nc, suspended_advection.nf
        self.ng, self.nb = suspended_advection.ng, suspended_advection.nb
        _require(self.ng == 1, "M2-B implicit GPU slice requires one sand fraction")
        _require(suspended_faces.nc == self.nc and suspended_faces.nf == self.nf and
                 suspended_faces.ng == self.ng and suspended_faces.nb == self.nb,
                 "M2-B suspended face/advection geometry differs")
        profile = load_suspension_profile(data.get("suspension_runtime_profile"))
        _require(data.get("suspension_profile_fingerprint") == profile.fingerprint,
                 "M2-B prepared/profile suspension fingerprint differs")
        self.profile = profile
        _require(profile.sediment_fraction_classes == ("sand",),
                 "M2-B implicit GPU slice requires one authenticated sand class")
        self.advection_dry = float(suspended_advection.dry)
        self.dry = float(profile.diffusion_dry_threshold_m)
        dicuv = float(profile.horizontal_diffusivity_m2_s)
        _require(np.isfinite(self.advection_dry) and self.advection_dry >= 0.0,
                 "M2-B advection wet/dry threshold must be finite and nonnegative")
        _require(np.isfinite(self.dry) and self.dry >= 0.0,
                 "M2-B diffusion DryFlc must be finite and nonnegative")
        _require(np.isfinite(dicuv) and dicuv >= 0.0,
                 "M2-B dicuv must be finite and nonnegative")
        _require(profile.horizontal_diffusivity_mode == "uniform_constant" and
                 profile.horizontal_diffusivity_provider ==
                 HORIZONTAL_DIFFUSIVITY_PROVIDER,
                 "M2-B GPU requires authenticated uniform dicuv")
        _require(profile.diffusion_face_exchange_area_provider == FACE_AREA_PROVIDER,
                 "M2-B accepted hydraulic face-area provider differs")
        _require(profile.diffusion_boundary_exterior_depth_provider ==
                 BOUNDARY_EXTERIOR_DEPTH_PROVIDER and
                 profile.diffusion_face_activity_contract ==
                 DIFFUSION_FACE_ACTIVITY_CONTRACT,
                 "M2-B boundary exterior-depth provider differs")

        self.prescribed_neumann = np.ascontiguousarray(
            profile.neumann_boundary_by_fraction, dtype=np.int32)
        return_time = np.asarray(data.get("boundary_return_time_s"))
        if return_time.shape == (self.nb,):
            return_time = np.broadcast_to(return_time[None, :], (self.ng, self.nb))
        _require(return_time.dtype.kind in "fiu" and
                 np.isfinite(return_time).all() and
                 return_time.shape == (self.ng, self.nb),
                 "M2-B boundary return time must be finite per logical boundary/fraction")
        bid_host = np.asarray(data["bid"])
        expanded = np.zeros((self.ng, self.nf), dtype=np.float64)
        boundary = bid_host >= 0
        expanded[:, boundary] = return_time[:, bid_host[boundary]]
        return_time = expanded
        self.return_time_host = np.ascontiguousarray(
            return_time, dtype=np.float64)
        self.return_time = cp.asarray(self.return_time_host)
        self.neumann = cp.asarray(self.prescribed_neumann)
        self.dicuv = cp.full(self.nc, dicuv, dtype=cp.float64)

        a = suspended_advection.geometry
        self.owner, self.neighbour, self.bid = a["owner"], a["neighbour"], a["bid"]
        self.offset, self.csr_faces, self.signs = a["offset"], a["faces"], a["signs"]
        self.length, self.area = a["length"], a["area"]
        distance = _host_array(data, "distance", (2, self.nf))
        neighbour = np.asarray(data["neighbour"])
        # Prepared distance[0] is owner-to-face.  Internal centres span both
        # half distances; an open boundary ends at that face and therefore
        # uses the owner half distance once (native v48 mid-owner geometry).
        centre_distance = np.where(
            neighbour >= 0, distance[0] + distance[1], distance[0])
        _require(np.all(centre_distance > 0.0),
                 "M2-B diffusion centre distances must be positive")
        self.centre_distance = cp.asarray(np.ascontiguousarray(centre_distance))
        bids = np.asarray(data["bid"])
        lists = [np.flatnonzero(bids == b).astype(np.int32) for b in range(self.nb)]
        if "suspended_amr_contract" in data:
            # M2-C return memory is owned by the exported, physically ordered
            # open-face CSR.  Re-sorting those faces by topology-local index
            # reverses some sections after regridding and breaks exact TH
            # memory handoff even though the physical faces are unchanged.
            boundary_faces_wide = np.asarray(data.get("boundary_faces"))
            boundary_offset_wide = np.asarray(data.get("boundary_offset"))
            _require(boundary_faces_wide.ndim == 1 and
                     boundary_faces_wide.dtype.kind in "iu" and
                     boundary_offset_wide.shape == (self.nb + 1,) and
                     boundary_offset_wide.dtype.kind in "iu" and
                     boundary_offset_wide[0] == 0 and
                     boundary_offset_wide[-1] == len(boundary_faces_wide) and
                     np.all(np.diff(boundary_offset_wide) >= 0) and
                     np.all((0 <= boundary_faces_wide) &
                            (boundary_faces_wide < self.nf)) and
                     len(np.unique(boundary_faces_wide)) ==
                     len(boundary_faces_wide) and
                     np.array_equal(np.sort(boundary_faces_wide),
                                    np.flatnonzero(bids >= 0)) and
                     all(np.all(bids[boundary_faces_wide[
                         boundary_offset_wide[b]:boundary_offset_wide[b + 1]
                     ]] == b) for b in range(self.nb)),
                     "M2-C ordered boundary-face CSR differs")
            boundary_faces = np.ascontiguousarray(
                boundary_faces_wide, dtype=np.int32)
            boundary_offset = np.ascontiguousarray(
                boundary_offset_wide, dtype=np.int32)
        else:
            boundary_faces = (np.concatenate(lists) if lists else
                              np.empty(0, np.int32))
            boundary_offset = np.r_[
                0, np.cumsum([len(v) for v in lists])].astype(np.int32)
        self.boundary_faces = cp.asarray(boundary_faces)
        self.boundary_offset = cp.asarray(boundary_offset)
        self.boundary_face_source_ids_host = None
        if "suspended_amr_contract" in data:
            source_ids = np.asarray(data.get("face_source_ids"))
            _require(source_ids.shape == (self.nf,) and
                     source_ids.dtype.kind in "iu" and
                     np.all(source_ids > 0),
                     "M2-C requires positive stable candidate-face IDs")
            selected = np.ascontiguousarray(
                source_ids[boundary_faces], dtype=np.int64)
            _require(len(np.unique(selected)) == len(selected),
                     "M2-C open boundary candidate-face IDs must be unique")
            self.boundary_face_source_ids_host = selected

        path = Path(source_path or HERE / "flow2dh_boundary_diffusion.cu")
        module = cp.RawModule(code=path.read_text(encoding="utf-8"),
                              options=OPTIONS, name_expressions=KERNELS)
        self.k = {name: module.get_function(name) for name in KERNELS}
        ng, nf, nc, nb = self.ng, self.nf, self.nc, self.nb
        self.accepted_last = cp.zeros((ng, nf), dtype=cp.float64)
        self.accepted_clock = cp.zeros((ng, nf), dtype=cp.float64)
        self.trial_last = cp.zeros_like(self.accepted_last)
        self.trial_clock = cp.zeros_like(self.accepted_clock)
        self.face_concentration = cp.zeros((ng, nf), dtype=cp.float64)
        self.expected_advective_mass_in = cp.zeros((ng, nf), dtype=cp.float64)
        self.expected_advective_ledger = cp.zeros((ng, nb), dtype=cp.float64)
        self.expected_advective_ledger_status = cp.zeros(
            (ng, nb), dtype=cp.int32)
        self.boundary_status = cp.zeros((ng, nf), dtype=cp.int32)
        self.accepted_face_area = cp.zeros(nf, dtype=cp.float64)
        self.transport_face_active = cp.zeros(nf, dtype=cp.int32)
        self.area_status = cp.zeros(nf, dtype=cp.int32)
        self.effective_diffusivity = cp.zeros((ng, nf), dtype=cp.float64)
        self.diffusive_face_mass_owner_out = cp.zeros((ng, nf), dtype=cp.float64)
        self.diffusive_face_active = cp.zeros((ng, nf), dtype=cp.int32)
        self.diffusive_face_status = cp.zeros((ng, nf), dtype=cp.int32)
        self.diffusive_theta = cp.ones((ng, nc), dtype=cp.float64)
        self.diffusive_theta_status = cp.zeros((ng, nc), dtype=cp.int32)
        self.next_mass = cp.zeros((ng, nc), dtype=cp.float64)
        self.diffusive_cell_status = cp.zeros((ng, nc), dtype=cp.int32)
        self.diffusive_ledger = cp.zeros((ng, nb), dtype=cp.float64)
        self.diffusive_ledger_status = cp.zeros((ng, nb), dtype=cp.int32)
        # The generic face kernel retains a per-logical-boundary input in its
        # stable ABI.  M2-B patches every open face from its resolved resident
        # value immediately afterwards, so this allocation is deliberately
        # persistent and never carries an unauthenticated prescription.
        self._generic_boundary_dummy = cp.zeros((ng, nb), dtype=cp.float64)

    def _launch(self, name, n, args):
        if n:
            self.k[name](((n + 127)//128,), (128,), args)

    def _array(self, value, shape, dtype, name):
        _require(isinstance(value, self.cp.ndarray) and value.shape == shape and
                 value.dtype == dtype and value.flags.c_contiguous,
                 f"{name} must be contiguous device {dtype} {shape}")

    def resolve_boundary(self, depth, suspended_mass, prescribed_by_id,
                         accepted_water_flux, dt):
        cp = self.cp
        self._array(depth, (self.nc,), cp.float64, "depth")
        self._array(suspended_mass, (self.ng, self.nc), cp.float64,
                    "suspended_mass")
        self._array(prescribed_by_id, (self.ng, self.nb), cp.float64,
                    "logical-boundary prescribed concentration")
        self._array(accepted_water_flux, (self.nf,), cp.float64,
                    "accepted water flux")
        _require(np.isfinite(dt) and dt > 0.0,
                 "M2-B boundary timestep must be positive and finite")
        self._launch("m2b_resolve_boundary", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            np.int32(self.nb),
            np.float64(self.advection_dry), np.float64(dt), depth,
            suspended_mass, prescribed_by_id, accepted_water_flux, self.length,
            self.owner, self.neighbour, self.bid, self.neumann,
            self.return_time, self.accepted_last, self.accepted_clock,
            self.face_concentration, self.expected_advective_mass_in,
            self.trial_last, self.trial_clock, self.boundary_status))
        self._launch("m2b_expected_advective_ledger", self.ng*self.nb, (
            np.int32(self.nf), np.int32(self.ng), np.int32(self.nb),
            self.boundary_offset, self.boundary_faces,
            self.expected_advective_mass_in, self.boundary_status,
            self.expected_advective_ledger,
            self.expected_advective_ledger_status))
        return dict(concentration=self.face_concentration,
                    expected_advective_mass_in=
                        self.expected_advective_mass_in,
                    expected_advective_ledger=
                        self.expected_advective_ledger,
                    status=(self.boundary_status,
                            self.expected_advective_ledger_status))

    def reconstruct(self, depth, suspended_mass, hydraulic):
        """Run generic kernels with the distinct per-face FLOW boundary input."""
        cp, face, a, b = self.cp, self.faces, self.faces.reconstruction.geometry, self.faces.buffers
        face._check(depth, (self.nc,), "depth", cp.float64)
        face._check(suspended_mass, (self.ng, self.nc), "suspended_mass", cp.float64)
        nc, nf, ng, nadj = map(np.int32, (self.nc, self.nf, self.ng, face.nadj))
        face._launch("sc_primitive", self.nc, (nc, ng, face.dry, depth,
            suspended_mass, b["primitive"], b["primitive_status"]))
        self._launch("m2b_sc_gradients", self.nc, (nc, nf, nadj, ng,
            face.dry, face.grav, face.outlet_extension, b["primitive"],
            hydraulic["primitive"], self.face_concentration, a["xy"], a["mid"],
            a["normal"], a["owner"], a["neighbour"], a["bid"],
            a["boundary_mode"], a["offset"], a["faces"], a["signs"],
            hydraulic["active"], a["hydraulic_ls_matrix"],
            a["hydraulic_ls_weight"], a["bed_ls_delta"], b["grad"], b["vlo"],
            b["vhi"], b["gradient_status"]))
        face._launch("sc_wall_values", self.nc, (nc, ng, b["primitive"],
            a["wall_stencil_ids"], hydraulic["wall_ready"], b["wall_values"]))
        # Generic boundary input is not read after the per-face patch for FCT;
        # use its existing allocation solely to satisfy the stable kernel ABI.
        face._launch("sc_faces", self.nf, (nc, nf, ng, np.int32(self.nb),
            face.dry, depth, b["primitive"], b["grad"], b["wall_values"],
            hydraulic["wall_ready"], self._generic_boundary_dummy,
            a["owner"], a["neighbour"],
            a["bid"], a["wall_slot"], a["weno_cells"],
            a["face_displacement"], b["left"], b["right"], b["face_status"]))
        self._launch("m2b_patch_boundary_faces", self.ng*self.nf, (
            nf, ng, a["neighbour"], a["bid"], self.face_concentration,
            b["right"], b["face_status"]))
        b["status"].fill(0)
        face._launch("sc_status_merge", max(self.nc, self.nf), (nc, nf,
            b["primitive_status"], b["gradient_status"], b["face_status"],
            b["status"]))
        return b

    def advect(self, depth, mass, next_depth, accepted_water_flux, left, right,
               hydro_constant, dt):
        """Run the unchanged generic FCT except for per-face boundary lookup."""
        cp, adv, a, b = self.cp, self.advection, self.advection.geometry, self.advection.buffers
        nc, nf, ng, nb = map(np.int32, (self.nc, self.nf, self.ng, self.nb))
        dt64 = np.float64(dt)
        adv._launch("sa_concentration", self.nc*self.ng, (nc, ng, depth, mass,
            np.float64(adv.dry), b["concentration"], b["concentration_status"]))
        self._launch("m2b_sa_face_low", self.nf*self.ng, (nc, nf, ng,
            accepted_water_flux, b["concentration"], left, right,
            self.face_concentration, a["owner"], a["neighbour"], a["bid"],
            b["concentration_status"], b["low_flux"], b["anti"], b["face_status"]))
        self._launch("m2b_sa_cell_bounds", self.nc*self.ng, (nc, nf, ng, mass,
            next_depth, b["concentration"], self.face_concentration,
            accepted_water_flux, b["low_flux"], b["anti"], a["area"],
            a["length"], a["normal"], a["rectangle_width"], hydro_constant,
            a["weno_cells"], a["owner"], a["neighbour"], a["bid"], a["offset"],
            a["faces"], a["signs"], b["concentration_status"], b["face_status"],
            dt64, b["low_state"], b["rplus"], b["rminus"], b["cell_status"]))
        adv._launch("sa_limit_faces", self.nf*self.ng, (nc, nf, ng,
            b["low_flux"], b["anti"], b["rplus"], b["rminus"], a["owner"],
            a["neighbour"], b["cell_status"], b["scaled_anti"], b["flux"],
            b["flux_status"]))
        adv._launch("sa_update", self.nc*self.ng, (nc, nf, ng, b["low_state"],
            b["scaled_anti"], next_depth, a["length"], a["area"], a["offset"],
            a["faces"], a["signs"], b["cell_status"], b["flux_status"], dt64,
            np.float64(adv.dry), b["next_mass"], b["next_concentration"],
            b["update_status"]))
        adv._launch("sa_boundary_ledger", self.nb*self.ng, (nf, ng, nb,
            a["boundary_offset"], a["boundary_faces"], b["flux"], a["length"],
            b["flux_status"], dt64, b["ledger"], b["ledger_status"]))
        return dict(next_mass=b["next_mass"], next_concentration=b["next_concentration"],
                    flux=b["flux"], ledger=b["ledger"], status=(
                        b["concentration_status"], b["face_status"],
                        b["cell_status"], b["flux_status"], b["update_status"],
                        b["ledger_status"]))

    def face_area_from_trace(self, accepted_trace):
        """Map accepted reconstructed face traces to the BRAIDS exchange area.

        This is the authenticated BRAIDS numerical mapping: arithmetic depth
        from the two accepted reconstructed face traces times face width.  It
        is not represented as source-identical FLOW ``areau``/``areav``.
        """
        self._array(accepted_trace, (10, self.nf), self.cp.float64,
                    "accepted hydraulic trace")
        self._launch("m2b_face_area_from_trace", self.nf, (
            np.int32(self.nf), self.neighbour, self.bid, accepted_trace,
            self.length, self.accepted_face_area, self.transport_face_active,
            self.area_status))
        return (self.accepted_face_area, self.transport_face_active,
                self.area_status)

    def diffuse(self, endpoint_depth, mass_after_advection,
                accepted_face_exchange_area, transport_face_active, dt):
        cp = self.cp
        self._array(endpoint_depth, (self.nc,), cp.float64, "endpoint depth")
        self._array(mass_after_advection, (self.ng, self.nc), cp.float64,
                    "post-advection mass")
        self._array(accepted_face_exchange_area, (self.nf,), cp.float64,
                    "accepted face exchange area")
        self._array(transport_face_active, (self.nf,), cp.int32,
                    "transport face active")
        _require(np.isfinite(dt) and dt > 0.0,
                 "M2-B diffusion timestep must be positive and finite")
        self._launch("m2b_diffusion_faces", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            np.float64(self.dry), np.float64(dt), endpoint_depth,
            mass_after_advection, self.face_concentration, self.dicuv,
            accepted_face_exchange_area, self.centre_distance,
            transport_face_active, self.owner, self.neighbour, self.bid,
            self.effective_diffusivity, self.diffusive_face_mass_owner_out,
            self.diffusive_face_active, self.diffusive_face_status))
        self._launch("m2b_diffusion_theta", self.ng*self.nc, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            mass_after_advection, self.diffusive_face_mass_owner_out,
            self.area, self.offset, self.csr_faces, self.signs,
            self.diffusive_face_status, self.diffusive_theta,
            self.diffusive_theta_status))
        self._launch("m2b_diffusion_limit", self.ng*self.nf, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            self.owner, self.neighbour, self.diffusive_theta,
            self.diffusive_theta_status,
            self.diffusive_face_mass_owner_out,
            self.diffusive_face_status))
        self._launch("m2b_diffusion_update", self.ng*self.nc, (
            np.int32(self.nc), np.int32(self.nf), np.int32(self.ng),
            mass_after_advection, self.diffusive_face_mass_owner_out, self.area,
            self.offset, self.csr_faces, self.signs, self.diffusive_face_status,
            self.next_mass, self.diffusive_cell_status))
        self._launch("m2b_diffusion_ledger", self.ng*self.nb, (
            np.int32(self.nf), np.int32(self.ng), np.int32(self.nb),
            self.boundary_offset, self.boundary_faces,
            self.diffusive_face_mass_owner_out, self.diffusive_face_status,
            self.diffusive_ledger, self.diffusive_ledger_status))
        return dict(next_mass=self.next_mass,
                    face_mass_owner_out=self.diffusive_face_mass_owner_out,
                    effective_diffusivity=self.effective_diffusivity,
                    active=self.diffusive_face_active,
                    donor_theta=self.diffusive_theta,
                    ledger=self.diffusive_ledger,
                    status=(self.diffusive_face_status,
                            self.diffusive_theta_status,
                            self.diffusive_cell_status,
                            self.diffusive_ledger_status))

    def commit_boundary_memory(self):
        self.cp.copyto(self.accepted_last, self.trial_last)
        self.cp.copyto(self.accepted_clock, self.trial_clock)

    def checkpoint_fields(self):
        return dict(
            flow2dh_boundary_last_outflow_concentration=
                self.cp.asnumpy(self.accepted_last),
            flow2dh_boundary_return_clock=self.cp.asnumpy(self.accepted_clock),
        )

    def boundary_memory_fields(self):
        """Return only physical open-face memory in stable boundary CSR order.

        Internal face numbering is topology-local and therefore cannot be
        copied across an AMR regrid.  ``boundary_faces`` is instead built in
        logical-boundary CSR order.  The selected-AMR handoff authenticates
        that those physical faces (owner node, geometry and forcing) are
        unchanged before using this compact representation.
        """
        faces = self.boundary_faces
        _require(self.boundary_face_source_ids_host is not None,
                 "M2-C boundary memory requires stable candidate-face IDs")
        return dict(
            flow2dh_amr_boundary_face_source_ids=
                self.boundary_face_source_ids_host.copy(),
            flow2dh_amr_boundary_last_outflow_concentration=
                self.cp.asnumpy(self.accepted_last[:, faces]),
            flow2dh_amr_boundary_return_clock=
                self.cp.asnumpy(self.accepted_clock[:, faces]),
        )

    def restore_boundary_memory(self, fields):
        """Install authenticated physical-face memory after a topology change.

        Non-boundary slots are scratch-only for this operator and are reset to
        zero.  No nearest-face interpolation or CPU numerical fallback is
        allowed: a different ordered boundary inventory fails closed.
        """
        last = np.asarray(
            fields["flow2dh_amr_boundary_last_outflow_concentration"])
        clock = np.asarray(fields["flow2dh_amr_boundary_return_clock"])
        source_ids = np.asarray(
            fields["flow2dh_amr_boundary_face_source_ids"])
        count = int(self.boundary_faces.size)
        shape = (self.ng, count)
        _require(self.boundary_face_source_ids_host is not None and
                 source_ids.shape == (count,) and
                 source_ids.dtype == np.int64 and
                 np.array_equal(source_ids,
                                self.boundary_face_source_ids_host) and
                 last.shape == clock.shape == shape and
                 last.dtype == clock.dtype == np.float64 and
                 np.isfinite(last).all() and np.isfinite(clock).all() and
                 np.all(last >= 0.0) and np.all(clock >= 0.0),
                 "M2-C ordered boundary-memory handoff differs")
        boundary = np.concatenate([
            np.arange(self.boundary_offset.get()[b],
                      self.boundary_offset.get()[b + 1], dtype=np.int64)
            for b in range(self.nb)
        ]) if self.nb else np.empty(0, dtype=np.int64)
        _require(np.array_equal(boundary, np.arange(count, dtype=np.int64)),
                 "M2-C boundary CSR memory order differs")
        return_time = self.return_time_host[:, self.cp.asnumpy(
            self.boundary_faces).astype(np.int64, copy=False)]
        positive = return_time > 0.0
        tolerance = (256.0 * np.finfo(np.float64).eps *
                     np.maximum(1.0, return_time))
        _require(np.all(clock[positive] <=
                        return_time[positive] + tolerance[positive]) and
                 np.all(clock[~positive] == 0.0),
                 "M2-C boundary-memory clock is outside its contract")
        self.accepted_last.fill(0.0)
        self.accepted_clock.fill(0.0)
        self.accepted_last[:, self.boundary_faces] = self.cp.asarray(last)
        self.accepted_clock[:, self.boundary_faces] = self.cp.asarray(clock)
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)

    def restore_checkpoint(self, fields):
        last = np.asarray(fields["flow2dh_boundary_last_outflow_concentration"])
        clock = np.asarray(fields["flow2dh_boundary_return_clock"])
        shape = (self.ng, self.nf)
        _require(last.shape == clock.shape == shape and
                 last.dtype == clock.dtype == np.float64 and
                 np.isfinite(last).all() and np.isfinite(clock).all() and
                 np.all(last >= 0.0) and np.all(clock >= 0.0),
                 "M2-B boundary checkpoint memory differs")
        positive = self.return_time_host > 0.0
        tolerance = (256.0 * np.finfo(np.float64).eps *
                     np.maximum(1.0, self.return_time_host))
        _require(np.all(clock[positive] <=
                        self.return_time_host[positive] + tolerance[positive]) and
                 np.all(clock[~positive] == 0.0),
                 "M2-B boundary checkpoint return clock is outside its contract")
        self.cp.copyto(self.accepted_last, self.cp.asarray(last))
        self.cp.copyto(self.accepted_clock, self.cp.asarray(clock))
        self.cp.copyto(self.trial_last, self.accepted_last)
        self.cp.copyto(self.trial_clock, self.accepted_clock)


__all__ = [
    "BOUNDARY_EXTERIOR_DEPTH_PROVIDER", "BOUNDARY_NUMERICAL_ID",
    "DIFFUSION_NUMERICAL_ID", "FACE_AREA_PROVIDER",
    "ResidentFlow2DHBoundaryDiffusion", "method_identity",
]
