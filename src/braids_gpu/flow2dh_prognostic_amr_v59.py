"""Resident-CUDA Q3-H3/v59 synchronous selected-mesh AMR component.

This is an isolated component gate, not a live solver or QGIS route.  It owns
the first Q3-H3 AMR slice: conservative signed ``hI`` family transfer,
canonical fine-segment signed advection, one composite backward-Euler scalar
diffusion solve, frozen accepted-start ``Betac`` segment fluxes, and exact
Coriolis rotation on active leaves.  Candidate state, topology, registers and
ledgers stay on the explicitly selected CUDA device until an atomic commit.

Recursive implicit subcycling, open boundaries, wet/dry cells, the hydraulic
controller, sediment/Exner work, restart serialization and production
admission are intentionally rejected/not claimed here.
"""
from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path

import numpy as np

from braids_core.sediment.prognostic_secondary_amr_execution import (
    EXECUTION_KEY as AMR_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as AMR_ORDERED_ROUTE_KEY,
    canonical_amr_instance_fingerprints,
    validate_prognostic_secondary_amr_execution_document,
    validate_prognostic_secondary_amr_ordered_route,
)
from braids_core.sediment.prognostic_secondary_nonzero_execution import (
    EXECUTION_KEY as NONZERO_EXECUTION_KEY,
    validate_prognostic_secondary_nonzero_execution_document,
)
from .flow2dh_secondary_operator import CUDA_OPTIONS


ENVELOPE_VERSION = 59
SCOPE_ID = "q3h3-classic-flow2dh-prognostic-amr-gpu-component-v1"
INTEGRATOR = "implicit_fwave_midpoint"
TIME_COUPLING = "synchronous_selected_mesh_one_clock_q3h_v1"
SCHMIDT_NUMBER = 0.7
STATUS_OK = 0
STATUS_INVALID_INPUT = 1
STATUS_INVALID_GEOMETRY = 2
STATUS_INVALID_LINEAR_SYSTEM = 3
STATUS_NOT_CONVERGED = 4
STATUS_CANCELLED = 5
STATUS_INVALID_OUTPUT = 6
STATUS_INVALID_TOPOLOGY = 7

CANONICAL_FIXTURE_SHA256 = (
    "c8431cabacc4ab09e13901bc656f1e99efa3999568728f0f8c99d63405d0588f"
)
SOURCE_CONTRACT_SHA256 = (
    "19d80d7e9f0e5fc07cc5a5bf09a5f22a07c42fa6e0bc969db827191efdd19df2"
)
BETAC_OPERATOR_AMENDMENT_SHA256 = (
    "afe7ee466e381efb37bbbcdcf40cd8d1528c54c3c51443b86a6ec086f7f0f034"
)
# Compile behavior is part of the hashed v59 wrapper rather than an unbound
# transitive import.  A change in the shared CUDA foundation therefore fails
# closed until this component is reviewed and its own source identity moves.
V59_CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
CUDA_SOURCE_SHA256 = (
    "f8cafd724c2cbd3a2064eaa142bdba9a87c82ed2e11c905ccfc79a414d2596fb"
)
PARENT_HASHES = {
    "flow2dh_prognostic_nonzero_v58.py":
        "57ed65517316f3990fe1e496ab45de4d033be0a9072f18f7c7151cce224659ae",
    "flow2dh_prognostic_nonzero_v58.cu":
        "14de487b7a9c212a157f2849ac553e1424554a541b2c1cd69be8360c6814c7f5",
    "implicit_midpoint_fwave_nonzero_v58.py":
        "4ff5aea41efeca75b17aa8af9c316e6fa4b191da27377e647eff35b5965270b9",
}


def _poison_open_trial(method):
    """Make every exceptional operator/commit rejection terminal."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            if getattr(self, "_trial_open", False):
                self._last_status = max(
                    getattr(self, "_last_status", STATUS_OK),
                    STATUS_INVALID_INPUT,
                )
                self._trial_rejected = True
            raise
    return guarded

KERNEL_NAMES = (
    "v59_restrict_family", "v59_prolong_family",
    "v59_advective_segments", "v59_apply_mass_increment",
    "v59_gather_scalar_register", "v59_gather_vector_register",
    "v59_diffusion_conductance", "v59_diffusion_setup",
    "v59_diffusion_matvec", "v59_pcg_initialize_vectors",
    "v59_pcg_initialize_control", "v59_pcg_alpha",
    "v59_pcg_update_xr", "v59_pcg_precondition",
    "v59_pcg_convergence_beta", "v59_pcg_update_p",
    "v59_diffusion_finish", "v59_diffusion_register",
    "v59_betac_cells", "v59_betac_segments",
    "v59_coriolis_active_leaves",
    "v59_reduce_vector_cell_increment_ordered",
)


def _scalar(data: dict, name: str, default=None):
    if name not in data:
        if default is None:
            raise ValueError("Q3-H3/v59 prepared input lacks " + name)
        return default
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError("Q3-H3/v59 control must be scalar: " + name)
    return value.item()


def _strict_bool_scalar(data: dict, name: str, default=None) -> bool:
    value = _scalar(data, name, default)
    if not isinstance(value, bool):
        raise ValueError("Q3-H3/v59 control must be Boolean: " + name)
    return value


def _strict_int_scalar(data: dict, name: str, default=None) -> int:
    value = _scalar(data, name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Q3-H3/v59 control must be an integer: " + name)
    return value


def _numeric_scalar(data: dict, name: str, default=None) -> float:
    value = _scalar(data, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Q3-H3/v59 control must be numeric: " + name)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Q3-H3/v59 control must be finite: " + name)
    return result


def _string_scalar(data: dict, name: str) -> str:
    value = _scalar(data, name)
    if not isinstance(value, str) or not value:
        raise ValueError("Q3-H3/v59 control must be a nonempty string: " + name)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_parent_sources() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    actual = {}
    for name, expected in PARENT_HASHES.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Q3-H3/v59 accepted v58 parent is missing or linked")
        actual[name] = _sha(path)
        if actual[name] != expected:
            raise ValueError("Q3-H3/v59 accepted v58 parent identity differs: " + name)
    return actual


def _bind_design_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    amr = (root / "braids_support/sediment/delft3d_current/q3h_integration/"
           "h3_amr")
    paths = (
        (amr / "source_contract.json", SOURCE_CONTRACT_SHA256),
        (amr / "fixtures/canonical_selected_mesh_v1.json",
         CANONICAL_FIXTURE_SHA256),
        (amr / "betac_operator_amendment_v1.json",
         BETAC_OPERATOR_AMENDMENT_SHA256),
    )
    for path, expected in paths:
        if not path.is_file() or path.is_symlink() or _sha(path) != expected:
            raise ValueError("Q3-H3/v59 design source identity differs: " + path.name)


def _host_array(data, name, dtype, shape):
    raw = np.asarray(data[name])
    expected_dtype = np.dtype(dtype)
    if raw.shape != shape or raw.dtype != expected_dtype:
        raise ValueError("Q3-H3/v59 prepared array differs: " + name)
    value = np.ascontiguousarray(raw)
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError("Q3-H3/v59 prepared array is nonfinite: " + name)
    return value


class ResidentPrognosticSecondaryAMRV59:
    """CUDA-resident transaction for the bounded synchronous AMR operators."""

    def __init__(self, data: dict, *, source_path: str | Path | None = None):
        if not isinstance(data, dict):
            raise TypeError("Q3-H3/v59 prepared input must be a mapping")
        if _string_scalar(data, "execution_backend") != "gpu":
            raise ValueError("Q3-H3/v59 backend must be explicitly GPU")
        if _string_scalar(data, "gpu_integrator") != INTEGRATOR:
            raise ValueError("Q3-H3/v59 requires implicit_fwave_midpoint")
        if not _strict_bool_scalar(data, "amr_enabled"):
            raise ValueError("Q3-H3/v59 requires an explicit AMR selection")
        if _string_scalar(data, "amr_time_coupling") != TIME_COUPLING:
            raise ValueError("Q3-H3/v59 supports one synchronous AMR clock only")
        if _strict_bool_scalar(data, "recursive_subcycling", False):
            raise ValueError("Q3-H3/v59 recursive implicit AMR is unsupported")
        if not _strict_bool_scalar(data, "fully_wet_scope"):
            raise ValueError("Q3-H3/v59 first slice is fully wet only")
        requested_device_id = _strict_int_scalar(data, "cuda_device_id")
        if requested_device_id < 0:
            raise ValueError(
                "Q3-H3/v59 requires an explicit nonnegative CUDA device ID")
        self.requested_device_id = requested_device_id

        _bind_design_sources()
        self.parent_hashes = _bind_parent_sources()
        root = Path(__file__).resolve().parents[1]
        fixture_path = (root / "braids_support/sediment/delft3d_current/"
                        "q3h_integration/h3_amr/fixtures/"
                        "canonical_selected_mesh_v1.json")
        import json
        canonical_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        canonical_topology = {
            "cells": canonical_fixture["composite_cells"],
            "family": canonical_fixture["transfer_family"],
        }
        canonical_faces = canonical_fixture["canonical_internal_segments"]
        topology_document = data.get("amr_topology_document")
        face_document = data.get("amr_face_catalog_document")
        boundary_document = data.get("amr_boundary_catalog_document")
        if (topology_document != canonical_topology or
                face_document != canonical_faces or boundary_document != []):
            raise ValueError(
                "Q3-H3/v59 bounded canonical instance documents differ")
        instance = canonical_amr_instance_fingerprints(
            topology_document=topology_document,
            face_catalog_document=face_document,
            boundary_catalog_document=boundary_document)
        self.parent_identity = \
            validate_prognostic_secondary_nonzero_execution_document(
                data.get(NONZERO_EXECUTION_KEY))
        self.wrapper_sha256 = _sha(Path(__file__).resolve())
        self.identity = validate_prognostic_secondary_amr_execution_document(
            data.get(AMR_EXECUTION_KEY), parent=self.parent_identity,
            source_contract_sha256=SOURCE_CONTRACT_SHA256,
            canonical_fixture_sha256=CANONICAL_FIXTURE_SHA256,
            betac_operator_amendment_sha256=
                BETAC_OPERATOR_AMENDMENT_SHA256,
            implementation_source_sha256=self.wrapper_sha256,
            **instance)
        validate_prognostic_secondary_amr_ordered_route(
            data.get(AMR_ORDERED_ROUTE_KEY), self.identity,
            self.parent_identity)
        if (self.identity["backend"] != "gpu" or
                self.identity["integrator"] != INTEGRATOR or
                self.identity["cuda_device_id"] != requested_device_id or
                self.identity["time_coupling"] != TIME_COUPLING or
                self.identity["automatic_cpu_fallback"] is not False):
            raise ValueError("Q3-H3/v59 shared execution identity differs")

        self.nc = _strict_int_scalar(data, "nc")
        self.nf = _strict_int_scalar(data, "nf")
        self.nfamily = _strict_int_scalar(data, "nfamily", 0)
        self.component_case = _string_scalar(data, "component_case")
        self.dry = _numeric_scalar(data, "dry")
        self.diffusivity = _numeric_scalar(data, "scalar_diffusivity_m2_s")
        self.coriolis = _numeric_scalar(
            data, "coriolis_constant_s_inverse", 0.0)
        self.betac = _numeric_scalar(data, "betac", 0.0)
        self.gravity = _numeric_scalar(data, "gravity", 9.81)
        self.von_karman = _numeric_scalar(data, "von_karman", 0.41)
        self.chzmin = _numeric_scalar(data, "chzmin", 20.0)
        self.relative_tolerance = _numeric_scalar(
            data, "relative_linear_tolerance", 1.0e-13)
        self.absolute_tolerance = _numeric_scalar(
            data, "absolute_linear_tolerance", 1.0e-14)
        self.maximum_iterations = _strict_int_scalar(
            data, "maximum_linear_iterations", 256)
        self.conservation_tolerance = float(
            self.identity["conservation_tolerance"])
        numeric = np.asarray((
            self.dry, self.diffusivity, self.coriolis, self.betac,
            self.gravity, self.von_karman, self.chzmin,
            self.relative_tolerance, self.absolute_tolerance,
            self.conservation_tolerance), dtype=np.float64)
        if (self.nc <= 0 or self.nf < 0 or self.nfamily < 0 or
                not np.isfinite(numeric).all() or self.dry <= 0.0 or
                self.diffusivity < 0.0 or self.gravity <= 0.0 or
                self.von_karman <= 0.0 or self.chzmin <= 0.0 or
                self.relative_tolerance < 0.0 or self.absolute_tolerance < 0.0 or
                self.conservation_tolerance <= 0.0 or
                isinstance(self.maximum_iterations, bool) or
                self.maximum_iterations < 1):
            raise ValueError("Q3-H3/v59 numerical controls are invalid")

        self.fixture_sha256 = _string_scalar(data, "fixture_sha256")
        if self.fixture_sha256 != CANONICAL_FIXTURE_SHA256:
            raise ValueError("Q3-H3/v59 canonical fixture identity differs")
        if (self.parent_identity["coriolis_constant_s_inverse"] != self.coriolis or
                self.parent_identity["betac"] != self.betac or
                self.parent_identity["scalar_diffusivity_m2_s"] !=
                    self.diffusivity or
                self.parent_identity["chzmin"] != self.chzmin or
                self.parent_identity["von_karman_constant"] != self.von_karman or
                self.identity["dry_tolerance_m"] != self.dry or
                self.identity["gravity_m_s2"] != self.gravity or
                self.identity["von_karman_constant"] != self.von_karman or
                self.identity["chzmin"] != self.chzmin or
                self.identity["relative_linear_tolerance"] !=
                    self.relative_tolerance or
                self.identity["absolute_linear_tolerance"] !=
                    self.absolute_tolerance or
                self.identity["maximum_linear_iterations"] !=
                    self.maximum_iterations):
            raise ValueError("Q3-H3/v59 physical controls differ from identity")
        fixture_controls = canonical_fixture["operator_controls"]
        if (self.relative_tolerance !=
                fixture_controls["relative_linear_tolerance"] or
                self.absolute_tolerance !=
                fixture_controls["absolute_linear_tolerance"] or
                self.maximum_iterations !=
                fixture_controls["maximum_linear_iterations"] or
                self.diffusivity !=
                fixture_controls["scalar_diffusivity_m2_s"] or
                self.dry != fixture_controls["dry_tolerance_m"] or
                fixture_controls["schmidt_number"] != SCHMIDT_NUMBER):
            raise ValueError("Q3-H3/v59 canonical numerical controls differ")

        cell_id = _host_array(data, "cell_id", np.int64, (self.nc,))
        level = _host_array(data, "level", np.int32, (self.nc,))
        area = _host_array(data, "area", np.float64, (self.nc,))
        centre = _host_array(data, "centre", np.float64, (2, self.nc))
        if (np.unique(cell_id).size != self.nc or np.any(level < 0) or
                np.any(area <= 0.0)):
            raise ValueError("Q3-H3/v59 cell catalog is invalid")
        if int(np.max(level)) > int(self.identity["maximum_level"]):
            raise ValueError("Q3-H3/v59 selected level exceeds its identity")

        owner = _host_array(data, "owner", np.int32, (self.nf,))
        neighbour = _host_array(data, "neighbour", np.int32, (self.nf,))
        source_id = _host_array(data, "source_id", np.int64, (self.nf,))
        segment_level = _host_array(
            data, "segment_level", np.int32, (self.nf,))
        length = _host_array(data, "length", np.float64, (self.nf,))
        normal = _host_array(data, "normal", np.float64, (2, self.nf))
        midpoint = _host_array(data, "midpoint", np.float64, (2, self.nf))
        distance = _host_array(data, "distance", np.float64, (self.nf,))
        if (np.unique(source_id).size != self.nf or np.any(source_id <= 0) or
                (self.nf > 1 and np.any(np.diff(source_id) <= 0)) or
                np.any(owner < 0) or
                np.any(owner >= self.nc) or np.any(neighbour < 0) or
                np.any(neighbour >= self.nc) or np.any(owner == neighbour) or
                np.any(length <= 0.0) or np.any(distance <= 0.0)):
            raise ValueError(
                "Q3-H3/v59 requires unique closed canonical face segments")
        if (np.any(np.abs(level[owner] - level[neighbour]) > 1) or
                np.any(segment_level != np.maximum(
                    level[owner], level[neighbour]))):
            raise ValueError(
                "Q3-H3/v59 face endpoint levels violate the 2:1 catalog")
        normal_norm = np.sqrt(np.sum(normal * normal, axis=0))
        if not np.allclose(normal_norm, 1.0, rtol=0.0, atol=2.0e-14):
            raise ValueError("Q3-H3/v59 segment normals are not unit vectors")

        family_parent = _host_array(
            data, "family_parent", np.int32, (self.nfamily,))
        child_offset = _host_array(
            data, "child_offset", np.int32, (self.nfamily + 1,))
        raw_children = np.asarray(data.get("children", []))
        if raw_children.ndim != 1 or raw_children.dtype != np.dtype(np.int32):
            raise ValueError("Q3-H3/v59 prepared array differs: children")
        children = np.ascontiguousarray(raw_children)
        if (children.ndim != 1 or child_offset[0] != 0 or
                child_offset[-1] != children.size or
                np.any(np.diff(child_offset) != 4) or
                np.any(family_parent < 0) or np.any(family_parent >= self.nc) or
                np.any(children < 0) or np.any(children >= self.nc)):
            raise ValueError("Q3-H3/v59 family catalog is incomplete")
        seen_children = set()
        for family in range(self.nfamily):
            row = children[child_offset[family]:child_offset[family + 1]]
            if (np.unique(row).size != 4 or family_parent[family] in row or
                    any(int(item) in seen_children for item in row)):
                raise ValueError("Q3-H3/v59 family catalog overlaps")
            seen_children.update(map(int, row))
            if not np.isclose(
                    area[family_parent[family]], np.sum(area[row]),
                    rtol=2.0e-13, atol=2.0e-13):
                raise ValueError("Q3-H3/v59 family areas do not close")

        if self.component_case == "canonical_transfer_family_v1":
            transfer = canonical_fixture["transfer_family"]
            expected_area = np.r_[transfer["parent_area_m2"],
                                  transfer["child_area_m2"]]
            expected_centre = np.vstack((
                np.r_[0.0, np.asarray(transfer["child_offset_m"])[:, 0]],
                np.r_[0.0, np.asarray(transfer["child_offset_m"])[:, 1]]))
            if (self.nc != 5 or self.nf != 0 or self.nfamily != 1 or
                    not np.array_equal(level, np.asarray(
                        [0, 1, 1, 1, 1], dtype=np.int32)) or
                    not np.array_equal(area, expected_area) or
                    not np.array_equal(centre, expected_centre) or
                    not np.array_equal(family_parent, [0]) or
                    not np.array_equal(child_offset, [0, 4]) or
                    not np.array_equal(children, [1, 2, 3, 4])):
                raise ValueError("Q3-H3/v59 canonical transfer geometry differs")
        elif self.component_case == "canonical_composite_segments_v1":
            cells = canonical_fixture["composite_cells"]
            segments = canonical_fixture["canonical_internal_segments"]
            by_id = {int(value): index for index, value in enumerate(
                cells["cell_id"])}
            expected_owner = np.asarray([by_id[int(value)] for value in
                                         segments["owner_cell_id"]],
                                        dtype=np.int32)
            expected_neighbour = np.asarray([by_id[int(value)] for value in
                                             segments["neighbour_cell_id"]],
                                            dtype=np.int32)
            checks = (
                self.nc == 3, self.nf == 2, self.nfamily == 0,
                np.array_equal(cell_id, cells["cell_id"]),
                np.array_equal(level, cells["level"]),
                np.array_equal(area, cells["area_m2"]),
                np.array_equal(centre, np.asarray(cells["centre_m"]).T),
                np.array_equal(owner, expected_owner),
                np.array_equal(neighbour, expected_neighbour),
                np.array_equal(source_id, segments["source_id"]),
                np.array_equal(segment_level, np.maximum(
                    level[expected_owner], level[expected_neighbour])),
                np.array_equal(length, segments["length_m"]),
                np.array_equal(normal, np.asarray(
                    segments["owner_to_neighbour_normal"]).T),
                np.array_equal(midpoint, np.asarray(segments["midpoint_m"]).T),
                np.array_equal(distance,
                               segments["normal_projected_centre_distance_m"]),
                family_parent.size == 0, children.size == 0,
            )
            if not all(checks):
                raise ValueError("Q3-H3/v59 canonical composite geometry differs")
        else:
            raise ValueError("Q3-H3/v59 component case is unsupported")

        # Build deterministic full-catalog CSR over the canonical segments.
        incidence = [[] for _ in range(self.nc)]
        for face, (left, right) in enumerate(zip(owner, neighbour)):
            incidence[int(left)].append(face);incidence[int(right)].append(face)
        offset = np.zeros(self.nc + 1, dtype=np.int32)
        for cell, row in enumerate(incidence):
            offset[cell + 1] = offset[cell] + len(row)
        faces = np.ascontiguousarray(
            [face for row in incidence for face in row], dtype=np.int32)

        digest = hashlib.sha256()
        digest.update(b"q3h3-v59-synchronous-selected-mesh-v1\0")
        for name, value in (
                ("cell_id", cell_id), ("level", level), ("area", area),
                ("centre", centre), ("owner", owner),
                ("neighbour", neighbour), ("source_id", source_id),
                ("segment_level", segment_level),
                ("length", length), ("normal", normal),
                ("midpoint", midpoint), ("distance", distance),
                ("family_parent", family_parent),
                ("child_offset", child_offset), ("children", children)):
            digest.update(name.encode("ascii") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes(order="C"))
        digest.update(self.fixture_sha256.encode("ascii"))
        self.mesh_fingerprint = digest.hexdigest()

        source = Path(source_path or Path(__file__).with_suffix(".cu")).resolve()
        if not source.is_file() or source.is_symlink():
            raise ValueError("Q3-H3/v59 CUDA source is missing or linked")
        self.source_path = source
        self.source_sha256 = _sha(source)
        if self.source_sha256 != CUDA_SOURCE_SHA256:
            raise ValueError("Q3-H3/v59 CUDA source identity differs")
        if tuple(CUDA_OPTIONS) != V59_CUDA_OPTIONS:
            raise ValueError("Q3-H3/v59 CUDA compile options identity differs")

        # CUDA is imported only after every host identity/geometry check.
        import cupy as cp
        self.cp = cp
        if requested_device_id >= int(cp.cuda.runtime.getDeviceCount()):
            raise ValueError("Q3-H3/v59 requested CUDA device is unavailable")
        self.device_id = requested_device_id
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source.read_text(encoding="utf-8"),
                options=V59_CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.module = module
            self.kernels = {
                name: module.get_function(name) for name in KERNEL_NAMES
            }
            self.cell_id = cp.asarray(cell_id)
            self.level = cp.asarray(level)
            self.area = cp.asarray(area)
            self.centre = cp.asarray(centre)
            self.owner = cp.asarray(owner)
            self.neighbour = cp.asarray(neighbour)
            self.source_id = cp.asarray(source_id)
            self.segment_level = cp.asarray(segment_level)
            self.length = cp.asarray(length)
            self.normal = cp.asarray(normal)
            self.midpoint = cp.asarray(midpoint)
            self.distance = cp.asarray(distance)
            self.family_parent = cp.asarray(family_parent)
            self.child_offset = cp.asarray(child_offset)
            self.children = cp.asarray(children)
            self.offset = cp.asarray(offset)
            self.faces = cp.asarray(faces)

            self.accepted_q_snapshot = cp.zeros((3, self.nc), dtype=cp.float64)
            self.accepted_mass_snapshot = cp.zeros(self.nc, dtype=cp.float64)
            self.accepted_active_snapshot = cp.zeros(self.nc, dtype=cp.int32)
            self.q_trial = cp.zeros((3, self.nc), dtype=cp.float64)
            self.mass_trial = cp.zeros(self.nc, dtype=cp.float64)
            self.active_trial = cp.zeros(self.nc, dtype=cp.int32)
            self.status_cell = cp.zeros(self.nc, dtype=cp.int32)
            self.status_face = cp.zeros(self.nf, dtype=cp.int32)
            self.status_one = cp.zeros(1, dtype=cp.int32)
            self.cancel_flag = cp.zeros(1, dtype=cp.int32)

            self.advective_register = cp.zeros(self.nf, dtype=cp.float64)
            self.diffusive_register = cp.zeros(self.nf, dtype=cp.float64)
            self.betac_flux = cp.zeros((2, self.nf), dtype=cp.float64)
            self.betac_register = cp.zeros((2, self.nf), dtype=cp.float64)
            self.advective_cell_increment = cp.zeros(self.nc, dtype=cp.float64)
            self.diffusive_cell_increment = cp.zeros(self.nc, dtype=cp.float64)
            self.betac_cell_increment = cp.zeros((2, self.nc), dtype=cp.float64)
            self.coriolis_cell_increment = cp.zeros((2, self.nc), dtype=cp.float64)
            self.coriolis_ledger = cp.zeros(2, dtype=cp.float64)
            self.tensor = cp.zeros((3, self.nc), dtype=cp.float64)

            self.accepted_advective_register = cp.zeros_like(self.advective_register)
            self.accepted_diffusive_register = cp.zeros_like(self.diffusive_register)
            self.accepted_betac_register = cp.zeros_like(self.betac_register)
            self.accepted_coriolis_ledger = cp.zeros_like(self.coriolis_ledger)

            self.conductance = cp.zeros(self.nf, dtype=cp.float64)
            self.diffusion_input = cp.zeros(self.nc, dtype=cp.float64)
            self.diffusion_candidate = cp.zeros(self.nc, dtype=cp.float64)
            self.x = cp.zeros(self.nc, dtype=cp.float64)
            self.rhs = cp.zeros(self.nc, dtype=cp.float64)
            self.diagonal = cp.ones(self.nc, dtype=cp.float64)
            self.r = cp.zeros(self.nc, dtype=cp.float64)
            self.z = cp.zeros(self.nc, dtype=cp.float64)
            self.p = cp.zeros(self.nc, dtype=cp.float64)
            self.ap = cp.zeros(self.nc, dtype=cp.float64)
            self.product = cp.zeros(self.nc, dtype=cp.float64)
            self.pcg_scalars = cp.zeros(6, dtype=cp.float64)
            self.pcg_control = cp.zeros(4, dtype=cp.int32)
            self.rhs2 = cp.zeros((), dtype=cp.float64)
            self.rr = cp.zeros((), dtype=cp.float64)
            self.rz = cp.zeros((), dtype=cp.float64)
            self.denominator = cp.zeros((), dtype=cp.float64)
            self.rz_new = cp.zeros((), dtype=cp.float64)

        self._trial_open = False
        self._trial_rejected = False
        self._completed_phases: set[str] = set()
        self._phase_rank = 0
        self._last_status = STATUS_OK
        self._have_accepted_state = False
        self._accepted_ptrs = None
        self.accepted_topology_generation = 0
        self.trial_topology_generation = 0
        self.accepted_predictor_cache_valid = True
        self.trial_predictor_cache_valid = True
        self._last_diffusion = {}

    def _launch(self, name, count, args):
        self._require_bound_device()
        if count <= 0:
            return
        block = 128
        self.kernels[name](((count + block - 1) // block,), (block,), args)

    def _require_trial(self):
        self._require_bound_device()
        if not self._trial_open:
            raise RuntimeError("Q3-H3/v59 has no open transaction")

    def _require_bound_device(self):
        if int(self.cp.cuda.runtime.getDevice()) != self.device_id:
            raise RuntimeError("Q3-H3/v59 CUDA device context differs")

    def _fp64(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError(name + " must be resident contiguous CUDA FP64")

    def _int32(self, name, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.int32 or
                value.shape != shape or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise ValueError(name + " must be resident contiguous CUDA int32")

    def _status(self, array) -> int:
        return int(self.cp.max(array).item()) if array.size else STATUS_OK

    @staticmethod
    def _memory_interval(value):
        begin = int(value.data.ptr)
        return begin, begin + int(value.nbytes)

    @classmethod
    def _memory_overlaps(cls, left, right) -> bool:
        left_begin, left_end = cls._memory_interval(left)
        right_begin, right_end = cls._memory_interval(right)
        return (left_end > left_begin and right_end > right_begin and
                left_begin < right_end and right_begin < left_end)

    def _resident_arrays(self):
        cp = self.cp
        return tuple(value for value in vars(self).values()
                     if isinstance(value, cp.ndarray))

    def _validate_external_state_storage(self, accepted_q, accepted_mass,
                                         accepted_active):
        external = (
            ("accepted hydraulics", accepted_q),
            ("accepted signed hI", accepted_mass),
            ("accepted active leaves", accepted_active),
        )
        for index, (left_name, left) in enumerate(external):
            for right_name, right in external[index + 1:]:
                if self._memory_overlaps(left, right):
                    raise ValueError(
                        "Q3-H3/v59 external accepted arrays overlap: "
                        f"{left_name} and {right_name}")
        resident = self._resident_arrays()
        for name, value in external:
            if any(self._memory_overlaps(value, internal)
                   for internal in resident):
                raise ValueError(
                    f"Q3-H3/v59 {name} aliases private resident storage")

    def _validate_active_ownership(self, active, *, accepted):
        mask = self.cp.asnumpy(active)
        children = self.cp.asnumpy(self.children)
        offsets = self.cp.asnumpy(self.child_offset)
        parents = self.cp.asnumpy(self.family_parent)
        for family, parent in enumerate(parents):
            row = children[offsets[family]:offsets[family + 1]]
            coarse = mask[parent] == 1 and np.all(mask[row] == 0)
            refined = mask[parent] == 0 and np.all(mask[row] == 1)
            if not (coarse or refined):
                kind = "accepted" if accepted else "candidate"
                raise ValueError(
                    f"Q3-H3/v59 {kind} family ownership overlaps or is incomplete")
        if self.nf and (np.any(mask[self.cp.asnumpy(self.owner)] != 1) or
                        np.any(mask[self.cp.asnumpy(self.neighbour)] != 1)):
            kind = "accepted" if accepted else "candidate"
            raise ValueError(
                f"Q3-H3/v59 {kind} face catalog has an inactive endpoint")

    def _enter_component_phase(self, name: str, rank: int, *, composite_only=False):
        if composite_only and self.component_case != "canonical_composite_segments_v1":
            raise RuntimeError(
                f"Q3-H3/v59 {name} is outside the selected component case")
        if name in self._completed_phases:
            raise RuntimeError(
                f"Q3-H3/v59 component phase {name} may execute only once")
        if rank <= self._phase_rank:
            raise RuntimeError(
                f"Q3-H3/v59 component phase {name} is out of order")
        self._completed_phases.add(name)
        self._phase_rank = rank

    def begin_trial(self, accepted_q, accepted_mass, accepted_active):
        self._require_bound_device()
        if self._trial_open:
            raise RuntimeError("Q3-H3/v59 transaction is already open")
        self._fp64("accepted hydraulics", accepted_q, (3, self.nc))
        self._fp64("accepted signed hI", accepted_mass, (self.nc,))
        self._int32("accepted active leaves", accepted_active, (self.nc,))
        self._validate_external_state_storage(
            accepted_q, accepted_mass, accepted_active)
        if (bool(self.cp.any(~self.cp.isfinite(accepted_q)).item()) or
                bool(self.cp.any(~self.cp.isfinite(accepted_mass)).item()) or
                bool(self.cp.any((accepted_active != 0) &
                                 (accepted_active != 1)).item()) or
                bool(self.cp.any((accepted_active == 1) &
                                 (accepted_q[0] <= self.dry)).item())):
            raise ValueError("Q3-H3/v59 accepted state is not finite fully wet")
        self._validate_active_ownership(accepted_active, accepted=True)
        self.cp.copyto(self.accepted_q_snapshot, accepted_q)
        self.cp.copyto(self.accepted_mass_snapshot, accepted_mass)
        self.cp.copyto(self.accepted_active_snapshot, accepted_active)
        self.cp.copyto(self.q_trial, accepted_q)
        self.cp.copyto(self.mass_trial, accepted_mass)
        self.cp.copyto(self.active_trial, accepted_active)
        self.advective_register.fill(0.0)
        self.diffusive_register.fill(0.0)
        self.betac_flux.fill(0.0);self.betac_register.fill(0.0)
        self.advective_cell_increment.fill(0.0)
        self.diffusive_cell_increment.fill(0.0)
        self.betac_cell_increment.fill(0.0)
        self.coriolis_cell_increment.fill(0.0);self.coriolis_ledger.fill(0.0)
        self.cancel_flag.fill(0);self._last_status = STATUS_OK
        self._trial_rejected = False
        self._completed_phases.clear();self._phase_rank = 0
        self.trial_topology_generation = self.accepted_topology_generation
        self.trial_predictor_cache_valid = self.accepted_predictor_cache_valid
        self._accepted_ptrs = (
            int(accepted_q.data.ptr), int(accepted_mass.data.ptr),
            int(accepted_active.data.ptr))
        self._trial_open = True
        return self.q_trial, self.mass_trial, self.active_trial

    @_poison_open_trial
    def propose_coarsen(self, family: int):
        self._require_trial()
        if isinstance(family, bool) or not isinstance(family, int) or not (
                0 <= family < self.nfamily):
            raise ValueError("Q3-H3/v59 family index differs")
        self.status_one.fill(0)
        self._launch("v59_restrict_family", 1, (
            np.int32(self.nc), np.int32(family), self.family_parent,
            self.child_offset, self.children, self.area, self.q_trial,
            self.mass_trial, self.active_trial, np.float64(self.dry),
            np.int32(1), np.int32(1),
            self.status_one))
        status = self._status(self.status_one)
        if status == STATUS_OK:
            self.trial_topology_generation += 1
            self.trial_predictor_cache_valid = False
        self._last_status = max(self._last_status, status)
        return status

    @_poison_open_trial
    def synchronize_parent_cache(self, family: int):
        self._require_trial()
        if isinstance(family, bool) or not isinstance(family, int) or not (
                0 <= family < self.nfamily):
            raise ValueError("Q3-H3/v59 family index differs")
        self.status_one.fill(0)
        self._launch("v59_restrict_family", 1, (
            np.int32(self.nc), np.int32(family), self.family_parent,
            self.child_offset, self.children, self.area, self.q_trial,
            self.mass_trial, self.active_trial, np.float64(self.dry),
            np.int32(0), np.int32(1),
            self.status_one))
        status = self._status(self.status_one)
        self._last_status = max(self._last_status, status)
        return status

    @_poison_open_trial
    def propose_refine(self, family: int, child_depth, intensity_gradient,
                       intensity_bounds):
        self._require_trial()
        if isinstance(family, bool) or not isinstance(family, int) or not (
                0 <= family < self.nfamily):
            raise ValueError("Q3-H3/v59 family index differs")
        self._fp64("child depth", child_depth, (self.nc,))
        self._fp64("intensity gradient", intensity_gradient, (2,))
        bounds = np.asarray(intensity_bounds, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError("Q3-H3/v59 intensity bounds differ")
        self.status_one.fill(0)
        self._launch("v59_prolong_family", 1, (
            np.int32(self.nc), np.int32(family), self.family_parent,
            self.child_offset, self.children, self.area, self.centre,
            child_depth, intensity_gradient, np.float64(bounds[0]),
            np.float64(bounds[1]), np.float64(self.dry), self.q_trial,
            self.mass_trial, self.active_trial, self.status_one))
        status = self._status(self.status_one)
        if status == STATUS_OK:
            self.trial_topology_generation += 1
            self.trial_predictor_cache_valid = False
        self._last_status = max(self._last_status, status)
        return status

    @_poison_open_trial
    def advect(self, normal_discharge_per_width, primitive_gradient,
               primitive_bounds, dt: float):
        """Apply signed upwind segment transfers reconstructed at midpoints."""
        self._require_trial()
        self._enter_component_phase("secondary_advection", 3, composite_only=True)
        self._fp64("normal discharge", normal_discharge_per_width, (self.nf,))
        self._fp64("primitive gradient", primitive_gradient, (2, self.nc))
        self._fp64("primitive bounds", primitive_bounds, (2, self.nc))
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3/v59 advection timestep differs")
        if int(self.cancel_flag.item()) != 0:
            self._last_status = STATUS_CANCELLED
            return {"status": STATUS_CANCELLED, "register": self.advective_register}
        self.advective_register.fill(0.0)
        self.advective_cell_increment.fill(0.0);self.status_face.fill(0)
        self.status_cell.fill(0)
        self._launch("v59_advective_segments", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.owner, self.neighbour,
            self.length, self.midpoint, self.centre,
            normal_discharge_per_width, self.accepted_q_snapshot,
            self.accepted_mass_snapshot, primitive_gradient, primitive_bounds,
            self.active_trial, np.float64(dt), self.advective_register,
            self.advective_cell_increment, self.status_face))
        status = self._status(self.status_face)
        if status == STATUS_OK:
            self._launch("v59_gather_scalar_register", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
                self.owner, self.neighbour, self.active_trial,
                self.advective_register, self.advective_cell_increment,
                self.status_cell))
            status = self._status(self.status_cell)
        if status == STATUS_OK:
            self.status_cell.fill(0)
            self._launch("v59_apply_mass_increment", self.nc, (
                np.int32(self.nc), self.area, self.active_trial,
                self.advective_cell_increment, self.mass_trial,
                self.status_cell))
            status = self._status(self.status_cell)
        self._last_status = max(self._last_status, status)
        return {"status": status, "register": self.advective_register,
                "cell_integrated_increment": self.advective_cell_increment}

    @_poison_open_trial
    def diffuse(self, depth_new, dt: float):
        """Solve one composite active-leaf backward-Euler graph system."""
        self._require_trial()
        self._enter_component_phase("secondary_diffusion", 4, composite_only=True)
        self._fp64("new depth", depth_new, (self.nc,))
        expected_depth = self.q_trial[0]
        if (self._memory_interval(depth_new) !=
                self._memory_interval(expected_depth)):
            raise ValueError(
                "Q3-H3/v59 diffusion depth must be the hydraulic trial depth")
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3/v59 diffusion timestep differs")
        cp = self.cp
        if (bool(cp.any(~cp.isfinite(depth_new)).item()) or
                bool(cp.any((self.active_trial == 1) &
                            (depth_new <= self.dry)).item())):
            raise ValueError(
                "Q3-H3/v59 diffusion depth is outside fully wet scope")
        self.diffusive_register.fill(0.0)
        self.diffusive_cell_increment.fill(0.0)
        cp.copyto(self.diffusion_input, self.mass_trial)
        if int(self.cancel_flag.item()) != 0:
            self._last_status = STATUS_CANCELLED
            self._last_diffusion = {"iterations": 0, "converged": False,
                                    "cancelled": True}
            return {"status": STATUS_CANCELLED,
                    "register": self.diffusive_register,
                    **self._last_diffusion}
        if self.diffusivity == 0.0 or dt == 0.0:
            self._last_diffusion = {"iterations": 0, "converged": True,
                                    "cancelled": False,
                                    "postsolve_patch": False,
                                    "conservation_residual_m4_s": 0.0}
            return {"status": STATUS_OK,
                    "register": self.diffusive_register,
                    **self._last_diffusion}

        self.status_face.fill(0);self.status_cell.fill(0)
        self._launch("v59_diffusion_conductance", self.nf, (
            np.int32(self.nc), np.int32(self.nf), depth_new,
            np.float64(self.diffusivity), np.float64(SCHMIDT_NUMBER),
            self.length, self.distance, self.owner, self.neighbour,
            self.active_trial, self.conductance, self.status_face))
        self._launch("v59_diffusion_setup", self.nc, (
            np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
            self.owner, self.neighbour, self.area, depth_new,
            self.conductance, self.diffusion_input, self.active_trial,
            np.float64(dt), self.x, self.rhs, self.diagonal,
            self.status_cell))
        setup_status = max(self._status(self.status_face),
                           self._status(self.status_cell))
        if setup_status != STATUS_OK:
            self._last_status = max(self._last_status, setup_status)
            return {"status": setup_status,
                    "register": self.diffusive_register,
                    "iterations": 0, "converged": False,
                    "cancelled": False, "postsolve_patch": False}

        self.pcg_control.fill(1)
        self._launch("v59_diffusion_matvec", self.nc, (
            np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
            self.owner, self.neighbour, self.area, depth_new,
            self.conductance, self.active_trial, np.float64(dt), self.x,
            self.ap, self.pcg_control, np.int32(1)))
        self._launch("v59_pcg_initialize_vectors", self.nc, (
            np.int32(self.nc), self.rhs, self.ap, self.diagonal,
            self.r, self.z, self.p))
        cp.multiply(self.rhs, self.rhs, out=self.product);cp.sum(self.product, out=self.rhs2)
        cp.multiply(self.r, self.r, out=self.product);cp.sum(self.product, out=self.rr)
        cp.multiply(self.r, self.z, out=self.product);cp.sum(self.product, out=self.rz)
        self._launch("v59_pcg_initialize_control", 1, (
            np.float64(self.relative_tolerance),
            np.float64(self.absolute_tolerance), self.rhs2, self.rr,
            self.rz, self.cancel_flag, self.pcg_scalars, self.pcg_control))
        for iteration in range(1, self.maximum_iterations + 1):
            self._launch("v59_diffusion_matvec", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
                self.owner, self.neighbour, self.area, depth_new,
                self.conductance, self.active_trial, np.float64(dt), self.p,
                self.ap, self.pcg_control, np.int32(0)))
            cp.multiply(self.p, self.ap, out=self.product);cp.sum(self.product, out=self.denominator)
            self._launch("v59_pcg_alpha", 1, (
                self.denominator, self.cancel_flag, self.pcg_scalars,
                self.pcg_control))
            self._launch("v59_pcg_update_xr", self.nc, (
                np.int32(self.nc), self.p, self.ap, self.pcg_scalars,
                self.pcg_control, self.x, self.r))
            self._launch("v59_pcg_precondition", self.nc, (
                np.int32(self.nc), self.r, self.diagonal,
                self.pcg_control, self.z))
            cp.multiply(self.r, self.r, out=self.product);cp.sum(self.product, out=self.rr)
            cp.multiply(self.r, self.z, out=self.product);cp.sum(self.product, out=self.rz_new)
            self._launch("v59_pcg_convergence_beta", 1, (
                np.int32(iteration), self.rr, self.rz_new,
                self.cancel_flag, self.pcg_scalars, self.pcg_control))
            self._launch("v59_pcg_update_p", self.nc, (
                np.int32(self.nc), self.z, self.pcg_scalars,
                self.pcg_control, self.p))

        cp.copyto(self.diffusion_candidate, self.diffusion_input)
        self._launch("v59_diffusion_finish", self.nc, (
            np.int32(self.nc), depth_new, self.x, self.active_trial,
            self.pcg_control, self.diffusion_input, self.diffusion_candidate,
            self.status_cell))
        self._launch("v59_diffusion_register", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.owner,
            self.neighbour, self.conductance, self.x, np.float64(dt),
            self.diffusive_register, self.diffusive_cell_increment,
            self.status_face))
        face_status = self._status(self.status_face)
        if face_status == STATUS_OK:
            self._launch("v59_gather_scalar_register", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
                self.owner, self.neighbour, self.active_trial,
                self.diffusive_register, self.diffusive_cell_increment,
                self.status_cell))
        control = cp.asnumpy(self.pcg_control)
        scalars = cp.asnumpy(self.pcg_scalars)
        status = max(self._status(self.status_cell), face_status,
                     int(control[1]))
        if status == STATUS_OK and not bool(control[3]):
            status = STATUS_NOT_CONVERGED
        actual_increment = self.area * (self.diffusion_candidate - self.diffusion_input)
        active_defect = cp.where(
            self.active_trial == 1,
            actual_increment - self.diffusive_cell_increment,
            0.0)
        residual = float(cp.max(cp.abs(active_defect)).item())
        scale = float(cp.max(cp.abs(actual_increment)).item())
        closure_limit = self.conservation_tolerance * (1.0 + scale)
        conservation_error = float(cp.abs(cp.sum(actual_increment)).item())
        conservation_scale = float(cp.sum(cp.abs(actual_increment)).item())
        conservation_limit = self.conservation_tolerance * (
            1.0 + conservation_scale)
        register_error = float(
            cp.abs(cp.sum(self.diffusive_cell_increment)).item())
        register_scale = float(
            cp.sum(cp.abs(self.diffusive_cell_increment)).item())
        register_limit = self.conservation_tolerance * (1.0 + register_scale)
        if status == STATUS_OK and (
                not np.isfinite((residual, conservation_error,
                                 register_error)).all() or
                residual > closure_limit or
                conservation_error > conservation_limit or
                register_error > register_limit):
            status = STATUS_INVALID_OUTPUT
        if status == STATUS_OK:
            cp.copyto(self.mass_trial, self.diffusion_candidate)
        self._last_status = max(self._last_status, status)
        self._last_diffusion = {
            "iterations": int(control[2]),
            "residual": float(scalars[2]),
            "target": float(scalars[3]),
            "converged": status == STATUS_OK,
            "cancelled": status == STATUS_CANCELLED,
            "postsolve_patch": False,
            "conservation_residual_m4_s": residual,
            "closed_inventory_error_m4_s": conservation_error,
            "register_sum_error_m4_s": register_error,
        }
        return {"status": status, "register": self.diffusive_register,
                "cell_integrated_increment": self.diffusive_cell_increment,
                **self._last_diffusion}

    @_poison_open_trial
    def freeze_betac(self, chezy, dt: float):
        """Freeze accepted-start fine-segment traction for implicit residuals."""
        self._require_trial()
        self._enter_component_phase("accepted_start_betac", 1, composite_only=True)
        self._fp64("Chezy", chezy, (self.nc,))
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3/v59 Betac timestep differs")
        self.tensor.fill(0.0);self.betac_flux.fill(0.0)
        self.betac_register.fill(0.0);self.betac_cell_increment.fill(0.0)
        self.status_cell.fill(0);self.status_face.fill(0)
        self._launch("v59_betac_cells", self.nc, (
            np.int32(self.nc), self.accepted_q_snapshot,
            self.accepted_mass_snapshot, chezy, self.active_trial,
            np.float64(self.dry), np.float64(self.betac),
            np.float64(self.gravity), np.float64(self.von_karman),
            np.float64(self.chzmin), self.tensor, self.status_cell))
        status = self._status(self.status_cell)
        if status == STATUS_OK:
            self._launch("v59_betac_segments", self.nf, (
                np.int32(self.nc), np.int32(self.nf), self.owner,
                self.neighbour, self.normal, self.length, self.tensor,
                self.active_trial, np.float64(dt), self.betac_flux,
                self.betac_register, self.betac_cell_increment,
                self.status_face))
            status = self._status(self.status_face)
        if status == STATUS_OK:
            self._launch("v59_gather_vector_register", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.offset, self.faces,
                self.owner, self.neighbour, self.active_trial,
                self.accepted_q_snapshot,
                self.betac_register, self.betac_cell_increment,
                self.status_cell))
            status = self._status(self.status_cell)
        self._last_status = max(self._last_status, status)
        return {"status": status, "flux_per_unit_length": self.betac_flux,
                "register": self.betac_register,
                "cell_integrated_increment": self.betac_cell_increment}

    @_poison_open_trial
    def apply_coriolis(self, dt: float):
        self._require_trial()
        self._enter_component_phase("endpoint_coriolis", 2)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("Q3-H3/v59 Coriolis timestep differs")
        self.coriolis_cell_increment.fill(0.0);self.coriolis_ledger.fill(0.0)
        self.status_cell.fill(0)
        self._launch("v59_coriolis_active_leaves", self.nc, (
            np.int32(self.nc), np.float64(self.coriolis), np.float64(dt),
            self.area, self.active_trial, self.q_trial,
            self.coriolis_cell_increment, self.status_cell))
        status = self._status(self.status_cell)
        if status == STATUS_OK:
            self._launch("v59_reduce_vector_cell_increment_ordered", 1, (
                np.int32(self.nc), self.coriolis_cell_increment,
                self.coriolis_ledger))
        self._last_status = max(self._last_status, status)
        return {"status": status, "state": self.q_trial,
                "cell_integrated_increment": self.coriolis_cell_increment,
                "ledger": self.coriolis_ledger}

    def request_cancel(self):
        self._require_bound_device()
        self.cancel_flag.fill(1)

    def _validate_trial_state_for_commit(self):
        cp = self.cp
        if (bool(cp.any(~cp.isfinite(self.q_trial)).item()) or
                bool(cp.any(~cp.isfinite(self.mass_trial)).item()) or
                bool(cp.any((self.active_trial != 0) &
                            (self.active_trial != 1)).item()) or
                bool(cp.any((self.active_trial == 1) &
                            (self.q_trial[0] <= self.dry)).item())):
            raise ValueError(
                "Q3-H3/v59 candidate state is not finite fully wet")
        self._validate_active_ownership(self.active_trial, accepted=False)

    @_poison_open_trial
    def finish_trial(self, accepted_q, accepted_mass, accepted_active, *, commit):
        self._require_trial()
        self._fp64("accepted hydraulics", accepted_q, (3, self.nc))
        self._fp64("accepted signed hI", accepted_mass, (self.nc,))
        self._int32("accepted active leaves", accepted_active, (self.nc,))
        self._validate_external_state_storage(
            accepted_q, accepted_mass, accepted_active)
        if not isinstance(commit, (bool, np.bool_)):
            raise TypeError("Q3-H3/v59 commit flag must be boolean")
        if self._accepted_ptrs != (
                int(accepted_q.data.ptr), int(accepted_mass.data.ptr),
                int(accepted_active.data.ptr)):
            raise RuntimeError("Q3-H3/v59 accepted ownership changed")
        if bool(commit):
            if (self._trial_rejected or self._last_status != STATUS_OK or
                    int(self.cancel_flag.item()) != 0):
                raise RuntimeError("Cannot commit an invalid Q3-H3/v59 transaction")
            self._validate_trial_state_for_commit()
            # Refined inactive parents are synchronized caches, never a second
            # physical owner.  Cache synchronization is private/fallible work
            # and therefore occurs before any external accepted copy.
            parent_mask = self.cp.asnumpy(
                self.active_trial[self.family_parent])
            for family, selected in enumerate(parent_mask):
                if selected == 0 and self.synchronize_parent_cache(family) != STATUS_OK:
                    raise RuntimeError(
                        "Cannot commit an unsynchronized Q3-H3/v59 family")
            self._validate_trial_state_for_commit()
            cp = self.cp
            cp.copyto(accepted_q, self.q_trial)
            cp.copyto(accepted_mass, self.mass_trial)
            cp.copyto(accepted_active, self.active_trial)
            cp.copyto(self.accepted_q_snapshot, self.q_trial)
            cp.copyto(self.accepted_mass_snapshot, self.mass_trial)
            cp.copyto(self.accepted_active_snapshot, self.active_trial)
            cp.copyto(self.accepted_advective_register, self.advective_register)
            cp.copyto(self.accepted_diffusive_register, self.diffusive_register)
            cp.copyto(self.accepted_betac_register, self.betac_register)
            cp.copyto(self.accepted_coriolis_ledger, self.coriolis_ledger)
            self.accepted_topology_generation = self.trial_topology_generation
            self.accepted_predictor_cache_valid = self.trial_predictor_cache_valid
            self._have_accepted_state = True
        self._trial_open = False;self._accepted_ptrs = None
        self._trial_rejected = False
        return accepted_q, accepted_mass, accepted_active

    def abort_trial(self):
        if self._trial_open:
            self._trial_open = False;self._accepted_ptrs = None
            self._trial_rejected = False

    def storage_identity(self):
        arrays = (
            self.cell_id, self.level, self.area, self.centre, self.owner,
            self.neighbour, self.source_id, self.segment_level, self.length,
            self.normal, self.midpoint, self.distance, self.family_parent,
            self.child_offset, self.children, self.offset, self.faces,
            self.accepted_q_snapshot, self.accepted_mass_snapshot,
            self.accepted_active_snapshot, self.q_trial, self.mass_trial,
            self.active_trial, self.advective_register,
            self.diffusive_register, self.betac_flux, self.betac_register,
            self.advective_cell_increment, self.diffusive_cell_increment,
            self.betac_cell_increment, self.coriolis_cell_increment,
            self.coriolis_ledger, self.tensor, self.conductance,
            self.diffusion_input, self.diffusion_candidate, self.x, self.rhs,
            self.diagonal, self.r, self.z, self.p, self.ap, self.product,
            self.pcg_scalars, self.pcg_control, self.cancel_flag,
        )
        return tuple((id(item), int(item.data.ptr), item.shape, item.dtype.str,
                      item.strides, int(item.device.id)) for item in arrays)

    def diagnostic(self):
        return {
            "scope_id": SCOPE_ID,
            "envelope_version": ENVELOPE_VERSION,
            "integrator": INTEGRATOR,
            "time_coupling": TIME_COUPLING,
            "device_id": self.device_id,
            "automatic_cpu_fallback": False,
            "live_controller_supported": False,
            "component_scope": (
                "fully_wet_closed_synchronous_selected_mesh_transfer_"
                "canonical_advection_composite_diffusion_betac_coriolis"),
            "unsupported": (
                "live_hydraulic_controller", "recursive_implicit_subcycling",
                "open_boundaries", "wet_dry", "sediment_exner",
                "restart_serialization", "production_qgis"),
            "component_case": self.component_case,
            "fixture_sha256": self.fixture_sha256,
            "execution_fingerprint": self.identity["fingerprint"],
            "topology_fingerprint": self.identity["topology_fingerprint"],
            "face_catalog_fingerprint": self.identity[
                "face_catalog_fingerprint"],
            "boundary_catalog_fingerprint": self.identity[
                "boundary_catalog_fingerprint"],
            "mesh_fingerprint": self.mesh_fingerprint,
            "cuda_source_sha256": self.source_sha256,
            "python_wrapper_sha256": self.wrapper_sha256,
            "accepted_parent_sha256": dict(self.parent_hashes),
            "topology_generation": self.accepted_topology_generation,
            "predictor_cache_valid": self.accepted_predictor_cache_valid,
            "diffusion": dict(self._last_diffusion),
        }


__all__ = [
    "BETAC_OPERATOR_AMENDMENT_SHA256", "CANONICAL_FIXTURE_SHA256",
    "CUDA_SOURCE_SHA256", "ENVELOPE_VERSION", "INTEGRATOR",
    "KERNEL_NAMES", "PARENT_HASHES", "SCOPE_ID", "STATUS_CANCELLED",
    "STATUS_INVALID_TOPOLOGY", "STATUS_OK", "TIME_COUPLING",
    "ResidentPrognosticSecondaryAMRV59",
]
