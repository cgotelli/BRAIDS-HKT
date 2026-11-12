"""Host-only QGIS AMR project to experimental schema-2 catalog preparation.

This internal route derives selected geometry and native decks from the saved
project, runs the unchanged validation zero-step exporter, and authenticates
the combined input package. Nothing here advances a solver or enables the
QGIS GPU Run action for AMR.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

import numpy as np

from braids_core.boundaries.amr import AMRHierarchy, AMRSettings, build_hierarchy
from braids_core.boundaries.cut_cell_io import (CutBoundary, CutSediment,
                                               CutSedimentFraction, write_cut_case)
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid

from .bank_inputs import load_bank_zones, map_bank_zones_to_cells
from .project import (_spatial_bed_values, initial_state, load_project,
                      terrain_averages)
from .refinement_inputs import (REFINEMENT_LAYER_NAME, load_refinement_zones,
                                map_refinement_zones)
from .raster_surface import surface_averages
from .roughness import MANNING_LAYER_NAME, load_manning_zones, map_manning_to_cells
from .structure_inputs import OBSTACLE_LAYER_NAME, load_solid_obstacles
from .water_body import WATER_BODY_LAYER_NAME, map_water_body_to_cells


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _node_digest(ids: np.ndarray) -> str:
    encoded = json.dumps(ids.tolist(), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GPUAMRTopologyPlan:
    name: str
    node_ids: np.ndarray
    selected_mesh: object
    node_ids_sha256: str


@dataclass(frozen=True)
class GPUAMRCatalogPlan:
    project_path: Path
    project_sha256: str
    native_executable: Path
    native_executable_sha256: str
    source_files_sha256: dict[str, str]
    hierarchy: AMRHierarchy
    topologies: tuple[GPUAMRTopologyPlan, ...]


def _validate_project_scope(project: dict) -> AMRSettings:
    """Admit only the physics represented by the current bank schema-2 reader."""
    numerics = project["numerics"]
    internal = project.get("internal_conditions", {})
    initial = project["initial"]
    amr = AMRSettings(**numerics.get("amr", {}))
    if not amr.enabled or numerics["scheme"] != 6 or not amr.subcycling:
        raise ValueError("GPU AMR catalog needs enabled subcycled AMR and scheme 6")
    if numerics.get('horizontal_viscosity_m2_s',0.) != 0.:
        raise ValueError('GPU AMR catalog does not support horizontal viscosity')
    if (not numerics.get("sediment_enabled", True) or
            numerics.get("bedload") != 2 or numerics.get("exchange") != 0 or
            numerics.get("bed_slope", 0) != 0 or
            numerics.get("hiding_exposure", 0) != 0 or
            numerics.get("mpm_friction_mode") != 1 or
            numerics.get("mpm_critical_shields") != .047):
        raise ValueError("GPU AMR bank schema 2 needs plain MPM bedload and exchange 0")
    if (len(project.get("sediment_fractions", ())) != 2 or
            initial.get("active_layer_thickness_m", 0) <= 0 or
            initial.get("substrate_layer_count") != 1 or
            initial.get("active_layer_mode", "constant") != "constant"):
        raise ValueError("GPU AMR bank schema 2 needs two fractions and two constant bed layers")
    bank = internal.get("bank_failure", {})
    if (not bank.get("enabled", False) or bank.get("front_tracking_enabled", False) or
            bank.get("toe_erosion_mode") != "noncohesive_transport_capacity" or
            not bank.get("zones_path")):
        raise ValueError("GPU AMR bank schema 2 needs mapped noncohesive physical-toe banks")
    if (not internal.get("general_avalanche", {}).get("enabled", False) or
            not internal.get("secondary_flow", {}).get("enabled", False) or
            internal.get("secondary_flow", {}).get("model") != "prognostic"):
        raise ValueError("GPU AMR bank schema 2 needs avalanche and prognostic secondary flow")
    for key in ("bank_morphology", "bedload_adaptation", "bedform_roughness",
                "morphological_acceleration"):
        if internal.get(key, {}).get("enabled", False):
            raise ValueError("GPU AMR bank schema 2 does not admit " + key)
    boundaries = project.get("boundaries", {})
    if (len(boundaries) != 1 or
            any(row.get("mode") != "stage" or row.get("hydraulic_series") or
                row.get("sediment_series") or
                row.get("bedload_boundary_mode", "prescribed") != "prescribed" or
                row.get("concentration_kg_m3", 0) != 0 or
                row.get("bed_input_kg_m_s", 0) != 0 or
                row.get("bed_input_kg_s") not in (None, 0) or
                any(row.get("fraction_concentration_kg_m3", {}).values()) or
                any(row.get("fraction_bed_input_kg_s", {}).values()) or
                row.get("fraction_sediment_series")
                for row in boundaries.values())):
        raise ValueError("GPU AMR bank schema 2 needs one unscheduled stage outlet and zero sediment feed")
    return amr


def _native_metadata_matches_project(metadata: dict, project: dict) -> bool:
    """Check selected-deck physics against the saved QGIS controls."""
    if not isinstance(metadata, dict):
        return False
    numerics = project["numerics"]
    internal = project["internal_conditions"]
    initial = project["initial"]
    bank = internal["bank_failure"]
    avalanche = internal["general_avalanche"]
    secondary = internal["secondary_flow"]
    material_record = metadata.get("materials")
    friction_record = metadata.get("mpm_friction_mode")
    if not isinstance(material_record, dict) or not isinstance(friction_record, dict):
        return False
    materials = material_record.get("fractions", ())
    if not isinstance(materials, (tuple, list)):
        return False
    if (metadata.get("amr", object()) is not None or
            metadata.get("reconstruction_scheme") != numerics["scheme"] or
            metadata.get("bedload_law") != numerics["bedload"] or
            metadata.get("suspended_exchange_name") != "off" or
            metadata.get("cfl") != numerics["cfl"] or
            metadata.get("mpm_critical_shields") != numerics["mpm_critical_shields"] or
            friction_record.get("mode") != numerics["mpm_friction_mode"] or
            len(materials) != len(project["sediment_fractions"])):
        return False
    for actual, expected in zip(materials, project["sediment_fractions"]):
        if not isinstance(actual, dict) or any(actual.get(key) != value
                                                for key, value in expected.items()):
            return False
    layers = metadata.get("bed_layers", {})
    if not isinstance(layers, dict):
        return False
    if (layers.get("active_layer_thickness_m") != initial["active_layer_thickness_m"] or
            layers.get("active_layer_mode") != initial["active_layer_mode"] or
            layers.get("substrate_layer_count") != initial["substrate_layer_count"]):
        return False
    failure = metadata.get("bank_failure", {})
    if not isinstance(failure, dict):
        return False
    if (failure.get("enabled") is not True or
            failure.get("toe_erosion_mode") != bank["toe_erosion_mode"] or
            failure.get("toe_capacity_fraction") != bank["toe_capacity_fraction"] or
            failure.get("critical_undercut_m") != bank["critical_undercut_m"] or
            failure.get("bank_front_tracking_enabled") is not False):
        return False
    native_avalanche = metadata.get("general_avalanche", {})
    if not isinstance(native_avalanche, dict):
        return False
    if (native_avalanche.get("enabled") is not True or any(
            native_avalanche.get(key) != avalanche[key] for key in (
                "wet_maximum_slope", "dry_maximum_slope", "avalanche_time_s",
                "relaxation_length_m", "wet_depth_switch_m"))):
        return False
    native_secondary = metadata.get("secondary_flow", {})
    if not isinstance(native_secondary, dict):
        return False
    if (native_secondary.get("enabled") is not True or any(
            native_secondary.get(key) != secondary[key] for key in (
                "model", "transport_direction", "momentum_beta", "espir"))):
        return False
    native_boundaries = metadata.get("boundaries", {})
    if not isinstance(native_boundaries, dict):
        return False
    if set(native_boundaries) != set(project["boundaries"]):
        return False
    for name, expected in project["boundaries"].items():
        actual = native_boundaries[name]
        if not isinstance(actual, dict) or any(actual.get(key) != expected[key]
                                                for key in expected):
            return False
    return True


def _source_files(project: dict) -> dict[str, str]:
    """Record the content of every external input used by this host route."""
    internal = project.get("internal_conditions", {})
    paths = (
        project["terrain"].get("dem_path"),
        project["terrain"].get("section_dem_path"),
        project.get("bed_inputs", {}).get("bedrock_raster_path"),
        project.get("bed_inputs", {}).get("thickness_raster_path"),
        project.get("bed_inputs", {}).get("zones_path"),
        project.get("roughness", {}).get("zones_path"),
        project.get("initial", {}).get("water_body_path"),
        project.get("mesh_refinement", {}).get("zones_path"),
        internal.get("solid_obstacles", {}).get("path"),
        internal.get("bank_failure", {}).get("zones_path"),
    )
    resolved = sorted({str(Path(value).expanduser().resolve())
                       for value in paths if value})
    return {path: _sha256(Path(path)) for path in resolved}


def _source_digest(files: dict[str, str]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _automatic_layouts(hierarchy: AMRHierarchy, minimum: np.ndarray | None,
                       project: dict) -> tuple[tuple[str, np.ndarray], ...]:
    """Choose one balanced interior four-child transition from QGIS inputs."""
    requested = np.zeros(hierarchy.count, dtype=np.int64)
    if minimum is not None:
        requested = np.maximum(requested, minimum)
    finest = hierarchy.offsets[-2]
    for face in hierarchy.finest.faces:
        if face.neighbour < 0 and face.bc_id:
            requested[finest + face.owner] = hierarchy.finest_level
    coarse = np.asarray(hierarchy.balanced_leaves(requested), dtype=np.int64)
    coarse_mesh = hierarchy.select(coarse)
    bank = project["internal_conditions"]["bank_failure"]
    zones = load_bank_zones(bank["zones_path"], project["domain"]["crs_wkt"],
                            bank["zones_layer"])
    coarse_material, _ = map_bank_zones_to_cells(coarse_mesh, zones)
    if not np.any(coarse_material):
        raise ValueError("GPU AMR catalog has no selected bank material")
    coarse_positions = {int(node): i for i, node in enumerate(coarse)}
    for parent in sorted(map(int, coarse),
                         key=lambda node: (-int(hierarchy.node_level[node]), node)):
        children = hierarchy.children[parent]
        if (len(children) != 4 or
                hierarchy.node_level[parent] >= hierarchy.maximum_level[parent]):
            continue
        refined = np.asarray(sorted((set(map(int, coarse)) - {parent}) |
                                    set(map(int, children))), dtype=np.int64)
        try:
            refined_mesh = hierarchy.select(refined)
        except ValueError:
            continue
        refined_material, _ = map_bank_zones_to_cells(refined_mesh, zones)
        positions = {int(node): i for i, node in enumerate(refined)}
        if all(refined_material[positions[child]] ==
               coarse_material[coarse_positions[parent]] for child in children):
            return (("refined", refined), ("coarse", coarse))
    raise ValueError("GPU AMR project has no balanced, bank-consistent interior refinement pair")


def plan_gpu_amr_catalog(
    project_path: str | Path,
    selections: Sequence[tuple[str, Sequence[int]]] | None = None,
    *, cancelled: Callable[[], bool] | None = None,
) -> GPUAMRCatalogPlan:
    """Validate selected layouts against an ordinary saved QGIS project.

    With no explicit selections, choose one bank-consistent interior four-child
    transition while honoring user refinement zones and finest outlet owners.
    This performs no filesystem writes or numerical solver steps.
    """
    path = Path(project_path).expanduser().resolve()
    project = load_project(path)
    amr = _validate_project_scope(project)
    names = set()
    requested = []
    if selections is not None:
        if not isinstance(selections, Sequence) or not selections:
            raise ValueError("GPU AMR catalog requires selected layouts")
        for item in selections:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("GPU AMR layout requires a name and global node IDs")
            name, nodes = item
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
                raise ValueError("Invalid or repeated GPU AMR topology name")
            names.add(name)
            ids = np.asarray(nodes)
            if (ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu" or
                    np.any(ids > np.iinfo(np.int64).max)):
                raise ValueError("GPU AMR selected nodes must be sorted unique integers")
            ids = ids.astype(np.int64, copy=True)
            if np.any(ids < 0) or np.any(np.diff(ids) <= 0):
                raise ValueError("GPU AMR selected nodes must be sorted unique integers")
            requested.append((name, ids))
    if cancelled and cancelled():
        raise InterruptedError("GPU AMR catalog planning cancelled")
    domain_data = project["domain"]
    domain = PolygonDomain(domain_data["domain_id"],
                           domain_data["vertices_xy"], domain_data["crs_wkt"])
    assignments = [BoundaryAssignment(**row) for row in project["assignments"]]
    grid = NativeGrid(**project["grid"])
    obstacle = project.get("internal_conditions", {}).get("solid_obstacles", {})
    obstacles = (() if not obstacle.get("path") else load_solid_obstacles(
        obstacle["path"], domain.crs_wkt,
        obstacle.get("layer", OBSTACLE_LAYER_NAME)))
    options = dict(levels=amr.levels, refinement_ratio=amr.refinement_ratio,
        minimum_area_fraction=project["numerics"].get(
            "minimum_boundary_cell_area_fraction", 0.))
    if "cancelled" in inspect.signature(build_hierarchy).parameters:
        options["cancelled"] = cancelled
    hierarchy = build_hierarchy(domain, assignments, grid, obstacles, **options)
    refinement = project.get("mesh_refinement", {})
    minimum = None
    if refinement.get("zones_path"):
        zones = load_refinement_zones(
            refinement["zones_path"], domain.crs_wkt,
            refinement.get("zones_layer", REFINEMENT_LAYER_NAME),
            maximum_level=amr.levels - 1)
        minimum, _ = map_refinement_zones(hierarchy, zones)
    if selections is None:
        requested = list(_automatic_layouts(hierarchy, minimum, project))
    planned = []
    finest = hierarchy.offsets[-2]
    for name, ids in requested:
        if cancelled and cancelled():
            raise InterruptedError("GPU AMR catalog planning cancelled")
        cover = hierarchy.leaf_cover(ids)
        if minimum is not None and np.any(
                hierarchy.node_level[cover[finest:]] < minimum[finest:]):
            raise ValueError("GPU AMR layout violates QGIS minimum refinement zones")
        mesh = hierarchy.select(ids, require_balance=True)
        open_faces = [face for face in mesh.faces if face.neighbour < 0 and face.bc_id]
        if (not open_faces or any(mesh.cells[face.owner].level != hierarchy.finest_level
                                  for face in open_faces)):
            raise ValueError("GPU AMR stage outlet must remain on finest selected leaves")
        planned.append(GPUAMRTopologyPlan(name, ids, mesh, _node_digest(ids)))
    executable = Path(project["native_executable"]).resolve()
    return GPUAMRCatalogPlan(path, _sha256(path), executable,
                             _sha256(executable), _source_files(project),
                             hierarchy, tuple(planned))


def _write_selected_deck(plan: GPUAMRCatalogPlan,
                         topology: GPUAMRTopologyPlan, folder: Path,
                         project: dict,
                         cancelled: Callable[[], bool] | None = None) -> Path:
    """Resolve one fixed selected mesh using the ordinary QGIS input mappers."""
    if cancelled and cancelled():
        raise InterruptedError("GPU AMR catalog preparation cancelled")
    folder.mkdir(parents=True, exist_ok=False)
    mesh = topology.selected_mesh
    terrain = project["terrain"]
    if terrain.get("interpretation", "bilinear_surface") == "cell_average_raster":
        bed, _ = terrain_averages(mesh, terrain["dem_path"], cancelled=cancelled)
    else:
        bed, _ = surface_averages(mesh, terrain["dem_path"], cancelled=cancelled)
    thickness, bedrock, _ = _spatial_bed_values(mesh, bed, project,
                                                cancelled=cancelled)
    initial = project["initial"]
    water = None
    if initial.get("water_body_path"):
        water, _ = map_water_body_to_cells(
            mesh, initial["water_body_path"], project["domain"]["crs_wkt"],
            initial.get("water_body_layer", WATER_BODY_LAYER_NAME))
    fractions = tuple(CutSedimentFraction(**row)
                      for row in project["sediment_fractions"])
    q, rock, _ = initial_state(
        mesh, bed, initial, CutSediment(**project["sediment"]),
        thickness_by_cell=thickness, bedrock_by_cell=bedrock,
        sediment_fractions=fractions, water_body_mask=water)
    roughness = project["roughness"]
    roughness_zones = (() if not roughness.get("zones_path") else
        load_manning_zones(roughness["zones_path"], project["domain"]["crs_wkt"],
                           roughness.get("zones_layer", MANNING_LAYER_NAME)))
    manning, _ = map_manning_to_cells(mesh, roughness["default_manning_n"],
                                      roughness_zones)
    bank = project["internal_conditions"]["bank_failure"]
    bank_zones = load_bank_zones(bank["zones_path"], project["domain"]["crs_wkt"],
                                bank["zones_layer"])
    bank_mask, _ = map_bank_zones_to_cells(mesh, bank_zones)
    if not np.any(bank_mask):
        raise ValueError("GPU AMR selected deck has no bank material")
    numerics = project["numerics"]
    general = project["internal_conditions"]["general_avalanche"]
    secondary = project["internal_conditions"]["secondary_flow"]
    deck = folder / "case.data"
    write_cut_case(
        deck, mesh, q, rock,
        {name: CutBoundary(**row) for name, row in project["boundaries"].items()},
        end_time_s=numerics["end_time_s"], dt_max_s=numerics["dt_max_s"],
        output_interval_s=numerics["output_interval_s"], cfl=numerics["cfl"],
        scheme=6, bedload=2, exchange=0,
        settling_velocity_m_s=numerics["settling_velocity_m_s"],
        horizontal_viscosity_m2_s=numerics.get('horizontal_viscosity_m2_s',0.),
        horizontal_viscosity_formulation=numerics.get('horizontal_viscosity_formulation','depth_conservative'),
        manning=roughness["default_manning_n"],
        manning_by_cell=manning if roughness.get("zones_path") else None,
        sediment_fractions=fractions,
        mpm_cf=numerics["mpm_cf"],
        mpm_critical_shields=numerics["mpm_critical_shields"],
        mpm_friction_mode=numerics["mpm_friction_mode"],
        active_layer_thickness_m=initial["active_layer_thickness_m"],
        active_layer_mode="constant",
        substrate_layer_count=1,
        substrate_layer_thickness_m=initial["substrate_layer_thickness_m"],
        bank_erodible_by_cell=bank_mask, bank_failure=True,
        bank_toe_erosion_mode="noncohesive_transport_capacity",
        bank_toe_capacity_fraction=bank["toe_capacity_fraction"],
        bank_critical_undercut_m=bank["critical_undercut_m"],
        general_avalanche=True,
        general_avalanche_wet_slope=general["wet_maximum_slope"],
        general_avalanche_dry_slope=general["dry_maximum_slope"],
        general_avalanche_time_s=general["avalanche_time_s"],
        general_avalanche_wet_depth_switch_m=general["wet_depth_switch_m"],
        general_avalanche_relaxation_length_m=general["relaxation_length_m"],
        secondary_flow=True, secondary_flow_model="prognostic",
        secondary_flow_transport_direction=secondary["transport_direction"],
        secondary_flow_momentum_beta=secondary["momentum_beta"],
        secondary_flow_espir=secondary["espir"],
        morphological_factor=1., compact_sidecars=True,
        source_sha256=dict(qgis_project_sha256=plan.project_sha256,
                           qgis_selected_nodes_sha256=topology.node_ids_sha256,
                           qgis_native_executable_sha256=plan.native_executable_sha256))
    mesh.write_geometry(folder / "geometry.npz")
    return deck


def _validate_initial_transition(plan: GPUAMRCatalogPlan,
                                 initialized: Mapping[str, dict]) -> None:
    """Admit a one-group catalog only if its initial state restricts exactly."""
    if tuple(row.name for row in plan.topologies) != ("refined", "coarse"):
        raise ValueError("Automatic GPU AMR catalog topology order changed")
    refined, coarse = plan.topologies
    larger, smaller = set(map(int, refined.node_ids)), set(map(int, coarse.node_ids))
    parent_only = smaller - larger
    children_only = larger - smaller
    if len(parent_only) != 1 or len(children_only) != 4:
        raise ValueError("Automatic GPU AMR layouts do not differ by one child group")
    parent = next(iter(parent_only))
    children = np.flatnonzero(plan.hierarchy.parent == parent)
    if set(map(int, children)) != children_only:
        raise ValueError("Automatic GPU AMR transition has incomplete children")
    a, b = initialized["refined"], initialized["coarse"]
    refined_at = {int(node): index for index, node in enumerate(refined.node_ids)}
    coarse_at = {int(node): index for index, node in enumerate(coarse.node_ids)}
    retained = sorted(larger & smaller)
    for key in ("state", "bed", "rock", "manning_cell", "bank_material_mask"):
        if not np.array_equal(
                np.asarray(a[key])[..., [refined_at[node] for node in retained]],
                np.asarray(b[key])[..., [coarse_at[node] for node in retained]]):
            raise ValueError("GPU AMR retained initial material/state differs: " + key)
    weights = np.asarray([plan.hierarchy.cells[int(child)].area_m2
                          for child in children], dtype=np.float64)
    child_at = [refined_at[int(child)] for child in children]
    for key in ("state", "bed", "rock"):
        expected = np.asarray(a[key])[..., child_at] @ weights / weights.sum()
        actual = np.asarray(b[key])[..., coarse_at[parent]]
        if not np.allclose(actual, expected, rtol=0., atol=1e-12):
            raise ValueError("GPU AMR initial parent/children are not conservative: " + key)


def package_gpu_amr_catalog(
    plan: GPUAMRCatalogPlan, destination: str | Path,
    *, native_exports: Mapping[str, str | Path], runtime_build: str | Path,
):
    """Package project-bound zero-step exports using the strict schema-2 gate.

    The QGIS producer writes each selected fixed-mesh deck with
    ``source_sha256`` markers for the saved project and selected node list.
    The native exporter copies those deck bytes into its authenticated origin.
    """
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("GPU AMR catalog destination must be new")
    if (_sha256(plan.project_path) != plan.project_sha256 or
            _sha256(plan.native_executable) != plan.native_executable_sha256):
        raise ValueError("QGIS GPU AMR project or native executable changed after planning")
    project = load_project(plan.project_path)
    if _source_files(project) != plan.source_files_sha256:
        raise ValueError("QGIS GPU AMR raster or polygon inputs changed after planning")
    if set(native_exports) != {row.name for row in plan.topologies}:
        raise ValueError("GPU AMR native exports differ from selected QGIS layouts")
    rows = []
    for topology in plan.topologies:
        folder = Path(native_exports[topology.name]).expanduser().resolve()
        metadata_path = folder / "case.data.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("GPU AMR selected native deck metadata is missing or linked")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("GPU AMR selected native deck metadata is invalid")
        markers = metadata.get("source_sha256", {})
        if (not isinstance(markers, dict) or
                markers.get("qgis_project_sha256") != plan.project_sha256 or
                markers.get("qgis_selected_nodes_sha256") != topology.node_ids_sha256):
            raise ValueError("GPU AMR native export is not bound to the QGIS project and layout")
        if not _native_metadata_matches_project(metadata, project):
            raise ValueError("GPU AMR native deck controls differ from the QGIS project")
        rows.append((topology.name, topology.node_ids, folder))
    # The exporter authenticates every native receipt, origin byte, operator,
    # sidecar, selected geometry and cross-topology control before it creates
    # the destination.  This package remains a host-only input artifact.
    from braids_gpu.amr_prepared import export_bank_amr_prepared_v2

    return export_bank_amr_prepared_v2(
        destination, plan.hierarchy, rows, runtime_build=runtime_build,
        source=dict(qgis_project_sha256=plan.project_sha256,
                    qgis_native_executable_sha256=plan.native_executable_sha256,
                    qgis_input_sources_sha256=_source_digest(plan.source_files_sha256),
                    native_initialization_only=True,
                    GPU_solver_executed=False,
                    general_GPU_AMR_qualified=False))


def prepare_gpu_amr_catalog_from_project(
    project_path: str | Path, destination: str | Path, *,
    export_runtime: str | Path,
    cancelled: Callable[[], bool] | None = None,
):
    """Prepare an internal host-only schema-2 catalog from a QGIS project.

    The unchanged G4 I/O-only native exporter is an explicit validation
    dependency here. Its runtime is verified before creating output; it takes
    zero numerical steps. This route is not wired into QGIS GPU Run.
    """
    plan = plan_gpu_amr_catalog(project_path, cancelled=cancelled)
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("GPU AMR catalog output must be new")
    runtime = Path(export_runtime).expanduser().resolve()
    from validation.gpu_fwave import g4_bank_active_front_export as exporter

    exporter.verify_build(runtime)
    if cancelled and cancelled():
        raise InterruptedError("GPU AMR catalog preparation cancelled")
    project = load_project(plan.project_path)
    if (_sha256(plan.project_path) != plan.project_sha256 or
            _sha256(plan.native_executable) != plan.native_executable_sha256 or
            _source_files(project) != plan.source_files_sha256):
        raise ValueError("QGIS GPU AMR input identity changed before native preparation")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "CATALOG_INPUTS.json").write_text(json.dumps({
        "schema": 1, "kind": "braids-qgis-gpu-amr-host-inputs",
        "project_sha256": plan.project_sha256,
        "native_executable_sha256": plan.native_executable_sha256,
        "source_files_sha256": plan.source_files_sha256,
        "source_files_digest_sha256": _source_digest(plan.source_files_sha256),
        "runtime_build_sha256": _sha256(runtime / "build.json"),
        "topologies": [{"name": row.name,
                        "selected_nodes_sha256": row.node_ids_sha256,
                        "selected_cells": len(row.node_ids)}
                       for row in plan.topologies],
        "native_solver_steps": 0, "GPU_solver_executed": False,
        "general_GPU_AMR_qualified": False,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    exports = {}
    initialized = {}
    for topology in plan.topologies:
        deck = _write_selected_deck(plan, topology,
                                    destination / topology.name / "deck", project,
                                    cancelled=cancelled)
        if cancelled and cancelled():
            raise InterruptedError("GPU AMR catalog preparation cancelled")
        prepared = destination / topology.name / "native_initialized"
        initialized[topology.name], _ = exporter.export(deck, prepared,
                                                        runtime=runtime)
        exports[topology.name] = prepared
    _validate_initial_transition(plan, initialized)
    if cancelled and cancelled():
        raise InterruptedError("GPU AMR catalog preparation cancelled")
    return package_gpu_amr_catalog(
        plan, destination / "amr_prepared", native_exports=exports,
        runtime_build=runtime / "build.json")
