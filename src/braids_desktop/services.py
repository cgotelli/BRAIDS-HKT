"""QGIS-free project services for the independently staged BRAIDS runtime.

The desktop bootstrap places ``desktop/runtime`` first on sys.path. This module
uses the same strict project resolver and portable folder format as that runtime;
it does not introduce a desktop numerical model or migrate older inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.river_model import new_model, resolve_model
from braids_core.river_open_boundary import new_state, resolve as resolve_boundaries
from braids_core.river_project import (
    SCHEMA, SLOPE_PARAMETERS, HIDING_PARAMETERS, ROUGHNESS_PARAMETERS,
    BEDFORM_METHODS, DEPENDENT_LAYER_PARAMETERS, VAN_RIJN_REFERENCE_PARAMETERS,
    SKIN_SHEAR_PARAMETERS, WET_AVALANCHE_PARAMETERS, validate_project,
)
from braids_core import river_folder
from braids_core.sediment.current_river_solver import CurrentRiverControls
from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
from braids_core.sediment.river_column_methods import method_ids
from braids_core.sediment.river_field_policy import FieldSettings


def geometry_grid(domain, minimum_size_m, levels):
    """Coarse covering grid; the requested size is the finest 2:1 cell width."""
    if type(levels) is not int or not 1 <= levels <= 6:
        raise ValueError("Choose between 1 and 6 refinement levels")
    if (type(minimum_size_m) not in (int, float)
            or not math.isfinite(minimum_size_m) or minimum_size_m <= 0):
        raise ValueError("Minimum mesh size must be positive finite metres")
    size = float(minimum_size_m) * 2 ** (levels - 1)
    xs, ys = zip(*domain.vertices_xy)
    x, y = math.floor(min(xs) / size) * size, math.floor(min(ys) / size) * size
    return asdict(NativeGrid(math.ceil((max(xs) - x) / size),
                            math.ceil((max(ys) - y) / size), x, y, size, size))


def default_document(dem_path, vertices_xy, crs, *, name="BRAIDS river", backend="cpu",
                     minimum_size_m=5., levels=3, mesh_mode="static_manual", end_time_s=60.):
    """Create a current hydrodynamics-only document, with no inferred boundaries.

    All optional physics is off. The initial surface is 0 m in the DEM datum,
    velocity is zero, and exterior edges are walls until explicitly assigned.
    The caller must review initial conditions for their terrain. Fresh checkpoint
    cadence equals the end time; the GUI records its automatic editor choice.
    """
    if mesh_mode not in ("dynamic_depth", "static_terrain", "static_manual"):
        raise ValueError("Choose static_manual, static_terrain or dynamic_depth mesh mode")
    if type(end_time_s) not in (int, float) or not math.isfinite(end_time_s) or end_time_s <= 0:
        raise ValueError("Simulation duration must be positive finite seconds")
    domain = PolygonDomain("desktop-domain", vertices_xy, crs)
    columns = ColumnMethodsV101(("sand",), (2650.,), (1590.,), (.0005,),
                               transport_thickness_m=.02, underlayer_thickness_m=.2)
    settings = dict(
        domain=dict(domain_id=domain.domain_id, vertices_xy=domain.vertices_xy,
                    spatial_reference=domain.crs_wkt),
        assignments=[], grid=geometry_grid(domain, minimum_size_m, levels), levels=levels,breaklines=[],solids=[],
        controls=asdict(CurrentRiverControls(45., .0005, 1., 0., 0., 0., threads=4, cfl=.225)),
        columns=asdict(columns), hydraulics={}, sediment_boundaries={},
        bank=dict(bank_theta=0., collapse=False, collapse_slope=.5, repose_angle_deg=32.),
        suspension=dict(bed_factor=1., sus_factor=0., diffusivity=0., temperature=20., cref=1600.),
        adaptation=asdict(FieldSettings(terrain_enabled=mesh_mode == "static_terrain",domain_border_enabled=False)),
        adaptive=mesh_mode == "dynamic_depth", end_time_s=float(end_time_s), dt_max_s=1.,
        output_interval_s=min(10., float(end_time_s)), checkpoint_interval_s=float(end_time_s),sediment_start_time_s=0.)
    request=new_model(backend=backend)
    request['hydrodynamics']['roughness']='roughness.manning'
    document = dict(
        schema=SCHEMA, name=name, model=resolve_model(request).document(),turbines=None,
        settings=settings, terrain=dict(dem_path=str(dem_path)), bank_survey=None,material_zones=dict(roughness=[],erodible=[]),
        initial=dict(stage_m=0., velocity_m_s=[0., 0.], layer_thickness_m=[.02, .2],
                     layer_fractions=[[1.], [1.]], concentration_kg_m3=[0.]),
        slope_parameters=deepcopy(SLOPE_PARAMETERS), hiding_parameters=deepcopy(HIDING_PARAMETERS),
        roughness_parameters=deepcopy(ROUGHNESS_PARAMETERS), bedform_methods=deepcopy(BEDFORM_METHODS),
        dependent_layer_parameters=deepcopy(DEPENDENT_LAYER_PARAMETERS),
        van_rijn_reference_parameters=deepcopy(VAN_RIJN_REFERENCE_PARAMETERS),
        skin_shear_parameters=deepcopy(SKIN_SHEAR_PARAMETERS),
        wet_avalanche_parameters=deepcopy(WET_AVALANCHE_PARAMETERS),
        column_methods=method_ids(columns), sediment_schedules={}, boundary_states={}, boundary_resolution={})
    return validate_document(json.loads(json.dumps(document)))


def validate_document(document):
    """Return a detached, fully validated current document (never normalize it)."""
    return validate_project(document)


def _resolve_boundary_state(document):
    settings = document["settings"]
    kinds = {a["bc_id"]: a["kind"] for a in settings["assignments"] if a["kind"] != "wall"}
    for name in ("hydraulics", "sediment_boundaries"):
        settings[name] = {key: value for key, value in settings[name].items() if key in kinds}
    for name in ("boundary_states", "sediment_schedules"):
        document[name] = {key: value for key, value in document[name].items() if key in kinds}
    document["boundary_resolution"] = resolve_boundaries(
        settings["hydraulics"], document["boundary_states"], kinds,
        sediment=document["model"]["physics"]["sediment"],
        ng=len(settings["columns"]["fraction_ids"]), end_time=settings["end_time_s"])
    return document


def assign_boundary(document, edge_ids, name, kind, controls=None,*,for_editing=False):
    """Replace a named section atomically, not accumulate its old faces.

    Selected edges are removed from any previous section. An invalid selection
    (stale, noncollinear or disconnected inlet/outlet) fails without
    changing the supplied document. Wall edges are allowed to remain cropped.
    """
    result = deepcopy(document)
    settings = result["settings"]
    domain = PolygonDomain(**settings["domain"])
    edge_ids = list(edge_ids)
    if not edge_ids or len(set(edge_ids)) != len(edge_ids):
        raise ValueError("Select at least one distinct domain edge")
    if set(edge_ids) - {edge.edge_id for edge in domain.edges}:
        raise ValueError("Unknown or stale domain edge; select the current geometry")
    rows = [asdict(BoundaryAssignment(edge, name, kind)) for edge in edge_ids]
    settings["assignments"] = [row for row in settings["assignments"]
                               if row["edge_id"] not in edge_ids and row["bc_id"] != name] + rows
    if kind != "wall":
        previous = settings["hydraulics"].get(name)
        if controls is None:
            controls = (previous if previous and previous["method"].startswith(kind + ".") else
                        dict(method="inlet.critical_discharge", discharge_m3_s=0.) if kind == "inlet" else
                        dict(method="outlet.stage", stage_m=result["initial"]["stage_m"]))
        settings["hydraulics"][name] = deepcopy(controls)
        sediment = result["model"]["physics"]["sediment"] is not None
        if previous is None or previous["method"] != controls["method"]:
            result["boundary_states"][name] = new_state(controls["method"], sediment=sediment)
        ng = len(settings["columns"]["fraction_ids"])
        if name not in settings["sediment_boundaries"] or (previous and not previous["method"].startswith(kind + ".")):
            settings["sediment_boundaries"][name] = (dict(kind="feed", solid_flux_m2_s=[0.] * ng,
                concentration_kg_m3=[0.] * ng) if kind == "inlet" else
                dict(kind="outflow", concentration_kg_m3=[0.] * ng))
            result["sediment_schedules"][name] = dict(method="forcing.constant")
    result=_resolve_boundary_state(result)
    return editable_geometry(result) if for_editing else validate_document(result)


def editable_geometry(document):
    """Validate editable geometry, including straight oblique open sections."""
    from braids_core.boundaries.domain import boundary_polyline
    result=deepcopy(document);domain=PolygonDomain(**result['settings']['domain'])
    rows=[BoundaryAssignment(**a) for a in result['settings']['assignments']]
    domain.boundary_segments(rows)
    for name in {a.bc_id for a in rows if a.kind!='wall'}:boundary_polyline(domain,rows,name)
    from braids_core.boundaries.solids import validate_solids
    validate_solids(result['settings']['solids'],domain)
    return result


def replace_domain(document, vertices_xy, crs=None,*,for_editing=False):
    """Explicit new geometry clears all prior assignments; no stale edge reuse."""
    result = deepcopy(document)
    old = result["settings"]["domain"]
    domain = PolygonDomain(old["domain_id"], vertices_xy, crs or old["spatial_reference"])
    settings = result["settings"]
    fine = settings["grid"]["dx_m"] / 2 ** (settings["levels"] - 1)
    settings["domain"] = dict(domain_id=domain.domain_id, vertices_xy=domain.vertices_xy,
                              spatial_reference=domain.crs_wkt)
    settings["grid"] = geometry_grid(domain, fine, settings["levels"])
    settings["assignments"] = []
    result=_resolve_boundary_state(result)
    return editable_geometry(result) if for_editing else validate_document(result)


def clear_boundary(document, name,*,for_editing=False):
    """Remove one named assignment; its former edges become ordinary walls."""
    result = deepcopy(document)
    result["settings"]["assignments"] = [a for a in result["settings"]["assignments"] if a["bc_id"] != name]
    result=_resolve_boundary_state(result)
    return editable_geometry(result) if for_editing else validate_document(result)


def aligned_document(document, new_domain, mapped, grid, *, new_assignment=None,for_editing=False):
    """Build a reviewable alignment proposal without changing physical forcing.

    The core alignment derives a new domain identity and exact edge mapping.
    Preserve that mapping (including any partial intervals), and retain all
    controls, exterior states, feeds and schedules for each unchanged named
    role. Only a genuinely new name or inlet/outlet role receives defaults.
    ``new_assignment`` is the optional UI proposal, not a second geometry edit.
    """
    if not isinstance(new_domain, PolygonDomain):
        raise TypeError("The derived PolygonDomain is required")
    result = deepcopy(document)
    settings = result["settings"]
    original_kinds = {row["bc_id"]: row["kind"] for row in settings["assignments"]}
    assignments = [row if isinstance(row, BoundaryAssignment) else BoundaryAssignment(**row) for row in mapped]
    # Check exact edge IDs, intervals, overlap and conflicting named roles before
    # resolving any numerical controls. No ID reconstruction or nearest matching.
    new_domain.boundary_segments(assignments)
    settings["domain"] = dict(domain_id=new_domain.domain_id, vertices_xy=new_domain.vertices_xy,
                              spatial_reference=new_domain.crs_wkt)
    settings["grid"] = deepcopy(grid)
    settings["assignments"] = [asdict(row) for row in assignments]
    kinds = {row.bc_id: row.kind for row in assignments if row.kind != "wall"}
    if new_assignment is not None:
        _, name, kind = new_assignment
        if kind != "wall" and kinds.get(name) != kind:
            raise ValueError("The derived mapping lost the proposed named boundary")
    sediment = result["model"]["physics"]["sediment"] is not None
    ng = len(settings["columns"]["fraction_ids"])
    for name, kind in kinds.items():
        if original_kinds.get(name) == kind:
            continue
        control = (dict(method="inlet.critical_discharge", discharge_m3_s=0.) if kind == "inlet" else
                   dict(method="outlet.stage", stage_m=result["initial"]["stage_m"]))
        settings["hydraulics"][name] = control
        result["boundary_states"][name] = new_state(control["method"], sediment=sediment)
        settings["sediment_boundaries"][name] = (dict(kind="feed", solid_flux_m2_s=[0.] * ng,
            concentration_kg_m3=[0.] * ng) if kind == "inlet" else
            dict(kind="outflow", concentration_kg_m3=[0.] * ng))
        result["sediment_schedules"][name] = dict(method="forcing.constant")
    result=_resolve_boundary_state(result)
    return editable_geometry(result) if for_editing else validate_document(result)


def save_project_folder(root, document, *, source_base=None, session=None, **kwargs):
    """Save all inputs/revisions using the shared folder contract; return resolved paths."""
    document=validate_document(document)
    # Windows antivirus/indexing can briefly hold a freshly staged directory.
    # Retry only an unpublished NEW folder: never repeat a partially published
    # update, overwrite a folder created by another session, or ignore a lock.
    import time
    for attempt in range(5):
        try:
            river_folder.save_folder(root, document, source_base=source_base,
                                     session=session, **kwargs)
            break
        except PermissionError as exc:
            if attempt==4 or getattr(exc,'winerror',None) not in (5,32,33) or Path(root).exists():raise
            if kwargs.get('cancelled') and kwargs['cancelled']():raise InterruptedError('Save cancelled') from exc
            time.sleep(.05*2**attempt)
    return river_folder.load_folder(root)


def load_project_folder(root):
    return river_folder.load_folder(root)


def project_history(root):
    return river_folder.list_history(root)


def project_session(root):
    if (Path(root)/river_folder.MANIFEST_FILE).is_file():return deepcopy(river_folder.folder_info(root)["session"])
    from .workspace import read_editor
    state=read_editor(root,verify_assets=False)
    return deepcopy(state['session']) if state else {}


def prepared_mesh(prepared):
    """Display exactly the prepared initial leaves; preparation is reused by Run."""
    from braids_core.river_worker import load_prepared
    from braids_core.sediment import adaptive_river_worker as worker
    project,receipt=load_prepared(prepared)
    metadata=Path(prepared)/'DESKTOP_PREPARATION.json'
    if metadata.is_file():
        import numpy as np
        from . import preparation
        from braids_core.sediment.river_leaf_io import sha
        display=json.loads(metadata.read_text(encoding='utf-8'));path=Path(prepared)/'MESH_PREVIEW.npz'
        if display.get('helper_sha256')!=sha(preparation.__file__) or display.get('preview_sha256')!=sha(path):
            raise ValueError('Prepared display geometry checksum/identity differs; prepare again')
        with np.load(path,allow_pickle=False) as f:
            vertices=f['vertices'];offsets=f['offsets'];faces=int(f['faces'])
        if (vertices.ndim!=2 or vertices.shape[1]!=2 or not np.isfinite(vertices).all() or
                offsets.ndim!=1 or len(offsets)!=display['cells']+1 or offsets[0]!=0 or
                offsets[-1]!=len(vertices) or np.any(np.diff(offsets)<3) or faces<=0):
            raise ValueError('Prepared display geometry shape differs; prepare again')
        return [vertices[a:b].tolist() for a,b in zip(offsets[:-1],offsets[1:])],faces
    profile,initial,tree=worker.load_prepared(Path(prepared)/'core/prepared')
    survey=json.loads((Path(prepared)/'BANK_SURVEY.json').read_text())
    required=() if survey is None else tuple(survey['mapping']['required_leaves'])
    turbines=json.loads((Path(prepared)/'TURBINES.json').read_text())
    if turbines is not None:required=tuple(sorted(set(required)|set(turbines['required_leaves'])))
    leaves=worker.initial_leaves(profile,initial,tree,required)
    mesh=tree.select(leaves)
    return [c.vertices_xy for c in mesh.cells],len(mesh.faces)


def update_project_session(root, **values):
    session = project_session(root)
    session.update(values)
    return river_folder.update_session(root, session)


def project_member(root, path):
    """Contained project member, following symlinks before the containment test."""
    root = Path(root).resolve()
    target = Path(path)
    target = (root / target).resolve() if not target.is_absolute() else target.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Select a file or directory inside this BRAIDS project folder")
    return target
