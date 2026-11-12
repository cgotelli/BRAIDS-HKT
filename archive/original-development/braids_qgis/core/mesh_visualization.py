"""Create QGIS vector products from the exact native cut-cell mesh."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from osgeo import ogr, osr

from braids_core.boundaries.agglomeration import agglomerate_small_cells
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.geometry_io import geometry_artifact,load_geometry
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.windows_paths import filesystem_path


MESH_LAYER_NAME = "braids_computational_mesh"


def build_initial_mesh(domain: PolygonDomain, grid: NativeGrid,
                       minimum_area_fraction: float = .5, obstacles=()):
    """Build the same unlabelled cell geometry used during run preparation."""
    mesh = map_cut_cells(domain, [], grid, obstacles)
    agglomeration = None
    if minimum_area_fraction > 0:
        mesh, _, agglomeration = agglomerate_small_cells(mesh, minimum_area_fraction)
    return mesh, agglomeration


def _value(cell: Any, name: str, default=None):
    return cell.get(name, default) if isinstance(cell, Mapping) else getattr(cell, name, default)


def _snapshot_rows(geometry: Mapping[str, Any]):
    snapshots = geometry.get("mesh_snapshots")
    if not snapshots:
        return [{"mesh_id": 0, "time_s": 0., "cells": geometry["cells"]}]
    rows = []
    for index, snapshot in enumerate(snapshots):
        cells = snapshot.get("cells")
        if not cells:
            raise ValueError(f"Mesh snapshot {index} contains no cells")
        rows.append({"mesh_id": int(snapshot.get("mesh_id", index)),
                     "time_s": float(snapshot.get("time_s", 0.)), "cells": cells})
    ids = [row["mesh_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Mesh snapshot IDs must be unique")
    return rows


def _frame_mesh_ids(geometry: Mapping[str, Any], frame_times_s: Sequence[float], snapshots):
    explicit = geometry.get("frame_mesh_ids")
    available = {row["mesh_id"] for row in snapshots}
    if explicit is not None:
        ids = [int(value) for value in explicit]
        if len(ids) != len(frame_times_s):
            raise ValueError("frame_mesh_ids must contain one mesh ID per saved result frame")
        if not set(ids) <= available:
            raise ValueError("frame_mesh_ids references an unavailable mesh snapshot")
        return ids
    ordered = sorted((row["time_s"], row["mesh_id"]) for row in snapshots)
    return [max((row for row in ordered if row[0] <= float(time_s)), default=ordered[0])[1]
            for time_s in frame_times_s]


def _write_mesh(path: Path, crs_wkt: str, snapshots, *, progress=None,
                cancelled=None, progress_range=(0.,100.)) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if driver is None:
        raise RuntimeError("QGIS GDAL installation does not provide the GeoPackage driver")
    if path.exists():
        driver.DeleteDataSource(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = driver.CreateDataSource(os.fspath(path))
    if dataset is None:
        raise RuntimeError(f"Could not create mesh dataset {path}")
    spatial = osr.SpatialReference()
    if spatial.ImportFromWkt(str(crs_wkt)) != ogr.OGRERR_NONE:
        dataset = None
        raise ValueError("Mesh CRS is not valid WKT")
    layer = dataset.CreateLayer(MESH_LAYER_NAME, spatial, ogr.wkbPolygon)
    if layer is None:
        dataset = None
        raise RuntimeError("Could not create computational mesh layer")
    for name, kind in (("mesh_id", ogr.OFTInteger), ("cell_id", ogr.OFTInteger64),
                       ("level", ogr.OFTInteger), ("time_s", ogr.OFTReal),
                       ("area_m2", ogr.OFTReal)):
        if layer.CreateField(ogr.FieldDefn(name, kind)) != ogr.OGRERR_NONE:
            dataset = None
            raise RuntimeError(f"Could not create mesh field {name}")
    definition = layer.GetLayerDefn()
    total=sum(len(snapshot["cells"]) for snapshot in snapshots)
    completed=0;first_progress,last_progress=map(float,progress_range)
    report_every=max(1,total//200)
    layer.StartTransaction()
    try:
        for snapshot in snapshots:
            for cell in snapshot["cells"]:
                if cancelled is not None and completed % report_every == 0 and cancelled():
                    raise InterruptedError("Adaptive mesh preparation cancelled")
                vertices = [tuple(map(float, point[:2])) for point in _value(cell, "vertices_xy")]
                if len(vertices) < 3:
                    raise ValueError("A computational cell needs at least three vertices")
                if vertices[0] != vertices[-1]:
                    vertices.append(vertices[0])
                polygon = ogr.Geometry(ogr.wkbPolygon)
                for points in (vertices,*(_value(cell,"holes_xy",()) or ())):
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for x, y in points:
                        ring.AddPoint_2D(float(x),float(y))
                    ring.CloseRings();polygon.AddGeometry(ring)
                feature = ogr.Feature(definition)
                feature["mesh_id"] = snapshot["mesh_id"]
                feature["cell_id"] = int(_value(cell, "cell_id"))
                feature["level"] = int(_value(cell, "level", 0))
                feature["time_s"] = snapshot["time_s"]
                feature["area_m2"] = float(_value(cell, "area_m2", polygon.GetArea()))
                feature.SetGeometry(polygon)
                if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
                    raise RuntimeError("Could not write a computational mesh cell")
                completed+=1
                if progress is not None and (completed % report_every == 0 or completed == total):
                    progress(first_progress+(last_progress-first_progress)*completed/max(1,total))
        if layer.CommitTransaction() != ogr.OGRERR_NONE:
            raise RuntimeError("Could not commit computational mesh cells")
    except Exception:
        layer.RollbackTransaction()
        dataset = None
        path.unlink(missing_ok=True)
        raise
    dataset = None


def write_initial_mesh(path: str | Path, domain: PolygonDomain, grid: NativeGrid,
                       minimum_area_fraction: float = .5, obstacles=()) -> dict[str, Any]:
    mesh, agglomeration = build_initial_mesh(domain, grid, minimum_area_fraction,obstacles)
    snapshot = {"mesh_id": 0, "time_s": 0., "cells": mesh.cells}
    path = filesystem_path(Path(path).expanduser()).resolve()
    _write_mesh(path, mesh.crs_wkt, [snapshot])
    return {"path": str(path), "layer": MESH_LAYER_NAME, "mesh_count": 1,
            "cell_count": len(mesh.cells), "partial_cell_count": mesh.metrics()["partial_cell_count"],
            "minimum_area_fraction": minimum_area_fraction,
            "obstacle_count":len(mesh.obstacle_ids),"solid_area_m2":mesh.solid_area_m2,
            "merged_cell_count": 0 if agglomeration is None else agglomeration["merged_cell_count"],
            "unresolved_small_cell_count": 0 if agglomeration is None else len(agglomeration["unresolved_small_cells"])}


def write_result_mesh(path: str | Path, geometry: Mapping[str, Any],
                      frame_times_s: Sequence[float], *, progress=None,
                      cancelled=None, progress_range=(0.,100.)) -> dict[str, Any]:
    """Write fixed or future adaptive meshes and map every frame to a mesh ID."""
    snapshots = _snapshot_rows(geometry)
    frame_ids = _frame_mesh_ids(geometry, frame_times_s, snapshots)
    path = filesystem_path(Path(path).expanduser()).resolve()
    expected = sum(len(row["cells"]) for row in snapshots)
    existing = ogr.Open(os.fspath(path)) if path.is_file() else None
    existing_layer = None if existing is None else existing.GetLayerByName(MESH_LAYER_NAME)
    valid_existing = existing_layer is not None and existing_layer.GetFeatureCount() == expected
    existing = None
    if not valid_existing:
        _write_mesh(path, str(geometry["crs_wkt"]), snapshots,progress=progress,
                    cancelled=cancelled,progress_range=progress_range)
    elif progress is not None:
        progress(float(progress_range[1]))
    levels = sorted({int(_value(cell, "level", 0)) for row in snapshots for cell in row["cells"]})
    return {"path": path.name, "layer": MESH_LAYER_NAME,
            "fixed": len(snapshots) == 1 and len(set(frame_ids)) <= 1,
            "mesh_count": len(snapshots), "feature_count": expected,
            "cell_counts": {str(row["mesh_id"]): len(row["cells"]) for row in snapshots},
            "levels": levels, "frame_mesh_ids": frame_ids}


def write_adaptive_initial_mesh(path,project,*,progress=None,cancelled=None):
    """Preview the exact native initial selection, using all physical inputs."""
    import json
    import os
    import subprocess
    import tempfile
    from .project import prepare_run,save_project,verified_native_executable
    from .results import read_frames
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.braids_mesh_',dir=path.parent) as temporary:
        root=Path(temporary)
        project_path=save_project(root/'project.json',project)
        run=prepare_run(project_path,root/'prepared',progress=progress,cancelled=cancelled)
        if cancelled and cancelled():raise InterruptedError('Mesh preview cancelled')
        environment={**os.environ,'OMP_NUM_THREADS':str(project['numerics']['threads']),
                     'OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'CLOSE','OMP_PLACES':'CORES',
                     'OMP_WAIT_POLICY':'PASSIVE'}
        environment.pop('BRAIDS_TIME_INTEGRATOR',None)
        executable=verified_native_executable(run)
        result=subprocess.run([str(executable),'case.data','--initial-mesh'],cwd=run,
            capture_output=True,text=True,timeout=60,env=environment)
        if result.returncode:
            raise ValueError('Adaptive initial mesh failed: '+result.stdout+result.stderr)
        frame=read_frames(run/'frames.txt')[0]
        tree=load_geometry(geometry_artifact(run,'amr_hierarchy'),
            cell_ids=frame['node_ids'],include_faces=False)
        cells=[{**cell,'cell_id':i} for i,cell in enumerate(tree['cells'])]
        _write_mesh(path,tree['crs_wkt'],[{'mesh_id':frame['mesh_id'],'time_s':0.,'cells':cells}])
        geometry=load_geometry(geometry_artifact(run,'geometry'),cell_ids=[],include_faces=False)
        amr=project['numerics']['amr'];levels=int(amr.get('levels',4))
        ratio=int(amr.get('refinement_ratio',2));finest=levels-1
        return {'path':str(path),'layer':MESH_LAYER_NAME,'mesh_count':1,'cell_count':len(cells),
            'partial_cell_count':sum(c['area_m2']<c['nominal_area_m2']*(1-1e-12) for c in cells),
            'merged_cell_count':sum(max(0,len(c['member_indices'])//(
                ratio**(finest-c['level']))**2-1) for c in cells),
            'obstacle_count':len(geometry.get('obstacle_ids',[])),
            'level_counts':{str(level):sum(c['level']==level for c in cells) for level in range(levels)}}
