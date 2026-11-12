"""Create, prepare and qualify a 10 m x 2 m Jiang--Cao flume project."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "desktop/runtime"
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Trapezoid231"
OUT = ROOT / "validation/_trapezoid231"
SOURCE = OUT / "inputs"
PYTHON = ROOT / "desktop/.venv/Scripts/python.exe"
MANIFEST = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"
SOURCE_ID = "e89341080f14ed1f87ea30e790784b43b086ede2a1ef55083c5aed20d00b71a2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def install_imports() -> None:
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        QT_QPA_PLATFORM="offscreen", OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)


def normal_depth(discharge: float, width: float, side: float,
                 manning: float, slope: float) -> float:
    def flow(depth: float) -> float:
        area = width * depth + side * depth * depth
        perimeter = width + 2.0 * depth * math.sqrt(1.0 + side * side)
        return area * (area / perimeter) ** (2.0 / 3.0) * math.sqrt(slope) / manning
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if flow(mid) < discharge:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_dem() -> tuple[Path, dict]:
    from osgeo import gdal, osr
    from scipy.ndimage import gaussian_filter

    SOURCE.mkdir(parents=True, exist_ok=True)
    pixel = 0.0125
    x0, y0 = -0.25, 2.25
    nx, ny = 840, 200
    x = x0 + (np.arange(nx) + 0.5) * pixel
    y = y0 - (np.arange(ny) + 0.5) * pixel
    xx, yy = np.meshgrid(x, y)
    taper = np.clip(np.minimum(xx, 10.0 - xx) / 0.5, 0.0, 1.0)
    center = 1.0 + taper * (
        0.015 * np.sin(2.0 * np.pi * xx / 3.1)
        + 0.008 * np.sin(2.0 * np.pi * xx / 1.7 + 0.4)
    )
    bottom_width, side_slope, bank_depth = 0.10, 2.0, 0.08
    transverse = np.clip(
        (np.abs(yy - center) - 0.5 * bottom_width) / side_slope,
        0.0, bank_depth,
    )
    rng = np.random.default_rng(231)
    fine = gaussian_filter(rng.normal(size=(ny, nx)), sigma=7.0, mode="reflect")
    inside = (xx >= 0.0) & (xx <= 10.0) & (yy >= 0.0) & (yy <= 2.0)
    fine -= fine[inside].mean()
    fine *= 0.00010 / np.max(np.abs(fine[inside]))
    patches = np.zeros_like(xx)
    for xc, yc, amplitude, sx, sy in (
        (2.1, 0.99, 0.00035, 0.35, 0.10),
        (4.4, 1.02, -0.00045, 0.45, 0.12),
        (6.8, 0.98, 0.00040, 0.40, 0.11),
        (8.5, 1.01, -0.00030, 0.32, 0.10),
    ):
        patches += amplitude * np.exp(
            -0.5 * ((xx - xc) / sx) ** 2 - 0.5 * ((yy - yc) / sy) ** 2
        )
    irregularity = taper * (fine + patches)
    bed = 0.01 * (10.0 - xx) + transverse + irregularity
    dem = SOURCE / "trapezoidal_flume_1mm.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(dem), nx, ny, 1, gdal.GDT_Float64,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(2056)
    dataset.SetGeoTransform((x0, pixel, 0.0, y0, 0.0, -pixel))
    dataset.SetProjection(reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(bed)
    dataset = None

    depth = normal_depth(0.002, bottom_width, side_slope, 0.03, 0.01)
    design = {
        "domain_m": {"length": 10.0, "width": 2.0},
        "longitudinal_slope": 0.01,
        "trapezoid": {
            "bottom_width_m": bottom_width, "side_slope_h_to_v": side_slope,
            "bankfull_depth_m": bank_depth,
            "top_width_m": bottom_width + 2.0 * side_slope * bank_depth,
        },
        "normal_flow_design_check": {
            "water_discharge_m3_s": 0.002, "manning_s_m_one_third": 0.03,
            "normal_depth_m": depth,
            "normal_top_width_m": bottom_width + 2.0 * side_slope * depth,
            "freeboard_m": bank_depth - depth,
            "note": "Geometry check only; runtime resistance is the Jiang-Cao automatic bundle.",
        },
        "irregularity": {
            "seed": 231, "random_component_max_abs_m": 0.00010,
            "four_smooth_patches_max_abs_m": 0.00045,
            "centerline_waviness_max_m": 0.023,
            "boundary_taper_m": 0.5,
        },
        "dem": {"pixel_m": pixel, "path": str(dem)},
    }
    save(OUT / "DESIGN.json", design)
    return dem, design


def copy_runtime() -> None:
    manifest = json.loads((RUNTIME / "RUNTIME.json").read_text(encoding="utf8"))
    target = PROJECT / "runtime"
    for index, (name, digest) in enumerate(manifest["files"].items(), 1):
        source = RUNTIME / name
        if sha(source) != digest:
            raise ValueError(f"Installed runtime changed: {name}")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if index % 500 == 0:
            print(f"Frozen runtime {index}/{len(manifest['files'])}", flush=True)
    shutil.copy2(RUNTIME / "RUNTIME.json", target / "RUNTIME.json")
    if sha(target / "RUNTIME.json") != MANIFEST:
        raise ValueError("Frozen project runtime identity differs")


def create_document(dem: Path, design: dict) -> dict:
    from PySide6.QtWidgets import QApplication
    from braids_desktop import services
    from braids_desktop.forms import SettingsPanel
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_model import new_model, new_bedload_selection, resolve_model
    from braids_core.river_jiang_cao import METHOD, DEFAULTS, COLLAPSE, ROUGHNESS
    from braids_core.river_open_boundary import new_state
    from braids_core.river_project import validate_project
    from braids_core.sediment.bed_columns_v101 import ColumnMethodsV101
    from braids_core.sediment.river_column_methods import method_ids

    app = QApplication.instance() or QApplication([])
    document = services.default_document(
        str(dem), [(0, 0), (10, 0), (10, 2), (0, 2)], "EPSG:2056",
        backend="cpu", minimum_size_m=0.05, levels=1,
        mesh_mode="static_terrain", end_time_s=18000.0,
    )
    water_discharge = 0.002
    ratio = 0.0004316981132075471
    solid_discharge = ratio * water_discharge
    mixture_discharge = water_discharge + solid_discharge
    domain = PolygonDomain(**document["settings"]["domain"])
    section_start, section_end = 0.79, 1.21
    for x, name, kind, control in (
        (0.0, "inlet", "inlet", {
            "method": "inlet.normal_discharge",
            "discharge_m3_s": mixture_discharge, "energy_slope": "mesh",
        }),
        (10.0, "outlet", "outlet", {"method": "outlet.critical"}),
    ):
        edge = next(e.edge_id for e in domain.edges if e.start_xy[0] == e.end_xy[0] == x)
        document = services.assign_boundary(document, [edge], name, kind, control)
        assignment = next(row for row in document["settings"]["assignments"]
                          if row["bc_id"] == name)
        assignment.update(start_m=section_start, end_m=section_end)

    model = new_model(backend="cpu")
    model["hydrodynamics"].update(
        secondary_flow="secondary.prognostic", roughness=ROUGHNESS,
    )
    selection = new_bedload_selection()
    selection["capacity"] = {
        "method": METHOD, "parameters": dict(DEFAULTS),
        "grain_sizes_m": {"d50": 0.001, "d90": 0.001},
    }
    selection["grains"][0]["diameter_m"] = 0.001
    selection["avalanche"] = COLLAPSE
    model["sediment"] = {"enabled": True, "selection": selection}
    document["model"] = resolve_model(model).document()
    columns = replace(
        ColumnMethodsV101(**document["settings"]["columns"]),
        fraction_ids=("sand",), diameter_m=(0.001,), bed="mixed", base_update=1,
    )
    document["settings"]["columns"] = asdict(columns)
    document["column_methods"] = method_ids(columns)
    document["settings"]["controls"].update(
        diameter=0.001, threads=8, morfac=1.0, betac=0.0,
        diffusivity=0.0, viscosity=0.0, cfl=0.5,
    )
    document["settings"]["adaptation"].update(
        domain_border_enabled=False, interval=1.0,
    )
    document["settings"].update(
        adaptive=False, dt_max_s=0.03, output_interval_s=60.0,
        checkpoint_interval_s=300.0,
    )
    document["initial"].update(
        stage_m=0.0, velocity_m_s=[0.0, 0.0], layer_thickness_m=[0.2],
        layer_fractions=[[1.0]],
    )
    for name in ("inlet", "outlet"):
        document["boundary_states"][name] = new_state(
            document["settings"]["hydraulics"][name]["method"], sediment=True,
        )
    document["boundary_states"]["inlet"]["composition"] = {
        "method": "composition.reference_volume_fraction", "fractions": [1.0],
    }
    document["settings"]["sediment_boundaries"]["inlet"] = {
        "kind": "inlet_solid_water_ratio", "solid_water_ratio": [ratio],
        "concentration_kg_m3": [0.0],
    }
    document = services._resolve_boundary_state(document)
    panel = SettingsPanel()
    panel.load_document(document)
    document = panel.document(document)
    panel.close()
    document["settings"]["checkpoint_interval_s"] = 300.0
    document["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 300.0,
    }
    document["name"] = "Jiang-Cao 10 m trapezoidal low-flow channel - 2 L/s"
    document = validate_project(document)
    audit = {
        "schema": "braids.trapezoidal-flume.v1",
        "runtime_manifest_sha256": MANIFEST, "source_identity": SOURCE_ID,
        "model": "bundle.jiang_cao_p15_bulk_diameters213_v1",
        "material": {"d50_m": 0.001, "d90_m": 0.001,
                     "solid_density_kg_m3": 2650.0, "erodible_depth_m": 0.2},
        "water_discharge_m3_s": water_discharge,
        "solid_water_ratio": ratio, "solid_discharge_m3_s": solid_discharge,
        "mixture_boundary_discharge_m3_s": mixture_discharge,
        "inlet_outlet_section_station_m": [section_start, section_end],
        "mesh": {"fixed": True, "dx_m": 0.05, "dy_m": 0.05,
                 "cells": 8000},
        "clocks": {"end_s": 18000.0, "output_s": 60.0,
                   "checkpoint_s": 300.0, "cfl": 0.5, "dt_max_s": 0.03},
        "design": design,
        "limitations": [
            "Synthetic geometry, not an experimental reproduction.",
            "Uniform D50=D90 idealization; no transported size classes.",
            "Manning normal depth is a geometry check only; Jiang-Cao owns runtime resistance.",
        ],
    }
    return document, audit


def figure(dem: Path, design: dict) -> None:
    from osgeo import gdal
    import matplotlib.pyplot as plt
    dataset = gdal.Open(str(dem))
    bed = dataset.GetRasterBand(1).ReadAsArray()
    transform = dataset.GetGeoTransform()
    x = transform[0] + (np.arange(bed.shape[1]) + 0.5) * transform[1]
    y = transform[3] + (np.arange(bed.shape[0]) + 0.5) * transform[5]
    domain_x = (x >= 0.0) & (x <= 10.0)
    domain_y = (y >= 0.0) & (y <= 2.0)
    z = bed[np.ix_(domain_y, domain_x)]
    xx, yy = x[domain_x], y[domain_y]
    row = np.argmin(np.abs(xx - 5.0))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    image = axes[0].pcolormesh(xx, yy, z, shading="nearest", cmap="terrain")
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Synthetic DEM")
    fig.colorbar(image, ax=axes[0], label="bed elevation (m)")
    local = z[:, row] - 0.01 * (10.0 - xx[row])
    axes[1].plot(yy, local, color="black", lw=1.5)
    depth = design["normal_flow_design_check"]["normal_depth_m"]
    axes[1].axhline(depth, color="tab:blue", ls="--", label="2 L/s design water level")
    axes[1].set(xlabel="transverse coordinate y (m)", ylabel="local elevation (m)",
                title="Cross-section near x = 5 m")
    axes[1].legend()
    fig.savefig(OUT / "DEM_AND_CROSS_SECTION.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if PROJECT.exists() or OUT.exists():
        raise FileExistsError("Preserve existing trapezoidal-channel project/evidence")
    if sha(RUNTIME / "RUNTIME.json") != MANIFEST:
        raise ValueError("Installed preview51 runtime required")
    install_imports()
    from braids_core.river_worker import source_identity
    if source_identity() != SOURCE_ID:
        raise ValueError("Installed preview51 source identity differs")
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core import river_worker

    dem, design = make_dem()
    document, audit = create_document(dem, design)
    services.save_project_folder(PROJECT, document, source_base=SOURCE)
    for name in ("logs", "qualification", "references"):
        (PROJECT / name).mkdir(exist_ok=True)
    save(PROJECT / "INPUT_AUDIT.json", audit)
    save(OUT / "INPUT_AUDIT.json", audit)
    copy_runtime()
    for name in ("launch.py", "worker.py"):
        shutil.copy2(ROOT / "desktop" / name, PROJECT / name)
    shutil.copy2(ROOT / "desktop/run_trapezoid231.py", PROJECT / "run_trapezoid231.py")
    (PROJECT / "Open project.cmd").write_text(
        '@echo off\nstart "" "' + str(PYTHON.with_name("pythonw.exe"))
        + '" -B "%~dp0launch.py" "%~dp0."\n', encoding="utf8",
    )
    (PROJECT / "Start long run.cmd").write_text(
        '@echo off\nstart "BRAIDS trapezoid" /b "' + str(PYTHON)
        + '" -X utf8 "%~dp0run_trapezoid231.py" run '
        + '> "%~dp0logs\\trapezoid231_18000.log" 2>&1\n', encoding="utf8",
    )
    (PROJECT / "Check long run.cmd").write_text(
        '@echo off\n"' + str(PYTHON)
        + '" -X utf8 "%~dp0run_trapezoid231.py" status\n', encoding="utf8",
    )
    (PROJECT / "Stop long run safely.cmd").write_text(
        '@echo off\ntype nul > "%~dp0STOP_18000.request"\n'
        'echo Stop requested. BRAIDS will seal a checkpoint before exiting.\npause\n',
        encoding="utf8",
    )
    (PROJECT / "README.md").write_text(
        "# Jiang-Cao trapezoidal low-flow channel\n\n"
        "Synthetic 10 m x 2 m fixed-grid project with a central 0.08 m-deep "
        "trapezoid, 1 mm uniform sand and 2 L/s water discharge. Open with "
        "`Open project.cmd`. Use the check and safe-stop commands while the "
        "independent run is active. Exact assumptions are in `INPUT_AUDIT.json`.\n",
        encoding="utf8",
    )
    figure(dem, design)
    shutil.copy2(OUT / "DEM_AND_CROSS_SECTION.png", PROJECT / "DEM_AND_CROSS_SECTION.png")

    # Installed short gate before preparing the independent run.
    gate = OUT / "gate"
    short = deepcopy(document)
    short["settings"].update(end_time_s=5.0, output_interval_s=5.0,
                             checkpoint_interval_s=5.0)
    short["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 5.0,
    }
    save(gate / "project.json", short)
    gate_prepared = prepare(gate / "project.json", gate / "prepared",
                            asset_cache=PROJECT / "cache")
    began = time.perf_counter()
    gate_result = river_worker.run(
        gate_prepared, gate / "run", checkpoint_dir=gate / "checkpoints",
    )
    gate_wall = time.perf_counter() - began
    if gate_result["status"] != "complete":
        raise RuntimeError(gate_result)
    prepared = prepare(PROJECT / "braids_project.json",
                       PROJECT / "prepared/trapezoid231",
                       asset_cache=PROJECT / "cache")
    services.update_project_session(
        PROJECT, prepared=prepared.relative_to(PROJECT).as_posix(),
        output="", log="logs/trapezoid231_18000.log",
    )
    qualification = {
        "runtime_manifest_sha256": MANIFEST, "source_identity": SOURCE_ID,
        "project": str(PROJECT), "prepared": str(prepared),
        "short_gate": gate_result, "short_gate_wall_s": gate_wall,
        "long_run_backend": "cpu", "threads": 8,
        "ready": True,
    }
    save(OUT / "QUALIFICATION.json", qualification)
    save(PROJECT / "QUALIFICATION.json", qualification)
    print(json.dumps(qualification, indent=2), flush=True)


if __name__ == "__main__":
    main()
