#!/usr/bin/env python3
"""Run a compact, reproducible native two-solver river-widening experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "braids_support/sediment/delft3d_current"))
from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction, write_cut_case
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.section import Section
from braids_qgis.core.results import read_frames
import runtime as delft_runtime


def load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


common = load_module(
    "synthetic_delft_common",
    ROOT / "braids_support/sediment/delft3d_intercomparison/settling_basin/run_comparison.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def native_frames_path(folder: Path) -> Path:
    """Use the actual solver format (small fixed grids write ASCII)."""
    for name in ('frames.bin', 'frames.txt'):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'No native frames in {folder}')


def specification() -> dict:
    return json.loads((HERE / "specification.json").read_text(encoding="utf-8"))


def verified_braids_build() -> tuple[Path, dict]:
    executable = ROOT / "solver/cut_cells/build/xbraids_cutcell.exe"
    manifest = ROOT / "solver/cut_cells/build/build.json"
    if not executable.is_file() or not manifest.is_file():
        raise FileNotFoundError("Build the current native solver with python tools/build_cutcell.py")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    changed = [source for source, digest in record["source_sha256"].items()
               if sha256(ROOT / source) != digest]
    if changed or sha256(executable) != record["executable_sha256"]:
        raise RuntimeError(f"Native BRAIDS executable is stale; rebuild before comparing: {changed}")
    return executable, record


def smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    value = np.clip(value, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * value))


def bottom_width(x, g):
    x = np.asarray(x, dtype=float)
    fraction = smoothstep((x - g["inlet_reach_end_m"]) / (g["expansion_end_m"] - g["inlet_reach_end_m"]))
    fraction -= smoothstep((x - g["wide_reach_end_m"]) / (g["contraction_end_m"] - g["wide_reach_end_m"]))
    return g["narrow_bottom_width_m"] + (g["wide_bottom_width_m"] - g["narrow_bottom_width_m"]) * fraction


def outer_width(x, g):
    return bottom_width(x, g) + 2.0 * g["bank_height_m"] * g["bank_side_slope_horizontal_per_vertical"]


def bed(x, y, g):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    width = bottom_width(x, g)
    bank = np.maximum(np.abs(y) - width / 2.0, 0.0) / g["bank_side_slope_horizontal_per_vertical"]
    middle = smoothstep((x - g["expansion_end_m"]) / 60.0) * (1.0 - smoothstep((x - g["wide_reach_end_m"] + 60.0) / 60.0))
    if g.get("bed_perturbation_mode", "two_sine") == "smooth":
        perturb = g["bed_perturbation_amplitude_m"] * middle * np.sin(
            2.0 * np.pi * x / 160.0 + 2.0 * np.pi * y / 100.0 + 0.37)
    else:
        perturb = g["bed_perturbation_amplitude_m"] * middle * (
            np.sin(2.0 * np.pi * x / 83.0 + 2.0 * np.pi * y / 27.0)
            * np.sin(2.0 * np.pi * y / 31.0 + 0.37)
        )
    return g["longitudinal_slope"] * (g["length_m"] - x) + bank + perturb


def section_area(depth: float, width: float, side: float) -> float:
    return width * depth + side * depth * depth


def critical_depth(q: float, width: float, side: float) -> float:
    low, high = 0.0, 10.0
    for _ in range(70):
        mid = (low + high) / 2.0
        area = section_area(mid, width, side)
        top = width + 2.0 * side * mid
        if q * q * top > 9.81 * area ** 3:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def normal_depth(q: float, width: float, side: float, n: float, slope: float) -> float:
    low, high = 0.0, 10.0
    for _ in range(70):
        mid = (low + high) / 2.0
        area = section_area(mid, width, side)
        perimeter = width + 2.0 * mid * math.sqrt(1.0 + side * side)
        conveyance = area * (area / perimeter) ** (2.0 / 3.0) * math.sqrt(slope) / n
        if conveyance < q:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def planning_mpm_capacity_kg_s(width: float, spec: dict) -> float:
    """Uniform-flow estimate only; native 2D transport is the actual result."""
    g, h, s = (spec[key] for key in ("geometry", "hydraulics", "sediment"))
    side, discharge, n = (g["bank_side_slope_horizontal_per_vertical"],
                          h["discharge_m3_s"], h["manning_n_s_m13"])
    depth = normal_depth(discharge, width, side, n, g["longitudinal_slope"])
    area = section_area(depth, width, side)
    perimeter = width + 2.0 * depth * math.hypot(1.0, side)
    tau = 1000.0 * 9.81 * area / perimeter * g["longitudinal_slope"]
    chezy = depth ** (1.0 / 6.0) / n
    grain_chezy = 18.0 * math.log10(12.0 * max(depth / s["d90_m"], 1.0))
    ripple_factor = min((chezy / grain_chezy) ** 1.5, 1.0)
    delta = (s["grain_density_kg_m3"] - 1000.0) / 1000.0
    excess = max(ripple_factor * tau / (1000.0 * delta * 9.81 * s["d50_m"]) - s["critical_shields"], 0.0)
    solid_flux_m2_s = 8.0 * math.sqrt(delta * 9.81 * s["d50_m"] ** 3) * excess ** 1.5
    return solid_flux_m2_s * width * s["grain_density_kg_m3"]


def initial_stage(x, g, h):
    width = bottom_width(x, g)
    depth = np.asarray([normal_depth(h["discharge_m3_s"], float(w), g["bank_side_slope_horizontal_per_vertical"], h["manning_n_s_m13"], g["longitudinal_slope"]) for w in np.ravel(width)]).reshape(np.shape(width))
    end_depth = critical_depth(h["discharge_m3_s"], g["narrow_bottom_width_m"], g["bank_side_slope_horizontal_per_vertical"])
    blend = 1.0 - smoothstep(x / 50.0) + smoothstep((x - g["length_m"] + 50.0) / 50.0)
    return g["longitudinal_slope"] * (g["length_m"] - x) + depth * (1.0 - blend) + end_depth * blend


def initial_velocity(x, g, h):
    """Use the *actual initialized stage* and bed to set section discharge."""
    x = np.asarray(x)
    unique, inverse = np.unique(x.ravel(), return_inverse=True)
    y = outer_width(unique, g)[:, None] * np.linspace(-0.5, 0.5, 1025)[None, :]
    stage = initial_stage(unique, g, h)[:, None]
    wet_area = np.trapezoid(np.maximum(stage - bed(unique[:, None], y, g), 0.0), y, axis=1)
    velocity = h["discharge_m3_s"] / np.maximum(wet_area, 1.0e-9)
    return velocity[inverse].reshape(x.shape)


def check_matched_resolution(g):
    dx = g["braids_cell_size_m"]
    longitudinal = g["length_m"] / g["delft3d_nx"]
    transverse = (g["wide_bottom_width_m"] + 2.0 * g["bank_height_m"] *
                  g["bank_side_slope_horizontal_per_vertical"]) / g["delft3d_ny"]
    if max(longitudinal, transverse) > dx + 1.0e-12:
        raise ValueError("Delft3D grid is coarser than the finest BRAIDS cell")
    return longitudinal, transverse


def domain_and_mesh(spec: dict):
    g = spec["geometry"]
    x = np.unique(np.concatenate((np.linspace(0.0, g["length_m"], 301), [g["inlet_reach_end_m"], g["expansion_end_m"], g["wide_reach_end_m"], g["contraction_end_m"]])))
    half = outer_width(x, g) / 2.0
    lower = tuple((float(a), float(-b)) for a, b in zip(x, half))
    upper = tuple((float(a), float(b)) for a, b in zip(x[::-1], half[::-1]))
    polygon = PolygonDomain("synthetic_expansion", lower + upper, "EPSG:2056")
    edges = []
    for edge in polygon.edges:
        if edge.start_xy[0] == edge.end_xy[0] == 0.0:
            edges.append(BoundaryAssignment(edge.edge_id, "inlet", "inlet"))
        elif edge.start_xy[0] == edge.end_xy[0] == g["length_m"]:
            edges.append(BoundaryAssignment(edge.edge_id, "outlet", "outlet"))
    dx = g["braids_cell_size_m"]
    ny = math.ceil((g["wide_bottom_width_m"] + 4.0) / dx)
    mesh = map_cut_cells(polygon, edges, NativeGrid(math.ceil(g["length_m"] / dx), ny, 0.0, -ny * dx / 2.0, dx, dx))
    return mesh


def prepare_braids(spec: dict, folder: Path, end_s: float, output_s: float,
                   hydraulic_only: bool = False,
                   spinup: Path | None = None,
                   hydraulic_seed: Path | None = None) -> tuple[int, Path]:
    _, build = verified_braids_build()
    if (folder / "case.data").exists() or (folder / "native.log").exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g, h, s, t = (spec[key] for key in ("geometry", "hydraulics", "sediment", "time"))
    mesh = domain_and_mesh(spec)
    # Native live critical control requires physical boundary-profile
    # endpoints; a separately declared Section alone freezes the target.
    from dataclasses import replace
    mesh = replace(mesh, faces=tuple(
        replace(face,
                bed_elevation_m=float(bed(face.start_xy[0], 0.5 * (face.start_xy[1] + face.end_xy[1]), g)),
                section_bed_start_m=float(bed(*face.start_xy, g)),
                section_bed_end_m=float(bed(*face.end_xy, g)))
        if face.kind == "inlet" else face for face in mesh.faces
    ))
    coords = np.asarray([cell.centroid_xy for cell in mesh.cells])
    elevation = bed(coords[:, 0], coords[:, 1], g)
    stage = initial_stage(coords[:, 0], g, h)
    depth = np.maximum(stage - elevation, 1.0e-4)
    velocity = initial_velocity(coords[:, 0], g, h)
    # The native roughness-corrected MPM path currently requires a layered
    # multi-fraction state. Use two numerically distinct but physically
    # identical 15 mm tracers (50/50): the material remains monodisperse.
    initial = np.zeros((len(mesh.cells), 9))
    initial[:, 0] = depth
    initial[:, 1] = depth * velocity
    bulk = s["grain_density_kg_m3"] * (1.0 - s["porosity"])
    initial[:, 5:7] = bulk * s["active_layer_m"] / 2.0
    initial[:, 7:9] = bulk * (s["mobile_layer_m"] - s["active_layer_m"]) / 2.0
    if hydraulic_seed is not None:
        if not hydraulic_only or spinup is not None:
            raise ValueError("A coarser hydraulic seed requires its own fixed-bed relaxation")
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        coarse = np.load(hydraulic_seed / "geometry.npz")
        state = read_frames(hydraulic_seed / "frames.txt")[-1]["q"]
        wet = state[:, 0] > 0.01
        values = np.column_stack((coarse["bed"][wet] + state[wet, 0],
                                  state[wet, 1:3] / state[wet, 0, None]))
        interpolated = LinearNDInterpolator(coarse["xy"][wet], values)(coords)
        missing = ~np.isfinite(interpolated).all(axis=1)
        interpolated[missing] = NearestNDInterpolator(coarse["xy"][wet], values)(coords[missing])
        initial[:, 0] = np.maximum(interpolated[:, 0] - elevation, 1.0e-4)
        initial[:, 1:3] = initial[:, 0, None] * interpolated[:, 1:3]
    if spinup is not None:
        spun_geometry = np.load(spinup / "braids" / "geometry.npz")
        if not np.allclose(spun_geometry["xy"], coords, atol=1.0e-9):
            raise ValueError("BRAIDS spin-up geometry does not match morphology geometry")
        if not np.allclose(spun_geometry["bed"], elevation, atol=1.0e-9):
            raise ValueError("BRAIDS spin-up bed does not match morphology bed")
        spun_frames = read_frames(spinup / "braids" / "frames.txt")
        if not np.allclose(spun_frames[0]["q"][:, 3:], spun_frames[-1]["q"][:, 3:]):
            raise ValueError("BRAIDS spin-up changed the bed or sediment inventory")
        last_b = (spun_frames[-1]["ledger"][:, 0] -
                  spun_frames[-2]["ledger"][:, 0]) / (
                  spun_frames[-1]["time_s"] - spun_frames[-2]["time_s"])
        if abs(np.sum(last_b)) > 0.01 * h["discharge_m3_s"]:
            raise ValueError("BRAIDS hydraulic spin-up has not reached flow balance")
        initial[:, :3] = spun_frames[-1]["q"][:, :3]
    materials = tuple(CutSedimentFraction(
        grain_density_kg_m3=s["grain_density_kg_m3"], water_density_kg_m3=1000.0,
        porosity=s["porosity"], d50_m=s["d50_m"], d90_m=s["d90_m"],
        z0_m=s["d50_m"] / 30.0, minimum_bedload_depth_m=0.005,
        roughness_mode="manning", fraction_id=f"gravel_{suffix}", sediment_type="sand",
        initial_volume_fraction=0.5, initial_active_volume_fraction=0.5,
    ) for suffix in ("a", "b"))
    boundaries = {
        "inlet": CutBoundary("inlet_critical", discharge_m3_s=h["discharge_m3_s"],
                             fraction_bed_input_kg_s={} if hydraulic_only else
                             {"gravel_a": s["bedload_supply_kg_s"] / 2.0,
                              "gravel_b": s["bedload_supply_kg_s"] / 2.0},
                             bedload_boundary_mode="prescribed"),
        "outlet": CutBoundary("critical"),
    }
    if 'prescribed_inlet_core_width_m' in h:
        # Validation-only controlled lateral distribution. Integrate the same
        # prescribed q(y) and sediment feed over each model's boundary faces.
        core=float(h['prescribed_inlet_core_width_m'])
        if not 0<core<=g['narrow_bottom_width_m']:raise ValueError('Invalid inlet core width')
        boundaries={'outlet':CutBoundary('stage',stage_m=h['fixed_outlet_stage_m'])};newfaces=[]
        for face in mesh.faces:
            if face.bc_id=='inlet':
                lo,hi=sorted((face.start_xy[1],face.end_xy[1]))
                share=max(0.,min(hi,core/2)-max(lo,-core/2))/core
                if share>0:
                    name=f'inlet_{face.face_id}'
                    boundaries[name]=CutBoundary('total_discharge',discharge_m3_s=h['discharge_m3_s']*share,
                        fraction_bed_input_kg_s={} if hydraulic_only else
                        {'gravel_a':s['bedload_supply_kg_s']*share/2,'gravel_b':s['bedload_supply_kg_s']*share/2})
                    face=replace(face,bc_id=name,bed_elevation_m=None,section_bed_start_m=None,section_bed_end_m=None)
                else:face=replace(face,bc_id='',kind='wall',bed_elevation_m=None,section_bed_start_m=None,section_bed_end_m=None)
            newfaces.append(face)
        mesh=replace(mesh,faces=tuple(newfaces))
    half_outer = float(outer_width(0.0, g)) / 2.0
    section_y = np.linspace(-half_outer, half_outer, 29)
    section = Section(
        "inlet", tuple((section_y + half_outer).tolist()),
        tuple(bed(0.0, section_y, g).tolist()), 0.0, (1.0, 0.0),
    )
    metadata = write_cut_case(
        folder / "case.data", mesh, initial, elevation - s["mobile_layer_m"], boundaries,
        end_time_s=end_s, dt_max_s=t["braids_dt_max_s"], output_interval_s=output_s,
        cfl=t.get("braids_cfl", 0.45), scheme=t.get("braids_scheme", 6),
        bedload=0 if hydraulic_only else 2,
        exchange=0, manning=h["manning_n_s_m13"],
        sediment_fractions=materials, active_layer_thickness_m=s["active_layer_m"],
        substrate_layer_count=1, substrate_layer_thickness_m=s["mobile_layer_m"] - s["active_layer_m"],
        mpm_cf=0.01, mpm_critical_shields=s["critical_shields"],
        mpm_friction_mode=0 if hydraulic_only else 1,
        bed_slope=0 if hydraulic_only else s.get("braids_bed_slope_mode", 2),
        morphological_factor=1.0 if hydraulic_only else s["morphological_factor"],
        source_sha256=build["source_sha256"], compact_sidecars=True,
        sections=None if 'prescribed_inlet_core_width_m' in h else {"inlet": section},
    )
    np.savez_compressed(folder / "geometry.npz", xy=coords, bed=elevation,
                        area=np.asarray([cell.area_m2 for cell in mesh.cells]))
    write(folder / "effective_specification.json", json.dumps(spec, indent=2))
    if hydraulic_seed is not None:
        write(folder / "hydraulic_seed.json", json.dumps({
            "source": str(hydraulic_seed), "frames_sha256": sha256(native_frames_path(hydraulic_seed)),
            "method": "interpolated wet-cell water surface and velocity, then separate fixed-bed relaxation"}, indent=2))
    write(folder / "preparation.json", json.dumps({"cells": len(mesh.cells), "faces": len(mesh.faces),
        "end_time_s": end_s, "output_interval_s": output_s,
        "deck": metadata, "specification_sha256": sha256(HERE / "specification.json"),
        "effective_specification_sha256": sha256(folder / "effective_specification.json"),
        "hydraulic_spinup": None if spinup is None else {
            "frames_sha256": sha256(native_frames_path(spinup / "braids")),
            "final_time_s": spun_frames[-1]["time_s"]}}, indent=2))
    return len(mesh.cells), folder


def delft_mdf(spec: dict, end_s: float, output_s: float) -> str:
    g, h, t = (spec[key] for key in ("geometry", "hydraulics", "time"))
    adapted = {
        "geometry": {"active_cells": [g["delft3d_nx"], g["delft3d_ny"]], "width_m": g["narrow_bottom_width_m"], "bed_elevation_m": 0.0},
        "hydraulics": {"initial_stage_m": 1.0, "inlet_total_discharge_m3_s": h["discharge_m3_s"], "gravity_m_s2": 9.81, "water_density_kg_m3": 1000.0, "manning_n_s_m13": h["manning_n_s_m13"], "horizontal_momentum_viscosity_m2_s": 0.0},
        "sediment": {"physical_horizontal_dispersion_m2_s": 0.0},
        "time": {"end_s": end_s, "output_interval_s": output_s},
    }
    value = common._delft3d_mdf("settling", adapted)
    value = re.sub(r"(?m)^Depuni=.*$", "Fildep= #basin.dep#\nFmtdep= #FR#", value)
    value = re.sub(r"(?m)^Dt\s*=.*$", f"Dt    = {t['delft3d_dt_s'] / 60.0:.16e}", value)
    value = re.sub(r"(?m)^Dryflc=.*$", "Dryflc= 5.000000000000e-03", value)
    value = value.replace("Dpsopt= #MAX#", "Dpsopt= #DP#")
    return value.rstrip() + "\nFilbcQ = #basin.bcq#\nFmtbcQ = #FR#\nFilic = #initial.ini#\nFmtic = #FR#\nPrecOut= 8\n"


def qh_table(q_values: np.ndarray, stages: np.ndarray) -> str:
    rows = "\n".join(f"{q:.16e} {z:.16e}" for q, z in zip(q_values, stages))
    return "\n".join((
        "table-name           'Critical outlet rating curve'", "contents             'Uniform             '",
        "location             'outlet              '", "xy-function          'non-equidistant'",
        "interpolation        'linear'", "parameter            'total discharge (t)  end A'               unit '[m3/s]'",
        "parameter            'water elevation (z)  end B'               unit '[m]'",
        f"records-in-table     {len(q_values)}", rows,
    ))


def sediment_concentration_tables(end_s: float) -> str:
    tables = []
    for number, location in enumerate(("outlet", "inlet"), start=1):
        tables.append("\n".join((
            f"table-name           'Boundary Section : {number}'",
            "contents             'Uniform             '",
            f"location             '{location:<20}'",
            "time-function        'non-equidistant'", "reference-time       20010601",
            "time-unit            'minutes'", "interpolation        'linear'",
            "parameter            'time                '                     unit '[min]'",
            "parameter            'sediment 1          end A uniform'       unit '[kg/m**3]'",
            "parameter            'sediment 1          end B uniform'       unit '[kg/m**3]'",
            "records-in-table     2", "0.0 0.0 0.0", f"{end_s / 60.0:.16e} 0.0 0.0",
        )))
    return "\n".join(tables)


def bedload_supply_table(spec: dict, end_s: float,
                         schedule: list[tuple[float, float]] | None = None) -> str:
    g, h, s = (spec[key] for key in ("geometry", "hydraulics", "sediment"))
    bulk = s["grain_density_kg_m3"] * (1.0 - s["porosity"])
    # IBedCond=4 reads volume including pores, so the inlet mass conversion
    # uses CDryB. This initial per-width rate is only a starting estimate;
    # changing wet width requires time-dependent calibration and SBTR auditing.
    flux = s["bedload_supply_kg_s"] / (bulk * s["delft3d_effective_inlet_feed_width_m"])
    records = ([(0.0, flux), (end_s, flux)] if schedule is None else schedule)
    if not records or records[0][0] != 0.0 or records[-1][0] != end_s or any(
            not np.isfinite(time) or not np.isfinite(value) or value <= 0
            for time, value in records) or any(
            records[i][0] >= records[i + 1][0] for i in range(len(records) - 1)):
        raise ValueError("Delft3D bedload schedule must be positive and cover the run")
    return "\n".join((
        "table-name           'Prescribed single-grain bedload supply'",
        "contents             'Uniform             '",
        "location             'inlet               '",
        "time-function        'non-equidistant'", "reference-time       20010601",
        "time-unit            'minutes'", "interpolation        'linear'",
        "parameter            'time                '                     unit '[min]'",
        "parameter            'transport incl pores Sediment 1'          unit '[m2/s]'",
        f"records-in-table     {len(records)}",
        *(f"{time / 60.0:.16e} {value:.16e}" for time, value in records),
    ))


def delft_solid_volume_to_mass_rate(output_m3_s, spec: dict):
    """Convert explicitly configured TranspType=2 output to solid kg/s."""
    return np.asarray(output_m3_s) * spec["sediment"]["grain_density_kg_m3"]


def prepare_delft(spec: dict, folder: Path, end_s: float, output_s: float,
                  hydraulic_only: bool = False,
                  spinup: Path | None = None) -> tuple[int, Path]:
    if (folder / "trim-basin.nc").exists() or (
        (folder / "execution.json").exists()
        and json.loads((folder / "execution.json").read_text(encoding="utf-8")).get("success")
    ):
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g, h, s = (spec[key] for key in ("geometry", "hydraulics", "sediment"))
    check_matched_resolution(g)
    nx, ny = g["delft3d_nx"], g["delft3d_ny"]
    xline = np.linspace(0.0, g["length_m"], nx + 1)
    eta = np.linspace(-0.5, 0.5, ny + 1)[:, None]
    xgrid = np.tile(xline, (ny + 1, 1))
    ygrid = eta * outer_width(xline, g)[None, :]
    write(folder / "basin.grd", "\n".join(("Coordinate System = Cartesian", f"{nx + 1:8d}{ny + 1:8d}", "       0       0       0", *common._grid_rows(xgrid), *common._grid_rows(ygrid))))
    write(folder / "basin.enc", "\n".join(f"{a:6d}{b:6d}" for a, b in ((1, 1), (nx + 2, 1), (nx + 2, ny + 2), (1, ny + 2), (1, 1))))
    xc = (xline[:-1] + xline[1:]) / 2.0
    eta_c = (np.arange(ny) + 0.5) / ny - 0.5
    xx = np.tile(xc[:, None], (1, ny))
    yy = outer_width(xc, g)[:, None] * eta_c[None, :]
    elevation = bed(xx, yy, g)
    padded_bed = np.pad(elevation, 1, mode="edge")
    np.savetxt(folder / "basin.dep", -padded_bed.T, fmt="%.17e")
    west, east = f"(1,2)..(1,{ny + 1})", f"({nx + 2},2)..({nx + 2},{ny + 1})"
    write(folder / "basin.bnd", "\n".join((
        f"{'outlet':<20} Z Q {nx + 2:5d} {2:5d} {nx + 2:5d} {ny + 1:5d} 0.000000 Uniform",
        f"{'inlet':<20} T T {1:5d} {2:5d} {1:5d} {ny + 1:5d} 0.000000 Uniform",
    )))
    def time_table(location: str, discharge: float) -> str:
        return "\n".join((
            "table-name           'Boundary Section : 1'", "contents             'Uniform             '",
            f"location             '{location:<20}'", "time-function        'non-equidistant'",
            "reference-time       20010601", "time-unit            'minutes'", "interpolation        'linear'",
            "parameter            'time                '                     unit '[min]'",
            "parameter            'total discharge (t)  end A'               unit '[m3/s]'",
            "parameter            'total discharge (t)  end B'               unit '[m3/s]'",
            "records-in-table     2", f"0.0 {discharge:.16e} {discharge:.16e}",
            f"{end_s / 60.0:.16e} {discharge:.16e} {discharge:.16e}",
        ))
    write(folder / "basin.bct", time_table("inlet", h["discharge_m3_s"]))
    q = np.linspace(0.0, 2.0 * h["discharge_m3_s"], 101)
    z = np.asarray([critical_depth(float(flow), g["narrow_bottom_width_m"], g["bank_side_slope_horizontal_per_vertical"]) for flow in q])
    write(folder / "basin.bcq", qh_table(q, z))
    write(folder / "basin.bcc", sediment_concentration_tables(end_s))
    write(folder / "basin.bcm", bedload_supply_table(spec, end_s))
    write(folder / "basin.sed", f"""[SedimentFileInformation]
   FileVersion = 02.00
[SedimentOverall]
   Cref = 1.000000000000e+06
   IopSus = 0
[Sediment]
   Name = #Sediment 1#
   SedTyp = gravel
   RhoSol = {s['grain_density_kg_m3']:.12e}
   SedDia = {s['d50_m']:.12e}
   CDryB = {s['grain_density_kg_m3'] * (1.0 - s['porosity']):.12e}
   IniSedThick = {s['mobile_layer_m']:.12e}
   FacDSS = 1.0
   TraFrm = #formula.tra#""")
    write(folder / "formula.tra", "[TransportFormulaFileInformation]\n   FileVersion = 01.00\n[TransportFormula]\n   Number = 2\n   ACal = 1.000000000000e+00")
    write(folder / "basin.mor", f"""[MorphologyFileInformation]
   FileVersion = 02.00
[Morphology]
   MorFac = {1.0 if hydraulic_only else s['morphological_factor']:.12e}
   MorStt = 0.0
   Thresh = 0.0
   MorUpd = {'false' if hydraulic_only else 'true'}
   EqmBc = false
   DensIn = false
   ISlope = 2
   IHidExp = 1
   AksFac = 1.0
   RWave = 2.0
   AlfaBs = 1.0
   AlfaBn = 1.5
   Sus = 0.0
   Bed = 1.0
   SusW = 0.0
   BedW = 0.0
   SedThr = 5.000000000000e-03
   Hmaxth = 1.0e-3
   EpsPar = false
   BcFil = #basin.bcm#
[Boundary]
   Name = #inlet#
   IBedCond = 4
[Boundary]
   Name = #outlet#
   IBedCond = 0
[Output]
   Default = true
   TranspType = 2
   BedTranspDueToCurrentsAtZeta = true
   BedTranspDueToCurrentsAtFlux = true
   RawTransportsAtZeta = true
   Taub = true""")
    write(folder / "basin.crs", f"inlet                   1      2      1      {ny + 1}\noutlet                 {nx + 1}      2     {nx + 1}      {ny + 1}")
    stage = initial_stage(xc, g, h)
    velocity = initial_velocity(xc, g, h)
    initial_fields = (np.pad(np.tile(stage[:, None], (1, ny)), 1, mode="edge"), np.pad(np.tile(velocity[:, None], (1, ny)), 1, mode="edge"), np.zeros_like(padded_bed), np.zeros_like(padded_bed))
    if spinup is not None:
        spun_native = common.inspect_delft3d(spinup / "delft3d")
        spun = spun_native["map"]
        if spun["S1"].shape[1:] != padded_bed.shape or not np.allclose(
                spun["DPS"][0], spun["DPS"][-1], atol=1.0e-10) or not np.allclose(
                spun["DPS"][0], -padded_bed, atol=1.0e-10):
            raise ValueError("Delft3D spin-up grid or fixed bed does not match")
        spun_history = spun_native["history"]
        last_d = np.diff(spun_history["FLTR"][-2:], axis=0)[0] / np.diff(
            spun_history["time"][-2:])[0]
        if abs(last_d[0] - last_d[-1]) > 0.01 * h["discharge_m3_s"]:
            raise ValueError("Delft3D hydraulic spin-up has not reached flow balance")
        initial_fields = (spun["S1"][-1], spun["U1"][-1, 0],
                          spun["V1"][-1, 0], np.zeros_like(padded_bed))
    with (folder / "initial.ini").open("w", encoding="utf-8") as stream:
        for field in initial_fields:
            np.savetxt(stream, field.T, fmt="%.17e")
    write(folder / "basin.mdf", delft_mdf(spec, end_s, output_s))
    write(folder / "config_d_hydro.xml", """<?xml version="1.0" encoding="iso-8859-1"?>
<deltaresHydro xmlns="http://schemas.deltares.nl/deltaresHydro" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schemas.deltares.nl/deltaresHydro http://content.oss.deltares.nl/schemas/d_hydro-1.00.xsd">
  <documentation><fileVersion>1.00</fileVersion><createdBy>BRAIDS</createdBy></documentation>
  <control><sequence><start>syntheticExpansion</start></sequence></control>
  <flow2D3D name="syntheticExpansion"><library>flow2d3d</library><mdfFile>basin.mdf</mdfFile></flow2D3D>
</deltaresHydro>""")
    write(folder / "input_provenance.json", json.dumps({
        "specification_sha256": sha256(HERE / "specification.json"),
        "hydraulic_spinup": None if spinup is None else {
            "map_sha256": sha256(spinup / "delft3d" / "trim-basin.nc")},
        "files": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}, indent=2))
    return nx * ny, folder


def run_braids(spec: dict, folder: Path, timeout_s: float) -> dict:
    executable, _ = verified_braids_build()
    env = dict(os.environ, OMP_NUM_THREADS=str(spec["time"]["braids_threads"]), OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
    env['BRAIDS_PROGRESS_STEPS']='500'
    if spec['time'].get('profile_native',False):
        env['BRAIDS_PROFILE']='1'
    start = time.perf_counter()
    with (folder / "native.log").open("w", encoding="utf-8") as log:
        result = subprocess.run((str(executable), "case.data"), cwd=folder, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    if result.returncode or not any((folder / name).is_file() for name in ('frames.bin','frames.txt')):
        raise RuntimeError(f"BRAIDS failed to produce native frames; inspect {folder / 'native.log'}")
    frames = read_frames(folder / 'frames.txt')
    expected = json.loads((folder / 'preparation.json').read_text()).get('end_time_s')
    if not frames or (expected is not None and not np.isclose(frames[-1]['time_s'],expected)):
        raise RuntimeError(f'Incomplete native output in {folder}')
    summary = {"return_code": result.returncode, "wall_s": time.perf_counter() - start,
               "threads": spec["time"]["braids_threads"], "executable_sha256": sha256(executable)}
    write(folder / "execution.json", json.dumps(summary, indent=2))
    return summary


def spatial_metrics(x: np.ndarray, area: np.ndarray, delta: np.ndarray, spec: dict) -> dict:
    g = spec["geometry"]
    bounds = (
        ("narrow_inlet", 0.0, g["inlet_reach_end_m"]),
        ("expansion", g["inlet_reach_end_m"], g["expansion_end_m"]),
        ("wide_1km", g["expansion_end_m"], g["wide_reach_end_m"]),
        ("contraction", g["wide_reach_end_m"], g["contraction_end_m"]),
        ("narrow_outlet", g["contraction_end_m"], g["length_m"] + 1.0),
    )
    result = {}
    for name, low, high in bounds:
        mask = np.isfinite(delta) & np.isfinite(area) & (area > 0) & (x >= low) & (x < high)
        if not np.any(mask):
            continue
        result[name] = {
            "mean_dz_m": float(np.average(delta[mask], weights=area[mask])),
            "deposition_m3": float(np.sum(np.maximum(delta[mask], 0) * area[mask])),
            "erosion_m3": float(np.sum(np.maximum(-delta[mask], 0) * area[mask])),
            "max_aggradation_m": float(np.max(delta[mask])),
            "max_degradation_m": float(np.min(delta[mask])),
        }
    return result


def write_longitudinal_profiles(reports: Path, spec: dict, fields: dict) -> None:
    if len(fields) != 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    length = spec["geometry"]["length_m"]
    edges = np.arange(0.0, length + 20.0, 20.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profiles = {}
    for name, (x, area, delta) in fields.items():
        denominator, _ = np.histogram(x, bins=edges, weights=area)
        numerator, _ = np.histogram(x, bins=edges, weights=area * delta)
        profiles[name] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    matrix = np.column_stack((centers, profiles["braids"], profiles["delft3d"]))
    np.savetxt(reports / "longitudinal_bed_change.csv", matrix, delimiter=",", header="x_m,braids_mean_dz_m,delft3d_mean_dz_m", comments="")
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.axhline(0.0, color="0.5", lw=0.8)
    for label, color in (("braids", "#1e6ab0"), ("delft3d", "#bd4f35")):
        axis.plot(centers, 1000.0 * profiles[label], label=label.upper(), color=color)
    g = spec["geometry"]
    for x in (g["inlet_reach_end_m"], g["expansion_end_m"], g["wide_reach_end_m"], g["contraction_end_m"]):
        axis.axvline(x, color="0.75", lw=0.8, ls="--")
    axis.set(xlabel="Distance downstream (m)", ylabel="Mean bed change (mm)", title="Native Exner bed response: nominal 5 kg/s gravel feed, MorFac 10")
    axis.legend()
    figure.tight_layout()
    figure.savefig(reports / "longitudinal_bed_change.png", dpi=160)
    plt.close(figure)


def write_geometry_figure(reports: Path, spec: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = spec["geometry"]
    x = np.linspace(0.0, g["length_m"], 601)
    bottom_half = bottom_width(x, g) / 2.0
    outer_half = outer_width(x, g) / 2.0
    figure, axis = plt.subplots(figsize=(11, 3.4), layout="constrained")
    axis.fill_between(x, -outer_half, outer_half, color="#d9c6a7", label="2H:1V banks")
    axis.fill_between(x, -bottom_half, bottom_half, color="#78adca", label="Channel bottom")
    axis.axvline(g["expansion_end_m"], color="0.35", lw=0.7, ls="--")
    axis.axvline(g["wide_reach_end_m"], color="0.35", lw=0.7, ls="--")
    axis.set(xlim=(0, g["length_m"]), ylim=(-30, 30), xlabel="Distance downstream (m)",
             ylabel="Cross-stream coordinate (m)", title="Synthetic trapezoidal expansion / 1 km wide reach / contraction")
    axis.set_aspect("auto")
    axis.legend(loc="upper right")
    figure.savefig(reports / "geometry.png", dpi=160)
    plt.close(figure)


def write_expansion_maps(reports: Path, maps: dict) -> None:
    if len(maps) != 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    values = np.concatenate([np.abs(row[2]) for row in maps.values()])
    limit = max(float(np.nanpercentile(values, 99)), 0.005)
    figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, sharey=True,
                                layout="constrained")
    for axis, (label, (x, y, dz)) in zip(axes, maps.items()):
        selected = (x >= 75.0) & (x <= 350.0) & np.isfinite(dz)
        artist = axis.scatter(x[selected], y[selected], c=1000.0 * dz[selected],
                              s=14 if label == "braids" else 30, marker="s",
                              cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0,
                                  vmin=-1000.0 * limit, vmax=1000.0 * limit), linewidths=0)
        axis.set(ylabel=f"{label.upper()}\ny (m)", ylim=(-28, 28), xlim=(75, 350))
        for line in (100.0, 250.0):
            axis.axvline(line, color="black", lw=0.6, ls="--")
    axes[-1].set_xlabel("Distance downstream (m)")
    figure.colorbar(artist, ax=axes, label="Bed change (mm)", shrink=0.8,
                    pad=0.02, fraction=0.03)
    figure.savefig(reports / "expansion_bed_change_maps.png", dpi=170)
    plt.close(figure)


def write_markdown_report(reports: Path, spec: dict, result: dict) -> None:
    if result["status"] != "paired_native_outputs":
        return
    b, d = result["braids"], result["delft3d"]
    spun = result.get("hydraulic_spinup", False)
    feed_audit = Path(result.get("delft3d_run", "")) / "feed_audit.json"
    calibrated_feed = ("feed_calibration_v2" in str(result.get("delft3d_run", ""))
                       or (feed_audit.is_file() and json.loads(feed_audit.read_text()).get("passed", False)))
    feed_note = (
        "The Delft3D inlet flux was independently calibrated to 5 kg/s at "
        "every saved 30 s sample within the recorded feed tolerance, so the remaining bed-change gap is not "
        "explained by inlet sediment mass."
        if calibrated_feed else
        "Delft3D's per-width inlet flux varies in total mass as the wet "
        "width changes; this run is not necessarily source-equivalent."
    )
    lines = [
        "# Native synthetic expansion comparison",
        "",
        "This is a paired diagnostic, not a validated braided-river simulation.",
        ("Both sediment runs began from their respective balanced fixed-bed"
         " hydraulic spin-ups." if spun else
         "Both sediment runs began from the analytic initial surface, before"
         " the 1.5 km reach had equilibrated hydraulically."),
        "The same 1% trapezoidal 10→50→10 m bottom-width profile, 5 m³/s",
        "water discharge, 15 mm noncohesive grain and MorFac 10 are used.",
        "The full-width reach is 1 km long, with 15 mm deterministic initial-bed perturbations.",
        "BRAIDS stores that one physical grain as two identical 50/50 numerical",
        "fractions to use its roughness-corrected MPM path; Delft3D uses one fraction.",
        f"The uniform-flow planning estimate gives {planning_mpm_capacity_kg_s(10.0, spec):.2f} kg/s",
        f"MPM capacity at 10 m bottom width and {planning_mpm_capacity_kg_s(50.0, spec):.2f} kg/s",
        "at 50 m; this is a forcing screen, not a native-model prediction.",
        "",
        "| Native result | BRAIDS | Delft3D4 |",
        "|---|---:|---:|",
        f"| Accepted inlet water (m³/s; BRAIDS cumulative mean, Delft3D cumulative transect mean) | {b['accepted_inlet_water_m3_s']:.3f} | {d['accepted_inlet_water_m3_s']:.3f} |",
        f"| Final-interval outlet water (m³/s) | {b['last_interval_outlet_water_m3_s']:.3f} | {d['last_interval_outlet_water_m3_s']:.3f} |",
        f"| Accepted inlet bedload (kg/s raw; BRAIDS cumulative mean, Delft3D sampled time-mean estimate) | {b['accepted_inlet_bedload_kg_s_raw']:.3f} | {d['accepted_inlet_bedload_kg_s_time_mean_estimate']:.3f} |",
        f"| Final instantaneous Delft3D inlet bedload (kg/s raw) | n/a | {d['accepted_inlet_bedload_kg_s_raw']:.3f} |",
        f"| First sampled Delft3D inlet bedload (kg/s raw) | n/a | {d['first_sampled_inlet_bedload_kg_s_raw']:.3f} |",
        f"| BRAIDS Exner mass residual (kg) | {b['exner_mass_residual_kg']:.3e} | n/a |",
        f"| Expansion mean bed change (mm) | {1000*b['zones']['expansion']['mean_dz_m']:.3f} | {1000*d['zones']['expansion']['mean_dz_m']:.3f} |",
        f"| Expansion maximum aggradation (m) | {b['zones']['expansion']['max_aggradation_m']:.3f} | {d['zones']['expansion']['max_aggradation_m']:.3f} |",
        f"| Expansion maximum degradation (m) | {b['zones']['expansion']['max_degradation_m']:.3f} | {d['zones']['expansion']['max_degradation_m']:.3f} |",
        f"| Wide 1 km deposition (m³) | {b['zones']['wide_1km']['deposition_m3']:.3f} | {d['zones']['wide_1km']['deposition_m3']:.3f} |",
        f"| Inlet-row max bed rise (m) | {b['inlet_row_max_aggradation_m']:.3f} | {d['inlet_row_max_aggradation_m']:.3f} |",
        f"| Inlet-row max bed fall (m) | {b['inlet_row_max_degradation_m']:.3f} | {d['inlet_row_max_degradation_m']:.3f} |",
        "",
        f"The sampled inlet bedload differs by {100.0 * result['inlet_feed_relative_difference']:.1f}%.",
        "Delft3D prescribes transport per unit wet width; BRAIDS holds the",
        "specified total 5 kg/s. " + feed_note,
        "The historical flow-directed Exner flux produced cell-alternating",
        "bed changes that were not physical bars. See the separate stability",
        "audit for this build; agreement of extrema alone is not braids_support.",
        "",
        "[Longitudinal bed-change plot](longitudinal_bed_change.png) ·",
        "[Full geometry](geometry.png) ·",
        "[Expansion maps](expansion_bed_change_maps.png) ·",
        "[20 m profile CSV](longitudinal_bed_change.csv) ·",
        "[Machine-readable metrics](comparison.json)",
        "",
        "The inlet and outlet controls are not fully equivalent: BRAIDS recomputes",
        "critical depth with the evolving bed, whereas Delft3D4 uses a discharge",
        "inlet initialized near critical depth and a fixed-initial-section critical",
        "Q–H outlet rating. Delft3D's per-width gravel feed is wet-width dependent,",
        "so the accepted feed is audited rather than assumed constant.",
        "",
        "MorFac 10 is exploratory here: the BRAIDS qualification envelope does not",
        "cover all steep, wet/dry, bar-forming transients. A smooth expansion and",
        "single grain size do not mathematically guarantee bars or braiding.",
        "A longer experiment should wait until boundary flux, transport capacity",
        "and Exner bed-change differences have been diagnosed in this small case.",
    ]
    write(reports / "COMPARISON.md", "\n".join(lines))


def analyze(spec: dict, braids: Path, delft: Path, reports: Path) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    result = {"status": "partial", "boundary_equivalence": "BRAIDS live critical inlet/outlet; Delft3D discharge inlet and fixed critical Q-H outlet", "braids": {}, "delft3d": {}, "delft3d_run": str(delft)}
    fields = {}
    maps = {}
    if (braids / "frames.txt").is_file():
        frames = read_frames(braids / "frames.txt")
        if frames:
            xy = np.load(braids / "geometry.npz")
            b0, bf = frames[0]["q"], frames[-1]["q"]
            bulk = spec["sediment"]["grain_density_kg_m3"] * (1.0 - spec["sediment"]["porosity"])
            nfrac = int(frames[-1]["nfrac"])
            bed_start = 3 + nfrac
            delta = (bf[:, bed_start:].sum(axis=1) - b0[:, bed_start:].sum(axis=1)) / bulk
            fields["braids"] = (xy["xy"][:, 0], xy["area"], delta)
            maps["braids"] = (xy["xy"][:, 0], xy["xy"][:, 1], delta)
            wide = (xy["xy"][:, 0] >= spec["geometry"]["expansion_end_m"]) & (xy["xy"][:, 0] <= spec["geometry"]["wide_reach_end_m"])
            result["braids"] = {"last_time_s": frames[-1]["time_s"], "mean_wide_bed_change_m": float(np.average(delta[wide], weights=xy["area"][wide])), "wide_deposition_m3": float(np.sum(np.maximum(delta[wide], 0) * xy["area"][wide])), "wide_erosion_m3": float(np.sum(np.maximum(-delta[wide], 0) * xy["area"][wide]))}
            result["braids"]["zones"] = spatial_metrics(*fields["braids"], spec)
            inlet_row = xy["xy"][:, 0] < spec["geometry"]["braids_cell_size_m"]
            result["braids"]["inlet_row_max_aggradation_m"] = float(np.max(delta[inlet_row]))
            result["braids"]["inlet_row_max_degradation_m"] = float(np.min(delta[inlet_row]))
            result["braids"]["accepted_inlet_water_m3_s"] = float(frames[-1]["ledger"][0, 0] / frames[-1]["time_s"])
            result["braids"]["last_interval_outlet_water_m3_s"] = float(
                -(frames[-1]["ledger"][1, 0] - frames[-2]["ledger"][1, 0]) /
                (frames[-1]["time_s"] - frames[-2]["time_s"]))
            result["braids"]["accepted_inlet_bedload_kg_s_raw"] = float(np.sum(frames[-1]["ledger"][0, 1 + nfrac:]) / (frames[-1]["time_s"] * spec["sediment"]["morphological_factor"]))
            bed_mass_change = float(np.sum((bf[:, bed_start:].sum(axis=1) - b0[:, bed_start:].sum(axis=1)) * xy["area"]))
            accepted_bed_mass = float(np.sum(frames[-1]["ledger"][:, 1 + nfrac:]))
            result["braids"]["exner_mass_residual_kg"] = bed_mass_change - accepted_bed_mass
    if (delft / "trim-basin.nc").is_file():
        data = common.inspect_delft3d(delft)["map"]
        active = data["KCS"] == 1
        x = data["XZ"]
        delta = data["DPS"][0] - data["DPS"][-1]
        wide = active & (x >= spec["geometry"]["expansion_end_m"]) & (x <= spec["geometry"]["wide_reach_end_m"])
        area = data.get("GSQS", np.ones_like(x))
        fields["delft3d"] = (x[active], area[active], delta[active])
        maps["delft3d"] = (x[active], data["YZ"][active], delta[active])
        result["delft3d"] = {"last_time_index": int(data["DPS"].shape[0] - 1), "mean_wide_bed_change_m": float(np.average(delta[wide], weights=area[wide])), "wide_deposition_m3": float(np.sum(np.maximum(delta[wide], 0) * area[wide])), "wide_erosion_m3": float(np.sum(np.maximum(-delta[wide], 0) * area[wide]))}
        result["delft3d"]["zones"] = spatial_metrics(*fields["delft3d"], spec)
        inlet_row = active & (x < spec["geometry"]["braids_cell_size_m"])
        result["delft3d"]["inlet_row_max_aggradation_m"] = float(np.max(delta[inlet_row]))
        result["delft3d"]["inlet_row_max_degradation_m"] = float(np.min(delta[inlet_row]))
        history = common.inspect_delft3d(delft)["history"]
        result["delft3d"]["accepted_inlet_water_m3_s"] = float(history["FLTR"][-1, 0] / history["time"][-1])
        result["delft3d"]["last_interval_outlet_water_m3_s"] = float(
            (history["FLTR"][-1, 1] - history["FLTR"][-2, 1]) /
            (history["time"][-1] - history["time"][-2]))
        # Delft3D's default TranspType=2 writes solid volume (excluding
        # pores), although IBedCond=4 reads prescribed volume *including*
        # pores. See rdmor.f90 and bott3d.f90 in the pinned Delft3D source.
        result["delft3d"]["accepted_inlet_bedload_kg_s_raw"] = float(
            delft_solid_volume_to_mass_rate(history["SBTR"][-1, 0, 0], spec))
        sampled_rates = delft_solid_volume_to_mass_rate(history["SBTR"][:, 0, 0], spec)
        result["delft3d"]["first_sampled_inlet_bedload_kg_s_raw"] = float(sampled_rates[1])
        result["delft3d"]["accepted_inlet_bedload_kg_s_time_mean_estimate"] = float(np.sum(sampled_rates[1:] * np.diff(history["time"])) / history["time"][-1])
        if "SBTRC" in history:
            # Native per-step cumulative transport; no output-sampling quadrature.
            result["delft3d"]["accepted_inlet_bedload_kg_s_cumulative_mean"] = float(
                delft_solid_volume_to_mass_rate(history["SBTRC"][-1,0,0],spec) / history["time"][-1])
        result["delft3d"]["native_morfac"] = float(data["MORFAC"][-1])
    if result["braids"] and result["delft3d"]:
        result["status"] = "paired_native_outputs"
        preparation = braids / "preparation.json"
        result["hydraulic_spinup"] = bool(json.loads(preparation.read_text(
            encoding="utf-8")).get("hydraulic_spinup")) if preparation.is_file() else False
        q1 = result["braids"]["accepted_inlet_bedload_kg_s_raw"]
        q2 = result["delft3d"]["accepted_inlet_bedload_kg_s_time_mean_estimate"]
        result["inlet_feed_relative_difference"] = abs(q1 - q2) / max(abs(q1), abs(q2), 1e-12)
        write_longitudinal_profiles(reports, spec, fields)
        write_geometry_figure(reports, spec)
        write_expansion_maps(reports, maps)
    write(reports / "comparison.json", json.dumps(result, indent=2))
    write_markdown_report(reports, spec, result)
    return result


def audit_hydraulics(spec: dict, braids: Path, delft: Path, reports: Path) -> dict:
    """Compare fixed-bed depth and discharge after both native solvers run."""
    reports.mkdir(parents=True, exist_ok=True)
    frames = read_frames(braids / "frames.txt")
    geometry = np.load(braids / "geometry.npz")
    native = common.inspect_delft3d(delft)
    mapping, history = native["map"], native["history"]
    g, h = spec["geometry"], spec["hydraulics"]
    bx, by = geometry["xy"].T
    dx, dy = mapping["XZ"], mapping["YZ"]
    active = mapping["KCS"] == 1
    stations = (2.0, 50.0, 175.0, 300.0, 750.0, 1300.0, 1450.0, 1498.0)
    rows = []
    for x in stations:
        central_x = bx[np.abs(by) <= 2.01]
        sample_x = float(central_x[np.argmin(np.abs(central_x - x))])
        width = float(bottom_width(sample_x, g))
        bottom = float(bed(sample_x, 0.0, g))
        # Use the central bottom strip, not bank cells of differing elevation.
        bmask = (np.abs(bx - sample_x) < 0.01) & (np.abs(by) <= min(2.01, width / 4.0))
        dmask = active & (np.abs(dx - sample_x) < 0.01) & (np.abs(dy) <= min(2.01, width / 4.0))
        if not np.any(bmask) or not np.any(dmask):
            continue
        for index, label in ((0, "initial"), (-1, "final")):
            bstage = frames[index]["q"][:, 0] + geometry["bed"]
            dstage = mapping["S1"][index]
            rows.append((sample_x, label, float(np.median(bstage[bmask]) - bottom),
                         float(np.median(dstage[dmask]) - bottom)))
    inlet_b = np.diff([frames[-2]["ledger"][0, 0], frames[-1]["ledger"][0, 0]])[0] / (
        frames[-1]["time_s"] - frames[-2]["time_s"])
    outlet_b = -np.diff([frames[-2]["ledger"][1, 0], frames[-1]["ledger"][1, 0]])[0] / (
        frames[-1]["time_s"] - frames[-2]["time_s"])
    inlet_d, outlet_d = np.diff(history["FLTR"][-2:], axis=0)[0] / np.diff(history["time"][-2:])[0]
    inlet_sections = np.loadtxt(braids / "boundary_sections.txt", skiprows=1)
    section_q, section_area_m2, section_top_m = (
        inlet_sections[-1, 3], inlet_sections[-1, 5], inlet_sections[-1, 6])
    section_froude = float(section_q * np.sqrt(
        section_top_m / (9.81 * section_area_m2 ** 3)))
    profile = np.asarray([(x, 0 if label == "initial" else 1, b, d)
                          for x, label, b, d in rows])
    np.savetxt(reports / "hydraulic_depth_profile.csv", profile, delimiter=",",
               header="x_m,time_index_0_initial_1_final,braids_center_depth_m,delft3d_center_depth_m",
               comments="")
    initial = profile[profile[:, 1] == 0]
    final = profile[profile[:, 1] == 1]
    result = {
        "mode": "fixed_bed_hydraulic_control",
        "grid": {"braids_nominal_dx_m": g["braids_cell_size_m"],
                 "delft_dx_m": check_matched_resolution(g)[0],
                 "delft_max_transverse_m": check_matched_resolution(g)[1],
                 "braids_cells": int(len(bx)), "delft_active_cells": int(np.sum(active))},
        "duration_s": float(frames[-1]["time_s"]),
        "last_interval_flow_m3_s": {
            "braids_inlet": float(inlet_b), "braids_outlet": float(outlet_b),
            "delft3d_inlet": float(inlet_d), "delft3d_outlet": float(outlet_d)},
        "braids_inlet_section_froude": section_froude,
        "initial_center_depth_rmse_m": float(np.sqrt(np.mean((initial[:, 2] - initial[:, 3]) ** 2))),
        "final_center_depth_rmse_m": float(np.sqrt(np.mean((final[:, 2] - final[:, 3]) ** 2))),
        "stations": [{"x_m": float(x), "state": label,
                      "braids_center_depth_m": b, "delft3d_center_depth_m": d}
                     for x, label, b, d in rows],
    }
    write(reports / "HYDRAULIC_AUDIT.json", json.dumps(result, indent=2))
    lines = [
        "# Matched-resolution fixed-bed hydraulic audit",
        "",
        "Bed elevation was held fixed in both solvers; initialization lineage",
        "is recorded with each deck (a warm-start seed need not match an analytic start).",
        f"Delft3D used {result['grid']['delft_dx_m']:g} m longitudinal and at most",
        f"{result['grid']['delft_max_transverse_m']:.3f} m transverse spacing; BRAIDS used",
        f"a {g['braids_cell_size_m']:g} m nominal Cartesian grid.",
        "",
        f"After their respective spin-ups (BRAIDS {result['duration_s']:.0f} s, Delft3D {history['time'][-1]:.0f} s), last-interval inlet/outlet discharge",
        f"was {inlet_b:.3f}/{outlet_b:.3f} m³/s in BRAIDS and",
        f"{inlet_d:.3f}/{outlet_d:.3f} m³/s in Delft3D.",
        f"BRAIDS' inlet section Froude was {section_froude:.6f}.",
        f"Starting centerline depth RMSE was {result['initial_center_depth_rmse_m']:.6f} m;",
        f"equilibrated depth RMSE was {result['final_center_depth_rmse_m']:.3f} m.",
        "",
        "| x (m) | Initial BRAIDS / Delft3D depth (m) | Final BRAIDS / Delft3D depth (m) |",
        "|---:|---:|---:|",
    ]
    by_x = {}
    for x, label, b, d in rows:
        by_x.setdefault(x, {})[label] = (b, d)
    for x, states in by_x.items():
        b0, d0 = states["initial"]
        bf, df = states["final"]
        lines.append(f"| {x:g} | {b0:.3f} / {d0:.3f} | {bf:.3f} / {df:.3f} |")
    lines.extend((
        "",
        "Initial states and equilibrated profiles are reported separately above.",
        "Agreement of final water flux alone does not establish equal depths or",
        "sediment physics. The Delft3D",
        "discharge-only inlet is not a live critical-depth inlet like BRAIDS.",
        "",
        "[Depth CSV](hydraulic_depth_profile.csv) ·",
        "[Machine-readable audit](HYDRAULIC_AUDIT.json)",
    ))
    write(reports / "HYDRAULIC_AUDIT.md", "\n".join(lines))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="run a 60 s hydraulic-time stability pilot")
    parser.add_argument("--screen", action="store_true", help="run a 900 s hydraulic-time first morphology screen")
    parser.add_argument("--hydraulic-only", action="store_true",
                        help="run an 1800 s fixed-bed hydraulic control")
    parser.add_argument("--spinup-from", type=Path,
                        help="initialize both morphology decks from a paired fixed-bed hydraulic run")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true", help="run existing immutable paired decks")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--run-name", help="immutable output suffix for a new paired run")
    args = parser.parse_args()
    spec = specification()
    if sum((args.pilot, args.screen, args.hydraulic_only)) > 1:
        parser.error("--pilot, --screen, and --hydraulic-only are mutually exclusive")
    if args.hydraulic_only and args.spinup_from:
        parser.error("--hydraulic-only cannot be initialized from a spin-up")
    if not (args.pilot or args.hydraulic_only or args.analyze_only or
            args.run_prepared or args.spinup_from):
        parser.error("Morphology decks require --spinup-from a balanced fixed-bed run")
    tag = args.run_name or ("hydraulic" if args.hydraulic_only else
                            "pilot" if args.pilot else "screen" if args.screen else "target")
    root = HERE / "runs" / tag
    braids, delft = root / "braids", root / "delft3d"
    reports = HERE / "reports" / tag
    if args.analyze_only:
        analyzer = audit_hydraulics if args.hydraulic_only else analyze
        print(json.dumps(analyzer(spec, braids, delft, reports), indent=2))
        return 0
    end_s = (1800.0 if args.hydraulic_only else
             spec["time"]["pilot_end_s" if args.pilot else "screen_end_s" if args.screen else "target_end_s"])
    output_s = (150.0 if args.hydraulic_only else
                spec["time"]["pilot_output_interval_s" if args.pilot else "screen_output_interval_s" if args.screen else "target_output_interval_s"])
    if not args.run_prepared:
        nb, _ = prepare_braids(spec, braids, end_s, output_s,
                              args.hydraulic_only, args.spinup_from)
        nd, _ = prepare_delft(spec, delft, end_s, output_s,
                             args.hydraulic_only, args.spinup_from)
        print(f"Prepared BRAIDS {nb} cells and Delft3D {nd} cells for {end_s:g} s", flush=True)
    elif not (braids / "case.data").is_file() or not (delft / "basin.mdf").is_file():
        raise FileNotFoundError("Both prepared native decks are required")
    if args.prepare_only:
        return 0
    print("Running native solvers concurrently", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        b_future = pool.submit(run_braids, spec, braids, args.timeout)
        d_future = pool.submit(delft_runtime.run_case, delft, timeout_s=args.timeout)
        braids_run = b_future.result()
        print("BRAIDS", json.dumps(braids_run), flush=True)
        delft_run = d_future.result()
        print("Delft3D", json.dumps(delft_run), flush=True)
    analyzer = audit_hydraulics if args.hydraulic_only else analyze
    print(json.dumps(analyzer(spec, braids, delft, reports), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
