"""Planar-bed and perturbed-bed short controls for transverse striping."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_transverse230"
MANIFEST = "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def main() -> None:
    if (OUT / "runs").exists():
        raise FileExistsError("Preserve the existing transverse controls")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify(); install(json.loads((RUNTIME/"PRECOMPILED.json").read_text()), RUNTIME)
    if hashlib.sha256((RUNTIME/"RUNTIME.json").read_bytes()).hexdigest() != MANIFEST:
        raise ValueError("Frozen runtime differs")
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core import river_worker
    from braids_core.river_project import validate_project
    document = services.load_project_folder(PROJECT)
    # Build an exactly planar DEM on the same raster support.  Only the
    # documented perturbation is removed; domain, mesh and all physics remain.
    from osgeo import gdal
    import numpy as np
    source = gdal.Open(str(PROJECT / document["terrain"]["dem_path"]))
    transform = source.GetGeoTransform(); projection = source.GetProjection()
    nx, ny = source.RasterXSize, source.RasterYSize
    x = transform[0] + (np.arange(nx)+.5)*transform[1]
    plane = np.broadcast_to(.01*(25-x), (ny, nx)).copy()
    flat_dem = OUT / "flume_planar.tif"; flat_dem.parent.mkdir(parents=True, exist_ok=True)
    target = gdal.GetDriverByName("GTiff").Create(str(flat_dem), nx, ny, 1,
                                                   gdal.GDT_Float64,
                                                   options=["COMPRESS=LZW"])
    target.SetGeoTransform(transform); target.SetProjection(projection)
    target.GetRasterBand(1).WriteArray(plane); target = None; source = None
    flat = deepcopy(document); flat["terrain"]["dem_path"] = str(flat_dem)
    flat["settings"].update(end_time_s=10., output_interval_s=10., checkpoint_interval_s=10.)
    flat["editor_state"]["checkpoint_clock"] = dict(mode="custom", custom_interval_s=10.)
    flat_json = OUT / "flat_project.json"; save(flat_json, validate_project(flat))
    flat_prepared = prepare(flat_json, OUT/"prepared_flat", asset_cache=OUT/"cache")
    results = {}
    results["flat"] = river_worker.run(flat_prepared, OUT/"runs/flat10", until=10.,
                                        checkpoint_dir=OUT/"checkpoints/flat10")
    results["perturbed"] = river_worker.run(PROJECT/"prepared/experiment229",
        OUT/"runs/perturbed10", until=10., checkpoint_dir=OUT/"checkpoints/perturbed10")
    save(OUT/"RUNS.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
