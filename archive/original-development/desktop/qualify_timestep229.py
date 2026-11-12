"""Qualify a modest Jiang--Cao outer-step increase at fixed CFL 0.5."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_experiment229/timestep"
RUNTIME_SHA = "f8f896a590b38f0a0563b2d04fb6db823948a42ad688829849a4dfab370da4c5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def imports() -> None:
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)


def last_frame(output: Path) -> dict[str, np.ndarray]:
    frames = output / "frames"
    index = json.loads((frames / "INDEX.json").read_text())
    with np.load(frames / index["frames"][-1]["file"], allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def execute(document: dict, label: str, dt_max: float) -> dict:
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    folder = OUT / label
    if folder.exists():
        raise FileExistsError(f"Preserve qualification evidence: {folder}")
    request = deepcopy(document["model"]["request"])
    request["execution"]["backend"] = "cpu"
    value = deepcopy(document)
    value["model"] = resolve_model(request).document()
    value["settings"]["controls"].update(cfl=.5, threads=8)
    value["settings"].update(dt_max_s=dt_max, end_time_s=30.,
                             output_interval_s=10., checkpoint_interval_s=30.)
    value["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 30.
    }
    project = folder / "project.json"
    save(project, validate_project(value))
    started = time.perf_counter()
    prepared = prepare(project, folder / "prepared", asset_cache=PROJECT / "cache")
    preparation_s = time.perf_counter() - started
    started = time.perf_counter()
    result = river_worker.run(prepared, folder / "run",
                              checkpoint_dir=folder / "checkpoints")
    execution_s = time.perf_counter() - started
    diagnostics = json.loads((folder / "run/DIAGNOSTICS.json").read_text())[-1]
    row = {"label": label, "cfl": .5, "dt_max_s": dt_max,
           "preparation_s": preparation_s, "execution_s": execution_s,
           "result": result, "diagnostics": diagnostics,
           "runtime": sha(RUNTIME / "RUNTIME.json")}
    save(folder / "RESULT.json", row)
    return row


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve qualification evidence: {OUT}")
    imports()
    from braids_desktop import services
    document = services.load_project_folder(PROJECT)
    control = execute(document, "cfl050_dt0020", .02)
    candidate = execute(document, "cfl050_dt0030", .03)
    a = last_frame(OUT / "cfl050_dt0020/run")
    b = last_frame(OUT / "cfl050_dt0030/run")
    fields = ("q", "z", "mobile_bedload_solid_m", "secondary_hI_m2_s",
              "bedload_vector_m2_s", "bedload_capacity_m2_s")
    differences = {name: float(np.max(np.abs(a[name] - b[name]))) for name in fields}
    checks = {
        "candidate_complete": candidate["result"]["status"] == "complete",
        "candidate_rejections": len(candidate["result"]["rejections"]),
        "solid_residual_m3": candidate["diagnostics"]["solid_residual_m3"],
        "water_plus_pore_residual_m3": candidate["diagnostics"]["water_plus_pore_residual_m3"],
        "bed_difference_mm": 1000. * differences["z"],
        "depth_difference_mm": 1000. * differences["q"],
    }
    accepted = (checks["candidate_complete"] and checks["candidate_rejections"] == 0 and
                abs(checks["solid_residual_m3"]) < 1e-12 and
                abs(checks["water_plus_pore_residual_m3"]) < 1e-12 and
                checks["bed_difference_mm"] < 1. and checks["depth_difference_mm"] < 2.)
    record = {"runtime": RUNTIME_SHA, "control": control, "candidate": candidate,
              "differences": differences, "checks": checks, "accepted": accepted,
              "selected": {"cfl": .5, "dt_max_s": .03} if accepted else None}
    save(OUT / "QUALIFICATION.json", record)
    print(json.dumps(record, indent=2), flush=True)
    if not accepted:
        raise RuntimeError("The 0.03 s candidate did not pass")


if __name__ == "__main__":
    main()
