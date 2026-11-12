"""Coupled CPU/GPU and restart qualification for the 2-D compact candidate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = ROOT / "validation/_transverse230/candidate_2d_compact"
OUT = ROOT / "validation/_transverse230/backends"
EXPECTED = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def frame(folder: Path, time_s: float) -> dict[str, np.ndarray]:
    root = folder / "frames"
    index = json.loads((root / "INDEX.json").read_text(encoding="utf8"))
    row = next(item for item in index["frames"] if abs(item["time_s"] - time_s) < 1.e-9)
    with np.load(root / row["file"]) as values:
        return {name: values[name].copy() for name in values.files}


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve existing backend qualification")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8",
        OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        QT_QPA_PLATFORM="offscreen",
    )
    if hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Candidate runtime differs")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install

    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    from braids_core import river_worker
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_desktop import services
    from braids_desktop.preparation import prepare

    original = services.load_project_folder(PROJECT)
    results = {}
    for backend in ("cpu", "gpu"):
        document = deepcopy(original)
        request = deepcopy(document["model"]["request"])
        request["execution"]["backend"] = backend
        document["model"] = resolve_model(request).document()
        document["settings"].update(
            end_time_s=20., output_interval_s=20., checkpoint_interval_s=10.
        )
        document["settings"]["controls"].update(cfl=.5, threads=8)
        document["editor_state"]["checkpoint_clock"] = {
            "mode": "custom", "custom_interval_s": 10.
        }
        project = OUT / backend / "project.json"
        save(project, validate_project(document))
        prepared = prepare(project, OUT / backend / "prepared", asset_cache=PROJECT / "cache")
        results[backend] = river_worker.run(
            prepared,
            OUT / backend / "run20",
            until=20.,
            checkpoint_dir=OUT / backend / "checkpoints20",
        )

    cpu = frame(OUT / "cpu/run20", 20.)
    gpu = frame(OUT / "gpu/run20", 20.)
    fields = (
        "q", "z", "state", "mobile_bedload_solid_m",
        "bedload_vector_m2_s", "bedload_capacity_m2_s",
    )
    differences = {
        name: float(np.max(np.abs(cpu[name] - gpu[name]))) for name in fields
    }
    diagnostics = {}
    for backend in ("cpu", "gpu"):
        diagnostics[backend] = json.loads((
            OUT / backend / "run20/DIAGNOSTICS.json"
        ).read_text(encoding="utf8"))[-1]
    record = {
        "runtime_sha256": EXPECTED,
        "results": results,
        "cpu_gpu_max_abs": differences,
        "diagnostics": diagnostics,
        "accepted": (
            max(differences.values()) < 1.e-10
            and all(abs(row["solid_residual_m3"]) < 1.e-12 for row in diagnostics.values())
            and all(abs(row["water_plus_pore_residual_m3"]) < 1.e-12 for row in diagnostics.values())
        ),
    }
    save(OUT / "QUALIFICATION.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
