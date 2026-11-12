"""Measure the prepared trapezoidal case on GPU while the CPU experiment runs."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Trapezoid231"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_trapezoid231/gpu_gate"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core import river_worker

    document = services.load_project_folder(PROJECT)
    request = deepcopy(document["model"]["request"])
    request["execution"]["backend"] = "gpu"
    document["model"] = resolve_model(request).document()
    document["settings"].update(
        end_time_s=5.0, output_interval_s=5.0, checkpoint_interval_s=5.0,
    )
    document["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 5.0,
    }
    save(OUT / "project.json", validate_project(document))
    prepared = prepare(OUT / "project.json", OUT / "prepared",
                       asset_cache=PROJECT / "cache")
    began = time.perf_counter()
    result = river_worker.run(prepared, OUT / "run",
                              checkpoint_dir=OUT / "checkpoints")
    record = {"result": result, "wall_s": time.perf_counter() - began}
    save(OUT / "RESULT.json", record)
    if result["status"] != "complete":
        raise RuntimeError(record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
