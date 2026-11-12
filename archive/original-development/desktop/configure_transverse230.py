"""Fresh installed preview51 smoke checks and experiment preparation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_transverse230/installed"
PREPARED = PROJECT / "prepared/experiment230"
MANIFEST = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"
SOURCE = "e89341080f14ed1f87ea30e790784b43b086ede2a1ef55083c5aed20d00b71a2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def frame(output: Path) -> dict[str, np.ndarray]:
    root = output / "frames"
    index = json.loads((root / "INDEX.json").read_text(encoding="utf8"))
    with np.load(root / index["frames"][-1]["file"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main() -> None:
    if OUT.exists() or PREPARED.exists():
        raise FileExistsError("Preserve existing preview51 qualification artifacts")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    from braids_core.river_worker import source_identity
    if sha(RUNTIME / "RUNTIME.json") != MANIFEST or source_identity() != SOURCE:
        raise ValueError("Installed preview51 identity differs")
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core import river_worker

    document = services.load_project_folder(PROJECT)
    document["name"] = "Jiang-Cao Case 1 Run 3 - preview51 two-direction capacity"
    document["settings"]["controls"].update(cfl=0.5, threads=8)
    document["settings"].update(
        dt_max_s=0.03, end_time_s=18000.0, output_interval_s=60.0,
        checkpoint_interval_s=300.0,
    )
    document["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 300.0,
    }
    services.save_project_folder(PROJECT, document, source_base=PROJECT,
                                 session=services.project_session(PROJECT))

    results, frames = {}, {}
    prepared_cpu = None
    for backend in ("cpu", "gpu"):
        folder = OUT / backend
        value = deepcopy(document)
        request = deepcopy(value["model"]["request"])
        request["execution"]["backend"] = backend
        value["model"] = resolve_model(request).document()
        value["settings"].update(
            end_time_s=5.0, output_interval_s=5.0, checkpoint_interval_s=5.0,
        )
        value["editor_state"]["checkpoint_clock"] = {
            "mode": "custom", "custom_interval_s": 5.0,
        }
        project = folder / "project.json"
        save(project, validate_project(value))
        prepared = prepare(project, folder / "prepared", asset_cache=PROJECT / "cache")
        if backend == "cpu":
            prepared_cpu = prepared
        began = time.perf_counter()
        result = river_worker.run(
            prepared, folder / "run", until=5.0,
            checkpoint_dir=folder / "checkpoints",
        )
        results[backend] = {
            "result": result,
            "execution_s": time.perf_counter() - began,
        }
        frames[backend] = frame(folder / "run")

    fields = (
        "q", "z", "mobile_bedload_solid_m", "secondary_hI_m2_s",
        "bedload_vector_m2_s", "bedload_capacity_m2_s",
    )
    parity = {
        key: float(np.max(np.abs(frames["cpu"][key] - frames["gpu"][key])))
        for key in fields
    }
    old = PROJECT / "checkpoints/experiment229_gate_300s"
    checkpoint = max(old.glob("t_*_step_*.npz"), default=None)
    rejection = OUT / "old_checkpoint_rejection"
    message = ""
    if checkpoint is not None:
        try:
            river_worker.run(prepared_cpu, rejection, restart=checkpoint, until=300.0)
        except Exception as exc:
            message = str(exc)
    rejected = (
        "Checkpoint project, physics, backend, runtime or payload differs" in message
        and not rejection.exists()
    )

    prepared = prepare(PROJECT / "braids_project.json", PREPARED,
                       asset_cache=PROJECT / "cache")
    shutil.copy2(ROOT / "desktop/run_experiment230.py", PROJECT / "run_experiment230.py")
    python = ROOT / "desktop/.venv/Scripts/python.exe"
    (PROJECT / "Start long run.cmd").write_text(
        '@echo off\nstart "BRAIDS Jiang-Cao corrected" /b "' + str(python)
        + '" -X utf8 "%~dp0run_experiment230.py" run '
        + '> "%~dp0logs\\experiment230_18000.log" 2>&1\n', encoding="utf8",
    )
    (PROJECT / "Check long run.cmd").write_text(
        '@echo off\n"' + str(python)
        + '" -X utf8 "%~dp0run_experiment230.py" status\n', encoding="utf8",
    )
    (PROJECT / "Stop long run safely.cmd").write_text(
        '@echo off\ntype nul > "%~dp0STOP_18000.request"\n'
        'echo Stop requested. BRAIDS will seal a checkpoint before exiting.\npause\n',
        encoding="utf8",
    )
    record = {
        "runtime_manifest": MANIFEST,
        "source_identity": SOURCE,
        "project": str(PROJECT),
        "prepared": str(PREPARED),
        "settings": {
            "cfl": 0.5, "dt_max_s": 0.03, "end_time_s": 18000.0,
            "output_interval_s": 60.0, "checkpoint_interval_s": 300.0,
            "threads": 8,
        },
        "results": results,
        "cpu_gpu_max_abs": parity,
        "preview50_checkpoint_rejected": rejected,
        "rejection_message": message,
    }
    save(OUT / "CHECK.json", record)
    save(PROJECT / "TRANSVERSE_QUALIFICATION_230.json", record)
    if not (
        all(row["result"]["status"] == "complete" for row in results.values())
        and max(parity.values()) < 1e-10 and rejected
    ):
        raise RuntimeError(record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
