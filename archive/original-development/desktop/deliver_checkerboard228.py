"""Create and qualify the preview49 Jiang--Cao Desktop experiment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "desktop/runtime"
OLD_PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment225"
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
OUT = ROOT / "validation/_checkerboard228"
REPORT = (ROOT / "validation/sediment/river_morphodynamics/reports/"
          "desktop_experiment225/checkerboard_delivery228")
REFERENCE = ROOT / "validation/_checkerboard227/runs"
RUNTIME_SHA = "f8f896a590b38f0a0563b2d04fb6db823948a42ad688829849a4dfab370da4c5"
SOURCE_ID = "873c28c8d90b7aa96661e63bf26ddbad51306ceff1d3df6590eac30fe5439e6c"
PYTHON = ROOT / "desktop/.venv/Scripts/python.exe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def install_imports() -> None:
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(QT_QPA_PLATFORM="offscreen", OMP_DYNAMIC="FALSE",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text(encoding="utf8")), RUNTIME)


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
    if sha(target / "RUNTIME.json") != RUNTIME_SHA:
        raise ValueError("Frozen project runtime identity differs")


def prepare_case(document: dict, backend: str, folder: Path) -> Path:
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_desktop.preparation import prepare
    request = deepcopy(document["model"]["request"])
    request["execution"]["backend"] = backend
    document = deepcopy(document)
    document["model"] = resolve_model(request).document()
    document["settings"]["controls"].update(threads=8, cfl=.8)
    document["settings"].update(dt_max_s=.02, end_time_s=60.,
                                output_interval_s=5., checkpoint_interval_s=5.)
    document["editor_state"]["checkpoint_clock"] = {
        "mode": "custom", "custom_interval_s": 5.
    }
    project = folder / "project.json"
    save(project, validate_project(document))
    started = time.perf_counter()
    prepared = prepare(project, folder / "prepared", asset_cache=PROJECT / "cache")
    save(folder / "PREPARATION.json", {
        "backend": backend,
        "seconds": time.perf_counter() - started,
        "runtime_manifest": RUNTIME_SHA,
        "source_identity": SOURCE_ID,
    })
    return prepared


def run_worker(backend: str, prepared: Path, folder: Path) -> dict:
    output = folder / "run"
    command = [str(PYTHON), "-X", "utf8", str(ROOT / "desktop/worker.py"),
               str(RUNTIME / "braids_core/river_launcher.py"), "run",
               "--input", str(prepared), "--output", str(output),
               "--until", "20", "--checkpoint-dir", str(folder / "checkpoints")]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=environment,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (folder / "worker.log").write_text(completed.stdout, encoding="utf8")
    if completed.returncode:
        raise RuntimeError(f"{backend} worker failed; see {folder/'worker.log'}")
    status = json.loads((output / "STATUS.json").read_text(encoding="utf8"))
    result = {
        "backend": backend,
        "wall_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "status": status,
        "runtime_manifest": RUNTIME_SHA,
        "source_identity": SOURCE_ID,
    }
    save(folder / "RESULT.json", result)
    return result


def frame_at(root: Path, time_s: float) -> dict[str, np.ndarray]:
    frames = root / "frames"
    index = json.loads((frames / "INDEX.json").read_text(encoding="utf8"))
    row = next(row for row in index["frames"] if abs(row["time_s"] - time_s) < 1e-10)
    with np.load(frames / row["file"], allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def differences(left: dict, right: dict) -> dict:
    fields = ("q", "z", "mobile_bedload_solid_m", "secondary_hI_m2_s",
              "bedload_vector_m2_s", "bedload_capacity_m2_s", "state")
    result = {}
    for name in fields:
        if left[name].shape != right[name].shape:
            raise ValueError(f"Field shape differs: {name}")
        result[name] = float(np.max(np.abs(left[name] - right[name])))
    return result


def reject_old_checkpoint(prepared: Path) -> dict:
    old = OLD_PROJECT / "checkpoints/experiment225/t_300_step_00015002.npz"
    folder = PROJECT / "qualification/old_checkpoint_rejection"
    output = folder / "run"
    command = [str(PYTHON), "-X", "utf8", str(ROOT / "desktop/worker.py"),
               str(RUNTIME / "braids_core/river_launcher.py"), "run",
               "--input", str(prepared), "--output", str(output),
               "--restart", str(old), "--until", "20"]
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "worker.log").write_text(completed.stdout, encoding="utf8")
    accepted = (completed.returncode != 0 and not output.exists() and
                "Checkpoint project, physics, backend, runtime or payload differs"
                in completed.stdout)
    result = {"old_checkpoint": str(old), "returncode": completed.returncode,
              "output_created": output.exists(), "rejected_before_run": accepted,
              "message": "Checkpoint project, physics, backend, runtime or payload differs"}
    save(folder / "RESULT.json", result)
    if not accepted:
        raise AssertionError(result)
    return result


def create_project() -> None:
    if PROJECT.exists():
        raise FileExistsError(f"Preserve existing project: {PROJECT}")
    install_imports()
    from braids_core.river_worker import source_identity
    from braids_desktop import services
    if sha(RUNTIME / "RUNTIME.json") != RUNTIME_SHA or source_identity() != SOURCE_ID:
        raise ValueError("Installed runtime identity differs")
    document = services.load_project_folder(OLD_PROJECT)
    document["name"] = "Jiang-Cao Case 1 Run 2 — corrected face-compatible MF1"
    services.save_project_folder(PROJECT, document, source_base=OLD_PROJECT)
    for name in ("logs", "references", "qualification"):
        (PROJECT / name).mkdir(exist_ok=True)
    for name in ("chino.pdf", "sup.pdf", "jian.pdf"):
        shutil.copy2(ROOT / name, PROJECT / "references" / name)
    copy_runtime()
    for name in ("launch.py", "worker.py"):
        shutil.copy2(ROOT / "desktop" / name, PROJECT / name)
    (PROJECT / "Open experiment.cmd").write_text(
        '@echo off\nstart "" "' + str(PYTHON.with_name("pythonw.exe")) +
        '" -B "%~dp0launch.py" "%~dp0."\n', encoding="utf8")

    evidence = PROJECT / "qualified_reference/evidence"
    evidence.mkdir(parents=True)
    source_report = (ROOT / "validation/sediment/river_morphodynamics/reports/"
                     "desktop_experiment225/checkerboard_fix227")
    for name in ("REPORT.md", "METRICS.json", "COUPLING_COMPARISON_60S.png",
                 "RAW_BED_240S.png", "TRANSVERSE_SIGNAL_240S.png",
                 "TRANSPORT_VARIABILITY_240S.png"):
        shutil.copy2(source_report / name, evidence / name)
    shutil.copytree(REFERENCE / "cpu_v3_restart240/run",
                    PROJECT / "qualified_reference/corrected_240s/run")
    shutil.copytree(REFERENCE / "cpu_v3_120s_b/prepared",
                    PROJECT / "qualified_reference/cpu_v3_120s_b/prepared")

    audit = {
        "schema": "braids.jiang-cao.delivery.v1",
        "runtime_manifest_sha256": RUNTIME_SHA,
        "source_identity": SOURCE_ID,
        "base_project": str(OLD_PROJECT),
        "physical_inputs_changed": False,
        "numerical_change": (
            "capacity-only face-compatible pair state and affine-exact "
            "flow-aligned bed derivative"
        ),
        "physical_formulas_changed": False,
        "qualified_reference": "qualified_reference/corrected_240s/run",
        "reference_is_raw": True,
        "display_smoothing_used_for_metrics": False,
    }
    save(PROJECT / "DELIVERY_AUDIT.json", audit)
    (PROJECT / "README.md").write_text(
        "# Jiang-Cao corrected MF1 experiment\n\n"
        "Open `Open experiment.cmd`. The project contains the same documented "
        "Jiang-Cao Case 1 reconstruction and a frozen preview49 runtime.\n\n"
        "`qualified_reference/corrected_240s/run` is the raw qualified 240 s "
        "trajectory. Its figures and audit are under `qualified_reference/evidence`. "
        "No visual smoothing was used to remove the former checkerboard pattern.\n\n"
        "The main project remains configured for the 18,000 s MF1 experiment. "
        "Prepare before starting a new run. Old prepared folders and checkpoints "
        "belonging to preview48 are intentionally rejected.\n",
        encoding="utf8")

    cpu_folder = PROJECT / "qualification/smoke_cpu"
    gpu_folder = PROJECT / "qualification/smoke_gpu"
    cpu_prepared = prepare_case(document, "cpu", cpu_folder)
    gpu_prepared = prepare_case(document, "gpu", gpu_folder)
    cpu_result = run_worker("cpu", cpu_prepared, cpu_folder)
    gpu_result = run_worker("gpu", gpu_prepared, gpu_folder)

    installed_cpu = frame_at(cpu_folder / "run", 20.)
    installed_gpu = frame_at(gpu_folder / "run", 20.)
    reference_cpu = frame_at(REFERENCE / "cpu_v3_120s_b/run", 20.)
    reference_gpu = frame_at(REFERENCE / "gpu_v3_20s/run", 20.)
    comparisons = {
        "installed_cpu_vs_qualified_cpu": differences(installed_cpu, reference_cpu),
        "installed_gpu_vs_qualified_gpu": differences(installed_gpu, reference_gpu),
        "installed_cpu_vs_installed_gpu": differences(installed_cpu, installed_gpu),
    }
    maximum = max(value for row in comparisons.values() for value in row.values())
    if maximum > 1e-10:
        raise AssertionError({"maximum_difference": maximum, "comparisons": comparisons})
    rejection = reject_old_checkpoint(cpu_prepared)
    services.update_project_session(
        PROJECT,
        prepared="qualified_reference/cpu_v3_120s_b/prepared",
        output="qualified_reference/corrected_240s/run",
        log="qualification/smoke_cpu/worker.log",
    )
    result = {
        "runtime_manifest_sha256": RUNTIME_SHA,
        "source_identity": SOURCE_ID,
        "project": str(PROJECT),
        "cpu": cpu_result,
        "gpu": gpu_result,
        "comparisons": comparisons,
        "maximum_saved_field_difference": maximum,
        "old_checkpoint": rejection,
        "qualified_240s_visible_on_open": True,
        "qualified_240s_display_smoothing": False,
    }
    save(OUT / "DELIVERY.json", result)
    save(PROJECT / "QUALIFICATION.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    create_project()
