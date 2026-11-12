"""Independent 300 -> 18000 s continuation for the Jiang--Cao experiment."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
LOCAL_COPY = (HERE / "runtime").is_dir()
PROJECT = (HERE if LOCAL_COPY
           else HERE.parent.parent / "BRAIDS_JiangCao_Experiment228")
ROOT = HERE.parent / "BRAIDS" if LOCAL_COPY else HERE.parent
RUNTIME = PROJECT / "runtime"
PREPARED = PROJECT / "prepared/experiment229"
SOURCE = PROJECT / "checkpoints/experiment229_gate_300s"
OUTPUT = PROJECT / "runs/experiment229_18000s"
CHECKPOINTS = PROJECT / "checkpoints/experiment229_18000s"
STOP = PROJECT / "STOP_18000.request"
STATE = PROJECT / "LONG_RUN_STATE.json"
MANIFEST = "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def latest_checkpoint() -> Path:
    rows = []
    for sidecar in SOURCE.glob("t_*_step_*.json"):
        row = json.loads(sidecar.read_text(encoding="utf8"))
        rows.append((float(row["time_s"]), sidecar.with_suffix(".npz")))
    if not rows or abs(max(rows)[0] - 300.0) > 1e-9:
        raise ValueError("The authenticated 300 s checkpoint is required")
    checkpoint = max(rows)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def status() -> None:
    while True:
        progress = OUTPUT / "PROGRESS.json"
        state = json.loads(progress.read_text(encoding="utf8")) if progress.exists() else {}
        now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        os.system("cls" if os.name == "nt" else "clear")
        print("BRAIDS Jiang-Cao MF1 long run\n")
        print("Project:", PROJECT)
        print("Updated:", now)
        print("Status:", state.get("status", "starting"))
        print("Phase:", state.get("phase", "loading frozen runtime"))
        print(f"Simulation: {float(state.get('time_s', 300.0)):.3f} / 18000 s")
        print(f"Progress: {100 * float(state.get('time_s', 300.0)) / 18000:.4f}%")
        print("CFL: 0.5 | maximum outer timestep: 0.03 s | CPU threads: 8")
        if state.get("error"):
            print("Error:", state["error"])
        if state.get("status") in {"complete", "failed", "stopped"}:
            input("\nRun ended. Press Enter to close.")
            return
        print("\nRefreshes every 10 seconds. Press Ctrl+C to close this monitor only.")
        time.sleep(10)


def run() -> None:
    if OUTPUT.exists() or CHECKPOINTS.exists():
        raise FileExistsError("Preserve the existing long-run output")
    STOP.unlink(missing_ok=True)
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    digest = hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest()
    if digest != MANIFEST:
        raise ValueError("Frozen project runtime differs")
    from braids_core import river_worker
    from braids_desktop import services
    checkpoint = latest_checkpoint()
    began = time.perf_counter()
    started = dict(status="starting", pid=os.getpid(), started_utc=dt.datetime.now(
        dt.timezone.utc).isoformat(), restart=str(checkpoint), target_s=18000.0,
        cfl=.5, dt_max_s=.03, threads=8, runtime=MANIFEST)
    atomic(STATE, started)
    services.update_project_session(PROJECT,
        prepared=str(PREPARED.relative_to(PROJECT)),
        output=str(OUTPUT.relative_to(PROJECT)),
        log="logs/experiment229_18000.log")
    try:
        result = river_worker.run(PREPARED, OUTPUT, restart=checkpoint, until=18000.,
                                  checkpoint_dir=CHECKPOINTS, stop_file=STOP)
        atomic(STATE, dict(started, status=result["status"], result=result,
                           wall_seconds=time.perf_counter() - began,
                           ended_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    except BaseException as exc:
        atomic(STATE, dict(started, status="failed", error=f"{type(exc).__name__}: {exc}",
                           wall_seconds=time.perf_counter() - began,
                           ended_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "status"), nargs="?", default="run")
    arguments = parser.parse_args()
    (run if arguments.action == "run" else status)()
