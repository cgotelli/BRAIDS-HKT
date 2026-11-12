"""Independent time-zero Jiang--Cao trapezoidal-channel experiment."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parent
RUNTIME = PROJECT / "runtime"
PREPARED = PROJECT / "prepared/trapezoid231"
OUTPUT = PROJECT / "runs/trapezoid231_18000s"
CHECKPOINTS = PROJECT / "checkpoints/trapezoid231_18000s"
STOP = PROJECT / "STOP_18000.request"
STATE = PROJECT / "LONG_RUN_STATE.json"
MANIFEST = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def status() -> None:
    while True:
        progress = OUTPUT / "PROGRESS.json"
        state = json.loads(progress.read_text(encoding="utf8")) if progress.exists() else {}
        run_state = json.loads(STATE.read_text(encoding="utf8")) if STATE.exists() else {}
        os.system("cls" if os.name == "nt" else "clear")
        print("BRAIDS Jiang-Cao trapezoidal-channel run\n")
        print("Project:", PROJECT)
        print("Updated:", dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
        print("Status:", state.get("status", run_state.get("status", "starting")))
        print("Phase:", state.get("phase", "loading frozen runtime"))
        current = float(state.get("time_s", 0.0))
        print(f"Simulation: {current:.3f} / 18000 s")
        print(f"Progress: {100.0 * current / 18000.0:.4f}%")
        print("CFL: 0.5 | maximum outer timestep: 0.03 s | CPU threads: 8")
        error = state.get("error", run_state.get("error"))
        if error:
            print("Error:", error)
        terminal = state.get("status", run_state.get("status"))
        if terminal in {"complete", "failed", "stopped"}:
            input("\nRun ended. Press Enter to close.")
            return
        print("\nRefreshes every 10 seconds. Press Ctrl+C to close this monitor only.")
        time.sleep(10)


def run() -> None:
    if OUTPUT.exists() or CHECKPOINTS.exists():
        raise FileExistsError("Preserve the existing trapezoidal-channel output")
    if not (PREPARED / "PREPARED.json").is_file():
        raise FileNotFoundError("Prepared trapezoidal-channel project is required")
    STOP.unlink(missing_ok=True)
    affinity = None
    if os.name == "nt" and (os.cpu_count() or 0) >= 32:
        # Keep this second eight-thread solver on the upper half of this PC's
        # logical CPUs.  The already-running experiment uses the lower half.
        count = int(os.cpu_count())
        affinity = ((1 << count) - 1) ^ ((1 << (count // 2)) - 1)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), affinity):
            raise ctypes.WinError(ctypes.get_last_error())
    sys.path.insert(0, str(RUNTIME))
    os.environ.update(
        OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    digest = hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest()
    if digest != MANIFEST:
        raise ValueError("Frozen project runtime differs")
    from braids_core import river_worker
    from braids_desktop import services
    began = time.perf_counter()
    started = {
        "status": "starting", "pid": os.getpid(),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "restart": None, "target_s": 18000.0, "cfl": 0.5,
        "dt_max_s": 0.03, "threads": 8, "runtime": MANIFEST,
        "processor_affinity_mask": affinity,
    }
    atomic(STATE, started)
    services.update_project_session(
        PROJECT, prepared=PREPARED.relative_to(PROJECT).as_posix(),
        output=OUTPUT.relative_to(PROJECT).as_posix(),
        log="logs/trapezoid231_18000.log",
    )
    try:
        result = river_worker.run(
            PREPARED, OUTPUT, until=18000.0, checkpoint_dir=CHECKPOINTS,
            stop_file=STOP,
        )
        atomic(STATE, dict(
            started, status=result["status"], result=result,
            wall_seconds=time.perf_counter() - began,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        ))
    except BaseException as exc:
        atomic(STATE, dict(
            started, status="failed", error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.perf_counter() - began,
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        ))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "status"), nargs="?", default="run")
    arguments = parser.parse_args()
    (run if arguments.action == "run" else status)()
