"""Synchronous native execution used by tests and non-QGIS automation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Sequence

from .project import (RUN_MARKER, update_run_status, verified_native_executable,
                      verified_prepared_inputs)


def _text(value):
    if value is None:return ""
    return value.decode("utf-8",errors="replace") if isinstance(value,bytes) else str(value)


def run_native(run_root: str | Path, *, checkpoint: str | Path | None = None,
               timeout_s: float | None = None, extra_arguments: Sequence[str] = ()) -> subprocess.CompletedProcess:
    run_root=Path(run_root).expanduser().resolve()
    import json
    marker=json.loads((run_root/RUN_MARKER).read_text(encoding="utf-8"))
    if marker.get("preparation_error"):
        raise ValueError("Run preparation failed: "+marker["preparation_error"])
    if marker.get("status") not in {"prepared","failed","cancelled"}:
        raise ValueError("BRAIDS run is not prepared for execution")
    verified_prepared_inputs(run_root,marker)
    executable=verified_native_executable(run_root,marker);arguments=["case.data",*map(str,extra_arguments)]
    if checkpoint is not None:arguments=["case.data",str(Path(checkpoint).expanduser().resolve()),*map(str,extra_arguments)]
    update_run_status(run_root,"running",checkpoint=str(checkpoint) if checkpoint else None)
    environment={**os.environ,"OMP_NUM_THREADS":str(marker["threads"]),"OMP_DYNAMIC":"FALSE",
                 "OMP_PROC_BIND":"FALSE","OMP_WAIT_POLICY":"ACTIVE","GOMP_SPINCOUNT":"1000000",
                 "BRAIDS_CHECKPOINT_STEPS":"10000"}
    # Diagnostic shell settings are not authenticated project inputs.  The
    # native solver's default/auto policy remains in force for this launcher.
    environment.pop("BRAIDS_TIME_INTEGRATOR",None)
    try:
        result=subprocess.run([str(executable),*arguments],cwd=run_root,capture_output=True,text=True,
                              env=environment,timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        (run_root/"native.log").write_text(_text(exc.stdout)+_text(exc.stderr),encoding="utf-8")
        update_run_status(run_root,"failed",execution_error="timeout")
        raise
    (run_root/"native.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    result_stream=(run_root/"frames.txt").is_file() or (run_root/"live_preview"/"frames.csv").is_file()
    complete=(result.returncode==0 and (run_root/"checkpoint.bin").is_file() and
              (run_root/"boundary_ledger.txt").is_file() and result_stream)
    update_run_status(run_root,"completed" if complete else "failed",returncode=result.returncode,
                      execution_error=None if complete else "native process failed or output is incomplete")
    if not complete:raise RuntimeError(f"Native BRAIDS run failed; inspect {run_root/'native.log'}")
    return result
