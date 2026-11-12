"""Check an external GPU Python without importing CUDA libraries into QGIS.

This is a capability check, not an installer or a signature for the selected
Python environment. The caller separately pins that executable and verifies
the extracted source worker before running the probe.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


_PROBE = r'''
import json
import sys

try:
    import cupy as cp
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise RuntimeError("No CUDA device is visible to the selected Python")
    device_id = int(cp.cuda.Device().id)
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    free_memory, total_memory = cp.cuda.runtime.memGetInfo()
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    source = ('extern "C" __global__ void braids_fp64_probe('
              'const double *x, double *y) { y[0] = 2.0 * x[0]; }')
    kernel = cp.RawKernel(source, "braids_fp64_probe")
    x = cp.asarray([1.0], dtype=cp.float64)
    y = cp.empty_like(x)
    kernel((1,), (1,), (x, y))
    cp.cuda.get_current_stream().synchronize()
    result = float(cp.asnumpy(y)[0])
    if result != 2.0:
        raise RuntimeError("FP64 CUDA probe returned an incorrect result")
    record = {"schema": 1, "status": "ready", "cupy": cp.__version__,
              "device_count": device_count, "device_id": device_id,
              "device_name": str(name),
              "compute_major": int(properties["major"]),
              "compute_minor": int(properties["minor"]),
              "device_free_memory_bytes": int(free_memory),
              "device_total_memory_bytes": int(total_memory),
              "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
              "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
              "fp64_probe": result}
    print(json.dumps(record, allow_nan=False), flush=True)
except Exception as exc:
    print(json.dumps({"schema": 1, "status": "unavailable",
                      "error": type(exc).__name__ + ": " + str(exc)[:500]}), flush=True)
    sys.exit(2)
'''


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker_cuda_requirements(worker_root: str | Path) -> tuple[str, int]:
    """Return the exact packaged CuPy version and minimum CUDA runtime API."""
    path = Path(worker_root).expanduser().resolve() / "requirements-gpu.txt"
    rows = path.read_text(encoding="utf-8").splitlines()
    cupy = []
    runtime = []
    for row in rows:
        item = row.split("#", 1)[0].strip()
        if not item:
            continue
        match = re.fullmatch(r"cupy-cuda12x(?:\[[^]]+\])?==([0-9]+\.[0-9]+\.[0-9]+)", item)
        if match:
            cupy.append(match.group(1))
            continue
        match = re.fullmatch(r"nvidia-cuda-runtime-cu12==([0-9]+)\.([0-9]+)\.[0-9]+", item)
        if match:
            runtime.append(1000 * int(match.group(1)) + 10 * int(match.group(2)))
    if len(cupy) != 1 or len(runtime) != 1:
        raise ValueError("GPU worker must pin one CuPy and CUDA runtime version")
    return cupy[0], runtime[0]


def probe_gpu_python(python_executable: str | Path, worker_root: str | Path, *,
                     expected_python_sha256: str | None = None,
                     timeout_seconds: float = 30.) -> dict:
    """Prove that the pinned external Python can execute one FP64 CUDA kernel.

    Run in isolated mode, outside QGIS. A failed or hung probe does not create
    a solver output and never selects CPU. The worker payload must already have
    passed ``inspect_gpu_source_worker``; this function reads its pinned
    requirements but does not execute worker source.
    """
    python = Path(python_executable).expanduser().resolve()
    if not python.is_file():
        raise ValueError("Select an existing external GPU Python executable")
    before = _sha256(python)
    if expected_python_sha256 is not None and before != expected_python_sha256:
        raise ValueError("External GPU Python changed since it was pinned")
    required_cupy, minimum_runtime = worker_cuda_requirements(worker_root)
    try:
        process = subprocess.run([str(python), "-I", "-c", _PROBE],
                                 capture_output=True, text=True,
                                 timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("GPU runtime preflight timed out; no solver was launched") from exc
    except OSError as exc:
        raise ValueError("Could not start external GPU Python: " + str(exc)) from exc
    if _sha256(python) != before:
        raise ValueError("External GPU Python changed during capability preflight")
    try:
        record = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        detail = (process.stderr or process.stdout).strip()[-500:]
        raise ValueError("External GPU Python returned no valid capability record: " + detail) from exc
    if not isinstance(record, dict) or record.get("schema") != 1:
        raise ValueError("External GPU Python returned an unsupported capability record")
    if process.returncode != 0 or record.get("status") != "ready":
        raise ValueError("GPU runtime preflight failed: " + str(record.get("error", "unknown CUDA error")))
    if record.get("cupy") != required_cupy:
        raise ValueError(f"GPU worker requires CuPy {required_cupy}; selected Python has {record.get('cupy')}")
    if (not isinstance(record.get("cuda_runtime"), int) or
            record["cuda_runtime"] < minimum_runtime):
        raise ValueError(f"GPU worker requires CUDA runtime {minimum_runtime}; selected Python has {record.get('cuda_runtime')}")
    free_memory = record.get("device_free_memory_bytes")
    total_memory = record.get("device_total_memory_bytes")
    if (not isinstance(record.get("cuda_driver"), int) or record["cuda_driver"] <= 0 or
            not isinstance(record.get("device_count"), int) or record["device_count"] < 1 or
            not isinstance(record.get("device_name"), str) or not record["device_name"] or
            not isinstance(free_memory, int) or free_memory <= 0 or
            not isinstance(total_memory, int) or total_memory <= 0 or
            free_memory > total_memory or
            record.get("fp64_probe") != 2.0):
        raise ValueError("GPU runtime preflight returned incomplete device/FP64 evidence")
    return {**record, "python_sha256": before,
            "required_cupy": required_cupy,
            "minimum_cuda_runtime": minimum_runtime}
