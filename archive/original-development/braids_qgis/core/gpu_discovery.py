"""Bounded Windows discovery for the external BRAIDS GPU runtime.

Discovery only proposes paths.  The dock still authenticates every worker
file, hashes the selected interpreter and executes the CUDA/FP64 probe before
GPU preparation is enabled.  It never changes the user's CPU/GPU selection.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from collections.abc import Iterable


def _path(value) -> Path | None:
    if not value:
        return None
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _unique(paths: Iterable[Path | None]):
    seen = set()
    for path in paths:
        if path is None:
            continue
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            yield path


def _worker_roots(plugin_root: str | Path | None = None):
    user = _path(os.environ.get("USERPROFILE"))
    local = _path(os.environ.get("LOCALAPPDATA"))
    plugin = _path(plugin_root)
    explicit = _path(os.environ.get("BRAIDS_GPU_WORKER"))
    direct = [explicit]
    if plugin is not None:
        direct.extend((plugin / "gpu-worker", plugin.parent / "gpu-worker",
                       plugin.parent / "BRAIDS-GPU-worker"))
    if user is not None:
        direct.extend((
            user / "BRAIDS-GPU-worker",
            user / "Desktop" / "BRAIDS-GPU-worker",
            user / "Desktop" / "BRAIDS" / "dist" / "gpu-worker",
            user / "Downloads",
        ))
    if local is not None:
        direct.extend((local / "BRAIDS" / "gpu-worker",
                       local / "BRAIDS"))
    return list(_unique(direct))


def discover_gpu_workers(plugin_root: str | Path | None = None) -> list[Path]:
    """Return likely extracted workers, newest first, without a drive scan."""
    candidates: list[Path] = []
    for root in _worker_roots(plugin_root):
        if (root / "PACKAGE-MANIFEST.json").is_file():
            candidates.append(root)
        if not root.is_dir():
            continue
        try:
            children = list(root.glob("BRAIDS-GPU-*-windows-x86_64"))
        except OSError:
            continue
        candidates.extend(child for child in children
                          if (child / "PACKAGE-MANIFEST.json").is_file())
    values = list(_unique(candidates))
    values.sort(key=lambda value: value.stat().st_mtime, reverse=True)
    return values


def discover_gpu_pythons(worker_root: str | Path | None = None) -> list[Path]:
    """Return likely external CUDA Python executables in preference order."""
    user = _path(os.environ.get("USERPROFILE"))
    local = _path(os.environ.get("LOCALAPPDATA"))
    worker = _path(worker_root)
    explicit = _path(os.environ.get("BRAIDS_GPU_PYTHON"))
    candidates: list[Path | None] = [explicit]
    if worker is not None:
        candidates.extend((worker / ".venv_gpu" / "Scripts" / "python.exe",
                           worker.parent / ".venv_gpu" / "Scripts" / "python.exe"))
    if user is not None:
        candidates.extend((
            user / "Desktop" / "BRAIDS" / "validation" / "gpu_fwave" /
                ".venv_gpu" / "Scripts" / "python.exe",
            user / "Desktop" / "BRAIDS" / ".venv_gpu" / "Scripts" / "python.exe",
            user / ".venv_gpu" / "Scripts" / "python.exe",
            user / "miniconda3" / "envs" / "braids-gpu" / "python.exe",
            user / "miniconda3" / "python.exe",
            user / "anaconda3" / "envs" / "braids-gpu" / "python.exe",
            user / "anaconda3" / "python.exe",
        ))
    if local is not None:
        programs = local / "Programs" / "Python"
        if programs.is_dir():
            candidates.extend(sorted(programs.glob("Python*/python.exe"),
                                     reverse=True))
    candidates.extend(_path(shutil.which(name)) for name in ("python", "python3"))
    return [path for path in _unique(candidates) if path.is_file()]


def discover_gpu_defaults(plugin_root: str | Path | None = None,
                          *, worker_hint: str | Path | None = None,
                          python_hint: str | Path | None = None) -> dict:
    """Find one worker/Python pair for populating the General tab.

    Hints (normally remembered QGIS settings) win when they still exist.
    Authentication and the CUDA probe intentionally remain separate.
    """
    hinted_worker = _path(worker_hint)
    workers = ([hinted_worker] if hinted_worker is not None and
               (hinted_worker / "PACKAGE-MANIFEST.json").is_file() else [])
    workers.extend(discover_gpu_workers(plugin_root))
    worker = next(iter(_unique(workers)), None)
    hinted_python = _path(python_hint)
    pythons = ([hinted_python] if hinted_python is not None and
               hinted_python.is_file() else [])
    pythons.extend(discover_gpu_pythons(worker))
    python = next(iter(_unique(pythons)), None)
    return {
        "worker_root": worker,
        "python_executable": python,
        "worker_candidates": len(list(_unique(workers))),
        "python_candidates": len(list(_unique(pythons))),
    }


__all__ = ["discover_gpu_defaults", "discover_gpu_pythons",
           "discover_gpu_workers"]
