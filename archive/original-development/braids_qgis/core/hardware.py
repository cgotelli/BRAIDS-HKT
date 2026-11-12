"""Small dependency-free hardware queries for native OpenMP defaults."""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
import subprocess
import sys


def _windows_physical_cores() -> int | None:
    if sys.platform != "win32":
        return None
    relation_processor_core = 0
    needed = ctypes.c_ulong(0)
    function = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
    function(relation_processor_core, None, ctypes.byref(needed))
    if needed.value < 8:
        return None
    buffer = ctypes.create_string_buffer(needed.value)
    if not function(relation_processor_core, buffer, ctypes.byref(needed)):
        return None
    count = 0
    offset = 0
    while offset + 8 <= needed.value:
        relationship = ctypes.c_ulong.from_buffer(buffer, offset).value
        size = ctypes.c_ulong.from_buffer(buffer, offset + 4).value
        if size < 8 or offset + size > needed.value:
            return None
        if relationship == relation_processor_core:
            count += 1
        offset += size
    return count or None


def _linux_physical_cores() -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    cores = set()
    for core_path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
        path = Path(core_path)
        try:
            package = (path.parent / "physical_package_id").read_text().strip()
            cores.add((package, path.read_text().strip()))
        except OSError:
            return None
    return len(cores) or None


def _macos_physical_cores() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"], capture_output=True,
            text=True, timeout=2, check=True,
        )
        value = int(result.stdout.strip())
        return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def physical_core_count() -> int:
    """Return physical cores when discoverable, otherwise logical CPUs."""
    logical = max(1, os.cpu_count() or 1)
    physical = (
        _windows_physical_cores()
        or _linux_physical_cores()
        or _macos_physical_cores()
    )
    if physical is None or physical < 1 or physical > logical:
        return logical
    return physical


def recommended_openmp_threads() -> int:
    """Favor physical cores; SMT remains selectable for sufficiently large runs."""
    return physical_core_count()
