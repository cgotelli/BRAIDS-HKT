"""Locate the native runtime in an installed plugin or source checkout."""
from __future__ import annotations

from pathlib import Path
import platform
import stat
import sys


PLUGIN_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PLUGIN_ROOT.parent


def platform_tag() -> str:
    """Return the distribution tag used below ``bin``."""
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(
        platform.system(), platform.system().lower())
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    return f"{system}-{machine}"


def bundled_native_executable() -> Path:
    name = "xbraids_cutcell.exe" if sys.platform == "win32" else "xbraids_cutcell"
    return PLUGIN_ROOT / "bin" / platform_tag() / name


def default_native_executable() -> Path | None:
    """Prefer the runtime bundled with a release, then the development build."""
    bundled = bundled_native_executable()
    if bundled.is_file():
        if sys.platform != "win32" and not bundled.stat().st_mode & stat.S_IXUSR:
            try:
                bundled.chmod(bundled.stat().st_mode | stat.S_IXUSR)
            except OSError:
                pass
        return bundled
    development = SOURCE_ROOT / "solver" / "cut_cells" / "build" / "xbraids_cutcell"
    return development if development.is_file() else None
