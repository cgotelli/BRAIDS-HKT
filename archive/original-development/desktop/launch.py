"""Bootstrap the standalone UI with a single, authenticated solver namespace."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True

BASE = Path(__file__).resolve().parent
if sys.stderr is None or sys.stdout is None:
    (BASE/'evidence').mkdir(exist_ok=True)
    _startup_log=(BASE/'evidence'/'STARTUP.log').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=_startup_log
    if sys.stdout is None:sys.stdout=_startup_log
RUNTIME = BASE / "runtime"
if not (RUNTIME / "RUNTIME.json").is_file():
    raise SystemExit("BRAIDS runtime is missing. Run desktop/setup.ps1 first.")
sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(RUNTIME))
# The isolated Qt environment can share the host scientific installation.
if os.name == "nt":
    _dll_dirs = []
    for path in (Path(sys.base_prefix) / "Library" / "bin", Path(sys.base_prefix)):
        if path.is_dir():
            _dll_dirs.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ["PATH"]

from braids_desktop.app import main

if __name__ == "__main__":
    raise SystemExit(main(runtime=RUNTIME))
