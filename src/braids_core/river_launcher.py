"""Authenticated installed entry for the single river worker."""
import hashlib
import json
from pathlib import Path
import sys
import warnings

# CuPy can operate correctly through the installed CUDA runtime even when its
# optional toolkit-path discovery cannot identify CUDA_PATH.  That advisory is
# not actionable in BRAIDS and was being printed as if a solver warning.  Keep
# every actual CUDA import, driver, compilation and allocation failure visible.
warnings.filterwarnings('ignore',
    message=r'CUDA path could not be detected\. Set CUDA_PATH environment variable if CuPy fails to load\.',
    category=UserWarning, module=r'cupy\._environment')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def verify():
    manifest = json.loads((ROOT/"RUNTIME.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("route") != "unified_river_v1":
        raise ValueError("Only the current unified river runtime is supported")
    # Authenticate these exact bootstrap bytes before executing any application
    # helper. It imports only stdlib; the rest of the application stays unloaded
    # until the full manifest check has completed.
    name='braids_core/runtime_verification.py';path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Runtime artifact differs: '+name)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['files'].get(name):
        raise ValueError('Runtime artifact differs: '+name)
    namespace=dict(__name__='_braids_verified_runtime',__file__=str(path))
    exec(compile(raw,str(path),'exec'),namespace)
    global LAST_VERIFICATION
    LAST_VERIFICATION=namespace['verify_files'](ROOT,manifest['files'])
    return manifest


def main():
    verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((ROOT/"PRECOMPILED.json").read_text(encoding="utf-8")), ROOT)
    from braids_core.river_worker import main as execute
    execute()


if __name__ == "__main__":
    main()
