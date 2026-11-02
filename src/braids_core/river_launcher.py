"""Authenticated installed entry for the single river worker."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def verify():
    manifest = json.loads((ROOT/"RUNTIME.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("route") != "unified_river_v1":
        raise ValueError("Only the current unified river runtime is supported")
    for name, checksum in manifest["files"].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Runtime artifact differs: "+name)
    return manifest


def main():
    verify()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((ROOT/"PRECOMPILED.json").read_text(encoding="utf-8")), ROOT)
    from braids_core.river_worker import main as execute
    execute()


if __name__ == "__main__":
    main()
