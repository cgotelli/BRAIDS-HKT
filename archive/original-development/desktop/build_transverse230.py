"""Build an isolated preview50 runtime with 2-D compact capacity forcing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "desktop/runtime"
OUT = ROOT / "validation/_transverse230"
TARGET = OUT / "candidate_2d_compact"
SOURCES = (
    "braids_core/sediment/jiang_cao_coupled.py",
    "braids_core/sediment/jiang_cao_coupled.h",
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def verify(root: Path) -> dict:
    manifest = json.loads((root / "RUNTIME.json").read_text(encoding="utf8"))
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Runtime payload differs: {name}")
    return manifest


def main() -> None:
    base = verify(BASE)
    if TARGET.exists():
        raise FileExistsError(f"Preserve existing candidate: {TARGET}")
    for name in base["files"]:
        target = TARGET / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    manifest = json.loads(json.dumps(base))
    for name in SOURCES:
        shutil.copy2(ROOT / name, TARGET / name)
        manifest["files"][name] = sha(TARGET / name)

    sys.path[:0] = [str(ROOT)]
    from braids_core.sediment.jiang_cao_coupled import NAMES
    from braids_core.sediment.portable_river_runtime import _bindings
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

    source = ROOT / "braids_core/sediment/jiang_cao_coupled.h"
    engine = WaveEngine("cpu", 8, source=source, names=NAMES)
    row = _bindings(engine)
    table = json.loads((BASE / "PRECOMPILED.json").read_text(encoding="utf8"))
    existing = next(
        value
        for key, value in table.items()
        if key.startswith(
            "validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:"
        )
    )
    relative = Path(next(iter(existing["libraries"].values()))["file"])
    dependencies = [
        path
        for path in (BASE / relative.parent).glob("*.dll")
        if path.name != relative.name
    ]
    added = []
    for attribute, path in list(row["libraries"].items()):
        library = Path(path)
        folder = TARGET / "precompiled" / sha(library)[:20]
        folder.mkdir(exist_ok=True)
        for item in [library, source, *dependencies]:
            destination = folder / item.name
            shutil.copy2(item, destination)
            name = destination.relative_to(TARGET).as_posix()
            manifest["files"][name] = sha(destination)
            added.append(name)
        row["libraries"][attribute] = {
            "file": (folder / library.name).relative_to(TARGET).as_posix(),
            "sha256": sha(library),
        }
    key = (
        "validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:"
        + sha(source)
        + ":"
        + ",".join(NAMES)
    )
    if key in table:
        raise ValueError("Candidate binding unexpectedly exists in baseline")
    table[key] = row
    save(TARGET / "PRECOMPILED.json", table)
    manifest["files"]["PRECOMPILED.json"] = sha(TARGET / "PRECOMPILED.json")
    save(TARGET / "RUNTIME.json", manifest)
    verify(TARGET)
    record = {
        "base_runtime_sha256": sha(BASE / "RUNTIME.json"),
        "candidate_runtime_sha256": sha(TARGET / "RUNTIME.json"),
        "changed": [name for name in SOURCES if base["files"][name] != manifest["files"][name]],
        "new_binding": key,
        "added": sorted(set(added)),
        "installed_runtime_changed": False,
        "physical_formulas_changed": False,
        "capacity_state": (
            "sequential along-flow and transverse shared-face pair reconstruction; "
            "dynamic-location affine recovery; actual hydraulics unchanged"
        ),
    }
    save(OUT / "BUILD_2D_COMPACT.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
