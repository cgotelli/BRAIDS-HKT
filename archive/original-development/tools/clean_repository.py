#!/usr/bin/env python3
"""Remove generated and reproducible bulk data from a BRAIDS checkout."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRECTORY_NAMES = {
    ".pytest_cache", ".venv", "__pycache__", "_output", "build", "history",
    "builds", "integration_runs", "output", "outputs", "prepared", "profile_runs",
    "qgis_results", "results", "run", "runs", "section_runs", "segments",
    "timing_probes", "tmp",
}
EXTERNAL_TREES = {
    "resources/Delft3D-main",
    "resources/GeoFlood-master",
    "resources/Clawpack_tutorial",
    "vendor",
    "validation/vendor",
}
GENERATED_CASE_TREES = {
    "solver/clawpack-v5.14.0/classic",
    "solver/clawpack-v5.14.0/conda-recipe",
    "solver/clawpack-v5.14.0/geoclaw/examples",
    "solver/clawpack-v5.14.0/geoclaw/scratch",
    "solver/clawpack-v5.14.0/geoclaw/tests",
    "solver/clawpack-v5.14.0/pyclaw",
    "solver/clawpack-v5.14.0/visclaw",
    "validation/amr/bounded_reservoir_deposition/amr_selectivity",
    "validation/amr/bounded_reservoir_deposition/continuation_600",
    "validation/amr/bounded_reservoir_deposition/corrected_amr",
    "validation/amr/bounded_reservoir_deposition/project",
    "validation/amr/reservoir_window/allwet_build",
    "validation/amr/reservoir_window/first_30s",
    "validation/amr/reservoir_window/production",
    "validation/sediment/reservoir/roundoff_conservative_integration",
    "validation/sediment/reservoir/roundoff_integration",
    "validation/sediment/reservoir/roundoff_integration_final",
    "validation/sediment/reservoir/roundoff_integration_qualified",
    "validation/sediment/reservoir/roundoff_stage_integration",
    "validation/sediment/reservoir/roundoff_terrain_integration",
    "validation/sediment/reservoir_polygon/bed_slope_comparison",
    "validation/sediment/reservoir_polygon/dry_source_proof",
    "validation/sediment/reservoir_polygon/dry_source_regression",
    "validation/sediment/reservoir_polygon/grid_comparison",
    "validation/sediment/reservoir_polygon/peak_limiter_application",
    "validation/sediment/reservoir_polygon/ready",
    "validation/sediment/reservoir_polygon/startup_fix",
}
GENERATED_FILE_SUFFIXES = {".dylib", ".log", ".mod", ".o", ".pyc", ".pyo", ".so"}
GENERATED_FILE_NAMES = {".DS_Store", "native.log", "pyclaw.log", "xgeoclaw",
                        "xbraids_cutcell"}
KEEP_ARCHIVE = "BRAIDS-0.1.0-alpha.37-QGIS-macos-arm64.zip"
PRESERVED_VALIDATION_DIRECTORY_NAMES = {
    "docs", "fixtures", "reports", "source", "sources", "tests",
}
PRESERVED_VALIDATION_SUFFIXES = {
    ".c", ".f", ".f90", ".h", ".ipynb", ".md", ".png", ".py", ".sh",
    ".svg",
}
PRESERVED_VALIDATION_TREES = {
    "validation/hydraulics/sections/generated",
    "validation/sediment/camnasio2013_cc/measurements",
}


def bytes_in(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def candidates() -> list[Path]:
    selected: set[Path] = set()
    for relative in EXTERNAL_TREES | GENERATED_CASE_TREES:
        path = ROOT / relative
        if path.exists():
            selected.add(path)
    for path in ROOT.rglob("*"):
        if any(parent in selected for parent in path.parents):
            continue
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in GENERATED_DIRECTORY_NAMES:
            selected.add(path)
        elif path.is_file() and (
            path.name in GENERATED_FILE_NAMES or path.suffix.lower() in GENERATED_FILE_SUFFIXES
        ):
            selected.add(path)
        elif (path.is_file() and relative.parts[:1] == ("validation",)
              and path.suffix.lower() not in {".pdf", ".png"}
              and path.stat().st_size > 2 * 1024**2):
            selected.add(path)
        elif path.is_file() and relative.parts[:1] == ("dist",) and path.name != KEEP_ARCHIVE:
            selected.add(path)
    # Numerical case trees frequently use unique run names rather than a common
    # `runs` directory. Remove bottom-level data-only directories while keeping
    # scripts, native probes, documentation, final figures and test fixtures.
    validation = ROOT / "validation"
    directories = sorted(
        (path for path in validation.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True)
    for path in directories:
        relative = path.relative_to(ROOT).as_posix()
        if any(relative == tree or relative.startswith(tree + "/")
               for tree in PRESERVED_VALIDATION_TREES):
            continue
        if path.name in PRESERVED_VALIDATION_DIRECTORY_NAMES:
            continue
        if any(parent in selected for parent in path.parents) or path in selected:
            continue
        retained_files = [item for item in path.iterdir()
                          if item.is_file() and item not in selected]
        retained_directories = [item for item in path.iterdir()
                                if item.is_dir() and item not in selected]
        has_source_or_report = any(
            item.suffix.lower() in PRESERVED_VALIDATION_SUFFIXES
            or item.name.lower().startswith("makefile")
            for item in retained_files)
        if not has_source_or_report and not retained_directories:
            selected.add(path)
    ordered = sorted(selected, key=lambda item: (len(item.parts), str(item)))
    roots: list[Path] = []
    for path in ordered:
        if not any(parent in roots for parent in path.parents):
            roots.append(path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform deletion")
    args = parser.parse_args()
    paths = candidates()
    records = [(path.relative_to(ROOT).as_posix(), bytes_in(path), path.is_dir())
               for path in paths]
    total = sum(size for _, size, _ in records)
    print(json.dumps({"items": len(records), "bytes": total,
                      "gib": round(total / 1024**3, 3)}, indent=2))
    for name, size, is_directory in sorted(records, key=lambda row: row[1], reverse=True)[:30]:
        print(f"{size / 1024**2:10.1f} MiB  {'dir ' if is_directory else 'file'}  {name}")
    if not args.apply:
        print("Dry run only; pass --apply to remove these paths.")
        return
    counts = Counter("directories" if is_directory else "files" for _, _, is_directory in records)
    for name, _, _ in records:
        path = ROOT / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    print(f"Removed {counts['directories']} directories and {counts['files']} files.")


if __name__ == "__main__":
    main()
