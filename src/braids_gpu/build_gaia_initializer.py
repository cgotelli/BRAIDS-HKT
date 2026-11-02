"""Build the distinct current-source v43/law-9 zero-step GPU initializer.

The authenticated v42/law-8 ``native_exporter_v3`` and the historical
pre-coupled-face v43/law-9 ``native_exporter_v4`` remain frozen.  This v5
generation snapshots the current native sources and adds record writers plus a
zero-step exit only; it does not execute or alter a numerical solver step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_cutcell import SOURCES
from braids_support.gpu_fwave import coupled_reference as reference


DESTINATION = ROOT / "braids_gpu/native_exporter_v5"
CURRENT_BUILD = ROOT / "solver/cut_cells/build/build.json"
FROZEN_V4 = ROOT / "braids_gpu/native_exporter_v4"
FROZEN_V4_BUILD_SHA256 = "d8d098c32cc74a89bcd93534d166ff721bd194bda957157a995158ac6b2abcc0"
FROZEN_V4_EXECUTABLE_SHA256 = "90eeb6a5740ba0fa7aee3613855380e6185724209a636d4603ff77a5db7c203f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Current v43 initializer source anchor differs:\n" + old)
    return source.replace(old, new, 1)


def normal_metadata_source() -> str:
    text = """
 subroutine normal_export_metadata()
  integer :: gpu_unit
  call gpu_open(gpu_unit,'normal_metadata.bin')
"""
    for name in ("nb", "nfrac", "deck_version"):
        text += reference.rec(name, kind="i")
    for name, dims in (
        ("nhydro_series", ("nb",)),
        ("nseries", ("nb",)),
        ("mode", ("nb",)),
        ("section_control", ("nb",)),
        ("nfraction_series", ("nfrac", "nb")),
        ("native_section_seed_node", ("nb",)),
        ("native_boundary_seed_position", ("nb",)),
    ):
        expression = name
        if name == "native_section_seed_node":
            expression = "section_seed_node"
        elif name == "native_boundary_seed_position":
            expression = "boundary_seed_position"
        text += reference.rec(name, expression, dims, "i")
    for name, dims in (
        ("section_energy_slope", ("nb",)),
        ("hydro_series_time", ("size(hydro_series_time,1)", "nb")),
        ("hydro_series_q", ("size(hydro_series_q,1)", "nb")),
        ("hydro_series_stage", ("size(hydro_series_stage,1)", "nb")),
        ("series_time", ("size(series_time,1)", "nb")),
        ("series_concentration", ("size(series_concentration,1)", "nb")),
        ("series_bed_input", ("size(series_bed_input,1)", "nb")),
        ("fraction_series_time", ("size(fraction_series_time,1)", "nfrac", "nb")),
        ("fraction_series_concentration", ("size(fraction_series_concentration,1)", "nfrac", "nb")),
        ("fraction_series_bed_input", ("size(fraction_series_bed_input,1)", "nfrac", "nb")),
    ):
        text += reference.rec(name, dims=dims)
    return text + "  call gpu_finish(gpu_unit)\n end subroutine\n"


def transform(source: str) -> str:
    transformed = reference.instrument(source)
    transformed = _replace_once(
        transformed,
        "  call gpu_export_input()\n  stop 'GPU native initialization exported; no solver timestep performed'",
        "  call gpu_export_input()\n  call normal_export_metadata()\n"
        "  stop 'GPU native initialization exported; no solver timestep performed'",
    )
    transformed = _replace_once(
        transformed,
        "end module\n\nprogram run_cutcell",
        normal_metadata_source() + "end module\n\nprogram run_cutcell",
    )
    return transformed


def build() -> dict:
    if DESTINATION.exists():
        raise FileExistsError("Preserve existing native_exporter_v5: " + str(DESTINATION))
    frozen_v4_executable = FROZEN_V4 / "xbraids_gaia_export.exe"
    if (sha256(FROZEN_V4 / "build.json") != FROZEN_V4_BUILD_SHA256 or
            sha256(frozen_v4_executable) != FROZEN_V4_EXECUTABLE_SHA256):
        raise ValueError("Historical native_exporter_v4 differs; refuse to create v5")
    production = json.loads(CURRENT_BUILD.read_text(encoding="utf-8"))
    current_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    if production.get("source_sha256") != current_hashes:
        raise ValueError("Rebuild the production native runtime from current sources first")
    executable = ROOT / "solver/cut_cells/build/xbraids_cutcell.exe"
    if sha256(executable) != production.get("executable_sha256"):
        raise ValueError("Current production executable differs from its receipt")
    DESTINATION.mkdir(parents=True)
    for name in SOURCES:
        source = ROOT / name
        target = DESTINATION / source.name
        if source.name == "cutcell_solver.f90":
            target.write_text(transform(source.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(source, target)

    compiler = shutil.which(production["command"][0])
    if compiler is None:
        raise FileNotFoundError("Current native compiler is unavailable")
    output = DESTINATION / "xbraids_gaia_export.exe"
    command = [
        compiler, "-O3", "-flto", "-fopenmp",
        "-ffree-line-length-none", "-ffixed-line-length-none",
        *[str(DESTINATION / Path(name).name) for name in SOURCES],
        "-o", str(output),
    ]
    with (DESTINATION / "build.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=DESTINATION, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    if current_hashes != {name: sha256(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Current native sources changed during v43 initializer build")
    receipt = dict(
        schema=4,
        generation="native_exporter_v5",
        command=command,
        executable=str(output),
        executable_sha256=sha256(output),
        compiler_sha256=sha256(Path(compiler)),
        compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        driver_sha256=sha256(Path(__file__)),
        production_build_sha256=sha256(CURRENT_BUILD),
        production_executable_sha256=production["executable_sha256"],
        parent_executable_sha256=production["executable_sha256"],
        current_source_sha256=current_hashes,
        snapshot_source_sha256={path.name: sha256(path) for path in DESTINATION.iterdir()
                                if path.suffix.lower() in (".f", ".f90", ".f95")},
        source_sha256={path.name: sha256(path) for path in DESTINATION.iterdir()
                       if path.suffix.lower() in (".f", ".f90", ".f95")},
        numerical_source_unchanged=True,
        numerical_source_unchanged_by_instrumentation=True,
        instrumentation_scope="self-describing initial-state and forcing records, then unconditional zero-step exit",
        initializer_scope_extension="v43 TELEMAC-GAIA law9, slope4, secondary model3",
        native_solver_steps_performed=0,
        frozen_v3_modified=False,
        frozen_v4_modified=False,
        frozen_v4_build_sha256=FROZEN_V4_BUILD_SHA256,
        frozen_v4_executable_sha256=FROZEN_V4_EXECUTABLE_SHA256,
        frozen_qualification_relabelled=False,
        Delft3D_source_modified=False,
    )
    (DESTINATION / "build.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(dict(executable=str(output), sha256=receipt["executable_sha256"])))
    return receipt


if __name__ == "__main__":
    build()
