"""Build the additive v43/law-9 mode-2 zero-step GPU initializer.

``native_exporter_v5`` is immutable evidence for the equilibrium fixed-face
contract.  This distinct v6 generation snapshots the rebuilt production core
that also admits public/native bed-boundary mode 2: a prescribed per-fraction
feed with the same immutable hydraulic face bed.  Instrumentation only writes
self-describing records and exits before the first solver timestep.
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

from braids_gpu import build_gaia_initializer as previous
from tools.build_cutcell import SOURCES


DESTINATION = ROOT / "braids_gpu/native_exporter_v6"
CURRENT_BUILD = ROOT / "solver/cut_cells/build/build.json"
FROZEN_V5 = ROOT / "braids_gpu/native_exporter_v5"
FROZEN_V5_BUILD_SHA256 = "24daa64a4ea7fd3a5d6600407f40d4f2ab91ff819c3d9614fab2d8110a64635d"
FROZEN_V5_EXECUTABLE_SHA256 = "970dc63b032a08de59ca18506c9eb804c7c0d3efdd646f698000dbf981018eb5"
TRANSFORM_HELPER_SHA256 = "ecde24cfd5f20e014f704140be8700d8518772938e510c7b1929d012d34dc40e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    if DESTINATION.exists():
        raise FileExistsError("Preserve existing native_exporter_v6: " + str(DESTINATION))
    frozen_executable = FROZEN_V5 / "xbraids_gaia_export.exe"
    if (sha256(FROZEN_V5 / "build.json") != FROZEN_V5_BUILD_SHA256 or
            sha256(frozen_executable) != FROZEN_V5_EXECUTABLE_SHA256):
        raise ValueError("Historical native_exporter_v5 differs; refuse to create v6")
    helper = Path(previous.__file__).resolve()
    if sha256(helper) != TRANSFORM_HELPER_SHA256:
        raise ValueError("Pinned zero-step exporter transform helper differs")

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
            target.write_text(previous.transform(source.read_text(encoding="utf-8")),
                              encoding="utf-8")
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
        subprocess.run(command, cwd=DESTINATION, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    if current_hashes != {name: sha256(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Current native sources changed during v6 initializer build")

    receipt = dict(
        schema=5,
        generation="native_exporter_v6",
        command=command,
        executable=str(output),
        executable_sha256=sha256(output),
        compiler_sha256=sha256(Path(compiler)),
        compiler=subprocess.check_output([compiler, "--version"],
                                         text=True).splitlines()[0],
        driver_sha256=sha256(Path(__file__)),
        transform_helper_sha256=TRANSFORM_HELPER_SHA256,
        production_build_sha256=sha256(CURRENT_BUILD),
        production_executable_sha256=production["executable_sha256"],
        parent_executable_sha256=production["executable_sha256"],
        current_source_sha256=current_hashes,
        snapshot_source_sha256={
            path.name: sha256(path) for path in DESTINATION.iterdir()
            if path.suffix.lower() in (".f", ".f90", ".f95")},
        source_sha256={
            path.name: sha256(path) for path in DESTINATION.iterdir()
            if path.suffix.lower() in (".f", ".f90", ".f95")},
        numerical_source_unchanged=True,
        numerical_source_unchanged_by_instrumentation=True,
        instrumentation_scope=(
            "self-describing initial-state and forcing records, then "
            "unconditional zero-step exit"),
        initializer_scope_extension=(
            "v43 TELEMAC-GAIA law9 public bed-boundary mode 2: prescribed "
            "per-fraction feed with immutable hydraulic face bed"),
        bed_boundary_modes=[0, 1, 2],
        mode_two_hydraulic_trace="fixed_initial_face_bed",
        mode_two_sediment_flux="prescribed_per_fraction_feed",
        native_solver_steps_performed=0,
        frozen_v3_modified=False,
        frozen_v4_modified=False,
        frozen_v5_modified=False,
        frozen_v5_build_sha256=FROZEN_V5_BUILD_SHA256,
        frozen_v5_executable_sha256=FROZEN_V5_EXECUTABLE_SHA256,
        frozen_qualification_relabelled=False,
        Delft3D_source_modified=False,
    )
    (DESTINATION / "build.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(executable=str(output),
                          sha256=receipt["executable_sha256"])))
    return receipt


if __name__ == "__main__":
    build()
