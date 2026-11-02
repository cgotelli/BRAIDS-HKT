#!/usr/bin/env python3
"""Build the current Fortran/OpenMP development runtime used by the QGIS dock."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "solver/cut_cells/build"
SOURCES = (
    "solver/cut_cells/geoclaw_parameters.f90",
    "solver/wave/braids_hydraulic_roughness_module.f90",
    "solver/wave/braids_soulsby_van_rijn_module.f90",
    "solver/wave/braids_delft_capacity_module.f90",
    "solver/wave/braids_scalar_capacity_runtime_module.f90",
    "solver/wave/braids_delft_channel_composer_module.f90",
    "solver/wave/braids_sediment_shear_module.f90",
    "solver/wave/braids_exner_mol_module.f90",
    "solver/wave/braids_van_rijn_1993_module.f90",
    "solver/wave/braids_bedload_vr1993_module.f90",
    "solver/wave/braids_bedload_mpm_module.f90",
    "solver/wave/braids_bedform_roughness_module.f90",
    "solver/wave/braids_bedload_current_module.f90",
    "solver/wave/braids_secondary_flow_module.f90",
    "solver/wave/braids_flow2dh_secondary_operator_module.f90",
    "solver/wave/braids_flow2dh_prognostic_secondary_module.f90",
    "solver/wave/braids_curvature_regularization_module.f90",
    "solver/wave/braids_cohesive_exchange_module.f90",
    "solver/wave/braids_hiding_exposure_module.f90",
    "solver/wave/braids_bed_slope_module.f90",
    "solver/wave/braids_morphodynamic_flux_module.f90",
    "solver/wave/braids_bank_toe_geometry_module.f90",
    "solver/wave/braids_bank_toe_supply_module.f90",
    "solver/wave/braids_bank_toe_flux_module.f90",
    "solver/wave/braids_distributed_bank_module.f90",
    "solver/wave/braids_bank_front_module.f90",
    "solver/wave/braids_accepted_flux_reconstruction_module.f90",
    "solver/wave/braids_bank_reconstruction_module.f90",
    "solver/cut_cells/braids_viscous_inlet_module.f90",
    "solver/clawpack-v5.14.0/riemann/src/rpn2_geoclaw.f",
    "solver/clawpack-v5.14.0/riemann/src/geoclaw_riemann_utils.f",
    "solver/cut_cells/cutcell_solver.f90",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(path):
    """Hash source content independent of Windows/POSIX line endings."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("FC", "gfortran")
    # The QGIS runtime resolves a conventional ``.exe`` name on Windows.
    # Supplying the extension here also prevents MinGW from emitting an
    # extensionless PE file that QProcess cannot locate by that runtime name.
    executable = BUILD / ("xbraids_cutcell.exe" if os.name == "nt" else "xbraids_cutcell")
    temporary = BUILD / ("xbraids_cutcell.building.exe" if os.name == "nt" else "xbraids_cutcell.building")
    # Runtime bounds/undefined-state checks are valuable for a debugging run,
    # but impose work in every numerical kernel.  Ship the ordinary optimised
    # OpenMP executable; developers and CI can request the checked equivalent
    # with BRAIDS_NATIVE_CHECKS=1 without changing any numerical options.
    checks_enabled = os.environ.get("BRAIDS_NATIVE_CHECKS") == "1"
    # Whole-program optimisation lets gfortran inline the many small pure
    # closure helpers used inside the face loop. It does not select a host CPU
    # ISA (the packaged executable remains portable across supported PCs) and
    # does not enable unsafe floating-point reassociation.
    command = [compiler, "-O3", "-flto", "-fopenmp",
               "-ffree-line-length-none", "-ffixed-line-length-none",
               *[str(ROOT / name) for name in SOURCES], "-o", str(temporary)]
    if checks_enabled:
        command.insert(3, "-fcheck=all")
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    text_hashes = {name: text_digest(ROOT / name) for name in SOURCES}
    started = time.perf_counter()
    result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True, timeout=60)
    (BUILD / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    if hashes != {name: digest(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Native sources changed during compilation; rebuild")
    if text_hashes != {name: text_digest(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Native source content changed during compilation; rebuild")
    temporary.replace(executable)
    if os.name == "nt":
        # Several headless validation callers use the POSIX-style basename.
        # Keep that compatibility copy synchronized with the QGIS .exe so its
        # build manifest can never describe a different executable.
        shutil.copy2(executable, BUILD / "xbraids_cutcell")
    record = {"command": command, "source_sha256": hashes,
              "source_text_sha256": text_hashes,
              "executable_sha256": digest(executable),
              "elapsed_wall_s": time.perf_counter() - started,
              "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
              "scope": "Optimised OpenMP runtime built from recorded sources",
              "runtime_checks": checks_enabled,
              "supported_case_envelopes": list(range(1, 58)),
              "prognostic_secondary_v57": True,
              "prognostic_secondary_state_units": "m2_s",
              "prognostic_secondary_integrated_ledger_units": "m4_s",
              "prognostic_secondary_bed_boundary_rotation_policy":
                  "roles12_trace_owner_equilibrium_role3_rotated_owner_v1"}
    (BUILD / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built {executable} in {record['elapsed_wall_s']:.2f} s")


if __name__ == "__main__":
    main()
