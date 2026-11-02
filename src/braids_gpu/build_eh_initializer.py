"""Build the authenticated law-8 zero-step GPU initializer generation.

``native_exporter_v2`` is a frozen braids_support/runtime artifact.  This builder
copies that complete source closure into a new generation and applies only the
input-admission and record-export changes needed by the constant-Chezy
Engelund--Hansen / Koch--Flokstra GPU route.  The program still exits directly
after writing the initial records: no numerical solver step is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .atomic_output import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "braids_gpu/native_exporter_v2"
DESTINATION = ROOT / "braids_gpu/native_exporter_v3"
SHA = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Law-8 initializer source anchor differs:\n" + old)
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    """Apply a narrow, auditable zero-step admission/export extension."""
    source = _replace_once(
        source,
        " real(8) :: critical_mud_fraction=.5d0,transport_calibration=1.d0\n",
        " real(8) :: critical_mud_fraction=.5d0,transport_calibration=1.d0\n"
        " real(8) :: engelund_hansen_chezy=0.d0\n",
    )
    source = _replace_once(
        source,
        "  if(version<1.or.version>41) error stop 'unsupported cut-cell deck version'\n",
        "  if(version<1.or.version>42) error stop 'unsupported cut-cell deck version'\n",
    )
    source = _replace_once(
        source,
        "  nviscous_inlets=0\n  if(version==41) then\n",
        "  nviscous_inlets=0\n"
        "  if(version==42) then\n"
        "   read(unit,*) engelund_hansen_chezy\n"
        "   if(.not.ieee_is_finite(engelund_hansen_chezy).or.engelund_hansen_chezy<=0.d0) &\n"
        "    error stop 'invalid constant Engelund-Hansen Chezy coefficient'\n"
        "   read(unit,*) version\n"
        "   if(version<1.or.version>41) error stop 'unsupported Engelund-Hansen base deck version'\n"
        "  endif\n"
        "  if(version==41) then\n",
    )
    source = _replace_once(
        source,
        "  if((scheme/=1.and.scheme/=5.and.scheme/=6).or.bedlaw<0.or.bedlaw>7.or.exchange_mode<0.or.exchange_mode>3) &\n",
        "  if((scheme/=1.and.scheme/=5.and.scheme/=6).or.bedlaw<0.or.bedlaw>8.or.exchange_mode<0.or.exchange_mode>3) &\n",
    )
    source = _replace_once(
        source,
        "  if(version<16.and.(bedlaw>2.or.exchange_mode>2)) error stop 'extended sediment laws require deck version 16'\n",
        "  if(version<16.and.((bedlaw>2.and.bedlaw/=8).or.exchange_mode>2)) &\n"
        "   error stop 'extended sediment laws require deck version 16'\n",
    )
    source = _replace_once(
        source,
        "  if((bed_slope/=0.and.bed_slope/=2).or. &\n"
        "   .not.all(ieee_is_finite([bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg])).or. &\n"
        "   min(bed_slope_alfabs,bed_slope_alfabn)<0.or.bed_friction_angle_deg<=0.or.bed_friction_angle_deg>=90) &\n"
        "   error stop 'invalid bed-slope parameters'\n",
        "  if((bed_slope/=0.and.bed_slope/=2.and.bed_slope/=3).or. &\n"
        "   .not.all(ieee_is_finite([bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg])).or. &\n"
        "   min(bed_slope_alfabs,bed_slope_alfabn,bed_friction_angle_deg)<0.or. &\n"
        "   (bed_slope==2.and.(bed_friction_angle_deg<=0.or.bed_friction_angle_deg>=90))) &\n"
        "   error stop 'invalid bed-slope parameters'\n",
    )
    source = _replace_once(
        source,
        "  if(bed_slope/=0.and.bedlaw==0) error stop 'bed-slope correction requires an active bed-load formulation'\n"
        "  if(secondary_flow_model==1.and.all(bedlaw/=[2,3,5,6,7])) &\n",
        "  if(bed_slope/=0.and.bedlaw==0) error stop 'bed-slope correction requires an active bed-load formulation'\n"
        "  if(bed_slope==3.and.bedlaw/=8) &\n"
        "   error stop 'Koch-Flokstra slope mode requires Engelund-Hansen bed load'\n"
        "  if(bedlaw==8.and.(envelope_version/=42.or.engelund_hansen_chezy<=0.d0.or. &\n"
        "     exchange_mode/=0.or.mpm_friction_mode/=0.or.hiding_exposure/=0.or. &\n"
        "     bedload_adaptation.or.bedform_roughness.or.transport_calibration/=1.d0.or. &\n"
        "     (bed_slope/=0.and.bed_slope/=3))) &\n"
        "   error stop 'GPU Engelund-Hansen requires v42 constant Chezy and the qualified bedload-only scope'\n"
        "  if(bed_slope==3.and.(bed_slope_alfabs/=0.d0.or.bed_slope_alfabn<=0.d0.or. &\n"
        "     bed_friction_angle_deg<=0.d0)) &\n"
        "   error stop 'GPU Koch-Flokstra requires AlfaBs=0, AShld>0 and BShld>0'\n"
        "  if(secondary_flow_model==1.and.all(bedlaw/=[2,3,5,6,7,8])) &\n",
    )
    source = _replace_once(
        source,
        "     all(bedlaw/=[1,2,3,5,6,7])) &\n",
        "     all(bedlaw/=[1,2,3,5,6,7,8])) &\n",
    )
    source = _replace_once(
        source,
        "  if(secondary_flow.and.any(manning_cell<=0)) error stop 'secondary flow requires positive Manning roughness in every cell'\n",
        "  if(secondary_flow.and.any(manning_cell<=0)) error stop 'secondary flow requires positive Manning roughness in every cell'\n"
        "  if(bedlaw==8.and.any(manning_cell<=0)) &\n"
        "   error stop 'Engelund-Hansen hydraulics require positive Manning roughness in every cell'\n",
    )
    source = _replace_once(
        source,
        "  call gpu_put_r(gpu_unit,'transport_calibration',[transport_calibration],[integer::])\n",
        "  call gpu_put_r(gpu_unit,'transport_calibration',[transport_calibration],[integer::])\n"
        "  call gpu_put_r(gpu_unit,'engelund_hansen_chezy',[engelund_hansen_chezy],[integer::])\n",
    )
    return source


def build() -> dict:
    parent = json.loads((PARENT / "build.json").read_text(encoding="utf-8"))
    parent_exe = PARENT / Path(parent["executable"]).name
    if SHA(parent_exe) != parent["executable_sha256"]:
        raise ValueError("Frozen v2 initializer executable differs")
    compiler = Path(parent["command"][0])
    if SHA(compiler) != parent["compiler_sha256"]:
        raise ValueError("Pinned initializer compiler differs")
    sources = [PARENT / Path(part).name for part in parent["command"]
               if Path(part).suffix.lower() in (".f", ".f90", ".f95")]
    expected = parent.get("source_sha256", {})
    if not sources or {p.name for p in sources} != set(expected):
        raise ValueError("Frozen v2 initializer source inventory differs")
    for path in sources:
        if SHA(path) != expected[path.name]:
            raise ValueError("Frozen v2 initializer source differs: " + path.name)

    DESTINATION.mkdir(parents=True, exist_ok=False)
    for path in sources:
        target = DESTINATION / path.name
        if path.name == "cutcell_solver.f90":
            target.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(path, target)

    executable = DESTINATION / "xbraids_eh_export.exe"
    command = [str(compiler), *parent["command"][1:]]
    command = [str(DESTINATION / Path(part).name)
               if Path(part).suffix.lower() in (".f", ".f90", ".f95") else part
               for part in command]
    output_index = command.index("-o") + 1
    command[output_index] = str(executable)
    with (DESTINATION / "build.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=DESTINATION, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    for path in sources:
        if SHA(path) != expected[path.name]:
            raise ValueError("Frozen v2 source changed while building: " + path.name)

    receipt = dict(
        schema=3,
        command=command,
        executable=str(executable),
        executable_sha256=SHA(executable),
        parent_executable_sha256=parent["executable_sha256"],
        parent_build_sha256=SHA(PARENT / "build.json"),
        parent_source_sha256=dict(sorted(expected.items())),
        source_sha256={path.name: SHA(DESTINATION / path.name) for path in sources},
        compiler_sha256=parent["compiler_sha256"],
        driver_sha256=SHA(__file__),
        numerical_source_unchanged=True,
        initializer_scope_extension="v42 constant-Chezy Engelund-Hansen law 8; Koch-Flokstra mode 3",
        native_solver_steps_performed=0,
        Delft3D_source_modified=False,
    )
    write_json_atomic(DESTINATION / "build.json", receipt)
    print(json.dumps({"executable": str(executable), "sha256": receipt["executable_sha256"]}))
    return receipt


if __name__ == "__main__":
    build()
