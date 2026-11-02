"""Short unmodified alpha-82 SSP43 trajectories for resident GPU qualification.

Only validation input controls (end time, maximum step, output interval and input
identity) change. Mesh, physical state, boundaries and native source/executable
are frozen and verified. These are correctness runs while an unrelated CPU job
is active: their wall times MUST NOT be used as performance benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import reconstruction_oracle as oracle
from . import resident_update as resident

TAG = "gpu_hydraulic_20261001_v1"
CASES = {"bend_small":.0005, "cutcell_lake":.0001, "dry_dam_break":.0001}
EXTENDED_CONTROLS = {
    "bend_small":dict(end=2., dt=.005, output=.2),
    "cutcell_lake":dict(end=.5, dt=.001, output=.05),
    "dry_dam_break":dict(end=.5, dt=.001, output=.05),
}


def transform_deck(text, *, end, dt, output):
    """Edit only the documented version-specific scalar-control line."""
    lines = text.splitlines()
    version = int(lines[0])
    position = 6 if version==24 else 4 if version==1 else None
    if position is None or len(lines[position].split()) != 3:
        raise ValueError("Unsupported frozen input deck/control-line layout")
    old = list(map(float, lines[position].split()))
    replacement = f"{end:.17g} {dt:.17g} {output:.17g}"
    previous_fingerprint = lines[1]
    identity = hashlib.sha256((previous_fingerprint+"\n"+replacement+"\nSSP43 GPU validation").encode()).hexdigest()
    lines[1] = identity
    lines[position] = replacement
    return "\n".join(lines)+"\n", dict(
        time_control_line_one_based=position+1,
        previous=dict(end_s=old[0], dt_max_s=old[1], output_s=old[2]),
        current=dict(end_s=end, dt_max_s=dt, output_s=output),
        previous_fingerprint=previous_fingerprint, new_fingerprint=identity,
        physical_input_changed=False)


def run_case(case, tag=TAG, *, end=.05, dt=None, output=.01):
    if case not in CASES:
        raise ValueError("Unknown short hydraulic validation case")
    dt = CASES[case] if dt is None else dt
    if not 0 < dt <= output <= end or abs(round(end/output)*output-end)>1.e-12:
        raise ValueError("Invalid or noncommensurate validation time controls")
    from braids_support.sediment.lfm_bend_2d import run_case as r
    identity = oracle.verify_parent()
    root, report = resident.paths(tag)
    root.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    source = oracle.HERE/"runs"/oracle.TAG/case
    original = json.loads((source/"input_provenance.json").read_text())
    for name, expected in original["input_files"].items():
        if resident.sha(source/name) != expected:
            raise ValueError(f"Immutable input changed: {case}/{name}")
    target = root/("cpu_"+case)
    target.mkdir(exist_ok=False)
    for name in ("case.data.bin", "geometry.npz", "analysis_geometry.npz"):
        if (source/name).exists():
            shutil.copy2(source/name, target/name)
    deck, changes = transform_deck((source/"case.data").read_text(),
                                   end=end, dt=dt, output=output)
    (target/"case.data").write_text(deck)
    shutil.copy2(Path(__file__), target/"hydraulic_reference.py")
    provenance = dict(case=case, original_input=str(source),
        original_input_provenance_sha256=resident.sha(source/"input_provenance.json"),
        source_input_sha256=resident.sha(source/"case.data"), changes=changes,
        input_files={p.name:resident.sha(p) for p in target.iterdir() if p.is_file()},
        integrator="ssp43", expected_outlet_extension=0, threads=8,
        runtime_path=str(oracle.PARENT), executable_sha256=identity["executable_sha256"],
        source_sha256=identity["source_sha256"], frozen_source_modified=False,
        delft_modified=False, timing_claim=False, concurrent_CPU_reference_job=True)
    resident.write(target/"input_provenance.json", provenance)
    exe = oracle.PARENT/"xbraids_candidate.exe"
    result = subprocess.run([str(exe), "case.data"], cwd=target,
        env=dict(os.environ, OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
                 BRAIDS_TIME_INTEGRATOR="ssp43", BRAIDS_PROFILE="1"),
        capture_output=True, text=True, timeout=60)
    (target/"native.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    frames_path = r.shared.native_frames_path(target)
    frames = r.shared.read_frames(frames_path)
    times = np.array([f["time_s"] for f in frames])
    states = np.stack([f["q"][:,:3].T for f in frames])
    ledger = np.stack([f["ledger"] for f in frames])
    steps = np.array([f["steps"] for f in frames], dtype=np.int64)
    np.testing.assert_allclose(times, np.arange(round(end/output)+1)*output, rtol=0, atol=2.e-13)
    if not np.isfinite(states).all() or np.min(states[:,0])<0:
        raise ValueError("Invalid native reference trajectory")
    initial = oracle.read_snapshot(source)
    np.testing.assert_array_equal(states[0], initial["q"])
    water = np.sum(states[:,0]*initial["area"], axis=1)
    water_ledger = np.sum(ledger[:,:,0], axis=1)
    residual = water-water[0]-(water_ledger-water_ledger[0])
    np.savez_compressed(target/"cpu_trajectory.npz", times=times, q=states, ledger=ledger,
                        steps=steps, dt=np.float64(dt), area=initial["area"],
                        water_residual=residual, outlet_extension=np.int32(0), integrator=np.int32(43))
    receipt = dict(case=case, final_time_s=float(times[-1]), steps=int(steps[-1]),
        dt_max_s=dt, frame_times_s=times.tolist(), frame_steps=steps.tolist(),
        q_layout="time,hydraulic_component,cell", ledger_layout="time,boundary,ledger_component",
        frames_sha256=resident.sha(frames_path),
        arrays_sha256=resident.sha(target/"cpu_trajectory.npz"),
        executable_sha256=identity["executable_sha256"],
        input_provenance_sha256=resident.sha(target/"input_provenance.json"),
        water_residual_max_m3=float(np.max(np.abs(residual))),
        min_depth_m=float(np.min(states[:,0])), max_state_change=float(np.max(np.abs(states[-1]-states[0]))),
        performance_measurement=False, source_or_production_modified=False)
    resident.write(target/"trajectory_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--profile", choices=("short", "extended"), default="short")
    args = parser.parse_args()
    for case in args.cases:
        run_case(case, args.tag, **(EXTENDED_CONTROLS[case] if args.profile=="extended" else {}))


if __name__ == "__main__":
    main()
