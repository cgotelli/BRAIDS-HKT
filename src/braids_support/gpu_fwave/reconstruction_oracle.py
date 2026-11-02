"""Immutable one-stage Fortran oracle for the resident GPU reconstruction port.

This instruments a COPY of the frozen alpha-82 CPU runtime, never production or
the reference. It stops after the first successful hydraulic Euler stage; this
is neither a full simulation nor a performance benchmark. Reconstruction,
boundary characteristics, augmented f-wave and conservative updating remain the
original Fortran calculations. Extra queries/export statements have no numerical
side effects. All snapshot destinations are exclusive and hashes are recorded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import resident_update as resident

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "gpu_reconstruction_20261001_v1"
PARENT = ROOT / "braids_support/sediment/lfm_bend_2d/runs/fast_cpu_default_20261001_v1/runtime_production"
SCHEMA = 13001
SCOPE = "One fixed-grid hydraulic scheme-6 Euler stage; no complete GPU simulation or timing claim"
CASES = ("bend_small", "bend_large", "bend_nonlinear", "bend_small_ssp43", "bend_developed",
         "cutcell_lake", "dry_dam_break", "steep_wave", "hydraulic_jump", "diagonal_wall")

DECLARATIONS = """
  real(8),allocatable :: gpu_face_values(:,:,:)
  integer :: gpu_recon_unit,gpu_recon_side,gpu_recon_cell,gpu_recon_bi
"""

CAPTURE = """
  ! Read-only face_value calls after the original reconstruction has completed.
  if(scheme/=6.or.nstate/=5.or.nprimitive/=5.or.nfrac/=1) &
   error stop 'GPU reconstruction snapshot requires five-state scheme-6 hydraulics'
  gpu_recon_bi=bed_primitive_index()
  allocate(gpu_face_values(4,2,nf));gpu_face_values=0.d0
  do f=1,nf
   do gpu_recon_side=1,2
    gpu_recon_cell=owner(f)
    if(gpu_recon_side==2) gpu_recon_cell=neighbour(f)
    if(gpu_recon_cell<=0) cycle
    left=face_value(gpu_recon_cell,f,v,grad,vlo,vhi)
    gpu_face_values(:,gpu_recon_side,f)=left([1,2,3,gpu_recon_bi])
   enddo
  enddo
"""

WRITE = """
  open(newunit=gpu_recon_unit,file='reconstruction_snapshot.bin',access='stream',form='unformatted',status='new')
  write(gpu_recon_unit) 13001,nc,nf,size(faces),nb,cpu_time_integrator,cpu_outlet_extension,size(boundary_faces)
  write(gpu_recon_unit) dx,dy,speed_limit,evaluation_time,manning_chezy_minimum_depth,manning_friction_chezy_minimum_depth
  write(gpu_recon_unit) xy,mid,hydraulic_ls_weight,hydraulic_ls_matrix,bed_ls_delta,rectangle_width
  write(gpu_recon_unit) weno_cells-1,face_displacement,merge(1,0,wall_triangle)
  write(gpu_recon_unit) wall_stencil_ids-1,wall_stencil_kind,wall_stencil_normal,wall_slot-1
  write(gpu_recon_unit) wall_box_center,wall_normal,wall_mean_distance
  write(gpu_recon_unit) bid-1,bc_hydraulic,forcing,manning_cell
  write(gpu_recon_unit) boundary_offset-1,boundary_faces-1,mode,reversal_policy,nhydro_series,nmorfac
  write(gpu_recon_unit) merge(1,0,production_section_geometry),section_control,section_energy_slope,trace
  write(gpu_recon_unit) v([1,2,3,gpu_recon_bi],:),grad([1,2,3,gpu_recon_bi],:,:)
  write(gpu_recon_unit) vlo([1,2,3,gpu_recon_bi],:),vhi([1,2,3,gpu_recon_bi],:)
  write(gpu_recon_unit) wall_values([1,2,3,gpu_recon_bi],:,:),merge(1,0,wall_ready),gpu_face_values
  close(gpu_recon_unit)
"""


def export_transform(source):
    """Add instrumentation to Euler only, leaving all original formulas intact."""
    source = resident.export_transform(source)
    source = resident.replace_once(source, resident.EXPORT_DECLARATIONS,
                                   resident.EXPORT_DECLARATIONS + DECLARATIONS)
    source = resident.replace_once(source, resident.EXPORT_CAPTURE,
                                   resident.EXPORT_CAPTURE + CAPTURE)
    return resident.replace_once(source,
        "  stop 'GPU hydraulic snapshot complete; this is not a completed simulation'",
        WRITE + "  stop 'GPU hydraulic snapshot complete; this is not a completed simulation'")


def verify_parent():
    identity = json.loads((PARENT / "build.json").read_text())
    for name, expected in identity["source_sha256"].items():
        if resident.sha(PARENT / Path(name).name) != expected:
            raise ValueError(f"Frozen CPU source changed: {name}")
    if resident.sha(PARENT / "xbraids_candidate.exe") != identity["executable_sha256"]:
        raise ValueError("Frozen CPU executable changed")
    return identity


def build(tag=TAG):
    root, report = resident.paths(tag)
    parent = verify_parent()
    root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    runtime = root / "runtime_export"
    runtime.mkdir()
    command = [r"C:\Strawberry\c\bin\gfortran.exe", "-O3", "-flto", "-fopenmp",
               "-ffree-line-length-none", "-ffixed-line-length-none"]
    hashes = {}
    for name in parent["source_sha256"]:
        source, target = PARENT / Path(name).name, runtime / Path(name).name
        if target.name == "cutcell_solver.f90":
            target.write_text(export_transform(source.read_text()))
        else:
            shutil.copy2(source, target)
        hashes[name] = resident.sha(target)
        command.append(str(target))
    exe = runtime / "xbraids_candidate.exe"
    command += ["-o", str(exe)]
    result = subprocess.run(command, cwd=runtime, capture_output=True, text=True, timeout=240)
    (runtime / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    resident.write(runtime / "build.json", dict(command=command, source_sha256=hashes,
        executable_sha256=resident.sha(exe), parent=str(PARENT),
        parent_executable_sha256=parent["executable_sha256"],
        parent_source_sha256=parent["source_sha256"], scope=SCOPE,
        modifications="Euler-only read-only queries and binary export, then STOP; no formula changes",
        production_modified=False, delft_modified=False))
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    shutil.copy2(HERE / "resident_update.py", root / "resident_export_dependency.py")
    resident.write(report / "oracle_build.json", dict(schema=SCHEMA, scope=SCOPE,
        runner_sha256=resident.sha(root / Path(__file__).name),
        dependency_sha256=resident.sha(root / "resident_export_dependency.py"),
        parent_executable_sha256=parent["executable_sha256"],
        export_executable_sha256=resident.sha(exe)))
    print(f"BUILT immutable reconstruction oracle: {runtime}", flush=True)


def _prepare_bend(root, folder, identity, *, large=False, nonlinear=False, developed=False):
    """Exact hydraulic bend geometry with an optional recorded analytic wave."""
    from braids_support.sediment.lfm_bend_2d import hydraulic_performance as h
    r = h.r
    folder.mkdir(exist_ok=False)
    dx = .053125 if large else .10625
    mesh, _ = r.native_mesh(dx)
    xy = np.array([c.centroid_xy for c in mesh.cells])
    s, n = r.e.sn(*xy.T)
    bed = r.e.initial_bed(s)
    angle = np.clip((s-r.e.APPROACH)/r.e.RADIUS, 0, r.e.ANGLE)
    depth = np.full_like(s, r.e.DEPTH)
    along = np.full_like(s, r.e.Q/r.e.WIDTH/r.e.DEPTH)
    across = np.zeros_like(s)
    if nonlinear:
        # A nonstationary analytic state, not experimental/developed model data.
        # Excites nonlinear WENO weights, cross-stream velocity and limiting.
        envelope = np.exp(-((s-(r.e.APPROACH+5.0))/5.0)**2)
        depth += .04*envelope*np.sin(2.3*s+.7*n)
        along += .12*envelope*np.cos(1.7*s-2*n)
        across = .06*envelope*np.sin(.9*s+3*n)
    q = np.zeros((len(s), 5))
    q[:, 0] = depth
    q[:, 1] = depth*(along*np.cos(angle)+across*np.sin(angle))
    q[:, 2] = depth*(-along*np.sin(angle)+across*np.cos(angle))
    extra = {}
    if developed:
        reference = r.HERE/"runs/explicit_multistep_20261001_v2/qualification_small_k6"
        prior = json.loads((reference/"input_provenance.json").read_text())
        for name, expected in prior["files"].items():
            if resident.sha(reference/name) != expected:
                raise ValueError(f"Developed hydraulic reference input changed: {name}")
        geometry = np.load(reference/"analysis_geometry.npz")
        np.testing.assert_array_equal(xy, geometry["xy"])
        np.testing.assert_array_equal(bed, geometry["bed"])
        frame_path = r.shared.native_frames_path(reference)
        frame = r.shared.read_frames(frame_path)[-1]
        if frame["time_s"] != 20. or frame["q"].shape != q.shape or np.any(frame["q"][:,3:]!=0):
            raise ValueError("Developed reference is not the expected 20-second bare hydraulic state")
        q = frame["q"].copy()
        execution = json.loads((reference/"execution.json").read_text())
        extra = dict(developed_reference=str(reference), seed_time_s=frame["time_s"],
            seed_output_sha256=resident.sha(frame_path),
            seed_execution_sha256=resident.sha(reference/"execution.json"),
            seed_runtime_sha256=execution.get("executable_sha256"),
            seed_method="Six-state-history hydraulic SSPMS, constant forcing; clock reset to zero for one-stage comparison")
    boundaries = dict(inlet=r.CutBoundary("total_discharge", discharge_m3_s=r.e.Q),
                      outlet=r.CutBoundary("stage", stage_m=r.e.DEPTH))
    metadata = r.write_cut_case(folder / "case.data", mesh, q, bed, boundaries,
        end_time_s=.0001, dt_max_s=.0001, output_interval_s=.0001,
        cfl=.4, scheme=6, dry_tolerance_m=.0001, bedload=0, exchange=0,
        settling_velocity_m_s=0, manning=r.e.MANNING,
        source_sha256=identity["source_sha256"], compact_sidecars=True)
    mesh.write_geometry(folder / "geometry.npz")
    np.savez_compressed(folder / "analysis_geometry.npz", xy=xy, s=s, n=n, bed=bed,
                        area=np.array([c.area_m2 for c in mesh.cells]))
    resident.write(folder / "preparation.json", metadata)
    return dict(dx_m=dx, analytic_perturbation=nonlinear, **extra,
                perturbation_definition="See frozen reconstruction_oracle.py::_prepare_bend" if nonlinear else None)


def snapshots(tag=TAG, cases=CASES):
    sys.path.insert(0, str(ROOT))
    from braids_support.sediment.lfm_bend_2d import hydraulic_performance as h
    from braids_support.sediment.lfm_bend_2d import implicit_fwave as m
    root, report = resident.paths(tag)
    exe, identity = h.runtime(root, "export")
    rows = []
    for name in cases:
        if name not in CASES:
            raise ValueError(f"Unknown immutable snapshot case: {name}")
        folder = root / name
        if name.startswith("bend"):
            extra = _prepare_bend(root, folder, identity,
                                  large=name=="bend_large", nonlinear=name=="bend_nonlinear",
                                  developed=name=="bend_developed")
        else:
            if name == "diagonal_wall":
                from braids_core.boundaries.domain import PolygonDomain
                from braids_core.boundaries.inlet_io import NativeGrid
                from braids_core.boundaries.cut_cells import map_cut_cells
                domain = PolygonDomain(name, [(0,0),(8,0),(0,8)], "EPSG:2056")
                mesh = map_cut_cells(domain, [], NativeGrid(32,32,0,0,.25,.25))
                xy = np.array([cell.centroid_xy for cell in mesh.cells])
                x, y = xy.T
                bed = .005*x
                q = np.zeros((len(x),5)); q[:,0] = .2+.01*np.sin(x+2*y)
                q[:,1] = q[:,0]*(.3+.03*np.cos(x-y))
                q[:,2] = q[:,0]*(.2+.02*np.sin(2*x+y))
                bc, end = {}, .0001
            else:
                mesh, q, bed, bc, end, *_ = m.fixture(name)
            folder.mkdir(exist_ok=False)
            h.r.write_cut_case(folder / "case.data", mesh, q, bed, bc,
                end_time_s=end, dt_max_s=.0001, output_interval_s=end,
                cfl=.4, scheme=6, bedload=0, exchange=0, settling_velocity_m_s=0,
                dry_tolerance_m=.0001, manning=0, source_sha256=identity["source_sha256"])
            mesh.write_geometry(folder / "geometry.npz")
            extra = dict(fixture_module="reconstruction_oracle.snapshots" if name=="diagonal_wall"
                         else "braids_support.sediment.lfm_bend_2d.implicit_fwave",
                         fixture_module_sha256=resident.sha(Path(__file__) if name=="diagonal_wall" else Path(m.__file__)))
        method = "ssp43" if name.endswith("ssp43") else "auto"
        shutil.copy2(Path(__file__), folder/"oracle_runner.py")
        provenance = dict(case=name, requested_integrator=method, **extra,
            input_files={p.name:resident.sha(p) for p in folder.iterdir() if p.is_file()},
            export_executable_sha256=identity["executable_sha256"],
            parent_executable_sha256=identity["parent_executable_sha256"],
            scope=SCOPE, original_source_formulas_changed=False)
        resident.write(folder / "input_provenance.json", provenance)
        result = subprocess.run([str(exe), "case.data"], cwd=folder,
            env=dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
                     BRAIDS_TIME_INTEGRATOR=method), capture_output=True, text=True, timeout=120)
        (folder / "snapshot.log").write_text(result.stdout+result.stderr)
        result.check_returncode()
        data = read_snapshot(folder)
        np.savez_compressed(folder / "reconstruction_oracle.npz", **data)
        row = dict(case=name, cells=data["q"].shape[1], faces=data["length"].size,
            integrator=data["integrator"], outlet_extension=data["outlet_extension"],
            snapshot_sha256=resident.sha(folder / "gpu_snapshot.bin"),
            reconstruction_snapshot_sha256=resident.sha(folder / "reconstruction_snapshot.bin"),
            arrays_sha256=resident.sha(folder / "reconstruction_oracle.npz"),
            input_provenance_sha256=resident.sha(folder / "input_provenance.json"),
            weno_face_sides=int(np.count_nonzero(data["weno_cells"][0]>=0)),
            ready_wall_slots=int(np.count_nonzero(data["wall_ready"])),
            flattened_cells=int(np.count_nonzero(data["hydro_constant"])),
            threads=2, timed=False, scope=SCOPE)
        resident.write(folder / "snapshot_receipt.json", row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Individual receipts allow independent, append-only snapshot generation.
    if not (report / "snapshots.json").exists():
        resident.write(report / "snapshots.json", rows)
    return rows


def summary(tag=TAG):
    """Consolidate independently generated append-only receipts once complete."""
    root, report = resident.paths(tag)
    rows = []
    for case in CASES:
        folder = root/case
        row = json.loads((folder/"snapshot_receipt.json").read_text())
        for filename, field in (("gpu_snapshot.bin", "snapshot_sha256"),
                                ("reconstruction_snapshot.bin", "reconstruction_snapshot_sha256"),
                                ("reconstruction_oracle.npz", "arrays_sha256"),
                                ("input_provenance.json", "input_provenance_sha256")):
            if resident.sha(folder/filename) != row[field]:
                raise ValueError(f"Immutable oracle changed: {case}/{filename}")
        rows.append(row)
    destination = report/"complete_snapshots.json"
    if destination.exists():
        destination = report/f"complete_snapshots_{len(rows)}cases.json"
    resident.write(destination, dict(scope=SCOPE, cases=rows, all_verified=True,
        production_modified=False, delft_modified=False, timings_performed=False))
    print(f"Verified {len(rows)} immutable reconstruction oracle cases", flush=True)


def read_snapshot(folder):
    """Combine strict original flux oracle and the reconstruction cache export."""
    folder = Path(folder)
    data = resident.read_snapshot(folder / "gpu_snapshot.bin")
    with (folder / "reconstruction_snapshot.bin").open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=8)
        if header.size != 8 or header[0] != SCHEMA:
            raise ValueError("Invalid reconstruction snapshot header/schema")
        _, nc, nf, nadj, nb, integrator, outlet, nboundary = map(int, header)
        if nc != data["q"].shape[1] or nf != data["length"].size or nadj != data["faces"].size or nb<0:
            raise ValueError("Reconstruction/flux snapshot dimensions differ")
        scalars = np.fromfile(stream, dtype="<f8", count=6)
        if scalars.size != 6:
            raise ValueError("Truncated reconstruction scalar header")
        data.update(zip(("dx", "dy", "speed_limit", "evaluation_time", "manning_chezy_minimum_depth",
                         "manning_friction_chezy_minimum_depth"), map(float, scalars)))
        data.update(integrator=integrator, outlet_extension=outlet, schema=SCHEMA)
        def array(name, shape, integer=False):
            shape = (shape,) if isinstance(shape, int) else shape
            count = int(np.prod(shape))
            values = np.fromfile(stream, dtype="<i4" if integer else "<f8", count=count)
            if values.size != count:
                raise ValueError(f"Truncated reconstruction snapshot field {name}")
            data[name] = np.ascontiguousarray(values.reshape(shape, order="F"))
        array("xy", (2,nc)); array("mid", (2,nf))
        array("hydraulic_ls_weight", nadj); array("hydraulic_ls_matrix", (4,nc))
        array("bed_ls_delta", (2,nadj)); array("rectangle_width", (2,nc))
        array("weno_cells", (5,2,nf), True); array("face_displacement", (2,2,nf))
        array("wall_triangle", nc, True); array("wall_stencil_ids", (5,2,nc), True)
        array("wall_stencil_kind", (5,2,nc), True); array("wall_stencil_normal", (2,5,2,nc))
        array("wall_slot", (2,nf), True)
        array("wall_box_center", (2,nc)); array("wall_normal", (2,nc)); array("wall_mean_distance", nc)
        array("bid", nf, True); array("bc_hydraulic", (2,nb)); array("forcing", (6,nb))
        array("manning_cell", nc)
        array("boundary_offset", nb+1, True); array("boundary_faces", nboundary, True)
        array("mode", nb, True); array("reversal_policy", nb, True); array("nhydro_series", nb, True)
        array("nmorfac", 1, True); array("production_section_geometry", 1, True)
        array("section_control", nb, True); array("section_energy_slope", nb); array("boundary_trace", (3,nf))
        array("primitive", (4,nc)); array("grad", (4,2,nc))
        array("vlo", (4,nc)); array("vhi", (4,nc)); array("wall_values", (4,4,nc))
        array("wall_ready", (4,nc), True); array("face_values", (4,2,nf))
        if stream.read(1):
            raise ValueError("Unexpected trailing reconstruction snapshot bytes")
    data["face_boundary_mode"] = data["boundary_mode"].copy()
    validate_snapshot(data)
    return data


def validate_snapshot(data):
    resident.validate_input(data)
    nc, nf = data["q"].shape[1], data["length"].size
    for name in ("weno_cells", "wall_stencil_ids"):
        if np.any((data[name]<-1)|(data[name]>=nc)):
            raise ValueError(f"Invalid normalized reconstruction topology: {name}")
    if not np.isin(data["wall_slot"], [-1,0,1,2,3]).all():
        raise ValueError("Invalid wall slot")
    if data["face_values"].shape != (4,2,nf) or data["grad"].shape != (4,2,nc):
        raise ValueError("Invalid reconstruction oracle shape")
    for value in data.values():
        if isinstance(value, np.ndarray) and value.dtype.kind=="f" and not np.isfinite(value).all():
            raise ValueError("Nonfinite reconstruction snapshot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "snapshots", "summary"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    if args.action == "build":
        build(args.tag)
    elif args.action == "snapshots":
        snapshots(args.tag, args.cases)
    else:
        summary(args.tag)


if __name__ == "__main__":
    main()
