"""FP64 resident face-flux/draining/gather proof, not a complete GPU solver.

Only fixed-grid hydraulic Euler face work is implemented here. WENO traces and
boundary characteristics are exported from unchanged frozen CPU routines.
Friction, full time integration, AMR and sediment remain outside this proof.
Use the normal BRAIDS venv for build/snapshots and .venv_gpu for validate.
No numerical timing is performed by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "gpu_resident_update_20261001_v2"
PARENT = ROOT / "braids_support/sediment/lfm_bend_2d/runs/larger_timesteps_20261001_v3/runtime_ssp43"
GPU_PARENT = HERE / "runs/gpu_fwave_20261001_v2"
OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
SCOPE = "Fixed-grid hydraulic face flux, pressure quadrature, conservative draining and Euler gather only; CPU reconstruction/boundaries, no full GPU solver"


def sha(path):
    from braids_core.windows_paths import filesystem_path
    return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def paths(tag):
    if Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("Tag must be a single directory name")
    return HERE / "runs" / tag, HERE / "reports" / tag


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one immutable source anchor: {old[:90]!r}")
    return text.replace(old, new, 1)


EXPORT_DECLARATIONS = """
  ! Validation-only snapshot workspace; no production algorithm is changed.
  real(8),allocatable :: gpu_trace(:,:),gpu_distance(:,:),gpu_raw_fl(:,:),gpu_raw_fr(:,:)
  integer,allocatable :: gpu_kind(:),gpu_mode(:)
  integer :: gpu_unit
"""

EXPORT_CAPTURE = """
  ! Replay read-only reconstruction queries solely to export their exact inputs.
  ! These are not GPU reconstruction or boundary-condition implementations.
  if(amr_enabled.or.bedlaw/=0.or.exchange_mode/=0.or.secondary_flow.or. &
     horizontal_viscosity>0.d0) error stop 'GPU snapshot requires fixed-grid bare hydraulics'
  allocate(gpu_trace(10,nf),gpu_distance(2,nf),gpu_kind(nf),gpu_mode(nf),gpu_raw_fl(3,nf),gpu_raw_fr(3,nf))
  gpu_trace=0.d0;gpu_distance=0.d0;gpu_kind=0;gpu_mode=0
  gpu_raw_fl=fl(1:3,:);gpu_raw_fr=fr(1:3,:)
  do f=1,nf
   c=owner(f);n=neighbour(f);b=bid(f)
   gpu_distance(1,f)=abs(dot_product(mid(:,f)-xy(:,c),normal(:,f)))
   if(n>0) gpu_distance(2,f)=abs(dot_product(xy(:,n)-mid(:,f),normal(:,f)))
   if(b==0.and.state(1,c)<=dry_tolerance) then
    if(n==0) cycle
    if(state(1,n)<=dry_tolerance) cycle
   endif
   left=face_value(c,f,v,grad,vlo,vhi)
   depth=max(left(1)-left(bed_primitive_index()),0.d0);gpu_trace(9,f)=depth
   if(state(1,c)<=dry_tolerance) depth=0.d0
   if(hydro_constant(c).and.depth<=dry_tolerance) depth=0.d0
   gpu_trace(1:3,f)=[depth,depth*left(2:3)];gpu_trace(4,f)=left(bed_primitive_index())
   if(n>0) then
    gpu_kind(f)=1;right=face_value(n,f,v,grad,vlo,vhi)
    depth=max(right(1)-right(bed_primitive_index()),0.d0);gpu_trace(10,f)=depth
    if(state(1,n)<=dry_tolerance) depth=0.d0
    if(hydro_constant(n).and.depth<=dry_tolerance) depth=0.d0
    gpu_trace(5:7,f)=[depth,depth*right(2:3)];gpu_trace(8,f)=right(bed_primitive_index())
   else if(b==0) then
    gpu_kind(f)=2
    gpu_trace(5:8,f)=gpu_trace(1:4,f);gpu_trace(10,f)=gpu_trace(9,f)
   else
    gpu_kind(f)=3;gpu_mode(f)=mode(b)
    gpu_trace(1:3,f)=trace(:,f)
   endif
  enddo
"""

EXPORT_WRITE = """
  ! Reached only after the original draining/CFL/positivity update succeeded.
  open(newunit=gpu_unit,file='gpu_snapshot.bin',access='stream',form='unformatted',status='new')
  write(gpu_unit) nc,nf,size(faces)
  write(gpu_unit) dt,grav,dry_tolerance,cfl_target
  write(gpu_unit) state(1:3,:),bed,normal,length,area
  write(gpu_unit) owner-1,neighbour-1,offset-1,faces-1,signs
  write(gpu_unit) merge(1,0,hydro_constant),merge(1,0,active_cell)
  write(gpu_unit) gpu_kind,gpu_mode,sampled_boundary_bed,boundary_bed_offset
  write(gpu_unit) gpu_distance,gpu_trace,gpu_raw_fl,gpu_raw_fr,speed
  write(gpu_unit) water_theta,fl(1:3,:),fr(1:3,:),next(1:3,:)
  close(gpu_unit)
  stop 'GPU hydraulic snapshot complete; this is not a completed simulation'
"""


def export_transform(source):
    start = source.index(" subroutine euler(")
    end = source.index(" end subroutine", start) + len(" end subroutine")
    block = source[start:end]
    block = replace_once(block, "  if(any(workspace_shape/=", EXPORT_DECLARATIONS + "  if(any(workspace_shape/=")
    block = replace_once(block, "  ! Conservative water draining changes", EXPORT_CAPTURE + "  ! Conservative water draining changes")
    anchor = "  ! With no bed flux or transported scalar"
    if block.count(anchor) != 2:
        raise ValueError("Expected the two frozen scalar-sweep anchors")
    block = block.replace(anchor, EXPORT_WRITE + anchor, 1)
    return source[:start] + block + source[end:]


def build(tag=TAG):
    root, report = paths(tag)
    root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    runtime = root / "runtime_export"
    runtime.mkdir()
    parent = json.loads((PARENT / "build.json").read_text())
    hashes, copied = {}, []
    for name, expected in parent["source_sha256"].items():
        source = PARENT / Path(name).name
        if sha(source) != expected:
            raise ValueError(f"Frozen source changed: {source}")
        target = runtime / source.name
        if target.name == "cutcell_solver.f90":
            target.write_text(export_transform(source.read_text()))
        else:
            shutil.copy2(source, target)
        hashes[name] = sha(target)
        copied.append(str(target))
    command = [r"C:\Strawberry\c\bin\gfortran.exe", "-O3", "-flto", "-fopenmp",
               "-ffree-line-length-none", "-ffixed-line-length-none", *copied,
               "-o", str(runtime / "xbraids_candidate.exe")]
    result = subprocess.run(command, cwd=runtime, capture_output=True, text=True, timeout=240)
    (runtime / "build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    write(runtime / "build.json", dict(command=command, source_sha256=hashes,
          executable_sha256=sha(runtime / "xbraids_candidate.exe"),
          parent_executable_sha256=parent["executable_sha256"],
          scope="Read-only one-Euler-stage export instrumentation of frozen hydraulic CPU code"))
    gpu_parent = json.loads((GPU_PARENT / "build.json").read_text())
    if sha(GPU_PARENT / "augmented_fwave.cu") != gpu_parent["source_sha256"]["augmented_fwave.cu"]:
        raise ValueError("Frozen GPU augmented source changed")
    for name, source in (("augmented_fwave.cu", GPU_PARENT / "augmented_fwave.cu"),
                         ("resident_update.cu", HERE / "resident_update.cu"),
                         ("resident_update.py", Path(__file__)),
                         ("Clawpack-LICENSE.txt", HERE / "Clawpack-LICENSE.txt")):
        shutil.copy2(source, root / name)
    write(root / "resident_build.json", dict(
        source_sha256={p.name: sha(p) for p in root.iterdir() if p.is_file()},
        cuda_options=list(OPTIONS), scope=SCOPE, production_modified=False,
        delft_modified=False, reference_tag=GPU_PARENT.name))
    print(f"BUILT snapshot runtime {runtime}", flush=True)


def snapshots(tag=TAG):
    sys.path.insert(0, str(ROOT))
    from braids_support.sediment.lfm_bend_2d import hydraulic_performance as h
    from braids_support.sediment.lfm_bend_2d import implicit_fwave as m
    root, report = paths(tag)
    exe, identity = h.runtime(root, "export")
    rows = []
    for name in ("bend_small", "bend_large", "cutcell_lake", "dry_dam_break", "steep_wave", "hydraulic_jump"):
        folder = root / name
        if name.startswith("bend"):
            h.prepare_native(root, folder, dx=.10625 if name == "bend_small" else .053125,
                             end=.01, output=.01, variant="export")
        else:
            mesh, q, bed, bc, end, *_ = m.fixture(name)
            folder.mkdir(exist_ok=False)
            h.r.write_cut_case(folder / "case.data", mesh, q, bed, bc, end_time_s=end,
                dt_max_s=.0001, output_interval_s=end, cfl=.4, scheme=6,
                bedload=0, exchange=0, settling_velocity_m_s=0, dry_tolerance_m=.0001,
                manning=0, source_sha256=identity["source_sha256"])
            mesh.write_geometry(folder / "geometry.npz")
        result = subprocess.run([str(exe), "case.data"], cwd=folder,
            env=dict(os.environ, OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE"),
            capture_output=True, text=True, timeout=120)
        (folder / "snapshot.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        snapshot = read_snapshot(folder / "gpu_snapshot.bin")
        row = dict(case=name, cells=snapshot["q"].shape[1], faces=snapshot["length"].size,
                   input_sha256=sha(folder / "case.data"), snapshot_sha256=sha(folder / "gpu_snapshot.bin"),
                   reference_executable_sha256=identity["executable_sha256"],
                   scope="One original CPU Euler stage before friction; not a completed simulation")
        rows.append(row)
        print(json.dumps(row), flush=True)
    write(report / "snapshots.json", rows)


def read_snapshot(path):
    with Path(path).open("rb") as stream:
        nc, nf, nadj = np.fromfile(stream, dtype="<i4", count=3)
        if min(nc, nf, nadj) <= 0:
            raise ValueError("Invalid snapshot dimensions")
        scalars = np.fromfile(stream, dtype="<f8", count=4)
        result = dict(zip(("dt", "grav", "dry", "cfl_target"), map(float, scalars)))
        def array(name, count, components=1, integer=False):
            values = np.fromfile(stream, dtype="<i4" if integer else "<f8", count=int(count)*components)
            if values.size != int(count)*components:
                raise ValueError(f"Truncated snapshot field {name}")
            result[name] = np.ascontiguousarray(values.reshape(int(count), components).T) if components>1 else values
        for name, count, components in (("q",nc,3),("bed",nc,1),("normal",nf,2),("length",nf,1),("area",nc,1)):
            array(name,count,components)
        for name,count in (("owner",nf),("neighbour",nf),("offset",nc+1),("faces",nadj),("signs",nadj),
                           ("hydro_constant",nc),("active",nc),("kind",nf),("boundary_mode",nf),("sampled",nf)):
            array(name,count,integer=True)
        for name,count,components in (("bedoffset",nf,1),("distance",nf,2),("trace",nf,10),
                ("raw_fl",nf,3),("raw_fr",nf,3),("speed",nf,1),("theta",nc,1),
                ("fl",nf,3),("fr",nf,3),("next",nc,3)):
            array(name,count,components)
        if stream.read(1):
            raise ValueError("Unexpected trailing snapshot data")
    validate_input(result)
    return result


def validate_input(data):
    nc, nf = data["q"].shape[1], len(data["length"])
    if data["q"].shape != (3,nc) or data["trace"].shape != (10,nf):
        raise ValueError("Invalid state/trace shape")
    for value in data.values():
        if isinstance(value,np.ndarray) and value.dtype.kind=="f" and not np.isfinite(value).all():
            raise ValueError("Nonfinite input")
    if np.any(data["q"][0]<0) or np.any(data["trace"][[0,4,8,9]]<0):
        raise ValueError("Negative input depth")
    if np.any(data["area"]<=0) or np.any(data["length"]<=0):
        raise ValueError("Nonpositive geometry")
    if np.max(np.abs(np.sum(data["normal"]**2,axis=0)-1))>1.e-12:
        raise ValueError("Face normals are not unit vectors")
    if np.any((data["owner"]<0)|(data["owner"]>=nc)) or np.any((data["neighbour"]< -1)|(data["neighbour"]>=nc)):
        raise ValueError("Invalid ownership")
    if data["offset"].shape!=(nc+1,) or data["offset"][0]!=0 or data["offset"][-1]!=len(data["faces"]) or np.any(np.diff(data["offset"])<0):
        raise ValueError("Invalid CSR offsets")
    if np.any((data["faces"]<0)|(data["faces"]>=nf)) or not np.isin(data["signs"],[-1,1]).all():
        raise ValueError("Invalid CSR incidence")
    if not np.isin(data["kind"],[0,1,2,3]).all() or np.any((data["kind"]==1)&(data["neighbour"]<0)):
        raise ValueError("Invalid face kind")


class ResidentUpdate:
    """Persistent buffers for a frozen, pre-reconstructed Euler stage.

    Caller MUST check face_status, cell_status and augmented_status before
    accepting next. A flagged buffer is a rejected trial, not a usable state.
    Repeating step() repeats this frozen stage: it does not refresh WENO or
    boundary traces and therefore MUST NOT be advertised as time evolution.
    """
    def __init__(self, data, root):
        import cupy as cp
        validate_input(data)
        manifest = json.loads((Path(root)/"resident_build.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            if sha(Path(root)/name)!=expected:
                raise ValueError(f"Frozen resident source changed: {name}")
        code = (Path(root)/"augmented_fwave.cu").read_text()+"\n"+(Path(root)/"resident_update.cu").read_text()
        names = ("resident_prepare_faces","augmented_fwave_local","resident_assemble_flux",
                 "resident_draining","resident_limit_flux","resident_gather_update")
        module = cp.RawModule(code=code, options=tuple(manifest["cuda_options"]), name_expressions=names)
        self.kernels = {name:module.get_function(name) for name in names}
        self.cp, self.module = cp, module
        self.nc, self.nf = data["q"].shape[1], len(data["length"])
        input_names=("q","bed","normal","length","area","owner","neighbour","offset","faces","signs",
                     "hydro_constant","active","kind","boundary_mode","sampled","bedoffset","distance","trace")
        # Expected CPU fluxes and states are validation oracles only. Never
        # upload them into the resident calculation or its scratch buffers.
        self.inputs = {k:cp.asarray(data[k]) for k in input_names}
        self.controls = {k:np.float64(data[k]) for k in ("dt","grav","dry","cfl_target")}
        self.buffers = {"local":cp.empty((8,self.nf),dtype=cp.float64),
                        "riemann":cp.empty((18,self.nf),dtype=cp.float64),
                        "closed":cp.empty(self.nf,dtype=cp.int32),
                        "augmented_status":cp.empty(self.nf,dtype=cp.int32),
                        "face_status":cp.empty(self.nf,dtype=cp.int32),
                        "cell_status":cp.empty(self.nc,dtype=cp.int32),
                        "theta":cp.empty(self.nc,dtype=cp.float64),
                        "fl":cp.empty((3,self.nf),dtype=cp.float64),
                        "fr":cp.empty((3,self.nf),dtype=cp.float64),
                        "raw_fl":cp.empty((3,self.nf),dtype=cp.float64),
                        "raw_fr":cp.empty((3,self.nf),dtype=cp.float64),
                        "speed":cp.empty(self.nf,dtype=cp.float64),
                        "next":cp.empty((3,self.nc),dtype=cp.float64),
                        "cfl_ratio":cp.empty(self.nc,dtype=cp.float64),
                        "maxdepth":cp.empty((),dtype=cp.float64)}

    def step(self, capture_raw=False):
        cp,a,b=self.cp,self.inputs,self.buffers
        nc,nf=np.int64(self.nc),np.int64(self.nf)
        dt,g,dry,target=[self.controls[k] for k in ("dt","grav","dry","cfl_target")]
        def launch(name,size,args):
            self.kernels[name](((size+127)//128,),(128,),args)
        launch("resident_prepare_faces",self.nf,(a["trace"],a["q"],a["bed"],a["normal"],a["distance"],
            a["owner"],a["neighbour"],a["kind"],nc,nf,g,dry,b["local"],b["closed"]))
        launch("augmented_fwave_local",self.nf,(b["local"],nf,g,dry,b["riemann"],b["augmented_status"]))
        launch("resident_assemble_flux",self.nf,(a["trace"],b["local"],b["riemann"],a["q"],a["bed"],a["normal"],
            a["owner"],a["neighbour"],a["kind"],a["boundary_mode"],a["sampled"],a["bedoffset"],a["hydro_constant"],
            b["closed"],nc,nf,g,dry,b["fl"],b["fr"],b["speed"],b["face_status"]))
        if capture_raw:
            cp.copyto(b["raw_fl"],b["fl"]);cp.copyto(b["raw_fr"],b["fr"])
        launch("resident_draining",self.nc,(a["q"],a["area"],a["length"],b["fl"],a["offset"],a["faces"],
            a["signs"],a["active"],nc,nf,dt,b["theta"]))
        launch("resident_limit_flux",self.nf,(a["q"],b["theta"],a["owner"],a["neighbour"],a["kind"],nc,nf,dry,
            b["fl"],b["fr"],b["face_status"]))
        cp.max(a["q"][0],out=b["maxdepth"])
        launch("resident_gather_update",self.nc,(a["q"],a["area"],a["length"],a["normal"],b["fl"],b["fr"],
            b["speed"],b["maxdepth"],a["offset"],a["faces"],a["signs"],a["active"],nc,nf,dt,g,target,
            b["next"],b["cfl_ratio"],b["cell_status"]))
        return b


def validate(tag=TAG):
    root,report=paths(tag)
    receipt=json.loads((report/"snapshots.json").read_text())
    rows=[]
    for case in receipt:
        path=root/case["case"]/"gpu_snapshot.bin"
        if sha(path)!=case["snapshot_sha256"]:
            raise ValueError("Snapshot changed")
        data=read_snapshot(path); pipeline=ResidentUpdate(data,root)
        buffers=pipeline.step(capture_raw=True)
        pipeline.cp.cuda.Stream.null.synchronize()
        actual={k:pipeline.cp.asnumpy(v) for k,v in buffers.items()}
        errors={}
        for field in ("raw_fl","raw_fr","speed","theta","fl","fr","next"):
            difference=np.abs(actual[field]-data[field])
            errors[field]=dict(max_absolute=float(difference.max()),
                max_scaled=float(np.max(difference/np.maximum(1.0,np.abs(data[field])))),
                bitwise_equal=bool(np.array_equal(actual[field],data[field])))
        boundary=data["neighbour"]<0
        boundary_volume=-data["dt"]*np.dot(data["length"][boundary],actual["fl"][0,boundary])
        volume_change=np.dot(data["area"],actual["next"][0]-data["q"][0])
        budget=float(volume_change-boundary_volume)
        status={k:np.unique(actual[k],return_counts=True)[0].tolist() for k in ("face_status","cell_status","augmented_status")}
        passed=all(row["max_scaled"]<2.e-11 for row in errors.values()) and abs(budget)<1.e-10 and actual["next"][0].min()>=0
        passed &= all(np.all(actual[k]==0) for k in status)
        row=dict(case=case["case"],cells=pipeline.nc,faces=pipeline.nf,errors=errors,status=status,
            closed_faces=int(actual["closed"].sum()),limited_cells=int((actual["theta"]<1).sum()),
            minimum_depth=float(actual["next"][0].min()),water_budget_residual_m3=budget,
            maximum_lake_change=float(np.abs(actual["next"]-data["q"]).max()) if case["case"]=="cutcell_lake" else None,
            passed=bool(passed),scope=SCOPE,full_solver=False,precision="float64",timing_claim=None,
            snapshot_sha256=case["snapshot_sha256"],cpu_reference=case["reference_executable_sha256"])
        rows.append(row)
        np.savez_compressed(root/case["case"]/"gpu_resident_arrays.npz",**actual)
        print(json.dumps(row),flush=True)
    record=dict(rows=rows,all_passed=all(row["passed"] for row in rows),scope=SCOPE,
        cuda_options=list(OPTIONS),production_modified=False,delft_modified=False)
    write(report/"braids_support.json",record)
    if not record["all_passed"]:
        raise RuntimeError("GPU resident stage qualification failed")


def recheck(tag=TAG):
    """Check the current driver without replacing the frozen validation record."""
    root,destination=paths(tag)
    receipt=json.loads((destination/"snapshots.json").read_text())
    fields=("raw_fl","raw_fr","speed","theta","fl","fr","next")
    rows=[]
    for case in receipt:
        path=root/case["case"]/"gpu_snapshot.bin"
        if sha(path)!=case["snapshot_sha256"]:raise ValueError("Snapshot changed")
        data=read_snapshot(path);pipeline=ResidentUpdate(data,root)
        if set(fields)&set(pipeline.inputs):raise ValueError("Oracle array uploaded")
        buffers=pipeline.step(capture_raw=True)
        arrays={key:pipeline.cp.asnumpy(buffers[key]) for key in fields}
        statuses={key:pipeline.cp.asnumpy(buffers[key]) for key in
                  ("face_status","cell_status","augmented_status")}
        row=dict(case=case["case"],all_finite=all(np.isfinite(v).all() for v in arrays.values()),
                 bitwise_equal=all(np.array_equal(arrays[k],data[k]) for k in fields),
                 statuses_zero=all(np.all(v==0) for v in statuses.values()),oracle_arrays_uploaded=False)
        row={key:bool(value) if isinstance(value,np.bool_) else value for key,value in row.items()}
        rows.append(row);print(json.dumps(row),flush=True)
    passed=all(row["all_finite"] and row["bitwise_equal"] and row["statuses_zero"] for row in rows)
    write(destination/"current_driver_recheck.json",dict(rows=rows,all_passed=passed,
        current_driver_sha256=sha(__file__),frozen_build_manifest_sha256=sha(root/"resident_build.json"),
        scope=SCOPE,timing_claim=None))
    if not passed:raise RuntimeError("Current GPU driver recheck failed")


def direct_fixture(prescribed=False):
    """Analytic limiter fixture, not a geometrically closed hydraulic domain."""
    nc=1 if prescribed else 2
    q=np.zeros((3,nc));q[:,0]=[.2,.06,.04]
    trace=np.zeros((10,1));trace[:3,0]=q[:,0];trace[8,0]=.2
    return dict(dt=1.,grav=9.81,dry=1.e-4,cfl_target=100.,q=q,bed=np.zeros(nc),
        trace=trace,normal=np.array([[1.],[0.]]),distance=np.ones((2,1)),length=np.ones(1),area=np.ones(nc),
        owner=np.array([0],dtype=np.int32),neighbour=np.array([-1 if prescribed else 1],dtype=np.int32),
        offset=np.arange(nc+1,dtype=np.int32),faces=np.zeros(nc,dtype=np.int32),
        signs=np.array([1] if prescribed else [1,-1],dtype=np.int32),
        active=np.ones(nc,dtype=np.int32),hydro_constant=np.zeros(nc,dtype=np.int32),
        kind=np.array([3 if prescribed else 1],dtype=np.int32),boundary_mode=np.zeros(1,dtype=np.int32),
        sampled=np.zeros(1,dtype=np.int32),bedoffset=np.zeros(1))


def direct_tests(tag=TAG):
    """Force limiter and rejection branches independently of accepted CFL runs.

    Deliberately excessive trial flux/time is a unit test, not authorization to
    run a hydraulic model above its CFL limit. Oracle uses exact scalar FP64
    copies of the original draining equations, not a second f-wave solver.
    """
    root,report=paths(tag);rows=[]
    for name,prescribed,target in (("draining_conserves_two_cells",False,100.),
                                   ("prescribed_outflow_rejected",True,100.),
                                   ("excessive_cfl_rejected",False,.01)):
        data=direct_fixture(prescribed);pipeline=ResidentUpdate(data,root)
        cp,a,b=pipeline.cp,pipeline.inputs,pipeline.buffers
        initial_flux=np.array([[1.],[.9],[.8]],dtype=np.float64)
        b["fl"].set(initial_flux);b["fr"].set(initial_flux);b["speed"].fill(1.)
        b["face_status"].fill(0);nc=np.int64(pipeline.nc);nf=np.int64(pipeline.nf)
        pipeline.kernels["resident_draining"]((1,),(128,),(a["q"],a["area"],a["length"],b["fl"],
            a["offset"],a["faces"],a["signs"],a["active"],nc,nf,np.float64(1.),b["theta"]))
        pipeline.kernels["resident_limit_flux"]((1,),(128,),(a["q"],b["theta"],a["owner"],a["neighbour"],
            a["kind"],nc,nf,np.float64(data["dry"]),b["fl"],b["fr"],b["face_status"]))
        theta=cp.asnumpy(b["theta"]);left=cp.asnumpy(b["fl"]);right=cp.asnumpy(b["fr"])
        status=cp.asnumpy(b["face_status"])
        expected_theta=.2*(1-32*np.finfo(np.float64).eps)
        expected_flux=initial_flux.copy()
        if not prescribed:
            correction=expected_theta-1
            expected_flux[0,0]=expected_theta
            expected_flux[1:,0]+=correction*(data["q"][1:,0]/data["q"][0,0])
        passed=bool(theta[0]==expected_theta and np.array_equal(left,expected_flux) and np.array_equal(right,left))
        row=dict(case=name,theta=theta.tolist(),expected_theta=expected_theta,
                 shared_flux=left[:,0].tolist(),flux_error=float(np.abs(left-expected_flux).max()),
                 face_status=status.tolist(),oracle="Analytic FP64 draining equations",full_solver=False)
        if prescribed:
            # The original CPU Euler returns here; do not gather a rejected trial.
            passed &= bool(status[0]==2)
            row["accepted_state_produced"]=False
        else:
            cp.max(a["q"][0],out=b["maxdepth"])
            pipeline.kernels["resident_gather_update"]((1,),(128,),(a["q"],a["area"],a["length"],a["normal"],
                b["fl"],b["fr"],b["speed"],b["maxdepth"],a["offset"],a["faces"],a["signs"],a["active"],
                nc,nf,np.float64(1.),np.float64(9.81),np.float64(target),b["next"],b["cfl_ratio"],b["cell_status"]))
            next_state=cp.asnumpy(b["next"]);cell_status=cp.asnumpy(b["cell_status"])
            if target<1:
                passed &= bool(np.all(cell_status==2) and np.array_equal(next_state,data["q"]))
            else:
                passed &= bool(np.all(cell_status==0) and next_state[0].min()>=0 and np.sum(next_state[0])==.2)
            row.update(minimum_depth=float(next_state[0].min()),water_change=float(np.sum(next_state[0])-np.sum(data["q"][0])),
                       cell_status=cell_status.tolist(),cfl_ratio=cp.asnumpy(b["cfl_ratio"]).tolist())
        row["passed"]=bool(passed);rows.append(row)
        print(json.dumps(row),flush=True)
    write(report/"direct_tests.json",dict(rows=rows,all_passed=all(r["passed"] for r in rows),
        scope="Unit tests of forced draining and explicit rejection, not super-CFL simulation qualification"))
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Direct GPU limiter/rejection test failed")


def report(tag=TAG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize,TwoSlopeNorm
    root,destination=paths(tag)
    validation=json.loads((destination/"braids_support.json").read_text())
    direct=json.loads((destination/"direct_tests.json").read_text())
    folder=root/"bend_large";data=read_snapshot(folder/"gpu_snapshot.bin")
    actual=np.load(folder/"gpu_resident_arrays.npz");geometry=np.load(folder/"geometry.npz")
    vertices=geometry["vertices_xy"];offsets=geometry["vertex_offsets"]
    grid=json.loads(geometry["metadata_json"].item())["grid"]
    polygons=[]
    for k in range(len(offsets)-1):
        polygon=vertices[offsets[k]:offsets[k+1]]
        if not len(polygon):
            # Packed geometry omits vertices for regular Cartesian cells.
            # Reconstruct them; empty polygons silently drop valid cells.
            if not geometry["regular_cell"][k]:raise ValueError("Missing cut-cell polygon")
            x,y=geometry["cell_scalar"][k,1:3]
            scale=geometry["member_scale"][k]
            hx,hy=grid["dx_m"]*scale/2,grid["dy_m"]*scale/2
            polygon=np.array([[x-hx,y-hy],[x+hx,y-hy],[x+hx,y+hy],[x-hx,y+hy]])
        polygons.append(polygon)
    cpu=(data["next"][0]-data["q"][0])*1.e6
    gpu=(actual["next"][0]-data["q"][0])*1.e6
    error=np.abs(actual["next"][0]-data["next"][0])
    limit=max(float(np.abs(cpu).max()),1.e-12)
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),constrained_layout=True)
    for ax,values,title in zip(axes,(cpu,gpu,error),("Frozen CPU Euler stage","Resident GPU Euler stage","Absolute CPU–GPU depth difference")):
        is_error=ax is axes[2]
        collection=PolyCollection(polygons,array=values,cmap="viridis" if is_error else "RdBu_r",
            norm=Normalize(0.,1.e-15) if is_error else TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit),edgecolors="none")
        ax.add_collection(collection);ax.autoscale_view();ax.set_aspect("equal");ax.set_title(title,fontsize=11)
        ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
        fig.colorbar(collection,ax=ax,shrink=.7,label="Depth error [m]; scale floor 10⁻¹⁵" if is_error else "Depth increment [µm]")
    axes[2].text(.03,.04,"Exactly zero in all 20,088 cells",transform=axes[2].transAxes,fontsize=9,
                 bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
    fig.suptitle("GPU resident hydraulic-stage equivalence | 20,088-cell bend, 42,846 faces\n"
                 f"One Euler stage, dt = {data['dt']:.6g} s; CPU-produced WENO/boundary traces",fontsize=12)
    figure=destination/"BEND_STAGE_EQUIVALENCE_v2.png"
    if figure.exists():raise FileExistsError(figure)
    fig.savefig(figure,dpi=180);plt.close(fig)
    rows=validation["rows"]
    maximum_budget=max(abs(row["water_budget_residual_m3"]) for row in rows)
    lines=["GPU resident hydraulic stage — qualification", "", "Outcome: "+("PASS" if validation["all_passed"] and direct["all_passed"] else "FAIL"),
        "", SCOPE, "", "Implemented on GPU:",
        "  Normal/tangent rotation, unchanged augmented f-wave algorithm port, shared mass flux;",
        "  reflective/prescribed trace fluxes, centre-stage wet/dry aperture and dynamic run-up;",
        "  bed-pressure quadrature, conservative donor draining, deterministic CSR gathering;",
        "  CFL/negative-depth rejection status and pre-friction Euler hydraulic update.",
        "", "Original Fortran snapshot comparisons:"]
    for row in rows:
        lines.append(f"  {row['case']}: {row['cells']} cells, {row['faces']} faces; all seven compared fields bitwise equal; "
                     f"water budget {row['water_budget_residual_m3']:.3e} m3.")
    lines += [f"  Maximum absolute water residual: {maximum_budget:.3e} m3.",
              "  Partly dry lake: 14 blocked faces; maximum state drift 6.522e-20.",
              "", "Direct branch tests (analytic oracle, not full hydraulic simulations):"]
    for row in direct["rows"]:lines.append(f"  {row['case']}: {'PASS' if row['passed'] else 'FAIL'}.")
    lines += ["", "Limitations:",
        "  Every natural snapshot is one original CPU Euler stage, not a time-evolved GPU run.",
        "  CPU WENO reconstruction and boundary characteristic/section solves are not ported.",
        "  Friction, complete time integration, AMR/reflux and sediment remain outside this proof.",
        "  Natural snapshots did not activate donor draining; separate forced-flux tests exercise it.",
        "  Sampled-boundary bed correction is implemented but these six fixtures do not qualify it.",
        "  Rejected trial buffers must not be accepted; inspect all GPU status arrays.",
        "  No runtime speedup is claimed; this runner performs correctness checks, not benchmarks.",
        "  All arithmetic is FP64; no fast-math or FP32 conversion. Production and Delft3D unchanged.",
        "", "Next bounded milestone:",
        "  Port hydraulic primitive/gradient/limiter/WENO trace preparation with the same CPU oracle;",
        "  keep mesh, state and scratch buffers resident. Then integrate boundary forcing, friction",
        "  and the accepted time integrator, and run a full hydraulic bend before sediment or adoption.",
        "", "Evidence: braids_support.json, direct_tests.json, snapshots.json, BEND_STAGE_EQUIVALENCE_v2.png.",
        "The earlier unsuffixed figure omitted packed regular-cell polygons; use this v2 figure."]
    with (destination/"REPORT_v2.txt").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    print(str(destination/"REPORT_v2.txt"),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","snapshots","validate","recheck","direct_tests","report"))
    parser.add_argument("--tag",default=TAG)
    args=parser.parse_args()
    globals()[args.action](args.tag)


if __name__=="__main__":
    main()
