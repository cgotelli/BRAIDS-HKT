"""Parallel pure diffusion extension for one combined v98 scalar owner.

The internal trace joins the frozen face law BEFORE the one inventory limiter.
It is never applied as a second independently limited diffusion update.
It owns no timestep, boundary memory, settling, exchange, or accepted state.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .flow2dh_boundary_diffusion import FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA
from .shared_turbulence_consumers_v91 import verify_locked_sources_v91

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_scalar_diffusion_trace_v98_module.f90"
ABI_ID = "braids-v98-native-combined-scalar-diffusion-internal-trace-v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name, *, integer=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiub":
        raise ValueError(f"native v98 diffusion {name} shape/type differs")
    if not np.isfinite(raw).all():
        raise ValueError(f"native v98 diffusion {name} is nonfinite")
    if integer and (np.any(raw != np.floor(raw)) or np.any(np.abs(raw) > 2147483647)):
        raise ValueError(f"native v98 diffusion {name} is not int32")
    result = np.ascontiguousarray(raw, dtype=np.int32 if integer else np.float64)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"native v98 diffusion {name} is negative")
    return result


def _cancel(value):
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return value
    raise ValueError("native v98 diffusion cancellation must be bool/callable")


def validate_diffusion_trace_v98(topology, mass, depth, trace_faces, trace_depth,
        trace_concentration, trace_dicuv, trace_wet_mask, trace_receipt_sha256,
        trace_source_sha256, trace_runtime_sha256, cap_receipt_sha256,
        trace_cap_receipt_sha256, evaluation_index):
    from .density_trace_projection_v98 import _hex
    nc,nf,nl = topology.cell_count,topology.face_count,np.asarray(mass).shape[0]
    raw = np.asarray(trace_faces)
    if raw.ndim != 1:
        raise ValueError("v98 diffusion trace face shape differs")
    nt = raw.size
    tf = _array(raw,(nt,),"trace faces",integer=True)
    th = _array(trace_depth,(nt,),"trace depth",nonnegative=True)
    tc = _array(trace_concentration,(nl,nt),"trace concentration",nonnegative=True)
    td = _array(trace_dicuv,(nt,),"trace pre-capped dicuv",nonnegative=True)
    tw = _array(trace_wet_mask,(nt,),"trace wet",integer=True)
    if (type(evaluation_index) is not int or evaluation_index < 1 or
            np.any(tf < 0) or np.any(tf >= nf) or np.unique(tf).size != nt or
            np.any(np.asarray(topology.neighbour)[tf] != -1) or
            np.any(np.asarray(topology.boundary_index)[tf] != -1) or
            np.any((tw != 0)&(tw != 1)) or np.any((tw != 0)&(th <= 0))):
        raise ValueError("v98 diffusion trace must be unique internal faces, not boundary surrogates")
    # Authenticate actual leaf incidence, including ordinary interior donors.
    offsets = np.asarray(topology.cell_face_offset)
    face,sign = np.asarray(topology.cell_faces),np.asarray(topology.cell_face_sign)
    cell = np.repeat(np.arange(nc),np.diff(offsets))
    own,nei = np.asarray(topology.owner),np.asarray(topology.neighbour)
    positive,negative = sign == 1,sign == -1
    if (np.any(cell[positive] != own[face[positive]]) or
            np.any(cell[negative] != nei[face[negative]]) or
            not np.array_equal(np.bincount(face[positive],minlength=nf),np.ones(nf)) or
            not np.array_equal(np.bincount(face[negative],minlength=nf),nei >= 0)):
        raise ValueError("v98 diffusion trace CSR is not true leaf topology")
    identity = dict(trace_receipt_sha256=_hex(trace_receipt_sha256,"trace receipt"),
        trace_source_sha256=_hex(trace_source_sha256,"trace source"),
        trace_runtime_sha256=_hex(trace_runtime_sha256,"trace runtime"),
        cap_receipt_sha256=_hex(cap_receipt_sha256,"cap receipt"),
        trace_cap_receipt_sha256=_hex(trace_cap_receipt_sha256,"trace cap receipt"),
        evaluation_index=evaluation_index)
    lookup = np.full(nf,-1,np.int32);lookup[tf] = np.arange(nt,dtype=np.int32)
    return tf,th,tc,td,tw,lookup,identity


class NativeScalarDiffusionTraceRuntimeV98:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        self._function = library.braids_scalar_diffusion_trace_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function.argtypes = [ctypes.c_int]*6 + [ia]*8 + [fa]*7 + [ia] + [
            ctypes.c_int, ia, fa, fa, fa, ia,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, fa, fa, fa, fa, fa,
            ctypes.POINTER(ctypes.c_int)]
        self._function.restype = None

    def evaluate(self, topology, dicuv_cell_m2_s, mass, depth, boundary,
                 exchange_area, face_active, *, dt_s, trace_faces, trace_depth, trace_concentration, trace_dicuv, trace_wet_mask,
                 trace_receipt_sha256, trace_source_sha256, trace_runtime_sha256,
                 cap_receipt_sha256, trace_cap_receipt_sha256, evaluation_index, native_threads=4,
                 cancelled=False):
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("native v98 diffusion cancelled before launch")
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v98 diffusion thread count differs")
        if (isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or
                not np.isfinite(dt_s) or dt_s <= 0):
            raise ValueError("native v98 diffusion dt must be finite and positive")
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v98 diffusion runtime receipt was changed")
        if (_sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v98 diffusion source/library identity differs")
        nc, nf, nb = topology.cell_count, topology.face_count, topology.boundary_count
        raw = np.asarray(mass)
        if raw.ndim != 2 or raw.shape[0] < 1:
            raise ValueError("native v98 diffusion requires nonempty combined lanes")
        nl = raw.shape[0]
        mass = _array(mass, (nl, nc), "mass", nonnegative=True)
        depth = _array(depth, (nc,), "depth", nonnegative=True)
        dic = _array(dicuv_cell_m2_s, (nc,), "dicuv", nonnegative=True)
        boundary = _array(boundary, (nl, nf), "boundary", nonnegative=True)
        exchange = _array(exchange_area, (nf,), "exchange area", nonnegative=True)
        active = _array(face_active, (nf,), "active mask", integer=True)
        if np.any((active != 0) & (active != 1)):
            raise ValueError("native v98 diffusion active mask is not boolean")
        owner = _array(topology.owner, (nf,), "owner", integer=True)
        neighbour = _array(topology.neighbour, (nf,), "neighbour", integer=True)
        bid = _array(topology.boundary_index, (nf,), "boundary indices", integer=True)
        offsets = _array(topology.cell_face_offset, (nc+1,), "cell offsets", integer=True)
        nadj = int(offsets[-1])
        if nadj < 1:
            raise ValueError("native v98 diffusion cell adjacency is empty")
        faces = _array(topology.cell_faces, (nadj,), "cell faces", integer=True)
        signs = _array(topology.cell_face_sign, (nadj,), "cell signs", integer=True)
        area = _array(topology.cell_area_m2, (nc,), "cell areas")
        distance = _array(topology.centre_distance_m, (nf,), "face distances")
        if (np.any(owner < 0) or np.any(owner >= nc) or np.any(neighbour < -1) or
                np.any(neighbour >= nc) or np.any(bid < -1) or np.any(bid >= nb) or
                offsets[0] != 0 or np.any(np.diff(offsets) < 0) or
                np.any(faces < 0) or np.any(faces >= nf) or
                np.any((signs != -1) & (signs != 1)) or
                np.any(area <= 0) or np.any(distance <= 0)):
            raise ValueError("native v98 diffusion topology is invalid")
        provenance = validate_diffusion_trace_v98(topology, mass, depth, trace_faces, trace_depth,
            trace_concentration, trace_dicuv, trace_wet_mask, trace_receipt_sha256,
            trace_source_sha256, trace_runtime_sha256, cap_receipt_sha256,
            trace_cap_receipt_sha256, evaluation_index)
        tf, th, tc, td, tw, lookup, identity = provenance
        dry = float(topology.profile.diffusion_dry_threshold_m)
        if not np.isfinite(dry) or dry < 0:
            raise ValueError("native v98 diffusion dry threshold is invalid")
        # Boundary-owned CSR keeps each named ledger deterministic without
        # atomics or an O(nb*nf) scan; no Python numerical cell/face loops.
        exterior = np.flatnonzero(bid >= 0)
        bfaces = np.ascontiguousarray(exterior[np.argsort(bid[exterior], kind="stable")], dtype=np.int32)
        boffset = np.ascontiguousarray(np.concatenate(([0], np.cumsum(
            np.bincount(bid[exterior], minlength=nb)))), dtype=np.int32)
        result = np.empty((nl, nc)); amount = np.empty((nl, nf))
        ledger = np.empty((nl, nb)); effective = np.empty(nf)
        internal_ledger = np.empty(nl)
        status = ctypes.c_int()
        self._function(nc, nf, nl, nb, nadj, len(bfaces), owner, neighbour,
            bid, offsets, faces, signs, boffset, bfaces, area, distance,
            dic, mass, depth, boundary, exchange, active, len(tf), lookup, th, tc, td, tw, float(dt_s), dry,
            native_threads, result, amount, ledger, internal_ledger, effective, ctypes.byref(status))
        if status.value or not all(np.isfinite(a).all() for a in (result, amount, ledger, internal_ledger, effective)):
            raise ValueError(f"native v98 diffusion kernel rejected input: status {status.value}")
        if cancel():
            raise InterruptedError("native v98 diffusion cancelled after launch")
        for a in (result, amount, ledger, internal_ledger, effective):
            a.setflags(write=False)
        return dict(mass_kg_m2=result, diffusive_face_mass_owner_out_kg=amount,
            diffusive_boundary_ledger_kg=ledger, internal_interface_mass_in_kg=internal_ledger,
            trace_receipt_sha256=identity["trace_receipt_sha256"], trace_provenance=identity,
            trace_inputs_sha256=_hash(dict(faces=tf.tolist(), depth=th.tolist(),
                concentration=tc.tolist(), dicuv=td.tolist(), wet=tw.tolist())),
            physical_boundary_surrogate_count=0, turbulence_cap_evaluation_count=0,
            effective_diffusivity_m2_s=effective,
            native_runtime_sha256=self.receipt["receipt_sha256"],
            native_threads=native_threads, scalar_diffusion_owner_count=1,
            python_numerical_cell_face_loop_count=0,
            state_or_clock_publication_count=0, native_diffusion_launch_count=1)

    def evaluate_arrays(self, *args, **kwargs):
        """One evaluation, returning the existing v92 four-array tuple."""
        result = self.evaluate(*args, **kwargs)
        return tuple(result[k] for k in (
            "mass_kg_m2", "diffusive_face_mass_owner_out_kg",
            "diffusive_boundary_ledger_kg", "effective_diffusivity_m2_s"))


def build_native_scalar_diffusion_trace_v98(build_directory, *, compiler=None):
    locked = verify_locked_sources_v91(ROOT)
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v96 scalar diffusion requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    source_sha = _sha(SOURCE)
    key = _hash(dict(source=source_sha, compiler=_sha(selected), flags=flags))[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v96_diffusion_{key}_", dir=build_root))
    name = "braids_scalar_diffusion_trace_v98.dll" if os.name == "nt" else "libbraids_scalar_diffusion_trace_v98.so"
    library_path = build/name
    partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, str(SOURCE), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v96 scalar diffusion build failed: "+process.stdout+process.stderr)
    if source_sha != _sha(SOURCE):
        raise ValueError("native v96 scalar diffusion source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run([str(selected), "--version"], capture_output=True,
                             text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=source_sha, wrapper_sha256=_sha(Path(__file__)), library_sha256=_sha(library_path),
        compiler_sha256=_sha(selected), compiler_identity=version, compile_flags=flags,
        locked_source_sha256=locked, source_equation=FLOW2DH_HORIZONTAL_DIFFUSION_SCHEMA,
        openmp_face_cell_boundary_loops=True, production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeScalarDiffusionTraceRuntimeV98(receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = ["NativeScalarDiffusionTraceRuntimeV98", "build_native_scalar_diffusion_trace_v98"]
