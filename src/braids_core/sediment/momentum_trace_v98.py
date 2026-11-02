"""Pure pre-capped v91 viscosity trace; no cap, hydraulic or state owner."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .density_active_projection_v97 import _array, _digest, _hash, _sha
from .density_trace_projection_v98 import (
    DensityTraceGeometryV98, geometry_token_v98, validate_density_trace_inputs_v98,
)
from .shared_turbulence_consumers_v91 import verify_locked_sources_v91

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_momentum_trace_v98_module.f90"
ABI = "braids-v98-pure-pre-capped-viscosity-internal-trace-v1"


def validate_momentum_trace_inputs_v98(geometry, midpoint_q, capped_vicuv_cell, *,
    cell_area, face_distance, wet_cell_mask, trace_faces, trace_q, trace_capped_vicuv,
    trace_wet_mask, trace_receipt_sha256, trace_source_sha256, trace_runtime_sha256,
    cap_receipt_sha256, trace_cap_receipt_sha256, evaluation_index,
    native_threads=4, cancelled=False):
    if type(geometry) is not DensityTraceGeometryV98:
        raise ValueError("v98 momentum trace geometry type differs")
    nc, nf = geometry.cell_count, geometry.face_count
    q = _array(midpoint_q, (3, nc), "momentum midpoint q")
    raw_faces = np.asarray(trace_faces)
    if raw_faces.ndim != 1:
        raise ValueError("v98 momentum trace face shape differs")
    nt = raw_faces.size
    tq = _array(trace_q, (3, nt), "momentum trace q")
    if np.any(q[0] < 0) or np.any(tq[0] < 0):
        raise ValueError("v98 momentum trace negative depth")
    # Reuse authenticated topology/wet/trace constraints, not its density operator.
    validated, lookup, provenance, cancel = validate_density_trace_inputs_v98(
        geometry, q[0], np.ones(nc), reference_density=1., gravity=1.,
        face_distance=face_distance, base_face_acceleration=np.zeros(nf),
        wet_cell_mask=wet_cell_mask, base_density_face_mask=np.zeros(nf, np.int32),
        trace_faces=trace_faces, trace_depth=tq[0], trace_density=np.ones(nt),
        trace_wet_mask=trace_wet_mask, trace_receipt_sha256=trace_receipt_sha256,
        trace_source_sha256=trace_source_sha256, trace_runtime_sha256=trace_runtime_sha256,
        evaluation_index=evaluation_index, native_threads=native_threads, cancelled=cancelled)
    from .density_trace_projection_v98 import _hex
    provenance.pop("reference_density"); provenance.pop("gravity")
    provenance.update(cap_receipt_sha256=_hex(cap_receipt_sha256, "cap receipt"),
        trace_cap_receipt_sha256=_hex(trace_cap_receipt_sha256, "trace cap receipt"))
    area = _array(cell_area, (nc,), "momentum cell area")
    vic = _array(capped_vicuv_cell, (nc,), "momentum capped viscosity")
    tv = _array(trace_capped_vicuv, (nt,), "momentum trace capped viscosity")
    if np.any(area <= 0) or np.any(vic < 0) or np.any(tv < 0):
        raise ValueError("v98 momentum trace area/capped viscosity domain differs")
    _, _, distance, _, wet, _, faces, _, _, tw = validated
    return (q, vic, area, distance, wet, faces, tq, tv, tw), lookup, provenance, cancel


def make_momentum_trace_result_v98(g, inputs, provenance, flux, rate, *, runtime_sha,
                                  backend, threads):
    if backend not in ("native_cpu", "actual_cuda"):
        raise ValueError("v98 momentum trace backend differs")
    q, vic, area, distance, wet, faces, tq, tv, tw = inputs
    flux = _array(flux, (2, g.face_count), "momentum trace face flux")
    rate = _array(rate, (2, g.cell_count), "momentum trace cell rate")
    expected = np.zeros_like(flux)
    chosen = (wet[g.owner[faces]] != 0) & (tw != 0)
    f, own = faces[chosen], g.owner[faces[chosen]]
    coefficient = .5*(q[0, own]+tq[0, chosen])*.5*(vic[own]+tv[chosen])*g.projection.length[f]/distance[f]
    expected[:, f] = coefficient*(tq[1:, chosen]/tq[0, chosen]-q[1:, own]/q[0, own])
    if np.any(np.abs(flux-expected) > 64*np.finfo(float).eps*np.maximum(np.abs(expected), np.finfo(float).tiny)):
        raise ValueError("v98 momentum trace face equation differs")
    cell = np.repeat(np.arange(g.cell_count), np.diff(g.projection.offset))
    gathered = np.vstack([np.bincount(cell, weights=flux[k, g.projection.faces]*g.projection.signs,
        minlength=g.cell_count) for k in (0, 1)])/area
    gathered[:, wet == 0] = 0.
    if np.any(np.abs(rate-gathered) > 64*np.finfo(float).eps*np.maximum(np.abs(gathered), np.finfo(float).tiny)):
        raise ValueError("v98 momentum trace CSR equation differs")
    meta = dict(abi=ABI, backend=backend, geometry_sha256=g.fingerprint,
        inputs_sha256=_digest(*inputs), provenance_sha256=_hash(provenance),
        outputs_sha256=_digest(flux, rate), runtime_sha256=runtime_sha,
        native_threads=threads, trace_face_evaluation_count=int(faces.size),
        turbulence_cap_evaluation_count=0, v91_internal_mask_modification_count=0,
        physical_boundary_surrogate_count=0, depth_source_evaluation_count=0,
        numerical_python_cell_face_loops=0, state_or_clock_publication_count=0,
        automatic_cpu_fallback=False, production_admission=False, **provenance)
    return dict(face_flux_m4_s2=flux, momentum_rate_m2_s2=rate,
                metadata=meta, receipt_sha256=_hash(meta))


class NativeMomentumTraceRuntimeV98:
    def __init__(self, receipt, library, dependency, path):
        self.receipt, self.library, self.dependency = receipt, library, dependency
        self.path = Path(path)
        self.function = library.braids_momentum_trace_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int]*4+[ia]*2+[fa]*5+[ia]*4+[fa]*2+[ia]+[ctypes.c_int, fa, fa, ia]
        self.function.restype = None

    def evaluate(self, geometry, midpoint_q, capped_vicuv_cell, **kwargs):
        inputs, lookup, provenance, cancel = validate_momentum_trace_inputs_v98(
            geometry, midpoint_q, capped_vicuv_cell, **kwargs)
        if (self.receipt["receipt_sha256"] != _hash({k:v for k,v in self.receipt.items() if k != "receipt_sha256"}) or
                _sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(self.path) != self.receipt["library_sha256"]):
            raise ValueError("v98 momentum trace native runtime/source identity differs")
        q, vic, area, distance, wet, faces, tq, tv, tw = inputs
        g = geometry.projection
        flux, rate = np.empty((2, geometry.face_count)), np.empty((2, geometry.cell_count))
        status = np.empty(geometry.face_count, np.int32)
        threads = kwargs.get("native_threads", 4)
        self.function(geometry.cell_count, geometry.face_count, len(faces), len(g.faces),
            geometry.owner, lookup, q, vic, g.length, distance, area,
            g.offset, g.faces, g.signs, wet, tq, tv, tw, threads, flux, rate, status)
        if np.any(status):
            raise ValueError("v98 momentum trace native face computation failed")
        if cancel():
            raise InterruptedError("v98 momentum trace cancelled after native launch")
        return make_momentum_trace_result_v98(geometry, inputs, provenance, flux, rate,
            runtime_sha=self.receipt["receipt_sha256"], backend="native_cpu", threads=threads)


def build_native_momentum_trace_v98(build_directory, *, compiler=None):
    locked = verify_locked_sources_v91(ROOT)
    chosen = compiler or shutil.which("gfortran")
    if chosen is None and os.name == "nt":
        chosen = Path(r"C:\Strawberry\c\bin\gfortran.exe")
    if chosen is None or not Path(chosen).is_file():
        raise ValueError("v98 momentum trace requires gfortran")
    chosen = Path(chosen).resolve()
    root = Path(build_directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="momentum_trace_v98_", dir=root))
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none", "-static-libgfortran", "-static-libgcc"]
    path = build/("momentum_trace_v98.dll" if os.name == "nt" else "libmomentum_trace_v98.so")
    partial = path.with_suffix(path.suffix+".partial")
    source_sha = _sha(SOURCE)
    proc = subprocess.run([str(chosen), *flags, str(SOURCE), "-o", str(partial)], cwd=build,
        capture_output=True, text=True, timeout=120)
    if proc.returncode or not partial.is_file():
        raise ValueError("v98 momentum trace native build failed: "+proc.stdout+proc.stderr)
    if source_sha != _sha(SOURCE):
        raise ValueError("v98 momentum trace source changed during build")
    os.replace(partial, path)
    receipt = dict(abi=ABI, source_sha256=source_sha, wrapper_sha256=_sha(Path(__file__)),
        library_sha256=_sha(path), compiler_sha256=_sha(chosen), compile_flags=flags,
        locked_source_sha256=locked, cap_evaluation_count=0,
        thread_policy="independent_faces_then_CSR_cells_no_nested_team", production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dep = os.add_dll_directory(str(chosen.parent)) if os.name == "nt" and hasattr(os, "add_dll_directory") else None
    return NativeMomentumTraceRuntimeV98(receipt, ctypes.CDLL(str(path)), dep, path)
