"""Pure native coarse/fine DensIn projection, not another hydraulic owner.

The trace extends only the density mask. The historical internal-only v91
viscosity mask is not changed. Augmented f-wave fluxes and all accepted clocks
remain with the caller. Both face construction and projection use native batches.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .density_active_projection_v97 import (
    POLICY, RANK_TOL, RESIDUAL_TOL, DensityProjectionGeometryV97,
    NativeDensityActiveProjectionRuntimeV97,
    _array, _digest, _geometry_token, _hash, _sha,
    build_native_density_active_projection_v97,
    validate_projection_inputs_v97, validate_projection_result_v97,
)
from .densin_fwave_v85 import verify_locked_sources_v85

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_density_trace_projection_v98_module.f90"
BASE_SOURCE = ROOT/"solver/wave/braids_densin_residual_v95_module.f90"
FROZEN_BASE_SHA = "e701c096e400402c4386cc2015459d5900ce073094b86f97c0ad94c2516a61e3"
ABI = "braids-v98-native-density-internal-trace-v1"


@dataclass(frozen=True)
class DensityTraceGeometryV98:
    projection: DensityProjectionGeometryV97
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_index: np.ndarray
    fingerprint: str

    @property
    def cell_count(self):
        return self.projection.cell_count

    @property
    def face_count(self):
        return self.projection.face_count


def geometry_token_v98(g):
    return _hash({"projection": g.projection.fingerprint,
                  "topology": _digest(g.owner, g.neighbour, g.boundary_index)})


def resolve_density_trace_geometry_v98(projection, owner, neighbour, boundary_index):
    if (type(projection) is not DensityProjectionGeometryV97 or
            projection.fingerprint != _geometry_token(projection)):
        raise ValueError("v98 trace projection geometry identity differs")
    nc, nf = projection.cell_count, projection.face_count
    own = _array(owner, (nf,), "trace owner", integer=True)
    nei = _array(neighbour, (nf,), "trace neighbour", integer=True)
    bid = _array(boundary_index, (nf,), "trace boundary index", integer=True)
    if (np.any(own < 0) or np.any(own >= nc) or
            np.any(nei < -1) or np.any(nei >= nc) or
            np.any(nei == own) or np.any(bid < -1) or
            np.any((nei >= 0) & (bid >= 0))):
        raise ValueError("v98 trace topology owner/neighbour/physical-boundary identity differs")
    cell = np.repeat(np.arange(nc), np.diff(projection.offset))
    face, sign = projection.faces, projection.signs
    positive, negative = sign == 1, sign == -1
    if (np.any(cell[positive] != own[face[positive]]) or
            np.any(cell[negative] != nei[face[negative]]) or
            not np.array_equal(np.bincount(face[positive], minlength=nf), np.ones(nf)) or
            not np.array_equal(np.bincount(face[negative], minlength=nf), nei >= 0)):
        raise ValueError("v98 trace CSR incidence is not the true leaf topology")
    g = DensityTraceGeometryV98(projection, own, nei, bid, "")
    return DensityTraceGeometryV98(**{**g.__dict__, "fingerprint": geometry_token_v98(g)})


def _hex(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"v98 {name} SHA256 identity differs")
    return value


def validate_density_trace_inputs_v98(
    geometry, midpoint_depth, accepted_density_cell, *, reference_density,
    gravity, face_distance, base_face_acceleration, wet_cell_mask,
    base_density_face_mask, trace_faces, trace_depth, trace_density,
    trace_wet_mask, trace_receipt_sha256, trace_source_sha256,
    trace_runtime_sha256, evaluation_index, native_threads=4, cancelled=False,
):
    cancel = cancelled if callable(cancelled) else lambda: cancelled
    if not callable(cancelled) and type(cancelled) is not bool:
        raise ValueError("v98 density trace cancellation must be bool/callable")
    if cancel():
        raise InterruptedError("v98 density trace cancelled before launch")
    if (type(geometry) is not DensityTraceGeometryV98 or
            geometry.fingerprint != geometry_token_v98(geometry) or
            geometry.projection.fingerprint != _geometry_token(geometry.projection)):
        raise ValueError("v98 density trace geometry identity differs")
    if (type(native_threads) is not int or not 1 <= native_threads <= 256 or
            type(evaluation_index) is not int or evaluation_index < 1 or
            isinstance(reference_density, bool) or isinstance(gravity, bool) or
            not np.isfinite(reference_density) or not np.isfinite(gravity) or
            reference_density <= 0 or gravity <= 0):
        raise ValueError("v98 density trace parameter/count identity differs")
    nc, nf = geometry.cell_count, geometry.face_count
    h, base, wet, mask = validate_projection_inputs_v97(
        geometry.projection, midpoint_depth, base_face_acceleration,
        wet_cell_mask, base_density_face_mask,
        policy=POLICY, rank_tol=RANK_TOL, residual_tol=RESIDUAL_TOL)
    rho = _array(accepted_density_cell, (nc,), "trace accepted density")
    distance = _array(face_distance, (nf,), "trace distance")
    raw = np.asarray(trace_faces)
    if raw.ndim != 1:
        raise ValueError("v98 density trace face dimensions differ")
    nt = raw.size
    faces = _array(raw, (nt,), "trace faces", integer=True)
    th = _array(trace_depth, (nt,), "trace depth")
    tr = _array(trace_density, (nt,), "trace density")
    tw = _array(trace_wet_mask, (nt,), "trace wet", integer=True)
    if (np.any(rho <= 0) or np.any(distance <= 0) or
            np.any(faces < 0) or np.any(faces >= nf) or
            np.unique(faces).size != nt or
            np.any(geometry.neighbour[faces] != -1) or
            np.any(geometry.boundary_index[faces] != -1) or
            np.any(mask[faces] != 0) or np.any(th < 0) or np.any(tr <= 0) or
            np.any((tw != 0) & (tw != 1)) or np.any((tw == 1) & (th <= 0))):
        raise ValueError("v98 density trace must be unique internal leaf faces, never physical boundary surrogates")
    provenance = {"trace_receipt_sha256": _hex(trace_receipt_sha256, "trace receipt"),
                  "trace_source_sha256": _hex(trace_source_sha256, "trace source"),
                  "trace_runtime_sha256": _hex(trace_runtime_sha256, "trace runtime"),
                  "evaluation_index": evaluation_index,
                  "reference_density": float(reference_density), "gravity": float(gravity)}
    inputs = (h, rho, distance, base, wet, mask, faces, th, tr, tw)
    lookup = np.full(nf, -1, dtype=np.int32)
    lookup[faces] = np.arange(nt, dtype=np.int32)
    return inputs, lookup, provenance, cancel


def make_density_trace_result_v98(g, inputs, provenance, acceleration, mask,
                                 projection, *, runtime_sha, projection_runtime_sha,
                                 backend, threads):
    if (backend not in ("native_cpu", "actual_cuda") or
            projection.get("backend") != backend or
            projection.get("native_threads") != threads):
        raise ValueError("v98 density trace backend/projection thread identity differs")
    accel = _array(acceleration, (g.face_count,), "augmented face acceleration")
    active = _array(mask, (g.face_count,), "augmented density mask", integer=True)
    h, rho, distance, base, wet, base_mask, faces, th, tr, tw = inputs
    expected = np.array(base, copy=True)
    expected_mask = np.array(base_mask, copy=True)
    expected[faces] = 0.
    chosen = (wet[g.owner[faces]] != 0) & (tw != 0)
    selected = faces[chosen]
    owner = g.owner[selected]
    expected[selected] = (-provenance["gravity"]*(tr[chosen]-rho[owner]) *
        (th[chosen]+h[owner])/(4*provenance["reference_density"]*distance[selected]))
    expected_mask[faces] = chosen.astype(np.int32)
    scale = np.maximum(np.finfo(float).tiny, np.abs(expected))
    if (not np.array_equal(active, expected_mask) or
            np.any(np.abs(accel-expected) > 32*np.finfo(float).eps*scale)):
        raise ValueError("v98 density trace acceleration/mask equation differs")
    validate_projection_result_v97(projection, g.projection, h, accel, wet,
                                  active, runtime_sha=projection_runtime_sha)
    metadata = dict(abi=ABI, backend=backend, geometry_sha256=g.fingerprint,
        inputs_sha256=_digest(*inputs), provenance_sha256=_hash(provenance),
        outputs_sha256=_digest(accel, active), runtime_sha256=runtime_sha,
        projection_runtime_sha256=projection_runtime_sha,
        projection_receipt_sha256=projection["receipt_sha256"], native_threads=threads,
        trace_face_evaluation_count=int(faces.size), density_projection_owner_count=1,
        physical_boundary_surrogate_count=0, v91_mask_modification_count=0,
        numerical_python_cell_face_loops=0, state_or_clock_publication_count=0,
        automatic_cpu_fallback=False, production_admission=False, **provenance)
    result = dict(face_acceleration_m_s2=accel, density_face_mask=active,
        momentum_rate_m2_s2=projection["momentum_rate_m2_s2"], projection=projection,
        metadata=metadata, receipt_sha256=_hash(metadata))
    return result


def validate_density_trace_result_v98(result, geometry, midpoint_depth,
        accepted_density_cell, *, runtime_sha, projection_runtime_sha, **kwargs):
    if type(result) is not dict or set(result) != {"face_acceleration_m_s2",
            "density_face_mask", "momentum_rate_m2_s2", "projection", "metadata", "receipt_sha256"}:
        raise ValueError("v98 density trace result type differs")
    inputs, _, provenance, _ = validate_density_trace_inputs_v98(
        geometry, midpoint_depth, accepted_density_cell, **kwargs)
    rebuilt = make_density_trace_result_v98(geometry, inputs, provenance,
        result["face_acceleration_m_s2"], result["density_face_mask"], result["projection"],
        runtime_sha=runtime_sha, projection_runtime_sha=projection_runtime_sha,
        backend=result["metadata"].get("backend"),
        threads=(128 if result["metadata"].get("backend") == "actual_cuda" else kwargs.get("native_threads", 4)))
    if (result["metadata"] != rebuilt["metadata"] or
            result["receipt_sha256"] != rebuilt["receipt_sha256"] or
            not np.array_equal(result["momentum_rate_m2_s2"], rebuilt["momentum_rate_m2_s2"])):
        raise ValueError("v98 density trace result/input/runtime identity differs")
    return result


class NativeDensityTraceProjectionRuntimeV98:
    def __init__(self, receipt, library, dependency, library_path, projection):
        self.receipt, self.library, self.dependency = receipt, library, dependency
        self.library_path, self.projection = Path(library_path), projection
        self.function = library.braids_density_trace_faces_v98
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.function.argtypes = [ctypes.c_int]*3+[ia]*2+[fa]*4+[ia]*2+[fa]*2+[
            ia, ctypes.c_double, ctypes.c_double, ctypes.c_int, fa, ia, ia]
        self.function.restype = None

    def evaluate(self, geometry, midpoint_depth, accepted_density_cell, **kwargs):
        inputs, lookup, provenance, cancel = validate_density_trace_inputs_v98(
            geometry, midpoint_depth, accepted_density_cell, **kwargs)
        if (self.receipt["receipt_sha256"] != _hash({k:v for k,v in self.receipt.items() if k != "receipt_sha256"}) or
                _sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(BASE_SOURCE) != FROZEN_BASE_SHA or
                _sha(self.library_path) != self.receipt["library_sha256"] or
                self.projection.receipt["receipt_sha256"] != self.receipt["projection_runtime_sha256"]):
            raise ValueError("v98 density trace native runtime/source identity differs")
        h, rho, distance, base, wet, base_mask, faces, th, tr, tw = inputs
        accel = np.empty(geometry.face_count)
        active = np.empty(geometry.face_count, dtype=np.int32)
        status = np.empty(geometry.face_count, dtype=np.int32)
        threads = kwargs.get("native_threads", 4)
        self.function(geometry.cell_count, geometry.face_count, len(faces),
            geometry.owner, lookup, h, rho, distance, base, base_mask, wet, th, tr, tw,
            provenance["reference_density"], provenance["gravity"], threads,
            accel, active, status)
        if np.any(status):
            raise ValueError("v98 density trace native face construction failed")
        if cancel():
            raise InterruptedError("v98 density trace cancelled after face launch")
        projected = self.projection.evaluate(geometry.projection, h, accel, wet, active,
            policy=POLICY, native_threads=threads, cancelled=cancel)
        return make_density_trace_result_v98(geometry, inputs, provenance, accel, active,
            projected, runtime_sha=self.receipt["receipt_sha256"],
            projection_runtime_sha=self.projection.receipt["receipt_sha256"],
            backend="native_cpu", threads=threads)


def build_native_density_trace_projection_v98(build_directory, *, compiler=None):
    if _sha(BASE_SOURCE) != FROZEN_BASE_SHA:
        raise ValueError("v98 frozen v95 density equation source differs")
    locked = verify_locked_sources_v85(ROOT)
    chosen = compiler or shutil.which("gfortran")
    if chosen is None and os.name == "nt":
        chosen = Path(r"C:\Strawberry\c\bin\gfortran.exe")
    if chosen is None or not Path(chosen).is_file():
        raise ValueError("v98 native density trace requires gfortran")
    chosen = Path(chosen).resolve()
    root = Path(build_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    projection = build_native_density_active_projection_v97(root, compiler=chosen)
    build = Path(tempfile.mkdtemp(prefix="density_trace_v98_", dir=root))
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none", "-static-libgfortran", "-static-libgcc"]
    name = "density_trace_v98.dll" if os.name == "nt" else "libdensity_trace_v98.so"
    path, partial = build/name, build/(name+".partial")
    source_sha = _sha(SOURCE)
    proc = subprocess.run([str(chosen), *flags, str(SOURCE), "-o", str(partial)], cwd=build,
                          capture_output=True, text=True, timeout=120)
    if proc.returncode or not partial.is_file():
        raise ValueError("v98 density trace native build failed: "+proc.stdout+proc.stderr)
    if source_sha != _sha(SOURCE):
        raise ValueError("v98 density trace source changed during build")
    os.replace(partial, path)
    version = subprocess.run([str(chosen), "--version"], capture_output=True,
        text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI, source_sha256=source_sha, wrapper_sha256=_sha(Path(__file__)),
        library_sha256=_sha(path), compiler_sha256=_sha(chosen), compiler_identity=version,
        compile_flags=flags, projection_runtime_sha256=projection.receipt["receipt_sha256"],
        frozen_base_equation_sha256=FROZEN_BASE_SHA, locked_source_sha256=locked,
        thread_policy="independent_faces_then_v97_cells_no_nested_team",
        numerical_python_cell_face_loops=0, production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = os.add_dll_directory(str(chosen.parent)) if os.name == "nt" and hasattr(os, "add_dll_directory") else None
    return NativeDensityTraceProjectionRuntimeV98(receipt, ctypes.CDLL(str(path)), dependency, path, projection)
