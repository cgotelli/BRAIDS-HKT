"""Native pure v83 producer; legacy receipt plus explicit native execution evidence.

Only the existing source-admitted current-only VR1993/FactCr route is accepted.
The compatibility receipt remains v83; its backend label does not establish native
execution. The separate runtime/output evidence is required by the outer owner.
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

from . import factcr_lifecycle_v83 as v83
from .scalar_boundary_native_cpu_v97 import _hash, _sha, _digest
from .scalar_diffusion_native_cpu_v96 import _cancel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_factcr_native_v97_module.f90"
WRAPPER = Path(__file__).resolve()
SOURCES = (ROOT/"solver/wave/braids_vr1993_route_v66_module.f90",
           ROOT/"solver/wave/braids_factcr_route_v83_module.f90", SOURCE)
ABI_ID = "braids-v97-native-pure-v83-factcr-vr1993-producer-v1"


@dataclass(frozen=True)
class NativeFactCrEvidenceV97:
    runtime_sha256: str
    execution_fingerprint: str
    input_sha256: str
    output_sha256: str
    v83_receipt_sha256: str
    native_threads: int
    native_producer_batch_owner_count: int = 1
    python_numerical_cell_face_loop_count: int = 0
    scalar_or_clock_publication_count: int = 0


def _array(raw, shape, name, **kwargs):
    # Legacy validator freezes arrays; do not change caller mutability.
    return v83._array(np.array(raw, copy=True), shape, name, **kwargs)


class NativeFactCrRuntimeV97:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._function = library.braids_factcr_native_v97
        self._function.argtypes = ([ctypes.c_int]*4 + [ctypes.c_double]*4 +
            [fa]*26 + [ia, ctypes.POINTER(ctypes.c_int)])
        self._function.restype = None

    def _validate_runtime(self):
        if self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}):
            raise ValueError("native v97 FactCr receipt was changed")
        current = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES}
        if (current != self.receipt["source_sha256"] or
                _sha(WRAPPER) != self.receipt["wrapper_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v97 FactCr source/library identity differs")

    def produce(self, execution, *, native_threads=4, cancelled=False,
                frozen_uniform_threshold=None, **inputs):
        cancel = _cancel(cancelled)
        if cancel():
            raise InterruptedError("native v97 FactCr cancelled before launch")
        self._validate_runtime()
        v83.execution_document_v83(execution)
        if execution.role != v83.ROLE_CPU_ORACLE:
            raise ValueError("native v97 FactCr requires the source-admitted CPU v83 route")
        if type(native_threads) is not int or not 1 <= native_threads <= 256:
            raise ValueError("native v97 FactCr thread count differs")
        ng, nc = execution.fraction_count, execution.cell_count
        required = {
            "depth_after_m", "endpoint_momentum_m2_s", "observation_height_m",
            "water_density_kg_m3", "gravity_m_s2", "kinematic_viscosity_m2_s",
            "additional_shear_stress_pa", "sediment_d50_m", "sediment_d90_m",
            "sediment_density_kg_m3", "molecular_schmidt", "aks_factor",
            "maximum_volume_concentration", "minimum_depth_m", "bed_factor",
            "suspended_factor", "settling_velocity_m_s", "roughness_z0_m", "surface_fraction",
        }
        if set(inputs) != required:
            raise ValueError("native v97 FactCr producer input inventory differs")
        cell_positive = ("observation_height_m", "water_density_kg_m3", "gravity_m_s2",
                         "kinematic_viscosity_m2_s")
        grain_positive = ("sediment_d50_m", "sediment_d90_m", "sediment_density_kg_m3",
                          "molecular_schmidt", "aks_factor", "maximum_volume_concentration")
        h = _array(inputs["depth_after_m"], (nc,), "endpoint depth", nonnegative=True)
        momentum = _array(inputs["endpoint_momentum_m2_s"], (2, nc), "endpoint momentum")
        cell = {name: _array(inputs[name], (nc,), name, positive=True) for name in cell_positive}
        cell["additional_shear_stress_pa"] = _array(
            inputs["additional_shear_stress_pa"], (nc,), "additional shear", nonnegative=True)
        grain = {name: _array(inputs[name], (ng,), name, positive=True) for name in grain_positive}
        grain.update({name: _array(inputs[name], (ng,), name, nonnegative=True) for name in
                      ("minimum_depth_m", "bed_factor", "suspended_factor")})
        ws = _array(inputs["settling_velocity_m_s"], (ng, nc), "settling velocity", nonnegative=True)
        z0 = _array(inputs["roughness_z0_m"], (ng, nc), "roughness z0", positive=True)
        fraction = _array(inputs["surface_fraction"], (ng, nc), "surface fraction", nonnegative=True)
        profile = execution.lifecycle_execution.base_execution.profile
        if (np.any(grain["sediment_d90_m"] < grain["sediment_d50_m"]) or
                np.any(grain["maximum_volume_concentration"] > 1.) or np.any(fraction > 1.) or
                np.any(grain["sediment_density_kg_m3"][:, None] <= cell["water_density_kg_m3"]) or
                not np.allclose(fraction.sum(axis=0), 1., rtol=0., atol=32*np.finfo(float).eps) or
                not np.array_equal(grain["sediment_d50_m"], profile.sediment_d50_m) or
                not np.array_equal(grain["sediment_density_kg_m3"], profile.sediment_density_kg_m3) or
                not np.all(cell["gravity_m_s2"] == profile.gravity_m_s2)):
            raise ValueError("native v97 FactCr physical profile, grains or fraction differs")
        frozen = np.zeros((4, ng, nc)); use_frozen = frozen_uniform_threshold is not None
        if use_frozen:
            if not isinstance(frozen_uniform_threshold, (tuple, list)) or len(frozen_uniform_threshold) != 4:
                raise ValueError("native v97 FactCr frozen threshold inventory differs")
            values = tuple(_array(a, (ng,), f"frozen field {i}", positive=True)
                           for i, a in enumerate(frozen_uniform_threshold))
            if not np.array_equal(values[3], execution.factcr*values[2]):
                raise ValueError("native v97 FactCr frozen threshold application differs")
            for index, value in enumerate(values):
                frozen[index] = value[:, None]
        q = np.ascontiguousarray(np.vstack((h, momentum)))
        arrays = (q, *(cell[name] for name in (*cell_positive, "additional_shear_stress_pa")),
            *(grain[name] for name in (*grain_positive, "minimum_depth_m", "bed_factor", "suspended_factor")),
            ws, z0, fraction, frozen)
        # Input order matches the immutable native ABI, not dictionary insertion.
        fields = np.empty((4, ng, nc)); rawbed = np.empty((2, ng, nc))
        rawconc = np.empty((ng, nc)); bedcap = np.empty((2, ng, nc))
        capacity = np.empty((ng, nc)); target = np.empty((ng, nc)); diag = np.empty((6, ng, nc))
        counts = np.empty((ng, nc), np.int32)
        outputs = (fields, rawbed, rawconc, bedcap, capacity, target, diag)
        status = ctypes.c_int()
        input_sha = _hash(dict(arrays=_digest(*arrays), factcr=execution.factcr,
            dry=execution.lifecycle_execution.base_execution.dry_tolerance_m,
            sediment_threshold=profile.sediment_threshold_m, hfloor=profile.reference_depth_floor_m,
            frozen=use_frozen, execution=execution.fingerprint))
        self._function(nc, ng, native_threads, int(use_frozen), float(execution.factcr),
            float(execution.lifecycle_execution.base_execution.dry_tolerance_m),
            float(profile.sediment_threshold_m), float(profile.reference_depth_floor_m),
            *arrays, *outputs, counts, ctypes.byref(status))
        if status.value or not np.all(counts == 1) or not all(np.isfinite(a).all() for a in outputs):
            raise ValueError(f"native v97 FactCr kernel rejected input: status {status.value}")
        self._validate_runtime()
        if cancel():
            raise InterruptedError("native v97 FactCr cancelled after launch")
        receipt = v83.make_factcr_producer_receipt_v83(execution,
            accepted_endpoint=q, active_fraction=fraction,
            incoming_settling_velocity_m_s=ws,
            background_kinematic_viscosity_m2_s=cell["kinematic_viscosity_m2_s"],
            dstar=fields[0], critical_shields=fields[1], unfactored_critical_shear_pa=fields[2],
            effective_critical_shear_pa=fields[3], raw_bed_capacity_kg_m_s=rawbed,
            raw_equilibrium_concentration_kg_m3=rawconc, bed_capacity_kg_m_s=bedcap,
            lifecycle_capacity_kg_m_s=capacity, equilibrium_target_kg_m3=target,
            producer_diagnostics=diag, backend=v83.ROLE_CPU_ORACLE)
        v83.validate_factcr_producer_receipt_v83(receipt, execution)
        evidence = NativeFactCrEvidenceV97(self.receipt["receipt_sha256"], execution.fingerprint,
            input_sha, _digest(*outputs, counts), receipt.receipt_sha256, native_threads)
        return receipt, evidence


def build_native_factcr_runtime_v97(build_directory, *, compiler=None):
    locked = v83.verify_locked_sources_v83(ROOT)
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        candidate = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = candidate if candidate.is_file() else None
    if selected is None:
        raise ValueError("native v97 FactCr requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    directory = Path(build_directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES}
    wrapper_sha = _sha(WRAPPER)
    build = Path(tempfile.mkdtemp(prefix="v97_factcr_", dir=directory))
    name = "braids_factcr_native_v97.dll" if os.name == "nt" else "libbraids_factcr_native_v97.so"
    library_path = build/name; partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, *(str(p) for p in SOURCES), "-o", str(partial)],
        cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v97 FactCr build failed: "+process.stdout+process.stderr)
    if (sources != {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in SOURCES} or
            wrapper_sha != _sha(WRAPPER)):
        raise ValueError("native v97 FactCr source changed during build")
    os.replace(partial, library_path)
    version = subprocess.run([str(selected), "--version"], capture_output=True,
        text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=sources, wrapper_sha256=wrapper_sha,
        library_sha256=_sha(library_path), compiler_sha256=_sha(selected),
        compiler_identity=version, compile_flags=flags, locked_source_sha256=locked,
        unchanged_v66_v83_primitives=True, cell_grain_openmp=True,
        production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dependency = os.add_dll_directory(str(selected.parent))
    return NativeFactCrRuntimeV97(receipt, ctypes.CDLL(str(library_path)), dependency, library_path)


__all__ = ["NativeFactCrRuntimeV97", "NativeFactCrEvidenceV97", "build_native_factcr_runtime_v97"]
