"""Pure OpenMP settling and exchange/dry-transfer batches for the Q4 owner.

Settling is prepared from accepted parent concentration, temperature, density
and DSS *before* transport.  Exchange consumes that prepared velocity *after*
transport.  The returned dry transfer is consumed once by the outer scalar/
bed owner.  These kernels publish no state, clock, EOS or second Exner update.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .flow2dh_exchange import (
    FLOW2DH_EXCHANGE_SCHEMA, Flow2DHExchangeResult, GalappattiTerms,
)
from .flow2dh_settling import (
    FLOW2DH_SETTLING_DYNAMIC_DSS_ID, FLOW2DH_SETTLING_DYNAMIC_DSS_SCHEMA,
    FLOW2DH_SETTLING_DYNAMIC_DSS_SEMANTIC_ID, FLOW2DH_SETTLING_ID,
    FLOW2DH_SETTLING_SCHEMA, FLOW2DH_SETTLING_SEMANTIC_ID, Flow2DHSettlingResult,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"solver/wave/braids_settling_exchange_v97_module.f90"
ABI_ID = "braids-v97-native-parent-settling-and-exchange-dry-transfer-v1"
LOCKED_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"
_PREFIX = "resources/Delft3D-main/src/utils_gpl/morphology/packages/morphology_kernel/src/"
_LOCKED = {
    _PREFIX+"eqtran.f90": "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    _PREFIX+"soursin_2d.f90": "a1cb9ea0b9dde77251ba2751dc5ec285e7e74171569e7b65a75d5ced1ada51ea",
    _PREFIX+"eqsettle.f90": "b742dd609f61e8d59c0480af35624a7f46ccb0ccd02962c8f42fe4c125a59437",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(*values):
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode())
        result.update(json.dumps(list(array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _array(value, shape, name, *, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"native v97 {name} shape/type differs")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if (not np.isfinite(result).all() or (positive and np.any(result <= 0)) or
            (nonnegative and np.any(result < 0))):
        raise ValueError(f"native v97 {name} is outside its physical domain")
    return result


def _fraction(value, ng, nc, name, *, scalar=False, positive=False):
    raw = np.asarray(value)
    if raw.ndim == 0 and scalar:
        raw = np.broadcast_to(raw, (ng, nc))
    elif raw.shape == (ng,):
        raw = np.broadcast_to(raw[:, None], (ng, nc))
    return _array(raw, (ng, nc), name, positive=positive, nonnegative=not positive)


def _wet(value, nc):
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape not in ((), (nc,)):
        raise ValueError("native v97 wet mask must be Boolean scalar/cell field")
    return np.ascontiguousarray(np.broadcast_to(raw, (nc,)), dtype=np.int32)


def _scalar(value, name, *, positive=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or (positive and value <= 0) or
            (not positive and value < 0)):
        raise ValueError(f"native v97 {name} is outside its physical domain")
    return float(value)


def _poll(value):
    if type(value) is bool:
        return lambda: value
    if callable(value):
        return value
    raise ValueError("native v97 cancellation must be Boolean/callable")


@dataclass(frozen=True)
class NativeExchangeDryTransferResultV97:
    exchange: Flow2DHExchangeResult
    wetdry_transfer_to_bed_kg_m2: np.ndarray
    final_suspended_mass_kg_m2: np.ndarray
    final_bed_mass_kg_m2: np.ndarray
    wetdry_water_ledger_kg: np.ndarray
    wetdry_bed_ledger_kg: np.ndarray
    final_mass_balance_residual_kg: np.ndarray
    native_runtime_sha256: str
    input_sha256: str
    result_sha256: str
    native_threads: int
    exchange_owner_count: int = 1
    wetdry_transfer_owner_count: int = 1
    python_numerical_cell_face_loop_count: int = 0
    state_or_clock_publication_count: int = 0
    native_exchange_launch_count: int = 1


class NativeSettlingExchangeRuntimeV97:
    def __init__(self, receipt, library, dependency, library_path):
        self.receipt = receipt
        self.library = library
        self.dependency = dependency
        self.library_path = Path(library_path).resolve()
        fa = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ia = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self._settle = library.braids_parent_settling_v97
        self._settle.argtypes = [ctypes.c_int]*3+[fa]*5+[ia]+[
            ctypes.c_double, ctypes.c_double, ctypes.c_int]+[fa]*5+[
            ctypes.POINTER(ctypes.c_int)]
        self._settle.restype = None
        self._exchange = library.braids_exchange_dry_transfer_v97
        self._exchange.argtypes = [ctypes.c_int]*2+[fa]*11+[ia]+[
            ctypes.c_double]*4+[ctypes.c_int]+[fa]*8+[ia, fa,
            ctypes.POINTER(ctypes.c_int)]
        self._exchange.restype = None

    def _begin(self, threads, cancelled):
        poll = _poll(cancelled)
        if poll():
            raise InterruptedError("native v97 exchange/settling cancelled before launch")
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError("native v97 thread count must be 1..256")
        if (self.receipt.get("receipt_sha256") != _hash({
                k: v for k, v in self.receipt.items() if k != "receipt_sha256"}) or
                _sha(SOURCE) != self.receipt["source_sha256"] or
                _sha(Path(__file__)) != self.receipt["wrapper_sha256"] or
                _sha(self.library_path) != self.receipt["library_sha256"]):
            raise ValueError("native v97 settling/exchange runtime identity differs")
        return poll

    def settle(self, suspended_concentration_kg_m3, sediment_d50_m,
               settling_fac_dss, sediment_density_kg_m3,
               water_density_kg_m3, temperature_c, *, settling_diameter_m=None,
               settling_reference_concentration_kg_m3=1.0e4, gravity_m_s2=9.81,
               wet_mask=True, native_threads=4, cancelled=False):
        """Parent-timed SettleFrm=2; output fraction axis last, as frozen oracle."""
        poll = self._begin(native_threads, cancelled)
        raw = np.asarray(suspended_concentration_kg_m3)
        if raw.ndim != 2 or min(raw.shape) < 1:
            raise ValueError("native v97 settling requires (cell, all-fractions)")
        nc, ns = raw.shape
        concentration = _array(raw, (nc, ns), "parent concentration", nonnegative=True)
        d50 = np.asarray(sediment_d50_m)
        if d50.ndim != 1 or len(d50) < 1:
            raise ValueError("native v97 settling requires target fractions")
        ng = len(d50)
        d50 = _array(d50, (ng,), "D50", positive=True)
        fac = _array(settling_fac_dss, (ng,), "FacDSS", positive=True)
        solid = _array(sediment_density_kg_m3, (ng,), "RhoSol", positive=True)
        rho = _array(np.broadcast_to(water_density_kg_m3, (nc,)), (nc,), "RhoWater", positive=True)
        temp = _array(np.broadcast_to(temperature_c, (nc,)), (nc,), "temperature")
        if np.any(temp <= -20) or np.any(solid[:, None] <= rho):
            raise ValueError("native v97 temperature/density is outside settling domain")
        if settling_diameter_m is None:
            dss = d50*fac
            diameter = np.ascontiguousarray(np.broadcast_to(dss[:, None], (ng, nc)))
        else:
            dss = np.asarray(settling_diameter_m)
            if dss.shape == (nc, ng):
                diameter = _array(dss.T, (ng, nc), "accepted parent DSS", positive=True)
            elif dss.shape == (ng,):
                diameter = _fraction(dss, ng, nc, "accepted parent DSS", positive=True)
            else:
                raise ValueError("native v97 accepted DSS shape differs")
        cref = _scalar(settling_reference_concentration_kg_m3, "Cref")
        gravity = _scalar(gravity_m_s2, "gravity")
        wet = _wet(wet_mask, nc)
        clear = np.empty((ng, nc)); velocity = np.empty((ng, nc))
        total = np.empty(nc); hindered = np.empty(nc); viscosity = np.empty(nc)
        status = ctypes.c_int()
        self._settle(nc, ns, ng, np.ascontiguousarray(concentration.T), diameter,
                     solid, rho, temp, wet, cref, gravity, native_threads,
                     clear, velocity, total, hindered, viscosity, ctypes.byref(status))
        if status.value or not all(np.isfinite(v).all() for v in (clear, velocity, total, hindered, viscosity)):
            raise ValueError("native v97 parent settling kernel rejected fields")
        if poll():
            raise InterruptedError("native v97 parent settling cancelled after launch")
        dynamic = settling_diameter_m is not None
        outputs = [np.array(v, order="C", copy=True) for v in (
            velocity.T, clear.T, dss, total, hindered, viscosity, wet.astype(bool))]
        for value in outputs:
            value.setflags(write=False)
        return Flow2DHSettlingResult(
            *outputs,
            method_id=FLOW2DH_SETTLING_DYNAMIC_DSS_ID if dynamic else FLOW2DH_SETTLING_ID,
            semantic_id=FLOW2DH_SETTLING_DYNAMIC_DSS_SEMANTIC_ID if dynamic else FLOW2DH_SETTLING_SEMANTIC_ID)

    def evaluate(self, suspended_mass_kg_m2, bed_mass_kg_m2,
                 suspended_transport_kg_m_s, surface_fraction, speed_m_s,
                 shear_velocity_m_s, depth_before_m, depth_after_m,
                 settling_velocity_m_s, cell_area_m2, *, dt_s, fac_tsd=1.0,
                 dimensionless_timescale=0.0, sediment_threshold_m=0.5,
                 minimum_reference_depth_m=0.01, flow_wet_mask=True,
                 native_threads=4, cancelled=False):
        """One native exchange and one endpoint dry transfer, without commit."""
        poll = self._begin(native_threads, cancelled)
        raw = np.asarray(suspended_mass_kg_m2)
        if raw.ndim != 2 or min(raw.shape) < 1:
            raise ValueError("native v97 exchange requires (fraction, cell)")
        ng, nc = raw.shape
        suspended = _array(raw, (ng, nc), "suspended", nonnegative=True)
        bed = _array(bed_mass_kg_m2, (ng, nc), "bed", nonnegative=True)
        capacity = _array(suspended_transport_kg_m_s, (ng, nc), "capacity", nonnegative=True)
        fraction = _fraction(surface_fraction, ng, nc, "surface fraction")
        if np.any(fraction > 1):
            raise ValueError("native v97 surface fraction exceeds one")
        speed = _array(speed_m_s, (nc,), "speed", nonnegative=True)
        shear = _array(shear_velocity_m_s, (nc,), "shear", nonnegative=True)
        h0 = _array(depth_before_m, (nc,), "parent depth", nonnegative=True)
        h1 = _array(depth_after_m, (nc,), "endpoint depth", nonnegative=True)
        settling = _fraction(settling_velocity_m_s, ng, nc, "parent settling")
        timescale = _fraction(dimensionless_timescale, ng, nc, "timescale", scalar=True)
        area = _array(cell_area_m2, (nc,), "area", positive=True)
        wet = _wet(flow_wet_mask, nc)
        dt = _scalar(dt_s, "dt"); factsd = _scalar(fac_tsd, "FacTsd")
        threshold = _scalar(sediment_threshold_m, "SedThr", positive=False)
        floor = _scalar(minimum_reference_depth_m, "reference depth floor")
        inputs = (suspended, bed, capacity, fraction, speed, shear, h0, h1,
                  settling, timescale, area, wet)
        input_sha = _hash({"arrays": _digest(*inputs), "dt_s": dt, "fac_tsd": factsd,
                           "sediment_threshold_m": threshold, "reference_floor": floor})
        next_s = np.empty((ng, nc)); next_b = np.empty((ng, nc)); eq = np.empty((ng, nc))
        terms = np.empty((6, ng, nc)); transfer = np.empty((ng, nc)); cleanup = np.empty((ng, nc))
        final_s = np.empty((ng, nc)); final_b = np.empty((ng, nc))
        active = np.empty(nc, dtype=np.int32); ledger = np.empty((ng, 6))
        status = ctypes.c_int()
        self._exchange(nc, ng, suspended, bed, capacity, fraction, speed, shear,
                       h0, h1, settling, timescale, area, wet,
                       dt, factsd, threshold, floor, native_threads,
                       next_s, next_b, eq, terms, transfer, cleanup, final_s, final_b,
                       active, ledger, ctypes.byref(status))
        outputs = (next_s, next_b, eq, terms, transfer, cleanup, final_s, final_b, active, ledger)
        if status.value or not all(np.isfinite(v).all() for v in outputs):
            raise ValueError("native v97 exchange kernel rejected fields")
        residual = ledger[:, 2]-ledger[:, 1]
        final_residual = ledger[:, 5]-ledger[:, 1]
        scale = np.maximum(1., np.maximum(np.abs(ledger[:, 1]), np.abs(ledger[:, 2])))
        if (np.any(np.abs(residual) > 4096*np.finfo(float).eps*scale) or
                np.any(np.abs(final_residual) > 4096*np.finfo(float).eps*scale)):
            raise ValueError("native v97 exchange/dry-transfer mass ledger differs")
        if poll():
            raise InterruptedError("native v97 exchange cancelled after launch")
        result_sha = _hash({"native_runtime_sha256": self.receipt["receipt_sha256"],
                            "input_sha256": input_sha, "outputs_sha256": _digest(*outputs),
                            "native_threads": native_threads, "exchange_owner_count": 1,
                            "wetdry_transfer_owner_count": 1})
        water_ledger = ledger[:, 0].copy(); bed_ledger = -water_ledger
        before = ledger[:, 1].copy(); after = ledger[:, 2].copy()
        correction = ledger[:, 3].copy(); cleanup_water = -ledger[:, 4].copy()
        cleanup_bed = -cleanup_water
        sediment_active = active.astype(bool)
        for value in (*outputs, water_ledger, bed_ledger, before, after, correction,
                      residual, cleanup_water, cleanup_bed, final_residual, sediment_active):
            value.setflags(write=False)
        exchange = Flow2DHExchangeResult(
            next_s, next_b, eq, GalappattiTerms(*terms), transfer,
            water_ledger, bed_ledger, before, after, residual, correction,
            bool(np.any(correction > 0)), sediment_active)
        return NativeExchangeDryTransferResultV97(
            exchange, cleanup, final_s, final_b, cleanup_water, cleanup_bed,
            final_residual, self.receipt["receipt_sha256"], input_sha, result_sha, native_threads)

    def evaluate_exchange(self, *args, **kwargs):
        """Compatibility-shaped result; callers must not apply dry transfer twice."""
        return self.evaluate(*args, **kwargs).exchange


def build_native_settling_exchange_runtime_v97(build_directory, *, compiler=None):
    for relative, expected in _LOCKED.items():
        if _sha(ROOT/relative) != expected:
            raise ValueError("native v97 locked Delft source identity differs: "+relative)
    selected = compiler or shutil.which("gfortran")
    if selected is None and os.name == "nt":
        fallback = Path(r"C:\Strawberry\c\bin\gfortran.exe")
        selected = fallback if fallback.is_file() else None
    if selected is None:
        raise ValueError("native v97 settling/exchange requires gfortran")
    selected = Path(selected).resolve()
    flags = ["-O3", "-fopenmp", "-shared", "-ffree-line-length-none",
             "-static-libgfortran", "-static-libgcc"]
    source_sha = _sha(SOURCE); wrapper_sha = _sha(Path(__file__))
    root = Path(build_directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    key = _hash(dict(source=source_sha, compiler=_sha(selected), flags=flags))[:16]
    build = Path(tempfile.mkdtemp(prefix=f"v97_exchange_{key}_", dir=root))
    name = "braids_settling_exchange_v97.dll" if os.name == "nt" else "libbraids_settling_exchange_v97.so"
    path = build/name; partial = build/(name+".partial")
    process = subprocess.run([str(selected), *flags, str(SOURCE), "-o", str(partial)],
                             cwd=build, capture_output=True, text=True, timeout=120)
    if process.returncode or not partial.is_file():
        raise ValueError("native v97 settling/exchange build failed: "+process.stdout+process.stderr)
    if source_sha != _sha(SOURCE) or wrapper_sha != _sha(Path(__file__)):
        raise ValueError("native v97 settling/exchange source changed during build")
    os.replace(partial, path)
    identity = subprocess.run([str(selected), "--version"], capture_output=True,
                              text=True, check=True, timeout=20).stdout.splitlines()[0]
    receipt = dict(abi=ABI_ID, source_sha256=source_sha, wrapper_sha256=wrapper_sha,
        library_sha256=_sha(path), compiler_sha256=_sha(selected), compiler_identity=identity,
        compile_flags=flags, locked_source_sha256=_LOCKED, locked_delft_commit=LOCKED_COMMIT,
        settling_equations=FLOW2DH_SETTLING_SCHEMA,
        dynamic_dss_equations=FLOW2DH_SETTLING_DYNAMIC_DSS_SCHEMA,
        exchange_equations=FLOW2DH_EXCHANGE_SCHEMA, openmp_cell_and_fixed_block_ledgers=True,
        ledger_block_size=1024, parent_settling_before_transport=True,
        exchange_and_dry_transfer_after_transport=True, accepted_state_or_clock_owner=False,
        production_admission=False)
    receipt["receipt_sha256"] = _hash(receipt)
    dependency = os.add_dll_directory(str(selected.parent)) if os.name == "nt" and hasattr(os, "add_dll_directory") else None
    return NativeSettlingExchangeRuntimeV97(receipt, ctypes.CDLL(str(path)), dependency, path)


__all__ = ["NativeExchangeDryTransferResultV97", "NativeSettlingExchangeRuntimeV97",
           "build_native_settling_exchange_runtime_v97"]
