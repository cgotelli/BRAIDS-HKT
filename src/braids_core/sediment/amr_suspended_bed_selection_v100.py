"""AMR suspended-bed method selection, separate from water exchange and Q3.

The numerical CPU batch is OpenMP Fortran; CUDA supplies the same receipt
after its resident batch. Host arrays here are immutable audit evidence, not
a CUDA numerical fallback. DT reuses the existing scalar transport amounts;
DF selects exact zero. Cleanup and Q3 remain separate, single owners.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .suspended_exner_owner_v93 import (
    ROUTE_COMBINE, SUSPENDED_VECTOR_EXACT_ZERO_V93,
)

SOURCE = Path(__file__).resolve().parents[2]/'solver/wave/braids_amr_bed_selector_v100.f90'


def mode_v100(execution):
    if not execution.neglect_entrainment:
        return 0
    return 2 if execution.suspended_vector_mode == SUSPENDED_VECTOR_EXACT_ZERO_V93 else 1


def _digest(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode('ascii'))
        h.update(json.dumps(list(a.shape)).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()


def _freeze(value):
    a = np.array(value, copy=True, order='C')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class AMRSuspendedBedSelectionV100:
    execution_fingerprint: str
    mode: int
    exchange_bed_delta_kg_m2: np.ndarray
    flux_divergence_bed_delta_kg_m2: np.ndarray
    selected_bed_delta_kg_m2: np.ndarray
    selected_bed_mass_before_q3_kg_m2: np.ndarray
    selected_exchange_ledger_kg: np.ndarray
    selected_boundary_ledger_kg: np.ndarray
    selected_internal_interface_ledger_kg: np.ndarray
    excluded_water_exchange_bed_counterpart_kg: np.ndarray
    backend: str
    runtime_sha256: str
    native_threads: int
    selector_launch_count: int
    receipt_sha256: str


def _token(value):
    return hashlib.sha256(json.dumps({
        'scope': 'braids.amr.suspended_bed_selection.v100',
        'execution': value.execution_fingerprint, 'mode': value.mode,
        'arrays': _digest(value.exchange_bed_delta_kg_m2,
            value.flux_divergence_bed_delta_kg_m2, value.selected_bed_delta_kg_m2,
            value.selected_bed_mass_before_q3_kg_m2, value.selected_exchange_ledger_kg,
            value.selected_boundary_ledger_kg, value.selected_internal_interface_ledger_kg,
            value.excluded_water_exchange_bed_counterpart_kg),
        'backend': value.backend, 'runtime': value.runtime_sha256,
        'threads': value.native_threads, 'launches': value.selector_launch_count,
        'q3_owner_count': 0, 'water_exchange_owner_count': 0,
        'clock_publication_count': 0,
    }, sort_keys=True).encode('ascii')).hexdigest()


def wrap_amr_bed_selection_v100(execution, numerical, *, area,
        advective_boundary, diffusive_boundary, internal_ledger,
        backend, runtime_sha256, native_threads):
    """Authenticate selected *bed* ledgers without relabelling water sources."""
    exchange, divergence, selected, final_bed = numerical
    mode = mode_v100(execution)
    e = np.sum(exchange*np.asarray(area)[None, :], axis=1)
    ex = e if mode == 0 else np.zeros_like(e)
    boundary = (advective_boundary+diffusive_boundary if mode == 1
                else np.zeros_like(advective_boundary))
    internal = (np.asarray(internal_ledger)[:execution.fraction_count] if mode == 1
                else np.zeros_like(np.asarray(internal_ledger)[:execution.fraction_count]))
    integrated = np.sum(selected*np.asarray(area)[None, :], axis=1)
    expected = ex+np.sum(boundary, axis=1)+np.sum(internal, axis=1)
    if not np.allclose(integrated, expected, rtol=0., atol=3e-11):
        raise ValueError('v100 selected AMR suspended-bed ledger does not close')
    fields = [_freeze(a) for a in (*numerical, ex, boundary, internal,
                                  e if mode != 0 else np.zeros_like(e))]
    value = AMRSuspendedBedSelectionV100(execution.fingerprint, mode,
        *fields, backend, runtime_sha256, native_threads, 1, '')
    value = AMRSuspendedBedSelectionV100(**{**value.__dict__, 'receipt_sha256': _token(value)})
    validate_amr_bed_selection_v100(value, execution)
    return value


def validate_amr_bed_selection_v100(value, execution):
    if (type(value) is not AMRSuspendedBedSelectionV100 or
        value.execution_fingerprint != execution.fingerprint or
        value.mode != mode_v100(execution) or value.selector_launch_count != 1 or
        value.receipt_sha256 != _token(value)):
        raise ValueError('v100 AMR suspended-bed selector identity differs')
    arrays = [v for v in value.__dict__.values() if isinstance(v, np.ndarray)]
    if any(a.flags.writeable or not np.isfinite(a).all() for a in arrays):
        raise ValueError('v100 AMR suspended-bed evidence is mutable or nonfinite')
    if value.mode == 2 and (np.any(value.selected_bed_delta_kg_m2 != 0) or
                           np.any(value.flux_divergence_bed_delta_kg_m2 != 0)):
        raise ValueError('v100 DF selected a nonzero suspended-bed update')
    if np.any(value.selected_bed_mass_before_q3_kg_m2 < 0):
        raise ValueError('v100 selected suspended-bed method overdraws finite stock')


@lru_cache(maxsize=1)
def _native_runtime():
    compiler = shutil.which('gfortran')
    if compiler is None:
        raise RuntimeError('v100 native AMR bed selector requires gfortran')
    compiler = Path(compiler).resolve()
    directory = Path(tempfile.mkdtemp(prefix='braids_amr_bed_v100_'))
    library_path = directory/('selector.dll' if os.name == 'nt' else 'selector.so')
    flags = ['-O3', '-fopenmp', '-shared', '-ffree-line-length-none',
             '-static-libgfortran', '-static-libgcc', '-ffp-contract=off']
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    built = subprocess.run([str(compiler), *flags, str(SOURCE), '-o', str(library_path)],
                           cwd=directory, capture_output=True, text=True, timeout=120)
    if built.returncode:
        raise RuntimeError(built.stdout+built.stderr)
    dependency = os.add_dll_directory(str(compiler.parent)) if os.name == 'nt' else None
    library = ctypes.CDLL(str(library_path))
    f = library.braids_amr_bed_selector_v100
    fa = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    ia = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    f.argtypes = [ctypes.c_int]*4+[ia]+[fa]*9+[ia]
    f.restype = None
    token = hashlib.sha256(json.dumps({'source': source_sha,
        'wrapper': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'library': hashlib.sha256(library_path.read_bytes()).hexdigest(),
        'compiler': hashlib.sha256(compiler.read_bytes()).hexdigest(),
        'flags': flags}, sort_keys=True).encode('ascii')).hexdigest()
    return f, library, dependency, token


def native_amr_bed_selection_v100(execution, bed, water, diffused, transfer,
                                cleanup, *, native_threads):
    inputs = [np.ascontiguousarray(a, dtype=np.float64)
              for a in (bed, water, diffused, transfer, cleanup)]
    shape = (execution.fraction_count, execution.cell_count)
    if (any(a.shape != shape or not np.isfinite(a).all() for a in inputs) or
            type(native_threads) is not int or not 1 <= native_threads <= 256):
        raise ValueError('v100 AMR selector input shape/thread count differs')
    outputs = [np.empty(shape) for _ in range(4)]
    status = np.empty(shape, dtype=np.int32)
    combine = np.ascontiguousarray(np.asarray(execution.routes) == ROUTE_COMBINE, dtype=np.int32)
    f, _, _, token = _native_runtime()
    f(shape[1], shape[0], native_threads, mode_v100(execution), combine, *inputs, *outputs, status)
    if np.any(status):
        raise ValueError('v100 selected suspended-bed method overdraws finite stock')
    return outputs, token
