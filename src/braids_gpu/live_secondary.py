"""Guarded, data-driven live-section/MPM/prognostic-secondary GPU adapter.

The generated numerical modules are the existing exact-anchor A1 composition.
Only its fixture validator is replaced in an isolated module.  The native
zero-step export, source hashes, and all numerical CUDA bodies remain intact.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

import numpy as np


VARIANT = 'live_normal_critical_secondary_mpm_ssp43'
PRIVATE_KEY = 'braids_support.gpu_fwave._public_live_secondary_adapter'
WETDRY_GATE = (Path(__file__).resolve().parents[1] / 'validation' / 'gpu_fwave' /
               'reports' / 'A2_PUBLIC_WETDRY_PRE_GPU_GATE_20261004.md')
WETDRY_GATE_SHA = '77d63ec6fe06de3da17c7a0cdfd1e4ed4d5acbac18a484658eba2377ce15df9d'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_case(folder):
    """Merge authenticated native zero-step records without fixture dimensions."""
    from braids_support.gpu_fwave.coupled_reference import read_records, read_snapshot
    from braids_support.gpu_fwave.boundary_schedule_oracle import validate_arrays

    folder = Path(folder)
    data = read_snapshot(folder)
    extra = read_records(folder / 'normal_metadata.bin')
    validate_arrays(extra)
    for key, value in extra.items():
        if key in data and not np.array_equal(data[key], value):
            raise ValueError('Normal-section duplicate export differs: ' + key)
        data[key] = value
    nb = int(data['nb'])
    for key in ('section_seed_node', 'boundary_seed_position'):
        raw = np.asarray(data['native_' + key])
        require(raw.dtype == np.int32 and raw.shape == (nb,) and np.all(raw >= 0),
                'Invalid native section seed: ' + key)
        data[key] = np.ascontiguousarray(raw - 1)
    return data


def validate_scope(data, *, start, end, interval, dtmax, launch_mode, integrator):
    """Admit a structural envelope; never change an unsupported native control."""
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.normal_outlet_boundary import validate_boundaries
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope

    require(launch_mode == 'plain' and integrator == 'ssp43',
            'Live secondary variant requires explicit plain SSPRK43')
    require(int(data['scheme']) == 6 and int(data['cpu_time_integrator']) == 43 and
            int(data['cpu_outlet_extension']) == 1,
            'Live secondary variant requires native scheme 6 SSPRK43 and outlet extension 1')
    require(int(data['nb']) == 2 and np.array_equal(data['mode'], [9, 4]) and
            np.array_equal(data['section_control'], [1, 0]),
            'Live secondary variant requires one mode9/control1 inlet and one mode4 outlet')
    offsets = np.asarray(data['boundary_offset'])
    require(offsets.shape == (3,) and offsets[1] - offsets[0] >= 4 and
            offsets[2] - offsets[1] >= 1,
            'Live secondary variant requires nonempty surveyed sections')
    require(int(data['nfrac']) == 2 and int(data['nbed_layers']) == 2 and
            int(data['bedlaw']) == 2 and int(data['mpm_friction_mode']) == 1 and
            int(data['exchange_mode']) == 0 and int(data['nstate']) == 10 and
            int(data['nledger']) == 6 and int(data['secondary_index']) == 9 and
            int(data['secondary_ledger_index']) == 5,
            'Live secondary variant requires two-class/two-layer MPM and prognostic state layout')
    grains = np.asarray(data['d50_fraction'])
    require(grains.shape == (2,) and np.all(np.isfinite(grains)) and
            np.all(grains > 0) and grains[0] != grains[1],
            'Live secondary variant requires two distinct positive grain sizes')
    require(int(data['secondary_flow_model']) == 2 and bool(data['secondary_flow']) and
            bool(data['secondary_flow_transport_direction']) and
            float(data['secondary_flow_momentum_beta']) == 1. and
            float(data['secondary_flow_espir']) == 1.,
            'Live secondary variant requires prognostic flow, direction rotation, Beta 1 and Espir 1')
    secondary_scope(data)
    state = np.asarray(data['q'])
    require(state.ndim == 2 and state.shape == (3, int(data['nc'])) and
            np.isfinite(state).all() and np.all(state[0] >= 0) and
            np.any(state[0] > float(data['dry_tolerance'])),
            'Live secondary variant requires finite nonnegative initial depth and a wet cell')
    initially_dry = state[0] <= float(data['dry_tolerance'])
    require(not np.any(state[1:, initially_dry]),
            'Live secondary variant requires zero initial momentum in dry cells')
    require(float(data['morphological_factor']) == 1. and
            float(data['horizontal_viscosity']) == 0. and
            int(data['bed_slope']) == 0 and not np.any(data['nseries']),
            'Live secondary variant excludes MorFac, viscosity, bed-slope and legacy scalar series')
    for key in ('amr_enabled', 'bedload_adaptation', 'bedform_roughness',
                'bank_failure', 'bank_avalanche', 'general_avalanche',
                'bank_front_tracking', 'bank_erosion_fraction', 'nmorfac',
                'suspended_transport_active', 'viscosity_equilibrated_inlet'):
        require(not np.any(data.get(key, 0)), 'Live secondary variant excludes ' + key)
    require(not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']) and
            not np.any(data.get('suspended_mass', 0)),
            'Live secondary variant excludes suspended inlet/exchange physics')
    validate_boundaries(data)
    # The stock table validator excludes live sections as a composition
    # policy.  Its numerical table checks are reusable with that policy flag
    # detached; the combined section validator above checks the real flags.
    tables = validate_tables(dict(data, section_control=np.zeros(2, np.int32),
                                  sampled=np.zeros(int(data['nf']), np.int32)))
    counts = np.asarray(data['nfraction_series'])
    count = int(data['nhydro_series'][0])
    require(count >= 2 and np.array_equal(counts[:, 0], [count, count]) and
            not np.any(counts[:, 1]) and not int(data['nhydro_series'][1]),
            'Live secondary variant requires two active class feeds on the inlet Q schedule')
    times = np.asarray(data['hydro_series_time'])[:count, 0]
    require(np.array_equal(data['fraction_series_time'][:count, :, 0],
                           np.repeat(times[:, None], 2, axis=1)) and
            times[0] <= start and times[-1] >= end,
            'Live secondary variant requires common Q/class-feed knots covering the run')
    require(not np.array_equal(tables['fraction_series_bed_input'][:count, 0, 0],
                               tables['fraction_series_bed_input'][:count, 1, 0]) and
            np.ptp(tables['hydro_series_q'][:count, 0]) > 0,
            'Live secondary variant requires distinct class feeds and time-varying Q')
    events = BoundaryEvents.from_tables(data)
    require(len(events.knots) >= 2 and np.array_equal(events.knots, times),
            'Live secondary variant requires one common active forcing clock')
    require(start >= 0 and end > start and interval > 0 and dtmax > 0,
            'Live secondary variant requires positive finite run controls')
    return events


@contextmanager
def isolated_adapter():
    """Use the proven source transformer without changing A1's global guard."""
    from braids_support.gpu_fwave import a1_combined_current_adapter as ancestor
    require(PRIVATE_KEY not in sys.modules, 'Live secondary adapter already loaded')
    spec = importlib.util.spec_from_file_location(PRIVATE_KEY, Path(ancestor.__file__))
    require(spec is not None and spec.loader is not None, 'Combined adapter source absent')
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_KEY] = module
    try:
        spec.loader.exec_module(module)
        # The runner validates full controls and event/time scope before use.
        # Constructors call this guard again, after source generation.
        module.validate_input = lambda data: validate_scope(
            data, start=float(data.get('time', 0.)), end=float(data['endtime']),
            interval=float(data['output_dt']), dtmax=float(data['dtmax']),
            launch_mode='plain', integrator='ssp43') is not None
        yield module
    finally:
        sys.modules.pop(PRIVATE_KEY, None)
