"""Host admission for surveyed mode-9 stage and critical inlet choices.

This module authenticates native zero-step records and checks the common
fixed-grid SSPRK33 envelope. It does not claim a coupled CUDA implementation:
the stage/control-0 and critical/control-2 bridge must still be composed with
the resident f-wave, sediment, and mode-4 outlet worker.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np


VARIANT = 'live_mode9_section_choice_ssp33'
STAGE_SLOTS = (0, 2, 1)  # SSPRK33 Euler clocks t, t+dt, t+dt/2.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_case(folder):
    """Merge the authenticated sidecar without the control-1 fixture guard."""
    from braids_support.gpu_fwave.coupled_reference import read_records, read_snapshot
    from braids_support.gpu_fwave.boundary_schedule_oracle import validate_arrays

    folder = Path(folder)
    data = read_snapshot(folder)
    extra = read_records(folder / 'normal_metadata.bin')
    validate_arrays(extra)
    for key, value in extra.items():
        if key in data and not np.array_equal(data[key], value):
            raise ValueError('Mode-9 section duplicate export differs: ' + key)
        data[key] = value
    nb = int(data['nb'])
    for key in ('section_seed_node', 'boundary_seed_position'):
        raw = np.asarray(data['native_' + key])
        require(raw.dtype == np.int32 and raw.shape == (nb,) and np.all(raw >= 0),
                'Invalid native section seed: ' + key)
        data[key] = np.ascontiguousarray(raw - 1)
    return data


def validate_boundaries(data):
    """Validate actual section metadata and detach disjoint inlet/outlet geometry.

    This checks topology independently of the run envelope, so constructors do
    not need to replace the native control codes to reuse a boundary operator.
    """
    from braids_support.gpu_fwave import live_normal_trace as trace
    from braids_support.gpu_fwave import live_critical_section as critical
    from braids_support.gpu_fwave import outlet_boundary as outlet

    def vector(key, dtype, shape=None):
        value = np.asarray(data[key])
        require(value.dtype == dtype and value.ndim == 1 and
                value.flags.c_contiguous,
                'Canonical section vector required: ' + key)
        if shape is not None:
            require(value.shape == shape, 'Section vector shape differs: ' + key)
        return value

    mode = vector('mode', np.int32)
    nb = len(mode)
    require(np.array_equal(mode, [9, 4]),
            'Inlet choices require one surveyed mode9 inlet and mode4 outlet')
    nc = int(data['nc'])
    require(nc > 0 and np.asarray(data['q']).shape == (3, nc),
            'Section hydraulic state dimensions differ')
    owner = vector('owner', np.int32)
    nf = len(owner)
    neighbour = vector('neighbour', np.int32, (nf,))
    bid = vector('bid', np.int32, (nf,))
    offsets = vector('boundary_offset', np.int32, (nb + 1,))
    faces = vector('boundary_faces', np.int32)
    require(offsets[0] == 0 and offsets[-1] == len(faces) and
            np.all(np.diff(offsets.astype(np.int64)) > 0),
            'Section CSR requires nonempty ordered intervals')
    require(np.all((owner >= 0) & (owner < nc)) and
            np.all((faces >= 0) & (faces < nf)) and
            len(np.unique(faces)) == len(faces) and
            np.all((bid >= -1) & (bid < nb)) and
            np.array_equal(np.sort(faces), np.flatnonzero(bid >= 0)),
            'Section CSR must cover each assigned exterior face once')
    for b in range(nb):
        selected = faces[offsets[b]:offsets[b + 1]]
        require(np.all(bid[selected] == b) and np.all(neighbour[selected] < 0),
                'Section CSR boundary/exterior identity differs')
    control = vector('section_control', np.int32, (nb,))
    seeds = vector('section_seed_node', np.int32, (nb,))
    positions = vector('boundary_seed_position', np.int32, (nb,))
    slopes = vector('section_energy_slope', np.float64, (nb,))
    require(control[0] in (0, 2) and control[1] == 0 and
            np.isfinite(slopes).all() and
            1 <= seeds[0] < offsets[1] and seeds[1] == -1 and
            (positions[0] == -1 or 0 <= positions[0] < offsets[1]) and
            positions[1] == -1,
            'Section controls or zero-based native seeds differ')
    flag = np.asarray(data['production_section_geometry'])
    require(flag.shape == () and flag.dtype.kind in 'biu' and int(flag) == 1,
            'Genuine production section geometry is required')
    sampled = vector('sampled', np.int32, (nf,))
    expected = np.zeros(nf, np.int32)
    inlet_faces = faces[:offsets[1]]
    expected[inlet_faces] = 1
    require(np.array_equal(sampled, expected),
            'Sampled flags must identify exactly the surveyed inlet')
    geometry = trace.validate_geometry(dict(
        owner=owner, faces=inlet_faces.copy(),
        offsets=np.array([0, len(inlet_faces)], np.int32),
        length=data['length'], normal=data['normal'],
        left_offset=data['boundary_bed_left_offset'],
        right_offset=data['boundary_bed_right_offset'],
        mid_offset=data['boundary_bed_offset'],
        manning=data['manning_cell'], production_section_geometry=True))
    require(len(geometry['manning']) == nc,
            'Section Manning/state dimensions differ')
    for actual, native in (('manning', 'manning_cell'),
                           ('bedoffset', 'boundary_bed_offset')):
        if actual in data:
            require(np.array_equal(data[actual], data[native]),
                    'Coupled/native section geometry differs: ' + actual)
    controls = dict(mode=mode[:1].copy(), section_control=control[:1].copy(),
                    energy_slope=slopes[:1].copy(), seed_node=seeds[:1].copy())
    if control[0] == 2:
        controls = critical.validate_controls(geometry, controls)
    else:
        controls = MappingProxyType({k: trace.freeze(v) for k, v in controls.items()})
    metadata = MappingProxyType({key: trace.freeze(np.asarray(data[key])) for key in
        ('mode', 'section_control', 'section_seed_node', 'boundary_seed_position',
         'boundary_offset', 'boundary_faces', 'bid', 'neighbour', 'sampled',
         'section_energy_slope')})
    inlet = dict(geometry=geometry, controls=controls,
                 global_boundary_ids=trace.freeze(np.array([0], np.int32)),
                 native_metadata=metadata)
    outlet_faces = faces[offsets[1]:].copy()
    host, count = outlet.component.validate_geometry(dict(nc=nc, owner=owner,
        faces=outlet_faces, offsets=np.array([0, len(outlet_faces)], np.int32),
        length=data['length'], normal=data['normal']))
    out = dict(geometry=MappingProxyType(dict(host, nc=count)),
               global_boundary_ids=trace.freeze(np.array([1], np.int32)),
               native_metadata=metadata)
    return dict(inlet=inlet, outlet=out)


def validate_sections(data):
    return validate_boundaries(data)['inlet']


def validate_outlets(data):
    return validate_boundaries(data)['outlet']


def validate_scope(data, *, start, end, interval, dtmax, launch_mode, integrator):
    """Validate real controls before CUDA or output; retain all native values."""
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables

    validate_boundaries(data)
    require(launch_mode == 'plain' and integrator == 'ssp33',
            'Mode-9 inlet choices require explicit plain SSPRK33')
    require(np.isfinite([start, end, interval, dtmax]).all() and
            start >= 0 and end > start and interval > 0 and dtmax > 0,
            'Mode-9 inlet choices require finite positive run controls')
    require(int(data['scheme']) == 6 and int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0,
            'Mode-9 inlet choices require native scheme 6 SSPRK33 and outlet extension 0')
    mode = np.asarray(data['mode'])
    control = np.asarray(data['section_control'])
    require(mode.dtype == np.int32 and control.dtype == np.int32 and
            np.array_equal(mode, [9, 4]) and control.shape == (2,) and
            int(control[0]) in (0, 2) and int(control[1]) == 0,
            'Mode-9 inlet choices require stage/control0 or critical/control2 and mode4 outlet')
    nc, nf = int(data['nc']), int(data['nf'])
    offsets = np.asarray(data['boundary_offset'])
    faces = np.asarray(data['boundary_faces'])
    sampled = np.asarray(data['sampled'])
    require(offsets.dtype == faces.dtype == sampled.dtype == np.int32 and
            offsets.shape == (3,) and offsets[0] == 0 and
            offsets[-1] == len(faces) and np.all(np.diff(offsets) > 0) and
            np.all((faces >= 0) & (faces < nf)) and len(np.unique(faces)) == len(faces),
            'Mode-9 inlet choices require canonical nonempty disjoint section faces')
    expected = np.zeros(nf, np.int32)
    expected[faces[:offsets[1]]] = 1
    require(sampled.shape == (nf,) and np.array_equal(sampled, expected) and
            bool(data['production_section_geometry']),
            'Mode-9 inlet choices require exactly the inlet surveyed faces')
    seed = np.asarray(data['section_seed_node'])
    position = np.asarray(data['boundary_seed_position'])
    require(seed.shape == position.shape == (2,) and
            1 <= seed[0] < offsets[1] and seed[1] == -1 and
            0 <= position[0] < offsets[1] and position[1] == -1,
            'Mode-9 inlet choices require genuine zero-based section seeds')
    require(int(data['nfrac']) == 2 and int(data['nbed_layers']) == 2 and
            int(data['bedlaw']) == 2 and int(data['mpm_friction_mode']) == 1 and
            int(data['exchange_mode']) == 0 and int(data['secondary_flow_model']) == 0 and
            int(data['nstate']) == 9 and int(data['nledger']) == 5,
            'Mode-9 inlet choices require two-class/two-layer MPM without exchange or secondary flow')
    require(float(data['morphological_factor']) == 1. and
            float(data['horizontal_viscosity']) == 0. and
            not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']),
            'Mode-9 inlet choices exclude MorFac, viscosity, and suspended concentration')
    for key in ('amr_enabled', 'bedload_adaptation', 'bedform_roughness',
                'bank_failure', 'bank_avalanche', 'general_avalanche',
                'bank_front_tracking', 'viscosity_equilibrated_inlet'):
        require(not np.any(data.get(key, 0)), 'Mode-9 inlet choices exclude ' + key)
    q = np.asarray(data['q'])
    require(q.shape == (3, nc) and np.isfinite(q).all() and
            np.all(q[0] >= 0) and np.any(q[0] > float(data['dry_tolerance'])),
            'Mode-9 inlet choices require finite initial hydraulics and a wet cell')
    require(not np.any(q[1:, q[0] <= float(data['dry_tolerance'])]),
            'Mode-9 inlet choices require zero initial dry-cell momentum')
    # Reuse the complete numerical table validation with only its component
    # composition policy detached. The real section flags above stay intact.
    tables = validate_tables(dict(data, section_control=np.zeros(2, np.int32),
                                  sampled=np.zeros(nf, np.int32)))
    count = int(data['nhydro_series'][0])
    require(count >= 2 and not int(data['nhydro_series'][1]) and
            np.array_equal(np.asarray(data['nfraction_series'])[:, 0], [count, count]) and
            not np.any(np.asarray(data['nfraction_series'])[:, 1]),
            'Mode-9 inlet choices require one common Q and two-class feed schedule')
    times = tables['hydro_series_time'][:count, 0]
    require(times[0] <= start and times[-1] >= end and
            np.array_equal(tables['fraction_series_time'][:count, :, 0],
                           np.repeat(times[:, None], 2, axis=1)) and
            np.all(tables['hydro_series_q'][:count, 0] > 0),
            'Mode-9 inlet choices require positive Q and class feed on a covering common clock')
    if control[0] == 0:
        require(np.isfinite(tables['hydro_series_stage'][:count, 0]).all(),
                'Prescribed mode-9 stage must be finite')
    else:
        require(not np.any(tables['hydro_series_stage'][:count, 0]),
                'Critical mode-9 stage placeholder must remain zero')
    events = BoundaryEvents.from_tables(data)
    require(np.array_equal(events.knots, times),
            'Mode-9 inlet choices require one active forcing event clock')
    return events
