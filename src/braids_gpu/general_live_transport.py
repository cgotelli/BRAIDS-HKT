"""Ordinary fixed-grid suspended transport with current surveyed boundaries.

This bridge composes existing resident operators under an explicitly selected
SSPRK(3,3) or SSPRK(4,3) method.  It copies CUDA verbatim and authenticates its
generated Python constructor graph. It does not select a validation fixture, a
CPU fallback, or a different closure.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np

from braids_core.windows_paths import filesystem_path
from braids_core.gpu_capabilities import has_suspended_transport
from . import g1_inlet_solver as inlet_bridge
from . import live_inlet_modes as inlet
from .implicit_midpoint_fwave import (
    TOKEN as IMPLICIT_TOKEN,
    LABEL as IMPLICIT_LABEL,
    ImplicitMidpointController,
    method_identity as implicit_method_identity,
    coupling_identity as implicit_coupling_identity,
    validate_scope as validate_implicit_scope,
)
from .m1_profile_transport import (
    archive_profile_sources,
    make_law11_transport,
    HIDING_PUBLIC_ADMISSION_ID,
    SCALAR_PUBLIC_ADMISSION_ID,
    PUBLIC_ADMISSION_KEY,
    SECONDARY_EXECUTION_KEY,
    PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    validate_scalar_capacity_prepared,
    validate_law11_prepared,
    validate_secondary_composition,
    validate_prognostic_secondary_composition,
    validate_bed_slope_composition,
    verify_profile_sources,
)
from .m1_rp_taub_transport import (
    make_rp_taub_transport,
    method_identity as rp_taub_method_identity,
    validate_rp_taub_prepared,
)
from .law11_soursin_exchange import (
    EXCHANGE_MODE as FLOW2DH_EXCHANGE_MODE,
    ResidentLaw11SoursinExchange,
)
from .flow2dh_boundary_diffusion import ResidentFlow2DHBoundaryDiffusion
from braids_core.sediment.prognostic_secondary_execution import (
    validate_prognostic_secondary_execution_document,
)


ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT/'braids_support/gpu_fwave'
VARIANT = 'live_normal_critical_transport_ssp33'
NORMAL_ROOT_RULE = 'reflective_endpoint_minimum_and_flat_bracket_v1'
NORMAL_TRACE_RULE = 'root_linear_face_moments_owner_hydrostatic_path_v1'
NATIVE_NORMAL_FACE_POLICY = 'native_mode9_midpoint_depth_uniform_speed_sampled_pressure_v1'
PROGNOSTIC_CONSTRUCTOR_SHADOW = '_q3h3_prognostic_constructor_shadow'
STAGE_SLOTS = (0, 2, 1)
STAGE_FRACTIONS = (0., 1., .5)
INTEGRATORS = MappingProxyType({
    'ssp33': MappingProxyType(dict(label='SSPRK33', euler_stages=3,
                                   stage_slots=STAGE_SLOTS,
                                   stage_fractions=STAGE_FRACTIONS)),
    'ssp43': MappingProxyType(dict(label='SSPRK43', euler_stages=4,
                                   stage_slots=(0, 1, 2, 1),
                                   stage_fractions=(0., .5, 1., .5))),
    IMPLICIT_TOKEN: MappingProxyType(dict(label=IMPLICIT_LABEL, euler_stages=1,
                                          stage_slots=(1,),
                                          stage_fractions=(.5,))),
})
AVALANCHE_CONTROLS = ('general_avalanche_wet_slope', 'general_avalanche_dry_slope',
    'general_avalanche_time', 'general_avalanche_wet_depth_switch',
    'general_avalanche_relaxation_length')
AVALANCHE_ORDER = 'each_euler_after_exner_before_rebalance'
PARENT_AVALANCHE_ORDER = 'parent_post_reflux_once_per_macro_v1'
AVALANCHE_PLACEMENTS = (AVALANCHE_ORDER, PARENT_AVALANCHE_ORDER)
VISCOSITY_ORDER = 'after_second_relaxation_before_pre_commit'
PARENT_VISCOSITY_ORDER = 'parent_post_reflux_ssprk2_once_per_macro_v1'
VISCOSITY_PLACEMENTS = (VISCOSITY_ORDER, PARENT_VISCOSITY_ORDER)
VISCOSITY_INTEGRATOR = 'SSPRK2_subcycled_area_weighted_velocity_L2_limited'
VISCOSITY_LIMITED_NORM = 'area_weighted_velocity_L2'
BEDFORM_ORDER = 'ssprk33_stage_capacity_and_pre_friction'
RECURSIVE_BEDFORM_ORDER = 'level_ssprk33_stage_capacity_and_pre_friction_v1'
BEDFORM_PLACEMENTS = (BEDFORM_ORDER, RECURSIVE_BEDFORM_ORDER)
RECURSIVE_BEDFORM_METHOD = MappingProxyType(dict(
    policy=RECURSIVE_BEDFORM_ORDER,
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False))
PRIVATE = {name: '_glt_' + name[:-3] for name in (
    'boundary_schedule.py', 'boundaries.py', 'sediment_transport.py',
    'coupled_solver.py', 'outlet_boundary.py', 'live_normal_boundary.py',
    'g1_live_inlet_boundary.py')}
PYTHON_HELPERS = ('general_live_transport.py', 'integrated_normal_trace.py',
    'g1_inlet_solver.py', 'live_inlet_modes.py', 'g1_live_inlet_boundary.py',
    'a1_full_array_transaction.py', 'sample_discharge_boundary.py',
    'implicit_midpoint_fwave.py', 'forcing_profile.py',
    'amr_typed_roughness.py',
    'hydraulic_roughness.py', 'typed_normal_section.py',
    'viscous_inlet.py', 'viscous_inlet_boundary.py',
    'm1_rp_taub_transport.py', 'm1_profile_transport.py',
    'm1_capacity_dispatch.py', 'scalar_capacity_runtime.py',
    'hiding_exposure_runtime.py',
    'flow2dh_prognostic_secondary.py',
    'channel_composition.py', 'sediment_skin_shear.py')
def SHA(path):
    from braids_core.sediment.river_runtime_lease import digest
    return digest(filesystem_path(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _typed_forcing_identity(data):
    """Resolve local fixed-grid or topology-invariant authenticated identity."""
    from .forcing_profile import BINDING_FIELD, forcing_identity
    if BINDING_FIELD not in data:
        return None
    from .amr_typed_roughness import (
        BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD, amr_forcing_identity)
    if AMR_ROUGHNESS_BINDING_FIELD in data:
        return amr_forcing_identity(data)
    return forcing_identity(data)


def integration_identity(integrator, launch_mode='plain'):
    """Return the exact selected GPU time-method contract."""
    require(type(integrator) is str and integrator in INTEGRATORS,
            'General live transport requires an explicit supported integrator')
    require(launch_mode == 'plain', 'General live transport requires explicit plain launch')
    spec = INTEGRATORS[integrator]
    identity = dict(integrator=spec['label'], euler_stages=spec['euler_stages'],
                    launch_mode=launch_mode, stage_slots=list(spec['stage_slots']),
                    stage_fractions=list(spec['stage_fractions']))
    return identity


def normal_face_policy(data, integrator):
    """Resolve the explicit surveyed-face closure bound into source identity.

    Only the schema-2 law-11 implicit route is the CPU-v51 parity route.  All
    historical explicit and schema-1 runs retain the qualified integrated
    subface closure; this is intentionally not a mesh- or backend heuristic.
    """
    require(type(integrator) is str and integrator in INTEGRATORS,
            'Normal face policy requires an explicit supported integrator')
    if (integrator == IMPLICIT_TOKEN and int(data.get('bedlaw', 0)) == 11 and
            int(data.get('m1_profile_runtime_schema', 1)) == 2):
        return NATIVE_NORMAL_FACE_POLICY
    return NORMAL_TRACE_RULE


def _integer(data, key, allowed):
    value = np.asarray(data.get(key))
    require(value.shape == () and value.dtype.kind in 'iu' and int(value) in allowed,
            'General live transport invalid integer control: ' + key)
    return int(value)


def validate_boundaries(data):
    from braids_support.gpu_fwave.boundaries import validate_scope as stock
    from braids_support.gpu_fwave.hydraulic_pipeline import boundary_data
    modes = np.asarray(data['mode'])
    if modes.ndim == 1 and modes.dtype == np.int32 and len(modes) == 0:
        # Closed-wall domains have no trace on which to apply an inlet/outlet
        # closure. Active sediment/secondary routes still require their exact
        # outer contracts (M2-C or Q3-H3); plain inert hydraulic carriers have
        # a separate narrow admission below. Bare model=2 still fails closed.
        shadow = data.get(PROGNOSTIC_CONSTRUCTOR_SHADOW)
        if shadow is None:
            prognostic = validate_prognostic_secondary_composition(data)
        else:
            # Law-11 authenticates the public Q3-H3 deck before it creates
            # this private allocator-only shadow.  The generic base must see
            # model=0 so it cannot allocate a second secondary-flow owner,
            # while closed-wall validation still needs a cryptographic link
            # to the already authenticated Q3-H3 identity.
            prognostic = validate_prognostic_secondary_execution_document(
                data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
            require(
                isinstance(shadow, str) and
                shadow == prognostic['fingerprint'] and
                int(data.get('secondary_flow_model', -1)) == 0 and
                int(data.get('secondary_flow', -1)) == 0 and
                int(data.get('secondary_flow_transport_direction', -1)) == 0,
                'Closed-wall Q3-H3 constructor shadow is not authenticated')
        # Inert fixed-grid hydraulic carriers also need genuine closed walls.
        # They have no native sediment/secondary/redistribution owner; an
        # enclosing solver may own those operations under its own identity.
        # Do not route an ordinary closed hydraulic case through law-11 AMR.
        inert_closed = (
            all(int(data.get(k, -1)) == 0 for k in
                ('bedlaw','exchange_mode','secondary_flow_model','secondary_flow',
                 'bed_slope','general_avalanche','bank_failure','bank_toe_erosion_mode',
                 'bedload_adaptation','bedform_roughness','amr_enabled')) and
            int(data.get('nfrac', -1)) == 1 and int(data.get('nbed_layers', -1)) == 1 and
            np.asarray(data.get('state')).shape == (5, int(data.get('nc', -1))) and
            not np.any(np.asarray(data['state'])[3]) and
            not np.any(data.get('secondary_mass', 0)))
        if prognostic is None and not inert_closed:
            from .implicit_suspension_amr import contract_from_data
            contract_from_data(data)
        controls = np.asarray(data['section_control'])
        offsets = np.asarray(data['boundary_offset'])
        faces = np.asarray(data['boundary_faces'])
        require(controls.shape == (0,) and controls.dtype == np.int32 and
                offsets.shape == (1,) and offsets.dtype == np.int32 and
                offsets[0] == 0 and faces.shape == (0,) and
                faces.dtype == np.int32 and
                np.all(np.asarray(data['bid']) == -1) and
                all(np.asarray(data[key]).shape == (0,) for key in
                    ('section_seed_node', 'boundary_seed_position')),
                'Closed-wall boundary inventory differs')
        result = dict(outlet=None,
                      prognostic_secondary_closed_wall=prognostic is not None)
        if inert_closed: result['inert_hydraulic_closed_wall'] = True
        return result
    require(modes.ndim == 1 and modes.dtype == np.int32 and len(modes) > 0 and
            np.isin(modes, (1, 2, 4, 6, 8, 9)).all(),
            'General live transport supports explicit boundary modes1/2/4/6/8/9 only')
    controls = np.asarray(data['section_control'])
    if np.any(modes == 9):
        require(modes.shape == (2,) and modes[0] == 9 and modes[1] in (2, 4) and
                controls.shape == (2,) and controls.dtype == np.int32 and
                controls[0] in (0, 1, 2) and controls[1] == 0,
                'Surveyed transport requires mode9/control0/1/2 and stage2 or critical4 outlet')
        if controls[0] == 1:
            from braids_support.gpu_fwave.normal_outlet_boundary import _VALIDATE_NORMAL
            normal = _VALIDATE_NORMAL(data)
            for key in ('section_seed_node', 'boundary_seed_position'):
                require(int(np.asarray(data[key])[1]) == -1, 'Outlet section seed must be absent')
            return dict(normal=normal, outlet=_outlet_geometry(data))
        # Reuse the full qualified surveyed geometry validator in a detached
        # outlet-only validation view. Actual mode2 and all executed arrays stay
        # unchanged; the returned inlet geometry contains no outlet operation.
        view = dict(data, mode=np.array([9, 4], np.int32))
        result = inlet.validate_boundaries(view)
        result['outlet'] = _outlet_geometry(data)
        from braids_support.gpu_fwave.live_normal_trace import freeze
        result['inlet']['native_metadata'] = MappingProxyType({
            key: freeze(np.asarray(data[key])) for key in result['inlet']['native_metadata']})
        return result
    require(controls.shape == modes.shape and controls.dtype == np.int32 and not controls.any(),
            'Stock boundaries cannot contain surveyed section controls')
    sampled_legacy = bool(np.any(data['sampled']))
    if sampled_legacy:
        from .sample_discharge_boundary import validate_geometry
        validate_geometry(data)
    else:
        for key in ('section_seed_node', 'boundary_seed_position'):
            require(np.shape(data[key]) == modes.shape and np.all(np.asarray(data[key]) == -1),
                    'Stock boundary seeds must be absent')
    view = boundary_data(data)
    # Mode4's geometry/trace is separately checked and evaluated below; the
    # base validator is used only for the common stock arrays and topology.
    view['mode'] = np.where(modes == 4, 2, modes).astype(np.int32)
    view['horizontal_viscosity'] = 0.
    if sampled_legacy:
        # Only detached common-topology braids_support. Executed sampled geometry
        # is independently checked above and consumed by its native-law helper.
        view['sampled'] = np.zeros_like(data['sampled'])
        view['bedoffset'] = np.zeros_like(data['bedoffset'])
    stock(view, allow_scheduled=True, allow_bedform_mode1=bool(np.any(modes == 1)))
    return dict(outlet=_outlet_geometry(data))


def _outlet_geometry(data):
    from braids_support.gpu_fwave import outlet_boundary as outlet
    selected = np.flatnonzero(np.asarray(data['mode']) == 4).astype(np.int32)
    if not len(selected):
        return None
    offsets, faces = np.asarray(data['boundary_offset']), np.asarray(data['boundary_faces'])
    counts = np.array([offsets[b+1]-offsets[b] for b in selected], np.int64)
    require(np.all(counts > 0), 'Critical outlet must have physical faces')
    compact = np.concatenate([faces[offsets[b]:offsets[b+1]] for b in selected]).astype(np.int32)
    host, nc = outlet.component.validate_geometry(dict(nc=int(data['nc']), owner=data['owner'],
        faces=compact, offsets=np.r_[0, np.cumsum(counts)].astype(np.int32),
        length=data['length'], normal=data['normal']))
    return dict(geometry=MappingProxyType(dict(host, nc=nc)), global_boundary_ids=outlet.freeze(selected),
        native_metadata=MappingProxyType({key:outlet.freeze(np.asarray(data[key])) for key in
            ('mode', 'section_control', 'sampled', 'boundary_offset', 'boundary_faces', 'bid', 'neighbour')}))


def validate_sections(data):
    return validate_boundaries(data)['normal']


def validate_outlets(data):
    result = validate_boundaries(data)['outlet']
    require(result is not None, 'Explicit critical outlet required for critical component')
    return result


def suspended_enabled(data):
    initial = data.get('suspended_mass', ())
    if 'state' in data and 'suspended_indices' in data:
        initial = (initial, np.asarray(data['state'])[np.asarray(data['suspended_indices'], dtype=int)])
    return has_suspended_transport(data['exchange_mode'], initial,
        data.get('fraction_concentration', ()), data.get('fraction_series_concentration', ()),
        explicit_active=bool(data.get('suspended_transport_active', False)))


def recursive_bedform_envelope(data):
    """Recognize the one legacy mode-1/mode-2 bedform AMR envelope.

    This predicate intentionally admits both members of the controlled pair:
    feedback enabled and feedback disabled.  It does not enable the closure;
    the explicit ``bedform_roughness`` flag remains the sole switch for that.
    Keeping the complete predicate shared with the AMR selector prevents a
    boundary layout from being admitted by one recursive owner but rejected by
    the other.
    """
    try:
        modes = np.asarray(data['mode'])
        viscosity = float(np.asarray(data['horizontal_viscosity']))
        factors = [float(np.asarray(data[name])) for name in
                   ('morfac', 'morphological_factor') if name in data]
        forcing = np.asarray(data['forcing'])
        inlet = np.flatnonzero(modes == 1)
        outlet = np.flatnonzero(modes == 2)
        forcing_ok = bool(
            inlet.size == 1 and outlet.size == 1 and
            forcing.shape in ((2, 2), (6, 2)) and
            forcing.dtype.kind in 'fiu' and np.isfinite(forcing).all() and
            forcing[0, inlet[0]] > 0. and forcing[0, outlet[0]] == 0. and
            (forcing.shape[0] == 2 or
             (not np.any(forcing[2:4]) and np.all(forcing[4] > 0.) and
              not np.any(forcing[5]))))

        def zero(name):
            return not bool(np.any(np.asarray(data.get(name, 0))))

        return bool(
            modes.shape == (2,) and modes.dtype.kind in 'iu' and
            sorted(map(int, modes)) == [1, 2] and
            forcing_ok and
            int(np.asarray(data['bedlaw'])) == 2 and
            int(np.asarray(data['mpm_friction_mode'])) == 1 and
            int(np.asarray(data['bed_slope'])) == 0 and
            int(np.asarray(data['hiding_exposure'])) == 0 and
            int(np.asarray(data['exchange_mode'])) == 0 and
            int(np.asarray(data['secondary_flow_model'])) == 0 and
            int(np.asarray(data['bedform_roughness'])) in (0, 1) and
            int(np.asarray(data['nbed_layers'])) == 2 and
            int(np.asarray(data['nsubstrate_layers'])) == 1 and
            np.isfinite(viscosity) and viscosity == 0. and
            len(factors) > 0 and np.isfinite(factors).all() and
            all(value == 1. for value in factors) and
            zero('section_control') and zero('sampled') and
            zero('bedoffset') and zero('boundary_bed_offset') and
            zero('boundary_bed_left_offset') and
            zero('boundary_bed_right_offset') and
            zero('nseries') and zero('nhydro_series') and
            zero('nfraction_series') and zero('bedload_adaptation') and
            zero('general_avalanche') and zero('secondary_flow') and
            zero('secondary_flow_transport_direction') and
            zero('horizontal_viscosity_mode') and
            zero('horizontal_viscosity_geometry') and
            zero('morphological_acceleration') and zero('nmorfac') and
            zero('bank_failure') and zero('bank_toe_erosion_mode') and
            zero('bank_avalanche') and zero('bank_front_tracking') and
            zero('bank_erosion_fraction') and
            zero('production_section_geometry') and
            zero('viscosity_equilibrated_inlet') and
            not suspended_enabled(data))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def read_records(folder):
    """Read the exact declared native state/geometry inventory, without CUDA."""
    from braids_support.gpu_fwave.coupled_reference import read_records as records, read_snapshot
    folder = filesystem_path(folder).resolve()
    require(folder.is_dir(), 'General live transport prepared folder missing')
    require((folder/'coupled_input.bin').is_file() and not (folder/'coupled_input.bin').is_symlink(),
            'General live transport linked/missing native state')
    header = records(folder/'coupled_input.bin')
    viscosity = np.asarray(header.get('horizontal_viscosity'))
    require(viscosity.shape == () and viscosity.dtype.kind in 'fiu' and
            np.isfinite(viscosity) and float(viscosity) >= 0.,
            'General live transport invalid horizontal viscosity')
    expected = {'coupled_input.bin', 'normal_metadata.bin'}
    if float(viscosity) > 0.:
        expected.add('coupled_diffusion.bin')
    if _integer(header, 'general_avalanche', (0, 1)):
        expected.add('avalanche_metadata.bin')
    require({p.name for p in folder.glob('*.bin') if p.name != 'case.data.bin'} == expected,
            'General live transport native sidecar inventory differs')
    require(all((folder/name).is_file() and not (folder/name).is_symlink()
                for name in expected), 'General live transport linked/missing native records')
    data = read_snapshot(folder)
    for name in ('normal_metadata.bin', 'avalanche_metadata.bin'):
        if name not in expected:
            continue
        for key, value in records(folder/name).items():
            require(key not in data or np.array_equal(data[key], value),
                    'General live transport duplicate native record differs: ' + key)
            data[key] = value
    for key in ('section_seed_node', 'boundary_seed_position'):
        value = np.asarray(data['native_' + key])
        require(value.dtype == np.int32 and value.shape == (int(data['nb']),) and
                np.all(value >= 0), 'General live transport invalid native seed: ' + key)
        data[key] = np.ascontiguousarray(value - 1)
    path = folder/'coupled_input.npz'
    if path.exists():
        require(not path.is_symlink(), 'General live transport NPZ is linked')
        with np.load(path, allow_pickle=False) as archive:
            expected = dict(data)
            # The frozen v3/v4 archives predate the explicit descriptive
            # marker added by the v5 snapshot reader.  They remain readable as
            # historical evidence, but no fixed-face arrays are synthesized.
            if (data.get('gaia_fixed_inlet_contract') == 'historical_absent' and
                    'gaia_fixed_inlet_contract' not in archive.files):
                expected.pop('gaia_fixed_inlet_contract')
            require(len(archive.files) == len(set(archive.files)) and
                    set(archive.files) == set(expected) and
                    all(np.array_equal(archive[k], expected[k]) for k in archive.files),
                    'General live transport NPZ differs from native records')
    return data


def _avalanche_controls(data):
    enabled = bool(_integer(data, 'general_avalanche', (0, 1)))
    if enabled:
        for key in AVALANCHE_CONTROLS:
            value = np.asarray(data.get(key))
            require(value.shape == () and value.dtype.kind in 'fiu' and np.isfinite(value) and
                    (float(value) >= 0. if key == 'general_avalanche_wet_depth_switch' else float(value) > 0.),
                    'General live transport invalid avalanche control: ' + key)
        require(not bool(data['bedform_roughness']),
                'Native bedform roughness forbids general avalanche')
    return enabled


def static_bed_enabled(data):
    """The native implicit hydraulic material has one inert, unsplit bed row."""
    return (int(data['nbed_layers']) == 1 and
            int(data['nsubstrate_layers']) == 0 and
            int(data['bedlaw']) == 0 and int(data['exchange_mode']) == 0)


def _validate_static_bed(data):
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope
    require(static_bed_enabled(data) and int(data['nfrac']) == 1 and
            int(data['bedlaw']) == 0 and int(data['exchange_mode']) == 0 and
            int(data['secondary_flow_model']) == 0 and not suspended_enabled(data),
            'Implicit single-layer bed requires hydraulic-only inactive sediment')
    for key in ('general_avalanche', 'bedload_adaptation', 'bedform_roughness',
                'bank_failure', 'bank_toe_erosion_mode', 'bank_avalanche', 'bank_erosion_fraction',
                'secondary_flow_transport_direction', 'mpm_friction_mode', 'bed_slope',
                'hiding_exposure', 'active_layer_mode', 'active_layer_thickness',
                'active_layer_depth_fraction', 'active_layer_minimum_thickness',
                'bed_boundary_mode', 'fraction_bed_input', 'bc_bed_input',
                'series_bed_input', 'fraction_series_bed_input'):
        require(not np.any(data.get(key, 0)), 'Implicit static bed forbids active control: ' + key)
    require(not np.any(np.asarray(data['forcing'])[3]), 'Implicit static bed forbids total bed feed')
    secondary_scope(data)
    require(np.shape(data.get('viscosity_equilibrated_inlet')) == (int(data['nb']),) and
            not np.any(data['viscosity_equilibrated_inlet']),
            'Implicit static bed requires explicit ordinary inlet profile controls')
    if 'secondary_mass' in data:
        require(np.shape(data['secondary_mass']) == (int(data['nc']),) and
                not np.any(data['secondary_mass']), 'Implicit static bed cannot carry secondary mass')
    nc = int(data['nc'])
    for key, shape in (('q', (3, nc)), ('layer_mass', (1, 1, nc)),
                       ('rock', (nc,)), ('manning_cell', (nc,)), ('bulk_fraction', (1,))):
        value = np.asarray(data[key])
        require(value.shape == shape and value.dtype.kind in 'fiu' and np.isfinite(value).all(),
                'Invalid static-bed state/parameter: ' + key)
        if key not in ('rock', 'q'):
            require(np.all(value > 0 if key == 'bulk_fraction' else value >= 0),
                    'Invalid static-bed inventory/parameter: ' + key)
    require(np.all(np.asarray(data['q'])[0] >= 0), 'Invalid static-bed water depth')
    for key in ('grav', 'dry_tolerance', 'rho', 'dx', 'dy', 'dtmax', 'cfl_target'):
        value = np.asarray(data[key])
        require(value.shape == () and value.dtype.kind in 'fiu' and np.isfinite(value) and float(value) > 0,
                'Invalid static-bed hydraulic control: ' + key)


class ResidentStaticBed:
    """Preserve the real native inert bed row; no fictitious active/substrate bed.

    Hydraulic-only execution still needs bed elevation and a finite-state
    status. Both are cell-parallel device expressions, and rebalance performs
    no material exchange. All storage is allocated before trial snapshots.
    """
    def __init__(self, data, **kwargs):
        import cupy as cp
        _validate_static_bed(data)
        self.cp, self.nc, self.ng, self.ns = cp, int(data['nc']), 1, 0
        self.config = {'bulk': np.asarray(data['bulk_fraction']).copy()}
        self.bulk = cp.asarray(self.config['bulk'], dtype=cp.float64)
        self.geometry = {key:cp.asarray(np.ascontiguousarray(data[source]), dtype=cp.float64)
                         for key, source in (('bc_hydraulic', 'bc_hydraulic'), ('feed', 'fraction_bed_input'))}
        self.bed = cp.zeros(self.nc, cp.float64)
        self.status = cp.zeros(self.nc, cp.int32)
        self.coupled_mpm = False
        self.coupled_gaia = False

    def _layers(self, layers):
        require(isinstance(layers, self.cp.ndarray) and layers.shape == (1, 1, self.nc) and
                layers.dtype == self.cp.float64 and layers.flags.c_contiguous,
                'Implicit static bed requires its original resident single-layer shape')

    def bed_elevation(self, layers, rock, out=None):
        self._layers(layers)
        out = self.bed if out is None else out
        require(isinstance(rock, self.cp.ndarray) and rock.shape == (self.nc,) and
                rock.dtype == self.cp.float64 and rock.flags.c_contiguous and
                isinstance(out, self.cp.ndarray) and out.shape == (self.nc,) and
                out.dtype == self.cp.float64 and out.flags.c_contiguous,
                'Implicit static bed elevation storage differs')
        self.cp.divide(layers[0, 0], self.bulk[0], out=out)
        self.cp.add(out, rock, out=out)
        return out

    def rebalance(self, layers):
        self._layers(layers)
        self.cp.copyto(self.status, self.cp.where(
            self.cp.isfinite(layers[0, 0]) & (layers[0, 0] >= 0),
            self.cp.int32(0), self.cp.int32(1)))
        return self.status


def _exchange_controls(data, nc, ng):
    """Host equivalent of ResidentNoncohesiveExchange's physical preflight."""
    exchange = int(np.asarray(data['exchange_mode']))
    # Legacy modes retain their supplied-velocity closure.  Mode 4 alone uses
    # the authenticated SettleFrm=2 total-concentration hindering path.
    _integer(data, 'settling_mode',
             (1,) if exchange == FLOW2DH_EXCHANGE_MODE else (0,))
    _integer(data, 'active_layer_mode', (0,))
    # Ordinary v49 preserves the historical one-fraction, one-bed-row CPU
    # layout.  In the authenticated M2-C route that row is the physical
    # active reservoir itself; there is no fictitious substrate and therefore
    # no positive active-layer target.  All other exchange routes retain the
    # explicit layered-deck requirement.
    single_m2c_reservoir = (
        exchange == FLOW2DH_EXCHANGE_MODE and
        int(data.get('nbed_layers', -1)) == 1 and
        int(data.get('nsubstrate_layers', -1)) == 0)
    if single_m2c_reservoir:
        from .implicit_suspension_amr import contract_from_data
        contract_from_data(data)
    require(int(data['deck_version']) >= 13 or single_m2c_reservoir,
            'General live transport requires explicit active/substrate layers')
    kinds = np.asarray(data['sediment_type_fraction'])
    require(kinds.shape == (ng,) and kinds.dtype.kind in 'iu' and not np.any(kinds),
            'General live transport excludes cohesive fractions')
    parameters = {}
    positive = ('bulk_fraction', 'd50_fraction', 'd90_fraction', 'rhos_fraction',
                'z0_fraction', 'reference_height_fraction', 'dstar_fraction', 'taucr_fraction')
    nonnegative = ('manning_cell', 'sedthr_fraction') + (
        () if exchange == FLOW2DH_EXCHANGE_MODE else ('ws_fraction',))
    for key in positive + nonnegative:
        value = np.asarray(data[key])
        require(value.shape == ((nc,) if key == 'manning_cell' else (ng,)) and
                value.dtype.kind in 'fiu' and np.isfinite(value).all() and
                np.all(value > 0 if key in positive else value >= 0),
                'General live transport invalid exchange parameter: ' + key)
        parameters[key] = value
    require(np.all(parameters['d90_fraction'] >= parameters['d50_fraction']) and
            np.all(parameters['rhos_fraction'] > float(data['rho'])),
            'General live transport grain diameter/density differs')
    roughness = np.asarray(data['sediment_roughness_fraction'])
    require(roughness.shape == (ng,) and roughness.dtype.kind in 'iu' and
            np.isin(roughness, (0, 1)).all(),
            'General live transport unsupported sediment roughness')
    for key in ('rho', 'grav', 'dry_tolerance',
                'hindered_settling_reference_concentration'):
        value = np.asarray(data.get(key))
        require(value.shape == () and value.dtype.kind in 'fiu' and
                np.isfinite(value) and float(value) > 0,
                'General live transport positive exchange control required: ' + key)
    active_thickness = np.asarray(data.get('active_layer_thickness'))
    require(active_thickness.shape == () and
            active_thickness.dtype.kind in 'fiu' and
            np.isfinite(active_thickness) and
            ((single_m2c_reservoir and float(active_thickness) == 0.) or
             (not single_m2c_reservoir and float(active_thickness) > 0.)),
            'General live transport active-layer control differs from its layout')
    for key, default in (('active_layer_depth_fraction', 0.),
                         ('active_layer_minimum_thickness', 1.e-6)):
        value = np.asarray(data.get(key, default))
        require(value.shape == () and value.dtype.kind in 'fiu' and
                np.isfinite(value) and float(value) >= 0,
                'General live transport invalid active-layer control: ' + key)


def validate_scope(data, *, start, end, interval, dtmax, launch_mode, integrator):
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.coupled_solver import normalized, validate_scope as base_scope
    from braids_support.gpu_fwave.sediment_transport import configuration, validate_topology

    secondary = _integer(data, 'secondary_flow_model', (0, 1, 2, 3))
    bedlaw = _integer(data, 'bedlaw', (0, 2, 5, 6, 7, 8, 9, 11))
    exchange = _integer(data, 'exchange_mode', (0, 1, 2, FLOW2DH_EXCHANGE_MODE))
    _integer(data, 'hiding_exposure', (0, 2, 3, 4, 5))
    slope = _integer(data, 'bed_slope', (0, 2, 3, 4, 5, 6))
    bedform = bool(_integer(data, 'bedform_roughness', (0, 1)))
    adaptation = bool(_integer(data, 'bedload_adaptation', (0, 1)))
    avalanche = _avalanche_controls(data)
    length = np.asarray(data.get('bedload_adaptation_length_diameters'))
    require(length.shape == () and length.dtype.kind in 'fiu' and np.isfinite(length) and
            (not adaptation or (bedlaw == 2 and float(length) > 0)),
            'Live transport adaptation requires MPM and a finite positive enabled length')
    data = normalized(data)
    public_marker = data.get(PUBLIC_ADMISSION_KEY)
    scalar_foundation = (isinstance(public_marker, dict) and
                         public_marker.get('admission_id') in (
                             SCALAR_PUBLIC_ADMISSION_ID,
                             HIDING_PUBLIC_ADMISSION_ID))
    hiding_foundation = (isinstance(public_marker, dict) and
                         public_marker.get('admission_id') ==
                             HIDING_PUBLIC_ADMISSION_ID)
    scalar_profile = (validate_scalar_capacity_prepared(data)
                      if scalar_foundation else None)
    bed_slope_foundation = (validate_bed_slope_composition(data)
                            if scalar_foundation else None)
    secondary_foundation = validate_secondary_composition(data)
    prognostic_secondary_foundation = \
        validate_prognostic_secondary_composition(data)
    law11_profile = (validate_law11_prepared(data)
                     if bedlaw == 11 and not scalar_foundation else None)
    rp_taub_profile = (validate_rp_taub_prepared(data)
                       if bedlaw in (5, 6, 7) else None)
    boundary_scope = validate_boundaries(data)
    integration_identity(integrator, launch_mode)
    implicit = integrator == IMPLICIT_TOKEN
    if implicit:
        validate_implicit_scope(data)
    if integrator == 'ssp43':
        require(bedlaw == 2,
                'General live transport SSPRK43 currently requires MPM bedlaw 2')
    if bedlaw == 8 and not scalar_foundation:
        require(integrator in ('ssp33', IMPLICIT_TOKEN) and slope in (0, 2, 3, 5) and
                int(data['mpm_friction_mode']) == 0 and
                int(data['hiding_exposure']) == 0 and
                np.all(np.asarray(data['manning_cell']) > 0),
                'Engelund-Hansen requires SSPRK33 or implicit f-wave, positive Manning and hiding OFF')
        if slope == 3:
            require(float(data['bed_slope_alfabs']) >= 0. and
                    float(data['bed_slope_alfabn']) > 0. and
                    float(data['bed_friction_angle_deg']) > 0.,
                    'Koch-Flokstra requires AlfaBs>=0, AShld>0, BShld>0')
    if bedlaw == 9:
        require(integrator == 'ssp33' and slope == 4 and secondary == 3 and
                int(data['mpm_friction_mode']) == 0 and
                int(data['hiding_exposure']) == 0 and
                int(data['exchange_mode']) == 0 and
                bool(int(np.asarray(data.get('secondary_flow', 0)))) and
                bool(int(np.asarray(data.get('secondary_flow_transport_direction', 0)))) and
                float(np.asarray(data.get('secondary_flow_momentum_beta', np.nan))) == 0. and
                float(np.asarray(data.get('horizontal_viscosity', np.nan))) == 0. and
                not bool(int(np.asarray(data.get('general_avalanche', 0)))) and
                np.all(np.asarray(data['manning_cell']) > 0),
                'TELEMAC-GAIA law 9 requires SSPRK33, positive Manning, bedload-only slope4/model3')
    if scalar_foundation:
        expected_hiding = ((2, 3, 4, 5) if hiding_foundation else (0,))
        require(bedlaw in ((2, 11) if hiding_foundation else (2, 8, 11)) and
                integrator == IMPLICIT_TOKEN and
                slope == (0 if bed_slope_foundation is None else
                           bed_slope_foundation['persistence_code']) and
                secondary == (2 if prognostic_secondary_foundation is not None
                              else 1 if secondary_foundation is not None else 0) and
                exchange == 0 and
                int(data['hiding_exposure']) in expected_hiding and
                (not hiding_foundation or int(data['nfrac']) == 2) and
                np.all(np.asarray(data['manning_cell']) > 0) and
                scalar_profile is not None,
                'Q3-C/Q3-H1/Q3-H2/Q3-H3 scalar profile requires implicit midpoint, '
                'typed positive roughness and its authenticated auxiliary scope')
    elif bedlaw == 11:
        require(integrator == IMPLICIT_TOKEN and slope == 0 and secondary == 0 and
                exchange in (0, FLOW2DH_EXCHANGE_MODE) and
                int(data['hiding_exposure']) == 0 and
                np.all(np.asarray(data['manning_cell']) > 0) and
                law11_profile is not None,
                'Law-11 profile slice requires implicit midpoint, Manning, hiding/slope/secondary OFF, and exchange OFF or 4')
    if bedlaw in (5, 6, 7):
        require(integrator == IMPLICIT_TOKEN and slope == 0 and secondary == 0 and
                exchange == 0 and int(data['hiding_exposure']) == 0 and
                rp_taub_profile is not None,
                'RP_TAUB laws 5/6/7 require authenticated fixed-grid implicit bedload-only forcing')
    if exchange in (1, 2):
        require(integrator == 'ssp33',
                 'General live transport exchange modes 1/2 require SSPRK33')
    if exchange == FLOW2DH_EXCHANGE_MODE:
        require(bedlaw == 11 and integrator == IMPLICIT_TOKEN,
                'Generic FLOW 2DH exchange mode 4 requires law 11 and implicit midpoint')
    if bedform or adaptation or avalanche:
        require(integrator == 'ssp33',
                'Bedform, adaptation and general avalanche require SSPRK33')
    require(np.isfinite([start, end, interval, dtmax]).all() and start >= 0 and
            end > start and interval > 0 and dtmax > 0 and
            end <= float(data['endtime']) and dtmax <= float(data['dtmax']),
            'General live transport clock is outside native export')
    # cpu_time_integrator records the zero-step native initializer.  It does
    # not select the GPU method; the explicit ``integrator`` argument does.
    require(int(data['scheme']) == 6 and int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0 and float(data['morfac']) == 1.,
            'General live transport requires the scheme6 SSPRK33/outlet0 native initializer and MorFac1')
    viscosity = np.asarray(data.get('horizontal_viscosity'))
    require(viscosity.shape == () and viscosity.dtype.kind in 'fiu' and
            np.isfinite(viscosity) and float(viscosity) >= 0.,
            'General live transport invalid horizontal viscosity')
    if float(viscosity) > 0.:
        from braids_support.gpu_fwave.momentum_diffusion import validate_scope as diffusion_scope
        _integer(data, 'horizontal_viscosity_mode', (1,))
        _integer(data, 'horizontal_viscosity_geometry', (1,))
        diffusion_scope(data)
    for key in ('amr_enabled', 'bank_failure',
                'bank_front_tracking', 'bank_toe_erosion_mode', 'bank_avalanche',
                'bank_erosion_fraction', 'nmorfac', 'morphological_acceleration',
                'nseries'):
        require(not np.any(data.get(key, 0)),
                'General live transport unsupported control: ' + key)
    if np.any(data.get('viscosity_equilibrated_inlet', 0)):
        from .viscous_inlet_boundary import validate_viscous_inlet_scope
        validate_viscous_inlet_scope(data, integrator=integrator)
    nc, nf, ng, nl, nb = (int(data[key]) for key in ('nc', 'nf', 'nfrac', 'nbed_layers', 'nb'))
    static_bed = static_bed_enabled(data)
    ns, physical_ledger = 3 + ng*(1+nl), 1 + 2*ng
    prognostic = secondary == 2
    # The v49 FLOW 2DH declaration appends two restart/accounting rows per
    # fraction (advective and horizontal-diffusive boundary exchange).  They
    # are not extra physical state, but they are part of the authenticated
    # native ledger inventory and must survive preparation unchanged.
    m2c_boundary_memory = (exchange == FLOW2DH_EXCHANGE_MODE and
                           'suspended_amr_contract' in data)
    if m2c_boundary_memory:
        from .implicit_suspension_amr import contract_from_data
        contract_from_data(data)
    boundary_memory_ledger = 2*ng if m2c_boundary_memory else 0
    ledger = physical_ledger + int(prognostic) + boundary_memory_ledger
    closed_wall_prognostic = bool(
        boundary_scope.get('prognostic_secondary_closed_wall', False))
    require(nc > 0 and nf > 0 and 1 <= ng <= 32 and 1 <= nl <= 33 and
            (nb >= 0 if (m2c_boundary_memory or closed_wall_prognostic or
                        boundary_scope.get('inert_hydraulic_closed_wall', False))
             else nb >= 1) and
            int(data['nsubstrate_layers']) == nl-1 and
            int(data['secondary_index']) == (ns if prognostic else -1) and
            int(data['secondary_ledger_index']) ==
                (physical_ledger if prognostic else -1) and
            int(data['nstate']) == ns + int(prognostic) and
            int(data['nphysical_ledger']) == physical_ledger and
            int(data['nledger']) == ledger and
            np.shape(data['state']) == (ns + int(prognostic), nc) and
            np.shape(data['ledger']) == (ledger, nb) and
            np.shape(data['q']) == (3, nc) and
            np.shape(data['layer_mass']) == (nl, ng, nc) and
            np.array_equal(data['suspended_indices'], np.arange(3, 3+ng)) and
            np.array_equal(data['bed_indices'], np.arange(3+ng, ns).reshape(nl, ng)),
            'General live transport native inventory layout differs')
    require(not int(data['hiding_exposure']) or ng > 1,
            'General live transport hiding requires multiple fractions')
    q = np.asarray(data['q'])
    # The inert hydraulic carrier also prepares initially empty river domains.
    # It must not manufacture a wet seed cell. Its full static-state checks
    # below still require finite, nonnegative depths and inactive material;
    # transported legacy bundles retain their separate wet-state admission.
    require((static_bed or np.any(q[0] > data['dry'])) and not np.any(q[1:, q[0] <= data['dry']]),
            'General live transport initial wet/dry momentum differs')
    validate_topology(data, nc, nf, nb)
    if bedform:
        from braids_support.gpu_fwave.bedform_roughness import validate_scope as bedform_scope
        require(not suspended_enabled(data), 'Bedform roughness forbids suspended transport')
        bedform_scope(data)
    if exchange:
        _exchange_controls(data, nc, ng)
    else:
        _integer(data, 'settling_mode', (0,))
    # Real physics remains intact. Schedule counts and the independently
    # checked adaptation flag are detached for the stock preflight, whose
    # public scheduled/adaptation variants have older composition envelopes.
    # Actual tables and live geometry are separately validated, then consumed
    # verbatim by the generated resident components below.
    if static_bed:
        _validate_static_bed(data)
    else:
        single_scalar_reservoir = (
            (bedlaw == 11 or scalar_foundation) and
            exchange in (0, FLOW2DH_EXCHANGE_MODE) and
            nl == 1 and int(data['nsubstrate_layers']) == 0)
        if single_scalar_reservoir and exchange == FLOW2DH_EXCHANGE_MODE:
            # ``base_scope`` is the historical SSPRK33 catalogue and requires
            # at least one substrate.  The authenticated v49 M2-C contract is
            # instead a real one-row active reservoir and has already passed
            # its own hydraulic, topology, boundary and exchange validators.
            from .implicit_suspension_amr import contract_from_data
            contract_from_data(data)
        base_data = dict(data, bedload_adaptation=np.int32(0), bedform_roughness=np.int32(0),
            general_avalanche=np.int32(0), nhydro_series=np.zeros(nb, np.int32),
            nfraction_series=np.zeros((ng, nb), np.int32), nseries=np.zeros(nb, np.int32),
            # The frozen base validator predates the composed CUDA inlet.  Its
            # view is inert only after the exact actual bundle above passed;
            # the runtime data and selected operator remain unchanged.
            viscosity_equilibrated_inlet=np.zeros(nb, np.int32))
        if scalar_foundation:
            # The frozen stock validator predates the common Q3-C scalar
            # capacity dispatcher.  Reuse only its topology/stratigraphy
            # checks under an inert MPM shadow after the authenticated Q3-C
            # profile above has checked the actual law and parameters.  The
            # runtime data and selected capacity operator remain unchanged.
            base_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                             exchange_mode=np.int32(0), settling_mode=np.int32(0),
                             hiding_exposure=np.int32(0), secondary_flow_model=np.int32(0),
                             secondary_flow=np.int32(0),
                             secondary_flow_transport_direction=np.int32(0),
                             mpm_friction_mode=np.int32(0))
        elif bedlaw == 11:
            base_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                             exchange_mode=np.int32(0), settling_mode=np.int32(0),
                             hiding_exposure=np.int32(0), secondary_flow_model=np.int32(0),
                             secondary_flow=np.int32(0),
                             secondary_flow_transport_direction=np.int32(0))
        elif bedlaw in (5, 6, 7):
            # Stock topology/stratigraphy validation is reused under an MPM
            # shadow only; the runtime capacity operator remains the selected
            # law and is authenticated separately above.
            base_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                             exchange_mode=np.int32(0), settling_mode=np.int32(0),
                             hiding_exposure=np.int32(0), secondary_flow_model=np.int32(0),
                             secondary_flow=np.int32(0),
                             secondary_flow_transport_direction=np.int32(0))
        if not single_scalar_reservoir:
            base_scope(base_data)
    modes = np.asarray(data['mode'])
    checked_modes = np.where(modes == 9, 6, np.where(modes == 4, 2, modes)).astype(np.int32)
    if not static_bed and not single_scalar_reservoir:
        configured_data = dict(data, mode=checked_modes)
        if scalar_foundation:
            # See the matching base-scope shadow above.  This call validates
            # generic resident geometry only; Q3-C owns the live law.
            configured_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                                   exchange_mode=np.int32(0), settling_mode=np.int32(0),
                                   hiding_exposure=np.int32(0), secondary_flow_model=np.int32(0),
                                   secondary_flow=np.int32(0),
                                   secondary_flow_transport_direction=np.int32(0),
                                   mpm_friction_mode=np.int32(0))
        elif bedlaw == 11:
            configured_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                                   exchange_mode=np.int32(0),
                                   settling_mode=np.int32(0),
                                   hiding_exposure=np.int32(0),
                                   secondary_flow_model=np.int32(0),
                                   secondary_flow=np.int32(0),
                                   secondary_flow_transport_direction=np.int32(0))
        elif bedlaw in (5, 6, 7):
            configured_data.update(bedlaw=np.int32(2), bed_slope=np.int32(0),
                                   exchange_mode=np.int32(0),
                                   settling_mode=np.int32(0),
                                   hiding_exposure=np.int32(0),
                                   secondary_flow_model=np.int32(0),
                                   secondary_flow=np.int32(0),
                                   secondary_flow_transport_direction=np.int32(0))
        configuration(configured_data, external_suspension=suspended_enabled(data),
                      allow_scheduled=True, allow_adaptation=adaptation,
                      allow_general_avalanche=avalanche,
                      allow_bedform=bedform or bool(np.any(modes == 1)))
    tables = validate_tables(dict(data, section_control=np.zeros(nb, np.int32),
                                   sampled=np.zeros(nf, np.int32)))
    for boundary in range(nb):
        hydro_count = int(tables['nhydro_series'][boundary])
        if hydro_count:
            times = tables['hydro_series_time'][:hydro_count, boundary]
            require(times[0] <= start and times[-1] >= end,
                    'General live transport hydraulic table must cover run')
            if modes[boundary] == 9:
                require(np.all(tables['hydro_series_q'][:hydro_count, boundary] > 0),
                        'Surveyed inlet requires positive scheduled Q')
        elif modes[boundary] == 9:
            require(float(tables['forcing'][0, boundary]) > 0,
                    'Surveyed inlet requires positive constant Q')
        for grain in range(ng):
            count = int(tables['nfraction_series'][grain, boundary])
            if count:
                times = tables['fraction_series_time'][:count, grain, boundary]
                require(times[0] <= start and times[-1] >= end,
                        'General live transport class forcing clock must cover run')
    if bedlaw == 0:
        require(secondary == 0, 'No-bedload transport requires secondary OFF')
        for key in ('fraction_bed_input', 'bc_bed_input', 'series_bed_input',
                    'fraction_series_bed_input'):
            require(not np.any(data.get(key, 0)), 'No-bedload transport forbids bed feed: ' + key)
    return BoundaryEvents.from_tables(data)


def _validate_prepared(data, *, integrator='ssp33'):
    return validate_scope(data, start=float(data.get('time', 0.)),
        end=float(data['endtime']), interval=float(data['output_dt']),
        dtmax=float(data['dtmax']), launch_mode='plain', integrator=integrator)


def _transform(name, source, *, integrator='ssp33'):
    spec = INTEGRATORS[integrator]
    if integrator == IMPLICIT_TOKEN and name == 'sediment_transport.py':
        anchor = "if c['bedlaw'] not in (2,3,4,8,9):"
        require(source.count(anchor) == 2,
                'Law-11 profile sediment face-dispatch anchor differs')
        source = source.replace(anchor, "if c['bedlaw'] not in (2,3,4,5,6,7,8,9,11):")
    if integrator == IMPLICIT_TOKEN and name == 'coupled_solver.py':
        anchor = "self.bedload_active=int(data['bedlaw']) in (2,8,9)"
        require(source.count(anchor) == 1,
                'Law-11 profile bedload-active anchor differs')
        source = source.replace(anchor,
            "self.bedload_active=int(data['bedlaw']) in (2,5,6,7,8,9,11)", 1)
    if name == 'live_normal_boundary.py':
        from braids_support.gpu_fwave.normal_outlet_boundary import transform_bridge
        result = transform_bridge(name, source).replace(
            'from .normal_outlet_boundary import validate_sections as validate_actual',
            'from braids_gpu.general_live_transport import validate_sections as validate_actual')
        anchor = ("            self.normal_root=root_component.ResidentNormalSection(\n"
                  "                dict(sections['geometry'],production_section_geometry=True),sections['controls'],inputs['bed'],self._normal_Q,\n"
                  "                source_path=folder/'live_normal_section.cu')")
        replacement = (
            "            from braids_gpu.typed_normal_section import make_resident_normal_section\n"
            "            self.normal_root=make_resident_normal_section(\n"
            "                data,dict(sections['geometry'],production_section_geometry=True),\n"
            "                dict(sections['controls']),inputs['bed'],self._normal_Q,\n"
            "                legacy_type=root_component.ResidentNormalSection,\n"
            "                legacy_source=folder/'live_normal_section.cu',\n"
            "                typed_source=folder/'typed_normal_section.cu',\n"
            "                roughness_source=folder/'hydraulic_roughness.cu')")
        require(result.count(anchor) == 1,
                'Typed normal-section constructor anchor differs')
        return result.replace(anchor, replacement, 1)
    if name == 'g1_live_inlet_boundary.py':
        replacements = {
            'from braids_gpu.live_inlet_modes import validate_boundaries':
                'from braids_gpu.general_live_transport import validate_boundaries',
            'np.array_equal(mode, [9, 4]) and int(control[0]) in (0, 2)':
                'mode.shape == (2,) and mode[0] == 9 and mode[1] in (2, 4) and int(control[0]) in (0, 2)',
            'control=self.inlet_control, mode=9, outlet_mode=4,':
                'control=self.inlet_control, mode=9, outlet_mode=int(self.cp.asnumpy(self.arrays["mode"])[1]),'}
        for before, after in replacements.items():
            require(source.count(before) == 1, 'Live transport inlet bridge anchor differs')
            source = source.replace(before, after, 1)
        if spec['euler_stages'] != 3:
            # The G1 wrapper stores diagnostics for every actual Euler call.
            # Its numerical section/root components are already stage-agnostic;
            # only these record dimensions and order guards were SSPRK33-specific.
            stages = spec['euler_stages']
            replacements = {
                'cp.empty((3, self.nc))': f'cp.empty(({stages}, self.nc))',
                'cp.empty((3, nfaces + self._g1_n))':
                    f'cp.empty(({stages}, nfaces + self._g1_n))',
                '(3, self._g1_n)': f'({stages}, self._g1_n)',
                'cp.empty((3, 3, nfaces))': f'cp.empty(({stages}, 3, nfaces))',
                'cp.zeros(3, dtype=cp.int32)': f'cp.zeros({stages}, dtype=cp.int32)',
                'self._g1_next_stage < 3': f'self._g1_next_stage < {stages}',
                '0 <= index < 3': f'0 <= index < {stages}',
                'three_ssprk33_euler_records=True':
                    f"euler_records={stages}, integrator='{spec['label']}'",
            }
            for before, after in replacements.items():
                require(source.count(before) >= 1,
                        'Live transport inlet stage anchor differs: ' + before)
                source = source.replace(before, after)
        return source
    result = inlet_bridge.transform(name, source)
    result = result.replace('from braids_gpu.live_inlet_modes import validate_boundaries',
                            'from braids_gpu.general_live_transport import validate_boundaries')
    result = result.replace('from braids_gpu.live_inlet_modes import validate_outlets',
                            'from braids_gpu.general_live_transport import validate_outlets')
    if name == 'boundaries.py':
        before = "            raise ValueError('Bedform mode-1 inlet requires unsampled zero-offset faces')"
        require(result.count(before) == 1, 'Sampled legacy boundary validation anchor differs')
        result = result.replace(before,
            '            from braids_gpu.sample_discharge_boundary import validate_geometry\n'
            '            validate_geometry(data)', 1)
    if name == 'sediment_transport.py':
        before = 'allowed_modes=[2,3,4,6,8,9]+([1] if allow_bedform else [])'
        require(result.count(before) == 1, 'Live transport sediment mode1 anchor differs')
        result = result.replace(before, 'allowed_modes=[1,2,4,6,8,9]', 1)
    if name == 'coupled_solver.py':
        for old, new in (('._g1_boundaries', '._glt_boundaries'),
                         ('._g1_sediment_transport', '._glt_sediment_transport')):
            require(result.count(old) == 1, 'Live transport constructor anchor differs: ' + old)
            result = result.replace(old, new, 1)
        anchor = "        stress=secondary.add_stress(a['trace'],a['kind'],b['closed'],b['fl'],b['fr'],b['speed'])"
        require(result.count(anchor) == 1, 'Live transport integrated flux anchor differs')
        result = result.replace(anchor,
            "        if hasattr(self, 'integrated_section_flux'):self.integrated_section_flux(b)\n" + anchor, 1)
        # The base solver now keeps the bedform output and sediment Manning
        # input in independent allocations so recursive rollback has no alias.
        # It also copies the freshly evaluated row at every Euler stage.  Keep
        # these as authenticated transform anchors instead of applying the old
        # generated-source workaround a second time.
        storage = "            if self.sediment.geometry['manning_cell'].shape!=(self.nc,):"
        require(result.count(storage) == 1,
                'Live transport bedform storage anchor differs')
        stage = ("            bedform_result=self.bedform.evaluate(q,state['layers'])\n"
                 "            bedform_status=bedform_result['status']\n"
                 "            cp.copyto(self.sediment.geometry['manning_cell'],\n"
                 "                      bedform_result['properties'][0])")
        require(result.count(stage) == 1,
                'Live transport bedform stage anchor differs')
    return result


def _original(name):
    return ROOT/'braids_gpu'/name if name == 'g1_live_inlet_boundary.py' else SPATIAL/name


def _source_files(origin):
    return [*sorted(origin.glob('*.cu')), *sorted(SPATIAL.glob('*.py')),
            *(ROOT/'braids_gpu'/name for name in PYTHON_HELPERS),
            ROOT/'braids_gpu/implicit_midpoint_fwave.cu',
            ROOT/'braids_gpu/hydraulic_roughness.cu',
            ROOT/'braids_gpu/typed_boundary_roughness.cu',
            ROOT/'braids_gpu/typed_normal_section.cu',
            ROOT/'braids_gpu/viscous_inlet.cu',
            ROOT/'braids_gpu/m1_capacity_dispatch.cu',
            ROOT/'braids_gpu/scalar_capacity_runtime.cu',
            ROOT/'braids_gpu/channel_composition.cu',
            ROOT/'braids_gpu/sediment_skin_shear.cu',
            ROOT/'braids_gpu/implicit_suspension_amr.cu',
            ROOT/'braids_gpu/implicit_suspension_amr.py',
            ROOT/'braids_core/windows_paths.py', ROOT/'braids_core/gpu_capabilities.py',
            ROOT/'braids_core/sediment/river_source_io.py',
            ROOT/'braids_core/sediment/river_runtime_lease.py']


def prepare_sources(folder, data, *, kernel_source=None, integrator='ssp33',
                    cancelled=lambda: None):
    from braids_core.sediment.river_source_io import map_files
    cancelled()
    _validate_prepared(data, integrator=integrator)
    spec = INTEGRATORS[integrator]
    folder = filesystem_path(folder).resolve()
    require(not folder.exists(), 'Preserve existing live transport source archive')
    origin = SPATIAL if kernel_source is None else Path(kernel_source).resolve()
    require(origin.is_dir() and origin.is_relative_to(ROOT.resolve()) and
            not origin.is_symlink() and any(origin.glob('*.cu')),
            'Live transport kernel source must be package-relative and nonempty')
    jobs = []
    destinations = set()
    for original in _source_files(origin):
        cancelled()
        require(original.is_file() and not original.is_symlink(),
                'Live transport source is missing or linked')
        relative = original.relative_to(ROOT).as_posix()
        target = folder/original.name if original.suffix == '.cu' else folder/'python_snapshot'/relative
        key = target.relative_to(folder).as_posix().casefold()
        require(key not in destinations, 'Live transport source destination collision: '+relative)
        destinations.add(key)
        jobs.append((original, target, relative))
    folder.mkdir(parents=True)
    for parent in sorted({target.parent for _, target, _ in jobs}):
        parent.mkdir(parents=True, exist_ok=True)
    def copy_source(job):
        original, target, relative = job
        shutil.copy2(original, target)
        return relative, SHA(original)
    sources = dict(map_files(copy_source, jobs, cancelled=cancelled))
    generated = {}
    for name, private in PRIVATE.items():
        cancelled()
        target = folder/(private + '.py')
        target.write_text(_transform(name, _original(name).read_text(encoding='utf-8'),
                                     integrator=integrator), encoding='utf-8')
        generated[name] = dict(name=target.name, sha256=SHA(target))
    forcing = _typed_forcing_identity(data)
    viscous_inlet = None
    if np.any(data.get('viscosity_equilibrated_inlet', 0)):
        from .viscous_inlet_boundary import validate_viscous_inlet_scope
        viscous_inlet = validate_viscous_inlet_scope(data, integrator=integrator)
    receipt = dict(schema=1, variant=VARIANT, sources=sources, generated=generated,
        CUDA_copied_verbatim=True, normal_root_rule=NORMAL_ROOT_RULE,
        normal_trace_rule=NORMAL_TRACE_RULE,
        normal_face_policy=normal_face_policy(data, integrator),
        integrator=spec['label'], integrator_token=integrator,
        euler_stages=spec['euler_stages'], stage_slots=list(spec['stage_slots']),
        stage_fractions=list(spec['stage_fractions']),
        native_cpu_time_integrator=int(data['cpu_time_integrator']),
        bedlaw=int(data['bedlaw']),
        m1_profile_runtime_schema=int(data.get('m1_profile_runtime_schema', 1)),
        nfrac=int(data['nfrac']),
        nbed_layers=int(data['nbed_layers']), exchange_mode=int(data['exchange_mode']),
        secondary_flow_model=int(data['secondary_flow_model']),
        delft_forcing=forcing,
        viscosity_equilibrated_inlet=viscous_inlet,
        general_avalanche=bool(data['general_avalanche']),
        avalanche_order=AVALANCHE_ORDER if bool(data['general_avalanche']) else 'disabled',
        physical_validation=False, Delft3D_source_modified=False)
    if kernel_source is not None:
        receipt['kernel_origin'] = origin.relative_to(ROOT).as_posix()
    (folder/'general_live_transport_sources.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    marker = data.get(PUBLIC_ADMISSION_KEY)
    if (int(data['bedlaw']) == 11 or
            (isinstance(marker, dict) and marker.get('admission_id') in (
             SCALAR_PUBLIC_ADMISSION_ID, HIDING_PUBLIC_ADMISSION_ID))):
        profile_receipt = archive_profile_sources(folder, data, root=ROOT)
        receipt.update(m1_profile_generation=profile_receipt['generation'],
            capacity_profile_fingerprint=profile_receipt['capacity_profile_fingerprint'],
            m1_profile_manifest_sha256=SHA(folder/'m1_profile_sources.json'))
        if SECONDARY_EXECUTION_KEY in profile_receipt:
            receipt['secondary_execution_fingerprint'] = \
                profile_receipt[SECONDARY_EXECUTION_KEY]['fingerprint']
        if PROGNOSTIC_SECONDARY_EXECUTION_KEY in profile_receipt:
            receipt['prognostic_secondary_execution_fingerprint'] = \
                profile_receipt[PROGNOSTIC_SECONDARY_EXECUTION_KEY][
                    'fingerprint']
        (folder/'general_live_transport_sources.json').write_text(
            json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    verify_sources(folder, cancelled=cancelled)
    cancelled()
    return receipt


def verify_sources(folder, *, cancelled=lambda: None):
    from braids_core.sediment.river_runtime_lease import verified_sources
    def inventory(receipt):
        origin=ROOT/receipt.get('kernel_origin','braids_support/gpu_fwave')
        # Only these two globbed sets can gain/lose members. Fixed helpers and
        # kernel files remain individually byte-leased and path-identity checked
        # by verified_sources. Do not rebuild hundreds of constant Path objects
        # and relative spellings for every AMR owner.
        return (tuple(sorted(p.name for p in origin.glob('*.cu'))),
                tuple(sorted(p.name for p in SPATIAL.glob('*.py'))),
                tuple(PYTHON_HELPERS))
    cancelled()
    result = verified_sources(folder, lambda path: _verify_sources(path, cancelled=cancelled), inventory)
    cancelled()
    return result


def _verify_sources(folder, *, cancelled=lambda: None):
    from braids_core.sediment.river_source_io import map_files
    cancelled()
    folder = filesystem_path(folder).resolve()
    receipt = json.loads((folder/'general_live_transport_sources.json').read_text(encoding='utf-8'))
    labels = {spec['label']: key for key, spec in INTEGRATORS.items()}
    integrator = labels.get(receipt.get('integrator'))
    require(integrator is not None, 'General live transport source integrator differs')
    spec = INTEGRATORS[integrator]
    require(receipt.get('schema') == 1 and receipt.get('variant') == VARIANT and
            receipt.get('CUDA_copied_verbatim') is True and
            receipt.get('normal_root_rule') == NORMAL_ROOT_RULE and
            receipt.get('normal_trace_rule') == NORMAL_TRACE_RULE and
            receipt.get('integrator_token') == integrator and
            receipt.get('normal_face_policy') == normal_face_policy(receipt, integrator) and
            receipt.get('euler_stages') == spec['euler_stages'] and
            receipt.get('stage_slots') == list(spec['stage_slots']) and
            receipt.get('stage_fractions') == list(spec['stage_fractions']) and
            receipt.get('native_cpu_time_integrator') == 33,
            'General live transport source contract differs')
    origin = Path(receipt.get('kernel_origin', 'braids_support/gpu_fwave'))
    require(not origin.is_absolute() and '..' not in origin.parts and
            (ROOT/origin).resolve().is_relative_to(ROOT.resolve()),
            'General live transport kernel origin escapes package')
    require(set(receipt['sources']) == {p.relative_to(ROOT).as_posix() for p in _source_files(ROOT/origin)}
            and set(receipt['generated']) == set(PRIVATE),
            'General live transport source inventory differs')
    def verify_source(item):
        relative, digest = item
        path = ROOT/relative
        saved = folder/path.name if path.suffix == '.cu' else folder/'python_snapshot'/relative
        require(not path.is_symlink() and not saved.is_symlink() and
                SHA(path) == digest and SHA(saved) == digest,
                'General live transport source drift: ' + relative)
    map_files(verify_source, receipt['sources'].items(), cancelled=cancelled)
    for name, row in receipt['generated'].items():
        cancelled()
        require(row['name'] == PRIVATE[name]+'.py', 'General live transport generated path differs')
        target = folder/row['name']
        require(not target.is_symlink() and SHA(target) == row['sha256'] and
                target.read_text(encoding='utf-8') ==
                _transform(name, _original(name).read_text(encoding='utf-8'),
                           integrator=integrator),
                'General live transport generated bridge differs: ' + name)
    cancelled()
    return receipt


def _stock_parent(Parent, *, flow2dh_return_owner=False):
    from .sample_discharge_boundary import wrap_boundary
    class StockModeParent(Parent):
        def __init__(self, data, inputs, source=None, **kwargs):
            # This switch activates the already implemented mode1 trace only;
            # the bedform physics flag remains the actual input value.
            parent_data = data
            if flow2dh_return_owner:
                # The stock hydraulic trace code does not consume reversal
                # policy.  Its legacy validator nevertheless rejects FLOW's
                # Thatcher-Harleman value.  Authenticate M2-C, then present
                # the inert no-reversal token only to that hydraulic parent;
                # ResidentFlow2DHBoundaryDiffusion retains and executes the
                # unmodified per-face return policy and memory from ``data``.
                from .implicit_suspension_amr import contract_from_data
                contract_from_data(data)
                parent_data = dict(data, reversal_policy=np.ones_like(
                    np.asarray(data['reversal_policy'])))
            super().__init__(parent_data, inputs, source=source, allow_scheduled=True,
                             allow_bedform_mode1=bool(np.any(np.asarray(data['mode']) == 1)))
    return wrap_boundary(StockModeParent)


def _without_critical_outlet(Parent, *, euler_stages=3, integrator='SSPRK33'):
    """Same stock trace reset once; no synthetic critical outlet is created."""
    class StockOutlet(Parent):
        def __init__(self, data, inputs, source=None, **kwargs):
            super().__init__(data, inputs, source=source, **kwargs)
            self.outlet_records = {}
            self._glt_q = None
            self._glt_bound_storage = None

        def _glt_storage(self):
            values = list(self.arrays.values()) + [self.inputs[key] for key in
                ('bed', 'normal', 'length', 'owner', 'trace')]
            if self._glt_q is not None:
                values.append(self._glt_q)
            return tuple((id(v), v.data.ptr, v.shape, v.dtype.str, v.strides, v.device.id) for v in values)

        def _bind_stage_state_outlet(self, q):
            require(self._glt_q is None and isinstance(q, self.cp.ndarray) and
                    q.shape == (3, self.nc) and q.dtype == self.cp.float64 and q.flags.c_contiguous,
                    'Stock boundary requires one stable resident stage state')
            self._glt_q = q
            self._glt_bound_storage = self._glt_storage()

        def _check_storage_outlet(self):
            require(self._glt_q is not None and self._glt_storage() == self._glt_bound_storage,
                    'Stock boundary storage changed')
            return True

        def _begin_trial_outlet(self):
            self._check_storage_outlet()

        def _apply_outlet(self, q, trace):
            self._check_storage_outlet()
            require(q is self._glt_q and trace is self.inputs['trace'], 'Stock boundary stage binding differs')
            return super().apply(q, trace)

        def _record_stage_outlet(self, index):
            require(type(index) is int and 0 <= index < euler_stages,
                    'Stock boundary '+integrator+' stage differs')

        def _audit_outlet(self):
            return self._check_storage_outlet()

        def _receipt_outlet(self):
            return dict(critical_outlet=False, modes=self.cp.asnumpy(self.arrays['mode']).tolist(),
                        actual_stock_traces=True, parent_reset_count_per_apply=1)
    return StockOutlet


def _boundary_factory(modules, data, *, source_folder, integrator_token,
                      euler_stages=3, integrator='SSPRK33',
                      face_policy=NORMAL_TRACE_RULE):
    def composed(boundary):
        if not np.any(data.get('viscosity_equilibrated_inlet', 0)):
            return boundary
        from .viscous_inlet_boundary import wrap_boundary
        return wrap_boundary(boundary, data, source_folder=source_folder,
                             integrator=integrator_token)

    parent = _stock_parent(
        modules['boundaries.py'].BoundaryControls,
        flow2dh_return_owner=(
            int(data['exchange_mode']) == FLOW2DH_EXCHANGE_MODE and
            'suspended_amr_contract' in data))
    has_critical = bool(np.any(np.asarray(data['mode']) == 4))
    outlet_module = modules['outlet_boundary.py'] if has_critical else SimpleNamespace(
        make_boundary_controls=lambda Parent: _without_critical_outlet(
            Parent, euler_stages=euler_stages, integrator=integrator))
    if np.any(np.asarray(data['mode']) == 9):
        if int(data['section_control'][0]) == 1:
            from .integrated_normal_trace import integrated_boundary
            from braids_support.gpu_fwave.normal_outlet_boundary import make_boundary_controls as normal
            return composed(integrated_boundary(
                normal(parent, modules['live_normal_boundary.py'], outlet_module),
                face_policy=face_policy))
        return composed(modules['g1_live_inlet_boundary.py'].make_boundary_controls(
            parent, outlet_module))
    Outlet = outlet_module.make_boundary_controls(parent)
    class StockBoundary(Outlet):
        def bind_sediment_hydraulic(self, value):
            require(value.shape == (2, self.nb) and value.dtype == self.cp.float64,
                    'Stock sediment hydraulic binding differs')
        def bind_stage_state(self, q):
            self._bind_stage_state_outlet(q)
        def check_storage(self):
            return self._check_storage_outlet()
        def begin_trial(self):
            self._begin_trial_outlet()
        def apply(self, q, trace):
            return self._apply_outlet(q, trace)
        def record_stage(self, index):
            self._record_stage_outlet(index)
        def receipt(self):
            return self._receipt_outlet()
    return composed(StockBoundary)


@contextmanager
def constructors(folder, data, *, integrator='ssp33',
                  avalanche_placement=AVALANCHE_ORDER,
                  viscosity_placement=VISCOSITY_ORDER,
                  bedform_placement=BEDFORM_ORDER):
    from .a1_full_array_transaction import wrap_solver
    from braids_support.gpu_fwave.boundary_events import tolerance
    _validate_prepared(data, integrator=integrator)
    require(avalanche_placement in AVALANCHE_PLACEMENTS,
            'General live transport avalanche placement differs')
    require(viscosity_placement in VISCOSITY_PLACEMENTS,
            'General live transport viscosity placement differs')
    require(bedform_placement in BEDFORM_PLACEMENTS,
            'General live transport bedform placement differs')
    integration = INTEGRATORS[integrator]
    implicit = integrator == IMPLICIT_TOKEN
    folder = filesystem_path(folder).resolve()
    source_receipt = verify_sources(folder)
    require(source_receipt['integrator'] == integration['label'],
            'General live transport source/selected integrator differs')
    if 'delft_forcing_binding' in data:
        require(source_receipt.get('delft_forcing') ==
                _typed_forcing_identity(data),
                'General live transport source/typed forcing identity differs')
    else:
        require(source_receipt.get('delft_forcing') is None,
                'Legacy live transport acquired a typed forcing identity')
    viscous_receipt = None
    if np.any(data.get('viscosity_equilibrated_inlet', 0)):
        from .viscous_inlet_boundary import validate_viscous_inlet_scope
        viscous_receipt = validate_viscous_inlet_scope(data, integrator=integrator)
    require(source_receipt.get('viscosity_equilibrated_inlet') == viscous_receipt,
            'General live transport source/viscous-inlet identity differs')
    marker = data.get(PUBLIC_ADMISSION_KEY)
    scalar_public = (isinstance(marker, dict) and
                     marker.get('admission_id') in (
                         SCALAR_PUBLIC_ADMISSION_ID,
                         HIDING_PUBLIC_ADMISSION_ID))
    if int(data['bedlaw']) == 11 or scalar_public:
        profile_source_receipt = verify_profile_sources(folder, data, root=ROOT)
        require(source_receipt.get('m1_profile_generation') ==
                    profile_source_receipt['generation'] and
                source_receipt.get('capacity_profile_fingerprint') ==
                    profile_source_receipt['capacity_profile_fingerprint'] and
                source_receipt.get('m1_profile_manifest_sha256') ==
                    SHA(folder/'m1_profile_sources.json'),
                'Law-11 profile/base source identities differ')
        secondary_identity = validate_secondary_composition(data)
        prognostic_secondary_identity = \
            validate_prognostic_secondary_composition(data)
        require(source_receipt.get('secondary_execution_fingerprint') ==
                (None if secondary_identity is None else
                 secondary_identity['fingerprint']),
                'Q3-H2 source/runtime identity differs')
        require(source_receipt.get(
                    'prognostic_secondary_execution_fingerprint') ==
                (None if prognostic_secondary_identity is None else
                 prognostic_secondary_identity['fingerprint']),
                'Q3-H3 source/runtime identity differs')
    prefix = 'braids_support.gpu_fwave.'
    require(not any(prefix+p in sys.modules for p in PRIVATE.values()),
            'General live transport private modules already loaded')
    loaded, modules = [], {}
    try:
        for name, private in PRIVATE.items():
            key = prefix+private
            spec = importlib.util.spec_from_file_location(key, folder/(private+'.py'))
            require(spec is not None and spec.loader is not None, 'Live transport private module missing')
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            loaded.append(key)
            spec.loader.exec_module(module)
            modules[name] = module
        core = modules['coupled_solver.py']
        modes, controls = np.asarray(data['mode']).copy(), np.asarray(data['section_control']).copy()
        control = int(controls[0]) if len(modes) and modes[0] == 9 else -1
        selected_face_policy = normal_face_policy(data, integrator)
        require(source_receipt.get('normal_face_policy') == selected_face_policy,
                'General live transport normal face policy/source differs')
        core.BoundaryControls = _boundary_factory(modules, data,
            euler_stages=integration['euler_stages'], integrator=integration['label'],
            source_folder=folder, integrator_token=integrator,
            face_policy=selected_face_policy)
        Transport = core.ResidentSedimentTransport
        Law11Transport = make_law11_transport(Transport)
        RPTaubTransport = make_rp_taub_transport(Transport)
        def sediment_operator(actual, **kwargs):
            marker = actual.get(PUBLIC_ADMISSION_KEY)
            if (int(actual['bedlaw']) == 11 or
                    (isinstance(marker, dict) and marker.get('admission_id') in (
                     SCALAR_PUBLIC_ADMISSION_ID, HIDING_PUBLIC_ADMISSION_ID))):
                return Law11Transport(actual, **kwargs)
            if int(actual['bedlaw']) in (5, 6, 7):
                return RPTaubTransport(actual, **kwargs)
            return (ResidentStaticBed(actual, **kwargs) if static_bed_enabled(actual)
                    else Transport(actual, **kwargs))
        core.ResidentSedimentTransport = sediment_operator
        original_scope = core.validate_scope
        def admitted_scope(actual, **flags):
            require(bool(flags.get('bedload_adaptation_variant')) == bool(actual['bedload_adaptation']) and
                    bool(flags.get('bedform_variant')) == bool(actual['bedform_roughness']) and
                    bool(flags.get('general_avalanche_variant')) == bool(actual['general_avalanche']) and
                    not any(value for key, value in flags.items()
                            if key not in ('bedload_adaptation_variant', 'bedform_variant', 'general_avalanche_variant')),
                    'Live transport excludes unmatched adaptation or special bank/research variants')
            return _validate_prepared(actual, integrator=integrator) is not None
        core.validate_scope = admitted_scope
        try:
            class GeneralLiveTransportSolver(core.CoupledSolver):
                def __init__(self, raw, *, source_folder=folder):
                    _validate_prepared(raw, integrator=integrator)
                    require(np.array_equal(raw['mode'], modes) and np.array_equal(raw['section_control'], controls) and
                            Path(source_folder).resolve() == folder,
                            'General live transport section/source changed')
                    runtime_raw = (core.normalized(raw)
                                   if int(raw['exchange_mode']) ==
                                   FLOW2DH_EXCHANGE_MODE else raw)
                    super().__init__(runtime_raw, source_folder=folder,
                                     integrator=('ssp33' if implicit else integrator),
                                     bedload_adaptation_variant=bool(raw['bedload_adaptation']),
                                     general_avalanche_variant=bool(raw['general_avalanche']),
                                     bedform_variant=bool(raw['bedform_roughness']))
                    self.classic_secondary_identity = \
                        validate_secondary_composition(raw)
                    self.prognostic_secondary_identity = \
                        validate_prognostic_secondary_composition(raw)
                    if self.classic_secondary_identity is not None:
                        require(self.secondary.model == 1 and
                                self.secondary.direction,
                                'Q3-H2 collided with a changed legacy secondary owner')
                        # Q3-H2 is equilibrium bed-direction composition only.
                        # The generic class was constructed from the real deck
                        # so its inventory is authentic, then made inert to
                        # prevent legacy D-Flow-FM curvature, momentum, or a
                        # second capacity rotation from entering the stage.
                        self.secondary.model = 0
                        self.secondary.direction = False
                        require(getattr(self.sediment, 'classic_secondary', None)
                                is not None,
                                'Q3-H2 resident secondary owner is missing')
                    self.prognostic_secondary = None
                    if self.prognostic_secondary_identity is not None:
                        require(self.classic_secondary_identity is None and
                                self.secondary.model == 2 and
                                self.secondary.direction and
                                getattr(self.sediment, 'classic_secondary', None)
                                    is None and
                                getattr(self.sediment,
                                        'prognostic_secondary_identity', None) ==
                                    self.prognostic_secondary_identity,
                                'Q3-H3 collided with another secondary owner')
                        from .flow2dh_prognostic_secondary import (
                            ResidentClassicFlow2DHPrognosticSecondary,
                        )
                        self.prognostic_secondary = \
                            ResidentClassicFlow2DHPrognosticSecondary(
                                raw, self.reconstruction,
                                self.sediment.geometry,
                                source_folder=source_folder,
                                transport=self.secondary)
                        self.sediment.prognostic_secondary = \
                            self.prognostic_secondary
                        # The generic model-2 object now supplies only the
                        # qualified scalar reconstruction/FCT kernels through
                        # the v57 owner.  Its exact source/filter/stress and
                        # capacity-rotation entry points are unreachable.
                        self.secondary.buffers.update(
                            zero_status=self.cp.zeros((), dtype=self.cp.int32),
                            source_integral=self.cp.zeros((), dtype=self.cp.float64),
                            equilibrium_intensity=self.cp.zeros(
                                self.nc, dtype=self.cp.float64),
                            equilibrium_mass=self.cp.zeros(
                                self.nc, dtype=self.cp.float64))
                        self.secondary.model = 0
                        self.secondary.direction = False
                    self.law11_soursin_exchange = None
                    self.flow2dh_boundary_diffusion = None
                    self.flow2dh_surface_fraction = None
                    if int(raw['exchange_mode']) == FLOW2DH_EXCHANGE_MODE:
                        require(self.has_suspended and
                                self.suspended_exchange is None and
                                self.exchange_reduction is None,
                                'FLOW 2DH exchange collided with a legacy exchange operator')
                        self.law11_soursin_exchange = ResidentLaw11SoursinExchange(
                            raw, source_path=folder/'law11_soursin_exchange.cu')
                        self.exchange_reduction = core.ResidentExchangeLedger(
                            self.suspended_advection.geometry['area'], self.ng,
                            source_path=folder/'exchange_ledger_parallel.cu')
                        self.exchange_ledger_algorithm = self.exchange_reduction.method
                        self.exchange_ledger = self.exchange_reduction.ledger
                        self.exchange_ledger_status = self.exchange_reduction.status
                        self.exchange_flag = self.cp.zeros((), dtype=self.cp.int32)
                        self.flow2dh_surface_fraction = self.cp.ones(
                            (self.ng, self.nc), dtype=self.cp.float64)
                        self.flow2dh_boundary_diffusion = (
                            ResidentFlow2DHBoundaryDiffusion(
                                raw, self.suspended_faces,
                                self.suspended_advection,
                                source_path=folder/
                                    'flow2dh_boundary_diffusion.cu'))
                    self._glt_avalanche_placement = (
                        avalanche_placement if self.general_avalanche_enabled
                        else 'disabled')
                    self._glt_viscosity_placement = (
                        viscosity_placement if self.diffusion is not None
                        else 'disabled')
                    self._glt_bedform_placement = (
                        bedform_placement if self.bedform is not None
                        else 'disabled')
                    if self._glt_avalanche_placement == PARENT_AVALANCHE_ORDER:
                        # Keep the authenticated resident operator, but prevent
                        # asynchronous level-local Euler calls from executing a
                        # nonlocal transfer.  The recursive owner invokes it
                        # once on the synchronized post-reflux leaf state.
                        self.general_avalanche_variant = False
                    self.flux.inputs['q'] = self.stepper.stage['q']
                    self.boundary.bind_sediment_hydraulic(self.sediment.geometry['bc_hydraulic'])
                    self.boundary.bind_stage_state(self.stepper.stage['q'])
                    if control == 1:
                        self.flux.integrated_section_flux = self.boundary.integrated_flux
                    self.schedule = modules['boundary_schedule.py'].ResidentBoundarySchedule(
                        raw, source_path=folder/'boundary_schedule.cu')
                    groups = [self.schedule.outputs, getattr(self.boundary, 'normal_records', {}),
                              self.boundary.outlet_records]
                    if self.has_suspended:
                        groups += [self.suspended_faces.buffers, self.suspended_advection.buffers]
                    for group in groups:
                        for value in group.values():
                            value.fill(0)
                    if self.suspended_exchange is not None:
                        for key in ('next_suspended', 'next_active', 'delta', 'status'):
                            getattr(self.suspended_exchange, key).fill(0)
                        self.exchange_reduction.ledger.fill(0)
                        self.exchange_reduction.status.fill(0)
                        for value in self.exchange_reduction.scratch:
                            value.fill(0)
                    if self.law11_soursin_exchange is not None:
                        for key in ('next_suspended', 'next_active',
                                    'equilibrium_concentration',
                                    'adaptation_velocity',
                                    'dimensionless_timescale',
                                    'clear_water_settling_velocity',
                                    'settling_velocity', 'hindered_factor', 'delta',
                                    'limiter_correction', 'sediment_active',
                                    'status'):
                            getattr(self.law11_soursin_exchange, key).fill(0)
                        self.exchange_reduction.ledger.fill(0)
                        self.exchange_reduction.status.fill(0)
                        for value in self.exchange_reduction.scratch:
                            value.fill(0)
                    if self.bedload_adaptation_variant:
                        for value in self.sediment.adaptation.work.values():
                            value.fill(0)
                        self.sediment.adaptation.residual = 0.
                    if self.bedform is not None:
                        for key in ('properties', 'iterations', 'status'):
                            getattr(self.bedform, key).fill(0)
                    if self.avalanche is not None:
                        # Initialize private work only, never geometry, bulk or
                        # accepted layers. Every trial then has a deterministic
                        # byte snapshot, including a rejected first avalanche.
                        for obj, keys in (
                            (self.avalanche, ('layers', 'bed', 'bed_status', 'status')),
                            (self.avalanche.proposal, ('down', 'intercept', 'rate', 'conductance',
                                'cell_rate', 'maximum_rate', 'subdt', 'proposed')),
                            (self.avalanche.transfer, ('scale', 'limited', 'next', 'cell_status', 'face_status'))):
                            for key in keys:
                                getattr(obj, key).fill(0)
                        for value in self.avalanche.proposal.status.values():
                            value.fill(0)
                    if hasattr(self.boundary, 'normal_root'):
                        self.boundary.normal_root.invalidate_cache()
                    self._glt_stage = self._glt_times = None
                    self._glt_accepted_time = float(raw.get('time', 0.))
                    self.implicit_midpoint = (ImplicitMidpointController(
                        self, source_path=folder/'implicit_midpoint_fwave.cu')
                        if implicit else None)

                def euler(self, state, dt, current):
                    index = self._glt_stage
                    require(type(index) is int and 0 <= index < integration['euler_stages'] and
                            abs(current-self._glt_times[index]) <= tolerance(current),
                            'General live transport '+integration['label']+' stage clock differs')
                    slot, values = integration['stage_slots'][index], self.schedule.outputs
                    self.cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][slot])
                    self.cp.copyto(self.sediment.geometry['bc_hydraulic'], values['hydraulic'][slot])
                    self.cp.copyto(self.sediment.geometry['feed'], values['feed'][slot])
                    if self.has_suspended:
                        self.cp.copyto(self.bc_concentration, values['concentration'][slot])
                    result = super().euler(state, dt, current)
                    self.boundary.record_stage(index)
                    self._merge(values['status'][slot])
                    self._glt_stage += 1
                    return result

                def _a1_trial_diagnostics(self):
                    saved = dict(accepted_time=self._glt_accepted_time,
                                 filter_receipt=self.secondary.last_filter_receipt,
                                 avalanche_substeps=list(self.avalanche_trial_substeps))
                    if hasattr(self.sediment, 'last_capacity_result'):
                        # Q3-C point results own freshly allocated CuPy result
                        # arrays.  They are diagnostic object references, not
                        # members of the segmented array inventory, so retain
                        # the accepted references explicitly across a rejected
                        # or cancelled trial.
                        saved['capacity_result_refs'] = (
                            self.sediment.last_capacity_result,
                            getattr(self.sediment, 'last_capacity_points', None))
                    if self.bedload_adaptation_variant:
                        saved['adaptation'] = copy.deepcopy((self.sediment.adaptation.iterations,
                            self.sediment.adaptation.residual, self.sediment.adaptation_result))
                    if self.implicit_midpoint is not None:
                        saved['implicit'] = dict(
                            have_predictor=self.implicit_midpoint.have_predictor,
                            last_diagnostics=copy.deepcopy(
                                self.implicit_midpoint.last_diagnostics),
                            accepted_steps=self.implicit_midpoint.accepted_steps,
                            rejected_steps=self.implicit_midpoint.rejected_steps,
                            fwave_evaluations=self.implicit_midpoint.fwave_evaluations,
                            linear_matvecs=self.implicit_midpoint.linear_matvecs)
                        bend = self.implicit_midpoint.bend_physics_v1
                        if bend is not None:
                            saved['bend_v1'] = bend.trial_metadata()
                    return saved

                def _a1_restore_diagnostics(self, saved, *, where=True):
                    if where:
                        self._glt_accepted_time = saved['accepted_time']
                        self.secondary.last_filter_receipt = saved['filter_receipt']
                        self.avalanche_trial_substeps = saved['avalanche_substeps']
                        if 'capacity_result_refs' in saved:
                            (self.sediment.last_capacity_result,
                             self.sediment.last_capacity_points) = \
                                saved['capacity_result_refs']
                        if self.bedload_adaptation_variant:
                            (self.sediment.adaptation.iterations, self.sediment.adaptation.residual,
                             self.sediment.adaptation_result) = saved['adaptation']
                        if self.implicit_midpoint is not None:
                            implicit_saved = saved['implicit']
                            self.implicit_midpoint.have_predictor = implicit_saved['have_predictor']
                            self.implicit_midpoint.last_diagnostics = implicit_saved['last_diagnostics']
                            self.implicit_midpoint.accepted_steps = implicit_saved['accepted_steps']
                            self.implicit_midpoint.rejected_steps = implicit_saved['rejected_steps']
                            self.implicit_midpoint.fwave_evaluations = implicit_saved['fwave_evaluations']
                            self.implicit_midpoint.linear_matvecs = implicit_saved['linear_matvecs']
                            if 'bend_v1' in saved:
                                self.implicit_midpoint.bend_physics_v1.restore_trial_metadata(saved['bend_v1'])

                def step(self, dt, current, *, commit=True):
                    require(self._glt_avalanche_placement != PARENT_AVALANCHE_ORDER,
                            'Parent-clock avalanche requires the recursive AMR owner')
                    require(self._glt_viscosity_placement != PARENT_VISCOSITY_ORDER,
                            'Parent-clock viscosity requires the recursive AMR owner')
                    require(self._glt_bedform_placement != RECURSIVE_BEDFORM_ORDER,
                            'Level-clock bedform feedback requires the recursive AMR owner')
                    require(self._glt_stage is None, 'Concurrent live transport trial')
                    self.boundary.check_storage()
                    saved = self._a1_trial_diagnostics()
                    self._glt_stage = 0
                    self._glt_times = tuple(float(current+f*dt) for f in integration['stage_fractions'])
                    try:
                        self.boundary.begin_trial()
                        self.schedule.evaluate(current, dt)
                        if self.implicit_midpoint is None:
                            result = super().step(dt, current, commit=commit)
                            require(self._glt_stage == integration['euler_stages'],
                                    'Live transport omitted Euler stage')
                        else:
                            values = self.schedule.outputs
                            slot = integration['stage_slots'][0]
                            self.cp.copyto(self.boundary.arrays['forcing'],
                                           values['hydraulic'][slot])
                            self.cp.copyto(self.sediment.geometry['bc_hydraulic'],
                                           values['hydraulic'][slot])
                            self.cp.copyto(self.sediment.geometry['feed'],
                                           values['feed'][slot])
                            if self.has_suspended:
                                self.cp.copyto(self.bc_concentration,
                                               values['concentration'][slot])
                            if bool(self.cp.any(values['status'][slot] != 0).item()):
                                raise RuntimeError('Implicit midpoint boundary schedule is invalid')

                            def weak_sediment(midpoint, macro_dt, midpoint_time, *, secondary=None):
                                # Only private SSPRK allocations are exposed to
                                # the unchanged Euler sediment/Exner operator.
                                # Its explicit hydraulic endpoint is discarded.
                                self.cp.copyto(self.stepper.stage['q'], midpoint)
                                self.cp.copyto(self.stepper.stage['layers'],
                                               self.state['layers'])
                                self.cp.copyto(self.stepper.stage['secondary'],
                                               self.state['secondary'] if secondary is None else secondary)
                                old_target = self.flux.controls['cfl_target']
                                self.flux.controls['cfl_target'] = np.float64(1.e100)
                                suspended_owner = self.has_suspended
                                if int(self.data['exchange_mode']) == FLOW2DH_EXCHANGE_MODE:
                                    self.has_suspended = False
                                try:
                                    result = self.euler(self.stepper.stage, macro_dt,
                                                        midpoint_time)
                                finally:
                                    self.has_suspended = suspended_owner
                                    self.flux.controls['cfl_target'] = old_target
                                return result

                            result = self.implicit_midpoint.step(
                                dt, current,
                                sediment_callback=(weak_sediment
                                    if int(self.data['bedlaw']) in (2, 5, 6, 7, 8, 11) else None),
                                commit=commit)
                            if (int(result['status'].item()) == 0 and
                                    int(self.data['bedlaw']) == 0):
                                self.boundary.record_stage(0)
                                self._glt_stage = 1
                            require(int(result['status'].item()) != 0 or
                                    self._glt_stage == integration['euler_stages'],
                                    'Implicit transport omitted midpoint stage')
                        accepted = commit and not bool(result['status'].item())
                        self._a1_restore_diagnostics(saved,
                            where=result['status'] != 0 if commit else True)
                        if accepted:
                            self._glt_accepted_time = float(current+dt)
                        return result
                    except Exception:
                        self._a1_restore_diagnostics(saved)
                        raise
                    finally:
                        self._glt_stage = self._glt_times = None

                def parent_post_reflux_avalanche(self, layers, depth, dt):
                    """Apply the admitted nonlocal repose operator on one root clock."""
                    require(self._glt_avalanche_placement == PARENT_AVALANCHE_ORDER and
                            self.general_avalanche_enabled and self.avalanche is not None and
                            self._glt_stage is None and self._glt_times is None,
                            'Parent-clock avalanche is unavailable or an RK stage is active')
                    result = self.avalanche.step(layers, depth, dt)
                    self.avalanche_trial_substeps.append(int(result['substeps']))
                    return result

                def parent_post_reflux_viscosity(self, q, layers, dt):
                    """Apply native SSPRK2 momentum diffusion on one root clock.

                    The recursive owner calls this only after all hyperbolic and
                    sediment levels have synchronized and refluxed.  The full
                    selected leaf graph is therefore updated through the native
                    pairwise velocity-flux divergence, including coarse/fine
                    faces, before any parent avalanche.  This formulation does
                    not imply momentum conservation at variable depth.
                    """
                    require(self._glt_viscosity_placement == PARENT_VISCOSITY_ORDER and
                            self.diffusion is not None and
                            self._glt_stage is None and self._glt_times is None,
                            'Parent-clock viscosity is unavailable or an RK stage is active')
                    return self.diffusion.step(q, self.bed(layers), dt)

                def fields(self, *, current=None):
                    require(self._glt_stage is None and (current is None or
                            (np.isfinite(current) and float(current) >= 0.)),
                            'Invalid accepted live transport output clock')
                    if self.bedload_adaptation_variant:
                        current = self._glt_accepted_time if current is None else float(current)
                        values = self.schedule.evaluate(current,
                            max(float(self.data['dtmax']), float(np.spacing(current))))
                        require(not bool(self.cp.any(values['status'][0]).item()),
                                'Invalid accepted transport adaptation forcing')
                        self.cp.copyto(self.sediment.geometry['feed'], values['feed'][0])
                    result = super().fields()
                    bend = getattr(self, 'bend_bundle_v1', None)
                    if bend is not None:
                        result.update(bend.fields())
                    return result

                def gaia_fixed_inlet_diagnostic(self, current):
                    require(self._glt_stage is None and np.isfinite(current) and
                            float(current) >= 0.,
                            'Invalid accepted GAIA endpoint diagnostic clock')
                    span=max(float(self.data['dtmax']),float(np.spacing(float(current))))
                    values=self.schedule.evaluate(float(current),span)
                    require(not bool(self.cp.any(values['status'][0]).item()),
                            'Invalid GAIA endpoint boundary forcing')
                    self.cp.copyto(self.boundary.arrays['forcing'],values['hydraulic'][0])
                    self.cp.copyto(self.sediment.geometry['bc_hydraulic'],values['hydraulic'][0])
                    self.cp.copyto(self.sediment.geometry['feed'],values['feed'][0])
                    if self.has_suspended:
                        self.cp.copyto(self.bc_concentration,values['concentration'][0])
                    return super().gaia_fixed_inlet_diagnostic(float(current))

                def candidate_receipt(self):
                    receipt = dict(variant=VARIANT, integrator=integration['label'], backend='gpu',
                        euler_stages=integration['euler_stages'],
                        stage_slots=list(integration['stage_slots']),
                        stage_fractions=list(integration['stage_fractions']),
                        native_cpu_time_integrator=int(self.data['cpu_time_integrator']),
                        normal_face_policy=selected_face_policy,
                        bedlaw=int(self.data['bedlaw']), exchange_mode=int(self.data['exchange_mode']),
                        secondary_flow_model=int(self.data['secondary_flow_model']),
                        hiding_exposure=int(self.data['hiding_exposure']),
                        bed_slope=int(self.data['bed_slope']),
                        adaptation_enabled=self.bedload_adaptation_variant,
                        adaptation_length_diameters=float(self.data['bedload_adaptation_length_diameters']),
                        adaptation_output_clock='accepted_time',
                        horizontal_viscosity=float(self.data['horizontal_viscosity']),
                        horizontal_viscosity_mode=int(self.data['horizontal_viscosity_mode']),
                        horizontal_viscosity_geometry=int(self.data['horizontal_viscosity_geometry']),
                        viscosity_order=self._glt_viscosity_placement,
                        viscosity_integrator=(VISCOSITY_INTEGRATOR
                            if self.diffusion is not None else 'disabled'),
                        viscosity_limited_norm=(VISCOSITY_LIMITED_NORM
                            if self.diffusion is not None else 'disabled'),
                        bedform_roughness=bool(self.data['bedform_roughness']),
                        bedform_order=self._glt_bedform_placement,
                        bedform_diagnostic_clock=('lazy_accepted_frame'
                            if self.bedform is not None else 'disabled'),
                        recursive_bedform_method=(dict(RECURSIVE_BEDFORM_METHOD)
                            if self._glt_bedform_placement == RECURSIVE_BEDFORM_ORDER
                            else None),
                        general_avalanche=self.general_avalanche_enabled,
                        avalanche_order=self._glt_avalanche_placement,
                        suspended_enabled=self.has_suspended,
                        static_bed=static_bed_enabled(self.data),
                        source_identity=SHA(folder/'general_live_transport_sources.json'),
                        boundary=self.boundary.receipt(), schedule=self.schedule.receipt(),
                        exchange_ledger_algorithm=self.exchange_ledger_algorithm,
                        automatic_CPU_fallback=False,
                        accepted_transaction='segmented_all_reachable_v5', physical_validation=False)
                    if self.implicit_midpoint is not None:
                        receipt.update(implicit_method_identity(
                            coupling=implicit_coupling_identity(self.data),
                            data=self.data))
                        if 'suspended_amr_contract' in self.data:
                            from braids_gpu.implicit_suspension_amr import (
                                method_identity as suspended_amr_identity)
                            receipt.update(suspended_amr_identity(self.data))
                    marker = self.data.get(PUBLIC_ADMISSION_KEY)
                    scalar_public = (isinstance(marker, dict) and
                        marker.get('admission_id') in (
                            SCALAR_PUBLIC_ADMISSION_ID,
                            HIDING_PUBLIC_ADMISSION_ID))
                    if int(self.data['bedlaw']) == 11 or scalar_public:
                        receipt.update(self.sediment.profile_receipt())
                    elif int(self.data['bedlaw']) in (5, 6, 7):
                        receipt.update(self.sediment.profile_receipt())
                    if 'delft_forcing_binding' in self.data:
                        receipt['delft_forcing'] = \
                            _typed_forcing_identity(self.data)
                    bend = getattr(self, 'bend_bundle_v1', None)
                    if bend is not None:
                        receipt.update(bend_bundle_v1=bend.identity, static_bed=not bend.morphology,
                            sediment_owner='experimental_bend_bundle_v1', public_support=False)
                    return receipt
            yield wrap_solver(GeneralLiveTransportSolver)
        finally:
            core.validate_scope = original_scope
    finally:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        verify_sources(folder)
