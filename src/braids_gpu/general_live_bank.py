"""General fixed-grid live-section, native secondary-flow and bank GPU bridge.

This adapter selects no fixture folder and copies current CUDA sources verbatim.
Those sources include the versioned current-interface physical-toe repair;
they are not numerically identical to the pre-repair bank implementation.
The native zero-step export supplies all geometry and controls.  Its private
Python bridges connect the existing mode-9 inlet, mode-4 outlet, resident
schedule and mode-2/3 bank transactions under plain SSPRK(3,3).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from . import g1_inlet_solver as inlet_bridge
from . import live_inlet_modes as inlet
from . import general_live_transport as common


ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / 'braids_support/gpu_fwave'
VARIANT = 'live_normal_critical_bank_ssp33'
TOE_INTERFACE_RULE = 'inventory_bounded_evolving_ordinary_toe_v3'
NORMAL_ROOT_RULE = 'reflective_endpoint_minimum_and_flat_bracket_v1'
NORMAL_TRACE_RULE = 'root_linear_face_moments_owner_hydrostatic_path_v1'
PRIVATE = {name: '_glb_' + name[:-3] for name in common.PRIVATE}
HISTORY = {
    'undercut': 'bank_undercut',
    'reference_height': 'bank_reference_height',
    'toe_height': 'bank_toe_height',
    'reference_length': 'bank_reference_length',
    'toe_cumulative': 'bank_toe_cumulative',
    'collapse_cumulative': 'bank_collapse_cumulative',
    'collapse_count': 'bank_collapse_count',
}
STAGE_SLOTS = (0, 2, 1)
STAGE_FRACTIONS = (0., 1., .5)
SHA = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _mode(data):
    mode = np.asarray(data.get('bank_toe_erosion_mode'))
    require(mode.shape == () and mode.dtype.kind in 'iu' and int(mode) in (2, 3),
            'General live-bank requires native bank mode 2 or 3')
    return int(mode)


def _secondary_model(data):
    value = np.asarray(data.get('secondary_flow_model'))
    require(value.shape == () and value.dtype.kind in 'iu' and
            int(value) in (0, 1, 2),
            'General live-bank secondary model must be integer 0, 1 or 2')
    return int(value)


def validate_boundaries(data):
    """Dispatch on the actual inlet law; never substitute its control code."""
    return common.validate_boundaries(data)


def validate_outlets(data):
    return common.validate_outlets(data)


def validate_sections(data):
    return common.validate_sections(data)


def read_records(folder):
    """Read one ordinary native export, not a validation-folder selector.

    The caller authenticates its schema-5 receipt.  This reader independently
    enforces the exact sidecar inventory and native/NPZ consistency before a
    constructor can use any record.  No native executable or GIS tool is loaded.
    """
    from braids_support.gpu_fwave.coupled_reference import (
        read_records as native_records, read_snapshot)

    folder = Path(folder).resolve()
    require(folder.is_dir(), 'General live-bank prepared folder is missing')
    data = read_snapshot(folder)
    mode = _mode(data)
    bank = ('bank_toe_metadata.bin' if mode == 2 else
            'distributed_bank_metadata.bin')
    sidecars = ('normal_metadata.bin', 'avalanche_metadata.bin',
                'bank_history_metadata.bin', bank)
    if float(data['horizontal_viscosity']) > 0.:
        sidecars += ('coupled_diffusion.bin',)
    require({path.name for path in folder.glob('*.bin') if path.name != 'case.data.bin'} ==
            {'coupled_input.bin', *sidecars},
            'General live-bank native binary sidecar inventory differs')
    for name in sidecars:
        candidate = folder / name
        require(candidate.is_file() and not candidate.is_symlink(),
                'General live-bank sidecar is missing or linked: ' + name)
        for key, value in native_records(candidate).items():
            if name == 'coupled_diffusion.bin':
                key = 'diffusion_' + key
            require(key not in data or np.array_equal(data[key], value),
                    'General live-bank duplicate native record differs: ' + name + '/' + key)
            data[key] = value
    nb = int(data['nb'])
    for name in ('section_seed_node', 'boundary_seed_position'):
        native = np.asarray(data['native_' + name])
        require(native.shape == (nb,) and native.dtype == np.int32 and
                np.all(native >= 0),
                'General live-bank native section seed differs: ' + name)
        data[name] = np.ascontiguousarray(native - 1)
    npz = folder / 'coupled_input.npz'
    if npz.exists():
        require(not npz.is_symlink(), 'General live-bank NPZ is linked')
        with np.load(npz, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)) and
                    set(archive.files) == set(data) and
                    all(np.array_equal(archive[key], data[key]) for key in archive.files),
                    'General live-bank NPZ differs from native records')
    return data


def validate_scope(data, *, start, end, interval, dtmax, launch_mode,
                   integrator, experimental_morfac_mode3=False,
                   experimental_morfac_mode3_max=25.0):
    """Admit legal native live-bank controls without fixture dimensions."""
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.coupled_solver import (
        normalized, validate_scope as base_scope)
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope
    from braids_support.gpu_fwave.sediment_transport import configuration as sediment_config

    secondary_model = _secondary_model(data)
    data = normalized(data)
    mode = _mode(data)
    validate_boundaries(data)
    require(launch_mode == 'plain' and integrator == 'ssp33',
            'General live-bank requires explicit plain SSPRK33')
    require(np.isfinite([start, end, interval, dtmax]).all() and
            start >= 0. and end > start and interval > 0. and dtmax > 0. and
            end <= float(data['endtime']) and dtmax <= float(data['dtmax']),
            'General live-bank run clock is outside the native export')
    require(type(experimental_morfac_mode3) is bool,
            'Experimental bank MorFac selector must be boolean')
    require(np.asarray(experimental_morfac_mode3_max).shape == () and
            np.isfinite(experimental_morfac_mode3_max) and
            1. <= float(experimental_morfac_mode3_max) <= 1000.,
            'Experimental bank MorFac limit must be in [1, 1000]')
    morfac = float(data['morphological_factor'])
    morfac_ok = (morfac == 1. or
        (experimental_morfac_mode3 and mode == 3 and
         1. < morfac <= float(experimental_morfac_mode3_max) and
         not np.any(data.get('nmorfac', 0))))
    require(int(data['scheme']) == 6 and int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0 and
            int(data['bedlaw']) == 2 and int(data['mpm_friction_mode']) in (0, 1) and
            int(data['exchange_mode']) == 0 and
            int(data['bank_failure']) == 1 and
            morfac_ok,
            'General live-bank requires unsuspended native-secondary MPM')
    viscosity = np.asarray(data['horizontal_viscosity'])
    require(viscosity.shape == () and viscosity.dtype.kind in 'fiu' and
            np.isfinite(viscosity) and float(viscosity) >= 0.,
            'General live-bank viscosity must be finite and nonnegative')
    if float(viscosity) > 0.:
        from braids_support.gpu_fwave.momentum_diffusion import validate_scope as diffusion_scope
        common._integer(data, 'horizontal_viscosity_mode', (1,))
        common._integer(data, 'horizontal_viscosity_geometry', (1,))
        diffusion_scope(data)
    for name, allowed in (('hiding_exposure', (0, 2, 3, 4, 5)), ('bed_slope', (0, 2)),
                          ('bedload_adaptation', (0, 1))):
        value = np.asarray(data.get(name))
        require(value.shape == () and value.dtype.kind in 'biu' and
                int(value) in allowed,
                'General live-bank unsupported graded control: ' + name)
    adaptation = bool(data['bedload_adaptation'])
    length = np.asarray(data.get('bedload_adaptation_length_diameters'))
    require(length.shape == () and length.dtype.kind in 'fiu' and
            np.isfinite(length) and (not adaptation or float(length) > 0.),
            'General live-bank adaptation requires a finite positive enabled length')
    # Check the actual material, slope, hiding and adaptation parameters before
    # constructing CUDA objects. Only boundary admission is detached here:
    # the real surveyed normal/critical laws were checked above. Their controls
    # and the full physics remain unchanged in every executed operator.
    modes = np.asarray(data['mode'])
    checked_modes = np.where(modes == 9, 6, np.where(modes == 4, 2, modes)).astype(np.int32)
    sediment_config(dict(data, mode=checked_modes), allow_bedform=bool(np.any(modes == 1)),
        allow_scheduled=True, allow_general_avalanche=True,
        allow_physical_toe_mode2=(mode == 2),
        allow_distributed_bank_mode3=(mode == 3), allow_adaptation=adaptation)
    for name in ('amr_enabled', 'bedform_roughness',
                 'bank_front_tracking', 'bank_avalanche', 'bank_erosion_fraction',
                 'nmorfac', 'morphological_acceleration',
                 'viscosity_equilibrated_inlet', 'suspended_transport_active'):
        require(not np.any(data.get(name, 0)),
                'General live-bank excludes unported control: ' + name)
    require(not np.any(data['suspended_mass']) and
            not np.any(data['fraction_concentration']) and
            not np.any(data.get('fraction_series_concentration', 0)),
            'General live-bank excludes suspended concentration/exchange')
    nc, ng, nl, nb = (int(data[key]) for key in
                      ('nc', 'nfrac', 'nbed_layers', 'nb'))
    prognostic = secondary_model == 2
    physical_states, physical_ledgers = 3 + ng * (1 + nl), 1 + 2 * ng
    require(nc > 0 and 1 <= ng <= 32 and 2 <= nl <= 33 and nb >= 1 and
            int(data['nsubstrate_layers']) == nl - 1 and
            np.shape(data['q']) == (3, nc) and
            np.shape(data['layer_mass']) == (nl, ng, nc) and
            int(data['secondary_index']) == (physical_states if prognostic else -1) and
            int(data['secondary_ledger_index']) == (physical_ledgers if prognostic else -1) and
            int(data['nstate']) == physical_states + int(prognostic) and
            int(data['nledger']) == physical_ledgers + int(prognostic) and
            ('state' not in data or np.shape(data['state']) ==
             (physical_states + int(prognostic), nc)),
            'General live-bank state/ledger layout differs from generic MPM')
    require(int(data['hiding_exposure']) == 0 or ng > 1,
            'General live-bank hiding/exposure requires multiple sediment fractions')
    q = np.asarray(data['q'])
    require(np.isfinite(q).all() and np.all(q[0] >= 0.) and
            np.any(q[0] > float(data['dry_tolerance'])) and
            not np.any(q[1:, q[0] <= float(data['dry_tolerance'])]),
            'General live-bank initial wet/dry hydraulic state differs')
    secondary_scope(data)
    require(np.asarray(data['secondary_mass']).shape == (nc,) and
            np.isfinite(data['secondary_mass']).all() and
            (prognostic or not np.any(data['secondary_mass'])),
            'General live-bank secondary inventory differs from selected model')
    for accepted, exported in HISTORY.items():
        value = np.asarray(data[exported])
        require(value.shape == (nc,) and
                value.dtype == (np.int32 if accepted == 'collapse_count'
                                else np.float64) and
                np.all(value >= 0) and
                (value.dtype != np.float64 or np.isfinite(value).all()),
                'General live-bank initial history differs: ' + exported)
    if _mode(data)==2:
        # Native static-map exports have no authenticated evolving support.
        # Continuing nonzero geometric memory requires our schema-2 checkpoint,
        # never guessing a support identity from a newly derived wet map.
        require(not any(np.any(data[HISTORY[key]]) for key in
            ('undercut','reference_height','toe_height','reference_length')),
            'Nonzero initial bank geometry requires an evolving-map checkpoint')
    # The stock bank validators remain valuable, but their scheduling policy
    # predates the live-section bridge.  Remove only schedule counts in this
    # detached validation view; validate the real tables immediately below.
    bank_view = dict(data,
        # The old bank-only validator predates composition with graded slope
        # transport/adaptation. Their real values were independently checked
        # above; this detached view retains all its bank geometry/map checks.
        # The actual friction0/1 capacity was validated by sediment_config;
        # the historical bank-only guard uses friction1 for geometry checks.
        hiding_exposure=np.int32(0), bed_slope=np.int32(0), mpm_friction_mode=np.int32(1),
        bedload_adaptation=np.int32(0),
        horizontal_viscosity=np.float64(0.),
        # The detached historical bank validator is intentionally bounded at
        # MorFac 25.  The actual validation-only factor was checked above and
        # is retained by the composed solver; clamp only this geometry view.
        morfac=np.float64(min(morfac, 25.)),
        morphological_factor=np.float64(min(morfac, 25.)),
        nhydro_series=np.zeros_like(data['nhydro_series']),
        nfraction_series=np.zeros_like(data['nfraction_series']),
        nseries=np.zeros_like(data['nseries']))
    # Historical bank validators exclude some secondary branches. The actual
    # branch and its native inventory layout were checked above; this detached
    # bank-geometry check must not impersonate the prognostic G3 specialization.
    bank_view.update(secondary_flow_model=0, secondary_flow=0,
                     secondary_flow_transport_direction=0,
                     secondary_flow_momentum_beta=0., secondary_index=-1,
                     secondary_mass=np.zeros(nc, np.float64))
    base_scope(bank_view, bank_avalanche_mode2_variant=(mode == 2),
               g3_bank_secondary_variant=False,
               evolving_toe_variant=(mode == 2),
               distributed_bank_avalanche_mode3_variant=(mode == 3),
               experimental_morfac_mode3=experimental_morfac_mode3)
    tables = validate_tables(dict(data,
        section_control=np.zeros(nb, np.int32),
        sampled=np.zeros(int(data['nf']), np.int32)))
    for boundary in range(nb):
        count = int(tables['nhydro_series'][boundary])
        if count:
            times = tables['hydro_series_time'][:count, boundary]
            require(times[0] <= start and times[-1] >= end,
                    'General live-bank hydraulic clock does not cover the run')
            if modes[boundary] == 9:
                require(np.all(tables['hydro_series_q'][:count, boundary] > 0),
                        'General live-bank surveyed inlet requires positive scheduled Q')
        elif modes[boundary] == 9:
            require(float(tables['forcing'][0, boundary]) > 0.,
                    'General live-bank surveyed inlet requires positive constant Q')
        for grain in range(ng):
            count = int(tables['nfraction_series'][grain, boundary])
            if count:
                times = tables['fraction_series_time'][:count, grain, boundary]
                require(times[0] <= start and times[-1] >= end,
                        'General live-bank class feed clock does not cover the run')
    return BoundaryEvents.from_tables(data)


def _validate_prepared(data, *, experimental_morfac_mode3=False,
                       experimental_morfac_mode3_max=25.0):
    return validate_scope(data, start=float(data.get('time', 0.)),
        end=float(data['endtime']), interval=float(data['output_dt']),
        dtmax=float(data['dtmax']), launch_mode='plain', integrator='ssp33',
        experimental_morfac_mode3=experimental_morfac_mode3,
        experimental_morfac_mode3_max=experimental_morfac_mode3_max)


def _transform(name, source):
    """Reuse common boundary composition; keep bank sources transactional."""
    result = common._transform(name, source).replace('._glt_', '._glb_')
    result = result.replace('from braids_gpu.general_live_transport import',
                            'from braids_gpu.general_live_bank import')
    if name == 'coupled_solver.py':
        old = '                                     post_source=self.post_source,'
        require(result.count(old) == 1, 'Bank diffusion source hook anchor differs')
        result = result.replace(old, '                                     post_source=None,', 1)
        old = '''                                     pre_commit=(self.bank_avalanche_pre_commit if self.bank_avalanche_mode2_variant else
                                                 self.distributed_bank_avalanche_pre_commit if self.distributed_bank_avalanche_mode3_variant else
                                                 self.physical_toe_pre_commit if self.physical_toe_mode2_variant else
                                                 self.distributed_bank_pre_commit if self.distributed_bank_mode3_variant else None),'''
        require(result.count(old) == 1, 'Bank final private source anchor differs')
        result = result.replace(old, '                                     pre_commit=self._glb_post_bank_sources,', 1)
    return result


def prepare_sources(folder, data, *, kernel_source=None):
    """Freeze the executed Python/CUDA graph without changing CUDA bodies."""
    _validate_prepared(data)
    folder = Path(folder).resolve()
    require(not folder.exists(), 'Preserve existing general live-bank source archive')
    folder.mkdir(parents=True)
    sources = {}
    origin=SPATIAL if kernel_source is None else Path(kernel_source).resolve()
    require(origin.is_dir() and origin.is_relative_to(ROOT.resolve()) and
            not origin.is_symlink(),
            'Live-bank kernel origin must be a package-relative source directory')
    originals=list(common._source_files(origin))
    require(originals and all(path.is_file() and not path.is_symlink() for path in originals),
            'Live-bank source closure is empty, missing, or linked')
    originals.append(ROOT/'braids_gpu/general_live_bank.py')
    for original in originals:
        relative = original.relative_to(ROOT).as_posix()
        target = (folder / original.name if original.suffix == '.cu' else
                  folder / 'python_snapshot' / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        sources[relative] = SHA(original)
    generated = {}
    for name, private in PRIVATE.items():
        target = folder / (private + '.py')
        target.write_text(_transform(name, common._original(name).read_text(encoding='utf-8')),
                          encoding='utf-8')
        generated[name] = dict(name=target.name, sha256=SHA(target))
    receipt = dict(schema=1, variant=VARIANT, sources=sources,
        generated=generated, numerical_CUDA_unchanged=False,
        CUDA_copied_verbatim=True, physical_toe_interface_rule=TOE_INTERFACE_RULE,
        normal_root_rule=NORMAL_ROOT_RULE,
        normal_trace_rule=NORMAL_TRACE_RULE,
        integrator='SSPRK33', stage_fractions=list(STAGE_FRACTIONS),
        bank_mode=int(data['bank_toe_erosion_mode']),
        nfrac=int(data['nfrac']), nbed_layers=int(data['nbed_layers']),
        physical_validation=False, Delft3D_source_modified=False)
    if kernel_source is not None:
        receipt['kernel_origin']=origin.relative_to(ROOT.resolve()).as_posix()
    (folder / 'general_live_bank_sources.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    verify_sources(folder)
    return receipt


def verify_sources(folder):
    folder = Path(folder).resolve()
    receipt = json.loads((folder / 'general_live_bank_sources.json').read_text(
        encoding='utf-8'))
    require(receipt.get('schema') == 1 and receipt.get('variant') == VARIANT and
            receipt.get('numerical_CUDA_unchanged') is False and
            receipt.get('CUDA_copied_verbatim') is True and
            receipt.get('physical_toe_interface_rule') == TOE_INTERFACE_RULE and
            receipt.get('normal_root_rule') == NORMAL_ROOT_RULE,
            'General live-bank source receipt differs')
    require(receipt.get('normal_trace_rule') == NORMAL_TRACE_RULE,
            'General live-bank normal trace rule differs')
    if 'kernel_origin' in receipt:
        origin=Path(receipt['kernel_origin'])
        require(not origin.is_absolute() and '..' not in origin.parts and
                (ROOT/origin).resolve().is_relative_to(ROOT.resolve()),
                'General live-bank kernel origin is not package-relative')
    origin = ROOT/receipt.get('kernel_origin', 'braids_support/gpu_fwave')
    expected_sources = {path.relative_to(ROOT).as_posix() for path in common._source_files(origin)}
    expected_sources.add('braids_gpu/general_live_bank.py')
    require(set(receipt['sources']) == expected_sources, 'General live-bank source inventory differs')
    require(set(receipt['generated']) == set(PRIVATE), 'General live-bank generated inventory differs')
    for relative, digest in receipt['sources'].items():
        relative_path=Path(relative)
        require(not relative_path.is_absolute() and '..' not in relative_path.parts and
                (ROOT/relative_path).resolve().is_relative_to(ROOT.resolve()),
                'General live-bank source path escapes package')
        original = ROOT / relative
        if original.suffix=='.cu' and 'kernel_origin' in receipt:
            require(relative_path.parent==Path(receipt['kernel_origin']) and
                    original.is_file() and not original.is_symlink(),
                    'General live-bank CUDA origin differs')
        saved = (folder / original.name if original.suffix == '.cu' else
                 folder / 'python_snapshot' / relative)
        require(SHA(original) == digest and SHA(saved) == digest,
                'General live-bank source drift: ' + relative)
    for name, row in receipt['generated'].items():
        require(row['name'] == PRIVATE[name]+'.py', 'General live-bank generated path differs')
        target = folder / row['name']
        require(SHA(target) == row['sha256'] and
                target.read_text(encoding='utf-8') ==
                _transform(name, common._original(name).read_text(encoding='utf-8')),
                'General live-bank generated bridge drift: ' + name)
    return receipt


@contextmanager
def constructors(folder, data, *, experimental_morfac_mode3=False,
                 experimental_morfac_mode3_max=25.0):
    """Yield a full-rollback solver with authenticated current bank kernels."""
    from .a1_full_array_transaction import wrap_solver
    from braids_support.gpu_fwave.boundary_events import tolerance

    require(type(experimental_morfac_mode3) is bool,
            'Experimental bank MorFac selector must be boolean')
    _validate_prepared(data,
        experimental_morfac_mode3=experimental_morfac_mode3,
        experimental_morfac_mode3_max=experimental_morfac_mode3_max)
    folder = Path(folder).resolve()
    verify_sources(folder)
    prefix = 'braids_support.gpu_fwave.'
    require(not any(prefix + private in sys.modules for private in PRIVATE.values()),
            'General live-bank private modules already loaded')
    loaded = []
    modules = {}
    try:
        for name, private in PRIVATE.items():
            key = prefix + private
            spec = importlib.util.spec_from_file_location(key,
                folder / (private + '.py'))
            require(spec is not None and spec.loader is not None,
                    'General live-bank private module is missing: ' + private)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            loaded.append(key)
            spec.loader.exec_module(module)
            modules[name] = module
        core = modules['coupled_solver.py']
        core.BoundaryControls = common._boundary_factory(
            modules, data, source_folder=folder, integrator_token='ssp33',
            euler_stages=3, integrator='SSPRK33')
        schedule = modules['boundary_schedule.py']
        original_scope = core.validate_scope
        mode = _mode(data)
        secondary_model = _secondary_model(data)
        g3_secondary = mode == 2 and secondary_model == 2
        adaptation = bool(data['bedload_adaptation'])

        def admitted_scope(actual, **flags):
            require(_mode(actual) == mode and
                    _secondary_model(actual) == secondary_model and
                    bool(actual['bedload_adaptation']) == adaptation and
                    bool(flags.get('bank_avalanche_mode2_variant')) == (mode == 2) and
                    bool(flags.get('g3_bank_secondary_variant')) == g3_secondary and
                    bool(flags.get('evolving_toe_variant')) == (mode == 2) and
                    bool(flags.get('distributed_bank_avalanche_mode3_variant')) == (mode == 3) and
                    bool(flags.get('bedload_adaptation_variant')) == adaptation and
                    not any(flags.get(key) for key in
                        ('scheduled', 'bedform_variant', 'general_avalanche_variant',
                         'physical_toe_mode2_variant', 'distributed_bank_mode3_variant')),
                    'General live-bank constructor bank-law flags differ')
            return _validate_prepared(actual,
                experimental_morfac_mode3=experimental_morfac_mode3,
                experimental_morfac_mode3_max=experimental_morfac_mode3_max) is not None

        core.validate_scope = admitted_scope
        try:
            class GeneralLiveBankSolver(core.CoupledSolver):
                def __init__(self, raw, *, source_folder=folder):
                    _validate_prepared(raw,
                        experimental_morfac_mode3=experimental_morfac_mode3,
                        experimental_morfac_mode3_max=experimental_morfac_mode3_max)
                    require(_mode(raw) == mode and
                            _secondary_model(raw) == secondary_model and
                            bool(raw['bedload_adaptation']) == adaptation and
                            np.array_equal(raw['mode'], data['mode']) and
                            np.array_equal(raw['section_control'], data['section_control']),
                            'General live-bank constructor physics or section law differs')
                    require(Path(source_folder).resolve() == folder,
                            'General live-bank source folder differs')
                    super().__init__(raw, source_folder=folder, integrator='ssp33',
                        bank_avalanche_mode2_variant=(mode == 2),
                        g3_bank_secondary_variant=g3_secondary,
                        evolving_toe_variant=(mode == 2),
                        distributed_bank_avalanche_mode3_variant=(mode == 3),
                        bedload_adaptation_variant=adaptation)
                    # Establish this stable Euler alias before the complete
                    # array transaction inventories any reachable GPU buffer.
                    self.flux.inputs['q'] = self.stepper.stage['q']
                    self.boundary.bind_sediment_hydraulic(
                        self.sediment.geometry['bc_hydraulic'])
                    self.boundary.bind_stage_state(self.stepper.stage['q'])
                    if int(raw['mode'][0]) == 9 and int(raw['section_control'][0]) == 1:
                        self.flux.integrated_section_flux = self.boundary.integrated_flux
                    self.schedule = schedule.ResidentBoundarySchedule(
                        raw, source_path=folder / 'boundary_schedule.cu')
                    for accepted, exported in HISTORY.items():
                        self.bank_history[accepted].set(np.asarray(raw[exported]))
                    for value in self.schedule.outputs.values():
                        value.fill(0)
                    for records in (getattr(self.boundary, 'normal_records', {}),
                                    self.boundary.outlet_records):
                        for value in records.values():
                            value.fill(0)
                    if hasattr(self.boundary, 'normal_root'):
                        self.boundary.normal_root.invalidate_cache()
                    self._glb_stage = None
                    self._glb_times = None
                    self._glb_accepted_time = float(raw.get('time', 0.))

                def _glb_post_bank_sources(self, state, dt):
                    """Bank/collapse, avalanche, then diffusion on private state.

                    Native diffusion sees the updated bed. Global state, bank
                    history, toe map and diffusion cache still commit together.
                    """
                    flag = (self.bank_avalanche_pre_commit(state, dt) if mode == 2 else
                            self.distributed_bank_avalanche_pre_commit(state, dt))
                    if self.diffusion is not None and int(flag.item()) == 0:
                        self._merge(self.post_source(state, dt), flag)
                    return flag

                def euler(self, state, dt, current):
                    index = self._glb_stage
                    require(type(index) is int and 0 <= index < 3 and
                            abs(float(current) - self._glb_times[index]) <=
                            tolerance(current),
                            'General live-bank SSPRK33 Euler clock/order differs')
                    slot = STAGE_SLOTS[index]
                    values = self.schedule.outputs
                    cp = self.cp
                    cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][slot])
                    cp.copyto(self.sediment.geometry['bc_hydraulic'],
                              values['hydraulic'][slot])
                    cp.copyto(self.sediment.geometry['feed'], values['feed'][slot])
                    result = super().euler(state, dt, current)
                    self.boundary.record_stage(index)
                    self._merge(values['status'][slot])
                    self._glb_stage += 1
                    return result

                def _a1_trial_diagnostics(self):
                    # wrap_solver snapshots every reachable device array,
                    # including root cache, bank histories and toe diagnostics.
                    saved = dict(accepted_time=self._glb_accepted_time)
                    if adaptation:
                        operator = self.sediment.adaptation
                        saved['adaptation'] = (operator.iterations, operator.residual,
                            None if self.sediment.adaptation_result is None else
                            dict(self.sediment.adaptation_result))
                    return saved

                def _a1_restore_diagnostics(self, saved, *, where=True):
                    if where:
                        self._glb_accepted_time = saved['accepted_time']
                        if adaptation:
                            (self.sediment.adaptation.iterations,
                             self.sediment.adaptation.residual,
                             self.sediment.adaptation_result) = saved['adaptation']

                def step(self, dt, current, *, commit=True):
                    require(self._glb_stage is None,
                            'Concurrent general live-bank trials are unsupported')
                    self.boundary.check_storage()
                    saved = self._a1_trial_diagnostics()
                    self._glb_stage = 0
                    self._glb_times = tuple(float(current + fraction * dt)
                                            for fraction in STAGE_FRACTIONS)
                    try:
                        self.boundary.begin_trial()
                        self.schedule.evaluate(current, dt)
                        result = super().step(dt, current, commit=commit)
                        require(self._glb_stage == 3,
                                'General live-bank SSPRK33 omitted an Euler stage')
                        accepted = commit and not bool(result['status'].item())
                        self._a1_restore_diagnostics(saved,
                            where=(result['status'] != 0) if commit else True)
                        if accepted:
                            self._glb_accepted_time = float(current + dt)
                        return result
                    except Exception:
                        self._a1_restore_diagnostics(saved)
                        raise
                    finally:
                        self._glb_stage = None
                        self._glb_times = None

                def fields(self, *, current=None):
                    if adaptation:
                        require(self._glb_stage is None,
                                'Accepted adaptation output cannot run inside a trial')
                        current = self._glb_accepted_time if current is None else float(current)
                        require(np.isfinite(current) and current >= 0.,
                                'Invalid accepted adaptation output clock')
                        # Schedule kernels require an advancing trial. Only
                        # slot zero is consumed here: it is exactly the accepted
                        # time, unlike the final SSPRK33 Euler stage t+dt/2.
                        diagnostic_dt = max(float(self.data['dtmax']),
                                            float(np.spacing(current)))
                        values = self.schedule.evaluate(current, diagnostic_dt)
                        require(not bool(self.cp.any(values['status'][0]).item()),
                                'Invalid accepted adaptation output forcing')
                        self.cp.copyto(self.sediment.geometry['feed'], values['feed'][0])
                    return super().fields()

                def candidate_receipt(self):
                    return dict(variant=VARIANT, integrator='SSPRK33',
                        stage_slots=list(STAGE_SLOTS), bank_mode=mode,
                        secondary_flow_model=int(self.data['secondary_flow_model']),
                        secondary_transport_direction=bool(self.data['secondary_flow_transport_direction']),
                        secondary_momentum_beta=float(self.data['secondary_flow_momentum_beta']),
                        hiding_exposure=int(self.data['hiding_exposure']),
                        bed_slope=int(self.data['bed_slope']), adaptation_enabled=adaptation,
                        horizontal_viscosity=float(self.data['horizontal_viscosity']),
                        horizontal_viscosity_mode=int(self.data['horizontal_viscosity_mode']),
                        horizontal_viscosity_geometry=int(self.data['horizontal_viscosity_geometry']),
                        experimental_morfac_mode3=experimental_morfac_mode3,
                        experimental_morfac_mode3_max=float(experimental_morfac_mode3_max),
                        diffusion_order='after_bank_and_avalanche',
                        adaptation_output_clock='accepted_time',
                        source_identity=SHA(folder / 'general_live_bank_sources.json'),
                        boundary=self.boundary.receipt(),
                        schedule=self.schedule.receipt(), backend='gpu',
                        automatic_CPU_fallback=False,
                        accepted_transaction='segmented_all_reachable_v5',
                        physical_validation=False)

            yield wrap_solver(GeneralLiveBankSolver)
        finally:
            core.validate_scope = original_scope
    finally:
        for key in reversed(loaded):
            sys.modules.pop(key, None)
        verify_sources(folder)
