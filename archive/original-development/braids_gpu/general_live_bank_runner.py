"""Ordinary-deck lifecycle for the common live-boundary GPU bank solver.

Explicit native controls select physical-toe or distributed-slope erosion.
Accepted state, bank histories and signed ledgers are checkpointed; derived
boundary workspaces are rebuilt. No fixture path selects numerical physics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .atomic_output import save_arrays_atomic as save_arrays, write_json_atomic as write_json

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'live_normal_critical_bank_ssp33'
EXPORTER_SHA = 'f9dc86987a04328e778ce790bb89133aa7c97943b550b2b335508aa3a8195ab5'
COMMON = frozenset(('coupled_input.bin', 'normal_metadata.bin',
                   'avalanche_metadata.bin', 'bank_history_metadata.bin'))
MODE_SIDECAR = {2:'bank_toe_metadata.bin', 3:'distributed_bank_metadata.bin'}
HISTORY = ('undercut', 'reference_height', 'toe_height', 'reference_length',
           'toe_cumulative', 'collapse_cumulative', 'collapse_count')
ADAPTATION_DIAGNOSTICS = frozenset(('bedload_adaptation_equilibrium',
    'bedload_adaptation_capacity', 'bedload_adaptation_iterations',
    'bedload_adaptation_residual'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def packaged_native_exporter(relative, executable_name, expected_sha256):
    """Resolve an authenticated zero-step exporter from an extracted worker."""
    from .graded_native_exporter import REQUIRED_DLLS, export_environment

    worker = ROOT
    manifest_path = worker/'PACKAGE-MANIFEST.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            'Packaged native exporter requires a worker manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest.get('files')
    require(isinstance(files, dict), 'Packaged native exporter file inventory is missing')
    runtime = worker/relative
    executable = runtime/executable_name
    build_path = runtime/'build.json'
    require(runtime.is_dir() and not runtime.is_symlink() and
            build_path.is_file() and not build_path.is_symlink() and
            files.get(relative+'/build.json') == sha(build_path),
            'Packaged native exporter receipt differs')
    build = json.loads(build_path.read_text(encoding='utf-8'))
    require(build.get('schema') == 1 and
            build.get('kind') == 'portable integrated native initializer' and
            build.get('executable') == executable_name and
            build.get('executable_sha256') == expected_sha256 and
            build.get('native_solver_steps_performed') == 0 and
            build.get('numerical_source_unchanged') is True and
            build.get('Delft3D_source_modified') is False and
            executable.is_file() and not executable.is_symlink() and
            files.get(relative+'/'+executable_name) == expected_sha256 and
            sha(executable) == expected_sha256,
            'Packaged native exporter identity differs')
    sources = build.get('source_sha256')
    source_dir = runtime/'source'
    require(isinstance(sources, dict) and sources and
            source_dir.is_dir() and not source_dir.is_symlink() and
            {path.name for path in source_dir.iterdir()} == set(sources),
            'Packaged native exporter source inventory differs')
    for name, identity in sources.items():
        require(Path(name).name == name and name not in ('.','..') and
                isinstance(identity, str) and len(identity) == 64,
                'Packaged native exporter source identity is invalid')
        source = source_dir/name
        require(source.is_file() and not source.is_symlink() and
                files.get(relative+'/source/'+name) == identity and sha(source) == identity,
                'Packaged native exporter source differs: '+name)
    dlls = build.get('dll_sha256')
    require(isinstance(dlls, dict) and set(REQUIRED_DLLS) <= set(dlls) and
            {path.name.lower() for path in runtime.glob('*.dll')} == set(dlls),
            'Packaged native exporter DLL inventory differs')
    for name, identity in dlls.items():
        require(Path(name).name == name and name.lower() == name and
                isinstance(identity, str) and len(identity) == 64,
                'Packaged native exporter DLL identity is invalid')
        path = runtime/name
        require(path.is_file() and not path.is_symlink() and
                files.get(relative+'/'+name) == identity and sha(path) == identity,
                'Packaged native exporter DLL differs: '+name)
    return runtime, executable, export_environment(runtime)


def digest(fields):
    value = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        value.update(name.encode()+b'\0'+array.dtype.str.encode()+b'\0')
        value.update(json.dumps(array.shape).encode()+b'\0'+array.tobytes())
    return value.hexdigest()


def artifact_names(mode, viscosity=0.):
    require(type(mode) is int and mode in MODE_SIDECAR, 'Unsupported native bank mode')
    return (COMMON | {MODE_SIDECAR[mode], 'coupled_input.npz'} |
            ({'coupled_diffusion.bin'} if viscosity > 0 else set()))


def authenticated_member(folder, name, identity, *, nested=False):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Invalid bank preparation member')
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and parts.as_posix() == name and
            all(part not in ('.', '..', '') for part in parts.parts) and
            (nested or len(parts.parts) == 1), 'Unsafe bank preparation path: '+name)
    target = folder
    for part in parts.parts:
        target /= part
        require(not target.is_symlink(), 'Linked bank preparation member: '+name)
    require(target.is_file() and target.resolve().is_relative_to(folder) and
            isinstance(identity, str) and len(identity) == 64 and sha(target) == identity,
            'Bank preparation member differs: '+name)


def read_prepared(folder):
    from . import general_live_bank as candidate
    folder = Path(folder).resolve()
    path = folder/'preparation_receipt.json'
    require(path.is_file() and not path.is_symlink(), 'Missing bank preparation receipt')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    require(receipt.get('schema') == 5 and receipt.get('variant') == VARIANT and
            receipt.get('exporter_executable_sha256') == EXPORTER_SHA and
            receipt.get('initialization_CPU_only') is True and
            type(receipt.get('native_solver_steps_performed')) is int and
            receipt['native_solver_steps_performed'] == 0 and
            receipt.get('GPU_solver_executed') is False, 'Bank zero-step preparation identity differs')
    mode = receipt.get('bank_mode')
    names = artifact_names(mode)
    artifacts = receipt.get('artifacts', {})
    require(set(artifacts) in (names, names | {'coupled_diffusion.bin'}),
            'Bank preparation artifact inventory differs')
    for name, identity in artifacts.items():
        authenticated_member(folder, name, identity)
    for name in ('scheduled_metadata.bin',
                 MODE_SIDECAR[3 if mode == 2 else 2]):
        require(not (folder/name).exists(), 'Bank preparation excludes sidecar: '+name)
    origins = receipt.get('origin', {}).get('files', {})
    require({'case.data', 'geometry.npz'} <= set(origins),
            'Bank native deck and geometry identity missing')
    for name, identity in origins.items():
        authenticated_member(folder, name, identity, nested=True)
    for line in (folder/'case.data').read_text().splitlines():
        tokens = line.split()
        if len(tokens) == 2 and re.fullmatch(r'[0-9a-fA-F]{64}', tokens[1]):
            require(origins.get(tokens[0]) == tokens[1].lower(),
                    'Bank native binary deck dependency is not authenticated')
    data = candidate.read_records(folder)
    require(int(data['bank_toe_erosion_mode']) == mode, 'Bank receipt/native mode differs')
    require(set(artifacts) == artifact_names(mode, float(data['horizontal_viscosity'])),
            'Bank viscosity/cache artifacts differ')
    with np.load(folder/'coupled_input.npz', allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == set(data)
            and all(archive[k].dtype == np.asarray(data[k]).dtype and
                    np.array_equal(archive[k], data[k]) for k in archive.files),
            'Bank NPZ differs from native records')
    identity = hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()
    return data, identity, sha(path)


def scope(data, **overrides):
    from . import general_live_bank as candidate
    controls = dict(start=float(data.get('time', 0.)), end=float(data['endtime']),
        interval=float(data['output_dt']), dtmax=float(data['dtmax']),
        launch_mode='plain', integrator='ssp33')
    controls.update(overrides)
    return candidate.validate_scope(data, **controls)


def prepare(deck, output, *, stop_file=None):
    from . import general_live_bank as candidate
    from .graded_native_exporter import export_environment, REQUIRED_DLLS
    from validation.gpu_fwave.coupled_reference import copy_inputs
    deck, output = Path(deck).resolve(), Path(output).resolve()
    require(deck.name == 'case.data', 'Bank preparation requires a native case.data deck')
    def check_stop():
        if stop_file is not None and Path(stop_file).exists():
            raise InterruptedError('GPU bank preparation cancelled')
    check_stop()
    if (ROOT/'PACKAGE-MANIFEST.json').exists():
        runtime, executable, env = packaged_native_exporter(
            'braids_gpu/native_bank_exporter_v1','xbraids_live_bank_export.exe',EXPORTER_SHA)
    else:
        runtime = ROOT/'validation/g3_live_bank/runs/live_combined_export_20261004_v2/runtime'
        executable = runtime/'xbraids_live_bank_export.exe'
        require(executable.is_file() and sha(executable) == EXPORTER_SHA,
                'Pinned combined bank zero-step exporter differs')
        build = json.loads((runtime/'build.json').read_text())
        require(build['executable_sha256'] == EXPORTER_SHA, 'Bank exporter build receipt differs')
        compiler = Path(build['command'][0]).parent
        require(all((compiler/name).is_file() for name in REQUIRED_DLLS), 'Bank initializer DLLs missing')
        env = export_environment(runtime)
        env['PATH'] = str(compiler)+os.pathsep+env['PATH']
    origin = copy_inputs(deck, output, stop_file=stop_file)
    with (output/'initial.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([str(executable), deck.name], cwd=output, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        began = time.monotonic()
        try:
            while process.poll() is None:
                check_stop()
                if time.monotonic()-began > 240:
                    raise TimeoutError('Bank zero-step initialization exceeded 240 seconds')
                time.sleep(.05)
        except (InterruptedError, TimeoutError):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, [str(executable), deck.name])
    check_stop()
    data = candidate.read_records(output)
    events = scope(data)
    mode = int(data['bank_toe_erosion_mode'])
    save_arrays(output/'coupled_input.npz', **data)
    receipt = dict(schema=5, variant=VARIANT, bank_mode=mode, origin=origin,
        artifacts={name:sha(output/name) for name in sorted(artifact_names(mode, float(data['horizontal_viscosity'])))},
        exporter_executable_sha256=EXPORTER_SHA, initialization_CPU_only=True,
        native_solver_steps_performed=0, GPU_solver_executed=False)
    write_json(output/'preparation_receipt.json', receipt)
    read_prepared(output)
    return dict(status='prepared', output=str(output), variant=VARIANT, bank_mode=mode,
        cells=int(data['nc']), fractions=int(data['nfrac']), layers=int(data['nbed_layers']),
        forcing_knots_s=events.knots.tolist(), native_solver_steps_performed=0,
        preparation_receipt_sha256=sha(output/'preparation_receipt.json'), archive_bound=False)


def archive_sources(folder, data):
    from . import general_live_bank as candidate
    folder = Path(folder)
    candidate.prepare_sources(folder, data)
    for name in ('general_live_bank_runner.py', 'fixed_grid_lifecycle.py', 'integrated.py',
                 'runner.py', '__main__.py', 'atomic_output.py'):
        copied = folder/'worker_python'/name
        copied.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'braids_gpu'/name, copied)
    shutil.copy2(ROOT/'braids_core/gpu_capabilities.py', folder/'worker_python/gpu_capabilities.py')
    files = {file.relative_to(folder).as_posix():sha(file) for file in folder.rglob('*') if file.is_file()}
    write_json(folder/'runtime_manifest.json', dict(schema=1, variant=VARIANT,
        source_files=files, compact_checkpoint_schema=(2 if int(data['bank_toe_erosion_mode'])==2 else 1),
        numerical_CUDA_unchanged=False,
        CUDA_copied_verbatim=True, physical_toe_interface_rule=candidate.TOE_INTERFACE_RULE,
        normal_root_rule=candidate.NORMAL_ROOT_RULE))
    return sha(folder/'runtime_manifest.json')


def checkpoint_preflight(path, data, events, *, identities, start, end, expected_schema=None):
    from .runner import validate_bank_toe_checkpoint, validate_distributed_bank_checkpoint
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'Duplicate bank checkpoint field')
        fields = {k:archive[k].copy() for k in archive.files}
    nc, nf, ng, nl, nb = [int(data[k]) for k in ('nc', 'nf', 'nfrac', 'nbed_layers', 'nb')]
    shapes = dict(q=(3, nc), layers=(nl, ng, nc), secondary=(nc,), bed_elevation=(nc,),
        ledger_water=(nb,), ledger_bed=(ng, nb), ledger_secondary=(nb,), ledger_source=())
    shapes.update({'bank_history_'+key:(nc,) for key in HISTORY})
    mode = int(data['bank_toe_erosion_mode'])
    schema_value = np.asarray(fields.get('bank_checkpoint_schema'))
    require(schema_value.shape==() and schema_value.dtype.kind in 'iu',
            'Bank checkpoint schema differs')
    schema = int(schema_value)
    require(schema in ((1,2) if mode==2 else (1,)) and
            (expected_schema is None or schema==expected_schema),
            'Bank checkpoint schema differs')
    map_specs = {}
    if mode == 2:
        shapes.update(bank_toe_face_signed=(ng, nf), bank_toe_face_gross=(ng, nf))
    if schema == 2:
        from validation.gpu_fwave.evolving_toe_map import checkpoint_specs
        map_specs = {'toe_map_'+key:spec for key,spec in checkpoint_specs(nc,nf).items()}
        shapes.update({key:spec[0] for key,spec in map_specs.items()})
    other = {'time', 'dt_hint', 'steps', 'rejected', 'next_boundary_knot',
             'bank_checkpoint_schema', 'bank_payload_sha256'}
    require(set(fields) == set(shapes) | set(identities) | other, 'Bank checkpoint inventory differs')
    payload = fields.pop('bank_payload_sha256')
    require(payload.shape == () and payload.dtype.kind == 'U' and payload.item() == digest(fields),
            'Bank checkpoint payload differs')
    for key, identity in identities.items():
        value = fields[key]
        require(value.shape == () and value.dtype.kind == 'U' and value.item() == identity,
                'Bank checkpoint identity differs: '+key)
    for key, shape in shapes.items():
        value = fields[key]
        dtype = map_specs[key][1] if key in map_specs else (
            np.int32 if key == 'bank_history_collapse_count' else np.float64)
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
                'Bank checkpoint field differs: '+key)
    for key in ('time', 'dt_hint', 'next_boundary_knot'):
        require(fields[key].shape == () and fields[key].dtype == np.float64 and np.isfinite(fields[key]),
                'Invalid bank checkpoint scalar: '+key)
    for key in ('steps', 'rejected', 'bank_checkpoint_schema'):
        require(fields[key].shape == () and fields[key].dtype.kind in 'iu' and int(fields[key]) >= 0,
                'Invalid bank checkpoint integer: '+key)
    current = float(fields['time'])
    require(start <= current <= end and float(fields['dt_hint']) > 0 and
            float(fields['next_boundary_knot']) == events.next_after(current) and
            int(fields['bank_checkpoint_schema']) == schema,
            'Bank checkpoint clock/schema differs')
    require(np.min(fields['q'][0]) >= 0 and np.min(fields['layers']) >= 0 and
            np.min(fields['bank_history_collapse_count']) >= 0, 'Bank checkpoint negative state')
    dry = fields['q'][0] <= float(data['dry_tolerance'])
    require(not np.any(fields['q'][1:, dry]) and not np.any(fields['secondary'][dry]),
            'Bank checkpoint contains dry-cell momentum or secondary mass')
    if int(data['secondary_flow_model']) != 2:
        require(not any(np.any(fields[key]) for key in
                        ('secondary', 'ledger_secondary', 'ledger_source')),
                'OFF/equilibrium bank checkpoint contains transported secondary inventory')
    if mode == 2:
        # Historical ledgers stay on inactive old toes. Only exterior faces
        # can never have carried an internal physical-toe flux.
        allowed = ((np.asarray(data['neighbour'])>=0).astype(np.int32)
                   if schema==2 else data['is_toe_face'])
        validate_bank_toe_checkpoint(fields, allowed)
        if schema==2:
            require(int(fields['toe_map_initialized'])==1,'Uninitialized evolving toe checkpoint')
    else:
        validate_distributed_bank_checkpoint(fields)
    return fields


def candidate_module():
    from . import general_live_bank
    return general_live_bank


def execution_contract(data):
    candidate = candidate_module()
    mode = int(data['bank_toe_erosion_mode'])
    schema = 2 if mode == 2 else 1
    return dict(schema=schema, schema_key='bank_checkpoint_schema',
        payload_key='bank_payload_sha256', diagnostics=ADAPTATION_DIAGNOSTICS,
        method=dict(bank_mode=mode, avalanche=bool(data['general_avalanche']),
            mpm_friction_mode=int(data['mpm_friction_mode']), mpm_cf=float(data['mpm_cf']),
            secondary_flow_model=int(data['secondary_flow_model']),
            hiding_exposure=int(data['hiding_exposure']), bed_slope=int(data['bed_slope']),
            bedload_adaptation=bool(data['bedload_adaptation']),
            bedload_adaptation_length_diameters=float(data['bedload_adaptation_length_diameters']),
            horizontal_viscosity=float(data['horizontal_viscosity']),
            horizontal_viscosity_mode=int(data['horizontal_viscosity_mode']),
            horizontal_viscosity_geometry=int(data['horizontal_viscosity_geometry']),
            diffusion_order='after_bank_and_avalanche',
            adaptation_output_clock='accepted_time',
            adaptation_checkpoint_policy='derived_recompute_not_persistent',
            physical_toe_interface_rule=candidate.TOE_INTERFACE_RULE,
            normal_root_rule=candidate.NORMAL_ROOT_RULE,
            bank_checkpoint_schema=schema,
            physical_toe_mapping=('geometry_connected_wet_v1' if mode == 2 else 'none'),
            toe_face_orientation=('owner_to_neighbour' if mode == 2 else 'none')),
        manifest=dict(bank_mode=mode, bank_checkpoint_schema=schema,
            numerical_CUDA_unchanged=False, CUDA_copied_verbatim=True,
            physical_toe_interface_rule=candidate.TOE_INTERFACE_RULE,
            normal_root_rule=candidate.NORMAL_ROOT_RULE,
            suspended_enabled=False, diffusion_enabled=bool(float(data['horizontal_viscosity']) > 0)))


def restore_extra(solver, restored):
    from .runner import bank_toe_persistent_arrays, distributed_bank_persistent_arrays
    solver._glb_accepted_time = float(restored['time'])
    mode = int(solver.data['bank_toe_erosion_mode'])
    arrays = bank_toe_persistent_arrays(solver) if mode == 2 else distributed_bank_persistent_arrays(solver)
    for key, value in arrays.items():
        value.set(restored[key])
    if solver.evolving_toe is not None:
        solver.evolving_toe.restore(
            {key[len('toe_map_'):]:value for key,value in restored.items() if key.startswith('toe_map_')},
            solver.bed(), solver.state['q'][0], solver.state['layers'])


def run(*, input_path, output_path, until=None, output_every=None, dt_max=None,
        max_steps=None, stop_file=None, resume=None, launch_mode='graph', integrator='ssp43',
        capture_stages=False, capture_step_trace=False, source_archive=None, cancelled=lambda:False):
    from .fixed_grid_lifecycle import run as run_fixed_grid
    import sys
    return run_fixed_grid(sys.modules[__name__], input_path=input_path, output_path=output_path,
        until=until, output_every=output_every, dt_max=dt_max, max_steps=max_steps,
        stop_file=stop_file, resume=resume, launch_mode=launch_mode, integrator=integrator,
        capture_stages=capture_stages, capture_step_trace=capture_step_trace,
        source_archive=source_archive, cancelled=cancelled)
