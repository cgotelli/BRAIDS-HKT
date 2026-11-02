"""I/O-only native exporter for bank-enabled selected G4 AMR operators.

The unchanged frozen CPU21b7 numerical/parser source is copied into a new
validation runtime. Only zero-step export calls/records are inserted. This
module never modifies the native source or Delft3D reference trees.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import coupled_reference as core
from . import cpu_filter_cold20h as parent
from . import i4_avalanche_export as avalanche
from . import i4_bank_avalanche_export as dual
from . import i4_physical_toe_mode2_export as toe


HERE = Path(__file__).resolve().parent
ROOT = HERE/'runs/g4_bank_active_front_native_20261004_v1'
RUNTIME = ROOT/'runtime'
PARENT_EXE_SHA = '21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5'
SIDECARS = ('bank_toe_metadata.bin', 'avalanche_metadata.bin',
            'bank_history_metadata.bin')
HISTORY_FLOATS = ('bank_undercut', 'bank_reference_height', 'bank_toe_height',
                  'bank_reference_length', 'bank_toe_cumulative',
                  'bank_collapse_cumulative')
HISTORY_INTS = ('bank_collapse_count',)
HISTORY_KEYS = set(toe.IDENTITY_KEYS) | {'bank_failure'} | \
    set(HISTORY_FLOATS) | set(HISTORY_INTS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bank_history_export():
    text = "\n subroutine g4_export_bank_history_metadata()\n  integer :: gpu_unit\n"
    text += "  if(.not.bank_failure.or..not.allocated(bank_undercut)) &\n"
    text += "   error stop 'G4 bank-history export requires enabled allocated bank physics'\n"
    text += "  call gpu_open(gpu_unit,'bank_history_metadata.bin')\n"
    for name in toe.IDENTITY_KEYS:
        text += core.rec(name, kind='i')
    text += core.rec('bank_failure', 'merge(1,0,bank_failure)', kind='i')
    for name in HISTORY_FLOATS:
        text += core.rec(name, dims=('nc',))
    text += core.rec('bank_collapse_count', dims=('nc',), kind='i')
    text += '  call gpu_finish(gpu_unit)\n end subroutine\n'
    return text


def transform(source):
    changed = dual.transform(source)
    anchor = '  call i4_export_avalanche_metadata()\n  stop '
    marker = 'end module\n\nprogram run_cutcell'
    require(changed.count(anchor) == 1 and changed.count(marker) == 1,
            'G4 zero-step sidecar export anchors differ')
    changed = changed.replace(anchor,
        '  call i4_export_avalanche_metadata()\n'
        '  call g4_export_bank_history_metadata()\n  stop ', 1)
    return changed.replace(marker, bank_history_export()+marker, 1)


def verify_transform(source, changed):
    restored = changed.replace(bank_history_export(), '', 1).replace(
        '  call g4_export_bank_history_metadata()\n', '', 1)
    require(changed == transform(source) and restored == dual.transform(source),
            'G4 exporter changed more than the third metadata I/O call')
    dual.verify_transform(source, restored)
    for name in ('load_case', 'build_physical_toe_map',
                 'process_noncohesive_bank_failure',
                 'avalanche_erodible_banks', 'advance_ssp33'):
        start = source.index('subroutine '+name+'(')
        end = source.index('end subroutine', start)
        require(changed.count(source[start:end]) == 1,
                'Native parser/numerical body changed: '+name)
    return True


def build(runtime=RUNTIME):
    runtime = Path(runtime).resolve()
    if runtime.exists():
        raise FileExistsError('G4 I/O-only runtime is write-once')
    identity, protected = parent.runtime_identity()
    require(parent.TIGHT_SHA == PARENT_EXE_SHA and
            sha(parent.TIGHT_EXE) == PARENT_EXE_SHA,
            'G4 exporter parent must be frozen unchanged CPU21b7')
    runtime.mkdir(parents=True)
    sources = []
    for name in identity['source_sha256']:
        original = parent.TIGHT_RUNTIME/name
        target = runtime/name
        if name == 'cutcell_solver.f90':
            source = original.read_text(encoding='utf-8')
            changed = transform(source)
            verify_transform(source, changed)
            target.write_text(changed, encoding='utf-8')
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for source in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(source, runtime/source.name)
    flags = identity['command'][1:6]
    require(flags == ['-O3', '-flto', '-fopenmp', '-ffree-line-length-none',
                      '-ffixed-line-length-none'],
            'Frozen CPU21b7 compiler flags differ')
    compiler = Path(identity['command'][0])
    executable = runtime/'xbraids_g4_bank_export.exe'
    command = [str(compiler), *flags, *sources, '-o', str(executable)]
    process = subprocess.run(command, cwd=runtime, capture_output=True,
                             text=True, timeout=300)
    (runtime/'build.log').write_text(process.stdout+process.stderr,
                                      encoding='utf-8')
    require(process.returncode == 0,
            'G4 I/O-only export build failed: '+process.stderr[-3000:])
    record = dict(schema=1, command=command, executable=str(executable),
        executable_sha256=sha(executable),
        parent_executable_sha256=PARENT_EXE_SHA,
        parent_protected_files=protected,
        source_sha256={Path(path).name: sha(path) for path in sources},
        compiler_sha256=sha(compiler), generator_sha256=sha(__file__),
        core_generator_sha256=sha(core.__file__),
        dual_generator_sha256=sha(dual.__file__),
        toe_generator_sha256=sha(toe.__file__),
        avalanche_generator_sha256=sha(avalanche.__file__),
        export_only=True, numerical_routines_unchanged=True,
        native_original_modified=False, Delft3D_source_modified=False)
    (runtime/'build.json').write_text(json.dumps(record, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')
    return verify_build(runtime)


def verify_build(runtime=RUNTIME):
    runtime = Path(runtime).resolve()
    record = json.loads((runtime/'build.json').read_text(encoding='utf-8'))
    identity, protected = parent.runtime_identity()
    require(record['schema'] == 1 and record['export_only'] is True and
            record['numerical_routines_unchanged'] is True and
            record['parent_executable_sha256'] == PARENT_EXE_SHA ==
                sha(parent.TIGHT_EXE) and
            record['parent_protected_files'] == protected and
            record['generator_sha256'] == sha(__file__) and
            record['core_generator_sha256'] == sha(core.__file__) and
            record['dual_generator_sha256'] == sha(dual.__file__) and
            record['toe_generator_sha256'] == sha(toe.__file__) and
            record['avalanche_generator_sha256'] == sha(avalanche.__file__) and
            record['compiler_sha256'] == sha(record['command'][0]) and
            record['executable_sha256'] == sha(record['executable']),
            'G4 export-only runtime identity differs')
    for name, digest in record['source_sha256'].items():
        require(sha(runtime/name) == digest,
                'G4 copied I/O-only source changed: '+name)
    for filename, digest in identity['source_sha256'].items():
        require(sha(parent.TIGHT_RUNTIME/filename) == digest,
                'Frozen CPU21b7 native source changed: '+filename)
    source = (parent.TIGHT_RUNTIME/'cutcell_solver.f90').read_text(encoding='utf-8')
    changed = (runtime/'cutcell_solver.f90').read_text(encoding='utf-8')
    verify_transform(source, changed)
    return record


def read_prepared(folder):
    folder = Path(folder).resolve()
    require((folder/'coupled_input.bin').is_file() and
            all((folder/name).is_file() for name in SIDECARS) and
            not (folder/'frames.txt').exists() and
            not (folder/'checkpoint.bin').exists(),
            'G4 zero-step core/sidecars absent or solver advanced')
    data = core.read_snapshot(folder)
    expected = (toe.EXPECTED_KEYS,
        set(avalanche.CONTROLS)|set(toe.IDENTITY_KEYS)|{'general_avalanche'},
        HISTORY_KEYS)
    for name, keys in zip(SIDECARS, expected):
        extra = core.read_records(folder/name)
        require(set(extra) == set(keys), 'G4 sidecar schema differs: '+name)
        for key, value in extra.items():
            require(key not in data or np.array_equal(data[key], value),
                    'G4 duplicate core/sidecar value differs: '+key)
            data[key] = value
    nc = int(data['nc'])
    require(int(data['scheme']) == 6 and int(data['bedlaw']) == 2 and
            int(data['exchange_mode']) == 0 and int(data['mpm_friction_mode']) == 1 and
            float(data['mpm_critical']) == .047 and
            int(data['cpu_time_integrator']) == 33 and
            int(data['nfrac']) == 2 and int(data['nbed_layers']) == 2 and
            int(data['nstate']) == 10 and int(data['bank_failure']) == 1 and
            int(data['bank_toe_erosion_mode']) == 2 and
            int(data['general_avalanche']) == 1 and
            int(data['secondary_flow']) == 1 and
            int(data['secondary_flow_model']) == 2 and
            int(data['secondary_index']) == 9 and
            int(data['amr_enabled']) == 0 and int(data['nb']) == 1 and
            int(data['mode'][0]) == 2 and
            float(data['morphological_factor']) == 1. and
            float(data['bank_toe_capacity_fraction']) == 1. and
            int(data['toe_mapping_status']) == 0 and
            np.all(data['state'][3:5] == 0.) and
            all(data[key].shape == (nc,) for key in
                (*HISTORY_FLOATS, *HISTORY_INTS)) and
            all(np.array_equal(data[key], np.zeros(nc)) for key in
                (*HISTORY_FLOATS, *HISTORY_INTS)),
            'G4 native parser controls or seven initialized bank histories differ')
    return data


def export(deck, folder, *, runtime=RUNTIME):
    deck, folder, runtime = Path(deck).resolve(), Path(folder).resolve(), \
        Path(runtime).resolve()
    built = verify_build(runtime)
    if folder.exists():
        raise FileExistsError('G4 selected native export is write-once')
    origin = core.copy_inputs(deck, folder)
    environment = {key:value for key,value in os.environ.items()
                   if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    environment.update(BRAIDS_GPU_INITIAL_EXPORT='1',
                       BRAIDS_TIME_INTEGRATOR='legacy', OMP_NUM_THREADS='1',
                       OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
    environment['PATH'] = str(runtime)+os.pathsep+environment.get('PATH', '')
    result = subprocess.run([built['executable'], deck.name], cwd=folder,
        env=environment, capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (folder/'initial.log').write_text(result.stdout+result.stderr,
                                       encoding='utf-8')
    require(result.returncode == 0,
            'G4 zero-step native parser/export failed: '+
            (result.stdout+result.stderr)[-2000:])
    data = read_prepared(folder)
    artifacts = {name:sha(folder/name) for name in
                 ('coupled_input.bin', *SIDECARS)}
    receipt = dict(schema=1, origin=origin, status='completed',
        runtime_build_sha256=sha(runtime/'build.json'),
        runtime_executable_sha256=built['executable_sha256'],
        exporter_source_sha256=sha(__file__),
        native_cpu_parent_sha256=PARENT_EXE_SHA,
        controls=dict(scheme=int(data['scheme']), bedlaw=int(data['bedlaw']),
            exchange_mode=int(data['exchange_mode']),
            bank_toe_erosion_mode=int(data['bank_toe_erosion_mode']),
            general_avalanche=int(data['general_avalanche']),
            secondary_flow_model=int(data['secondary_flow_model']),
            cpu_time_integrator=int(data['cpu_time_integrator'])),
        cells=int(data['nc']), faces=int(data['nf']),
        bank_cells=int(np.count_nonzero(data['bank_material_mask'])),
        toe_faces=int(np.count_nonzero(data['is_toe_face'])),
        artifacts=artifacts, initialization_CPU_only=True,
        solver_steps_performed=0, numerical_routines_unchanged=True,
        Delft3D_source_modified=False, GPU_solver_executed=False)
    (folder/'preparation_receipt.json').write_text(json.dumps(receipt,
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return data, receipt
