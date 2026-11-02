"""Zero-step, I/O-only export of native I4 physical bank-toe mode-2 inputs.

The exporter is built from a frozen COPY of the native runtime.  Its only new
work is the established initial-state export plus a call to the unchanged
native physical-toe mapper, followed by a self-describing metadata write.
Neither the native numerical/parser body nor Delft3D is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import coupled_reference as cr
from . import cpu_filter_cold20h as parent
from . import i4_physical_toe_mode2_native_case as native


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/i4_physical_toe_mode2_prepared_20261004_v2'
RUNTIME = ROOT / 'runtime'
PARENT_EXE_SHA = '21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5'
# Freeze these after the unchanged-native ON/OFF gate is final.  Preparation
# refuses to run while they are unset; the gate must precede any GPU trajectory.
PARENT_PLAN_SHA = 'a3cd05aecb4837ded7e4402c922c02d973f5de5325c07ad9cab6bb99d61f2370'
PARENT_GATE_SHA = 'bcecacd10113076b7e5347858f3328480361b0cd8488b100d0f5fe56728abc8f'
SIDECAR = 'bank_toe_metadata.bin'
IDENTITY_KEYS = ('nc', 'nf', 'nfrac', 'nbed_layers', 'deck_version')
INTEGER_SCALARS = ('bank_toe_erosion_mode', 'bank_failure', 'toe_mapping_status')
REAL_SCALARS = ('bank_toe_capacity_fraction', 'bank_critical_undercut_m')
CELL_FLAGS = ('bank_material_mask', 'is_toe_cell', 'is_toe_donor')
FACE_FLAGS = ('is_toe_face',)
CELL_IDS = ('physical_toe_cell', 'physical_toe_donor')
FACE_IDS = ('physical_toe_face',)
FACE_CELL_IDS = ('candidate_bank', 'candidate_toe')
OTHER_ARRAYS = ('physical_toe_distance', 'physical_toe_status')
EXPECTED_KEYS = (set(IDENTITY_KEYS) | set(INTEGER_SCALARS) |
                 set(REAL_SCALARS) | set(CELL_FLAGS) | set(FACE_FLAGS) |
                 set(CELL_IDS) | set(FACE_IDS) | set(FACE_CELL_IDS) |
                 set(OTHER_ARRAYS))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def extra_export():
    text = "\n subroutine i4_export_physical_toe_metadata()\n  integer :: gpu_unit\n"
    text += "  if(bank_toe_erosion_mode/=2.or..not.bank_failure) &\n"
    text += "   error stop 'GPU physical-toe export requires native mode 2'\n"
    text += "  if(.not.bank_toe_mapping_ready) call build_physical_toe_map(q)\n"
    text += "  call gpu_open(gpu_unit,'bank_toe_metadata.bin')\n"
    for key in IDENTITY_KEYS:
        text += cr.rec(key, kind='i')
    text += cr.rec('bank_toe_erosion_mode', kind='i')
    text += cr.rec('bank_failure', 'merge(1,0,bank_failure)', kind='i')
    text += cr.rec('toe_mapping_status', '0', kind='i')
    text += cr.rec('bank_toe_capacity_fraction')
    text += cr.rec('bank_critical_undercut_m', 'bank_critical_undercut')
    for key, expression, size in (
        ('bank_material_mask', 'merge(1,0,bank_erodible)', 'nc'),
        ('physical_toe_cell', 'physical_toe_cell-1', 'nc'),
        ('physical_toe_donor', 'physical_toe_donor-1', 'nc'),
        ('physical_toe_face', 'physical_toe_face-1', 'nc'),
        ('physical_toe_distance', 'physical_toe_distance', 'nc'),
        ('is_toe_cell', 'merge(1,0,physical_is_toe_cell)', 'nc'),
        ('is_toe_donor', 'merge(1,0,physical_is_toe_donor)', 'nc'),
        ('is_toe_face', 'merge(1,0,physical_is_toe_face)', 'nf'),
        ('candidate_bank', 'physical_candidate_bank-1', 'nf'),
        ('candidate_toe', 'physical_candidate_toe-1', 'nf'),
        ('physical_toe_status', 'physical_toe_status', 'nc'),
    ):
        text += cr.rec(key, expression, (size,), 'r' if key == 'physical_toe_distance' else 'i')
    text += '  call gpu_finish(gpu_unit)\n end subroutine\n'
    return text


def transform(source):
    changed = cr.instrument(source)
    anchor = '  call gpu_export_input()\n  stop '
    marker = 'end module\n\nprogram run_cutcell'
    require(changed.count(anchor) == 1 and changed.count(marker) == 1,
            'Unexpected native zero-step exporter anchors')
    changed = changed.replace(anchor,
        '  call gpu_export_input()\n  call i4_export_physical_toe_metadata()\n  stop ', 1)
    return changed.replace(marker, extra_export() + marker, 1)


def verify_transform(source, changed):
    restored = changed.replace(extra_export(), '', 1).replace(
        '  call i4_export_physical_toe_metadata()\n', '', 1)
    require(changed == transform(source) and restored == cr.instrument(source),
            'I4 mode-2 exporter changed more than declared initial I/O')
    for name in ('load_case', 'build_physical_toe_map',
                 'process_noncohesive_bank_failure'):
        start = source.index(' subroutine ' + name + '(')
        end = source.index(' end subroutine', start)
        require(changed.count(source[start:end]) == 1,
                'Native numerical/parser body changed: ' + name)
    return True


def _frozen_native():
    require(len(PARENT_PLAN_SHA) == 64 and len(PARENT_GATE_SHA) == 64,
            'Native mode-2 plan and pre-GPU gate must be frozen first')
    native.verify_plan(native.DEFAULT_ROOT)
    plan = native.DEFAULT_ROOT / 'plan.json'
    gate = HERE / 'reports/i4_physical_toe_mode2_native_20261004_v2/GPU_ACCEPTANCE_GATE.txt'
    require(sha(plan) == PARENT_PLAN_SHA and sha(gate) == PARENT_GATE_SHA,
            'Frozen native mode-2 plan or pre-GPU gate differs')
    require(sha(native.SOURCE_EXECUTABLE) == PARENT_EXE_SHA,
            'Frozen native mode-2 executable differs')
    return {'plan_sha256': PARENT_PLAN_SHA, 'gate_sha256': PARENT_GATE_SHA}


def build(runtime=RUNTIME):
    runtime = Path(runtime).resolve()
    require(not runtime.exists(), 'I4 mode-2 export runtime already exists')
    _frozen_native()
    identity, protected = parent.runtime_identity()
    require(parent.TIGHT_SHA == PARENT_EXE_SHA,
            'I4 mode-2 exporter parent must be the frozen native executable')
    runtime.mkdir(parents=True)
    sources = []
    for name in identity['source_sha256']:
        original = parent.TIGHT_RUNTIME / name
        target = runtime / name
        if name == 'cutcell_solver.f90':
            source = original.read_text(encoding='utf-8')
            changed = transform(source)
            verify_transform(source, changed)
            target.write_text(changed, encoding='utf-8')
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for path in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(path, runtime / path.name)
    flags = identity['command'][1:6]
    require(flags == ['-O3', '-flto', '-fopenmp', '-ffree-line-length-none',
                      '-ffixed-line-length-none'],
            'Frozen native compiler flags differ')
    compiler = Path(identity['command'][0])
    executable = runtime / 'xbraids_mode2_export.exe'
    command = [str(compiler), *flags, *sources, '-o', str(executable)]
    process = subprocess.run(command, cwd=runtime, capture_output=True,
                             text=True, timeout=300)
    (runtime / 'build.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    require(process.returncode == 0,
            'I4 mode-2 I/O-only export compilation failed: ' + process.stderr[-2000:])
    record = dict(schema=1, command=command, executable=str(executable),
        executable_sha256=sha(executable), parent_executable_sha256=PARENT_EXE_SHA,
        parent_protected_files=protected,
        source_sha256={Path(path).name: sha(path) for path in sources},
        compiler_sha256=sha(compiler), driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        transformation='cr.instrument initialization I/O plus unchanged native map query and sidecar write',
        solver_steps_performed=0, numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    save(runtime / 'build.json', record)
    return verify_build(runtime)


def verify_build(runtime=RUNTIME):
    runtime = Path(runtime).resolve()
    record = json.loads((runtime / 'build.json').read_text(encoding='utf-8'))
    parent.runtime_identity()
    require(record['schema'] == 1 and
            record['parent_executable_sha256'] == PARENT_EXE_SHA == parent.TIGHT_SHA and
            record['driver_sha256'] == sha(__file__) and
            record['coupled_reference_sha256'] == sha(cr.__file__) and
            record['compiler_sha256'] == sha(record['command'][0]) and
            record['executable_sha256'] == sha(record['executable']),
            'I4 mode-2 export runtime or dependencies changed')
    for name, digest in record['source_sha256'].items():
        require(sha(runtime / name) == digest, 'I4 mode-2 export source changed: ' + name)
    for path, digest in record['parent_protected_files'].items():
        require(sha(path) == digest, 'I4 original native source/runtime changed: ' + path)
    verify_transform((parent.TIGHT_RUNTIME / 'cutcell_solver.f90').read_text(encoding='utf-8'),
                     (runtime / 'cutcell_solver.f90').read_text(encoding='utf-8'))
    return record


def read_prepared(folder, case):
    require(case in native.CASES, 'Unknown frozen I4 mode-2 case')
    folder = Path(folder).resolve()
    require((folder / 'coupled_input.bin').is_file() and
            (folder / SIDECAR).is_file() and
            not any((folder / name).exists() for name in
                    ('normal_metadata.bin', 'avalanche_metadata.bin',
                     'coupled_diffusion.bin')),
            'I4 mode-2 prepared sidecars absent or incompatible sidecar present')
    data = cr.read_snapshot(folder)
    extra = cr.read_records(folder / SIDECAR)
    require(set(extra) == EXPECTED_KEYS, 'I4 mode-2 sidecar schema differs')
    for key in IDENTITY_KEYS + ('bank_toe_erosion_mode', 'bank_failure'):
        require(key in data and int(data[key]) == int(extra[key]),
                'I4 mode-2 duplicate native identity differs: ' + key)
    data.update(extra)
    nc, nf = int(data['nc']), int(data['nf'])
    require(nc == 18 and nf > 0 and int(data['nfrac']) == 2 and
            int(data['nbed_layers']) == 2 and
            int(data['bank_toe_erosion_mode']) == 2 and
            int(data['bank_failure']) == 1 and int(data['toe_mapping_status']) == 0 and
            int(data['scheme']) == 6 and int(data['bedlaw']) == 2 and
            int(data['exchange_mode']) == 0 and int(data['amr_enabled']) == 0 and
            int(data['general_avalanche']) == 0 and int(data['bank_avalanche']) == 0 and
            float(data['morphological_factor']) == 1 and
            float(data['bank_critical_undercut_m']) > 0 and
            float(data['bank_toe_capacity_fraction']) == (1. if case == 'erosion' else 0.),
            'I4 mode-2 prepared numerical scope differs')
    for key in CELL_FLAGS + CELL_IDS + FACE_IDS + ('physical_toe_distance', 'physical_toe_status'):
        require(np.shape(data[key]) == (nc,), 'I4 mode-2 cell map shape differs: ' + key)
    for key in FACE_FLAGS + FACE_CELL_IDS:
        require(np.shape(data[key]) == (nf,), 'I4 mode-2 face map shape differs: ' + key)
    for key in CELL_FLAGS + FACE_FLAGS + CELL_IDS + FACE_IDS + FACE_CELL_IDS + ('physical_toe_status',):
        require(data[key].dtype == np.dtype('<i4'), 'I4 mode-2 map integer dtype differs: ' + key)
    require(data['physical_toe_distance'].dtype == np.dtype('<f8'),
            'I4 mode-2 map distance dtype differs')
    for key in CELL_FLAGS + FACE_FLAGS:
        require(np.all(np.isin(data[key], (0, 1))), 'I4 mode-2 mask invalid: ' + key)
    for key in CELL_IDS + FACE_CELL_IDS:
        require(np.all((data[key] >= -1) & (data[key] < nc)),
                'I4 mode-2 mapped cell ID invalid: ' + key)
    for key in FACE_IDS:
        require(np.all((data[key] >= -1) & (data[key] < nf)),
                'I4 mode-2 mapped face ID invalid: ' + key)
    mask = data['bank_material_mask'].astype(bool)
    require(mask.any() and np.all(data['physical_toe_status'][mask] == 1) and
            np.all(data['physical_toe_status'][~mask] == 0) and
            np.all(data['physical_toe_cell'][mask] >= 0) and
            np.all(data['physical_toe_donor'][mask] >= 0) and
            np.all(data['physical_toe_face'][mask] >= 0) and
            np.all(np.isfinite(data['physical_toe_distance'])) and
            np.all(data['physical_toe_distance'][mask] >= 0) and
            np.array_equal(data['is_toe_face'].astype(bool),
                (data['candidate_bank'] >= 0) & (data['candidate_toe'] >= 0)),
            'I4 mode-2 native material-to-toe map invalid')
    candidate = data['is_toe_face'].astype(bool)
    bank_ids = data['candidate_bank'][candidate]
    toe_ids = data['candidate_toe'][candidate]
    owners = data['owner'][candidate]
    neighbours = data['neighbour'][candidate]
    require(np.all(mask[bank_ids]) and np.all(~mask[toe_ids]) and
            np.all(((owners == bank_ids) & (neighbours == toe_ids)) |
                   ((owners == toe_ids) & (neighbours == bank_ids))) and
            np.array_equal(data['is_toe_donor'].astype(bool),
                           np.isin(np.arange(nc), bank_ids)) and
            np.array_equal(data['is_toe_cell'].astype(bool),
                           np.isin(np.arange(nc), toe_ids)),
            'I4 mode-2 native candidate face topology invalid')
    mapped_donors = data['physical_toe_donor'][mask]
    mapped_faces = data['physical_toe_face'][mask]
    mapped_toes = data['physical_toe_cell'][mask]
    require(np.all(mask[mapped_donors]) and
            np.all(data['is_toe_face'][mapped_faces] == 1) and
            np.array_equal(data['candidate_bank'][mapped_faces], mapped_donors) and
            np.array_equal(data['candidate_toe'][mapped_faces], mapped_toes),
            'I4 mode-2 native mapped bank-toe topology invalid')
    original = np.load(native.DEFAULT_ROOT / 'inputs' / case / 'initial_analysis.npz',
                       allow_pickle=False)
    require(np.array_equal(data['state'].T, original['state']) and
            np.array_equal(data['xy'].T, original['xy']) and
            np.array_equal(data['area'], original['area']) and
            np.array_equal(data['rock'], original['rock']) and
            np.array_equal(data['bulk_fraction'], original['bulk']) and
            np.array_equal(data['bank_material_mask'], original['bank_mask']) and
            np.array_equal(data['layer_mass'],
                original['state'][:, 5:5+int(data['nfrac'])*int(data['nbed_layers'])].T.reshape(
                    int(data['nbed_layers']), int(data['nfrac']), nc)),
            'I4 mode-2 prepared geometry/state/mask differs from frozen native initial state')
    return data


def pair_map_identity(on, off):
    for key in ('state', 'layer_mass', 'q', 'rock', 'area', 'xy', 'owner', 'neighbour',
                'mid', 'normal', 'length', 'bank_material_mask',
                'physical_toe_cell', 'physical_toe_donor', 'physical_toe_face',
                'physical_toe_distance', 'is_toe_cell', 'is_toe_donor', 'is_toe_face',
                'candidate_bank', 'candidate_toe', 'physical_toe_status'):
        require(np.array_equal(on[key], off[key]), 'I4 mode-2 ON/OFF initial map differs: ' + key)
    require(float(on['bank_critical_undercut_m']) ==
            float(off['bank_critical_undercut_m']),
            'I4 mode-2 ON/OFF critical undercut differs')
    return True


def prepare(case, root=ROOT, runtime=RUNTIME):
    require(case in native.CASES, 'Unknown frozen I4 mode-2 case')
    root = Path(root).resolve()
    frozen = _frozen_native()
    built = verify_build(runtime)
    source = native.DEFAULT_ROOT / 'inputs' / case / 'case.data'
    destination = root / case
    require(not destination.exists(), 'I4 mode-2 prepared case already exists')
    origin = cr.copy_inputs(source, destination)
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    environment.update(BRAIDS_GPU_INITIAL_EXPORT='1', BRAIDS_TIME_INTEGRATOR='legacy',
        OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
    environment['PATH'] = str(Path(runtime).resolve()) + os.pathsep + environment.get('PATH', '')
    process = subprocess.run([built['executable'], source.name], cwd=destination,
        env=environment, capture_output=True, text=True, timeout=60)
    (destination / 'initial.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    require(process.returncode == 0,
            'I4 mode-2 zero-step export failed: ' + process.stderr[-2000:])
    require(not (destination / 'frames.txt').exists() and
            not (destination / 'checkpoint.bin').exists(),
            'I4 mode-2 zero-step export unexpectedly evolved native state')
    data = read_prepared(destination, case)
    artifacts = {name: sha(destination / name) for name in ('coupled_input.bin', SIDECAR)}
    receipt = dict(schema=1, status='completed', case=case, origin=origin,
        native_plan_sha256=frozen['plan_sha256'],
        native_pre_gpu_gate_sha256=frozen['gate_sha256'],
        parent_native_executable_sha256=PARENT_EXE_SHA,
        exporter_build_sha256=sha(Path(runtime) / 'build.json'),
        exporter_executable_sha256=built['executable_sha256'],
        exporter_driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        solver_steps_performed=0, initialization_CPU_only=True,
        native_map_built_from_initial_state=True,
        native_ids_converted_to_zero_based_minus_one_unmapped=True,
        controls={key: float(data[key]) for key in REAL_SCALARS},
        geometry=dict(cells=int(data['nc']), faces=int(data['nf']),
            fractions=int(data['nfrac']), layers=int(data['nbed_layers']),
            bank_material_cells=int(np.count_nonzero(data['bank_material_mask'])),
            toe_faces=int(np.count_nonzero(data['is_toe_face']))),
        artifacts=artifacts, numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    save(destination / 'preparation_receipt.json', receipt)
    return dict(folder=str(destination), receipt_sha256=sha(destination / 'preparation_receipt.json'),
                artifacts=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'prepare', 'verify'))
    parser.add_argument('--case', choices=native.CASES)
    args = parser.parse_args()
    if args.action == 'build':
        result = build()
    else:
        require(args.case is not None, '--case is required')
        if args.action == 'prepare':
            result = prepare(args.case)
        else:
            folder = ROOT / args.case
            result = dict(case=args.case,
                receipt_sha256=sha(folder / 'preparation_receipt.json'),
                cells=read_prepared(folder, args.case)['nc'])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
