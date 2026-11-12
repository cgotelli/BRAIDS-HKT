"""GDAL-free constructor for an authenticated graded-bedload source archive.

The seven generated Python modules and CUDA files are the same private source
snapshot used by the historical target-B composition.  This loader only binds
those frozen classes; it does not transform or rewrite numerical source at run
time, nor import the native validation fixture used to generate the snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from . import live_bedload_adaptation as live


PRIVATE = (
    '_a1_boundary_schedule', '_a1_boundaries', '_a1_sediment_transport',
    '_a1_coupled_solver', '_a1_scheduled_coupled', '_a1_normal_bridge',
    '_a1_outlet_bridge',
)
PREFIX = 'validation.gpu_fwave.'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_archive(folder):
    """Hash the complete bundle and generated modules without fixture imports."""
    from .runner import sha
    folder = Path(folder).resolve()
    manifest_path = folder / 'runtime_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('schema') == 1 and
            manifest.get('variant') == live.GENERAL_VARIANT and
            manifest.get('runner_sha256') == sha(Path(__file__).parent / 'runner.py') and
            manifest.get('adapter_sha256') == sha(live.__file__) and
            manifest.get('prebuilt_adapter_sha256') == sha(__file__) and
            manifest.get('preparation_contract_schema') == 1,
            'Graded-bedload prebuilt source identity differs')
    files = manifest.get('source_files', {})
    require(isinstance(files, dict) and
            {name + '.py' for name in PRIVATE} <= set(files),
            'Graded-bedload generated Python module inventory differs')
    actual = {str(path.relative_to(folder)).replace('\\', '/')
              for path in folder.rglob('*') if path.is_file()}
    require(actual == set(files) | {'runtime_manifest.json'},
            'Graded-bedload prebuilt source inventory differs')
    for name, digest in files.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder) and path.is_file() and
                sha(path) == digest,
                'Graded-bedload prebuilt source file differs: ' + name)
    root = Path(__file__).resolve().parents[1]
    transaction_files = {
        'a1_full_array_transaction.py':
            root / 'braids_gpu/a1_full_array_transaction.py',
        'a1_source_lock_v2.py': root / 'braids_gpu/a1_source_lock_v2.py',
        'a2_segmented_array_copy.py':
            root / 'validation/gpu_fwave/a2_segmented_array_copy.py',
        'a2_solver_array_inventory.py':
            root / 'validation/gpu_fwave/a2_solver_array_inventory.py',
    }
    pinned = manifest.get('transaction_sources', {})
    require(set(pinned) == set(transaction_files),
            'Graded-bedload transaction source inventory differs')
    for name, path in transaction_files.items():
        require(path.is_file() and sha(path) == pinned[name] and
                files.get('transaction_python/' + name) == pinned[name],
                'Graded-bedload live/archived transaction source differs: ' + name)
    require(sha(folder / 'candidate_sources.json') ==
            manifest.get('candidate_receipt_sha256'),
            'Graded-bedload generated-source receipt differs')
    candidate = json.loads((folder / 'candidate_sources.json').read_text(
        encoding='utf-8'))
    require(candidate.get('schema') == 1 and
            candidate.get('kind') == 'isolated_current_source_combined_boundary' and
            candidate.get('numerical_cuda_unchanged') is True and
            candidate.get('native_controls_retained') is True,
            'Graded-bedload generated-source provenance differs')
    for name, digest in candidate['original_cuda_sha256'].items():
        require(files.get(name) == digest,
                'Graded-bedload archived CUDA differs from original: ' + name)
    for name, row in candidate['generated_python'].items():
        require(files.get(row['generated_name']) == row['generated_sha256'],
                'Graded-bedload generated Python differs: ' + name)
    return manifest


@contextmanager
def constructors(folder, data):
    """Load the prebuilt private class bindings, leaving legacy paths alone."""
    from validation.gpu_fwave import coupled_solver as stock_solver
    from validation.gpu_fwave import normal_outlet_boundary as boundary_bridge

    folder = Path(folder).resolve()
    live.validate_general_scope(data, start=float(data.get('time', 0.)),
        end=float(data['endtime']), interval=float(data['output_dt']),
        dtmax=float(data['dtmax']), launch_mode='plain', integrator='ssp33')
    verify_archive(folder)
    require(not any(PREFIX + name in sys.modules for name in PRIVATE),
            'Graded-bedload private modules already loaded')
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    loaded = []
    modules = {}
    try:
        for name in PRIVATE:
            key = PREFIX + name
            spec = importlib.util.spec_from_file_location(key, folder / (name + '.py'))
            require(spec is not None and spec.loader is not None,
                    'Graded-bedload generated module is absent: ' + name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            loaded.append(key)
            spec.loader.exec_module(module)
            modules[name] = module
        core = modules['_a1_coupled_solver']
        core.SecondaryTransport = stock_solver.SecondaryTransport
        core.BoundaryControls = boundary_bridge.make_boundary_controls(
            modules['_a1_boundaries'].BoundaryControls,
            modules['_a1_normal_bridge'], modules['_a1_outlet_bridge'])

        class _Intermediate(core.CoupledSolver):
            pass

        class _PrivateShell(_Intermediate):
            pass

        yield _PrivateShell
    finally:
        for key in reversed(loaded):
            sys.modules.pop(key, None)
        sys.dont_write_bytecode = previous_bytecode
        verify_archive(folder)


@contextmanager
def adapter():
    """Match the historical candidate interface without importing its fixture."""
    yield SimpleNamespace(constructors=constructors)
