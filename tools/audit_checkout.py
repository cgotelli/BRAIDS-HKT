"""Verify the source recovery inventory using only the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(path, expected):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Inventory path escapes checkout')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Source checksum differs: ' + str(path.relative_to(ROOT)))


def main():
    current = json.loads((ROOT / 'packaging/source-snapshot.json').read_text(encoding='utf-8'))
    for name, expected in current['files'].items():
        check(ROOT / 'src' / name, expected)
    original = json.loads((ROOT / 'archive/development-sources.json').read_text(encoding='utf-8'))
    for name, row in original['files'].items():
        check(ROOT / row['file'], row['sha256'])
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tools').glob('*.py'))
    for path in files:
        # compile() checks syntax without creating __pycache__ or importing Qt.
        compile(path.read_bytes(), str(path), 'exec')
    spec = ROOT / 'src/tools/build_cutcell.py'
    namespace = {'__file__': str(spec), '__name__': '_recovery_source_check'}
    exec(compile(spec.read_bytes(), str(spec), 'exec'), namespace)
    for name in namespace['SOURCES']:
        if not (ROOT / 'src' / name).is_file():
            raise FileNotFoundError('Missing native build input: ' + name)
    if not (ROOT / 'src/braids_support/sediment/struiksma_t2_2d/wall_reconstruction_probe.f90').is_file():
        raise FileNotFoundError('Missing wall reconstruction build helper')
    exporter = ROOT / 'src/braids_gpu/native_exporter_v1'
    recipe = json.loads((exporter / 'build.json').read_text(encoding='utf-8'))
    for value in recipe['command']:
        if value.lower().endswith(('.f', '.f90')):
            # Historical recipes contain Windows paths. Resolve by basename;
            # no historical drive or user directory is required.
            name = value.replace('\\', '/').rsplit('/', 1)[-1]
            if not (exporter / name).is_file():
                raise FileNotFoundError('Missing frozen geometry source: ' + name)
    print(json.dumps(dict(installed_source_files=len(current['files']),
                          original_development_files=len(original['files']),
                          python_files_syntax_checked=len(files), native_build_inputs='complete'), indent=2))


if __name__ == '__main__':
    main()
