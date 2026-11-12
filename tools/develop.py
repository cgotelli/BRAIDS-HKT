"""Stage an authenticated, source-built development runtime and open BRAIDS.

The installed launcher deliberately disables compilation. A development stage
uses the same numerical constructors with compilation enabled instead. Only the
two bootstraps and the child-process compiler PATH are adapted; solver code is
copied byte for byte. An immutable, content-addressed stage keeps a running
simulation isolated from edits to the checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path, before, after):
    text = path.read_text(encoding='utf-8')
    if text.count(before) != 1:
        raise RuntimeError('Development bootstrap marker changed: ' + str(path))
    path.write_text(text.replace(before, after), encoding='utf-8')


def stage():
    source = ROOT / 'src'
    files = sorted(p for p in source.rglob('*') if p.is_file()
                   and not set(p.relative_to(source).parts) & {'build', '__pycache__'}
                   and p.suffix not in {'.pyc', '.pyo', '.dll', '.exe', '.so', '.mod', '.o'})
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in files}
    key = hashlib.sha256((json.dumps(hashes, sort_keys=True) + digest(Path(__file__))).encode()).hexdigest()[:20]
    runtime = ROOT / '.development' / key / 'runtime'
    receipt = runtime / 'RUNTIME.json'
    if receipt.exists():
        manifest = json.loads(receipt.read_text(encoding='utf-8'))
        if manifest.get('development_source_sha256') != hashes:
            raise ValueError('Existing development source identity differs')
        for name, expected in manifest['files'].items():
            if digest(runtime / name) != expected:
                raise ValueError('Staged source was modified; remove this stage and restage: ' + name)
        return runtime
    for path in files:
        target = runtime / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # Keep the production authentication checks. Only packaged-DLL restoration
    # is omitted: original constructors build their own source-keyed caches.
    replace_once(runtime / 'braids_core/river_launcher.py',
                 '    install(json.loads((ROOT/"PRECOMPILED.json").read_text(encoding="utf-8")), ROOT)',
                 '    # Development: original constructors compile authenticated source.')
    replace_once(runtime / 'desktop/worker.py',
                 "    install(json.loads((runtime/'PRECOMPILED.json').read_text(encoding='utf-8')),runtime)",
                 '    # Development: original constructors compile authenticated source.')
    compiler_setup = ('import os\n'
                      'if os.environ.get("BRAIDS_DEV_COMPILER_BIN"):\n'
                      '    os.environ["PATH"] = os.environ["BRAIDS_DEV_COMPILER_BIN"] + os.pathsep + os.environ.get("PATH", "")\n')
    replace_once(runtime / 'desktop/worker.py', 'from osgeo import osr',
                 compiler_setup + 'from osgeo import osr')
    replace_once(runtime / 'braids_core/river_launcher.py', 'import warnings',
                 'import warnings\n' + compiler_setup)
    replace_once(runtime / 'braids_desktop/execution.py',
                 '        environment.insert("PATH", os.pathsep.join(search))',
                 '        compiler_bin = os.environ.get("BRAIDS_DEV_COMPILER_BIN")\n'
                 '        if compiler_bin: search.insert(0, compiler_bin)\n'
                 '        environment.insert("PATH", os.pathsep.join(search))')
    manifest = dict(schema=1, route='unified_river_v1', compiler_required=True,
                    distribution='BRAIDS source-built development stage',
                    development_source_sha256=hashes,
                    files={name: digest(runtime / name) for name in hashes})
    receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return runtime


def configure_compilers(directory=None):
    if directory:
        directory = str(Path(directory).resolve())
        os.environ['PATH'] = directory + os.pathsep + os.environ.get('PATH', '')
    compiler = shutil.which('gfortran')
    if not compiler or not shutil.which('g++'):
        raise RuntimeError('Install matching 64-bit GNU gfortran and g++ with OpenMP; '
                           'put their bin directory on PATH or pass --compiler-bin.')
    os.environ['BRAIDS_DEV_COMPILER_BIN'] = str(Path(compiler).parent)
    if os.name == 'nt':
        # Retain the handle for all ctypes imports in this GUI process.
        global _compiler_dll_directory
        _compiler_dll_directory = os.add_dll_directory(str(Path(compiler).parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--compiler-bin', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('project', nargs='?')
    args = parser.parse_args()
    configure_compilers(args.compiler_bin)
    runtime = stage()
    print(runtime, flush=True)
    if args.stage_only:
        return
    sys.path.insert(0, str(runtime))
    sys.argv = [str(runtime / 'launch.py')]
    if args.project:
        sys.argv.append(args.project)
    if args.smoke:
        sys.argv.append('--smoke')
    from braids_desktop.app import main as app_main
    raise SystemExit(app_main(runtime=runtime))


if __name__ == '__main__':
    main()
