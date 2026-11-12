"""Stage only the frozen 163/164 corrections; compile portable CPU bindings."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/_shoal165'
TARGET = OUT / 'candidate3'
FILES = ('braids_core/sediment/bed_wave_flux.py', 'braids_core/sediment/river_capacity.py',
         'braids_core/sediment/river_suspension.py', 'braids_core/sediment/wall_velocity_bounds.py',
         'braids_core/sediment/wall_bed_hydraulics.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    frozen = json.loads((ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_wave164/MANIFEST.json').read_text())
    assert sha(ROOT / 'desktop/runtime/RUNTIME.json') == frozen['installed_runtime_sha256']
    shutil.copytree(ROOT / 'desktop/runtime', TARGET)
    manifest = json.loads((TARGET / 'RUNTIME.json').read_text())
    for name in FILES:
        assert sha(ROOT / name) == frozen['sources_sha256'][name], name
        shutil.copy2(ROOT / name, TARGET / name)
        manifest['files'][name] = sha(TARGET / name)
    # Installed runtimes deliberately omit build-only development sources.
    # Compile from the authenticated maintained checkout, then test the actual
    # packaged runtime with its compiler-free loaders in a fresh process.
    sys.path[:0] = [str(ROOT), str(ROOT / 'desktop')]
    from braids_core.sediment import portable_river_runtime as portable
    from braids_core import river_worker
    from braids_gpu.atomic_output import write_json_atomic
    from check_shoal165 import make_document
    # Capture real construction on all exposed formula paths, not guessed keys.
    portable.install()
    for law in ('mpm', 'ashida_michiue_generalized', 'wang_fredsoe'):
        folder = OUT / 'binding_runs3' / law
        folder.mkdir(parents=True, exist_ok=False)
        doc = make_document(folder, 'cpu', law, .0025)
        doc['settings']['sediment_start_time_s'] = 0.
        write_json_atomic(folder / 'project.json', doc)
        prepared = river_worker.prepare(folder / 'project.json', folder / 'prepared')
        status = river_worker.run(prepared, folder / 'run', until=.01)
        assert status['status'] == 'segment_complete', status
    table = json.loads((TARGET / 'PRECOMPILED.json').read_text())
    compiler_folder = Path(shutil.which('gfortran')).parent

    def payload(filename):
        src = Path(filename)
        folder = TARGET / 'precompiled' / sha(src)[:20]
        folder.mkdir(parents=True, exist_ok=True)
        # Native libraries load their Fortran/OpenMP dependencies beside them.
        # The authenticated shoreline probe computes its constructor key from
        # the generated Fortran source even when no runtime compiler is used.
        files = [src, *src.parent.glob('*.f90')]
        files += [compiler_folder / n for n in ('libgomp-1.dll', 'libwinpthread-1.dll',
                  'libgcc_s_seh-1.dll', 'libgfortran-5.dll', 'libquadmath-0.dll', 'libdl.dll')
                  if (compiler_folder / n).is_file()]
        for file in files:
            target = folder / file.name
            if target.exists():
                assert sha(target) == sha(file)
            else:
                shutil.copy2(file, target)
            manifest['files'][target.relative_to(TARGET).as_posix()] = sha(target)
        return dict(file=(folder / src.name).relative_to(TARGET).as_posix(), sha256=sha(src))

    for key, row in portable.RECORDS.items():
        for attr in ('native', 'library'):
            if attr in row:
                row[attr] = payload(row[attr])
        for attr, filename in row.get('libraries', {}).items():
            row['libraries'][attr] = payload(filename)
        table[key] = row
    write_json_atomic(TARGET / 'PRECOMPILED.json', table)
    manifest['files']['PRECOMPILED.json'] = sha(TARGET / 'PRECOMPILED.json')
    write_json_atomic(TARGET / 'RUNTIME.json', manifest)
    write_json_atomic(OUT / 'STAGE.json', dict(runtime_sha256=sha(TARGET / 'RUNTIME.json'),
                      source_files=list(FILES), captured_bindings=list(portable.RECORDS),
                      installed_changed=False))
    print(sha(TARGET / 'RUNTIME.json'), flush=True)


if __name__ == '__main__':
    main()
