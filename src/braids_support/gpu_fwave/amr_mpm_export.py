"""Private read-only coupled geometry/material export for selected AMR leaves.

The copied-source executable stops after native initialization. The unchanged
CPU21b7 reference still runs the separate AMR case; this fixed selected-leaf
deck merely exposes its geometry/material arrays to the GPU adapter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_qgis.core.results import read_frames
from . import amr_hydro_native_fixture as base
from . import amr_mpm_native_fixture_v2 as case
from . import amr_mpm_native_fixture as geometry_case
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen


ROOT = case.ROOT / 'mixed_export_v1'


def build(destination=ROOT):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Private AMR MPM export is write-once')
    parent = frozen.TIGHT_EXE.parent
    identity = json.loads((parent / 'build.json').read_text(encoding='utf-8'))
    if base.sha(frozen.TIGHT_EXE) != frozen.TIGHT_SHA:
        raise ValueError('Frozen CPU21b7 runtime changed')
    for name, digest in identity['source_sha256'].items():
        if base.sha(parent / name) != digest:
            raise ValueError('Frozen CPU21b7 source changed: ' + name)
    runtime = destination / 'runtime'
    runtime.mkdir(parents=True)
    sources = []
    for name in identity['source_sha256']:
        old = parent / name
        new = runtime / name
        if name == 'cutcell_solver.f90':
            new.write_text(exporter.instrument(old.read_text(encoding='utf-8')),
                           encoding='utf-8')
        else:
            shutil.copy2(old, new)
        sources.append(new)
    executable = runtime / 'xbraids_coupled_export.exe'
    command = [r'C:\Strawberry\c\bin\gfortran.exe', '-O3', '-flto', '-fopenmp',
        '-ffree-line-length-none', '-ffixed-line-length-none',
        *(str(path) for path in sources), '-o', str(executable)]
    result = subprocess.run(command, cwd=runtime, capture_output=True,
        text=True, timeout=300)
    (runtime / 'build.log').write_text(result.stdout + result.stderr,
        encoding='utf-8')
    receipt = dict(parent_executable_sha256=frozen.TIGHT_SHA,
        frozen_sources=identity['source_sha256'],
        export_sources={path.name: base.sha(path) for path in sources},
        executable_sha256=base.sha(executable) if result.returncode == 0 else None,
        export_transform_sha256=base.sha(exporter.__file__),
        driver_sha256=base.sha(__file__), command=command,
        exit_code=result.returncode, original_native_formulas_unchanged=True,
        Delft3D_modified=False)
    base.save(destination / 'build.json', receipt)
    if result.returncode:
        raise RuntimeError('Private AMR MPM geometry export failed to compile')
    return receipt


def export(destination=ROOT):
    destination = Path(destination).resolve()
    build_record = json.loads((destination / 'build.json').read_text(encoding='utf-8'))
    runtime = destination / 'runtime/xbraids_coupled_export.exe'
    if base.sha(runtime) != build_record['executable_sha256']:
        raise ValueError('Private export executable changed')
    for name, digest in build_record['export_sources'].items():
        if base.sha(destination / 'runtime' / name) != digest:
            raise ValueError('Private export source changed: ' + name)
    plan = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Native AMR MPM source/input changed: ' + name)
    tree, _, _, _ = geometry_case.geometry()
    first = read_frames(case.ROOT / 'cpu/frames.txt')[0]
    ids = np.asarray(first['node_ids'], dtype=np.int64)
    levels = tree.node_level[ids]
    if len(ids) != 160 or np.count_nonzero(levels == 1) != 32 or np.count_nonzero(levels == 2) != 128:
        raise ValueError('V2 native AMR selected topology changed')
    q = np.ascontiguousarray(first['q'])
    mesh = tree.select(ids)
    selected = destination / 'mesh_0'
    selected.mkdir(exist_ok=False)
    deck = selected / 'case.data'
    write_cut_case(deck, mesh, q, np.full(len(q), -.1), {},
        end_time_s=.0001, dt_max_s=.0001, output_interval_s=.0001,
        cfl=.4, scheme=6, bedload=2, exchange=0, manning=.03,
        dry_tolerance_m=.0001, sediment_fractions=base.fraction(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047, secondary_flow=False,
        morphological_factor=1., compact_sidecars=True,
        source_sha256={'export_driver': base.sha(__file__),
                       'native_plan': base.sha(case.ROOT / 'plan.json')})
    mesh.write_geometry(selected / 'geometry.npz')
    np.savez_compressed(selected / 'leaf_identity.npz', node_ids=ids,
        levels=levels, area=np.asarray([cell.area_m2 for cell in mesh.cells]),
        initial=q, xy=np.asarray([cell.centroid_xy for cell in mesh.cells]))
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE',
        OMP_WAIT_POLICY='PASSIVE', BRAIDS_TIME_INTEGRATOR='auto',
        BRAIDS_GPU_INITIAL_EXPORT='1')
    result = subprocess.run([str(runtime), deck.name], cwd=selected, env=env,
        capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (selected / 'export.log').write_text(result.stdout + result.stderr,
        encoding='utf-8')
    if result.returncode or not (selected / 'coupled_input.bin').is_file():
        raise RuntimeError('Private AMR MPM selected-leaf export failed')
    data = exporter.read_snapshot(selected)
    np.testing.assert_array_equal(data['state'].T, q)
    if data['amr_enabled'] or data['bedlaw'] != 2 or data['nfrac'] != 2:
        raise ValueError('Selected-leaf material export lost graded MPM controls')
    receipt = dict(native_plan_sha256=base.sha(case.ROOT / 'plan.json'),
        native_frames_sha256=base.sha(case.ROOT / 'cpu/frames.txt'),
        private_export_executable_sha256=base.sha(runtime),
        source_sha256=base.sha(__file__), selected_leaves=len(ids),
        coarse=int(np.count_nonzero(levels == 1)),
        fine=int(np.count_nonzero(levels == 2)),
        exported_static_amr_flag=int(data['amr_enabled']),
        exported_bedlaw=int(data['bedlaw']),
        files={path.name: base.sha(path) for path in selected.iterdir() if path.is_file()},
        GPU_steps_performed=0, Delft3D_modified=False)
    base.save(destination / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'export'))
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.destination), indent=2))


if __name__ == '__main__':
    main()
