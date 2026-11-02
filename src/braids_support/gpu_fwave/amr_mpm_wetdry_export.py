"""Private native geometry/material exports for the V4 wet/dry MPM AMR case.

The unchanged CPU21b7 run supplies accepted states and topology. Copied-source
diagnostic executables expose immutable arrays and stop without a numerical
step. GPU evolution must use its own state, never a later native frame.
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
from . import amr_mpm_native_fixture_v2 as prior
from . import amr_mpm_wetdry_native_fixture_v4 as case
from . import amr_mpm_provenance_v2 as prior_lock
from . import coupled_reference as exported


ROOT = case.ROOT / 'mixed_export_v1'
AMR_DIAGNOSTIC_RUNTIME = prior.ROOT / 'native_path_probe_v2/runtime/xbraids_amr_stage_probe.exe'
FIXED_DIAGNOSTIC_RUNTIME = prior.ROOT / 'mixed_export_v1/runtime/xbraids_coupled_export.exe'


def _execute(runtime, folder, *, restart=False):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE',
        OMP_WAIT_POLICY='PASSIVE', BRAIDS_TIME_INTEGRATOR='auto',
        BRAIDS_GPU_INITIAL_EXPORT='1')
    args = [str(runtime), 'case.data']
    if restart:
        args.append('checkpoint.bin')
    result = subprocess.run(args, cwd=folder, capture_output=True, text=True,
        timeout=120, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        env=env)
    (folder / 'export.log').write_text(result.stdout + result.stderr,
        encoding='utf-8')
    if result.returncode or not (folder / 'coupled_input.bin').is_file():
        raise RuntimeError('Private native AMR MPM geometry export failed: ' + str(folder))
    return exported.read_snapshot(folder)


def _matched_geometry(left, right):
    names = ('state', 'xy', 'area', 'owner', 'neighbour', 'length', 'normal',
             'distance', 'mid', 'hydraulic_ls_weight', 'manning_cell')
    return {name: bool(np.array_equal(left[name], right[name])) for name in names}


def export(output=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Wet/dry MPM AMR geometry export is write-once')
    parent = prior_lock.preflight()
    plan = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen native V4 source/input changed: ' + name)
    if base.sha(AMR_DIAGNOSTIC_RUNTIME) != \
            json.loads((prior.ROOT / 'native_path_probe_v2/READBACK.json').read_text())[
                'private_amr_diagnostic_runtime_sha256']:
        raise ValueError('Private AMR exporter identity changed')
    identity = json.loads((prior.ROOT / 'mixed_export_v1/build.json').read_text())
    if base.sha(FIXED_DIAGNOSTIC_RUNTIME) != identity['executable_sha256']:
        raise ValueError('Private fixed selected-leaf exporter identity changed')
    frames = read_frames(case.ROOT / 'cpu/frames.txt')
    if len(frames) != 5 or len(frames[0]['node_ids']) != 91 or \
            len(frames[3]['node_ids']) != 64 or \
            not np.array_equal(frames[3]['node_ids'], frames[4]['node_ids']):
        raise ValueError('Native V4 topology is not the declared 91-to-64 case')
    tree, _, _, _ = case.geometry()
    for name, restarted in (('amr_initial', False), ('amr_final', True)):
        folder = output / name
        exported.copy_inputs(plan['deck'], folder)
        if restarted:
            shutil.copy2(case.ROOT / 'cpu/checkpoint.bin', folder / 'checkpoint.bin')
        data = _execute(AMR_DIAGNOSTIC_RUNTIME, folder, restart=restarted)
        target = frames[4] if restarted else frames[0]
        if data['nc'] != len(target['node_ids']) or \
                not np.array_equal(data['state'].T, target['q']):
            raise ValueError('Native AMR exporter state differs from frozen frame')
        if int(data['amr_enabled']) != 1:
            raise ValueError('Private AMR export lost adaptive identity')
    for index in (0, 3):
        frame = frames[index]
        ids = np.asarray(frame['node_ids'], dtype=np.int64)
        mesh = tree.select(ids)
        state = np.asarray(frame['q'], dtype=np.float64)
        folder = output / f'mesh_{index}'
        folder.mkdir(parents=True)
        deck = folder / 'case.data'
        write_cut_case(deck, mesh, state, np.full(len(state), -.1), {},
            end_time_s=.0001, dt_max_s=.0001, output_interval_s=.0001,
            cfl=.4, scheme=6, bedload=2, exchange=0, manning=.03,
            dry_tolerance_m=.0001, sediment_fractions=base.fraction(),
            active_layer_thickness_m=.05, substrate_layer_count=1,
            substrate_layer_thickness_m=.05, mpm_friction_mode=0,
            mpm_critical_shields=.047,
            secondary_flow=False, morphological_factor=1.,
            compact_sidecars=True, source_sha256={'driver': base.sha(__file__),
                                                  'native_plan': base.sha(case.ROOT / 'plan.json')})
        mesh.write_geometry(folder / 'geometry.npz')
        np.savez_compressed(folder / 'leaf_identity.npz', node_ids=ids,
            levels=tree.node_level[ids],
            area=np.asarray([cell.area_m2 for cell in mesh.cells]),
            xy=np.asarray([cell.centroid_xy for cell in mesh.cells]),
            initial=state, native_time_s=np.float64(frame['time_s']))
        data = _execute(FIXED_DIAGNOSTIC_RUNTIME, folder)
        if int(data['amr_enabled']) != 0 or data['nc'] != len(ids) or \
                data['bedlaw'] != 2 or data['nfrac'] != 2 or \
                not np.array_equal(data['state'].T, state):
            raise ValueError('Selected-leaf MPM material export differs from native')
    init_amr = exported.read_snapshot(output / 'amr_initial')
    init_fixed = exported.read_snapshot(output / 'mesh_0')
    final_amr = exported.read_snapshot(output / 'amr_final')
    final_fixed = exported.read_snapshot(output / 'mesh_3')
    geometry = dict(initial=_matched_geometry(init_amr, init_fixed),
                    final={key: value for key, value in
                           _matched_geometry(final_amr, final_fixed).items()
                           if key != 'state'})
    if not all(geometry['initial'].values()) or not all(geometry['final'].values()):
        raise ValueError('Private AMR/static topology geometry mismatch')
    weno_initial = np.any(init_amr['weno_cells'] != init_fixed['weno_cells'],
                          axis=(0, 1))
    weno_final = np.any(final_amr['weno_cells'] != final_fixed['weno_cells'],
                        axis=(0, 1))
    record = dict(scope='Private immutable geometry/material exports; no GPU numerical step',
        native_plan_sha256=base.sha(case.ROOT / 'plan.json'),
        native_frames_sha256=base.sha(case.ROOT / 'cpu/frames.txt'),
        native_checkpoint_sha256=base.sha(case.ROOT / 'cpu/checkpoint.bin'),
        parent_source_lock_sha256=parent['sha256'],
        native_runtime_sha256=plan['runtime_sha256'],
        private_AMR_export_runtime_sha256=base.sha(AMR_DIAGNOSTIC_RUNTIME),
        private_fixed_export_runtime_sha256=base.sha(FIXED_DIAGNOSTIC_RUNTIME),
        matched_geometry=geometry,
        initial_native_AMR_WENO_different_faces=int(np.count_nonzero(weno_initial)),
        final_native_AMR_WENO_different_faces=int(np.count_nonzero(weno_final)),
        exported_mesh_counts=[int(init_fixed['nc']), int(final_fixed['nc'])],
        files={str(path.relative_to(output)): base.sha(path)
               for path in output.rglob('*') if path.is_file()},
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(output / 'receipt.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
