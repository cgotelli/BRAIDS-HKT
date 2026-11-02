"""Frozen one-step wet/dry AMR composition reference and read-only exports.

This narrows the next GPU gate to the first synchronous AMR step while the
private subcycled AMR adapter remains 9-row/unsuspended.  Unchanged CPU21b7
produces the 10-row post-step state; copied-source diagnostic exporters only
read that state and selected-leaf geometry, without taking numerical steps.
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
from . import amr_composed_native_fixture_v2 as case
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_export as prior_export
from . import coupled_reference as exported


ROOT = case.ROOT / 'one_step_composed_reference_v1'


def prepare():
    if ROOT.exists():
        raise FileExistsError('Composed one-step reference is write-once')
    original = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**original['source_sha256'], **original['inputs'],
                         original['runtime']: original['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen full native source/input changed: '+name)
    tree, settings, state, rock = case.geometry()
    folder = ROOT / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=.002, dt_max_s=.002, output_interval_s=.002, cfl=.4,
        scheme=6, bedload=2, exchange=1, manning=.03,
        dry_tolerance_m=.0001,
        sediment_fractions=case.parent.fractions(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047,
        secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_espir=1., secondary_flow_transport_direction=True,
        secondary_flow_momentum_beta=0., morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__),
                       'full_native_plan': base.sha(case.ROOT / 'plan.json')})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(runtime=original['runtime'],
        runtime_sha256=original['runtime_sha256'],
        parent_plan_sha256=base.sha(case.ROOT / 'plan.json'),
        source_sha256={str(Path(__file__).resolve()): base.sha(__file__),
                       str(Path(case.__file__).resolve()): base.sha(case.__file__)},
        inputs={str(path): base.sha(path) for path in folder.iterdir() if path.is_file()},
        case='0.002 s active two-grain MPM/Exner with exchange and prognostic hI on wet/dry AMR mesh',
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(ROOT / 'plan.json', plan)
    return plan


def run_native():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen one-step source/input changed: '+name)
    folder = ROOT / 'cpu'
    exported.copy_inputs(ROOT / 'input/case.data', folder)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
        BRAIDS_TIME_INTEGRATOR='auto')
    with (folder / 'native.log').open('x', encoding='utf-8') as log:
        result = subprocess.run([plan['runtime'], 'case.data'], cwd=folder,
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError('Unchanged native composed one-step failed')
    frames = read_frames(folder / 'frames.txt')
    if (len(frames) != 2 or frames[1]['time_s'] != .002 or
            frames[1]['steps'] != 1 or
            len(frames[0]['node_ids']) != 91 or
            not np.array_equal(frames[0]['node_ids'], frames[1]['node_ids'])):
        raise ValueError('One-step native wet/dry AMR schedule changed')
    secondary = np.loadtxt(folder / 'secondary_flow_budget.txt',
                           comments='#', ndmin=2)
    if secondary.shape != (2, 6) or abs(secondary[1, 1]) <= 1.e-8:
        raise ValueError('Native one-step signed hI inactive')
    record = dict(exit_code=0, time_s=.002, accepted_steps=1,
        initial_leaves=91, final_leaves=91,
        native_frames_sha256=base.sha(folder / 'frames.txt'),
        native_checkpoint_sha256=base.sha(folder / 'checkpoint.bin'),
        secondary_integral=float(secondary[1, 1]),
        secondary_source_integral=float(secondary[1, 4]),
        secondary_budget_residual=float(secondary[1, 5]),
        runtime_sha256=plan['runtime_sha256'],
        plan_sha256=base.sha(ROOT / 'plan.json'),
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(folder / 'execution.json', record)
    return record


def export():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen one-step source/input changed: '+name)
    frames = read_frames(ROOT / 'cpu/frames.txt')
    if len(frames) != 2:
        raise ValueError('Two accepted native frames required')
    folder = ROOT / 'amr_final'
    exported.copy_inputs(ROOT / 'input/case.data', folder)
    shutil.copy2(ROOT / 'cpu/checkpoint.bin', folder / 'checkpoint.bin')
    final = prior_export._execute(prior_export.AMR_DIAGNOSTIC_RUNTIME,
                                  folder, restart=True)
    if (final['state'].shape != (10, 91) or
            not np.array_equal(final['state'][:9].T, frames[1]['q'])):
        raise ValueError('Native final full state differs from one-step frame')
    tree, _, _, _ = case.geometry()
    ids = np.asarray(frames[0]['node_ids'], dtype=np.int64)
    mesh = tree.select(ids)
    first = np.asarray(frames[0]['q'])
    fixed_folder = ROOT / 'mesh_0'
    fixed_folder.mkdir()
    fixed_deck = fixed_folder / 'case.data'
    write_cut_case(fixed_deck, mesh, first, np.full(len(first), -.1), {},
        end_time_s=.0001, dt_max_s=.0001, output_interval_s=.0001,
        cfl=.4, scheme=6, bedload=2, exchange=1, manning=.03,
        dry_tolerance_m=.0001,
        sediment_fractions=case.parent.fractions(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047,
        secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_espir=1., secondary_flow_transport_direction=True,
        secondary_flow_momentum_beta=0., morphological_factor=1.,
        compact_sidecars=True,
        source_sha256={'driver': base.sha(__file__),
                       'native_plan': base.sha(ROOT / 'plan.json')})
    mesh.write_geometry(fixed_folder / 'geometry.npz')
    fixed = prior_export._execute(prior_export.FIXED_DIAGNOSTIC_RUNTIME,
                                   fixed_folder)
    initial = exported.read_snapshot(case.ROOT /
        'amr_full_state_export_v1/initial')
    geometry = ('state', 'xy', 'area', 'owner', 'neighbour', 'length',
                'normal', 'distance', 'mid', 'hydraulic_ls_weight',
                'manning_cell')
    matches = {name: bool(np.array_equal(fixed[name], initial[name]))
               for name in geometry}
    if not all(matches.values()) or final['nc'] != initial['nc']:
        raise ValueError('Fixed selected-leaf/native AMR initial geometry differs')
    if (int(fixed['exchange_mode']) != 1 or
            int(fixed['secondary_flow_model']) != 2 or
            int(fixed['nfrac']) != 2 or int(fixed['amr_enabled']) != 0):
        raise ValueError('Fixed diagnostic lost coupled physics')
    record = dict(native_one_step_plan_sha256=base.sha(ROOT / 'plan.json'),
        native_one_step_frames_sha256=base.sha(ROOT / 'cpu/frames.txt'),
        native_one_step_checkpoint_sha256=base.sha(ROOT / 'cpu/checkpoint.bin'),
        native_initial_state_sha256=base.sha(case.ROOT /
            'amr_full_state_export_v1/initial/coupled_input.bin'),
        native_final_state_sha256=base.sha(folder / 'coupled_input.bin'),
        fixed_selected_leaf_state_sha256=base.sha(fixed_folder/'coupled_input.bin'),
        geometry_matches=matches,
        native_amr_weno_different_faces=int(np.count_nonzero(np.any(
            initial['weno_cells'] != fixed['weno_cells'], axis=(0, 1)))),
        native_amr_diagnostic_sha256=base.sha(prior_export.AMR_DIAGNOSTIC_RUNTIME),
        fixed_diagnostic_sha256=base.sha(prior_export.FIXED_DIAGNOSTIC_RUNTIME),
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(ROOT / 'export_receipt.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run_native', 'export'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
