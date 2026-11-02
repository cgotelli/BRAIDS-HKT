"""Frozen unchanged-native 0–0.02 s composed mixed-level AMR reference.

Identical physical input to the full V2 case, with end time at its first
output. The native run and copied-source diagnostic export expose the ten-row
checkpoint before any Gate-B1 GPU evolution.
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
from . import amr_composed_one_step_native as one
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_export as prior_export
from . import coupled_reference as exported


ROOT = case.ROOT / 'mixed_composed_reference_b1'


def prepare():
    if ROOT.exists():
        raise FileExistsError('Composed B1 native reference is write-once')
    parent = json.loads((case.ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**parent['source_sha256'], **parent['inputs'],
                         parent['runtime']: parent['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen full native source/input changed: '+name)
    tree, settings, state, rock = case.geometry()
    folder = ROOT / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=.02, dt_max_s=.002, output_interval_s=.02, cfl=.4,
        scheme=6, bedload=2, exchange=1, manning=.03,
        dry_tolerance_m=.0001,
        sediment_fractions=case.parent.fractions(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047, secondary_flow=True,
        secondary_flow_model='prognostic', secondary_flow_espir=1.,
        secondary_flow_transport_direction=True,
        secondary_flow_momentum_beta=0., morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__),
                       'full_native_plan': base.sha(case.ROOT / 'plan.json')})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(runtime=parent['runtime'],
        runtime_sha256=parent['runtime_sha256'],
        parent_plan_sha256=base.sha(case.ROOT / 'plan.json'),
        source_sha256={str(Path(__file__).resolve()):base.sha(__file__),
                       str(Path(case.__file__).resolve()):base.sha(case.__file__)},
        inputs={str(path):base.sha(path) for path in folder.iterdir() if path.is_file()},
        case='0–0.02 s native mixed-level MPM/Exner + two-class suspension/exchange + signed hI',
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(ROOT / 'plan.json', plan)
    return plan


def run_native():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']:plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen B1 native source/input changed: '+name)
    folder = ROOT / 'cpu'
    exported.copy_inputs(ROOT / 'input/case.data', folder)
    env = {key:value for key,value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
               BRAIDS_TIME_INTEGRATOR='auto')
    with (folder / 'native.log').open('x', encoding='utf-8') as log:
        result = subprocess.run([plan['runtime'], 'case.data'], cwd=folder,
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError('Unchanged native composed B1 failed')
    frames = read_frames(folder / 'frames.txt')
    full = read_frames(case.ROOT / 'cpu/frames.txt')
    if (len(frames) != 2 or frames[1]['time_s'] != .02 or
            frames[1]['steps'] != 10 or
            not np.array_equal(frames[0]['node_ids'], frames[1]['node_ids']) or
            not np.array_equal(frames[1]['node_ids'], full[1]['node_ids']) or
            not np.array_equal(frames[1]['q'], full[1]['q'])):
        raise ValueError('B1 native output differs from frozen full V2 prefix')
    budget = np.loadtxt(folder / 'secondary_flow_budget.txt',
                         comments='#', ndmin=2)
    full_budget = np.loadtxt(case.ROOT / 'cpu/secondary_flow_budget.txt',
                              comments='#', ndmin=2)
    if budget.shape != (2,6) or not np.array_equal(budget[1],full_budget[1]):
        raise ValueError('B1 signed hI budget differs from full V2 prefix')
    record = dict(exit_code=0, time_s=.02, accepted_outer_steps=10,
        native_leaves=91, native_l1_substeps=18, native_l2_substeps=36,
        prefix_matches_full_native_bitwise=True,
        frames_sha256=base.sha(folder/'frames.txt'),
        checkpoint_sha256=base.sha(folder/'checkpoint.bin'),
        secondary_budget_sha256=base.sha(folder/'secondary_flow_budget.txt'),
        plan_sha256=base.sha(ROOT/'plan.json'), no_GPU_steps=True,
        Delft3D_modified=False)
    base.save(folder/'execution.json',record)
    return record


def export():
    frames = read_frames(ROOT / 'cpu/frames.txt')
    if len(frames) != 2:
        raise ValueError('B1 native frames required')
    folder = ROOT / 'amr_final'
    exported.copy_inputs(ROOT / 'input/case.data', folder)
    shutil.copy2(ROOT/'cpu/checkpoint.bin',folder/'checkpoint.bin')
    result = prior_export._execute(prior_export.AMR_DIAGNOSTIC_RUNTIME,
                                    folder, restart=True)
    if (result['state'].shape != (10, 91) or
            not np.array_equal(result['state'][:9].T,frames[1]['q']) or
            int(result['secondary_flow_model']) != 2 or
            int(result['exchange_mode']) != 1):
        raise ValueError('B1 native ten-row final export changed')
    initial = exported.read_snapshot(case.ROOT /
        'amr_full_state_export_v1/initial')
    if not np.array_equal(initial['state'][:9].T,frames[0]['q']):
        raise ValueError('B1 initial state changed')
    record = dict(plan_sha256=base.sha(ROOT/'plan.json'),
        frames_sha256=base.sha(ROOT/'cpu/frames.txt'),
        checkpoint_sha256=base.sha(ROOT/'cpu/checkpoint.bin'),
        final_full_state_sha256=base.sha(folder/'coupled_input.bin'),
        initial_full_state_sha256=base.sha(case.ROOT/
            'amr_full_state_export_v1/initial/coupled_input.bin'),
        diagnostic_runtime_sha256=base.sha(prior_export.AMR_DIAGNOSTIC_RUNTIME),
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(ROOT/'export_receipt.json',record)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run_native','export'))
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](),indent=2))
