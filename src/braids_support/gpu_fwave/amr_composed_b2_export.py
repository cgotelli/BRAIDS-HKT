"""Read-only composed 64-leaf fixed exporter for post-coarsening geometry.

The exported *state* is a diagnostic placeholder and is never used as the GPU
post-regrid state. The GPU obtains all ten accepted rows from CUDA restriction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_qgis.core.results import read_frames
from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_b2_native as native
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_export as exporter
from . import coupled_reference


ROOT = case.ROOT / 'composed_mesh3_export_v1'


def run():
    if ROOT.exists():
        raise FileExistsError('Composed post-regrid export is write-once')
    plan = json.loads((native.ROOT/'plan.json').read_text(encoding='utf-8'))
    for path, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(path) != digest:
            raise ValueError('Frozen native B2 source/input changed: '+path)
    frames = read_frames(native.ROOT/'cpu/frames.txt')
    if len(frames) != 4 or frames[3]['time_s'] != .06:
        raise ValueError('Native post-coarsening frame required')
    tree, _, _, _ = case.geometry()
    ids = np.asarray(frames[3]['node_ids'], dtype=np.int64)
    if (len(ids) != 64 or
            not np.array_equal(ids, np.flatnonzero(tree.node_level == 1))):
        raise ValueError('Unexpected native L1 selected leaves')
    mesh = tree.select(ids)
    ROOT.mkdir(parents=True)
    deck = ROOT/'case.data'
    # Native 0.06 s state is only a parseable exporter placeholder. The GPU
    # post-transfer constructor replaces every row before any Euler stage.
    write_cut_case(deck, mesh, np.asarray(frames[3]['q']),
        np.full(len(ids), -.1), {},
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
        source_sha256={'driver':base.sha(__file__),
                       'native_plan':base.sha(native.ROOT/'plan.json')})
    mesh.write_geometry(ROOT/'geometry.npz')
    fixed = exporter._execute(exporter.FIXED_DIAGNOSTIC_RUNTIME, ROOT)
    amr = coupled_reference.read_snapshot(native.ROOT/'amr_final')
    equal = {key:bool(np.array_equal(fixed[key], amr[key])) for key in
             ('xy', 'area', 'owner', 'neighbour', 'length', 'normal',
              'distance', 'mid', 'hydraulic_ls_weight', 'manning_cell')}
    if not all(equal.values()) or fixed['state'].shape != (10, 64):
        raise ValueError('Post-regrid fixed/native AMR geometry mismatch')
    record = dict(native_plan_sha256=base.sha(native.ROOT/'plan.json'),
        native_final_state_sha256=base.sha(native.ROOT/'amr_final/coupled_input.bin'),
        fixed_export_sha256=base.sha(ROOT/'coupled_input.bin'),
        native_weno_changed_faces=int(np.count_nonzero(np.any(
            fixed['weno_cells'] != amr['weno_cells'], axis=(0,1)))),
        matching_geometry=equal, placeholder_state_not_a_GPU_input=True,
        no_GPU_steps=True, Delft3D_modified=False)
    base.save(ROOT/'RECEIPT.json', record)
    return record


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
