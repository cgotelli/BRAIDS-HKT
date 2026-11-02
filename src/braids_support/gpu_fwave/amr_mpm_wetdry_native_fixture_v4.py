"""Native-only pocket closure, subcycling, delayed coarsening MPM AMR case.

The wet/dry trigger initially refines the pocket; once it wets, native AMR's
three-event coarsening delay gives a window with active classwise reflux
before the local fine halo can coarsen. This remains exploratory until an
unchanged-native trajectory and pre-GPU gate are frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_qgis.core.results import read_frames
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_native_fixture_v3 as prior
from . import amr_mpm_wetdry_native_fixture_v2 as prior2
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen


TAG = 'amr_mpm_wetdry_regrid_native_20261004_v4'
ROOT = base.HERE / 'runs' / TAG


def geometry():
    tree, settings, state, rock = prior.geometry()
    return tree, replace(settings, refine_hydraulics=False), state, rock


def prepare(output=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Native wet/dry MPM AMR V4 fixture is write-once')
    _, sources = frozen.runtime_identity()
    if base.sha(frozen.TIGHT_EXE) != frozen.TIGHT_SHA:
        raise ValueError('Frozen CPU executable identity changed')
    tree, settings, state, rock = geometry()
    folder = output / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=.08, dt_max_s=.002, output_interval_s=.02, cfl=.4,
        scheme=6, bedload=2, exchange=0, manning=.03,
        dry_tolerance_m=.0001, sediment_fractions=base.fraction(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047,
        secondary_flow=False, morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__),
                       'geometry_parent': base.sha(prior.__file__),
                       'geometry_grandparent': base.sha(prior2.__file__)})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(tag=TAG, deck=str(deck), runtime=str(frozen.TIGHT_EXE),
        runtime_sha256=frozen.TIGHT_SHA,
        source_sha256={**sources, str(Path(__file__).resolve()): base.sha(__file__),
                       str(Path(prior.__file__).resolve()): base.sha(prior.__file__),
                       str(Path(prior2.__file__).resolve()): base.sha(prior2.__file__)},
        inputs={str(path): base.sha(path) for path in folder.iterdir() if path.is_file()},
        settings=dict(levels=3, active_levels=[1, 2], ratio=2,
            regrid_interval_s=.02, native_coarsen_delay_events=3,
            subcycling=True, scheme=6, bedlaw=2,
            grains_m=[.0002, .0003], critical_shields=.047,
            initial_dry_finest_cells=1, dry_pocket_xy=[14.25, 1.25],
            wet_level=1, refine_wet_dry=True, refine_hydraulics=False,
            friction_mode=0, secondary_flow=False, exchange_mode=0,
            morfac=1.), Delft3D_modified=False, GPU_steps_performed=0)
    base.save(output / 'plan.json', plan)
    return plan


def run_native(output=ROOT):
    output = Path(output).resolve()
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    for name, digest in {**plan['source_sha256'], **plan['inputs'],
                         plan['runtime']: plan['runtime_sha256']}.items():
        if base.sha(name) != digest:
            raise ValueError('Frozen native source/input changed: ' + name)
    folder = output / 'cpu'
    exporter.copy_inputs(plan['deck'], folder)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE',
        OMP_WAIT_POLICY='PASSIVE', BRAIDS_TIME_INTEGRATOR='auto')
    with (folder / 'native.log').open('x', encoding='utf-8') as log:
        result = subprocess.run([plan['runtime'], 'case.data'], cwd=folder,
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    tree, _, _, _ = geometry()
    frames = read_frames(folder / 'frames.txt') if result.returncode == 0 else []
    records = []
    for frame in frames:
        ids = frame['node_ids']
        levels = tree.node_level[ids]
        q = np.asarray(frame['q'])
        area = np.asarray([tree.cells[int(i)].area_m2 for i in ids])
        records.append(dict(time_s=float(frame['time_s']), mesh_id=int(frame['mesh_id']),
            leaves=len(ids), levels=[int(np.count_nonzero(levels == level))
                for level in range(tree.levels)],
            dry=int(np.count_nonzero(q[:, 0] <= .0001)),
            min_depth_m=float(np.min(q[:, 0])),
            class_mass_kg=[float(np.sum(area*(q[:, 3+g]+q[:, 5+g]+q[:, 7+g])))
                           for g in range(2)]))
    record = dict(exit_code=result.returncode, frames=records,
        files={path.name: base.sha(path) for path in folder.iterdir() if path.is_file()},
        plan_sha256=base.sha(output / 'plan.json'), Delft3D_modified=False)
    base.save(folder / 'execution.json', record)
    if result.returncode:
        raise RuntimeError('Native wet/dry MPM AMR V4 failed; inspect native.log')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run_native'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
