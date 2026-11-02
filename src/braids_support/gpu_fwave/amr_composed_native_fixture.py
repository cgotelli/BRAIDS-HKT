"""Unchanged-native exploratory AMR trajectory: MPM, suspension and hI.

This is a new write-once reference *input*, not a GPU pass claim.  It inherits
the validated 91-to-64 wet-pocket topology, adds two nonzero suspended grain
inventories with settling exchange and prognostic secondary-flow transport,
then records exact native source/input/runtime identities before execution.
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
from . import amr_mpm_wetdry_native_fixture_v4 as parent
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen


TAG = 'amr_composed_native_20261004_v1'
ROOT = base.HERE / 'runs' / TAG


def geometry():
    tree, settings, state, rock = parent.geometry()
    xy = np.asarray([cell.centroid_xy for cell in tree.finest.cells])
    # S = h C.  The two concentrations have different spatial gradients, so
    # advection cannot be represented by uniform settling alone.
    c0 = .03*(1. + .2*np.sin(2.*np.pi*xy[:, 0]/16.) +
              .1*np.cos(2.*np.pi*xy[:, 1]/4.))
    c1 = .02*(1. - .15*np.sin(2.*np.pi*xy[:, 0]/16.) +
              .1*np.sin(2.*np.pi*xy[:, 1]/4.))
    state[:, 3] = state[:, 0]*c0
    state[:, 4] = state[:, 0]*c1
    if np.min(state[:, 3:5]) < 0 or np.max(state[:, 3:5]) <= 0:
        raise ValueError('Both water-column classes must be active')
    return tree, settings, state, rock


def fractions():
    first, second = base.fraction()
    return (replace(first, settling_velocity_m_s=.025),
            replace(second, settling_velocity_m_s=.04))


def prepare(output=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Composed native AMR V1 fixture is write-once')
    _, sources = frozen.runtime_identity()
    if base.sha(frozen.TIGHT_EXE) != frozen.TIGHT_SHA:
        raise ValueError('Frozen CPU executable identity changed')
    tree, settings, state, rock = geometry()
    folder = output / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=.08, dt_max_s=.002, output_interval_s=.02, cfl=.4,
        scheme=6, bedload=2, exchange=1, manning=.03,
        dry_tolerance_m=.0001, sediment_fractions=fractions(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047,
        secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_espir=1., secondary_flow_transport_direction=True,
        secondary_flow_momentum_beta=0., morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__),
                       'geometry_parent': base.sha(parent.__file__)})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(tag=TAG, deck=str(deck), runtime=str(frozen.TIGHT_EXE),
        runtime_sha256=frozen.TIGHT_SHA,
        source_sha256={**sources, str(Path(__file__).resolve()): base.sha(__file__),
                       str(Path(parent.__file__).resolve()): base.sha(parent.__file__)},
        inputs={str(path): base.sha(path) for path in folder.iterdir() if path.is_file()},
        settings=dict(levels=3, ratio=2, regrid_interval_s=.02,
            native_coarsen_delay_events=3, subcycling=True, scheme=6,
            bedlaw=2, exchange_mode=1, secondary_flow_model=2,
            grains_m=[.0002, .0003], settling_velocity_m_s=[.025, .04],
            initial_dry_finest_cells=1, dry_pocket_xy=[14.25, 1.25],
            morfac=1.), GPU_steps_performed=0, Delft3D_modified=False)
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
    frames = read_frames(folder / 'frames.txt') if result.returncode == 0 else []
    tree, _, _, _ = geometry()
    records = []
    for frame in frames:
        ids = frame['node_ids']
        q = np.asarray(frame['q'])
        area = np.asarray([tree.cells[int(i)].area_m2 for i in ids])
        levels = tree.node_level[ids]
        if q.shape[1] != 10:
            raise ValueError('Native prognostic secondary row missing')
        records.append(dict(time_s=float(frame['time_s']),
            mesh_id=int(frame['mesh_id']), leaves=len(ids),
            levels=[int(np.count_nonzero(levels == level))
                    for level in range(tree.levels)],
            dry=int(np.count_nonzero(q[:, 0] <= .0001)),
            water_volume_m3=float(np.sum(area*q[:, 0])),
            suspended_class_kg=[float(np.sum(area*q[:, 3+g])) for g in range(2)],
            bed_class_kg=[float(np.sum(area*(q[:, 5+g]+q[:, 7+g])))
                          for g in range(2)],
            total_class_kg=[float(np.sum(area*(q[:, 3+g]+q[:, 5+g]+q[:, 7+g])))
                            for g in range(2)],
            secondary_min=float(np.min(q[:, 9])),
            secondary_max=float(np.max(q[:, 9])),
            secondary_signed_integral=float(np.sum(area*q[:, 9]))))
    record = dict(exit_code=result.returncode, frames=records,
        files={path.name: base.sha(path) for path in folder.iterdir() if path.is_file()},
        plan_sha256=base.sha(output / 'plan.json'), Delft3D_modified=False)
    base.save(folder / 'execution.json', record)
    if result.returncode:
        raise RuntimeError('Composed native AMR V1 failed; inspect native.log')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run_native'))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
