"""Write-once unchanged-native mixed-level graded-MPM/Exner AMR reference.

This deliberately begins with a fully wet static selected topology.  It
exercises real level-1/2 subcycling and classwise bedload/reflux, but not
regridding, bank motion, suspension, or a QGIS execution path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from braids_core.boundaries.amr import AMRSettings, build_hierarchy
from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_qgis.core.results import read_frames
from . import amr_hydro_native_fixture as base
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen


TAG = 'amr_mpm_subcycled_native_20261004_v1'
ROOT = base.HERE / 'runs' / TAG
END = .08


def geometry():
    domain = PolygonDomain('Two-grain active MPM AMR GPU prototype',
        [(0., 0.), (16., 0.), (16., 4.), (0., 4.)], 'EPSG:2056')
    settings = AMRSettings(enabled=True, levels=3, wet_level=1,
        regrid_interval_s=1., buffer_cells=2,
        refine_wet_dry=False, refine_walls=False, refine_bathymetry=False,
        refine_hydraulics=True, refine_sediment=False,
        surface_tolerance_m=.03, velocity_tolerance_m_s=.2,
        subcycling=True)
    tree = build_hierarchy(domain, [], NativeGrid(8, 2, 0., 0., 2., 2.), levels=3)
    xy = np.asarray([cell.centroid_xy for cell in tree.finest.cells])
    state = np.zeros((len(xy), 9), dtype=np.float64)
    state[:, 0] = np.where(xy[:, 0] < 4., .5, .1)
    velocity = .75 + .10 * np.sin(np.pi * xy[:, 0] / 8.)
    state[:, 1] = state[:, 0] * velocity
    state[:, 2] = state[:, 0] * .08 * np.sin(np.pi * xy[:, 1] / 4.)
    state[:, 5:7] = .025 * base.BULK
    state[:, 7:9] = .025 * base.BULK
    rock = np.full(len(xy), -.1)
    return tree, settings, state, rock


def prepare(output=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Native MPM AMR fixture is write-once')
    _, sources = frozen.runtime_identity()
    if base.sha(frozen.TIGHT_EXE) != frozen.TIGHT_SHA:
        raise ValueError('Frozen CPU executable identity changed')
    tree, settings, state, rock = geometry()
    folder = output / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=END, dt_max_s=.002, output_interval_s=.02, cfl=.4,
        scheme=6, bedload=2, exchange=0, manning=.03,
        dry_tolerance_m=.0001, sediment_fractions=base.fraction(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        secondary_flow=False, morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__)})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(tag=TAG, deck=str(deck), runtime=str(frozen.TIGHT_EXE),
        runtime_sha256=frozen.TIGHT_SHA,
        source_sha256={**sources, str(Path(__file__).resolve()): base.sha(__file__)},
        inputs={str(path): base.sha(path) for path in folder.iterdir() if path.is_file()},
        settings=dict(levels=3, active_levels=[1, 2], ratio=2,
            regrid_interval_s=1., scheme=6, bedlaw=2,
            grains_m=[.0002, .0003], friction_mode=0, secondary_flow=False,
            exchange_mode=0, morfac=1.),
        Delft3D_modified=False, GPU_steps_performed=0)
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
        levels = tree.node_level[frame['node_ids']]
        q = np.asarray(frame['q'])
        area = np.asarray([tree.cells[int(i)].area_m2 for i in frame['node_ids']])
        records.append(dict(time_s=float(frame['time_s']), mesh_id=int(frame['mesh_id']),
            leaves=len(levels), coarse=int(np.count_nonzero(levels == 1)),
            fine=int(np.count_nonzero(levels == 2)),
            class_mass_kg=[float(np.sum(area * (q[:, 3+g] + q[:, 5+g] + q[:, 7+g])))
                           for g in range(2)],
            active_mass_change_max_kg_m2=(float(np.max(np.abs(q[:, 5:7] - frames[0]['q'][:, 5:7])))
                if np.array_equal(frame['node_ids'], frames[0]['node_ids']) else None)))
    record = dict(exit_code=result.returncode, frames=records,
        files={path.name: base.sha(path) for path in folder.iterdir() if path.is_file()},
        plan_sha256=base.sha(output / 'plan.json'), Delft3D_modified=False)
    base.save(folder / 'execution.json', record)
    if result.returncode:
        raise RuntimeError('Native MPM AMR failed; inspect native.log')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run_native'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.output), indent=2))


if __name__ == '__main__':
    main()
