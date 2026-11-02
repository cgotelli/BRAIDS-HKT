"""Thresholded two-grain native AMR fixture for active classwise MPM/Exner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from braids_core.boundaries.cut_cell_io import write_cut_case
from . import amr_hydro_native_fixture as base
from . import amr_mpm_native_fixture as prior
from . import cpu_filter_cold20h as frozen


TAG = 'amr_mpm_subcycled_native_20261004_v2'
ROOT = base.HERE / 'runs' / TAG


def prepare(output=ROOT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Thresholded native MPM AMR fixture is write-once')
    _, sources = frozen.runtime_identity()
    if base.sha(frozen.TIGHT_EXE) != frozen.TIGHT_SHA:
        raise ValueError('Frozen CPU executable identity changed')
    tree, settings, state, rock = prior.geometry()
    folder = output / 'input'
    folder.mkdir(parents=True)
    deck = folder / 'case.data'
    write_cut_case(deck, tree.finest, state, rock, {},
        end_time_s=prior.END, dt_max_s=.002, output_interval_s=.02, cfl=.4,
        scheme=6, bedload=2, exchange=0, manning=.03,
        dry_tolerance_m=.0001, sediment_fractions=base.fraction(),
        active_layer_thickness_m=.05, substrate_layer_count=1,
        substrate_layer_thickness_m=.05, mpm_friction_mode=0,
        mpm_critical_shields=.047,
        secondary_flow=False, morphological_factor=1.,
        amr_hierarchy=tree, amr_settings=settings, compact_sidecars=True,
        source_sha256={'fixture': base.sha(__file__), 'geometry': base.sha(prior.__file__)})
    tree.finest.write_geometry(folder / 'geometry.npz')
    plan = dict(tag=TAG, deck=str(deck), runtime=str(frozen.TIGHT_EXE),
        runtime_sha256=frozen.TIGHT_SHA,
        source_sha256={**sources, str(Path(__file__).resolve()): base.sha(__file__),
                       str(Path(prior.__file__).resolve()): base.sha(prior.__file__)},
        inputs={str(path): base.sha(path) for path in folder.iterdir() if path.is_file()},
        settings=dict(levels=3, active_levels=[1, 2], ratio=2,
            regrid_interval_s=1., scheme=6, bedlaw=2,
            grains_m=[.0002, .0003], critical_shields=.047, friction_mode=0,
            secondary_flow=False, exchange_mode=0, morfac=1.),
        Delft3D_modified=False, GPU_steps_performed=0)
    base.save(output / 'plan.json', plan)
    return plan


def run_native(output=ROOT):
    return prior.run_native(Path(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run_native'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.output), indent=2))


if __name__ == '__main__':
    main()
