"""Summarize bounded public CPU/GPU AMR-record qualification, not timings."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braids_core.sediment.adaptive_river_worker import read_live_frame
from braids_gpu.atomic_output import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for path in args.input.glob('test_public*/PUBLIC_RECORDS132.json'):
        row = json.loads(path.read_text())
        key = ('bedload' if row['sediment'] else 'hydro', row['backend'])
        if key in runs:
            raise ValueError('Duplicate public run: ' + str(key))
        frame, mesh, fields = read_live_frame(path.parent/'whole')
        runs[key] = (row, frame, mesh, fields)
    if set(runs) != {(mode, backend) for mode in ('hydro', 'bedload') for backend in ('cpu', 'gpu')}:
        raise ValueError('All four completed CPU/GPU hydro/bedload trajectories required')
    comparison = {}
    for mode in ('hydro', 'bedload'):
        c, g = runs[mode, 'cpu'], runs[mode, 'gpu']
        if c[2].keys() != g[2].keys() or c[3].keys() != g[3].keys():
            raise ValueError('CPU/GPU field or geometry inventory differs')
        for name in c[2]:
            np.testing.assert_array_equal(c[2][name], g[2][name], err_msg=name)
        errors = {}
        for name, values in c[3].items():
            reference = g[3][name]
            if np.shape(values) != np.shape(reference):
                raise ValueError('CPU/GPU field shape differs: ' + name)
            errors[name] = float(np.max(np.abs(values-reference), initial=0.))
        if max(errors.values()) > 1e-8:
            raise ValueError('CPU/GPU public AMR trajectory difference exceeds 1e-8')
        comparison[mode] = dict(geometry_identical=True, field_max_errors=errors,
            final_time_s=c[1]['time_s'],
            maximum_bed_change_from_initial_m={backend:float(np.max(np.abs(
                runs[mode, backend][3]['z']-runs[mode, backend][3]['rock']-.22)))
                for backend in ('cpu', 'gpu')},
            evidence={backend:runs[mode, backend][0] for backend in ('cpu', 'gpu')})
    write_json_atomic(args.output, dict(comparison=comparison, timing_claim=False,
        scope='0.2 s cropped-wall dry inundation with actual dynamic AMR, cancellation and exact restart'))
    print(json.dumps({mode:dict(max_error=max(row['field_max_errors'].values()),
        bed_change=row['maximum_bed_change_from_initial_m'])
        for mode, row in comparison.items()}, indent=2))


if __name__ == '__main__':
    main()
