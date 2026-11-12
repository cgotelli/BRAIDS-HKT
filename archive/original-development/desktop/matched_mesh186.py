"""Isolate refinement from initial wetness on the held v6 runtime.

No production edits, project migrations, physical toggles or outlet/face
prototypes. Each job starts from the original wet initial condition.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'validation/_secondary185/candidate'
OUT = ROOT/'validation/_mesh186'
BASE = ROOT/'desktop/examples/two_meanders184/advanced_on'
sys.path[:0] = [str(RUNTIME), str(ROOT)]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from stage_secondary185 import sha, save


def document(mesh, backend):
    original = json.loads((BASE/'runs/to_200/PROJECT.json').read_text())
    refined = json.loads((BASE/'runs/run_20261027_084649_a62dd04f/PROJECT.json').read_text())
    doc = deepcopy(original)
    doc.pop('editor_state', None)
    doc['terrain']['dem_path'] = str((BASE/doc['terrain']['dem_path']).resolve())
    if mesh == 'refined':
        for key in ('grid', 'levels'):
            doc['settings'][key] = deepcopy(refined['settings'][key])
    doc['settings']['output_interval_s'] = 1.
    from braids_core.river_model import resolve_model
    request = doc['model']['request']
    request['execution']['backend'] = backend
    doc['model'] = resolve_model(request).document()
    assert doc['initial']['stage_m'] == 1.6
    assert not doc['settings']['adaptive']
    assert doc['model']['physics']['sediment']['bed_slope'] == 'bed_slope.off'
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', choices=('coarse', 'refined'), required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    folder = OUT/f'{args.mesh}_{args.backend}'
    folder.mkdir(parents=True, exist_ok=False)
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    verify()
    assert sha(RUNTIME/'RUNTIME.json') == '1b078961ad95f0038ba07da0a4aca9c3dd15a7f3cbffff2ec413986d27b4723b'
    install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()), RUNTIME)
    doc = document(args.mesh, args.backend)
    services.save_project_folder(folder/'project', doc)
    save(folder/'INPUT.json', dict(mesh=args.mesh, backend=args.backend, runtime=sha(RUNTIME/'RUNTIME.json'),
         source=str(BASE/'runs/to_200/PROJECT.json'), initial=doc['initial'],
         settings=doc['settings'], physics=doc['model']['physics'],
         scope='Cold wet start; only grid/levels and backend differ; output every 1 s'))
    begin = time.perf_counter()
    prepare(folder/'project/braids_project.json', folder/'prepared', asset_cache=folder/'cache')
    print(args.mesh, args.backend, 'PREPARED', round(time.perf_counter()-begin, 2), flush=True)
    def progress(output, row):
        t = row['time_s']
        if t >= 199 or t % 20 == 0:
            print(args.mesh, args.backend, 'time', t, 'wall', round(time.perf_counter()-begin, 2), flush=True)
    status = river_worker.run(folder/'prepared', folder/'run', until=220., on_frame=progress)
    results = ResultsStore(folder/'run')
    first, last = results.read(0), results.read(-1)
    assert status['status'] == 'segment_complete' and status['time_s'] == 220., status
    assert np.isfinite(last.q).all() and last.q[0].min() >= 0
    assert np.isfinite(last.z).all() and last.arrays['mass'].min() >= 0
    diagnostics = json.loads((folder/'run/DIAGNOSTICS.json').read_text())
    for row in diagnostics:
        assert abs(row['water_residual_m3']) <= row['water_roundoff_limit_m3']
        assert max(abs(v) for v in row['class_residual_kg']) <= row['class_roundoff_limit_kg']
        assert not any(row['wall_integrals'])
    save(folder/'CHECK.json', dict(status=status, elapsed_s=time.perf_counter()-begin,
         cells=len(last.z), minimum_area_m2=float(last.arrays['area'].min()),
         bed_min_mm=float(1000*(last.z-first.z).min()), bed_max_mm=float(1000*(last.z-first.z).max()),
         water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),
         sediment_residual_kg=max(max(abs(v) for v in r['class_residual_kg']) for r in diagnostics),
         runtime=sha(RUNTIME/'RUNTIME.json')))
    print('DONE', args.mesh, args.backend, flush=True)


if __name__ == '__main__':
    main()
