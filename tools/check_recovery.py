"""Exercise checkout-only development recovery on a small synthetic flume.

No measurements, saved mesh, installed native libraries or existing project
are used. Generated files stay in .recovery-check (or the chosen output).
This is a recovery smoke check, not a replacement for physical validation.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from develop import ROOT, configure_compilers, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-bin', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--features', choices=('hydraulics', 'turbine', 'sediment', 'combined'), default='hydraulics')
    parser.add_argument('--output', type=Path, default=ROOT / '.recovery-check')
    args = parser.parse_args()
    configure_compilers(args.compiler_bin)
    runtime = stage()
    sys.path.insert(0, str(runtime))
    import numpy as np
    from osgeo import gdal, osr
    from braids_desktop.services import default_document, assign_boundary, validate_document
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_model import HLLC_INTEGRATOR, resolve_model, new_bedload_selection
    from braids_gpu.atomic_output import write_json_atomic
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dem = output / 'synthetic_dem.tif'
    osr.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)
    gdal.UseExceptions()
    ds = gdal.GetDriverByName('GTiff').Create(str(dem), 60, 40, 1, gdal.GDT_Float64)
    ds.SetGeoTransform((-1, .1, 0, 3, 0, -.1))
    ds.SetProjection(srs.ExportToWkt())
    x = -1 + (np.arange(60) + .5) * .1
    ds.GetRasterBand(1).WriteArray(np.broadcast_to(-.001 * x, (40, 60)).copy())
    ds = None
    document = default_document(str(dem), [[0, 0], [4, 0], [4, 2], [0, 2]],
                                'EPSG:2056', minimum_size_m=.5, levels=1,
                                backend=args.backend, end_time_s=.05)
    domain = PolygonDomain(**document['settings']['domain'])
    for kind, coordinate, controls in (
        ('inlet', 0., dict(method='inlet.total_discharge', discharge_m3_s=.05)),
        ('outlet', 4., dict(method='outlet.stage', stage_m=.2))):
        edge = next(e for e in domain.edges if e.start_xy[0] == coordinate and e.end_xy[0] == coordinate)
        document = assign_boundary(document, [edge.edge_id], kind, kind, controls=controls)
    document['initial']['stage_m'] = .2
    document['settings'].update(dt_max_s=.01, output_interval_s=.05, checkpoint_interval_s=.05)
    results = {}
    invocation = args.features + '_' + uuid4().hex[:8]
    environment = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1')
    for name, integrator in (('fwave', document['model']['request']['execution']['integrator']),
                             ('hllc', HLLC_INTEGRATOR)):
        case = deepcopy(document)
        request = case['model']['request']
        request['execution']['integrator'] = integrator
        if args.features in ('sediment', 'combined'):
            selected = new_bedload_selection()
            selected['capacity'] = dict(method='capacity.mpm', parameters={'ACal': 1.},
                                        grain_sizes_m={'d50': .0005, 'd90': .00075})
            selected['bed'].update(composition='bed.layered', transport_layer='transport_layer.constant')
            request['sediment'] = dict(enabled=True, selection=selected)
        if args.features == 'combined':
            request['hydrodynamics']['secondary_flow'] = 'secondary.prognostic'
            case['settings']['controls']['betac'] = 1.
        if args.features in ('turbine', 'combined'):
            from braids_core.river_turbines import default, definition
            turbines = default()
            rotor = definition([1., 1.], [1., 0.], diameter=.2, hub_height=.15,
                               minimum_bed_clearance=.02, minimum_surface_clearance=.02)
            rotor['orientation'] = dict(mode='fixed_axis', yaw_deg=20.)
            turbines['turbines'] = [rotor]
            case['turbines'] = turbines
            case['initial']['stage_m'] = .5
            case['initial']['velocity_m_s'] = [.2, 0.]
            case['settings']['hydraulics']['outlet']['stage_m'] = .5
        case['model'] = resolve_model(request).document()
        from braids_core.river_open_boundary import new_state, resolve as resolve_boundaries
        enabled = case['model']['physics']['sediment'] is not None
        case['boundary_states'] = {key: new_state(value['method'], sediment=enabled)
                                   for key, value in case['settings']['hydraulics'].items()}
        case['boundary_resolution'] = resolve_boundaries(
            case['settings']['hydraulics'], case['boundary_states'],
            dict(inlet='inlet', outlet='outlet'), sediment=case['model']['physics']['sediment'],
            ng=1, end_time=case['settings']['end_time_s'])
        case = validate_document(case)
        folder = output / runtime.parent.name / invocation / (name + '_' + args.backend)
        folder.mkdir(parents=True, exist_ok=True)
        project = folder / 'project.json'
        write_json_atomic(project, case)
        prepared, run = folder / 'prepared', folder / 'run'
        commands = [
            [sys.executable, '-B', str(runtime / 'desktop/worker.py'), str(runtime / 'braids_core/river_launcher.py'),
             'prepare', '--project', str(project), '--output', str(prepared), '--asset-cache', str(folder / 'assets')],
            [sys.executable, '-B', str(runtime / 'desktop/worker.py'), str(runtime / 'braids_core/river_launcher.py'),
             'run', '--input', str(prepared), '--output', str(run)]]
        started = time.perf_counter()
        for i, command in enumerate(commands):
            print(f'{name}/{args.backend}: {"prepare" if i == 0 else "run"}', flush=True)
            with (folder / f'phase_{i}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, env=environment, cwd=runtime, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=900)
            if result.returncode:
                raise RuntimeError(f'Recovery check failed: {folder / f"phase_{i}.log"}')
        from braids_desktop.results import ResultsStore
        store = ResultsStore(run)
        assert store.times[-1] >= .05 - 1e-10, store.times
        frame = store.read(len(store.times) - 1)
        for field in ('q', 'z'):
            assert np.isfinite(frame.arrays[field]).all(), field
        assert (frame.arrays['q'][0] >= 0).all()
        results[name] = dict(last_time_s=float(store.times[-1]), frames=len(store.times),
                             cells=int(frame.arrays['q'].shape[1]), elapsed_s=time.perf_counter() - started)
    receipt = dict(schema=1, runtime=str(runtime), backend=args.backend, features=args.features, cases=results,
                   source_only=True, installed_solver_libraries_used=False)
    write_json_atomic(output / f'RECOVERY_{args.backend}_{args.features}.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
