"""Fresh current-schema shallow shoal through the real public Desktop worker.

No old preparation/checkpoint is reused. Public bedload laws only; this is not
the historical EH/KF/spiral experiment and must not be labelled as such.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_document(out, backend, law, dt):
    import numpy as np
    from osgeo import gdal, osr
    from braids_core.boundaries.domain import PolygonDomain
    from braids_core.river_model import resolve_model
    from braids_core.river_open_boundary import resolve
    doc = json.loads((ROOT / 'validation/_hydraulics162/mobile/braids_mpm_cpu/project/braids_project.json').read_text())
    doc['name'] = f'Desktop165 shallow shoal {law} {backend}'
    # The .25 m DEM uses the same analytic cell-centre bed as the retained shoal.
    x = (np.arange(36) - 1.5) * .25
    y = (np.arange(12) - 1.5) * .25
    xx, yy = np.meshgrid(x, y[::-1])
    bed = .11 * np.exp(-((xx - 3.) ** 4 + ((yy - 1.) / .7) ** 4))
    dem = out / 'shoal.tif'
    dataset = gdal.GetDriverByName('GTiff').Create(str(dem), 36, 12, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform((-.5, .25, 0., 2.5, 0., -.25))
    crs = osr.SpatialReference(); crs.ImportFromEPSG(2056)
    dataset.SetProjection(crs.ExportToWkt()); dataset.GetRasterBand(1).WriteArray(bed)
    dataset.FlushCache(); dataset = None
    doc['terrain']['dem_path'] = str(dem)
    settings = doc['settings']
    settings['domain']['vertices_xy'] = [[0., 0.], [8., 0.], [8., 2.], [0., 2.], [0., 0.]]
    domain = PolygonDomain(**settings['domain'])
    settings['assignments'] = []
    for xpos, name, kind in ((0., 'Upstream', 'inlet'), (8., 'Downstream', 'outlet')):
        edge = next(e for e in domain.edges if e.start_xy[0] == e.end_xy[0] == xpos)
        settings['assignments'].append(dict(bc_id=name, edge_id=edge.edge_id, kind=kind, start_m=0., end_m=None))
    settings['grid'] = dict(dx_m=.25, dy_m=.25, nx=32, ny=8, xlower_m=0., ylower_m=0.)
    settings.update(end_time_s=28., dt_max_s=dt, output_interval_s=1., checkpoint_interval_s=2., sediment_start_time_s=24.)
    settings['hydraulics'] = dict(Upstream=dict(method='inlet.stage_discharge', discharge_m3_s=.04, stage_m=.1),
                                  Downstream=dict(method='outlet.stage', stage_m=.08))
    settings['sediment_boundaries']['Upstream']['solid_discharge_m3_s'] = [0.]
    diameter = float(np.sqrt(.00044 * .00046))
    settings['controls'].update(chezy=28.8, diameter=diameter, morfac=10., cfl=1., threads=2)
    settings['columns']['diameter_m'] = [diameter]
    doc['roughness_parameters']['chezy_m_half_s'] = 28.8
    doc['initial'].update(stage_m=.1, velocity_m_s=[0., 0.], layer_thickness_m=[.5])
    request = copy.deepcopy(doc['model']['request'])
    request['execution']['backend'] = backend
    request['sediment']['selection']['grains'][0]['diameter_m'] = diameter
    # Retain the exact exposed, already resolved formula parameter set.
    source = ROOT / f'validation/_hydraulics162/mobile/braids_{law}_cpu/project/braids_project.json'
    request['sediment']['selection']['capacity'] = json.loads(source.read_text())['model']['request']['sediment']['selection']['capacity']
    doc['model'] = resolve_model(request).document()
    doc['boundary_resolution'] = resolve(settings['hydraulics'], doc['boundary_states'],
        dict(Upstream='inlet', Downstream='outlet'), sediment=doc['model']['physics']['sediment'], ng=1, end_time=28.)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--law', default='mpm', choices=['mpm', 'ashida_michiue_generalized', 'wang_fredsoe'])
    ap.add_argument('--dt', type=float, default=.0025)
    ap.add_argument('--tag', default='base')
    ap.add_argument('--until', type=float, default=28.)
    ap.add_argument('--restart-from', type=Path)
    ap.add_argument('--restart-time', type=float)
    ap.add_argument('--runtime', type=Path)
    args = ap.parse_args()
    if args.runtime:
        args.runtime = args.runtime.resolve()
        sys.path.insert(0, str(args.runtime))
        from braids_core.sediment.portable_river_runtime import install
        install(json.loads((args.runtime / 'PRECOMPILED.json').read_text()), args.runtime)
    from braids_core import river_worker
    from braids_gpu.atomic_output import write_json_atomic
    out = ROOT / f'validation/_shoal165/{args.tag}/{args.law}_{args.backend}'
    out.mkdir(parents=True, exist_ok=False)
    if args.restart_from:
        prepared = args.restart_from / 'prepared'
        checkpoint = args.restart_from / 'run/checkpoint.npz'
        if args.restart_time is not None:
            matches = list((args.restart_from / 'checkpoints').glob(f't_{args.restart_time:g}_step_*.npz'))
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one saved checkpoint: {matches}')
            checkpoint = matches[0]
    else:
        write_json_atomic(out / 'project.json', make_document(out, args.backend, args.law, args.dt))
        prepared = river_worker.prepare(out / 'project.json', out / 'prepared')
        checkpoint = None
    status = river_worker.run(prepared, out / 'run', until=args.until, restart=checkpoint,
                              checkpoint_dir=out / 'checkpoints')
    write_json_atomic(out / 'STATUS.json', status)
    print(status, flush=True)


if __name__ == '__main__':
    main()
