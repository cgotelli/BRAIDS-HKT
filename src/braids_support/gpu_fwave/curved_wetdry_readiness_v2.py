"""NEW prepare-only wet/dry fixture with an outlet-connected draining shoal.

V1 is retained: it rewetted the emergent bar but did not produce a robust
wet-to-dry crossing. V2 adds a submerged near-outlet floodplain shoal, initially
quiescent there, and allows4s to drain. No gate or numerical method is relaxed.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from . import curved_wetdry_readiness as old

TAG = 'curved_wetdry_readiness_20261002_v2'
PARENT = old.HERE/'runs/curved_wetdry_readiness_20261002_v1/plan.json'
PARENT_SHA = 'fea00c17f0ffbb2703ec786f0954d9b7824b3c279396b1deda1ae069fdea62f9'
SHOAL_HEIGHT = .096


def specifications():
    return [dict(row, duration=4.) for row in old.specifications()]


def initial_fields(xy):
    q, bar, _ = old.initial_fields(xy)
    x, y = np.asarray(xy).T
    shoal = SHOAL_HEIGHT*np.exp(-((x-7.75)/.75)**4-((y-1.875)/.40)**4)
    bed = bar+shoal
    velocity = np.divide(q[:, 1:3], q[:, 0, None], out=np.zeros((len(q), 2)), where=q[:, 0, None] > old.DRY)
    # A quiescent inundated shoal next to a flowing channel; changes only the
    # newly added downstream feature, not a seeded secondary intensity.
    velocity *= (1.-shoal/SHOAL_HEIGHT)[:, None]
    q[:, 0] = np.maximum(old.STAGE-bed, 0.)
    q[:, 1:3] = q[:, 0, None]*velocity
    q[q[:, 0] <= old.DRY, 1:3] = 0.
    return q, bed, bed-old.MOBILE, shoal


def prepare_case(row, folder):
    from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction, write_cut_case
    from braids_core.boundaries.cut_cells import map_cut_cells
    from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    from .cpu_cold_ulp24 import decode_binary
    if row not in specifications(): raise ValueError('Only the explicit V2 controls are supported')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    dx = row['dx']; nx, ny = round(old.LENGTH/dx), round(old.WIDTH/dx)
    domain = PolygonDomain('NEW draining-shoal active-secondary wet/dry fixture',
        [(0., 0.), (old.LENGTH, 0.), (old.LENGTH, old.WIDTH), (0., old.WIDTH)], 'EPSG:2056')
    mesh = map_cut_cells(domain, [BoundaryAssignment(domain.edges[3].edge_id, 'in', 'inlet'),
        BoundaryAssignment(domain.edges[1].edge_id, 'out', 'outlet')], NativeGrid(nx, ny, 0., 0., dx, dx))
    xy = np.array([c.centroid_xy for c in mesh.cells]); area = np.array([c.area_m2 for c in mesh.cells])
    if len(xy) != nx*ny or not np.all(area == dx*dx): raise ValueError('Uniform rectangle required')
    edges = np.array([(f.owner, f.neighbour) for f in mesh.faces if f.neighbour >= 0], dtype=np.int32)
    q, bed, rock, shoal = initial_fields(xy)
    raw, production, activity = old.host_activity(xy, q, bed, area, edges)
    intended_drain = (shoal > old.OUTLET_STAGE+.005) & (q[:, 0] >= old.GATES['initial_wet_test_depth_m'])
    if not intended_drain.any(): raise ValueError('No initially wet shoal above lowered tailwater')
    activity.update(initially_wet_shoal_above_tailwater_cells=int(intended_drain.sum()),
        shoal_maximum_elevation_m=float(shoal.max()),
        outlet_shoal_minimum_initial_depth_m=float(q[(xy[:, 0] > old.LENGTH-dx) & (shoal > .085), 0].min()),
        actual_shoal_drying_observed=False)
    fractions = tuple(CutSedimentFraction(fraction_id=f'sand_{k}', d50_m=.0002, d90_m=.0003,
        grain_density_kg_m3=2650., porosity=.4, settling_velocity_m_s=0., z0_m=.0002/30.,
        minimum_bedload_depth_m=old.DRY, minimum_exchange_depth_m=old.DRY, roughness_mode='manning',
        initial_volume_fraction=.5, initial_active_volume_fraction=.5) for k in range(2))
    boundaries = {'in': CutBoundary('total_discharge', discharge_m3_s=old.WIDTH*old.STAGE*old.SPEED),
        'out': CutBoundary('stage', stage_m=old.OUTLET_STAGE)}
    write_cut_case(folder/'case.data', mesh, q, rock, boundaries, end_time_s=row['duration'],
        dt_max_s=row['dt'], output_interval_s=.25, cfl=.4, scheme=6, bedload=2, exchange=0,
        manning=old.MANNING, dry_tolerance_m=old.DRY, sediment_fractions=fractions,
        active_layer_thickness_m=old.ACTIVE, substrate_layer_count=1,
        substrate_layer_thickness_m=old.MOBILE-old.ACTIVE, mpm_friction_mode=1, mpm_critical_shields=.047,
        bed_slope=2, secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_transport_direction=True, secondary_flow_momentum_beta=1., morphological_factor=1.,
        compact_sidecars=True, source_sha256={'fixture_v2': old.sha(__file__), 'unchanged_v1_helpers': old.sha(old.__file__)})
    decoded = decode_binary((folder/'case.data.bin').read_bytes())
    for key, expected in (('q', q), ('rock', rock), ('xy', xy), ('area', area)):
        if not np.array_equal(decoded[key].view(np.uint64), expected.view(np.uint64)):
            raise ValueError('Common physical input roundtrip differs: '+key)
    mesh.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'initial_analysis.npz', xy=xy, area=area, q=q, bed=bed, rock=rock,
        edges=edges, hI=np.zeros(len(q)), added_shoal=shoal, intended_drain_cells=intended_drain,
        diagnostic_raw_curvature=raw, diagnostic_unfiltered_hI_production=production)
    return dict(**row, cells=len(q), input=str(folder/'case.data'), initial_activity=activity,
        output_times_s=(np.arange(17)*.25).tolist(),
        input_files={str(p): old.sha(p) for p in folder.iterdir() if p.is_file()},
        native_initializer_checked=False, device_scope_checked=False)


def prepare(output=None):
    output = Path(output) if output is not None else old.HERE/'runs'/TAG
    if output.exists(): raise FileExistsError('Preserve existing attempt')
    if old.sha(PARENT) != PARENT_SHA: raise ValueError('Frozen V1 plan changed')
    parent = json.loads(PARENT.read_text())
    for path, digest in parent['source_files'].items():
        if old.sha(path) != digest: raise ValueError('Frozen V1 helper changed')
    if old.sha(old.CPU) != old.CPU_SHA: raise ValueError('Frozen intended CPU runtime changed')
    output.mkdir(parents=True); rows = [prepare_case(row, output/row['name']) for row in specifications()]
    plan = dict(parent, tag=TAG, fixture_revision=2, cases=rows,
        source_files={**parent['source_files'], str(Path(__file__).resolve()): old.sha(__file__)},
        parent_plan=str(PARENT), parent_plan_sha256=PARENT_SHA,
        parent_failed_gate='No robust initial-wet>=.002m to dry<=.0005m transition by2s; preserved without relaxing its gate',
        explicit_changes=['Add .096m submerged quartic shoal centred(7.75,1.875)m, widths(.75,.40)m, touching stage outlet',
            'Damp initial velocity smoothly only over added shoal, with multiplier1-shoal/.096; hI still starts at0',
            'Duration2s ->4s, same .25s output spacing, timestep/CFL/MF/grains/layers/boundaries/methods unchanged'],
        original_gate_values_unchanged=True, observed_transitions_still_required=True,
        not_a_relabelled_V1_pass=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(plan=str(output/'plan.json'), sha256=old.sha(output/'plan.json'), cases=len(rows),
        native_executions=0, GPU_calls=0, prepared_only=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path)
    print(json.dumps(prepare(p.parse_args().output), indent=2))
