"""Prepare-only small active-secondary wet/dry mobile-bar qualification inputs.

This is a NEW transient synthetic diagnostic, not the old Delft3D bar validation
or Johannot's experiment. Geometry is rectangular; *streamlines* are curved.
No CuPy import, compiler, exporter or native/GPU solver is invoked. The native
initializer must subsequently prove zero conserved hI and exact physical input
parity before matched CPU/GPU execution. Preparation is not qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAG = "curved_wetdry_readiness_20261002_v1"
CPU_SHA = "21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5"
CPU = HERE / "runs/cpu_filter_convergence_control_20261002_v1/runtime/xbraids_filter_tight.exe"
LENGTH, WIDTH = 8., 2.
DRY, MANNING, BULK = .001, 1. / 36., 1590.
STAGE, OUTLET_STAGE, SPEED = .1, .08, .4
MOBILE, ACTIVE = .1, .05

# Predeclared small-test numerical gates, not experimental error bounds.
# These are deliberately separate from the 5 mm full-bend acceptance limit.
GATES = dict(
    minimum_depth_m=-1e-12, minimum_layer_inventory_kg_m2=-1e-12,
    water_budget_absolute_m3=1e-9, sediment_budget_each_label_absolute_kg=1e-7,
    secondary_budget_absolute_m4_s=1e-9,
    CPU_GPU_bed_maximum_m=1e-6, CPU_GPU_bed_RMS_m=2e-7,
    CPU_GPU_stage_maximum_m=1e-5, CPU_GPU_depth_maximum_m=1e-5,
    CPU_GPU_velocity_RMS_m_s=1e-3, velocity_comparison_minimum_depth_m=.01,
    minimum_initially_wet_cells_that_dry=1, minimum_initially_dry_cells_that_wet=1,
    initial_wet_test_depth_m=.002, initial_dry_test_depth_m=.0005,
    final_wet_test_depth_m=.002, final_dry_test_depth_m=.0005,
    minimum_evolving_filter_calls_with_positive_iterations=1,
    minimum_evolving_maximum_abs_hI_m2_s=1e-8,
    minimum_accumulated_absolute_local_secondary_source_m4_s=1e-10,
    minimum_mobile_bed_maximum_change_m=1e-7,
    half_dt_bed_maximum_change_m=1e-4, half_dt_bed_RMS_change_m=2e-5,
    half_dt_stage_maximum_change_m=1e-3,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications():
    """The refined grid is prepared, but execution remains separately authorized."""
    return [dict(name="bar256", dx=.25, duration=2., dt=.002),
            dict(name="bar256_half_dt", dx=.25, duration=2., dt=.001),
            dict(name="bar1024", dx=.125, duration=2., dt=.001)]


def validate_spec(row):
    if row not in specifications():
        raise ValueError("Only the three declared fresh fixture controls are supported")
    return row


def initial_fields(xy):
    """Cell-centre sampled initial field; all inventories are per retained area.

    Smooth quartic bar is inspired by the existing 256-cell bar test, but height,
    flow, material, layers and secondary controls are explicitly NEW. hI is not
    artificially seeded: the native v37+ loader appends its original zero value.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Finite n-by-2 cell centres required")
    x, y = xy.T
    if np.any((x < 0.) | (x > LENGTH) | (y < 0.) | (y > WIDTH)):
        raise ValueError("Initial field only defined inside the declared rectangle")
    bed = .105 * np.exp(-((x - 3.) / 1.) ** 4 - ((y - 1.) / .7) ** 4)
    q = np.zeros((len(xy), 9), dtype=np.float64)
    q[:, 0] = np.maximum(STAGE - bed, 0.)
    envelope = np.sin(np.pi * x / LENGTH) ** 2 * np.exp(-((x - 3.) / 1.5) ** 2)
    v = -.10 * envelope * np.sin(2. * np.pi * y / WIDTH)
    wet = q[:, 0] > DRY
    q[wet, 1] = q[wet, 0] * SPEED
    q[wet, 2] = q[wet, 0] * v[wet]
    q[:, 5:7] = BULK * ACTIVE / 2.
    q[:, 7:9] = BULK * (MOBILE - ACTIVE) / 2.
    return q, bed, bed - MOBILE


def raw_curvature_diagnostic(xy, state, edges):
    """Host-only LS diagnostic following st_raw_curvature, not a solver oracle.

    Interior face neighbours only; dry-neighbour velocity is zero. Boundary
    ghost values are deliberately not invented. This demonstrates a nonzero
    curvature RHS without initializing a device or advancing a timestep.
    """
    xy = np.asarray(xy, dtype=float); state = np.asarray(state, dtype=float)
    if xy.shape != (len(state), 2) or state.shape != (len(xy), 9):
        raise ValueError("Physical state/geometry shape mismatch")
    if not np.isfinite(xy).all() or not np.isfinite(state).all() or np.any(state[:, 0] < 0):
        raise ValueError("Finite nonnegative depth required")
    neighbours = [[] for _ in xy]
    for first, second in edges:
        first, second = int(first), int(second)
        if not (0 <= first < len(xy) and 0 <= second < len(xy)) or first == second:
            raise ValueError("Interior face index out of range")
        neighbours[first].append(second); neighbours[second].append(first)
    velocity = np.zeros((len(xy), 2)); wet = state[:, 0] >= DRY
    velocity[wet] = state[wet, 1:3] / state[wet, 0, None]
    raw = np.zeros(len(xy))
    for c, adjacent in enumerate(neighbours):
        u, v = velocity[c]; speed = float(np.hypot(u, v))
        if not wet[c] or speed < .001:
            continue
        a11 = a12 = a22 = bx = by = cx = cy = 0.
        for n in adjacent:
            rx, ry = xy[n] - xy[c]; w = 1. / max(float(np.hypot(rx, ry)), 1e-6) ** 2
            un, vn = velocity[n]
            a11 += w * rx * rx; a12 += w * rx * ry; a22 += w * ry * ry
            bx += w * rx * (un - u); by += w * ry * (un - u)
            cx += w * rx * (vn - v); cy += w * ry * (vn - v)
        det = a11 * a22 - a12 * a12
        if det < 1e-6:
            continue
        dudx = (bx * a22 - a12 * by) / det; dudy = (a11 * by - bx * a12) / det
        dvdx = (cx * a22 - a12 * cy) / det; dvdy = (a11 * cy - cx * a12) / det
        raw[c] = -(u * u * dvdx - v * v * dudy + u * v * (dvdy - dudx)) / speed ** 3
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite diagnostic curvature")
    return raw


def host_activity(xy, q, bed, area, edges):
    raw = raw_curvature_diagnostic(xy, q, edges)
    wet = q[:, 0] > DRY; speed = np.zeros(len(q))
    speed[wet] = np.hypot(q[wet, 1], q[wet, 2]) / q[wet, 0]
    # Zero-hI instantaneous source before Helmholtz filtering. This is NOT the
    # actual filtered source receipt or proof that any evolving filter iterates.
    production = np.zeros(len(q)); chezy = q[wet, 0] ** (1. / 6.) / MANNING
    alpha = np.sqrt(9.81) / (.41 * np.maximum(chezy, 20.))
    scale = (1. - 2. * alpha) / (2. * .41 ** 2 * alpha)
    if np.any(scale <= 0.):
        raise ValueError("Invalid secondary relaxation scale")
    production[wet] = q[wet, 0] * speed[wet] ** 2 * raw[wet] / scale
    record = dict(wet_cells=int(wet.sum()), dry_cells=int((~wet).sum()),
        strongly_wet_cells=int(np.sum(q[:, 0] >= .002)),
        strongly_dry_cells=int(np.sum(q[:, 0] <= .0005)),
        raw_curvature_maximum_abs_per_m=float(np.max(abs(raw))),
        raw_curvature_positive_cells=int(np.sum(raw > 1e-6)),
        raw_curvature_negative_cells=int(np.sum(raw < -1e-6)),
        unfiltered_zero_hI_production_maximum_abs=float(np.max(abs(production))),
        initial_water_m3=float(q[:, 0] @ area),
        initial_sediment_each_label_kg=[float((q[:, 5+k] + q[:, 7+k]) @ area) for k in range(2)],
        maximum_initial_bed_m=float(np.max(bed)), actual_filtered_source_evaluated=False,
        actual_filter_iterations_observed=False, evolved_wet_dry_transitions_observed=False,
        imposed_nonzero_initial_hI=False)
    if (min(record['strongly_wet_cells'], record['strongly_dry_cells'],
            record['raw_curvature_positive_cells'], record['raw_curvature_negative_cells']) <= 0
            or record['raw_curvature_maximum_abs_per_m'] <= 1e-3
            or record['unfiltered_zero_hI_production_maximum_abs'] <= 1e-8):
        raise ValueError("Fixture does not expose the intended initial activity")
    return raw, production, record


def prepare_case(row, folder):
    """Write one common deck. No source modifications or numerical execution."""
    from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction, write_cut_case
    from braids_core.boundaries.cut_cells import map_cut_cells
    from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
    from braids_core.boundaries.inlet_io import NativeGrid
    from .cpu_cold_ulp24 import decode_binary

    validate_spec(row); folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    dx = row['dx']; nx, ny = round(LENGTH / dx), round(WIDTH / dx)
    domain = PolygonDomain("NEW active-secondary wet/dry bar", [(0., 0.), (LENGTH, 0.),
        (LENGTH, WIDTH), (0., WIDTH)], "EPSG:2056")
    assignments = [BoundaryAssignment(domain.edges[3].edge_id, 'in', 'inlet'),
                   BoundaryAssignment(domain.edges[1].edge_id, 'out', 'outlet')]
    mesh = map_cut_cells(domain, assignments, NativeGrid(nx, ny, 0., 0., dx, dx))
    xy = np.array([c.centroid_xy for c in mesh.cells]); area = np.array([c.area_m2 for c in mesh.cells])
    if len(xy) != nx * ny or not np.all(area == dx * dx):
        raise ValueError("Expected exact uncut uniform rectangle")
    edges = np.array([(f.owner, f.neighbour) for f in mesh.faces if f.neighbour >= 0], dtype=np.int32)
    q, bed, rock = initial_fields(xy); raw, production, activity = host_activity(xy, q, bed, area, edges)
    fractions = tuple(CutSedimentFraction(fraction_id=f'sand_{k}', d50_m=.0002, d90_m=.0003,
        grain_density_kg_m3=2650., porosity=.4, settling_velocity_m_s=0., z0_m=.0002/30.,
        minimum_bedload_depth_m=DRY, minimum_exchange_depth_m=DRY, roughness_mode='manning',
        initial_volume_fraction=.5, initial_active_volume_fraction=.5) for k in range(2))
    boundaries = {'in': CutBoundary('total_discharge', discharge_m3_s=WIDTH * STAGE * SPEED),
                  'out': CutBoundary('stage', stage_m=OUTLET_STAGE)}
    write_cut_case(folder/'case.data', mesh, q, rock, boundaries,
        end_time_s=row['duration'], dt_max_s=row['dt'], output_interval_s=.25, cfl=.4,
        scheme=6, bedload=2, exchange=0, manning=MANNING, dry_tolerance_m=DRY,
        sediment_fractions=fractions, active_layer_thickness_m=ACTIVE,
        substrate_layer_count=1, substrate_layer_thickness_m=MOBILE-ACTIVE,
        mpm_friction_mode=1, mpm_critical_shields=.047, bed_slope=2,
        secondary_flow=True, secondary_flow_model='prognostic',
        secondary_flow_transport_direction=True, secondary_flow_momentum_beta=1.,
        morphological_factor=1., compact_sidecars=True,
        source_sha256={'fixture': sha(__file__)})
    decoded = decode_binary((folder/'case.data.bin').read_bytes())
    for name, expected in (('q', q), ('rock', rock), ('xy', xy), ('area', area)):
        if not np.array_equal(decoded[name].view(np.uint64), expected.view(np.uint64)):
            raise ValueError('Common input roundtrip changed ' + name)
    mesh.write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'initial_analysis.npz', xy=xy, area=area, q=q, rock=rock, bed=bed,
        edges=edges, hI=np.zeros(len(q)), diagnostic_raw_curvature=raw,
        diagnostic_unfiltered_hI_production=production)
    return dict(**row, cells=len(q), input=str(folder/'case.data'),
        initial_activity=activity, output_times_s=(np.arange(9, dtype=float)*.25).tolist(),
        input_files={str(p): sha(p) for p in folder.iterdir() if p.is_file()},
        native_initializer_checked=False, device_scope_checked=False)


def prepare(output=None):
    output = Path(output) if output is not None else HERE/'runs'/TAG
    if output.exists():
        raise FileExistsError('Preserve every existing fixture attempt: ' + str(output))
    if sha(CPU) != CPU_SHA:
        raise ValueError('Intended frozen CPU21b7 runtime changed')
    output.mkdir(parents=True)
    rows = [prepare_case(row, output/row['name']) for row in specifications()]
    # Authenticate input-writing and diagnostic source; this is not a runtime
    # launch plan. Actual GPU and native-exporter selection remains root-owned.
    from braids_core.boundaries import cut_cell_io, cut_cells, domain, inlet_io
    from . import cpu_cold_ulp24
    helpers = [Path(m.__file__).resolve() for m in (cut_cell_io, cut_cells, domain, inlet_io, cpu_cold_ulp24)]
    plan = dict(schema=1, tag=TAG, cases=rows, prepared_only=True, native_executions=0,
        GPU_calls=0, numerical_steps=0, source_files={str(p): sha(p) for p in [Path(__file__).resolve(), *helpers]},
        intended_CPU=dict(executable=str(CPU), sha256=CPU_SHA, threads=8, integrator='ssp43',
            outlet_extension=1, filter_rtol=1e-8),
        intended_GPU='Explicit backend; qualified secondary-only outlet-bound repair, tight MG1e-8; exact source/graph plan not yet frozen',
        controls=dict(MorFac=1., CFL=.4, scheme=6, dry_m=DRY, secondary_model=2, secondary_beta=1.,
            active_layer_m=ACTIVE, substrate_layers=1, substrate_layer_m=MOBILE-ACTIVE,
            identical_labels=2, suspended_exchange=0, hiding=0, adaptation=False, avalanche=False,
            inlet_m3_s=WIDTH*STAGE*SPEED, inlet_sediment_feed_kg_s=0., outlet_stage_m=OUTLET_STAGE),
        numerical_execution_gates=GATES,
        execution_order=['native zero-step export and unchanged host scope guards',
            'prove exact common q/layers/rock and zero hI/ledgers before graph capture',
            'noncommitting graph warmup and one plain/graph step',
            'matched CPU/GPU bar256, then half-dt, then refined grid only after passes'],
        execution_evidence_required=['finite/nonnegative accepted states, native SSP43 and actual thread count',
            'water, each sediment label and hI/source budgets at each saved frame; no signed-source cancellation',
            'actual positive evolving filter iterations and nonzero local source/hI, not manufactured hI',
            'at least one robust wet-to-dry and dry-to-wet cell transition per backend',
            'mobile bed change and CPU/GPU original numerical gates at aligned output clocks',
            'restart preserves all physical inventories, hI and accumulated ledgers',
            'half-dt sensitivity plus area-conservative coarse/fine diagnostics; no automatic threshold relaxation'],
        scope='Short synthetic port/activity diagnostic, not a physical experiment or general wet/dry qualification',
        changes_from_old_bar=['bar height .093 -> .105m; no old outputs reused',
            'fresh analytic curved initial flow and .08m constant stage outlet, not time-series forcing',
            'D50 .5 -> .2mm; two identical labels/two layers; roughness-MPM and prognostic secondary enabled'],
        observational_limitations=['Quarter-second output proves only observed transitions, not all-trial positivity by itself',
            'Host raw-curvature diagnostic is neither filtered native/GPU parity nor measured solver activity',
            'Exact native hI zero must be checked after initialization; analysis.npz hI is the declared expectation',
            'If activity gates fail, retain this attempt and design a new input; never call an inert pass qualified'],
        not_Johannot_configuration=True, production_modified=False, Delft3D_modified=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(plan=str(output/'plan.json'), sha256=sha(output/'plan.json'), cases=len(rows),
        prepared_only=True, native_executions=0, GPU_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
