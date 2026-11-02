"""Explicit small GPU wet/dry qualification using the frozen qualified kernels.

Host-only preparation pins each fixture/export and the completed secondary-only
GPU source. Execution requires the plan SHA. No production edits or CPU fallback.
The host observations deliberately add synchronization: this is not a benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from braids_gpu.runner import sha, write_json, save_arrays

HERE = Path(__file__).resolve().parent
QUALIFIED_PLAN = HERE/'reports/inlet_secondary_gpu20h_20261002_v1/plan.json'
QUALIFIED_SHA = '1cdb23843a7a8c03db5f9d5e9451228a9a34adb7b5fed8e79427c6c84734e0bc'
FILTER = dict(rtol=1e-8, atol=1e-10, maxiter=500, graph_batch=5)
TRACE_COLUMNS = ('step', 'time_s', 'dt_s', 'status', 'CFL', 'minimum_depth_m',
    'minimum_layer_inventory_kg_m2', 'finite', 'maximum_abs_hI_m2_s', 'wet_cells',
    'initially_wet_ever_dry_cells', 'initially_dry_ever_wet_cells')


def checked_hashes(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Authenticated file changed: '+str(path))
    return True


def controls(case, fixture):
    dt, end = float(case['dt']), float(case['duration'])
    interval = .25
    if not np.isfinite([dt, end]).all() or min(dt, end) <= 0:
        raise ValueError('Finite positive fixture clocks required')
    steps, stride = round(end/dt), round(interval/dt)
    if (steps < 1 or stride < 1 or abs(steps*dt-end) > 1e-12 or
            abs(stride*dt-interval) > 1e-12 or steps % stride):
        raise ValueError('Fixture output and endpoint must be integer timestep multiples')
    if case['output_times_s'] != (np.arange(steps//stride+1)*interval).tolist():
        raise ValueError('Declared output clocks differ')
    c = fixture['controls']
    return dict(dt=dt, end=end, interval=interval, steps=steps, stride=stride,
        morfac=float(c['MorFac']), cfl=float(c['CFL']), dry=float(c['dry_m']),
        cells=int(case['cells']), scheme=int(c['scheme']), secondary_model=int(c['secondary_model']))


def snapshot_identity(data, case, expected):
    """Prove exact common inventories before applying constructor metadata only."""
    checks = {'time': 0., 'endtime': expected['end'], 'dtmax': expected['dt'],
        'output_dt': expected['interval'], 'morphological_factor': expected['morfac'],
        'cfl_target': expected['cfl'], 'dry_tolerance': expected['dry'],
        'scheme': expected['scheme'], 'secondary_flow_model': expected['secondary_model'],
        'nc': expected['cells']}
    for key, value in checks.items():
        if key not in data or float(data[key]) != value:
            raise ValueError('Native initialization control differs: '+key)
    with np.load(Path(case['input']).parent/'initial_analysis.npz', allow_pickle=False) as reference:
        arrays = dict(state9=(data['state'][:9], reference['q'].T),
            q=(data['q'], reference['q'][:, :3].T), rock=(data['rock'], reference['rock']))
        for name, (actual, wanted) in arrays.items():
            if not same_bits(actual, wanted):
                raise ValueError('Initializer changed declared physical '+name)
    indices = np.asarray(data['bed_indices'], dtype=int)
    if not same_bits(data['layer_mass'], data['state'][indices]):
        raise ValueError('Layer index/state mapping differs')
    if int(data['secondary_index']) != 9 or np.any(data['state'][9]):
        raise ValueError('Declared initially zero hI was not preserved')
    if np.any(data['state'][np.asarray(data['suspended_indices'], dtype=int)]):
        raise ValueError('Suspended inventory must remain zero')
    return dict(common_state9_bitwise=True, common_q_bitwise=True, common_rock_bitwise=True,
        layers_match_state_bitwise=True, initial_secondary_exactly_zero=True)


def same_bits(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def source_identity():
    if sha(QUALIFIED_PLAN) != QUALIFIED_SHA:
        raise ValueError('Qualified source plan changed')
    parent = json.loads(QUALIFIED_PLAN.read_text())
    folder = Path(parent['output_path'])/'source'
    manifest = json.loads((folder/'manifest.json').read_text())
    checked_hashes({str(folder/name): digest for name, digest in manifest.items()})
    expected = dict(parent['baseline_source_sha256'])
    expected.update(parent['candidate_source_sha256'])
    expected.update({name: spec['generated_file_sha256'] for name, spec in parent['repair_specs'].items()})
    for name, digest in expected.items():
        if manifest.get(name) != digest:
            raise ValueError('Qualified archived source mismatch: '+name)
    # Numerical Python is imported normally, but authenticated against the
    # qualified archive; coarse Python is loaded from its new archive directly.
    live = {}
    for name, digest in parent['baseline_source_sha256'].items():
        if name.endswith('.py'):
            path = (HERE.parents[1]/'braids_gpu'/name.removeprefix('backend_')) if name.startswith('backend_') else HERE/name
            if sha(path) != digest:
                raise ValueError('Live qualified dependency changed: '+name)
            live[str(path)] = digest
    live[str(Path(__file__).resolve())] = sha(__file__)
    return folder, manifest, live


def prepare(fixture_plan, case_name, oracle, destination, plan_path):
    from .coupled_reference import read_snapshot
    from .coupled_solver import normalized, validate_scope
    fixture_plan, oracle = Path(fixture_plan).resolve(), Path(oracle).resolve()
    destination, plan_path = Path(destination).resolve(), Path(plan_path).resolve()
    if destination.exists() or plan_path.exists():
        raise FileExistsError('Fresh execution plan and output required')
    fixture = json.loads(fixture_plan.read_text())
    rows = [r for r in fixture['cases'] if r['name'] == case_name]
    if len(rows) != 1:
        raise ValueError('Unique declared fixture case required')
    case = rows[0]; clock = controls(case, fixture)
    protected = {str(fixture_plan): sha(fixture_plan), **fixture['source_files'], **case['input_files']}
    checked_hashes(protected)
    prep = oracle/'preparation_receipt.json'
    receipt = json.loads(prep.read_text())
    if receipt.get('schema') != 1 or receipt.get('solver_steps_performed') != 0:
        raise ValueError('Native zero-step initialization receipt required')
    artifacts = receipt['artifacts']
    if 'coupled_input.bin' not in artifacts:
        raise ValueError('Missing native input identity')
    for name, digest in artifacts.items():
        if Path(name).name != name:
            raise ValueError('Initializer artifacts must be direct children')
        protected[str(oracle/name)] = digest
    protected[str(prep)] = sha(prep); checked_hashes(protected)
    data = read_snapshot(oracle); initial = snapshot_identity(data, case, clock)
    requested = {k: int(data[k]) for k in ('outlet_extension', 'cpu_outlet_extension')}
    if any(v not in (0, 1) for v in requested.values()):
        raise ValueError('Unexpected outlet metadata')
    # No step can execute between this constructor compatibility projection
    # and setting the actual qualified reconstruction extension to one.
    projected = dict(data, outlet_extension=0, cpu_outlet_extension=0)
    validate_scope(normalized(projected))
    folder, manifest, live = source_identity()
    protected[str(QUALIFIED_PLAN)] = QUALIFIED_SHA
    plan = dict(schema=1, entrypoint=Path(__file__).name, fixture_plan=str(fixture_plan), case=case,
        controls=clock, input_folder=str(oracle), output_path=str(destination),
        qualified_source_plan=str(QUALIFIED_PLAN), qualified_source_plan_sha256=QUALIFIED_SHA,
        source_folder=str(folder), source_manifest=manifest,
        source_manifest_sha256=sha(folder/'manifest.json'), authenticated_files=protected,
        live_python_sha256=live, initial_identity=initial, filter_controls=FILTER,
        requested_outlet_metadata=requested, constructor_outlet_metadata=0,
        effective_outlet_extension=1, repair_variant='secondary-only',
        reconstruction_unchanged=True, augmented_fwave_unchanged=True,
        sediment_guards=False, backend='gpu', integrator='SSPRK43',
        fixed_dt=True, rejection_policy='abort; no retry or CPU fallback',
        graph_preflight='Noncommitting plain and graph trial must be bitwise identical',
        timing_qualified=False, diagnostic_scope='Every accepted-step scalar positivity/activity; extra synchronization; not performance timing',
        production_modified=False, delft_modified=False)
    plan_path.parent.mkdir(parents=True, exist_ok=True); write_json(plan_path, plan)
    return dict(plan=str(plan_path), sha256=sha(plan_path), prepared_only=True, GPU_calls=0)


def verify_plan(path, approved):
    path = Path(path).resolve()
    if sha(path) != approved:
        raise ValueError('Reviewed exact execution plan SHA required')
    plan = json.loads(path.read_text())
    if (plan.get('schema') != 1 or plan.get('entrypoint') != Path(__file__).name or
            plan.get('repair_variant') != 'secondary-only' or plan.get('sediment_guards') is not False or
            plan.get('filter_controls') != FILTER or plan.get('effective_outlet_extension') != 1):
        raise ValueError('Explicit qualified worker selection differs')
    checked_hashes(plan['authenticated_files']); checked_hashes(plan['live_python_sha256'])
    folder = Path(plan['source_folder'])
    if sha(folder/'manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Qualified source manifest changed')
    checked_hashes({str(folder/n): d for n, d in plan['source_manifest'].items()})
    fixture = json.loads(Path(plan['fixture_plan']).read_text())
    if controls(plan['case'], fixture) != plan['controls']:
        raise ValueError('Derived controls changed')
    return plan


@contextmanager
def constructors(folder, manifest, bindings):
    """Apply only the already-qualified constructor bindings, before capture."""
    from . import coupled_solver, secondary_transport, coupled_graph
    old_secondary = coupled_solver.SecondaryTransport
    old_filter = secondary_transport.ResidentCurvature
    old_scope = coupled_graph.validate_graph_scope
    name = 'braids_support.gpu_fwave._wetdry_coarse_archive'
    if name in sys.modules:
        raise ValueError('Private archived module already registered')
    spec = importlib.util.spec_from_file_location(name, folder/'curvature_coarse_block.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    def secondary(*args, source_path=None, **kwargs):
        if Path(source_path).resolve() != folder/'secondary_transport.cu':
            raise ValueError('Unexpected secondary constructor path')
        target = folder/'secondary_transport_bounds_fixed.cu'
        # This archived kernel implements extension ONE, regardless of the
        # old constructor's hydraulic compatibility projection to zero. Bind
        # its real identity now; restore the projection until the caller
        # performs the existing explicit hydraulic promotion before stepping.
        # Never change archived bytes or weaken the exact-SHA ABI guard.
        if len(args) != 2 or set(kwargs)-{'curvature_source_path'} or args[1].outlet_extension != 0:
            raise ValueError('Unexpected archived secondary constructor contract')
        data, reconstruction = args
        projected = dict(data, outlet_extension=1, cpu_outlet_extension=1)
        reconstruction.outlet_extension = 1
        try:
            result = old_secondary(projected, reconstruction, source_path=target, **kwargs)
        finally:
            reconstruction.outlet_extension = 0
        bindings.append(dict(kind='secondary-only', executed=str(target), sha256=sha(target),
            before_graph_capture=True, secondary_outlet_extension=1,
            hydraulic_constructor_projection=0, required_hydraulic_extension_before_step=1))
        return result
    def coarse(data, source_path=None, *, parallel_coarse=True):
        if not parallel_coarse or Path(source_path).resolve() != folder/'secondary_curvature.cu':
            raise ValueError('Unexpected curvature constructor selection')
        result = module.CoarseBlockCurvature(data, source_path, parallel_coarse=True,
            max_coarse_cells=1024, block_size=256)
        bindings.append(dict(kind='coarse_block_v1', executed_python=str(folder/'curvature_coarse_block.py'),
            python_sha256=sha(folder/'curvature_coarse_block.py'),
            executed_cuda=str(folder/'curvature_coarse_block.cu'), cuda_sha256=sha(folder/'curvature_coarse_block.cu'),
            levels=[int(g['nc']) for g in result.levels], coarse_first=int(result.coarse_first),
            before_graph_capture=True))
        return result
    def scope(data, dt, extension=0):
        if extension != 1:
            raise ValueError('Qualified effective outlet extension must be one')
        old_scope(data, dt, 0)  # Only the independently qualified guard expansion.
    coupled_solver.SecondaryTransport = secondary
    secondary_transport.ResidentCurvature = coarse
    coupled_graph.validate_graph_scope = scope
    try:
        yield
    finally:
        coupled_solver.SecondaryTransport = old_secondary
        secondary_transport.ResidentCurvature = old_filter
        coupled_graph.validate_graph_scope = old_scope
        sys.modules.pop(name, None)


class Activity:
    """Receipts are assigned acceptance only after the enclosing trial returns."""
    def __init__(self, solver):
        self.solver = solver; self.phase = 'setup'; self.step = 0
        self.filters = []; self.sources = []
        self.original_filter = solver.secondary.filter.solve
        self.original_relax = solver.secondary.relax
        solver.secondary.filter.solve = self.filter
        solver.secondary.relax = self.relax
    def filter(self, *args, **kwargs):
        kwargs = dict(kwargs); kwargs.update(FILTER)
        field, receipt = self.original_filter(*args, **kwargs)
        self.filters.append(dict(phase=self.phase, step=self.step, accepted=False,
            **receipt, effective_controls=dict(FILTER)))
        return field, receipt
    def relax(self, *args, **kwargs):
        result = self.original_relax(*args, **kwargs)
        cp = self.solver.cp; b = self.solver.secondary.buffers
        self.sources.append(dict(phase=self.phase, step=self.step, accepted=False,
            absolute_local_source_m4_s=float(cp.sum(cp.abs(b['source_work'])).item()),
            signed_local_source_m4_s=float(cp.sum(b['source_work']).item()),
            maximum_abs_hI_m2_s=float(cp.max(cp.abs(args[1])).item()),
            maximum_abs_raw_curvature_per_m=float(cp.max(cp.abs(b['raw_curvature'])).item()),
            maximum_abs_filtered_curvature_per_m=float(cp.max(cp.abs(b['curvature'])).item())))
        return result
    def accept(self):
        for rows in (self.filters, self.sources):
            selected = [r for r in rows if r['phase'] == 'evolution' and r['step'] == self.step]
            if len(selected) != 2:
                raise RuntimeError('Expected exactly two source/filter calls per accepted trial')
            for row in selected:
                row['accepted'] = True
    def summary(self):
        filters = [r for r in self.filters if r['accepted']]
        sources = [r for r in self.sources if r['accepted']]
        return dict(evolving_filter_calls=len(filters),
            evolving_filter_calls_with_positive_iterations=sum(r['iterations'] > 0 for r in filters),
            accumulated_absolute_local_secondary_source_m4_s=sum(r['absolute_local_source_m4_s'] for r in sources),
            evolving_maximum_abs_hI_m2_s=max((r['maximum_abs_hI_m2_s'] for r in sources), default=0.),
            accepted_source_calls=len(sources), warmup_and_preflight_excluded=True,
            local_source_definition='sum over accepted source halves and cells of abs(area * local hI increment), not abs(net source ledger)')


def host_trial(solver, result):
    cp = solver.cp
    return {**{k: cp.asnumpy(v) for k, v in result['state'].items()},
        **{'ledger_'+k: cp.asnumpy(v) for k, v in result['ledger'].items()},
        'status': cp.asnumpy(result['status']), 'cfl': cp.asnumpy(result['cfl'])}


def accepted_unchanged(solver, before):
    cp = solver.cp
    return {key: same_bits(cp.asnumpy(value), before[key]) for owner, prefix in
        ((solver.state, ''), (solver.ledger, 'ledger_')) for name, value in owner.items()
        for key in (prefix+name,)}


def run(path, approved):
    from . import coupled_solver, coupled_graph
    from .coupled_reference import read_snapshot
    plan = verify_plan(path, approved); destination = Path(plan['output_path'])
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True); folder = (destination/'source').resolve()
    shutil.copytree(plan['source_folder'], folder)
    shutil.copy2(__file__, destination/Path(__file__).name)
    checked_hashes({str(folder/n): d for n, d in plan['source_manifest'].items()})
    data = read_snapshot(plan['input_folder']); clock = plan['controls']
    snapshot_identity(data, plan['case'], clock)
    data = dict(data, outlet_extension=0, cpu_outlet_extension=0)
    receipt = dict(status='initializing', plan=str(Path(path).resolve()), plan_sha256=approved,
        controls=clock, filter_controls=FILTER, backend='gpu', precision='FP64',
        integrator='SSPRK43', fixed_dt=True, timing_qualified=False,
        source_manifest_sha256=sha(folder/'manifest.json'), initial_identity=plan['initial_identity'],
        physical_arrays_unchanged_by_constructor_projection=True,
        constructor_outlet_metadata=0, effective_outlet_extension=1,
        positivity_scope='Each attempted result has native device stage/final checks; additional scalar observations cover each accepted state',
        restart_tested=False, production_modified=False, delft_modified=False)
    write_json(destination/'run.json', receipt)
    start = time.perf_counter(); activity = None; graph = None; solver = None
    frames = []; times = []; frame_steps = []; trace = []; bindings = []; accepted = 0; error = None
    def save(current, step):
        fields = solver.fields(); frames.append(fields); times.append(float(current)); frame_steps.append(step)
        save_arrays(destination/'checkpoint.npz', time=np.float64(current), dt_hint=np.float64(clock['dt']),
            steps=np.int64(step), input_identity=np.asarray(approved),
            source_identity=np.asarray(sha(folder/'manifest.json')), **fields)
    try:
        with constructors(folder, plan['source_manifest'], bindings):
            solver = coupled_solver.CoupledSolver(data, source_folder=folder); cp = solver.cp
            if solver.reconstruction.outlet_extension != 0:
                raise RuntimeError('Unexpected initial reconstruction metadata')
            solver.reconstruction.outlet_extension = 1
            if not all(bool(cp.all(v == 0).item()) for v in solver.ledger.values()):
                raise RuntimeError('Fresh fixture must start with zero ledgers')
            activity = Activity(solver)
            before = {**{k: cp.asnumpy(v) for k, v in solver.state.items()},
                **{'ledger_'+k: cp.asnumpy(v) for k, v in solver.ledger.items()}}
            activity.phase = 'preflight_plain'
            plain = host_trial(solver, solver.step(clock['dt'], 0., commit=False))
            if int(plain['status']):
                raise RuntimeError('Noncommitting plain trial rejected')
            activity.phase = 'graph_warmup'
            graph = coupled_graph.CoupledTransportGraph(solver)
            warm = accepted_unchanged(solver, before)
            if not all(warm.values()):
                raise RuntimeError('Noncommitting preflight/warmup mutated accepted fields')
            activity.phase = 'preflight_graph'
            replay = host_trial(solver, solver.step(clock['dt'], 0., commit=False))
            parity = {key: same_bits(plain[key], replay[key]) for key in plain}
            if not all(parity.values()) or not all(accepted_unchanged(solver, before).values()):
                raise RuntimeError('Plain/graph noncommitting one-step identity failed: '+str(parity))
            receipt['preflight'] = dict(plain_graph_bitwise=parity, accepted_state_ledger_unchanged=warm,
                warmup_committed=False, plain_status=int(plain['status']), graph_status=int(replay['status']))
            save(0., 0); activity.phase = 'evolution'
            initial_depth = solver.state['q'][0].copy()
            ever_dry = cp.zeros(solver.nc, dtype=cp.bool_); ever_wet = cp.zeros_like(ever_dry)
            for step in range(1, clock['steps']+1):
                activity.step = step; result = solver.step(clock['dt'], (step-1)*clock['dt'])
                code = int(result['status'].item()); peak = float(result['cfl'].item())
                if code:
                    receipt['rejected_trial'] = dict(step=step, status=code, cfl=peak)
                    raise RuntimeError('Fixed-step trial rejected; no adaptive retry')
                accepted = step; activity.accept()
                q, layers, secondary = (solver.state[k] for k in ('q', 'layers', 'secondary'))
                finite = all(bool(cp.isfinite(v).all().item()) for v in (*solver.state.values(), *solver.ledger.values()))
                hmin = float(q[0].min().item()); mmin = float(layers.min().item())
                ever_dry |= (initial_depth >= .002) & (q[0] <= .0005)
                ever_wet |= (initial_depth <= .0005) & (q[0] >= .002)
                trace.append((step, step*clock['dt'], clock['dt'], code, peak, hmin, mmin,
                    int(finite), float(cp.abs(secondary).max().item()), int((q[0] > clock['dry']).sum().item()),
                    int(ever_dry.sum().item()), int(ever_wet.sum().item())))
                if not finite or hmin < 0 or mmin < 0 or not np.isfinite(peak) or peak > clock['cfl']+1e-12:
                    raise RuntimeError('Per-step finite/positivity/CFL gate failed')
                if step % clock['stride'] == 0:
                    save(step*clock['dt'], step)
                    print(json.dumps(dict(step=step, time_s=step*clock['dt'], percent=100*step/clock['steps'])), flush=True)
            if graph.replays != clock['steps']+1:
                raise RuntimeError('Unexpected graph replay count including one preflight')
            if activity.summary()['evolving_filter_calls'] != 2*clock['steps']:
                raise RuntimeError('Unexpected accepted filter call count')
            receipt.update(status='completed', initially_wet_ever_dry_cells=int(ever_dry.sum().item()),
                initially_dry_ever_wet_cells=int(ever_wet.sum().item()))
    except Exception as exc:
        error = exc; receipt.update(status='failed', error=repr(exc))
    finally:
        if frames:
            stacked = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
            save_arrays(destination/'frames.npz', time=np.array(times), times=np.array(times),
                step=np.array(frame_steps, dtype=np.int64), bed=stacked['bed_elevation'], **stacked)
        save_arrays(destination/'step_trace.npz', trace=np.asarray(trace, dtype=float).reshape(-1, len(TRACE_COLUMNS)),
            columns=np.asarray(TRACE_COLUMNS))
        write_json(destination/'filter_history.json', activity.filters if activity else [])
        write_json(destination/'source_history.json', activity.sources if activity else [])
        receipt.update(accepted_steps=accepted, final_time_s=accepted*clock['dt'],
            bindings=bindings, graph=graph.receipt() if graph else None,
            activity=activity.summary() if activity else {}, wall_s=time.perf_counter()-start,
            artifacts={p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != 'run.json'})
        try:
            verify_plan(path, approved)
            checked_hashes({str(folder/n): d for n, d in plan['source_manifest'].items()})
            receipt['all_authenticated_identities_retained'] = True
        except Exception as exc:
            receipt.update(status='failed', identity_error=repr(exc), all_authenticated_identities_retained=False)
            if error is None: error = exc
        write_json(destination/'run.json', receipt)
    if error is not None:
        raise error
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--fixture-plan', type=Path, required=True)
    s.add_argument('--case', required=True); s.add_argument('--oracle', type=Path, required=True)
    s.add_argument('--destination', type=Path, required=True); s.add_argument('--plan', type=Path, required=True)
    s = sub.add_parser('run'); s.add_argument('--plan', type=Path, required=True)
    s.add_argument('--approved-plan-sha256', required=True)
    a = p.parse_args(); result = (prepare(a.fixture_plan, a.case, a.oracle, a.destination, a.plan)
        if a.action == 'prepare' else run(a.plan, a.approved_plan_sha256))
    print(json.dumps(result, indent=2, allow_nan=False))
