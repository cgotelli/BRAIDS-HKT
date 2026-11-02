"""Private composed AMR Gate B2: ten-row 91-to-64 CUDA restriction at .06 s.

This adds one native-scheduled coarsening transaction to the passing B1
wet/dry and mixed-level path. It ends immediately after topology commit; it
does not claim general AMR or a post-regrid GPU transport trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from braids_qgis.core.results import read_frames

from . import amr_composed_b1_gpu_v2 as b1
from . import amr_composed_b2_native as native
from . import amr_composed_b2_export as exported
from . import amr_composed_one_step_gpu as gate_a
from . import amr_composed_native_fixture_v2 as case
from . import amr_hydro_native_fixture as base
from . import coupled_reference
from . import coupled_solver
from .amr_transfer import ResidentAMRTransfer
from .amr_hydro_route import requires_synchronous_wet_front


ROOT = case.ROOT / 'gpu_composed_b2_candidate_v1'


class ComposedB2(b1.ComposedB1):
    def __init__(self):
        super().__init__()
        self.tree, _, _, _ = case.geometry()
        self.mesh_index = 0
        self.regrid_evaluations = 0
        self.regrid_events = []
        self.euler_calls_completed = 0

    def _restricted_mesh(self):
        cp = self.cp
        wanted = np.flatnonzero(self.tree.node_level == 1).astype(np.int64)
        native_ids = np.asarray(read_frames(native.ROOT/'cpu/frames.txt')[3]['node_ids'])
        if len(wanted) != 64 or not np.array_equal(wanted, native_ids):
            raise ValueError('Expected 64 native L1 selected leaves')
        old_map = {int(node): i for i, node in enumerate(self.ids)}
        new_map = {int(node): i for i, node in enumerate(wanted)}
        old = self._full(self.state)
        result = cp.full((64, 10), cp.nan, dtype=cp.float64)
        assigned = np.zeros(64, dtype=bool)
        for node, position in new_map.items():
            if node in old_map:
                result[position] = old[old_map[node]]
                assigned[position] = True
        parents = sorted(node for node in wanted if node not in old_map)
        children = [self.tree.children[parent] for parent in parents]
        if (len(parents) != 9 or any(len(group) != 4 or
                any(child not in old_map for child in group)
                for group in children)):
            raise ValueError('Expected nine complete four-child groups')
        cells = self.tree.cells
        weights = np.asarray([[cells[child].area_m2 for child in group]
                              for group in children], dtype=np.float64)
        parent_areas = np.asarray([cells[parent].area_m2 for parent in parents])
        positions = np.asarray([[np.subtract(cells[child].centroid_xy,
            cells[parent].centroid_xy) for child in group]
            for parent, group in zip(parents, children)])
        # Constructor values merely allocate a valid 10-row CUDA buffer;
        # all accepted child rows are replaced by current GPU state below.
        template = np.zeros((9, 4, 10), dtype=np.float64)
        template[:, :, 0] = .1
        template[:, :, 5:9] = .025*base.BULK
        transfer = ResidentAMRTransfer(np.ascontiguousarray(template),
            weights, parent_areas, np.full((9, 4), -.1), positions,
            np.asarray([base.BULK, base.BULK]), dry=.0001,
            prognostic_secondary=True,
            source_path=gate_a.SOURCE/'amr_transfer.cu')
        child_index = cp.asarray(np.asarray([[old_map[child] for child in group]
                                             for group in children]))
        cp.copyto(transfer.accepted, cp.ascontiguousarray(old[child_index]))
        reduced = transfer.restrict()
        for group, parent in enumerate(parents):
            result[new_map[parent]] = reduced[group]
            assigned[new_map[parent]] = True
        if not assigned.all() or not bool(cp.all(cp.isfinite(result))):
            raise RuntimeError('GPU restriction left a selected leaf unassigned')
        before_water, before_class, before_secondary = self._budgets(self.state)
        new_area = np.asarray([cells[int(node)].area_m2 for node in wanted])
        area = cp.asarray(new_area)
        after_water = float(cp.sum(area*result[:, 0]).item())
        after_class = cp.asnumpy(cp.sum(area[:, None]*
            (result[:, 3:5]+result[:, 5:7]+result[:, 7:9]), axis=0))
        after_secondary = float(cp.sum(area*result[:, 9]).item())
        water_residual = after_water-before_water
        class_residual = after_class-before_class
        secondary_residual = after_secondary-before_secondary
        if (abs(water_residual) > 1e-10 or
                max(abs(class_residual)) > 1e-8 or
                abs(secondary_residual) > 1e-8 or
                bool(cp.any(result[:, 0] < -1e-12)) or
                bool(cp.any(result[:, 3:9] < -1e-12))):
            raise RuntimeError('Ten-row GPU coarsening conservation rejected')
        event = dict(time_s=.06, old_leaves=91, new_leaves=64,
            restricted_parents=[int(node) for node in parents],
            transfer_water_residual_m3=water_residual,
            transfer_total_class_residual_kg=class_residual.tolist(),
            transfer_secondary_residual=secondary_residual,
            all_ten_rows_restricted_on_CUDA=True)
        return result, wanted, new_area, event

    def _post_solver(self, full):
        cp = self.cp
        data = coupled_reference.read_snapshot(exported.ROOT)
        amr = coupled_reference.read_snapshot(native.ROOT/'amr_final')
        for key in ('xy', 'area', 'owner', 'neighbour', 'length', 'normal',
                    'distance', 'mid', 'hydraulic_ls_weight', 'manning_cell'):
            if not np.array_equal(data[key], amr[key]):
                raise ValueError('Post-regrid fixed/native geometry changed: '+key)
        data['weno_cells'] = np.ascontiguousarray(amr['weno_cells'].copy())
        data['cpu_time_integrator'] = 33
        # Replace the native-exported future state with an independent
        # non-solution placeholder *before* constructing any CUDA operator.
        placeholder = np.zeros((10, 64), dtype=np.float64)
        placeholder[0] = .1
        placeholder[5:9] = .025*base.BULK
        data['state'] = placeholder
        solver = coupled_solver.CoupledSolver(data,
            source_folder=gate_a.SOURCE, integrator='ssp33')
        if (solver.nb or solver.ng != 2 or not solver.has_suspended or
                solver.secondary.model != 2):
            raise ValueError('Post-regrid composed GPU physics changed')
        proposal = self._unpack(full)
        for key in b1.KEYS:
            cp.copyto(solver.state[key], proposal[key])
        if not bool(cp.array_equal(full, self._full_for_n(solver.state, 64))):
            raise RuntimeError('Post-regrid ten-row state copy changed')
        return solver, data

    def _full_for_n(self, state, n):
        cp = self.cp
        full = cp.empty((n, 10), dtype=cp.float64)
        full[:, :3] = state['q'].T
        full[:, 3:5] = state['suspended'].T
        full[:, 5:7] = state['layers'][0].T
        full[:, 7:9] = state['layers'][1].T
        full[:, 9] = state['secondary']
        return cp.ascontiguousarray(full)

    def regrid(self, time, *, inject_failure=False):
        if (abs(time-.06) > 1e-12 or self.mesh_index != 0 or
                self.regrid_evaluations != 2 or self.accepted_steps != 30 or
                requires_synchronous_wet_front(self.state['q'], self.owner,
                                                self.neighbour)):
            raise ValueError('Only third fully wet 0.06 s regrid supported')
        full, ids, area, event = self._restricted_mesh()
        if inject_failure:
            raise RuntimeError('injected post-restriction pre-commit regrid rejection')
        # All GPU transfer, geometry checks and new CUDA allocations succeed
        # before any accepted topology/state pointer is replaced.
        solver, data = self._post_solver(full)
        self.euler_calls_completed += self.solver.euler_calls
        self.solver = solver
        self.state = solver.state
        self.ids = ids
        self.area = area
        self.area_device = self.cp.asarray(area)
        self.n = len(ids)
        self.levels = np.ones(len(ids), dtype=np.int64)
        self.index = {1:self.cp.arange(len(ids)),
                      2:self.cp.empty(0, dtype=self.cp.int64)}
        self.owner = self.cp.asarray(data['owner'])
        self.neighbour = self.cp.asarray(data['neighbour'])
        self.length = self.cp.asarray(data['length'])
        self.interface = self.cp.empty(0, dtype=self.cp.int64)
        self.reflux = None
        self.mesh_index = 3
        self.regrid_evaluations += 1
        self.regrid_events.append(event)
        return event

    def regrid_evaluation(self, time, *, inject_failure=False):
        if time not in (.02, .04, .06):
            raise ValueError('Only native B2 output clocks supported')
        if time == .06:
            return self.regrid(time, inject_failure=inject_failure)
        if self.regrid_evaluations != (0 if time == .02 else 1):
            raise ValueError('Native regrid evaluation order changed')
        self.regrid_evaluations += 1
        return None

    def persistent(self, time):
        cp = self.cp
        if self.mesh_index == 0:
            result = super().persistent(time)
        else:
            empty = np.empty((0,), dtype=np.float64)
            result = dict(state=cp.asnumpy(self._full_for_n(self.state, 64)),
                reflux_accepted=empty, reflux_register=empty,
                reflux_pending=np.bool_(False),
                accepted_steps=np.int64(self.accepted_steps),
                sync_steps=np.int64(self.sync_steps),
                mixed_steps=np.int64(self.mixed_steps),
                level_steps=np.asarray(self.level_steps, dtype=np.int64),
                euler_calls=np.int64(self.solver.euler_calls),
                maximum_cfl=np.float64(self.maximum_cfl),
                source_integral=np.float64(self.source_integral),
                exchange_ledger=self.exchange_ledger.copy(),
                secondary_peaks_json=np.asarray(json.dumps(
                    self.secondary_peaks, sort_keys=True)),
                time_s=np.float64(time), node_ids=self.ids.copy())
            result.update({f'peak_{name}':cp.asnumpy(value)
                           for name, value in self.peaks.items()})
        result.update(mesh_index=np.int64(self.mesh_index),
            regrid_evaluations=np.int64(self.regrid_evaluations),
            regrid_events_json=np.asarray(json.dumps(self.regrid_events,
                                                      sort_keys=True)),
            euler_calls_completed=np.int64(self.euler_calls_completed))
        return result

    def restore(self, archive):
        if int(archive['mesh_index']) != 0:
            raise ValueError('B2 restart is qualified only before coarsening')
        super().restore(archive)
        self.regrid_evaluations = int(archive['regrid_evaluations'])
        self.regrid_events = json.loads(str(archive['regrid_events_json']))
        self.euler_calls_completed = int(archive['euler_calls_completed'])


def compare(model):
    if model.mesh_index != 3 or len(model.ids) != 64:
        raise ValueError('Expected post-coarsening GPU topology')
    if not np.array_equal(model.ids, read_frames(
            native.ROOT/'cpu/frames.txt')[3]['node_ids']):
        raise ValueError('Native/GPU post-regrid leaf IDs differ')
    reference = coupled_reference.read_snapshot(native.ROOT/'amr_final')['state']
    actual = model.cp.asnumpy(model._full_for_n(model.state, 64)).T
    data = model.solver.data
    bulk = np.asarray(data['bulk_fraction'])
    rock = np.asarray(data['rock'])
    bed = lambda a: rock+np.sum(a[5:7]/bulk[:, None]+a[7:9]/bulk[:, None], axis=0)
    return dict(depth_max_abs_m=float(np.max(np.abs(actual[0]-reference[0]))),
        bed_max_abs_m=float(np.max(np.abs(bed(actual)-bed(reference)))),
        suspended_class_max_abs_kg_m2=np.max(np.abs(actual[3:5]-reference[3:5]),
                                            axis=1).tolist(),
        total_bed_class_max_abs_kg_m2=np.max(np.abs(
            actual[5:7]+actual[7:9]-reference[5:7]-reference[7:9]),
            axis=1).tolist(),
        secondary_hI_max_abs=float(np.max(np.abs(actual[9]-reference[9]))),
        dry_mask_equal=bool(np.array_equal(actual[0] <= .0001,
                                          reference[0] <= .0001)))


def run(output=ROOT):
    from . import amr_composed_b2_provenance as provenance
    lock = provenance.preflight()
    output = Path(output)
    if output.exists():
        raise FileExistsError('B2 readback is write-once')
    output.mkdir(parents=True)
    model = ComposedB2()
    rows = []
    error = None
    try:
        for outer in range(30):
            rows.append(model.step(.002, outer*.002))
            if (outer+1) % 10 == 0:
                model.regrid_evaluation(round((outer+1)*.002, 12))
    except Exception as exc:
        error = str(exc)
    metrics = compare(model) if error is None else None
    np.savez_compressed(output/'trajectory.npz', **model.persistent(
        model.accepted_steps*.002))
    failures = [error] if error else []
    if metrics is not None and (metrics['depth_max_abs_m'] > 2e-5 or
            metrics['bed_max_abs_m'] > 2e-5 or
            max(metrics['suspended_class_max_abs_kg_m2']) > 1e-5 or
            max(metrics['total_bed_class_max_abs_kg_m2']) > .02 or
            metrics['secondary_hI_max_abs'] > 1e-7 or
            not metrics['dry_mask_equal']):
        failures.append('Native/GPU B2 post-regrid difference exceeds frozen limits')
    event = model.regrid_events[0] if model.regrid_events else {}
    if (model.accepted_steps != 30 or model.sync_steps != 1 or
            model.mixed_steps != 29 or model.level_steps != [0, 58, 116] or
            model.mesh_index != 3 or len(model.regrid_events) != 1 or
            abs(event.get('transfer_water_residual_m3', 1)) > 1e-10 or
            max(np.abs(event.get('transfer_total_class_residual_kg', [1]))) > 1e-8 or
            abs(event.get('transfer_secondary_residual', 1)) > 1e-8):
        failures.append('Native B2 route/topology or transfer conservation failed')
    if rows and (max(abs(row['water_residual_m3']) for row in rows) > 1e-9 or
            max(max(abs(x) for x in row['class_residual_kg']) for row in rows) > 1e-8 or
            max(abs(row['secondary_source_residual']) for row in rows) > 1e-8 or
            min(row['minimum_depth_m'] for row in rows) < -1e-12):
        failures.append('Accepted outer conservation/source/positivity failed')
    peaks = {key:model.cp.asnumpy(value).tolist()
             for key, value in model.peaks.items()}
    secondary_peaks = model.secondary_peaks.copy()
    if (any(any(x <= threshold for x in peaks[name]) for name, threshold in
            (('bed_flux', 1e-8), ('suspended_flux', 1e-8),
             ('coarse_bed', 0.), ('fine_bed', 0.),
             ('bed_reflux', 1e-12), ('coarse_suspended', 0.),
             ('fine_suspended', 0.), ('suspended_reflux', 1e-12))) or
            any(secondary_peaks[name] <= threshold for name, threshold in
                (('face_flux', 0.), ('coarse_integral', 0.),
                 ('fine_integral', 0.), ('reflux_delta', 1e-12)))):
        failures.append('Classwise or signed-hI interface activity missing')
    record = dict(scope='Private composed ten-row GPU AMR through 0.06 s coarsening',
        rows=rows, accepted_steps=model.accepted_steps,
        level_steps=model.level_steps, mesh_index=model.mesh_index,
        regrid_events=model.regrid_events, metrics=metrics,
        peak_flux_and_integral=peaks,
        peak_secondary=secondary_peaks,
        source_integral=model.source_integral,
        exchange_ledger=model.exchange_ledger.tolist(),
        gate_B2_numerical_passed=not failures, failures=failures,
        source_preflight=lock,
        trajectory_sha256=base.sha(output/'trajectory.npz'),
        no_CPU_numerical_fallback=True, no_general_AMR_claim=True,
        Delft3D_modified=False, plugin_modified=False)
    base.save(output/'READBACK.json', record)
    if failures:
        raise RuntimeError('B2 numerical gate failed: '+'; '.join(failures))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
