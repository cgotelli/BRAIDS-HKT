"""Independent 67-leaf selected-mesh GPU numerical smoke, not full AMR.

This proves the new portable prepared-input loader can drive coupled GPU
f-wave/MPM/suspension/hI kernels on a topology absent from the B archive.  It
does NOT assert mixed-level subcycling, reflux, adaptation, or native-AMR
equivalence.  Those require separate general AMR qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from braids_gpu.amr_prepared import load_amr_prepared
from braids_gpu.amr_catalog_transition import coarsen_selected_state
from . import coupled_solver


SOURCE = Path(__file__).resolve().parent/'runs'/\
    'amr_composed_native_20261004_v2'/'gpu_composed_source_v1'
DEFAULT = Path(__file__).resolve().parent/'runs'/\
    'amr_general_prepared_20261004_v4'/'amr_prepared'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _totals(solver):
    cp = solver.cp
    area = cp.asarray(solver.data['area'])
    state = solver.state
    return (float(cp.sum(area*state['q'][0]).item()),
            cp.asnumpy(cp.sum(area[None, :]*(state['suspended']+
                cp.sum(state['layers'], axis=0)), axis=1)),
            float(cp.sum(area*state['secondary']).item()))


def _full(solver):
    cp = solver.cp
    data = solver.data
    state = solver.state
    full = cp.empty((solver.nc, int(data['nstate'])), dtype=cp.float64)
    full[:, :3] = state['q'].T
    for grain, row in enumerate(np.asarray(data['suspended_indices'])):
        full[:, int(row)] = state['suspended'][grain]
    for layer, rows in enumerate(np.asarray(data['bed_indices'])):
        for grain, row in enumerate(rows):
            full[:, int(row)] = state['layers'][layer, grain]
    full[:, int(data['secondary_index'])] = state['secondary']
    return cp.ascontiguousarray(full)


def _install(solver, full):
    cp = solver.cp
    data = solver.data
    cp.copyto(solver.state['q'], cp.ascontiguousarray(full[:, :3].T))
    for grain, row in enumerate(np.asarray(data['suspended_indices'])):
        cp.copyto(solver.state['suspended'][grain], full[:, int(row)])
    for layer, rows in enumerate(np.asarray(data['bed_indices'])):
        for grain, row in enumerate(rows):
            cp.copyto(solver.state['layers'][layer, grain], full[:, int(row)])
    cp.copyto(solver.state['secondary'], full[:, int(data['secondary_index'])])
    if not bool(cp.array_equal(full, _full(solver))):
        raise RuntimeError('AMR coarsened state did not install exactly')


def run(package=DEFAULT, output=None):
    prepared = load_amr_prepared(package)
    first = prepared.topologies[0]
    if len(first.node_ids) != 67:
        raise ValueError('Independent 67-leaf topology required for this smoke')
    data = first.data
    solver = coupled_solver.CoupledSolver(data, source_folder=SOURCE,
                                          integrator='ssp33')
    if (solver.nb or not solver.bedload_active or not solver.has_suspended or
            solver.secondary.model != 2 or solver.ng != 2):
        raise ValueError('Active coupled GPU physics was not constructed')
    before = _totals(solver)
    initial = {key: solver.cp.asnumpy(value).copy()
               for key, value in solver.state.items()}
    rows = []
    source_before = 0.
    transition = None
    pre_change = None
    post_initial = None
    for outer in range(4):
        if outer == 2:
            pre_change = {key: float(np.max(np.abs(solver.cp.asnumpy(value)-initial[key])))
                          for key, value in solver.state.items()}
            coarsened, transition = coarsen_selected_state(prepared,
                prepared.topologies[0], prepared.topologies[1], _full(solver),
                transfer_source=SOURCE/'amr_transfer.cu')
            solver = coupled_solver.CoupledSolver(prepared.topologies[1].data,
                source_folder=SOURCE, integrator='ssp33')
            _install(solver, coarsened)
            post_initial = {key: solver.cp.asnumpy(value).copy()
                            for key, value in solver.state.items()}
            transferred = _totals(solver)
            if (abs(transferred[0]-before[0]) > 1e-10 or
                    max(np.abs(transferred[1]-before[1])) > 1e-8 or
                    abs(transferred[2]-before[2]) > 1e-8):
                raise RuntimeError('AMR post-install budget changed')
            source_before = 0.
        result = solver.step(.002, .002*outer)
        if int(result['status'].item()) != 0:
            raise RuntimeError('Independent selected-leaf GPU step rejected')
        after = _totals(solver)
        source = float(solver.ledger['source'].item())
        rows.append(dict(time_s=.002*(outer+1),
            leaves=solver.nc,
            cfl=float(result['cfl'].item()),
            water_residual_m3=after[0]-before[0],
            sediment_class_residual_kg=(after[1]-before[1]).tolist(),
            secondary_source_residual=after[2]-before[2]-(source-source_before),
            depth_min_m=float(solver.cp.min(solver.state['q'][0]).item()),
            bed_flux_peak_kg_m_s=solver.cp.asnumpy(solver.cp.max(
                solver.cp.abs(solver.exner.buffers['flux']), axis=1)).tolist(),
            suspended_flux_peak_kg_m_s=solver.cp.asnumpy(solver.cp.max(
                solver.cp.abs(solver.suspended_advection.buffers['flux']),
                axis=1)).tolist(),
            secondary_face_flux_peak=float(solver.cp.max(solver.cp.abs(
                solver.secondary.buffers['flux'])).item()),
            suspended_exchange_kg=solver.cp.asnumpy(
                solver.ledger['exchange']).tolist()))
        before = after
        source_before = source
    state_change = dict(pre_coarsen=pre_change,
        post_coarsen={key: float(np.max(np.abs(solver.cp.asnumpy(value)-post_initial[key])))
                      for key, value in solver.state.items()})
    failures = []
    for row in rows:
        if (not np.isfinite(row['cfl']) or row['cfl'] > .4+1e-12 or
                abs(row['water_residual_m3']) > 1e-9 or
                max(np.abs(row['sediment_class_residual_kg'])) > 1e-8 or
                abs(row['secondary_source_residual']) > 1e-8 or
                row['depth_min_m'] < -1e-12):
            failures.append('Independent accepted GPU step failed conservation/CFL')
        if (min(row['bed_flux_peak_kg_m_s']) <= 1e-8 or
                min(row['suspended_flux_peak_kg_m_s']) <= 1e-8 or
                row['secondary_face_flux_peak'] <= 1e-12 or
                not np.any(row['suspended_exchange_kg'])):
            failures.append('One or more coupled CUDA physics paths were inactive')
    record = dict(package_identity=prepared.identity,
        topology=first.name, cells=len(first.node_ids),
        transition=transition,
        source_sha256={name: _sha(SOURCE/name) for name in
            ('augmented_fwave.cu','amr_reflux.cu','amr_transfer.cu')},
        steps=rows, max_state_changes=state_change,
        passed=not failures, failures=failures,
        scope='Two GPU steps on new 67-leaf topology, conservative CUDA 67-to-64 restriction, two post-coarsening GPU steps; no mixed-level subcycling/reflux or native-AMR equivalence',
        no_CPU_numerical_fallback=True, Delft3D_modified=False)
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError('Independent GPU smoke receipt is write-once')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8')
    if failures:
        raise RuntimeError('; '.join(failures))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.package, args.output), indent=2))
