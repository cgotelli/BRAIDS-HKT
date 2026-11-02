"""Isolated hydraulic dry-cutoff sensitivity; V2 physical state is unchanged.

V2's draining cells retained a film just below its1mm hydraulic cutoff and did
not cross the frozen0.5mm activity threshold. V3 lowers ONLY the hydraulic dry
cutoff to0.25mm. Initial velocities and sediment activation still use1mm. This
is an input sensitivity, not a solver bug fix, and V1/V2 failures are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from . import curved_wetdry_readiness_v2 as parent

old = parent.old
TAG = 'curved_wetdry_readiness_20261002_v3'
PARENT = old.HERE/'runs/curved_wetdry_readiness_20261002_v2/plan.json'
PARENT_SHA = '48e90d8f021b5632ef171dea8b9719dbcc17ca3ce349add1f3a1238a35b6ef08'
HYDRAULIC_DRY = .00025
SEDIMENT_MIN_DEPTH = .001


def specifications():
    return parent.specifications()


def initial_fields(xy):
    # Retain all4 exact V2 arrays, including the source shoal diagnostic.
    return parent.initial_fields(xy)


def authenticate():
    if old.sha(PARENT) != PARENT_SHA:
        raise ValueError('Frozen V2 fixture plan changed')
    value = json.loads(PARENT.read_text())
    for path, digest in value['source_files'].items():
        if old.sha(path) != digest:
            raise ValueError('Frozen V1/V2 fixture helper changed: '+path)
    for row in value['cases']:
        for path, digest in row['input_files'].items():
            if old.sha(path) != digest:
                raise ValueError('Frozen V2 input changed: '+path)
    if old.sha(old.CPU) != old.CPU_SHA:
        raise ValueError('Frozen intended CPU runtime changed')
    return value


@contextmanager
def writer_context():
    """Change one public writer argument; restore even on preparation failure."""
    from braids_core.boundaries import cut_cell_io
    original = cut_cell_io.write_cut_case

    def hydraulic_cutoff_only(*args, **kwargs):
        if kwargs.get('dry_tolerance_m') != SEDIMENT_MIN_DEPTH:
            raise ValueError('Unexpected parent hydraulic cutoff')
        fractions = kwargs.get('sediment_fractions', ())
        if len(fractions) != 2 or any(f.minimum_bedload_depth_m != SEDIMENT_MIN_DEPTH
            or f.minimum_exchange_depth_m != SEDIMENT_MIN_DEPTH for f in fractions):
            raise ValueError('Sediment1mm activation must remain unchanged')
        kwargs = dict(kwargs, dry_tolerance_m=HYDRAULIC_DRY,
            source_sha256={**kwargs['source_sha256'], 'fixture_v3':old.sha(__file__)})
        return original(*args, **kwargs)

    cut_cell_io.write_cut_case = hydraulic_cutoff_only
    try:
        yield
    finally:
        cut_cell_io.write_cut_case = original


def prepare_case(row, folder, original_row):
    if row not in specifications():
        raise ValueError('Only retained V2 physical controls allowed')
    with writer_context():
        result = parent.prepare_case(row, folder)
    folder = Path(folder); original_folder = Path(original_row['input']).parent
    # Binary geometry/state/face inputs contain no changed dry cutoff. Demand
    # the WHOLE payload remains identical, not just selected physical arrays.
    if (folder/'case.data.bin').read_bytes() != (original_folder/'case.data.bin').read_bytes():
        raise ValueError('V3 changed physical binary payload')
    with np.load(folder/'initial_analysis.npz') as new, np.load(original_folder/'initial_analysis.npz') as before:
        if set(new.files) != set(before.files) or any(not np.array_equal(new[k], before[k]) for k in new.files):
            raise ValueError('V3 analytic initial-state arrays changed')
    prior_lines = (original_folder/'case.data').read_text().splitlines()
    lines = (folder/'case.data').read_text().splitlines()
    changes = [i for i,(a,b) in enumerate(zip(prior_lines, lines)) if a!=b]
    if len(lines)!=len(prior_lines) or changes != [1,7]:
        raise ValueError('More than fingerprint and hydraulic dry-cutoff row changed')
    a,b = prior_lines[7].split(),lines[7].split()
    if float(a[1])!=SEDIMENT_MIN_DEPTH or float(b[1])!=HYDRAULIC_DRY or a[:1]+a[2:] != b[:1]+b[2:]:
        raise ValueError('Unauthorized scalar control changed')
    result.update(fixture_revision=3, hydraulic_dry_tolerance_m=HYDRAULIC_DRY,
        sediment_minimum_depth_m=SEDIMENT_MIN_DEPTH, initial_velocity_cutoff_m=SEDIMENT_MIN_DEPTH,
        original_V2_input=original_row['input'], physical_binary_identical_to_V2=True,
        all_initial_analysis_arrays_identical_to_V2=True, changed_native_text_lines_one_based=[2,8],
        source_diagnostic_note='Retained initial raw-curvature diagnostic uses original1mm mask; evolving solver uses.25mm hydraulic mask',
        actual_hydraulic_mask_not_yet_observed=True)
    return result


def prepare(output=None):
    source = authenticate()
    output = Path(output) if output is not None else old.HERE/'runs'/TAG
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = [prepare_case(spec, output/spec['name'], row)
             for spec,row in zip(specifications(), source['cases'])]
    plan = dict(source, tag=TAG, fixture_revision=3, cases=cases,
        parent_plan=str(PARENT), parent_plan_sha256=PARENT_SHA,
        source_files={**source['source_files'], str(Path(__file__).resolve()):old.sha(__file__)},
        controls={**source['controls'], 'dry_m':HYDRAULIC_DRY,
                  'sediment_minimum_depth_m':SEDIMENT_MIN_DEPTH, 'initial_velocity_cutoff_m':SEDIMENT_MIN_DEPTH},
        original_V2_files={p:h for row in source['cases'] for p,h in row['input_files'].items()},
        parent_failed_gate='V2 final draining film.000984854m below1mm hydraulic cutoff but above frozen.0005m crossing gate',
        explicit_changes=['Hydraulic dry tolerance.001m ->.00025m only; source identity/fingerprint updated truthfully',
            'Initial q/bed/rock/geometry/materials,1mm bedload/exchange activation,timestep,CFL,MorFac,boundaries and duration unchanged'],
        hydraulic_cutoff_sensitivity_not_solver_fix=True, original_gate_values_unchanged=True,
        physical_binary_identical_to_V2=True, observed_transitions_still_required=True,
        not_a_relabelled_V1_or_V2_pass=True, solver_executions=0, GPU_calls=0, prepared_only=True,
        analysis_notes=['Inherited dry_cells/wet-mask-disagreement plots using fixture.DRY are1mm display metrics, NOT the new.25mm solver mask.',
            'Robust wet/dry activity thresholds remain2mm/.5mm exactly; native/GPU t0 initialization must still be verified.'])
    if plan['numerical_execution_gates'] != old.GATES:
        raise ValueError('Frozen activity gates changed')
    (output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    authenticate()
    return dict(plan=str(output/'plan.json'), sha256=old.sha(output/'plan.json'), cases=len(cases),
        hydraulic_dry_tolerance_m=HYDRAULIC_DRY, sediment_minimum_depth_m=SEDIMENT_MIN_DEPTH,
        physical_binary_identical_to_V2=True, prepared_only=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path)
    print(json.dumps(prepare(p.parse_args().output),indent=2))
