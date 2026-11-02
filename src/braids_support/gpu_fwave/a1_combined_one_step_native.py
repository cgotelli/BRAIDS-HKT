"""Write-once unchanged-native 0.002 s comparator for the A1 combined GPU gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from braids_core.boundaries import cut_cell_io as writer
from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as full


HERE = Path(__file__).resolve().parent
TAG = 'a1_combined_live_scheduled_one_step_native_20261004_v1'
ROOT = HERE/'runs'/TAG
GATE = HERE/'reports/a1_combined_live_scheduled_gpu_candidate_20261004_v1/GPU_ONE_STEP_GATE.txt'
GATE_SHA = '1704fbcc1968bb1ee91dfe286b294b64d34b93428f95be848d7564523ed47b93'
SOURCE_LOCK = GATE.with_name('CURRENT_SOURCE_LOCK.txt')
END_S = .002


def require(test, message):
    if not test:
        raise ValueError(message)


def source_identity():
    files = {**full.source_files(), str(Path(__file__).resolve()): full.sha(__file__),
             str(GATE.resolve()): GATE_SHA, str(SOURCE_LOCK.resolve()): full.sha(SOURCE_LOCK)}
    for raw in SOURCE_LOCK.read_text(encoding='utf-8').splitlines():
        if not raw.startswith('braids_support/gpu_fwave/'):
            continue
        name, digest = raw.split()
        path = Path(name).resolve()
        require(full.sha(path) == digest, 'Current-source anchor changed: '+name)
        files[str(path)] = digest
    full.verify_files(files)
    return files


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh one-step native output')
    require(full.sha(GATE) == GATE_SHA, 'One-step GPU gate changed')
    build, native_files, archives = full.verify_exporter()
    source = {**source_identity(), **native_files,
              str(full.EXPORT_RUNTIME/'build.json'): full.sha(full.EXPORT_RUNTIME/'build.json'),
              str(full.EXPORT_RUNTIME/'xbraids_normal_export.exe'): build['executable_sha256'],
              **archives}
    full.verify_files(source)
    output.mkdir(parents=True)
    (output/'inputs').mkdir()
    rows = []
    for variant in full.VARIANTS:
        data = full.fixture(variant)
        folder = output/'inputs'/variant
        folder.mkdir()
        writer.write_cut_case(folder/'case.data', data['mesh'], data['q'], data['rock'],
            data['boundaries'], sections={'in': data['section']},
            end_time_s=END_S, dt_max_s=END_S, output_interval_s=END_S,
            cfl=.4, scheme=6, bedload=2, exchange=0, manning=.03,
            mpm_friction_mode=1, mpm_critical_shields=.047,
            sediment_fractions=data['fractions'], active_layer_thickness_m=.05,
            substrate_layer_count=1, substrate_layer_thickness_m=.05,
            secondary_flow=False, morphological_factor=1., compact_sidecars=True,
            source_sha256={'one_step_fixture':full.sha(__file__),
                           'original_full_fixture':full.sha(full.__file__),
                           'one_step_gate':GATE_SHA})
        data['mesh'].write_geometry(folder/'geometry.npz')
        np.savez_compressed(folder/'initial_analysis.npz', q=data['q'], bed=data['bed'],
            rock=data['rock'], xy=data['xy'],
            area=np.array([cell.area_m2 for cell in data['mesh'].cells]),
            q_knots_s=full.KNOTS_S, q_m3_s=full.Q_M3_S,
            feed_kg_s=full.FEED_KG_S, grain_d50_m=full.D50_M)
        rows.append(dict(variant=variant, deck=str(folder/'case.data'),
            files={str(path): full.sha(path) for path in folder.iterdir() if path.is_file()}))
    plan = dict(schema=1, tag=TAG, output=str(output), cases=rows,
        source_files=source, one_step_gate_sha256=GATE_SHA,
        source_lock_sha256=full.sha(SOURCE_LOCK),
        full_native_plan_sha256=full.sha(full.ROOT/'plan.json'),
        native_executable=str(full.frozen_cpu.TIGHT_EXE),
        native_executable_sha256=full.frozen_cpu.TIGHT_SHA,
        exporter_executable=str(full.EXPORT_RUNTIME/'xbraids_normal_export.exe'),
        exporter_executable_sha256=build['executable_sha256'],
        exporter_parent_sha256=build['parent_executable_sha256'],
        controls=dict(end_s=END_S, dt_max_s=END_S, output_s=END_S,
            cfl=.4, morfac=1., q_knots_s=full.KNOTS_S,
            q_m3_s=full.Q_M3_S, bed_feed_kg_s=full.FEED_KG_S,
            grain_d50_m=full.D50_M, initial_stage_m=full.VARIANTS),
        GPU_solver_executed=False, Delft3D_source_modified=False)
    full.save(output/'plan.json', plan)
    return dict(plan=str(output/'plan.json'), sha256=full.sha(output/'plan.json'))


def load_plan(output=ROOT):
    output = Path(output).resolve()
    require(full.sha(GATE) == GATE_SHA, 'One-step GPU gate changed')
    plan = json.loads((output/'plan.json').read_text(encoding='utf-8'))
    require(plan['schema'] == 1 and plan['tag'] == TAG and
            plan['one_step_gate_sha256'] == GATE_SHA and
            plan['source_lock_sha256'] == full.sha(SOURCE_LOCK) and
            plan['full_native_plan_sha256'] == full.sha(full.ROOT/'plan.json'),
            'One-step native plan identity differs')
    full.verify_files(plan['source_files'])
    for row in plan['cases']:
        full.verify_files(row['files'])
    full.verify_files({plan['native_executable']:plan['native_executable_sha256'],
                       plan['exporter_executable']:plan['exporter_executable_sha256']})
    return plan


def run(variant, output=ROOT):
    output = Path(output).resolve()
    require(variant in full.VARIANTS, 'Unknown one-step variant')
    plan = load_plan(output)
    row = next(item for item in plan['cases'] if item['variant'] == variant)
    deck = Path(row['deck'])
    prepared, cpu = output/variant/'prepared', output/variant/'cpu'
    require(not prepared.exists() and not cpu.exists(), 'Preserve previous one-step outputs')
    prepared.mkdir(parents=True)
    raw = prepared/'raw_export'
    origin = full.export_reader.copy_inputs(deck, raw)
    with (raw/'initial.log').open('x', encoding='utf-8') as log:
        process = subprocess.run([plan['exporter_executable'], deck.name], cwd=raw,
            env=full.environment(exporter=True), stdout=log, stderr=subprocess.STDOUT,
            timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    require(process.returncode == 0, 'One-step zero-step exporter failed')
    for name in ('coupled_input.bin', 'normal_metadata.bin'):
        (prepared/name).write_bytes((raw/name).read_bytes())
    data = full.ancestor.read_case(prepared)
    checks = full.check_prepared(data, variant, output)
    full.save(prepared/'preparation_receipt.json', dict(schema=1, variant=variant,
        plan_sha256=full.sha(output/'plan.json'), one_step_gate_sha256=GATE_SHA,
        source_lock_sha256=full.sha(SOURCE_LOCK), origin=origin,
        artifacts={name:full.sha(prepared/name) for name in ('coupled_input.bin','normal_metadata.bin')},
        solver_steps_performed=0, checks=checks))
    full.export_reader.copy_inputs(deck, cpu)
    with (cpu/'native.log').open('x', encoding='utf-8') as log:
        process = subprocess.run([plan['native_executable'], deck.name], cwd=cpu,
            env=full.environment(), stdout=log, stderr=subprocess.STDOUT,
            timeout=90, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    full.save(cpu/'execution.json', dict(schema=1, variant=variant,
        plan_sha256=full.sha(output/'plan.json'), return_code=process.returncode,
        native_executable_sha256=full.sha(plan['native_executable']),
        files={str(path):full.sha(path) for path in cpu.iterdir() if path.is_file()},
        GPU_solver_executed=False, Delft3D_source_modified=False))
    require(process.returncode == 0, 'Unchanged native one-step evolution failed')
    return dict(variant=variant, prepared=str(prepared), cpu=str(cpu))


def analyze(variant, output=ROOT):
    output = Path(output).resolve()
    require(variant in full.VARIANTS, 'Unknown one-step variant')
    plan = load_plan(output)
    prep = output/variant/'prepared'
    cpu = output/variant/'cpu'
    receipt = json.loads((prep/'preparation_receipt.json').read_text(encoding='utf-8'))
    execution = json.loads((cpu/'execution.json').read_text(encoding='utf-8'))
    require(receipt['one_step_gate_sha256'] == GATE_SHA and
            receipt['plan_sha256'] == full.sha(output/'plan.json') and
            receipt['solver_steps_performed'] == 0 and
            execution['return_code'] == 0 and
            execution['plan_sha256'] == receipt['plan_sha256'],
            'One-step execution provenance differs')
    full.verify_files({str(prep/name):digest for name,digest in receipt['artifacts'].items()})
    full.verify_files(execution['files'])
    data = full.ancestor.read_case(prep)
    frames = read_frames(cpu/'frames.txt')
    times = np.array([frame['time_s'] for frame in frames])
    state = np.stack([frame['q'] for frame in frames])
    ledger = np.stack([frame['ledger'] for frame in frames])
    area = data['area']
    inlet = int(np.flatnonzero(data['mode']==9)[0])
    suspended = state[:,:,3:5]
    layers = state[:,:,5:9].reshape(2,128,2,2)
    water = state[:,:,0]@area
    classes = (suspended+layers.sum(axis=2)).transpose(0,2,1)@area
    water_residual = water-water[0]-(ledger[:,:,0].sum(axis=1)-ledger[0,:,0].sum())
    class_ledger = ledger[:,:,1:3].sum(axis=1)+ledger[:,:,3:5].sum(axis=1)
    class_residual = classes-classes[0]-(class_ledger-class_ledger[0])
    expected_q = full.integral_at(full.KNOTS_S,full.Q_M3_S,END_S)
    expected_feed = np.array([full.integral_at(full.KNOTS_S,full.FEED_KG_S[g],END_S)
                              for g in range(2)])
    match = re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)',
                      (cpu/'native.log').read_text(encoding='utf-8'))
    dry = state[:,:,0] <= float(data['dry_tolerance'])
    checks = dict(two_frames=bool(state.shape==(2,128,9) and ledger.shape==(2,2,5) and
                                  np.allclose(times,[0.,END_S],atol=1e-12,rtol=0)),
        one_accepted_zero_rejected=bool(match and int(match.group(1))==1 and int(match.group(2))==0),
        finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
        nonnegative=bool(np.min(state[:,:,0])>=-1e-12 and
                         np.min(suspended)>=-1e-12 and np.min(layers)>=-1e-12),
        wetdry_scope=bool(not dry[0].any() if variant=='pool' else
                          dry[0].any() and (~dry[0]).any()),
        water_budget=bool(np.max(np.abs(water_residual))<=1e-8),
        class_budget=bool(np.max(np.abs(class_residual))<=1e-7),
        q_integral=bool(abs(ledger[-1,inlet,0]-expected_q)<=1e-6),
        class_feed_integral=bool(np.max(np.abs(ledger[-1,inlet,3:5]-expected_feed))<=1e-6))
    result = dict(schema=1, variant=variant, passed=all(checks.values()), checks=checks,
        plan_sha256=full.sha(output/'plan.json'), gate_sha256=GATE_SHA,
        native_executable_sha256=plan['native_executable_sha256'],
        frame_sha256=full.sha(cpu/'frames.txt'), times_s=times,
        accepted_steps=int(match.group(1)) if match else None,
        rejected_trials=int(match.group(2)) if match else None,
        initial_dry=int(dry[0].sum()), final_dry=int(dry[-1].sum()),
        maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
        maximum_class_residual_kg=np.max(np.abs(class_residual),axis=0),
        inlet_q_integral_error_m3=float(ledger[-1,inlet,0]-expected_q),
        inlet_class_feed_integral_error_kg=ledger[-1,inlet,3:5]-expected_feed,
        GPU_solver_executed=False,Delft3D_source_modified=False)
    full.save(output/variant/'analysis.json', result)
    require(result['passed'], 'Native one-step pre-GPU gate failed: '+str(checks))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','analyze'))
    parser.add_argument('--variant', choices=tuple(full.VARIANTS))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action=='prepare': value=prepare(args.output)
    else:
        if args.variant is None:parser.error('--variant required')
        value=run(args.variant,args.output) if args.action=='run' else analyze(args.variant,args.output)
    print(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist()))


if __name__=='__main__':main()
