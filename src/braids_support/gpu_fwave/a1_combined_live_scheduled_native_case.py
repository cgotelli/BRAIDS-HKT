"""Frozen unchanged-native precursor for one combined GPU boundary route.

This 128-cell two-class fixture composes a sampled mode-9 live-normal inlet,
mode-4 critical outlet, scheduled discharge and prescribed per-class bed feed.
It performs no GPU numerical work and never changes a native source file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from braids_core.boundaries import cut_cell_io as writer
from braids_core.boundaries.cut_cell_io import (
    CutBoundary, CutHydraulicBoundarySeries, CutSedimentBoundarySeries,
)
from braids_qgis.core.results import read_frames
from . import coupled_reference as export_reader
from . import cpu_filter_cold20h as frozen_cpu
from . import normal_coupled_oracle as original
from . import normal_outlet_cases as ancestor


HERE = Path(__file__).resolve().parent
TAG = 'a1_combined_live_scheduled_native_20261004_v1'
ROOT = HERE/'runs'/TAG
GATE = HERE/'reports'/TAG/'GPU_ACCEPTANCE_GATE.txt'
GATE_SHA = '2600fc5727c19664bbe68acd500321abb1db78dbdf0eb937c4a46cc93fcd22bc'
EXPORT_RUNTIME = HERE/'runs/i3_suspension_native_export_20261003_v1/runtime'
VARIANTS = {'pool': .14, 'wetdry': .095}
KNOTS_S = np.array([0., .07, .13, .20])
Q_M3_S = np.array([.08, .095, .065, .085])
FEED_KG_S = np.array([[.003, .005, .002, .004],
                      [.004, .002, .006, .003]])
D50_M = np.array([.00015, .00025])
END_S, DT_MAX_S, OUTPUT_S = .20, .002, .025
FRAME_TIMES = np.arange(9, dtype=np.float64)*OUTPUT_S
ENV = dict(OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE',
           BRAIDS_TIME_INTEGRATOR='ssp43')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return ancestor.sha(path)


def save(path, value):
    def plain(item):
        if isinstance(item,np.ndarray):
            return item.tolist()
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError('Unserializable receipt value: '+type(item).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=plain)+'\n',
                          encoding='utf-8')


def verify_files(mapping):
    for name, digest in mapping.items():
        require(Path(name).is_file() and sha(name)==digest,
                'Frozen file identity differs: '+str(name))


def verify_exporter():
    """Pin the old zero-step binary to its archived sources, not changed live helpers."""
    _, native_files = frozen_cpu.runtime_identity()
    build_path = EXPORT_RUNTIME/'build.json'
    build = json.loads(build_path.read_text(encoding='utf-8'))
    executable = EXPORT_RUNTIME/'xbraids_normal_export.exe'
    require(build['schema']==1 and
            Path(build['runtime']).resolve()==EXPORT_RUNTIME.resolve() and
            Path(build['executable']).resolve()==executable.resolve() and
            build['parent_executable_sha256']==frozen_cpu.TIGHT_SHA and
            build['source_equations_unchanged'] is True and
            build['transformation']==
                'cr.instrument I/O hooks plus zero-step raw tables/metadata exporter; no schedule oracle evaluation or guard change' and
            build['production_modified'] is False and
            build['Delft3D_modified'] is False,
            'Zero-step exporter provenance differs')
    verify_files({str(executable):build['executable_sha256'],
                  str(build_path):sha(build_path), **native_files})
    # Historical build.json records the live helper paths as they stood when
    # compiled; those helpers have since changed. Match their archived copies
    # inside the frozen runtime instead of silently accepting live-source drift.
    verify_files(build['protected_files'])
    verify_files(build['private_files'])
    verify_files(build['compiler_dlls'])
    archives = {str(EXPORT_RUNTIME/Path(path).name):digest
                for path,digest in build['dependencies'].items()}
    verify_files(archives)
    return build, native_files, archives


def source_files():
    return {str(Path(module.__file__).resolve()):sha(module.__file__)
            for module in (writer,export_reader,frozen_cpu,original,ancestor)} | {
            str(Path(__file__).resolve()):sha(__file__), str(GATE.resolve()):sha(GATE)}


def fixture(variant):
    require(variant in VARIANTS, 'Unknown combined-boundary variant')
    base = ancestor.fixture('prescribed')
    xy, bed = base['xy'], base['bed']
    q = base['q'].copy()
    q[:,0] = np.maximum(VARIANTS[variant]-.002*xy[:,0]-bed, 0.)
    q[:,1] = .35*q[:,0]
    q[:,2] = .015*np.sin(np.pi*xy[:,0]/4.)*np.sin(np.pi*xy[:,1]/2.)*q[:,0]
    fractions = tuple(replace(fraction,d50_m=float(diameter),d90_m=float(1.5*diameter))
                      for fraction,diameter in zip(base['fractions'],D50_M))
    feed = {f'sand_{g}':CutSedimentBoundarySeries(tuple(KNOTS_S),
            (0.,)*len(KNOTS_S),tuple(FEED_KG_S[g])) for g in range(2)}
    inlet = CutBoundary('inlet_normal',energy_slope=.002,boundary_manning_n=.03,
        hydraulic_series=CutHydraulicBoundarySeries(tuple(KNOTS_S),tuple(Q_M3_S)),
        fraction_sediment_series=feed)
    return dict(base, q=q, fractions=fractions,
                boundaries={'in':inlet,'out':CutBoundary('critical')})


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh folder; the reference is immutable')
    require(sha(GATE)==GATE_SHA, 'Pre-GPU gate changed')
    build, native_files, archives = verify_exporter()
    sources = {**source_files(), **native_files,
               str(EXPORT_RUNTIME/'build.json'):sha(EXPORT_RUNTIME/'build.json'),
               str(EXPORT_RUNTIME/'xbraids_normal_export.exe'):build['executable_sha256'],
               **archives}
    verify_files(sources)
    output.mkdir(parents=True)
    source_folder = output/'source'
    source_folder.mkdir()
    source_archive = {}
    for index,(name,digest) in enumerate(source_files().items()):
        archived = source_folder/(f'{index:02d}_'+Path(name).name)
        shutil.copy2(name,archived)
        require(sha(archived)==digest, 'Source snapshot differs: '+name)
        source_archive[str(archived)] = digest
    cases = []
    for variant in VARIANTS:
        data = fixture(variant)
        folder = output/'inputs'/variant
        folder.mkdir(parents=True)
        writer.write_cut_case(folder/'case.data',data['mesh'],data['q'],data['rock'],
            data['boundaries'],sections={'in':data['section']},
            end_time_s=END_S,dt_max_s=DT_MAX_S,output_interval_s=OUTPUT_S,
            cfl=.4,scheme=6,bedload=2,exchange=0,manning=.03,
            mpm_friction_mode=1,mpm_critical_shields=.047,
            sediment_fractions=data['fractions'],active_layer_thickness_m=.05,
            substrate_layer_count=1,substrate_layer_thickness_m=.05,
            secondary_flow=False,morphological_factor=1.,compact_sidecars=True,
            source_sha256={'fixture':sha(__file__),'gate':GATE_SHA,
                           'ancestor':sha(ancestor.__file__)})
        data['mesh'].write_geometry(folder/'geometry.npz')
        np.savez_compressed(folder/'initial_analysis.npz',q=data['q'],
            bed=data['bed'],rock=data['rock'],xy=data['xy'],
            area=np.array([cell.area_m2 for cell in data['mesh'].cells]),
            q_knots_s=KNOTS_S,q_m3_s=Q_M3_S,feed_kg_s=FEED_KG_S,
            grain_d50_m=D50_M)
        cases.append(dict(variant=variant,input=str(folder/'case.data'),
            input_files={str(path):sha(path) for path in folder.iterdir() if path.is_file()}))
    plan = dict(schema=1,tag=TAG,output=str(output),cases=cases,
        source_files=sources,source_archive=source_archive,
        pre_gpu_gate_sha256=GATE_SHA,
        native_executable=str(frozen_cpu.TIGHT_EXE),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        native_runtime_build=str(frozen_cpu.TIGHT_RUNTIME/'build.json'),
        native_runtime_build_sha256=sha(frozen_cpu.TIGHT_RUNTIME/'build.json'),
        zero_step_export_runtime=str(EXPORT_RUNTIME),
        zero_step_export_build_sha256=sha(EXPORT_RUNTIME/'build.json'),
        zero_step_export_executable_sha256=build['executable_sha256'],
        zero_step_export_parent_sha256=build['parent_executable_sha256'],
        environment=ENV,controls=dict(end_s=END_S,dt_max_s=DT_MAX_S,
            output_s=OUTPUT_S,cfl=.4,morfac=1.,q_knots_s=KNOTS_S,
            q_m3_s=Q_M3_S,bed_feed_kg_s=FEED_KG_S,grain_d50_m=D50_M,
            initial_stage_m=VARIANTS),
        GPU_solver_executed=False,Delft3D_source_modified=False,
        intention='Pre-GPU unchanged-native combined live-section/schedule engineering comparator')
    save(output/'plan.json',plan)
    return dict(plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'),
                gate_sha256=GATE_SHA,cases=list(VARIANTS))


def load_plan(output=ROOT):
    output = Path(output).resolve()
    require(sha(GATE)==GATE_SHA, 'Pre-GPU gate changed')
    plan_path = output/'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    require(plan['schema']==1 and plan['tag']==TAG and
            plan['pre_gpu_gate_sha256']==GATE_SHA and
            {row['variant'] for row in plan['cases']}==set(VARIANTS),
            'Combined native plan differs')
    verify_files(plan['source_files'])
    verify_files(plan['source_archive'])
    verify_files({plan['native_executable']:plan['native_executable_sha256'],
                  plan['native_runtime_build']:plan['native_runtime_build_sha256'],
                  str(EXPORT_RUNTIME/'build.json'):plan['zero_step_export_build_sha256'],
                  str(EXPORT_RUNTIME/'xbraids_normal_export.exe'):
                      plan['zero_step_export_executable_sha256']})
    verify_exporter()
    for row in plan['cases']:
        verify_files(row['input_files'])
    return plan, sha(plan_path)


def environment(*, exporter=False):
    env = {key:value for key,value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_','OMP_'))}
    env.update(ENV)
    if exporter:
        env['BRAIDS_GPU_INITIAL_EXPORT']='1'
        env['PATH']=str(EXPORT_RUNTIME)+os.pathsep+env.get('PATH','')
    else:
        env['PATH']=str(frozen_cpu.TIGHT_RUNTIME)+os.pathsep+env.get('PATH','')
    return env


def check_prepared(data,variant,output):
    inlet = int(np.flatnonzero(data['mode']==9)[0])
    outlet = int(np.flatnonzero(data['mode']==4)[0])
    initial = np.load(Path(output)/'inputs'/variant/'initial_analysis.npz',allow_pickle=False)
    offsets = data['boundary_offset']
    checks = dict(cells=int(data['nc'])==128,fractions=int(data['nfrac'])==2,
        layers=int(data['nbed_layers'])==2,deck_version=int(data['deck_version'])>=16,
        mode9_mode4=np.array_equal(data['mode'],np.array([9,4])),
        normal_control=np.array_equal(data['section_control'],np.array([1,0])),
        eight_faces_each=np.array_equal(np.diff(offsets),np.array([8,8])),
        surveyed_inlet=int(np.count_nonzero(data['sampled']))==8 and
            bool(data['production_section_geometry']),
        bedlaw2=int(data['bedlaw'])==2,exchange0=int(data['exchange_mode'])==0,
        secondary_off=int(data['secondary_flow_model'])==0,
        ssp43=int(data['cpu_time_integrator'])==43,
        hydraulic_schedule=int(data['nhydro_series'][inlet])==4 and
            np.array_equal(data['hydro_series_time'][:4,inlet],KNOTS_S) and
            np.array_equal(data['hydro_series_q'][:4,inlet],Q_M3_S),
        class_schedules=np.array_equal(data['nfraction_series'][:,inlet],np.array([4,4])) and
            np.array_equal(data['fraction_series_time'][:4,:,inlet],
                           np.broadcast_to(KNOTS_S[:,None],(4,2))) and
            np.array_equal(data['fraction_series_bed_input'][:4,:,inlet],FEED_KG_S.T) and
            not np.any(data['nfraction_series'][:,outlet]),
        no_suspended_input=not np.any(data['fraction_series_concentration']),
        initial_state=np.array_equal(data['state'].T,initial['q']),
        pool_or_wetdry=(bool(np.all(data['q'][0]>data['dry_tolerance'])) if variant=='pool'
                        else bool(np.any(data['q'][0]<=data['dry_tolerance']) and
                                  np.any(data['q'][0]>data['dry_tolerance']))))
    require(all(checks.values()), 'Zero-step combined preparation differs: '+str(checks))
    return checks


def execute_case(variant,output=ROOT):
    require(variant in VARIANTS, 'Unknown combined-boundary variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    row = next(item for item in plan['cases'] if item['variant']==variant)
    deck = Path(row['input'])
    prepared, cpu = output/variant/'prepared',output/variant/'cpu'
    require(not prepared.exists() and not cpu.exists(), 'Case execution must be fresh')
    prepared.mkdir(parents=True)
    raw = prepared/'raw_export'
    origin = export_reader.copy_inputs(deck,raw)
    build,_,_ = verify_exporter()
    with (raw/'initial.log').open('x',encoding='utf-8') as log:
        process = subprocess.run([build['executable'],deck.name],cwd=raw,
            env=environment(exporter=True),stdout=log,stderr=subprocess.STDOUT,
            timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    require(process.returncode==0 and (raw/'coupled_input.bin').is_file() and
            (raw/'normal_metadata.bin').is_file(),
            'Unchanged zero-step exporter failed; see initial.log')
    for name in ('coupled_input.bin','normal_metadata.bin'):
        (prepared/name).write_bytes((raw/name).read_bytes())
    data = ancestor.read_case(prepared)
    checks = check_prepared(data,variant,output)
    save(prepared/'preparation_receipt.json',dict(schema=1,plan_sha256=plan_sha,
        exporter_executable_sha256=build['executable_sha256'],
        exporter_parent_executable_sha256=build['parent_executable_sha256'],
        artifacts={name:sha(prepared/name) for name in ('coupled_input.bin','normal_metadata.bin')},
        origin=origin,solver_steps_performed=0,initialization_CPU_only=True,
        checks=checks,pre_gpu_gate_sha256=GATE_SHA))
    export_reader.copy_inputs(deck,cpu)
    started = time.perf_counter()
    with (cpu/'native.log').open('x',encoding='utf-8') as log:
        process = subprocess.run([plan['native_executable'],deck.name],cwd=cpu,
            env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(cpu/'execution.json',dict(schema=1,plan_sha256=plan_sha,
        status='completed' if process.returncode==0 else 'failed',
        return_code=process.returncode,environment=ENV,
        native_executable_sha256=sha(plan['native_executable']),
        elapsed_s=time.perf_counter()-started,
        output_files={str(path):sha(path) for path in cpu.iterdir() if path.is_file()},
        GPU_solver_executed=False,Delft3D_source_modified=False))
    require(process.returncode==0, 'Unchanged native combined run failed; see native.log')
    return dict(variant=variant,prepared=str(prepared),cpu=str(cpu),status='completed')


def integral_at(times,values,sample):
    total = 0.
    for left,right,low,high in zip(times[:-1],times[1:],values[:-1],values[1:]):
        if sample<=left:
            break
        stop=min(sample,right)
        top=low+(high-low)*(stop-left)/(right-left)
        total += .5*(low+top)*(stop-left)
        if sample<=right:
            break
    return total


def analyze_case(variant,output=ROOT):
    require(variant in VARIANTS, 'Unknown combined-boundary variant')
    output = Path(output).resolve()
    plan,plan_sha = load_plan(output)
    prepared,cpu = output/variant/'prepared',output/variant/'cpu'
    preparation = json.loads((prepared/'preparation_receipt.json').read_text(encoding='utf-8'))
    require(preparation['solver_steps_performed']==0 and
            preparation['pre_gpu_gate_sha256']==GATE_SHA and
            preparation['plan_sha256']==plan_sha,'Preparation identity differs')
    verify_files({str(prepared/name):digest for name,digest in preparation['artifacts'].items()})
    data = ancestor.read_case(prepared)
    execution = json.loads((cpu/'execution.json').read_text(encoding='utf-8'))
    require(execution['status']=='completed' and execution['return_code']==0 and
            execution['plan_sha256']==plan_sha and
            execution['native_executable_sha256']==plan['native_executable_sha256'],
            'Native execution identity differs')
    verify_files(execution['output_files'])
    frames = read_frames(cpu/'frames.txt')
    times = np.array([frame['time_s'] for frame in frames])
    state = np.stack([frame['q'] for frame in frames])
    ledger = np.stack([frame['ledger'] for frame in frames])
    require(np.allclose(times,FRAME_TIMES,rtol=0,atol=1e-10) and
            state.shape==(9,128,9) and ledger.shape==(9,2,5),
            'Native frame clock/layout differs')
    area = data['area']
    inlet,outlet = int(np.flatnonzero(data['mode']==9)[0]),int(np.flatnonzero(data['mode']==4)[0])
    suspended = state[:,:,3:5]
    layers = state[:,:,5:9].reshape(9,128,2,2)
    water_mass = state[:,:,0]@area
    class_mass = (suspended+layers.sum(axis=2)).transpose(0,2,1)@area
    water_ledger = ledger[:,:,0].sum(axis=1)
    class_ledger = ledger[:,:,1:3].sum(axis=1)+ledger[:,:,3:5].sum(axis=1)
    water_residual = water_mass-water_mass[0]-(water_ledger-water_ledger[0])
    class_residual = class_mass-class_mass[0]-(class_ledger-class_ledger[0])
    expected_q = np.array([integral_at(KNOTS_S,Q_M3_S,t) for t in times])
    expected_feed = np.array([[integral_at(KNOTS_S,FEED_KG_S[g],t) for g in range(2)]
                              for t in times])
    dry = state[:,:,0]<=float(data['dry_tolerance'])
    log = (cpu/'native.log').read_text(encoding='utf-8')
    match = re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)',log)
    checks = dict(nine_frames=True,
        finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
        nonnegative=bool(np.min(state[:,:,0])>=-1e-12 and
                         np.min(suspended)>=-1e-12 and np.min(layers)>=-1e-12),
        pool_or_wetdry=(bool(not dry[0].any()) if variant=='pool' else
                        bool(dry[0].any() and (~dry[0]).any() and
                             np.any(dry!=dry[0]))),
        water_budget=bool(np.max(np.abs(water_residual))<=1e-8),
        each_class_budget=bool(np.max(np.abs(class_residual))<=1e-7),
        inlet_q_analytic=bool(np.max(np.abs(ledger[:,inlet,0]-expected_q))<=1e-6),
        inlet_feed_analytic=bool(np.max(np.abs(ledger[:,inlet,3:5]-expected_feed))<=1e-6),
        positive_inlet=bool(ledger[-1,inlet,0]>0 and
                            np.all(ledger[-1,inlet,3:5]>0)),
        outlet_water_nonpositive=bool(np.all(ledger[:,outlet,0]<=1e-12)),
        no_suspended_inlet=bool(np.all(np.abs(ledger[:,inlet,1:3])<=1e-12)),
        internal_knots_between_frames=bool(all(not np.any(np.isclose(times,k,atol=1e-14,rtol=0))
                                                 for k in KNOTS_S[1:-1])),
        accepted_without_rejection=bool(match and int(match.group(1))>0 and
                                        int(match.group(2))==0))
    result = dict(schema=1,variant=variant,passed=all(checks.values()),checks=checks,
        pre_gpu_gate_sha256=GATE_SHA,plan_sha256=plan_sha,
        native_executable_sha256=execution['native_executable_sha256'],
        prepared_receipt_sha256=sha(prepared/'preparation_receipt.json'),
        prepared_input_sha256=sha(prepared/'coupled_input.bin'),
        prepared_schedule_sha256=sha(prepared/'normal_metadata.bin'),
        native_frames_sha256=sha(cpu/'frames.txt'),times_s=times,
        accepted_steps=int(match.group(1)) if match else None,
        rejected_trials=int(match.group(2)) if match else None,
        dry_cell_counts=dry.sum(axis=1),
        maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
        maximum_class_residual_kg=np.max(np.abs(class_residual),axis=0),
        maximum_inlet_q_integral_error_m3=float(np.max(np.abs(ledger[:,inlet,0]-expected_q))),
        maximum_inlet_feed_integral_error_kg=np.max(np.abs(ledger[:,inlet,3:5]-expected_feed),axis=0),
        final_inlet_water_m3=float(ledger[-1,inlet,0]),
        final_inlet_bed_kg=ledger[-1,inlet,3:5],
        final_outlet_water_m3=float(ledger[-1,outlet,0]),
        minimum_depth_m=float(np.min(state[:,:,0])),
        GPU_solver_executed=False,physical_validation=False,Delft3D_source_modified=False)
    save(output/variant/'analysis.json',result)
    require(result['passed'],'Combined native prerequisite failed: '+str(checks))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','analyze'))
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--variant',choices=tuple(VARIANTS))
    args = parser.parse_args()
    if args.action=='prepare':
        result=prepare(args.output)
    else:
        if args.variant is None:
            parser.error('--variant is required')
        result=(execute_case(args.variant,args.output) if args.action=='run' else
                analyze_case(args.variant,args.output))
    print(json.dumps(result,indent=2,allow_nan=False,
                     default=lambda value:value.tolist()))


if __name__=='__main__':
    main()
