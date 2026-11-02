"""Write-once unchanged-native A2 combined prognostic/MPM comparator.

This does not execute CUDA or patch any native/Delft3D numerical source.
The pre-GPU gate is a separate hashed artifact, written before the deck/run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from braids_core.boundaries import cut_cell_io as writer
from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as a1
from . import coupled_reference as export_reader
from . import cpu_filter_cold20h as frozen_cpu
from . import normal_outlet_cases as sections


HERE = Path(__file__).resolve().parent
TAG = 'a2_combined_prognostic_native_20261004_v1'
ROOT = HERE/'runs'/TAG
GATE = HERE/'reports'/TAG/'GPU_ACCEPTANCE_GATE.txt'
GATE_SHA = '030a6bd86f3ccbab1ae7b048af0cd690001aa962a9756e72b59b20dcb984acf7'
ENV = dict(a1.ENV)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return a1.sha(path)


def save(path, row):
    a1.save(path, row)


def verify_files(mapping):
    a1.verify_files(mapping)


def prepare(output=ROOT):
    """Write the exact A1 physical pool state with v37 secondary controls ON."""
    output=Path(output).resolve()
    require(not output.exists(), 'A2 native reference is write-once')
    require(sha(GATE)==GATE_SHA, 'A2 predeclared gate changed')
    exporter, native_files, archived_dependencies=a1.verify_exporter()
    frozen_cpu.runtime_identity()
    sources={str(path.resolve()):sha(path) for path in
        (Path(__file__),GATE,Path(a1.__file__),Path(writer.__file__),
         Path(sections.__file__),Path(export_reader.__file__))}
    sources.update(native_files)
    sources.update(archived_dependencies)
    sources[str(a1.EXPORT_RUNTIME/'build.json')]=sha(a1.EXPORT_RUNTIME/'build.json')
    sources[str(a1.EXPORT_RUNTIME/'xbraids_normal_export.exe')]=exporter['executable_sha256']
    sources[str(frozen_cpu.TIGHT_EXE)]=frozen_cpu.TIGHT_SHA
    sources[str(frozen_cpu.TIGHT_RUNTIME/'build.json')]=sha(frozen_cpu.TIGHT_RUNTIME/'build.json')
    verify_files(sources)
    folder=output/'input';folder.mkdir(parents=True)
    base=a1.fixture('pool')
    metadata=writer.write_cut_case(folder/'case.data',base['mesh'],base['q'],base['rock'],
        base['boundaries'],sections={'in':base['section']},
        end_time_s=a1.END_S,dt_max_s=a1.DT_MAX_S,output_interval_s=a1.OUTPUT_S,
        cfl=.4,scheme=6,bedload=2,exchange=0,manning=.03,
        mpm_friction_mode=1,mpm_critical_shields=.047,
        sediment_fractions=base['fractions'],active_layer_thickness_m=.05,
        substrate_layer_count=1,substrate_layer_thickness_m=.05,
        secondary_flow=True,secondary_flow_model='prognostic',
        secondary_flow_transport_direction=True,secondary_flow_momentum_beta=1.,
        secondary_flow_espir=1.,morphological_factor=1.,compact_sidecars=True,
        source_sha256={'a2_fixture':sources[str(Path(__file__).resolve())],
                       'a2_gate':GATE_SHA,'a1_ancestor':sources[str(Path(a1.__file__).resolve())]})
    require(metadata['format']=='xbraids_cutcell-v37' and
            metadata['secondary_flow']['model']=='prognostic',
            'Writer did not preserve v37 prognostic controls')
    base['mesh'].write_geometry(folder/'geometry.npz')
    np.savez_compressed(folder/'initial_analysis.npz',q=base['q'],bed=base['bed'],
        rock=base['rock'],xy=base['xy'],
        area=np.array([cell.area_m2 for cell in base['mesh'].cells]),
        q_knots_s=a1.KNOTS_S,q_m3_s=a1.Q_M3_S,
        feed_kg_s=a1.FEED_KG_S,grain_d50_m=a1.D50_M)
    plan=dict(schema=1,tag=TAG,gate_sha256=GATE_SHA,
        gate_path=str(GATE.resolve()),source_files=sources,
        native_executable=str(frozen_cpu.TIGHT_EXE.resolve()),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        native_runtime_build=str((frozen_cpu.TIGHT_RUNTIME/'build.json').resolve()),
        native_runtime_build_sha256=sha(frozen_cpu.TIGHT_RUNTIME/'build.json'),
        zero_step_export_executable=str((a1.EXPORT_RUNTIME/'xbraids_normal_export.exe').resolve()),
        zero_step_export_executable_sha256=exporter['executable_sha256'],
        zero_step_export_build=str((a1.EXPORT_RUNTIME/'build.json').resolve()),
        zero_step_export_build_sha256=sha(a1.EXPORT_RUNTIME/'build.json'),
        input_files={str(path.resolve()):sha(path) for path in folder.iterdir() if path.is_file()},
        environment=ENV,controls=dict(cells=128,scheme=6,bedlaw=2,exchange=0,
            secondary_model=2,secondary_direction=True,secondary_beta=1.,
            end_s=a1.END_S,dtmax_s=a1.DT_MAX_S,output_s=a1.OUTPUT_S,
            q_knots_s=a1.KNOTS_S,q_m3_s=a1.Q_M3_S,
            class_feed_kg_s=a1.FEED_KG_S,grain_d50_m=a1.D50_M),
        native_source_modified=False,Delft3D_source_modified=False,
        GPU_solver_executed=False,physical_validation=False)
    save(output/'plan.json',plan)
    return dict(plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'),
                gate_sha256=GATE_SHA,input_files=plan['input_files'])


def load_plan(output=ROOT):
    output=Path(output).resolve()
    require(sha(GATE)==GATE_SHA, 'A2 predeclared gate changed')
    plan_path=output/'plan.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
    require(plan['schema']==1 and plan['tag']==TAG and plan['gate_sha256']==GATE_SHA,
            'A2 plan identity differs')
    verify_files(plan['source_files']);verify_files(plan['input_files'])
    verify_files({plan['native_executable']:plan['native_executable_sha256'],
        plan['native_runtime_build']:plan['native_runtime_build_sha256'],
        plan['zero_step_export_executable']:plan['zero_step_export_executable_sha256'],
        plan['zero_step_export_build']:plan['zero_step_export_build_sha256']})
    a1.verify_exporter();frozen_cpu.runtime_identity()
    return plan,sha(plan_path)


def environment(*,exporter=False):
    return a1.environment(exporter=exporter)


def check_prepared(data):
    inlet=int(np.flatnonzero(data['mode']==9)[0])
    outlet=int(np.flatnonzero(data['mode']==4)[0])
    checks=dict(cells=int(data['nc'])==128,
        scheme6=int(data['scheme'])==6,bedlaw2=int(data['bedlaw'])==2,
        exchange0=int(data['exchange_mode'])==0,
        two_classes_layers=int(data['nfrac'])==2 and int(data['nbed_layers'])==2,
        secondary2=int(data['secondary_flow_model'])==2 and
            bool(data['secondary_flow']) and bool(data['secondary_flow_transport_direction']) and
            float(data['secondary_flow_momentum_beta'])==1. and
            float(data['secondary_flow_espir'])==1.,
        appended_state=int(data['nstate'])==10 and int(data['secondary_index'])==9 and
            int(data['nledger'])==6 and int(data['secondary_ledger_index'])==5,
        native_ssp43=int(data['cpu_time_integrator'])==43,
        section_pair=np.array_equal(data['mode'],[9,4]) and
            np.array_equal(data['section_control'],[1,0]) and
            np.array_equal(np.diff(data['boundary_offset']),[8,8]) and
            int(np.count_nonzero(data['sampled']))==8 and
            bool(data['production_section_geometry']),
        hydraulic_schedule=int(data['nhydro_series'][inlet])==4 and
            np.array_equal(data['hydro_series_time'][:4,inlet],a1.KNOTS_S) and
            np.array_equal(data['hydro_series_q'][:4,inlet],a1.Q_M3_S),
        class_feed=np.array_equal(data['nfraction_series'][:,inlet],[4,4]) and
            np.array_equal(data['fraction_series_bed_input'][:4,:,inlet],a1.FEED_KG_S.T) and
            not np.any(data['nfraction_series'][:,outlet]),
        two_distinct_grains=np.array_equal(data['d50_fraction'],a1.D50_M),
        no_suspension=not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']),
        initial_pool=np.all(data['q'][0]>data['dry_tolerance']))
    require(all(checks.values()),'A2 zero-step preparation differs: '+str(checks))
    return checks


def execute(output=ROOT):
    output=Path(output).resolve();plan,plan_sha=load_plan(output)
    deck=output/'input/case.data';prepared=output/'prepared';cpu=output/'cpu'
    require(not prepared.exists() and not cpu.exists(),'A2 execution is write-once')
    prepared.mkdir();raw=prepared/'raw_export'
    origin=export_reader.copy_inputs(deck,raw)
    exporter=plan['zero_step_export_executable']
    with (raw/'initial.log').open('x',encoding='utf-8') as log:
        proc=subprocess.run([exporter,deck.name],cwd=raw,env=environment(exporter=True),
            stdout=log,stderr=subprocess.STDOUT,timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    require(proc.returncode==0,'A2 zero-step export failed: '+str(raw/'initial.log'))
    for name in ('coupled_input.bin','normal_metadata.bin'):
        (prepared/name).write_bytes((raw/name).read_bytes())
    data=sections.read_case(prepared);checks=check_prepared(data)
    save(prepared/'preparation_receipt.json',dict(schema=1,plan_sha256=plan_sha,
        exporter_executable_sha256=plan['zero_step_export_executable_sha256'],
        artifacts={name:sha(prepared/name) for name in ('coupled_input.bin','normal_metadata.bin')},
        origin=origin,solver_steps_performed=0,initialization_CPU_only=True,
        checks=checks,gate_sha256=GATE_SHA))
    export_reader.copy_inputs(deck,cpu)
    started=time.perf_counter()
    with (cpu/'native.log').open('x',encoding='utf-8') as log:
        proc=subprocess.run([plan['native_executable'],deck.name],cwd=cpu,
            env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(cpu/'execution.json',dict(schema=1,plan_sha256=plan_sha,
        status='completed' if proc.returncode==0 else 'failed',return_code=proc.returncode,
        elapsed_s=time.perf_counter()-started,environment=ENV,
        native_executable_sha256=plan['native_executable_sha256'],
        output_files={str(path.resolve()):sha(path) for path in cpu.iterdir() if path.is_file()},
        GPU_solver_executed=False,Delft3D_source_modified=False))
    require(proc.returncode==0,'A2 unchanged native run failed: '+str(cpu/'native.log'))
    return dict(prepared=str(prepared),cpu=str(cpu),return_code=proc.returncode)


def analyze(output=ROOT):
    output=Path(output).resolve();plan,plan_sha=load_plan(output)
    prepared=output/'prepared';cpu=output/'cpu'
    prep=json.loads((prepared/'preparation_receipt.json').read_text(encoding='utf-8'))
    execution=json.loads((cpu/'execution.json').read_text(encoding='utf-8'))
    require(prep['plan_sha256']==plan_sha and prep['solver_steps_performed']==0 and
            execution['plan_sha256']==plan_sha and execution['return_code']==0,
            'A2 native preparation/execution receipt differs')
    verify_files({str(prepared/name):digest for name,digest in prep['artifacts'].items()})
    verify_files(execution['output_files'])
    data=sections.read_case(prepared);check_prepared(data)
    frames=read_frames(cpu/'frames.txt')
    times=np.array([f['time_s'] for f in frames])
    state=np.stack([f['q'] for f in frames])
    ledger=np.stack([f['ledger'] for f in frames])
    require(state.shape==(9,128,10) and ledger.shape==(9,2,6) and
            np.allclose(times,a1.FRAME_TIMES,rtol=0,atol=1e-10),
            'A2 frame clock/state/ledger layout differs')
    area=data['area'];inlet=int(np.flatnonzero(data['mode']==9)[0]);outlet=1-inlet
    initial=np.load(output/'input/initial_analysis.npz',allow_pickle=False)
    bed0=np.asarray(initial['bed'])
    # Native frames store physical rows [h,hu,hv,suspended_g,layer_lg,M_I].
    layers=state[:,:,5:9].reshape(9,128,2,2)
    water=state[:,:,0]@area
    class_mass=(state[:,:,3:5]+layers.sum(axis=2)).transpose(0,2,1)@area
    water_residual=water-water[0]-(ledger[:,:,0].sum(axis=1)-ledger[0,:,0].sum())
    class_ledger=ledger[:,:,1:3].sum(axis=1)+ledger[:,:,3:5].sum(axis=1)
    class_residual=class_mass-class_mass[0]-(class_ledger-class_ledger[0])
    expected_q=np.array([a1.integral_at(a1.KNOTS_S,a1.Q_M3_S,t) for t in times])
    expected_feed=np.array([[a1.integral_at(a1.KNOTS_S,a1.FEED_KG_S[g],t)
                             for g in range(2)] for t in times])
    log=(cpu/'native.log').read_text(encoding='utf-8')
    match=re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)',log)
    secondary=state[:,:,9]
    secondary_inventory=secondary@area
    secondary_ledger=ledger[:,:,5].sum(axis=1)
    # The text frame format stores layer masses, not a separate bed column.
    # Reconstruct bed from the exported native rock and per-class dry bulk
    # densities, the same identity used by the native layer representation.
    bulk=np.asarray(data['bulk_fraction'])
    require(bulk.shape==(2,) and np.all(bulk>0), 'A2 native bulk density differs')
    bed=np.asarray(data['rock'])[None,:]+np.sum(layers/bulk[None,None,None,:],axis=(2,3))
    require(np.max(np.abs(bed[0]-bed0))<=1e-12,
            'Reconstructed initial native bed differs from frozen physical input')
    # The public frame stream does not contain the native secondary source
    # integral. Preserve this as an explicit pending gate, not a fake closure.
    checks=dict(nine_frames=True,
        finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
        nonnegative=bool(np.min(state[:,:,0])>=-1e-12 and
                         np.min(state[:,:,3:9])>=-1e-12),
        active_prognostic_field=bool(np.any(np.abs(secondary[1:])>1e-12) and
                                      np.any(np.abs(secondary[1:]-secondary[0])>1e-12)),
        active_secondary_boundary_ledger=bool(np.any(np.abs(ledger[1:,:,5])>1e-12)),
        active_each_class=bool(np.all(np.max(np.abs(layers[-1]-layers[0]),axis=(0,1))>1e-12)),
        active_bed=bool(np.max(np.abs(bed[-1]-bed0))>1e-12),
        water_budget=bool(np.max(np.abs(water_residual))<=1e-8),
        each_class_budget=bool(np.max(np.abs(class_residual))<=1e-7),
        inlet_q_integral=bool(np.max(np.abs(ledger[:,inlet,0]-expected_q))<=1e-6),
        inlet_class_feed_integral=bool(np.max(np.abs(ledger[:,inlet,3:5]-expected_feed))<=1e-6),
        positive_inlet=bool(ledger[-1,inlet,0]>0 and np.all(ledger[-1,inlet,3:5]>0)),
        critical_outlet_water_nonpositive=bool(np.all(ledger[:,outlet,0]<=1e-12)),
        no_native_rejections=bool(match and int(match.group(1))>0 and int(match.group(2))==0))
    result=dict(schema=1,tag=TAG,passed=all(checks.values()),checks=checks,
        secondary_source_budget='pending: source integral absent from public frame stream',
        gate_sha256=GATE_SHA,plan_sha256=plan_sha,
        native_executable_sha256=execution['native_executable_sha256'],
        prepared_receipt_sha256=sha(prepared/'preparation_receipt.json'),
        prepared_input_sha256=sha(prepared/'coupled_input.bin'),
        prepared_normal_metadata_sha256=sha(prepared/'normal_metadata.bin'),
        frames_sha256=sha(cpu/'frames.txt'),times_s=times,
        accepted_steps=int(match.group(1)) if match else None,
        rejected_trials=int(match.group(2)) if match else None,
        maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
        maximum_each_class_residual_kg=np.max(np.abs(class_residual),axis=0),
        max_inlet_q_integral_error_m3=float(np.max(np.abs(ledger[:,inlet,0]-expected_q))),
        max_inlet_feed_integral_error_kg=np.max(np.abs(ledger[:,inlet,3:5]-expected_feed),axis=0),
        secondary_inventory=secondary_inventory,
        secondary_boundary_ledger=secondary_ledger,
        max_abs_secondary_m2_s=float(np.max(np.abs(secondary))),
        max_abs_bed_change_m=float(np.max(np.abs(bed[-1]-bed0))),
        max_each_class_layer_change_kg_m2=np.max(np.abs(layers[-1]-layers[0]),axis=(0,1)),
        GPU_solver_executed=False,physical_validation=False,Delft3D_source_modified=False)
    save(output/'analysis.json',result)
    require(result['passed'],'A2 unchanged-native prerequisite failed: '+str(checks))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','analyze'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    result={'prepare':prepare,'run':execute,'analyze':analyze}[args.action](args.output)
    print(json.dumps(result,indent=2,allow_nan=False,
                     default=lambda value:value.tolist()))


if __name__=='__main__':main()
