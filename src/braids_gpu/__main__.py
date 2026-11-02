"""Command-line API suitable for an external QGIS GPU worker process."""
import argparse
import json
import os
from pathlib import Path
from braids_core.windows_paths import filesystem_path
import shutil
import subprocess
from .backend import dispatch
from braids_core.sediment.qgis_product_v109 import (
    PRODUCT_SIDECAR, load_product_document_v109,
)

A1_VARIANT='combined_live_scheduled_a1_ssp43'
AMR_VARIANT='authenticated_composed_amr_ssp33'
ON_DEMAND_AMR_VARIANT='on_demand_bank_amr_ssp33'
ON_DEMAND_TRANSPORT_AMR_VARIANT='on_demand_transport_amr_ssp33'
SECONDARY_LIVE_VARIANT='live_normal_critical_secondary_mpm_ssp43'
LIVE_SUSPENSION_VARIANT='live_normal_critical_suspension_mpm_ssp33'
LIVE_VR_VARIANT='live_normal_critical_van_rijn_mpm_ssp33'
GRADED_LIVE_VR_VARIANT='live_normal_critical_van_rijn_graded_ssp33'
TARGET_B_ADAPTATION_VARIANT='live_normal_critical_graded_adaptation_ssp33'
GRADED_BEDLOAD_GENERAL_VARIANT='live_normal_critical_graded_bedload_ssp33'
LIVE_INLET_VARIANT='live_mode9_section_choice_ssp33'
GENERAL_LIVE_BANK_VARIANT='live_normal_critical_bank_ssp33'
GENERAL_LIVE_TRANSPORT_VARIANT='live_normal_critical_transport_ssp33'
INTEGRATED_VARIANT='integrated'
G3_VARIANT='physical_toe_mode2_avalanche_secondary_ssp33'
G3_LIVE_BANK_MODE2_VARIANT='g3_live_bank_mode2_ssp33'
G3_LIVE_BANK_MODE3_VARIANT='g3_live_bank_mode3_ssp33'
G3_LIVE_BANK_VARIANTS=(G3_LIVE_BANK_MODE2_VARIANT,G3_LIVE_BANK_MODE3_VARIANT)
QGIS_PREPARED_BINDING='qgis_product_prepared_v109.json'


def publish_qgis_product_result(input_folder,output_folder):
    """Carry the prepared product identity into results/restart provenance."""
    source=Path(input_folder)/PRODUCT_SIDECAR
    binding=Path(input_folder)/QGIS_PREPARED_BINDING
    if not source.is_file() and not binding.is_file():
        return None
    if not source.is_file() or not binding.is_file():
        raise ValueError('Incomplete Q10 prepared product identity')
    profile=load_product_document_v109(json.loads(source.read_text(encoding='ascii')))
    output=Path(output_folder)
    output_sidecar=output/PRODUCT_SIDECAR
    output_binding=output/QGIS_PREPARED_BINDING
    if output_sidecar.exists() or output_binding.exists():
        raise ValueError('Q10 result product identity already exists')
    shutil.copy2(source,output_sidecar);shutil.copy2(binding,output_binding)
    digest=__import__('hashlib').sha256(output_sidecar.read_bytes()).hexdigest()
    run_path=output/'run.json'
    run=json.loads(run_path.read_text(encoding='utf-8'))
    run['qgis_product_v109']={
        'fingerprint':profile.fingerprint,'backend':profile.backend,
        'integrator':profile.integrator,'sidecar':PRODUCT_SIDECAR,
        'sidecar_sha256':digest,'prepared_binding':QGIS_PREPARED_BINDING,
    }
    temporary=run_path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(run,sort_keys=True,separators=(',',':'),
                                    allow_nan=False),encoding='utf-8')
    temporary.replace(run_path)
    return run['qgis_product_v109']


def source_worker_scope():
    """Return the source-archive restrictions, absent in the live checkout."""
    manifest_path=Path(__file__).resolve().parents[1]/'PACKAGE-MANIFEST.json'
    if not manifest_path.is_file():return None
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('kind')!='experimental Windows source worker':
        raise ValueError('Unrecognized GPU worker package manifest')
    scope=manifest.get('source_worker_scope')
    if not isinstance(scope,dict) or not all(isinstance(scope.get(key),dict) for key in
            ('excluded_gpu_variants','excluded_prepared_sidecars')):
        raise ValueError('GPU source-worker scope is missing from package manifest')
    return scope


def prepare_frozen_a1(deck,output):
    """Copy only a fully authenticated native A1 zero-step export."""
    from .runner import sha
    from .a1_source_lock_v2 import compatible_source_lock
    from braids_support.gpu_fwave import a1_combined_full_gate_preflight as gate
    from braids_support.g2_target_b import prepare_input as native
    from braids_support.gpu_fwave import normal_coupled_oracle as exporter_environment
    deck=filesystem_path(deck).resolve();output=filesystem_path(output).resolve()
    if deck.name!='case.data':raise ValueError('A1 preparation requires a native case.data deck')
    with compatible_source_lock():plan,_=native.load_plan()
    matches=[]
    for row in plan['cases']:
        source=Path(row['input']).parent
        expected={Path(name).name:digest for name,digest in row['input_files'].items()}
        if deck.parent!=source and (all((deck.parent/name).is_file() and
              sha(deck.parent/name)==digest for name,digest in expected.items())):
            matches.append(row['variant'])
        elif deck.parent==source and all(sha(source/name)==digest for name,digest in expected.items()):
            matches.append(row['variant'])
    if len(matches)!=1:raise ValueError('A1 native deck and all sidecars must match one frozen pool/wetdry case')
    variant=matches[0]
    with compatible_source_lock():gate.preflight(variant)
    prepared=native.ROOT/variant/'prepared'
    receipt=json.loads((prepared/'preparation_receipt.json').read_text(encoding='utf-8'))
    names=set(receipt['artifacts'])|{'preparation_receipt.json'}
    if names!={'coupled_input.bin','normal_metadata.bin','preparation_receipt.json'}:
        raise ValueError('A1 prepared sidecar manifest differs from the frozen binary route')
    output.mkdir(parents=True,exist_ok=False)
    for name in sorted(names):shutil.copy2(prepared/name,output/name)
    if any(sha(output/name)!=sha(prepared/name) for name in names):
        raise ValueError('A1 prepared copy changed during authenticated installation')
    return dict(status='prepared',output=str(output),variant=A1_VARIANT,
        case=variant,normal_metadata_sha256=sha(output/'normal_metadata.bin'),
        preparation_receipt_sha256=sha(output/'preparation_receipt.json'),
        frozen_fixture_only=True)


def prepare_live_secondary(deck,output):
    """Run the locked I/O-only native exporter for a supported live deck."""
    from .runner import load_prepared,sha,time_controls
    from .live_secondary import validate_scope
    from braids_support.gpu_fwave import a1_combined_live_scheduled_native_case as native
    from braids_support.gpu_fwave.coupled_reference import copy_inputs

    deck=filesystem_path(deck).resolve();output=filesystem_path(output).resolve()
    if deck.name!='case.data':raise ValueError('Live secondary preparation requires a native case.data deck')
    build,_,_=native.verify_exporter()
    executable=native.EXPORT_RUNTIME/'xbraids_normal_export.exe'
    origin=copy_inputs(deck,output)
    process=subprocess.run([str(executable),deck.name],cwd=output,
        env=native.environment(exporter=True),capture_output=True,text=True,timeout=240,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (output/'initial.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    process.check_returncode()
    artifacts={name:sha(output/name) for name in ('coupled_input.bin','normal_metadata.bin')
               if (output/name).is_file()}
    if set(artifacts)!={'coupled_input.bin','normal_metadata.bin'}:
        raise ValueError('Locked live exporter omitted a required prepared record')
    from .live_secondary import read_case
    data=read_case(output)
    start,end,interval,dtmax=time_controls(data,None,None,None)
    events=validate_scope(data,start=start,end=end,interval=interval,dtmax=dtmax,
        launch_mode='plain',integrator='ssp43')
    receipt=dict(schema=1,variant=SECONDARY_LIVE_VARIANT,artifacts=artifacts,
        origin=origin,exporter_executable_sha256=build['executable_sha256'],
        initialization_CPU_only=True,solver_steps_performed=0)
    from .atomic_output import write_json_atomic
    write_json_atomic(output/'preparation_receipt.json',receipt)
    load_prepared(output,general_live_sections=True)
    return dict(status='prepared',output=str(output),variant=SECONDARY_LIVE_VARIANT,
        cells=int(data['nc']),faces=int(data['nf']),forcing_knots_s=events.knots.tolist(),
        preparation_receipt_sha256=sha(output/'preparation_receipt.json'),
        exact_fixture_hash_required=False)


def prepare_graded_bedload(deck,output,*,stop_file=None):
    """Produce the common graded-bedload contract from an ordinary native deck.

    The pinned native executable exports initial state and section metadata
    only.  No native or GPU time step is taken during preparation.
    """
    import time
    import numpy as np
    from .atomic_output import write_json_atomic
    from .runner import load_prepared, sha, time_controls
    from .live_bedload_adaptation import validate_general_scope, read_case
    from braids_support.gpu_fwave.coupled_reference import copy_inputs

    deck=filesystem_path(deck).resolve();output=filesystem_path(output).resolve()
    if deck.name!='case.data':
        raise ValueError('General graded-bedload preparation requires a native case.data deck')
    if stop_file is not None and filesystem_path(stop_file).exists():
        raise InterruptedError('GPU preparation cancelled before copying inputs')
    if source_worker_scope() is None:
        from braids_support.g2_target_b import prepare_input as native
        from braids_support.gpu_fwave import normal_coupled_oracle as exporter_environment
        build=native.verify_archived_exporter()
        executable=Path(build['executable'])
        environment=exporter_environment.environment(True,native.RUNTIME)
        environment['BRAIDS_TIME_INTEGRATOR']='legacy'
    else:
        from .graded_native_exporter import verify_packaged_exporter,export_environment
        build=verify_packaged_exporter()
        executable=Path(build['executable'])
        environment=export_environment(build['runtime'])
    origin=copy_inputs(deck,output,stop_file=stop_file)
    process=subprocess.Popen([str(executable),deck.name],cwd=output,
        env=environment,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    began=time.monotonic()
    try:
        while process.poll() is None:
            if stop_file is not None and filesystem_path(stop_file).exists():
                raise InterruptedError('GPU preparation cancelled during zero-step native export')
            if time.monotonic()-began>240.:
                raise TimeoutError('GPU zero-step native export exceeded 240 seconds')
            time.sleep(.05)
    except (InterruptedError,TimeoutError):
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
        raise
    stdout,stderr=process.communicate()
    (output/'initial.log').write_text(stdout+stderr,encoding='utf-8')
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode,
            [str(executable),deck.name],output=stdout,stderr=stderr)
    if not all((output/name).is_file() for name in
               ('coupled_input.bin','normal_metadata.bin')):
        raise ValueError('Native zero-step exporter omitted a required graded-bedload record')
    data=read_case(output)
    start,end,interval,dtmax=time_controls(data,None,None,None)
    events=validate_general_scope(data,start=start,end=end,interval=interval,
        dtmax=dtmax,launch_mode='plain',integrator='ssp33')
    if stop_file is not None and filesystem_path(stop_file).exists():
        raise InterruptedError('GPU preparation cancelled before sealing artifacts')
    np.savez_compressed(output/'coupled_input.npz',**data)
    artifacts={name:sha(output/name) for name in
               ('coupled_input.bin','coupled_input.npz','normal_metadata.bin')}
    receipt=dict(schema=3,variant=GRADED_BEDLOAD_GENERAL_VARIANT,
        artifacts=artifacts,origin=origin,
        exporter_executable_sha256=build['executable_sha256'],
        initialization_CPU_only=True,native_solver_steps_performed=0,
        GPU_solver_executed=False)
    write_json_atomic(output/'preparation_receipt.json',receipt)
    loaded,_=load_prepared(output,general_live_sections=True)
    if not all(np.array_equal(loaded[key],data[key]) for key in data):
        raise ValueError('Graded-bedload prepared round trip changed native records')
    return dict(status='prepared',output=str(output),
        variant=GRADED_BEDLOAD_GENERAL_VARIANT,cells=int(data['nc']),
        faces=int(data['nf']),forcing_knots_s=events.knots.tolist(),
        preparation_receipt_sha256=sha(output/'preparation_receipt.json'),
        native_solver_steps_performed=0,archive_bound=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('capabilities',help='Show explicit backend and current physical scope')
    adaptive_prep=commands.add_parser('prepare-adaptive-river',
        help='Prepare the combined common-clock river bundle (checkout only, explicit SSPRK2)')
    adaptive_prep.add_argument('--profile',required=True,type=Path)
    adaptive_prep.add_argument('--fields',required=True,type=Path)
    adaptive_prep.add_argument('--output',required=True,type=Path)
    adaptive_prep.add_argument('--stop-file',type=Path)
    adaptive_run=commands.add_parser('run-adaptive-river',
        help='Run combined adaptive river physics with explicit CPU/GPU choice (checkout only)')
    adaptive_run.add_argument('--input',required=True,type=Path)
    adaptive_run.add_argument('--output',required=True,type=Path)
    adaptive_run.add_argument('--backend',choices=('cpu','gpu'),required=True)
    adaptive_run.add_argument('--until',type=float)
    adaptive_run.add_argument('--resume',type=Path)
    adaptive_run.add_argument('--stop-file',type=Path)
    current_prep=commands.add_parser('prepare-current-river',
        help='Prepare the maintained fully wet river bundle (checkout integration scope)')
    current_prep.add_argument('--input',required=True,type=Path)
    current_prep.add_argument('--profile',required=True,type=Path)
    current_prep.add_argument('--output',required=True,type=Path)
    current_prep.add_argument('--stop-file',type=Path)
    current_run=commands.add_parser('run-current-river',
        help='Run the maintained river bundle with explicit CPU/GPU choice')
    current_run.add_argument('--input',required=True,type=Path)
    current_run.add_argument('--output',required=True,type=Path)
    current_run.add_argument('--backend',choices=('cpu','gpu'),required=True)
    current_run.add_argument('--until',type=float)
    current_run.add_argument('--resume',type=Path)
    current_run.add_argument('--stop-file',type=Path)
    current_run.add_argument('--max-steps',type=int)
    prep=commands.add_parser('prepare',help='Initialize GPU geometry/configuration from a native deck on CPU')
    prep.add_argument('--input',required=True,type=Path);prep.add_argument('--output',required=True,type=Path)
    prep.add_argument('--stop-file',type=Path,
                      help='Cooperative cancellation marker for native GPU input preparation')
    prep.add_argument('--evolving-banks',action='store_true',
                      help='Versioned evolving bank/history contract for on-demand AMR preparation')
    prep.add_argument('--recursive-bank-bedform-avalanche',action='store_true',
                      help='Select the exact recursive mode-3 bank, bedform and parent-avalanche initializer')
    prep.add_argument('--gpu-amr-selector-policy',choices=('local_eta_plane_v1',),
                      help='Explicit versioned slope-aware selector for nonbank general GPU AMR')
    prep.add_argument('--native-integrator',choices=('ssp43','legacy'),default='ssp43',
                      help='Native zero-step initializer method; legacy initializes SSPRK33 GPU cases')
    prep.add_argument('--gpu-preparation-variant',choices=('stock',INTEGRATED_VARIANT,A1_VARIANT,SECONDARY_LIVE_VARIANT,GRADED_BEDLOAD_GENERAL_VARIANT,LIVE_INLET_VARIANT,GENERAL_LIVE_BANK_VARIANT,GENERAL_LIVE_TRANSPORT_VARIANT,ON_DEMAND_AMR_VARIANT,ON_DEMAND_TRANSPORT_AMR_VARIANT),default='stock',
                      help='Select stock native export, frozen A1 copy, or guarded live-secondary zero-step export')
    run=commands.add_parser('run',help='Run exactly the backend selected by the user')
    run.add_argument('--backend',choices=('cpu','gpu'),required=True)
    run.add_argument('--input',required=True,type=Path,help='GPU: prepared folder or exact authenticated AMR input folder; CPU: native case.data')
    run.add_argument('--output',required=True,type=Path,help='New output folder; existing folders are never overwritten')
    run.add_argument('--cpu-executable',type=Path)
    run.add_argument('--gpu-launch-mode',choices=('graph','plain'),default='graph',help='GPU kernel launch mode; never changes execution backend')
    run.add_argument('--gpu-integrator',choices=('ssp43','ssp33','implicit_fwave_midpoint'),default='ssp43',
                     help='GPU time integrator; route admission rejects unsupported method/launch combinations')
    run.add_argument('--gpu-numerical-variant',
                     choices=('auto','scheduled_normal9_critical4',INTEGRATED_VARIANT,A1_VARIANT,SECONDARY_LIVE_VARIANT,LIVE_SUSPENSION_VARIANT,LIVE_VR_VARIANT,GRADED_LIVE_VR_VARIANT,TARGET_B_ADAPTATION_VARIANT,GRADED_BEDLOAD_GENERAL_VARIANT,LIVE_INLET_VARIANT,GENERAL_LIVE_BANK_VARIANT,GENERAL_LIVE_TRANSPORT_VARIANT,'qualified_curved_wetdry',
                              AMR_VARIANT,ON_DEMAND_AMR_VARIANT,ON_DEMAND_TRANSPORT_AMR_VARIANT,
                              'scheduled_inlet_ssp33','bedform_feedback_ssp33',
                              'general_avalanche_plain_ssp33',
                              'physical_toe_mode2_plain_ssp33',
                              'physical_toe_mode2_avalanche_ssp33',
                              G3_VARIANT,
                              G3_LIVE_BANK_MODE2_VARIANT,G3_LIVE_BANK_MODE3_VARIANT,
                              'distributed_slope_mode3_plain_ssp33',
                              'distributed_slope_mode3_avalanche_ssp33',
                              'bedload_adaptation_plain_ssp33'),default='auto',
                     help='Explicit GPU numerical scope; unsupported prepared inputs fail closed')
    run.add_argument('--until',type=float);run.add_argument('--output-every',type=float);run.add_argument('--dt-max',type=float)
    run.add_argument('--max-steps',type=int);run.add_argument('--stop-file',type=Path);run.add_argument('--resume',type=Path)
    run.add_argument('--gpu-capture-step-trace',action='store_true',
                     help='Write per-accepted-step clock/CFL trace for a focused GPU validation run')
    run.add_argument('--gpu-source-archive',type=Path,
                     help='Authenticated, fixture-independent graded-bedload numerical source bundle')
    args=parser.parse_args(argv)
    package_scope=source_worker_scope()
    if args.command in ('prepare-adaptive-river','run-adaptive-river'):
        if package_scope is not None:
            parser.error('Combined adaptive river bundle is not yet qualified in a packaged QGIS worker')
        from braids_core.sediment import adaptive_river_worker as adaptive
        try:
            if args.command=='prepare-adaptive-river':
                import numpy as np
                with np.load(args.fields,allow_pickle=False) as fields:
                    output=adaptive.prepare(json.loads(args.profile.read_text(encoding='utf8')),
                        {k:fields[k] for k in fields.files},args.output,stop_file=args.stop_file)
                result=dict(status='prepared',input=str(output))
            else:
                result=adaptive.run(args.input,args.output,backend=args.backend,until=args.until,
                    restart=args.resume,stop_file=args.stop_file,
                    on_frame=lambda output,value:print(json.dumps(dict(value,phase='accepted_frame',output=str(output))),flush=True))
        except InterruptedError as exc:
            print(json.dumps(dict(status='cancelled',error=str(exc))));return 130
        print(json.dumps(result));return 130 if result['status']=='cancelled' else 0
    if args.command in ('prepare-current-river','run-current-river'):
        if package_scope is not None:
            parser.error('Current river bundle is not yet qualified in a packaged QGIS worker')
        from braids_core.sediment.current_river_worker import prepare,run
        try:
            if args.command=='prepare-current-river':
                profile=json.loads(args.profile.read_text(encoding='utf8'))
                result=prepare(args.input,profile,args.output,stop_file=args.stop_file)
            else:
                result=run(args.input,args.output,backend=args.backend,until=args.until,
                    resume=args.resume,stop_file=args.stop_file,max_steps=args.max_steps,
                    progress=lambda value:print(json.dumps(value),flush=True))
        except InterruptedError as exc:
            print(json.dumps(dict(status='cancelled',error=str(exc))));return 130
        print(json.dumps(result));return 130 if result['status']=='cancelled' else 0
    if args.command=='capabilities':
        capabilities=dict(backends=['cpu','gpu'],selection='explicit user choice',automatic_fallback=False,
            gpu_amr_selector_policies=['local_eta_plane_v1'],
            GPU_scope=['fixed-grid scheme-6 augmented f-wave',
                'plain SSPRK43 for eligible fixed-grid nonbank MPM/exchange-off runs; explicit plain SSPRK33 for the remaining integrated routes',
                'Manning friction','no-bedload hydraulics and MPM bedload in bounded fixed-grid paths',
                'noncohesive suspended advection and exchange modes 0/1/2 in bounded plain cases',
                'secondary flow OFF/equilibrium/prognostic',
                'graded MPM with mapped hiding/exposure controls',
                'one active plus 1-32 substrate layers in the resident fixed-grid operator',
                'conservative arbitrary-depth 2:1 recursive AMR for the admitted nonbank, physical-toe and distributed-bank routes',
                'horizontal viscosity mode1 geometry1',
                'GPU wet/dry diffusion geometry rebuild','constant supported boundary controls'],
            qualified_engineering_cases=['I1 OFF/equilibrium two-grain 400-step native comparison and restart',
                'I2 OFF/equilibrium seven-grain four-layer 500-step native comparison and restart',
                'I2 specified-Cf MPM friction mode 0 bounded native comparison, rollback and restart',
                'I3 no-bedload six native/GPU hydraulic, settling and van Rijn cases',
                'I4 physical bank-toe mode 2 bounded ON/OFF native comparison and restart',
                'I4 distributed-slope bank mode 3 bounded ON/OFF current-source comparison and restart; 251-frame active donor-chain gate passed',
                'I4 distributed-slope bank mode 3 plus general avalanche: four-case 251-frame native gate, active-chain rollback and bitwise restart passed'],
            selectable_research_variants=['scheduled_normal9_critical4',
                INTEGRATED_VARIANT+' (common ordinary-preparation entry; selects connected physical operators, never a CPU/GPU heuristic; full inventory integration in progress)',
                GENERAL_LIVE_TRANSPORT_VARIANT+' (ordinary hydraulics/bedload/passive and exchanging suspension; secondary/adaptation/viscosity and legal bedform branch; eligible nonbank MPM/exchange-off runs admit plain SSPRK43, otherwise plain SSPRK33)',
                ON_DEMAND_AMR_VARIANT+' (packaged on-demand bank AMR with synchronous stepping plus recursive static/evolving physical-toe mode 2 and recursive evolving distributed-bank mode 3; exact constant-stage bedform/parent-avalanche compositions are available for evolving modes 2 and 3)',
                ON_DEMAND_TRANSPORT_AMR_VARIANT+' (packaged nonbank AMR with synchronous broad-physics stepping plus recursive MPM/exchange-2/prognostic-secondary stepping, full-leaf bedload adaptation, parent post-reflux mode1/geometry1 velocity-Laplacian viscosity and general avalanche)',
                GENERAL_LIVE_BANK_VARIANT+' (packaged ordinary bank modes2/3 with MPM, secondary flow, avalanche and supported viscosity; plain SSPRK33)',
                AMR_VARIANT+' (exact archived composed 91-to-64 AMR fixture only; plain SSPRK33; not general AMR or QGIS release)',
                A1_VARIANT+' (frozen 128-cell pool/wetdry live Q and two-class feed; plain SSPRK43 only; no QGIS/package claim)',
                SECONDARY_LIVE_VARIANT+' (guarded initially wet or mixed wet/dry two-class live sections, scheduled Q/feed, MPM and prognostic secondary; plain SSPRK43; no QGIS/package claim)',
                'qualified_curved_wetdry',
                'scheduled_inlet_ssp33 (bounded native comparison)',
                'bedform_feedback_ssp33 (bounded original 50/50 numerical gate passed)',
                'general_avalanche_plain_ssp33 (bounded ON/OFF native comparison passed)',
                'physical_toe_mode2_plain_ssp33 (bounded ON/OFF native comparison passed)',
                'physical_toe_mode2_avalanche_ssp33 (bounded four-switch native comparison, rollback and restart passed; broader combinations unqualified)',
                G3_VARIANT+' (frozen four-case engineering gate in progress; no QGIS/package or observational claim)',
                *(variant+' (archive-bound 128-cell live-bank engineering gate in progress; no QGIS/package or observational claim)'
                  for variant in G3_LIVE_BANK_VARIANTS),
                LIVE_VR_VARIANT+' (archive-bound frozen two-case live Van Rijn exchange-2 gate; not packaged/QGIS-qualified)',
                GRADED_LIVE_VR_VARIANT+' (archive-bound seven-class/four-layer live Van Rijn research gate; not packaged/QGIS-qualified)',
                TARGET_B_ADAPTATION_VARIANT+' (archive-bound seven-class/four-layer live bedload-adaptation OFF/ON research gate; not packaged/QGIS-qualified)',
                GRADED_BEDLOAD_GENERAL_VARIANT+' (general schema-3 prepared 128/200-cell numerical gate; package and reach-scale qualification pending)',
                LIVE_INLET_VARIANT+' (surveyed stage/depth/critical inlet with mode4 outlet, two-class MPM plain SSPRK33; common worker engineering route, QGIS/package qualification pending)',
                'distributed_slope_mode3_plain_ssp33 (bounded ON/OFF native comparison, active donor-chain collapse, rollback and restart passed; reach-scale still unqualified)',
                'distributed_slope_mode3_avalanche_ssp33 (bounded four-case native comparison, active donor chains, rollback and restart passed; reach-scale still unqualified)',
                'bedload_adaptation_plain_ssp33 (bounded two-grain ON/OFF native comparison and bitwise restart passed; broader combinations unqualified)'],
            unported_in_unified_runtime=['bedform/bank compositions outside the exact admitted recursive mode-2/mode-3 routes','cohesive suspended exchange (excluded from the noncohesive goal)',
                'reach-scale active-overlap bank and avalanche cases',
                'viscosity-equilibrated inlet profiles','full Delft3D sediment process coverage'],
            experimental=True)
        if package_scope is not None:
            capabilities['selectable_research_variants']=[entry for entry in capabilities['selectable_research_variants']
                if entry.split(' (',1)[0] not in package_scope['excluded_gpu_variants']
                and entry.split(' (',1)[0] not in (A1_VARIANT,SECONDARY_LIVE_VARIANT,AMR_VARIANT)]
            capabilities['source_worker_package']=package_scope
            capabilities['qualified_engineering_cases_note']=(
                'These are checkout validation records; archive-dependent cases are not shipped in this source worker.')
        print(json.dumps(capabilities,indent=2));return 0
    if args.command=='prepare':
        if (args.gpu_amr_selector_policy is not None and
                args.gpu_preparation_variant not in (INTEGRATED_VARIANT,
                    ON_DEMAND_AMR_VARIANT,ON_DEMAND_TRANSPORT_AMR_VARIANT)):
            parser.error('--gpu-amr-selector-policy requires general AMR preparation')
        if args.evolving_banks and args.gpu_preparation_variant not in (ON_DEMAND_AMR_VARIANT,INTEGRATED_VARIANT):
            parser.error('--evolving-banks requires the on-demand AMR preparation variant')
        if (args.recursive_bank_bedform_avalanche and
                (not args.evolving_banks or
                 args.gpu_preparation_variant not in
                    (ON_DEMAND_AMR_VARIANT,INTEGRATED_VARIANT))):
            parser.error('--recursive-bank-bedform-avalanche requires evolving on-demand bank AMR')
        if args.gpu_preparation_variant!='stock' and args.native_integrator!='ssp43':
            parser.error('Specialized GPU preparation variants own their native integrator')
        if args.gpu_preparation_variant==INTEGRATED_VARIANT:
            if (package_scope is not None and INTEGRATED_VARIANT not in
                    package_scope.get('supported_gpu_variants',())):
                parser.error('GPU source worker does not package integrated preparation')
            from .integrated import prepare as prepare_integrated
            try:
                result=prepare_integrated(args.input,args.output,stop_file=args.stop_file,
                    evolving_banks=args.evolving_banks,
                    **(dict(recursive_bank_bedform_avalanche=True)
                       if args.recursive_bank_bedform_avalanche else {}),
                    **(dict(selector_policy=args.gpu_amr_selector_policy)
                       if args.gpu_amr_selector_policy is not None else {}))
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==ON_DEMAND_AMR_VARIANT:
            if package_scope is not None and ON_DEMAND_AMR_VARIANT not in package_scope.get('supported_gpu_variants',[]):
                parser.error('Source worker does not package the on-demand AMR initializer')
            from .amr_on_demand_runner import prepare as prepare_on_demand
            try:
                result=prepare_on_demand(args.input,args.output,stop_file=args.stop_file,
                    **(dict(evolving_banks=True) if args.evolving_banks else {}),
                    **(dict(recursive_bank_bedform_avalanche=True)
                       if args.recursive_bank_bedform_avalanche else {}),
                    **(dict(selector_policy=args.gpu_amr_selector_policy)
                       if args.gpu_amr_selector_policy is not None else {}))
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==ON_DEMAND_TRANSPORT_AMR_VARIANT:
            if (package_scope is not None and ON_DEMAND_TRANSPORT_AMR_VARIANT not in
                    package_scope.get('supported_gpu_variants',())):
                parser.error('Source worker does not package the general transport AMR initializer')
            from .amr_on_demand_runner import prepare as prepare_on_demand
            try:
                result=prepare_on_demand(args.input,args.output,stop_file=args.stop_file,
                    **(dict(selector_policy=args.gpu_amr_selector_policy)
                       if args.gpu_amr_selector_policy is not None else {}))
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==GENERAL_LIVE_TRANSPORT_VARIANT:
            if (package_scope is not None and GENERAL_LIVE_TRANSPORT_VARIANT not in
                    package_scope.get('supported_gpu_variants',())):
                parser.error('GPU source worker does not package the common transport route')
            from .general_live_transport_runner import prepare as prepare_transport
            try:
                result=prepare_transport(args.input,args.output,stop_file=args.stop_file)
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==GENERAL_LIVE_BANK_VARIANT:
            if (package_scope is not None and GENERAL_LIVE_BANK_VARIANT not in
                    package_scope.get('supported_gpu_variants',())):
                parser.error('GPU source worker does not package the common live-bank route')
            from .general_live_bank_runner import prepare as prepare_general_bank
            try:
                result=prepare_general_bank(args.input,args.output,stop_file=args.stop_file)
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==LIVE_INLET_VARIANT:
            if package_scope is not None:
                parser.error('GPU source worker does not yet package the stage/critical inlet numerical route')
            from .live_inlet_runner import prepare as prepare_live_inlet
            try:
                result=prepare_live_inlet(args.input,args.output,stop_file=args.stop_file)
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==SECONDARY_LIVE_VARIANT:
            if package_scope is not None:
                parser.error('GPU source worker excludes the live-secondary native exporter and sidecar')
            if args.stop_file is not None:
                parser.error('Live-secondary preparation does not support --stop-file')
            print(json.dumps(prepare_live_secondary(args.input,args.output)));return 0
        if args.gpu_preparation_variant==GRADED_BEDLOAD_GENERAL_VARIANT:
            if (package_scope is not None and
                    (GRADED_BEDLOAD_GENERAL_VARIANT not in package_scope['supported_gpu_variants'] or
                     GRADED_BEDLOAD_GENERAL_VARIANT in package_scope['excluded_gpu_variants'])):
                parser.error('GPU source worker excludes the graded-bedload native exporter and sidecar')
            try:
                result=prepare_graded_bedload(args.input,args.output,
                    stop_file=args.stop_file)
            except InterruptedError as exc:
                print(json.dumps(dict(status='cancelled',output=str(args.output),
                    reason=str(exc))))
                return 130
            print(json.dumps(result));return 0
        if args.gpu_preparation_variant==A1_VARIANT:
            if package_scope is not None:
                parser.error('GPU source worker excludes frozen A1 native preparation and sidecars')
            if args.stop_file is not None:
                parser.error('Frozen A1 preparation does not support --stop-file')
            print(json.dumps(prepare_frozen_a1(args.input,args.output)));return 0
        from braids_support.gpu_fwave.coupled_reference import prepare
        try:prepare(args.input,args.output,stop_file=args.stop_file,
                    native_integrator=args.native_integrator)
        except InterruptedError as exc:
            print(json.dumps(dict(status='cancelled',output=str(args.output),reason=str(exc))))
            return 130
        print(json.dumps(dict(status='prepared',output=str(args.output))));return 0
    if package_scope is not None and args.backend=='gpu':
        selected_variant=args.gpu_numerical_variant
        if selected_variant==INTEGRATED_VARIANT:
            from .integrated import resolve_prepared
            selected_variant=resolve_prepared(args.input)
        excluded=package_scope['excluded_gpu_variants']
        if selected_variant in excluded:
            parser.error('GPU source worker excludes '+selected_variant+': '+excluded[selected_variant])
        if selected_variant not in package_scope.get('supported_gpu_variants',()):
            parser.error('GPU source worker does not package '+selected_variant)
        if (args.gpu_numerical_variant==ON_DEMAND_AMR_VARIANT and
                ON_DEMAND_AMR_VARIANT not in package_scope.get('supported_gpu_variants',[])):
            parser.error('Source worker does not package the on-demand AMR numerical route')
        if args.gpu_numerical_variant==LIVE_INLET_VARIANT:
            parser.error('GPU source worker does not yet package the stage/critical inlet numerical route')
        if args.gpu_numerical_variant==AMR_VARIANT:
            parser.error('GPU source worker excludes the archived composed AMR validation fixture and source locks')
        if args.gpu_numerical_variant==A1_VARIANT:
            parser.error('GPU source worker excludes frozen A1 current-source archive and native reference')
        if args.gpu_numerical_variant==SECONDARY_LIVE_VARIANT:
            parser.error('GPU source worker excludes the live-secondary current-source archive and native exporter')
        if args.gpu_numerical_variant==LIVE_SUSPENSION_VARIANT:
            parser.error('GPU source worker excludes the live-suspension current-source archive and native exporter')
        if args.gpu_numerical_variant==G3_VARIANT:
            parser.error('GPU source worker excludes the frozen G3 bank/avalanche/secondary research fixture')
        if args.gpu_numerical_variant in G3_LIVE_BANK_VARIANTS:
            parser.error('GPU source worker excludes the frozen G3 live-bank research fixture')
        for sidecar,reason in package_scope['excluded_prepared_sidecars'].items():
            if (args.input/sidecar).is_file():
                allowed=package_scope.get('allowed_prepared_sidecars_by_variant',{}).get(
                    selected_variant,())
                if (selected_variant in (GENERAL_LIVE_TRANSPORT_VARIANT,
                                         GENERAL_LIVE_BANK_VARIANT) and sidecar in allowed):
                    receipt_path=args.input/'preparation_receipt.json'
                    try:receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                    except (OSError,ValueError):
                        parser.error('GPU integrated sidecar lacks a readable preparation receipt')
                    expected_schema=(6 if selected_variant==GENERAL_LIVE_TRANSPORT_VARIANT else 5)
                    if (receipt.get('schema')!=expected_schema or
                            receipt.get('variant')!=selected_variant or
                            sidecar not in receipt.get('artifacts',{})):
                        parser.error('GPU integrated sidecar lacks the selected preparation contract')
                    continue
                if (sidecar=='normal_metadata.bin' and
                        args.gpu_numerical_variant==GRADED_BEDLOAD_GENERAL_VARIANT and
                        sidecar in allowed):
                    receipt_path=args.input/'preparation_receipt.json'
                    try:receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                    except (OSError,ValueError):
                        parser.error('GPU graded-bedload sidecar lacks a readable preparation receipt')
                    if (receipt.get('schema')!=3 or
                            receipt.get('variant')!=GRADED_BEDLOAD_GENERAL_VARIANT or
                            set(receipt.get('artifacts',{}))!={
                                'coupled_input.bin','coupled_input.npz',
                                'normal_metadata.bin'}):
                        parser.error('GPU graded-bedload sidecar lacks the exact schema-3 contract')
                    continue
                parser.error('GPU source worker excludes prepared '+sidecar+': '+reason)
        if args.gpu_numerical_variant==GRADED_BEDLOAD_GENERAL_VARIANT:
            relative=package_scope.get('source_archives_by_variant',{}).get(
                GRADED_BEDLOAD_GENERAL_VARIANT)
            if relative not in ('braids_gpu/frozen_sources/graded_bedload_v1',
                                'braids_gpu/frozen_sources/graded_bedload_v2',
                                'braids_gpu/frozen_sources/graded_bedload_v3',
                                'braids_gpu/frozen_sources/graded_bedload_v8'):
                parser.error('GPU graded-bedload package has no approved numerical archive')
            bundle=Path(__file__).resolve().parents[1]/relative
            if args.gpu_source_archive is None:
                args.gpu_source_archive=bundle
            elif args.gpu_source_archive.resolve()!=bundle.resolve():
                parser.error('GPU graded-bedload package requires its own numerical archive')
    from .runner import run_cpu,run_gpu
    result=dispatch(args.backend,cpu_runner=run_cpu,gpu_runner=run_gpu,input_path=args.input,output_path=args.output,
        cpu_executable=args.cpu_executable,until=args.until,output_every=args.output_every,dt_max=args.dt_max,
        max_steps=args.max_steps,stop_file=args.stop_file,resume=args.resume,launch_mode=args.gpu_launch_mode,
        numerical_variant=args.gpu_numerical_variant,integrator=args.gpu_integrator,
        capture_step_trace=args.gpu_capture_step_trace,
        source_archive=args.gpu_source_archive)
    if args.backend=='gpu':
        product=publish_qgis_product_result(args.input,args.output)
        if product is not None and isinstance(result,dict):
            result['qgis_product_v109']=product
    print(json.dumps(result,default=str));return 0


if __name__=='__main__':raise SystemExit(main())
