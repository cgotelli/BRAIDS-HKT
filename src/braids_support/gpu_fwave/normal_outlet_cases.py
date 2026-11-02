"""New live-normal inlet / characteristic-critical outlet transient fixtures.

The only physical change from normal_coupled_oracle.fixture is stage outlet
mode2 -> critical outlet mode4. Existing sampled inlet, forcing, initial state,
materials and prognostic secondary flow are retained. Preparation is host-only;
the separate explicit initializer requires a reviewed execution plan.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from . import normal_coupled_oracle as normal

HERE=Path(__file__).resolve().parent
ROOT=HERE/'runs'/'normal_outlet_cases_20261003_v1'
NAMES=('equilibrium','prescribed')
FROZEN_NORMAL_SHA='281739098167c395fd16ed51edaf574197a5bbfdcb19845d6bed749f2cc43aab'
sha=normal.sha


def verify_dependencies():
    sources={str(Path(normal.__file__).resolve()):FROZEN_NORMAL_SHA}
    normal.verify_files(sources);return sources


def fixture(name):
    verify_dependencies()
    if name not in NAMES:raise ValueError('Unknown normal/critical-outlet fixture')
    from braids_core.boundaries.cut_cell_io import CutBoundary
    data=normal.fixture(name)
    data['boundaries']=dict(data['boundaries'],out=CutBoundary('critical'))
    return data


def prepare_fixture(name,folder):
    from braids_core.boundaries import cut_cell_io as writer
    dependencies=verify_dependencies();data=fixture(name)
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    sources={str(Path(__file__).resolve()):sha(__file__),str(Path(writer.__file__).resolve()):sha(writer.__file__),**dependencies}
    writer.write_cut_case(folder/'case.data',data['mesh'],data['q'],data['rock'],data['boundaries'],
        sections={'in':data['section']},end_time_s=.04,dt_max_s=.001,output_interval_s=.005,
        cfl=.4,scheme=6,bedload=2,exchange=0,manning=.03,mpm_friction_mode=1,mpm_critical_shields=.047,
        sediment_fractions=data['fractions'],active_layer_thickness_m=.05,substrate_layer_count=1,
        substrate_layer_thickness_m=.05,secondary_flow=True,secondary_flow_model='prognostic',
        secondary_flow_transport_direction=True,secondary_flow_momentum_beta=1.,morphological_factor=1.,
        compact_sidecars=True,source_sha256={'fixture':sha(__file__),'writer':sha(writer.__file__),
            'normal_geometry_initial_forcing':FROZEN_NORMAL_SHA})
    np.savez(folder/'initial_analysis.npz',q=data['q'],rock=data['rock'],bed=data['bed'],xy=data['xy'],
        area=np.array([c.area_m2 for c in data['mesh'].cells]))
    row=dict(schema=1,name=name,input=str(folder/'case.data'),cells=128,inlet_faces=8,outlet_faces=8,
        duration=.04,dt=.001,dt_max=.001,cfl=.4,morphological_factor=1.,
        output_times_s=(np.arange(9)*.005).tolist(),hydraulic_knots_s=[0.,.02,.04],
        sediment_knots_s=[0.,.02,.04] if name=='prescribed' else [],interior_forcing_events_s=[.02],
        inlet='Unchanged sampled live normal mode9/control1, Manning .03 and energy slope .002; Q .08/.09/.08 m3/s',
        outlet='Current-owner characteristic critical mode4, no imposed Q or stage',
        bed_is_mobile=True,section_control=1,
        sediment='Two identical sand labels; .05m active plus .05m substrate; MPM2, exchange0',
        secondary='Prognostic model2, beta1 momentum coupling and transport direction retained; initial hI0',
        initial_state='Exactly the original normal fixture; trapezoidal bed, stage .14-.002*x, u=.35 and nonuniform transverse velocity; not steady normal/critical equilibrium',
        changes_from_normal=['Only stage outlet mode2 -> characteristic critical outlet mode4'],
        retained_feed='Original two per-label prescribed feeds' if name=='prescribed' else 'Original equilibrium-capacity feed',
        native_evolution_executable=str(normal.parent.TIGHT_EXE),native_evolution_executable_sha256=normal.parent.TIGHT_SHA,
        export_runtime=str(normal.RUNTIME),metadata_reader='normal_outlet_cases.read_case',
        native_or_GPU_execution_performed=False,scope='Short coupled port fixture, not experiment, steady flow, long-term morphology or performance proof',
        source_files=sources,files={str(p):sha(p) for p in folder.iterdir() if p.is_file()})
    normal.save(folder/'fixture.json',row);normal.verify_files(sources);return row


def read_case(folder):
    """Unchanged exporter records; explicit combined mode9/1 and mode4 scope."""
    verify_dependencies();folder=Path(folder)
    data=normal.cr.read_snapshot(folder);extra=normal.cr.read_records(folder/'normal_metadata.bin')
    normal.schedules.validate_arrays(extra)
    for key,value in extra.items():
        if key in data and not np.array_equal(data[key],value):raise ValueError('Duplicate export differs: '+key)
        data[key]=value
    nb=int(data['nb']);offsets=np.asarray(data['boundary_offset']);faces=np.asarray(data['boundary_faces'])
    if nb!=2 or offsets.dtype!=np.int32 or offsets.shape!=(3,) or offsets[0]!=0 or offsets[-1]!=len(faces) or np.any(np.diff(offsets)!=8):
        raise ValueError('Two eight-face boundary partitions required')
    mode=np.asarray(data['mode']);control=np.asarray(data['section_control'])
    if mode.dtype!=np.int32 or mode.shape!=(nb,) or sorted(mode.tolist())!=[4,9]:raise ValueError('Exactly mode9 inlet and mode4 outlet required')
    if control.dtype!=np.int32 or control.shape!=(nb,) or not np.array_equal(control,(mode==9).astype(np.int32)):
        raise ValueError('Only normal control1 on mode9 allowed')
    if not bool(data['production_section_geometry']):raise ValueError('Production section geometry required')
    for key in ('section_seed_node','boundary_seed_position'):
        raw=np.asarray(data['native_'+key])
        if raw.dtype!=np.int32 or raw.shape!=(nb,) or np.any(raw<0) or np.any(raw[mode!=9]):raise ValueError('Invalid native seed metadata')
        data[key]=np.ascontiguousarray(raw-1)
    b=int(np.flatnonzero(mode==9)[0]);count=offsets[b+1]-offsets[b]
    if not 1<=data['section_seed_node'][b]<count:raise ValueError('Interior normal seed required')
    if not offsets[b]<=data['boundary_seed_position'][b]<offsets[b+1]:raise ValueError('Boundary seed outside inlet CSR partition')
    expected=np.zeros_like(data['sampled']);expected[faces[offsets[b]:offsets[b+1]]]=1
    if not np.array_equal(data['sampled'],expected):raise ValueError('Only every normal inlet face must be sampled')
    return data


def prepare_case(deck,folder,runtime=normal.RUNTIME):
    """Explicit zero-step export, no rebuild or native evolution."""
    verify_dependencies();built=normal.verify_build(runtime);folder=Path(folder).resolve()
    origin=normal.cr.copy_inputs(deck,folder);command=[built['executable'],Path(deck).name]
    process=subprocess.run(command,cwd=folder,env=normal.environment(True,runtime),capture_output=True,text=True,timeout=60)
    (folder/'initial.log').write_text(process.stdout+process.stderr)
    receipt=dict(schema=1,status='failed',return_code=process.returncode,command=command,origin=origin,
        parent_executable_sha256=normal.parent.TIGHT_SHA,export_runtime_sha256=built['executable_sha256'],
        build=str(Path(runtime).resolve()/'build.json'),build_sha256=sha(Path(runtime)/'build.json'),
        helper_sha256=sha(__file__),solver_steps_performed=0,initialization_CPU_only=True,
        inlet_mode=9,outlet_mode=4,section_control=1,native_parser_sorting_seed_checks_unchanged=True,
        source_files={str(Path(__file__).resolve()):sha(__file__),**verify_dependencies()},
        metadata_conversion='Native local section node / global boundary CSR position:1based ->0based; absent0 ->-1; raw arrays retained',
        real_native_schedule_tables=list(normal.schedules.TABLES))
    try:
        process.check_returncode();data=read_case(folder)
        if int(data['cpu_time_integrator'])!=43 or int(data['cpu_outlet_extension'])!=1:raise ValueError('Expected unchanged CPU21b7 initialized controls')
        np.savez_compressed(folder/'coupled_input.npz',**data);normal.verify_build(runtime);receipt['status']='completed'
    except Exception as exc:receipt['error']=repr(exc);raise
    finally:
        receipt['artifacts']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        normal.save(folder/'preparation_receipt.json',receipt)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=ROOT/'inputs');args=p.parse_args()
    print(json.dumps([prepare_fixture(n,args.folder/n) for n in NAMES],indent=2))


if __name__=='__main__':main()
