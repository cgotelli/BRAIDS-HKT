"""Small unchanged-native wet/dry general-avalanche trajectory for GPU I4.

Creates new validation decks and executes a copied, hash-locked native binary.
It never patches or rebuilds native or Delft3D source. The non-avalanche twin
shares precisely the same initial state, mesh and hydraulics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from braids_core.boundaries.cut_cell_io import CutSedimentFraction, write_cut_case
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_qgis.core.results import read_frames


HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=HERE/'runs/i4_general_avalanche_native_20261003_v1'
SOURCE_EXECUTABLE=HERE/'runs/cpu_filter_convergence_control_20261002_v1/runtime/xbraids_filter_tight.exe'
CASES=('avalanche','control')
FRACTIONS=(
    CutSedimentFraction(fraction_id='fine',d50_m=.0002,d90_m=.0003,
        settling_velocity_m_s=0.,initial_volume_fraction=.5,
        initial_active_volume_fraction=.5),
    CutSedimentFraction(fraction_id='coarse',d50_m=.001,d90_m=.0015,
        settling_velocity_m_s=0.,initial_volume_fraction=.5,
        initial_active_volume_fraction=.5),
)
BULK=np.array([grain.bulk_kg_m3 for grain in FRACTIONS])
END_S=.05
DT_MAX_S=.005
OUTPUT_S=.01
DRY_TOLERANCE_M=.00025
WET_DEPTH_SWITCH_M=.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def file_map(folder):
    return {item.name:sha(item) for item in sorted(Path(folder).iterdir()) if item.is_file()}


def geometry_and_initial():
    domain=PolygonDomain('i4_general_avalanche',[(0,0),(4,0),(4,2),(0,2)],'EPSG:2056')
    mesh=map_cut_cells(domain,[],NativeGrid(4,2,0,0,1,1))
    require(len(mesh.cells)==8,'Expected eight full cells')
    xy=np.array([cell.centroid_xy for cell in mesh.cells],dtype=float)
    area=np.array([cell.area_m2 for cell in mesh.cells],dtype=float)
    state=np.zeros((8,9),dtype=float)
    substrate_by_row=np.array([[.02,.08,.34,.02],[.03,.06,.29,.04]])
    depth_by_row=np.array([[.10,.02,0.,.06],[.07,.01,0.,.08]])
    for cell,(x,y) in enumerate(xy):
        column=int(np.floor(x));row=int(np.floor(y))
        active_fine=.20+.15*column+.03*row
        substrate_fine=.75-.12*column+.02*row
        active=np.array([active_fine,1-active_fine])*.04
        substrate=np.array([substrate_fine,1-substrate_fine])*substrate_by_row[row,column]
        state[cell,0]=depth_by_row[row,column]
        state[cell,5:7]=active*BULK
        state[cell,7:9]=substrate*BULK
    require(np.count_nonzero(state[:,0]==0)==2 and
            np.count_nonzero(state[:,0]>WET_DEPTH_SWITCH_M)>=3 and
            np.count_nonzero((state[:,0]>0)&(state[:,0]<WET_DEPTH_SWITCH_M))>=2,
            'Fixture must span dry/shallow/deep avalanche mobility')
    require(np.ptp(state[:,5]/BULK[0])>0 and
            np.ptp(state[:,7]/BULK[0])>0,
            'Grain composition must differ across cells and layers')
    return mesh,xy,area,state


def prepare(output=DEFAULT_ROOT):
    output=Path(output).resolve()
    require(not output.exists(),'Avalanche native output already exists')
    require(SOURCE_EXECUTABLE.is_file(),'Frozen unchanged-native executable is missing')
    output.mkdir(parents=True)
    runtime=output/'runtime';runtime.mkdir()
    executable=runtime/'xbraids_avalanche_reference.exe'
    shutil.copy2(SOURCE_EXECUTABLE,executable)
    mesh,xy,area,state=geometry_and_initial()
    for case in CASES:
        folder=output/'inputs'/case;folder.mkdir(parents=True)
        metadata=write_cut_case(folder/'case.data',mesh,state,np.zeros(8),{},
            end_time_s=END_S,dt_max_s=DT_MAX_S,output_interval_s=OUTPUT_S,
            cfl=.4,scheme=6,bedload=0,exchange=0,manning=0.,
            sediment_fractions=FRACTIONS,active_layer_thickness_m=.04,
            dry_tolerance_m=DRY_TOLERANCE_M,
            general_avalanche=case=='avalanche',
            general_avalanche_wet_slope=.08,
            general_avalanche_dry_slope=.16,
            general_avalanche_time_s=.02,
            general_avalanche_wet_depth_switch_m=WET_DEPTH_SWITCH_M,
            general_avalanche_relaxation_length_m=1.,
            source_sha256={'i4_case_script':sha(__file__)})
        np.savez_compressed(folder/'initial_analysis.npz',xy=xy,area=area,state=state,
            bulk=BULK,rock=np.zeros(8))
        require(bool(metadata['general_avalanche']['enabled'])==(case=='avalanche'),
                'Native writer did not encode requested avalanche switch')
    plan=dict(schema=1,tag=output.name,
        native_executable=str(executable),native_executable_sha256=sha(executable),
        copied_from=str(SOURCE_EXECUTABLE.resolve()),
        copied_from_sha256=sha(SOURCE_EXECUTABLE),
        source_sha256={'case_script':sha(__file__)},
        controls=dict(end_s=END_S,dtmax_s=DT_MAX_S,output_s=OUTPUT_S,cfl=.4,
            scheme=6,bedlaw=0,exchange=0,morfac=1.,dry_tolerance_m=DRY_TOLERANCE_M,
            wet_depth_switch_m=WET_DEPTH_SWITCH_M),
        inputs={case:file_map(output/'inputs'/case) for case in CASES},
        Delft3D_source_modified=False,GPU_solver_executed=False)
    save(output/'plan.json',plan)
    return dict(plan=str(output/'plan.json'),plan_sha256=sha(output/'plan.json'),
                executable_sha256=sha(executable))


def verify_plan(output):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text())
    require(plan['schema']==1 and plan['tag']==output.name and
            plan['native_executable_sha256']==sha(plan['native_executable']) and
            plan['source_sha256']['case_script']==sha(__file__),
            'Native avalanche plan/source/runtime changed')
    for case in CASES:
        require(file_map(output/'inputs'/case)==plan['inputs'][case],
                'Native avalanche input changed: '+case)
    return plan


def execute(case,output=DEFAULT_ROOT):
    require(case in CASES,'Unknown avalanche native case')
    output=Path(output).resolve();plan=verify_plan(output)
    folder=output/case/'cpu'
    require(not folder.exists(),'Native avalanche execution already exists')
    folder.mkdir(parents=True)
    for source in (output/'inputs'/case).iterdir():
        if source.is_file():shutil.copy2(source,folder/source.name)
    environment={key:value for key,value in os.environ.items()
        if not key.upper().startswith(('BRAIDS_','OMP_'))}
    environment.update(BRAIDS_TIME_INTEGRATOR='legacy',OMP_NUM_THREADS='1',
        OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
    command=[plan['native_executable'],'case.data']
    completed=subprocess.run(command,cwd=folder,capture_output=True,text=True,
        timeout=120,env=environment)
    (folder/'stdout.txt').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    result=dict(status='completed' if completed.returncode==0 else 'failed',
        return_code=completed.returncode,command=command,
        native_executable_sha256=sha(plan['native_executable']),
        plan_sha256=sha(output/'plan.json'),
        environment={key:environment[key] for key in
            ('BRAIDS_TIME_INTEGRATOR','OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY')},
        output_files=file_map(folder))
    save(folder/'execution.json',result)
    require(completed.returncode==0,'Native avalanche execution failed: '+completed.stderr[-2000:])
    return result


def analyze(output=DEFAULT_ROOT):
    output=Path(output).resolve();plan=verify_plan(output)
    values={}
    for case in CASES:
        folder=output/case/'cpu'
        execution=json.loads((folder/'execution.json').read_text())
        require(execution['return_code']==0 and
                execution['plan_sha256']==sha(output/'plan.json') and
                execution['native_executable_sha256']==plan['native_executable_sha256'],
                'Native avalanche execution receipt differs')
        for name,digest in execution['output_files'].items():
            require(sha(folder/name)==digest,'Native avalanche output changed: '+name)
        frames=read_frames(folder/'frames.txt')
        state=np.stack([row['q'] for row in frames])
        times=np.array([row['time_s'] for row in frames])
        initial=np.load(output/'inputs'/case/'initial_analysis.npz',allow_pickle=False)
        area=initial['area'];bulk=initial['bulk']
        class_mass=np.sum((state[:,:,5:7]+state[:,:,7:9])*area[None,:,None],axis=1)
        water=np.sum(state[:,:,0]*area[None,:],axis=1)
        bed=np.sum(state[:,:,5:7]/bulk[None,None,:]+state[:,:,7:9]/bulk[None,None,:],axis=2)
        values[case]=dict(times=times,state=state,class_mass=class_mass,water=water,bed=bed,
            initial_state=initial['state'],area=area)
    on,off=values['avalanche'],values['control']
    require(on['state'].shape==(6,8,9) and off['state'].shape==on['state'].shape and
            np.allclose(on['times'],np.arange(6)*OUTPUT_S,rtol=0,atol=1e-10) and
            np.array_equal(on['initial_state'],off['initial_state']),
            'Avalanche ON/OFF trajectory shape/clocks/initial state differ')
    dry=on['state'][0,:,0]<=DRY_TOLERANCE_M
    on_off_bed=np.abs(on['bed']-off['bed'])
    wetdry_on=on['state'][:,:,0]<=DRY_TOLERANCE_M
    wetdry_off=off['state'][:,:,0]<=DRY_TOLERANCE_M
    class_movement=np.max(np.abs(on['state'][-1,:,5:9]-off['state'][-1,:,5:9]),axis=0)
    checks=dict(six_finite_frames=np.isfinite(on['state']).all() and
                np.isfinite(off['state']).all(),
        all_nonnegative=np.min(on['state'][:,:,0])>=0 and
                        np.min(on['state'][:,:,5:9])>=0,
        unchanged_twin_initial=np.array_equal(on['state'][0],off['state'][0]),
        wet_dry_present_at_all_frames=bool(np.all(np.any(wetdry_on,axis=1)) and
                                           np.all(np.any(~wetdry_on,axis=1))),
        wet_dry_classification_matches_control=np.array_equal(wetdry_on,wetdry_off),
        dry_donor_moves=bool(np.max(on_off_bed[-1,dry])>.02),
        wet_recipient_moves=bool(np.max(on_off_bed[-1,~dry])>.02),
        both_grains_and_layers_move=bool(np.all(class_movement>.01)),
        active_avalanche_trajectory=bool(np.max(on_off_bed[-1])>.05 and
                                         np.max(on_off_bed[1])>.01),
        per_class_conservative=bool(np.max(np.abs(on['class_mass']-on['class_mass'][0]))<1e-9),
        water_conservative=bool(np.max(np.abs(on['water']-on['water'][0]))<1e-9))
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(schema=1,plan_sha256=sha(output/'plan.json'),
        native_executable_sha256=plan['native_executable_sha256'],
        passed=all(checks.values()),checks=checks,
        frame_count=len(on['times']),frame_times_s=on['times'].tolist(),
        accepted_steps_by_frame=[int(row['steps']) for row in
            read_frames(output/'avalanche/cpu/frames.txt')],
        initial_dry_cells=int(np.count_nonzero(on['state'][0,:,0]<=DRY_TOLERANCE_M)),
        final_dry_cells=int(np.count_nonzero(on['state'][-1,:,0]<=DRY_TOLERANCE_M)),
        maximum_avalanche_on_off_bed_difference_m=float(np.max(np.abs(on['bed']-off['bed']))),
        maximum_dry_donor_bed_difference_m=float(np.max(on_off_bed[-1,dry])),
        maximum_wet_recipient_bed_difference_m=float(np.max(on_off_bed[-1,~dry])),
        maximum_bed_difference_by_frame_m=np.max(on_off_bed,axis=1).tolist(),
        final_layer_class_movement_kg_m2=class_movement.tolist(),
        maximum_avalanche_on_off_layer_difference_kg_m2=float(np.max(np.abs(
            on['state'][:,:,5:9]-off['state'][:,:,5:9]))),
        maximum_class_mass_residual_kg=float(np.max(np.abs(on['class_mass']-on['class_mass'][0]))),
        maximum_water_residual_m3=float(np.max(np.abs(on['water']-on['water'][0]))),
        minimum_depth_m=float(np.min(on['state'][:,:,0])),
        minimum_layer_mass_kg_m2=float(np.min(on['state'][:,:,5:9])),
        avalanche_frame_sha256=sha(output/'avalanche/cpu/frames.txt'),
        control_frame_sha256=sha(output/'control/cpu/frames.txt'))
    save(output/'analysis.json',result)
    require(result['passed'],'Native avalanche wet/dry activation or conservation prerequisite failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','execute','analyze'))
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.output)
    elif args.action=='execute':
        require(args.case is not None,'--case required for execute')
        result=execute(args.case,args.output)
    else:result=analyze(args.output)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
