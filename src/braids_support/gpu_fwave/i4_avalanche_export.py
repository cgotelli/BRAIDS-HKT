"""I/O-only native export of the frozen I4 general-avalanche validation deck.

This adds only a sidecar writer to a *copy* of the frozen native CPU source.
The parent parser, geometry, avalanche and hydraulic routines are unchanged.
The resulting zero-step snapshot is for validation; GPU coupling is separate.
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

from . import coupled_reference as cr
from . import cpu_filter_cold20h as parent
from . import i4_avalanche_native_case as native


HERE=Path(__file__).resolve().parent
ROOT=HERE/'runs/i4_general_avalanche_prepared_20261003_v2'
RUNTIME=ROOT/'runtime'
PARENT_PLAN_SHA='0bb40fad5c9f14e61fdd7fe23076604d9bee764059629472848068d494e26cc3'
PARENT_EXE_SHA='21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5'
CONTROLS=('general_avalanche_wet_slope','general_avalanche_dry_slope',
          'general_avalanche_time','general_avalanche_wet_depth_switch',
          'general_avalanche_relaxation_length')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def extra_export():
    text="\n subroutine i4_export_avalanche_metadata()\n  integer :: gpu_unit\n"
    text+="  call gpu_open(gpu_unit,'avalanche_metadata.bin')\n"
    for name in ('nc','nf','nfrac','nbed_layers','deck_version'):
        text+=cr.rec(name,kind='i')
    text+=cr.rec('general_avalanche','merge(1,0,general_avalanche)',kind='i')
    for name in CONTROLS:text+=cr.rec(name)
    text+='  call gpu_finish(gpu_unit)\n end subroutine\n'
    return text


def transform(source):
    changed=cr.instrument(source)
    anchor='  call gpu_export_input()\n  stop '
    marker='end module\n\nprogram run_cutcell'
    require(changed.count(anchor)==1 and changed.count(marker)==1,
            'Unexpected native zero-step exporter anchors')
    changed=changed.replace(anchor,
        '  call gpu_export_input()\n  call i4_export_avalanche_metadata()\n  stop ',1)
    return changed.replace(marker,extra_export()+marker,1)


def verify_transform(source,changed):
    restored=changed.replace(extra_export(),'',1).replace(
        '  call i4_export_avalanche_metadata()\n','',1)
    require(changed==transform(source) and restored==cr.instrument(source),
            'I4 exporter changed more than the declared sidecar I/O')
    for name in ('load_case','avalanche_erodible_banks','advance_ssp33'):
        start=source.index('subroutine '+name+'(')
        end=source.index('end subroutine',start)
        require(changed.count(source[start:end])==1,
                'Native numerical/parser body changed: '+name)
    return True


def build(runtime=RUNTIME):
    runtime=Path(runtime).resolve()
    require(not runtime.exists(),'I4 export runtime already exists')
    require(sha(native.DEFAULT_ROOT/'plan.json')==PARENT_PLAN_SHA and
            sha(native.SOURCE_EXECUTABLE)==PARENT_EXE_SHA,
            'Frozen native I4 plan/runtime differs')
    identity,protected=parent.runtime_identity()
    require(parent.TIGHT_SHA==PARENT_EXE_SHA,
            'I4 exporter parent must be the frozen native executable')
    runtime.mkdir(parents=True)
    sources=[]
    for name in identity['source_sha256']:
        original=parent.TIGHT_RUNTIME/name;target=runtime/name
        if name=='cutcell_solver.f90':
            source=original.read_text(encoding='utf-8')
            changed=transform(source)
            verify_transform(source,changed)
            target.write_text(changed,encoding='utf-8')
        else:shutil.copy2(original,target)
        sources.append(str(target))
    for path in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(path,runtime/path.name)
    flags=identity['command'][1:6]
    require(flags==['-O3','-flto','-fopenmp','-ffree-line-length-none',
                    '-ffixed-line-length-none'],'Frozen native compiler flags differ')
    compiler=Path(identity['command'][0]);executable=runtime/'xbraids_avalanche_export.exe'
    command=[str(compiler),*flags,*sources,'-o',str(executable)]
    process=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=300)
    (runtime/'build.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    require(process.returncode==0,'I4 I/O-only exporter compilation failed: '+process.stderr[-2000:])
    record=dict(schema=1,command=command,executable=str(executable),
        executable_sha256=sha(executable),parent_executable_sha256=parent.TIGHT_SHA,
        parent_protected_files=protected,
        source_sha256={str(Path(path).name):sha(path) for path in sources},
        compiler_sha256=sha(compiler),driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        transformation='cr.instrument initialization I/O plus I4 avalanche controls sidecar only',
        solver_steps_performed=0,numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    save(runtime/'build.json',record)
    return verify_build(runtime)


def verify_build(runtime=RUNTIME):
    runtime=Path(runtime).resolve();record=json.loads((runtime/'build.json').read_text())
    parent.runtime_identity()
    require(record['schema']==1 and
            record['parent_executable_sha256']==parent.TIGHT_SHA==PARENT_EXE_SHA and
            record['driver_sha256']==sha(__file__) and
            record['coupled_reference_sha256']==sha(cr.__file__) and
            record['compiler_sha256']==sha(record['command'][0]) and
            record['executable_sha256']==sha(record['executable']),
            'I4 export runtime or dependencies changed')
    for name,digest in record['source_sha256'].items():
        require(sha(runtime/name)==digest,'I4 export source changed: '+name)
    for path,digest in record['parent_protected_files'].items():
        require(sha(path)==digest,'I4 original native source/runtime changed: '+path)
    verify_transform((parent.TIGHT_RUNTIME/'cutcell_solver.f90').read_text(encoding='utf-8'),
        (runtime/'cutcell_solver.f90').read_text(encoding='utf-8'))
    return record


def read_prepared(folder,case):
    require(case in native.CASES,'Unknown frozen I4 avalanche case')
    folder=Path(folder).resolve()
    require(not (folder/'normal_metadata.bin').exists() and
            (folder/'coupled_input.bin').is_file() and
            (folder/'avalanche_metadata.bin').is_file(),
            'I4 prepared sidecars absent or mode9 sidecar unexpectedly present')
    data=cr.read_snapshot(folder)
    extra=cr.read_records(folder/'avalanche_metadata.bin')
    require(set(extra)==set(CONTROLS)|{'nc','nf','nfrac','nbed_layers',
                                     'deck_version','general_avalanche'},
            'I4 avalanche sidecar schema differs')
    for name,value in extra.items():
        require(name not in data or np.array_equal(data[name],value),
                'I4 duplicate native avalanche value differs: '+name)
        data[name]=value
    expected=(dict(general_avalanche_wet_slope=.08,
        general_avalanche_dry_slope=.16,general_avalanche_time=.02,
        general_avalanche_wet_depth_switch=native.WET_DEPTH_SWITCH_M,
        general_avalanche_relaxation_length=1.) if case=='avalanche' else
        dict(general_avalanche_wet_slope=10.,general_avalanche_dry_slope=1.,
            general_avalanche_time=86400.,general_avalanche_wet_depth_switch=.1,
            general_avalanche_relaxation_length=1.))
    require(int(data['nc'])==8 and int(data['nfrac'])==2 and
            int(data['nbed_layers'])==2 and int(data['nb'])==0 and
            int(data['scheme'])==6 and int(data['bedlaw'])==0 and
            int(data['exchange_mode'])==0 and int(data['amr_enabled'])==0 and
            int(data['general_avalanche'])==int(case=='avalanche') and
            int(data['bank_avalanche'])==0 and int(data['bank_failure'])==0 and
            all(float(data[key])==value for key,value in expected.items()) and
            float(data['dry_tolerance'])==native.DRY_TOLERANCE_M and
            float(data['morphological_factor'])==1.,
            'I4 prepared numerical/avalanche scope differs')
    original=np.load(native.DEFAULT_ROOT/'inputs'/case/'initial_analysis.npz',allow_pickle=False)
    require(np.array_equal(data['state'].T,original['state']) and
            np.array_equal(data['xy'].T,original['xy']) and
            np.array_equal(data['area'],original['area']) and
            np.array_equal(data['rock'],original['rock']) and
            np.array_equal(data['bulk_fraction'],original['bulk']) and
            np.array_equal(data['layer_mass'],original['state'][:,5:9].T.reshape(2,2,8)),
            'I4 prepared geometry/layer state differs from frozen native initial state')
    require(np.count_nonzero(data['q'][0]<=native.DRY_TOLERANCE_M)==2 and
            np.all(np.isfinite(data['state'])) and
            np.min(data['layer_mass'])>=0,
            'I4 prepared wet/dry or layer inventory differs')
    return data


def prepare(case,root=ROOT,runtime=RUNTIME):
    require(case in native.CASES,'Unknown frozen I4 avalanche case')
    root=Path(root).resolve();built=verify_build(runtime)
    native.verify_plan(native.DEFAULT_ROOT)
    require(sha(native.DEFAULT_ROOT/'plan.json')==PARENT_PLAN_SHA,
            'Frozen I4 native plan changed')
    source=native.DEFAULT_ROOT/'inputs'/case/'case.data';destination=root/case
    require(not destination.exists(),'I4 prepared case already exists')
    origin=cr.copy_inputs(source,destination)
    environment={key:value for key,value in os.environ.items()
        if not key.upper().startswith(('BRAIDS_','OMP_'))}
    environment.update(BRAIDS_GPU_INITIAL_EXPORT='1',BRAIDS_TIME_INTEGRATOR='legacy',
        OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
    environment['PATH']=str(Path(runtime).resolve())+os.pathsep+environment.get('PATH','')
    process=subprocess.run([built['executable'],source.name],cwd=destination,
        env=environment,capture_output=True,text=True,timeout=60)
    (destination/'initial.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    require(process.returncode==0,'I4 zero-step export failed: '+process.stderr[-2000:])
    require(not (destination/'frames.txt').exists() and
            not (destination/'checkpoint.bin').exists(),
            'I4 zero-step export unexpectedly evolved native state')
    data=read_prepared(destination,case)
    artifacts={name:sha(destination/name) for name in
        ('coupled_input.bin','avalanche_metadata.bin')}
    receipt=dict(schema=1,status='completed',case=case,origin=origin,
        native_plan_sha256=PARENT_PLAN_SHA,
        parent_native_executable_sha256=PARENT_EXE_SHA,
        exporter_build_sha256=sha(Path(runtime)/'build.json'),
        exporter_executable_sha256=built['executable_sha256'],
        exporter_driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        solver_steps_performed=0,initialization_CPU_only=True,
        no_mode9_adapter_sidecar=True,
        controls={key:float(data[key]) for key in CONTROLS},
        geometry=dict(cells=int(data['nc']),faces=int(data['nf']),
            fractions=int(data['nfrac']),layers=int(data['nbed_layers']),
            initially_dry_cells=int(np.count_nonzero(data['q'][0]<=native.DRY_TOLERANCE_M))),
        artifacts=artifacts,numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    save(destination/'preparation_receipt.json',receipt)
    return dict(folder=str(destination),receipt_sha256=sha(destination/'preparation_receipt.json'),
        artifacts=artifacts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build','prepare','verify'))
    parser.add_argument('--case',choices=native.CASES)
    args=parser.parse_args()
    if args.action=='build':result=build()
    else:
        require(args.case is not None,'--case is required')
        if args.action=='prepare':result=prepare(args.case)
        else:
            folder=ROOT/args.case
            result=dict(case=args.case,receipt_sha256=sha(folder/'preparation_receipt.json'),
                cells=read_prepared(folder,args.case)['nc'])
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
