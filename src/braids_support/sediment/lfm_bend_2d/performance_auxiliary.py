"""Sediment/secondary-flow performance only; all hydraulic routines preserved."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from . import run_case as r
from . import optimize_performance as old
from . import performance_methods as methods
from . import profile_current as audit

TAG='performance_auxiliary_20260930_v1'
BASE=methods.BASE
PREVIOUS=r.HERE/'runs/performance_methods_20260930_v2'
PARALLEL=r.HERE/'native/secondary_curvature_parallel.f90'


def sediment_source(text):
    source=(r.ROOT/'solver/cut_cells/cutcell_solver.f90').read_text()
    text=audit.replace_once(text,' real(8), allocatable :: face_displacement(:,:,:)',
        ' real(8), allocatable :: face_displacement(:,:,:)\n'
        ' real(8), allocatable :: bed_ls_delta(:,:),bed_ls_weight(:),bed_ls_matrix(:,:)')
    for name in ('bed_downslopes','bedload_capacity_gradients'):
        text=audit.replace_once(text,methods.routine(text,name),methods.routine(source,name))
    text=audit.replace_once(text,' subroutine cache_face_geometry()\n  integer :: f',
        ' subroutine cache_face_geometry()\n  integer :: f\n  call cache_sediment_geometry()')
    text=audit.replace_once(text,' subroutine cache_one_face(c,f,side)',
        methods.routine(source,'cache_sediment_geometry')+'\n\n subroutine cache_one_face(c,f,side)')
    return text


def hydraulic_guard(candidate, baseline):
    checks={}
    for name in ('gradients','prepare_wall_traces','prepare_wet_dry_traces','wave_flux',
                 'traces','euler','prepare_hydraulic_stage'):
        checks[name]=methods.routine(candidate,name)==methods.routine(baseline,name)
    for name in ('face_value','dynamic_runup_opens_face'):
        import re
        pattern=rf'(?ms)^ .*?function {name}\(.*?^ end function'
        # Limit start to one line; do not include preceding routines.
        pattern=rf'(?ms)^ [^\n]*function {name}\(.*?^ end function'
        checks[name]=re.search(pattern,candidate)[0]==re.search(pattern,baseline)[0]
    if not all(checks.values()):raise ValueError(f'Hydrodynamic source changed: {checks}')
    return checks


def build(root,variant):
    runtime=root/('runtime_'+variant);runtime.mkdir(parents=True,exist_ok=False)
    prior=json.loads((BASE/'runtime/build.json').read_text())
    command=[prior['command'][0],'-O3','-flto=8','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none']
    hashes={};guard={}
    for name,expected in prior['source_sha256'].items():
        source=BASE/'runtime'/Path(name).name
        assert r.sha256(source)==expected
        text=source.read_text()
        if source.name=='cutcell_solver.f90':
            original=text;text=sediment_source(text);guard=hydraulic_guard(text,original)
        elif source.name=='secondary_curvature_regularization.f90':
            text=(PARALLEL if variant=='parallel' else PARALLEL.with_name('secondary_curvature_fast.f90')).read_text()
        target=runtime/source.name;target.write_text(text);hashes[name]=r.sha256(target);command.append(str(target))
    exe=runtime/'xbraids_candidate.exe';command+=['-o',str(exe)]
    begin=time.perf_counter();proc=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=180)
    (runtime/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    record=dict(source_sha256=hashes,executable_sha256=r.sha256(exe),command=command,
        depth_factor=2.,constant_mf_limit=25,series_mf_limit=20,variant=variant,
        compile_wall_s=time.perf_counter()-begin,hydrodynamic_routines_unchanged=guard,
        reference=str(BASE),relative_tolerance=1e-13,delft_modified=False)
    r.write_json(runtime/'build.json',record);print('BUILT '+variant,flush=True)


def exe(root,name):
    if name=='baseline':runtime=BASE/'runtime'
    else:runtime=root/('runtime_'+name)
    identity=json.loads((runtime/'build.json').read_text());path=runtime/'xbraids_candidate.exe'
    assert r.sha256(path)==identity['executable_sha256']
    for source,expected in identity['source_sha256'].items():assert r.sha256(runtime/Path(source).name)==expected
    return path


CASES=[('fine',audit.SOURCE,2880,.5,audit.SOURCE/'checkpoint.bin',8),
       ('large',BASE/'large_input',0,.1,None,16),
       ('larger',PREVIOUS/'larger_input',0,.025,None,16)]


def run(tag,stage):
    root=r.HERE/'runs'/tag;report=r.HERE/'reports'/tag
    if stage=='build':
        root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
        r.write_json(report/'plan.json',dict(hydrodynamics='Preserve GeoClaw reconstruction, fluxes and integration byte-for-byte at routine level',
            scope=['sediment geometry','secondary-flow filter numerical solver'],limits=methods.LIMITS,
            parallel_modes=['serial MIC','parallel matrix/vector work with serial MIC','parallel block MIC'],
            policy='Choose parallel work when measured faster; retain serial only with paired evidence',delft_modified=False))
        for variant in ('serial','parallel'):build(root,variant)
        return
    if stage=='screen':
        rows=[];checks={}
        for size,source,start,duration,checkpoint,threads in CASES:
            for name,mode in [('serial','serial'),('parallel','serial'),('parallel','matvec'),('parallel','block')]:
                label=f'{size}_{name}_{mode}'
                rows.append(old.execute(root,label,exe(root,name),source,threads,duration,start,checkpoint,
                                        extra_env={'BRAIDS_CURVATURE_MODE':mode}))
                if label!=f'{size}_serial_serial':
                    checks[label]=methods.compare(root/f'{size}_serial_serial',root/label)
                r.write_json(report/'screen.json',dict(rows=rows,checks=checks))
                assert all(x['passed'] for x in checks.values())
        return
    if stage=='benchmark':
        rows=[];checks={}
        for size,source,start,duration,checkpoint,threads in CASES:
            for repeat,order in enumerate((('baseline','serial'),('serial','baseline')),1):
                for name in order:
                    label=f'{size}_r{repeat}_{name}'
                    rows.append(old.execute(root,label,exe(root,name),source,threads,2*duration,start,checkpoint))
                checks[f'{size}_r{repeat}']=methods.compare(root/f'{size}_r{repeat}_baseline',root/f'{size}_r{repeat}_serial')
                r.write_json(report/'benchmark.json',dict(rows=rows,checks=checks,
                    scope='No competing braids_support/compilation jobs during paired timings; hydraulic routines unchanged'))
                assert all(x['passed'] for x in checks.values())
        return
    if stage=='regression':
        row=old.execute(root,'long_serial',exe(root,'serial'),audit.SOURCE,8,24,2880,audit.SOURCE/'checkpoint.bin')
        check=methods.compare(BASE/'long_optimized',root/'long_serial')
        r.write_json(report/'regression.json',dict(row=row,check=check,reference=str(BASE/'long_optimized')))
        assert check['passed'],check
        rows=[]
        for label,threads,duration,start,checkpoint in [
            ('large_t1',1,.2,0,None),('large_split_a',16,.1,0,None),
            ('large_split_b',16,.1,.1,root/'large_split_a/checkpoint.bin')]:
            rows.append(old.execute(root,label,exe(root,'serial'),BASE/'large_input',threads,duration,start,checkpoint))
        sha=r.sha256(root/'large_r1_serial/checkpoint.bin')
        flags={label:r.sha256(root/label/'checkpoint.bin')==sha for label in ('large_t1','large_split_b')}
        r.write_json(report/'reproducibility.json',dict(rows=rows,checks=flags))
        assert all(flags.values())
        return
    if stage=='outlet_audit':
        import sys
        import numpy as np
        sys.path.insert(0,str(r.ROOT/'tests'))
        from test_stage_outlet_wetting import run_stage
        rows=[];arrays=[]
        for name in ('baseline','serial'):
            folder=root/('outlet_'+name)
            q=run_stage(folder,stage=.02,executable=exe(root,name))[-1]
            arrays.append(q)
            hc=((.2+2*np.sqrt(9.81*.1))/3)**2/9.81
            rows.append(dict(variant=name,executable_sha256=r.sha256(exe(root,name)),folder=str(folder),
                measured_discharge_m3_s=float(-q['ledger'][0,0]/q['time_s']),
                expected_discharge_m3_s=float(2*hc*np.sqrt(9.81*hc))))
        identical=all(np.array_equal(arrays[0][k],arrays[1][k]) for k in ('q','ledger'))
        r.write_json(report/'known_failure.json',dict(test='test_resolved_free_outfall_matches_characteristic_critical_flux[0.02]',
            preexisting=True,baseline_and_candidate_identical=identical,rows=rows,
            action='Recorded without changing the hydraulic outlet or loosening the test'))
        assert identical
        return
    raise ValueError(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['build','screen','benchmark','regression','outlet_audit'])
    p.add_argument('--tag',default=TAG);args=p.parse_args();run(args.tag,args.stage)
