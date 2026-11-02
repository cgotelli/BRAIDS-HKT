"""Isolated constant-MF25 input allowance; production's <=20 guard remains.

Only validation input guards change, not any time integration or sediment law.
Piecewise factor-series range is deliberately not widened by this experiment.
"""
import inspect
import shutil
import subprocess
from . import run_case as r
from . import regularized_mobile as mobile
from braids_core.boundaries import cut_cell_io


def native_source(source):
    changes={
        'morphological_factor>20.d0':'morphological_factor>25.d0',
        "error stop 'morphological factor must be finite and within the qualified 1 to 20 range'":
            "error stop 'constant morphological factor outside experimental validation range 1 to 25'"}
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('MF input guard source context changed')
        source=source.replace(old,new)
    return source


def validation_writer(runtime):
    source=inspect.getsource(cut_cell_io.write_cut_case)
    changes={'not 1<=morphological_factor<=20':'not 1<=morphological_factor<=25',
        "raise ValueError('Morphological factor must be within the qualified 1 to 20 range')":
        "raise ValueError('Constant morphological factor outside experimental validation range 1 to 25')"}
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('Writer MF guard context changed')
        source=source.replace(old,new)
    target=runtime/'validation_writer.py';r.write(target,source)
    # Retain the original dataclass/type identities; no mutation of its globals.
    namespace=dict(vars(cut_cell_io))
    exec(compile(source,str(target),'exec'),namespace)
    return namespace['write_cut_case']


def build(tag):
    parent=r.HERE/'runs/fine_regularized_v1_runtime'
    original_exe,original=mobile.locked_candidate(parent)
    runtime=r.HERE/'runs'/f'{tag}_runtime';runtime.mkdir(parents=True,exist_ok=False)
    source=(parent/'cutcell_solver.f90').read_text()
    target=runtime/'cutcell_solver.f90';r.write(target,native_source(source))
    shutil.copy2(parent/'secondary_curvature_regularization.f90',runtime/'secondary_curvature_regularization.f90')
    names=list(original['source_sha256'])
    paths=[runtime/name.split('/')[-1] if name.split('/')[-1]=='cutcell_solver.f90' else
           runtime/'secondary_curvature_regularization.f90' if r.Path(name).name=='secondary_curvature_regularization.f90'
           else r.ROOT/name for name in names]
    hashes={name:r.sha256(path) for name,path in zip(names,paths)}
    exe=runtime/'xbraids_candidate.exe'
    command=[shutil.which('gfortran'),'-O3','-flto','-fopenmp','-ffree-line-length-none',
             '-ffixed-line-length-none',*[str(p) for p in paths],'-o',str(exe)]
    result=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=120)
    r.write(runtime/'build.log',result.stdout+result.stderr);result.check_returncode()
    if hashes!={name:r.sha256(path) for name,path in zip(names,paths)}:raise RuntimeError('Sources changed during build')
    record=dict(depth_factor=2.,source_sha256=hashes,executable_sha256=r.sha256(exe),command=command,
        original_executable_sha256=r.sha256(original_exe),original_source_sha256=original['source_sha256'],
        role='Validation-only constant MF <=25; arithmetic unchanged from ell=2h MF1 candidate',
        constant_mf_limit=25,series_mf_limit=20,production_guard_changed=False)
    r.write_json(runtime/'build.json',record)
    return exe,record,validation_writer(runtime)
