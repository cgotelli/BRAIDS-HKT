"""Freeze alpha81, build guarded CPU defaults, and retain an MF25 validation copy.

This builder never installs or edits production. Promotion uses the generated,
reviewed source through apply_patch, followed by the normal release builder.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools import build_cutcell as build
from . import run_case as r
from . import fast_cpu_default_transform as transform
from . import mf25_validation_runtime as mf25

TAG = 'fast_cpu_default_20261001_v1'
ROOT = r.HERE/'runs'/TAG
REPORT = r.HERE/'reports'/TAG


def checkpoint_contract(text):
    """Append a resolved-method tag; old-format restarts explicitly use legacy.

    The state prefix stays readable by analysis tooling. A multistep restart
    rebuilds history with SSP43; this is logged and is not byte-exact continuity.
    """
    replace = transform.p.replace
    text = replace(text, ' character(64) :: restart_fingerprint',
        " character(8) :: restart_integrator_magic\n"
        " integer :: restart_integrator=33\n"
        ' character(64) :: restart_fingerprint')
    marker = "  close(unit)\n  else if(amr_enabled) then"
    text = replace(text, marker, """  read(unit,iostat=ios) restart_integrator_magic
  if(ios<0) then
   ! Pre-adoption checkpoint: never silently switch its integrator.
   restart_integrator=33
  else
   if(ios/=0.or.restart_integrator_magic/='BRDINT01') error stop 'invalid checkpoint integrator trailer'
   read(unit,iostat=ios) restart_integrator
   if(ios/=0.or.(restart_integrator/=33.and.restart_integrator/=43.and.restart_integrator/=6)) &
    error stop 'invalid checkpoint time integrator'
  endif
""" + marker)
    marker = ' call resolve_cpu_time_integrator(len_trim(restart)>0)'
    text = replace(text, marker, marker + """
 if(len_trim(restart)>0.and.restart_integrator/=cpu_time_integrator) then
  error stop 'restart time integrator differs; resume using its original BRAIDS_TIME_INTEGRATOR'
 endif""")
    text = replace(text, '  close(checkpoint_unit)',
        "  write(checkpoint_unit) 'BRDINT01',cpu_time_integrator\n  close(checkpoint_unit)")
    return text


def source(original):
    return checkpoint_contract(transform.transform(original))


def freeze():
    ROOT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    work = ROOT/'runtime_legacy'; work.mkdir()
    manifest = json.loads((build.BUILD/'build.json').read_text())
    assert r.sha256(build.BUILD/'xbraids_cutcell.exe') == manifest['executable_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert r.sha256(r.ROOT/name) == digest, name
        shutil.copy2(r.ROOT/name, work/Path(name).name)
    shutil.copy2(build.BUILD/'xbraids_cutcell.exe', work/'xbraids_candidate.exe')
    shutil.copy2(build.BUILD/'build.json', work/'build.json')
    shutil.copy2(build.BUILD/'build.log', work/'build.log')
    r.write_json(REPORT/'production_before.json', manifest)
    r.write_json(REPORT/'plan.json', dict(
        default='auto: SSPMS6 eligible hydraulics; SSP43 eligible fixed-grid MPM; legacy otherwise',
        existing_CFL_preserved=True, cold_comparison_CFL=.8,
        original_reference_wall_s=22570.255672,
        bed_limits=dict(rms_m=.003, max_m=.005, every_shared_frame=True),
        production_MF_limit=20, validation_constant_MF_limit=25,
        delft3d_source_modified=False, gpu_not_production=True))
    print('FROZEN '+str(work), flush=True)


def build_candidate(variant):
    assert variant in ('production', 'mf25')
    old, parent = runtime('legacy')
    work = ROOT/('runtime_'+variant); work.mkdir(exist_ok=False)
    hashes = {}
    for name in build.SOURCES:
        src = old.parent/Path(name).name; target = work/src.name
        if name == build.SOURCES[-1]:
            value = source(src.read_text())
            if variant == 'mf25': value = mf25.native_source(value)
            target.write_text(value)
        else: shutil.copy2(src, target)
        hashes[name] = r.sha256(target)
    exe = work/'xbraids_candidate.exe'
    command = [shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe',
        '-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
        *[str(work/Path(s).name) for s in build.SOURCES], '-o',str(exe)]
    started = time.perf_counter()
    result = subprocess.run(command,cwd=work,text=True,capture_output=True,timeout=180)
    (work/'build.log').write_text(result.stdout+result.stderr); result.check_returncode()
    record = dict(variant=variant, source_sha256=hashes,
        source_text_sha256={s:build.text_digest(work/Path(s).name) for s in build.SOURCES},
        executable_sha256=r.sha256(exe), command=command,
        parent_executable_sha256=parent['executable_sha256'],
        compile_wall_s=time.perf_counter()-started, constant_mf_limit=25 if variant=='mf25' else 20,
        compiler=subprocess.check_output([command[0],'--version'],text=True).splitlines()[0],
        scripts={f.name:r.sha256(f) for f in (Path(__file__),Path(transform.__file__),
            r.HERE/'native/fast_cpu_integrators.inc')},
        checkpoint_method_trailer=True, cfl_override=False, delft3d_modified=False)
    r.write_json(work/'build.json',record)
    print('BUILT '+str(exe),flush=True)


def runtime(variant='mf25'):
    folder=ROOT/('runtime_'+variant)
    identity=json.loads((folder/'build.json').read_text())
    exe=folder/'xbraids_candidate.exe'
    assert r.sha256(exe)==identity['executable_sha256']
    for name,value in identity['source_sha256'].items():
        assert r.sha256(folder/Path(name).name)==value,name
    return exe,identity


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','production','mf25'))
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:build_candidate(args.action)
