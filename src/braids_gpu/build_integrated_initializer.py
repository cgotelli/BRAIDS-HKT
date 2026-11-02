"""Build a versioned I/O-only initializer; keep all frozen references intact.

Only a conditional avalanche-control sidecar is added to the pinned a50f
initializer. Removing the inserted text must recover the parent source exactly.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .atomic_output import write_json_atomic
from .graded_native_exporter import EXPORTER_SHA256

ROOT=Path(__file__).resolve().parents[1]
DESTINATION=ROOT/'braids_gpu/native_exporter_v2'
CONTROLS=('general_avalanche_wet_slope','general_avalanche_dry_slope',
          'general_avalanche_time','general_avalanche_wet_depth_switch',
          'general_avalanche_relaxation_length')
SHA=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extra_export():
    text="\n subroutine integrated_export_avalanche_metadata()\n  integer :: gpu_unit\n"
    text+="  call gpu_open(gpu_unit,'avalanche_metadata.bin')\n"
    for name in ('nc','nf','nfrac','nbed_layers','deck_version'):
        text+=f"  call gpu_put_i(gpu_unit,'{name}',[{name}],[integer::])\n"
    text+="  call gpu_put_i(gpu_unit,'general_avalanche',[merge(1,0,general_avalanche)],[integer::])\n"
    for name in CONTROLS:
        text+=f"  call gpu_put_r(gpu_unit,'{name}',[{name}],[integer::])\n"
    return text+'  call gpu_finish(gpu_unit)\n end subroutine\n'


def transform(source):
    marker='end module\n\nprogram run_cutcell'
    anchor="  call normal_export_metadata()\n  stop 'GPU native initialization exported; no solver timestep performed'"
    inserted='  if(general_avalanche) call integrated_export_avalanche_metadata()\n'
    if source.count(marker)!=1 or source.count(anchor)!=1:
        raise ValueError('Pinned initializer I/O anchors differ')
    output=source.replace(marker,extra_export()+marker,1).replace(anchor,
        anchor.replace('  stop ',inserted+'  stop ',1),1)
    if output.replace(extra_export(),'',1).replace(inserted,'',1)!=source:
        raise ValueError('Initializer transformation changed more than sidecar I/O')
    return output


def build():
    prior=ROOT/'braids_gpu/native_exporter_v1/build.json'
    identity=json.loads(prior.read_text())
    parent_exe=Path(identity['executable'])
    if SHA(parent_exe)!=EXPORTER_SHA256 or SHA(identity['command'][0])!=identity['compiler_sha256']:
        raise ValueError('Pinned initializer/compiler identity differs')
    source_args=[Path(arg) for arg in identity['command'] if Path(arg).suffix.lower() in ('.f','.f90')]
    for source in source_args:
        if SHA(source)!=identity['private_files'].get(str(source)):
            raise ValueError('Frozen initializer source differs: '+str(source))
    DESTINATION.mkdir(parents=True,exist_ok=False)
    replacements={}
    parents={}
    for source in source_args:
        target=DESTINATION/source.name
        parents[str(source)]=SHA(source)
        if source.name=='cutcell_solver.f90':
            target.write_text(transform(source.read_text(encoding='utf-8')),encoding='utf-8')
        else:
            shutil.copy2(source,target)
        replacements[str(source)]=str(target)
    executable=DESTINATION/'xbraids_integrated_export.exe'
    replacements[str(parent_exe)]=str(executable)
    command=[replacements.get(arg,arg) for arg in identity['command']]
    with (DESTINATION/'build.log').open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=DESTINATION,stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=300)
    # Existing compiler-side DLL closure is unchanged. Packaging later copies
    # the redistributable dependency closure, not the compiler installation.
    for source,expected in parents.items():
        if SHA(source)!=expected:raise ValueError('Frozen parent changed during build')
    receipt=dict(schema=2,command=command,executable=str(executable),
        executable_sha256=SHA(executable),parent_executable_sha256=EXPORTER_SHA256,
        parent_sources=parents,source_sha256={p.name:SHA(DESTINATION/p.name) for p in source_args},
        compiler_sha256=identity['compiler_sha256'],driver_sha256=SHA(__file__),
        original_build_sha256=SHA(prior),conditional_avalanche_metadata=True,
        numerical_source_unchanged=True,native_solver_steps_performed=0,
        Delft3D_source_modified=False)
    write_json_atomic(DESTINATION/'build.json',receipt)
    print(json.dumps(dict(executable=str(executable),sha256=receipt['executable_sha256'])))


if __name__=='__main__':build()
