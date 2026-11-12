"""Stage the sole exact-wall implementation, retaining the installed runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
OUT=ROOT/'validation/_delivery172'
TARGET=OUT/'candidate'
FILES=('braids_core/sediment/reflecting_wall.py','validation/sediment/river_morphodynamics/face_activity.py')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')


def main():
    if TARGET.exists():raise FileExistsError(TARGET)
    receipt=json.loads((ROOT/'validation/sediment/river_morphodynamics/reports/desktop_reflection171/MANIFEST.json').read_text())
    assert receipt['qualified'] and sha(BASE/'RUNTIME.json')==receipt['runtime_manifest_sha256']
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():assert sha(BASE/name)==digest,name
    # A complete untouched copy remains reproducible after installed promotion.
    shutil.copytree(BASE,OUT/'preview26_runtime',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from braids_core.sediment import reflecting_wall
    from validation.sediment.river_morphodynamics import face_activity,shoreline_hydraulic_probe as probe
    table=json.loads((BASE/'PRECOMPILED.json').read_text())
    key,row=next((k,v) for k,v in table.items() if k.startswith(probe.__name__+'.build:') and v['native']['file']=='precompiled/d440a06059c179ec44ed/hydraulics.dll')
    source=BASE/Path(row['native']['file']).parent/'shoreline_probe.f90'
    code=reflecting_wall.native_source(source.read_text())
    # Check the real constructor transform, not just a guessed precompiled key.
    parent=table['braids_core.sediment.shoreline_solver.build_native']['native']['file']
    fresh=face_activity.native_transform(probe.native_code((BASE/Path(parent).parent/'shoreline.f90').read_text(),'positive_bounded'))
    assert fresh==code,'Canonical CPU chain differs from qualified pressure-only transform'
    build=OUT/'build'/hashlib.sha256(code.encode()).hexdigest()[:20];build.mkdir(parents=True)
    generated=build/'shoreline_probe.f90';generated.write_text(code)
    library=build/'hydraulics.dll';command=list(row['identity']['command'])
    old=next(x for x in command if x.endswith('shoreline_probe.f90'))
    command[command.index(old)]=str(generated);command[command.index('-o')+1]=str(library)
    result=subprocess.run(command,cwd=build,capture_output=True,text=True,timeout=180)
    (build/'BUILD.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    identity=dict(row['identity'],method=reflecting_wall.METHOD,command=command,
        source_sha256=sha(generated),library_sha256=sha(library),
        reflecting_wall_source_sha256=sha(TARGET/'braids_core/sediment/reflecting_wall.py'),
        production_modified=True,Delft3D_modified=False)
    save(build/'IDENTITY.json',identity)
    package=TARGET/'precompiled'/sha(library)[:20];package.mkdir(parents=True)
    deps=[p for p in source.parent.glob('*.dll') if p.name!='hydraulics.dll']
    for p in [library,generated,*deps]:
        dst=package/p.name;shutil.copy2(p,dst);manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
    newkey=probe.__name__+'.build:'+hashlib.sha256(code.encode()).hexdigest()
    del table[key]
    table[newkey]=dict(native=dict(file=(package/library.name).relative_to(TARGET).as_posix(),sha256=sha(library)),identity=identity)
    save(TARGET/'PRECOMPILED.json',table);manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest)
    save(OUT/'STAGE.json',dict(previous_runtime_sha256=sha(BASE/'RUNTIME.json'),runtime_sha256=sha(TARGET/'RUNTIME.json'),
        canonical_files={name:sha(ROOT/name) for name in FILES},old_binding=key,new_binding=newkey,
        build=identity,installed_changed=False,Delft3D_modified=False))
    print('Staged exact-wall runtime:',sha(TARGET/'RUNTIME.json'),flush=True)


if __name__=='__main__':main()
