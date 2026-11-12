"""Freeze the approved directional limiter over the verified inlet187 payload."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from stage_secondary185 import ROOT,save,sha
BASE=ROOT/'validation/_inlet187/prescribed_candidate'
OUT=ROOT/'validation/_wall188'
TARGET=OUT/'candidate_v3'
FILES=tuple('braids_core/sediment/'+n for n in (
    'wall_bed_hydraulics.py','directional_hydraulic_limiter.py',
    'directional_hydraulic_limiter.f90.inc','directional_hydraulic_limiter.cu.inc'))


def main():
    assert sha(BASE/'RUNTIME.json')=='8b27294c161aec91332a74c89b9de99c9c91cd78a94630369bc0557ae8081015'
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for n,h in manifest['files'].items():assert sha(BASE/n)==h,n
    assert not (OUT/'STAGE.json').exists(),'Sealed candidates are immutable'
    if TARGET.exists():
        assert sha(TARGET/'RUNTIME.json')==sha(BASE/'RUNTIME.json'),'Only an unsealed stage may resume'
    else:shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for n in FILES:
        shutil.copy2(ROOT/n,TARGET/n);manifest['files'][n]=sha(TARGET/n)
    shutil.copy2(ROOT/'desktop/release188/__init__.py',TARGET/'braids_desktop/__init__.py')
    manifest['files']['braids_desktop/__init__.py']=sha(TARGET/'braids_desktop/__init__.py')
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from braids_core.sediment import directional_hydraulic_limiter as limiter
    from validation.sediment.river_morphodynamics.shoreline_hydraulic_probe import native_code
    from validation.sediment.river_morphodynamics.face_activity import native_transform
    from braids_core.sediment.reflecting_wall import FORTRAN_HELPER
    table=json.loads((BASE/'PRECOMPILED.json').read_text())
    command=table['braids_core.sediment.wall_bed_hydraulics.build']['identity']['command']
    dependencies=[s for s in command if s.endswith(('.f','.f90'))][:-1]
    records=[];shore=None
    # Remove the obsolete pre-wall-pressure probe binding; retain its immutable
    # files as evidence, but do not advertise it as an executable alternative.
    del table['validation.sediment.river_morphodynamics.shoreline_hydraulic_probe.build:d86562a4b506cd79bba58f80a155b9f4282e731aa77d698df595385a3d21d3a9']
    keys=[k for k in table if k.startswith(('braids_core.sediment.wall_bed_hydraulics.build',
        'braids_core.sediment.shoreline_solver.build_native','validation.sediment.river_morphodynamics.shoreline_hydraulic_probe.build'))]
    for key in keys:
        row=table.pop(key);source_dir=(BASE/row['native']['file']).parent
        sources=list(source_dir.glob('*.f90'));assert len(sources)==1,sources
        source=sources[0];code=limiter.native_source(source.read_text())
        if 'shoreline_hydraulic_probe' in key:
            assert shore is not None
            current=native_transform(native_code(shore,row['identity']['mode']))
            # The two additions commute numerically, but append their helper
            # routines in opposite order. Require all other source bytes equal.
            assert current.replace(FORTRAN_HELPER,'')==code.replace(FORTRAN_HELPER,'')
            code=current
        folder=OUT/'package_build'/hashlib.sha256(code.encode()).hexdigest()[:20]
        folder.mkdir(parents=True,exist_ok=True)
        generated=folder/source.name
        if generated.exists():assert generated.read_text()==code
        else:generated.write_text(code)
        binary=folder/'hydraulics.dll'
        compile=[command[0],'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
            *dependencies,str(generated),'-o',str(binary),'-Wl,--export-all-symbols']
        if not binary.exists():
            run=subprocess.run(compile,cwd=folder,capture_output=True,text=True,timeout=180)
            (folder/'build.log').write_text(run.stdout+run.stderr)
            assert run.returncode==0,str(folder/'build.log')
        identity=dict(method=limiter.METHOD,parent=row['identity'],generated_sha256=sha(generated),
            library_sha256=sha(binary),source_sha256={n:sha(TARGET/n) for n in FILES},
            command=compile,Delft3D_modified=False)
        save(folder/'identity.json',identity)
        package=TARGET/'precompiled'/sha(binary)[:20];package.mkdir(exist_ok=False)
        for p in (binary,generated,*[p for p in source_dir.glob('*.dll') if p.name!='hydraulics.dll']):
            dst=package/p.name;shutil.copy2(p,dst);manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        if key=='braids_core.sediment.shoreline_solver.build_native':shore=code
        if 'shoreline_hydraulic_probe' in key:
            assert shore is not None
            mode=row['identity']['mode']
            # New source digest selects only the matching compiled probe.
            current=native_transform(native_code(shore,mode))
            assert current==code
            key=key.split(':')[0]+':'+hashlib.sha256(current.encode()).hexdigest()
        table[key]=dict(native=dict(file=(package/binary.name).relative_to(TARGET).as_posix(),sha256=sha(binary)),identity=identity)
        records.append(dict(key=key,source=sha(source),generated=sha(generated),library=sha(binary)))
        print('BUILT',key,flush=True)
    save(TARGET/'PRECOMPILED.json',table);manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest)
    save(OUT/'STAGE.json',dict(baseline=sha(BASE/'RUNTIME.json'),candidate=sha(TARGET/'RUNTIME.json'),files=FILES,native=records))
    print('STAGED',sha(TARGET/'RUNTIME.json'),flush=True)


if __name__=='__main__':main()
