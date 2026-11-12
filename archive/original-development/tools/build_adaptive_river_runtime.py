"""Build a compiler-free combined river runtime from measured CPU bindings.

Only source, kernels, precompiled BRAIDS binaries and notices enter the payload.
No case, checkpoint, validation result or Delft3D source/runtime is collected.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core.sediment.portable_river_runtime import sha,install,RECORDS

def source_closure(seeds):
    files=set();pending=list(seeds)
    while pending:
        p=Path(pending.pop()).resolve()
        if p in files or not p.is_file() or not p.is_relative_to(ROOT):continue
        files.add(p)
        if p.suffix!='.py':continue
        parts=p.relative_to(ROOT).with_suffix('').parts
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base='.'.join(parts[:-node.level]) if node.level else ''
                base='.'.join(x for x in (base,node.module) if x)
                candidates=[base]+[base+'.'+a.name for a in node.names]
            for module in candidates:
                if module.split('.')[0] not in ('braids_core','braids_gpu','validation','tools'):continue
                child=ROOT.joinpath(*module.split('.')).with_suffix('.py')
                if child.is_file():pending.append(child)
        for ext in ('.h','.cu','.cpp','.f90','.inc'):
            companion=p.with_suffix(ext)
            if companion.is_file():pending.append(companion)
    return files

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);parser.add_argument('--evidence',required=True)
    args=parser.parse_args();dst=Path(args.destination).resolve();evidence=Path(args.evidence).resolve()
    if dst.exists():raise FileExistsError(dst)
    evidence.mkdir(parents=True,exist_ok=True)
    install()
    from braids_core.sediment import adaptive_river_worker as worker
    from braids_core.sediment.river_field_policy import AutomaticRiver,FieldSettings
    from braids_core.sediment.river_leaf_controller import RiverLeafController
    old=ROOT/'validation/sediment/river_morphodynamics/reports/adaptive_worker_v1/input_cropped_automatic'
    from validation.sediment.river_morphodynamics.worker_seed import read_seed
    p,a=read_seed(old);t=worker.hierarchy(p)
    prep=worker.prepare(p,a,evidence/'prepared')
    for backend in ('cpu','gpu'):
        result=worker.run(prep,evidence/backend,backend=backend)
        print(backend,result,flush=True)
    # The supported bed switch changes generated code; include both exact kernels.
    p['suspension']['bed_factor']=0.
    cache=worker.cache_for(evidence/'zero_bed_cache',p,a,t)
    make=worker.factory('cpu',p,t,cache,evidence/'zero_bed_owner')
    leaves=tuple(range(t.offsets[-2],t.count));zero=make(leaves,t.select(leaves));zero.step(.0005)
    table=json.loads(json.dumps(RECORDS));dst.mkdir(parents=True)
    def copy(source,target):
        source=Path(source).resolve();target=dst/target;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(source)!=sha(target):raise ValueError('Duplicate destination '+str(target))
        else:shutil.copy2(source,target)
        return {'file':target.relative_to(dst).as_posix(),'sha256':sha(target)}
    deps=('libgcc_s_seh-1.dll','libgfortran-5.dll','libgomp-1.dll','libquadmath-0.dll','libwinpthread-1.dll','libdl.dll')
    def dll(source,native=False):
        source=Path(source);folder=Path('precompiled')/sha(source)[:20] if native else Path('precompiled/engines')
        name=source.name if native else sha(source)[:20]+'_'+source.name
        result=copy(source,folder/name)
        for dep in deps:copy(Path('C:/Strawberry/c/bin')/dep,folder/dep)
        if native:
            for child in source.parent.glob('*.f90'):copy(child,folder/child.name)
        return result
    for row in table.values():
        for attr,path in list(row.get('libraries',{}).items()):row['libraries'][attr]=dll(path)
        if 'native' in row:row['native']=dll(row['native'],True)
        if 'library' in row:row['library']=dll(row['library'])
    seeds=[Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(getattr(m,'__file__','')).endswith('.py')]
    seeds+=[ROOT/'braids_core/sediment/adaptive_river_worker.py',ROOT/'braids_core/sediment/adaptive_river_launcher.py',ROOT/'braids_core/sediment/adaptive_river_gis.py']
    # prepare_sources deliberately snapshots these exact current GPU sources.
    for folder in ('braids_gpu','validation/gpu_fwave'):
        seeds+=list((ROOT/folder).glob('*.py'))
    files=source_closure(seeds)
    # Kernel/header includes and executable catalogues have no input/run data.
    for folder in ('braids_core/sediment','braids_gpu','validation/gpu_fwave','validation/sediment/river_morphodynamics'):
        for pth in (ROOT/folder).iterdir():
            if pth.is_file() and pth.suffix in ('.h','.cu','.inc','.cpp','.f90'):files.add(pth)
    for pth in (ROOT/'braids_core/sediment').glob('*.json'):files.add(pth)
    for pth in (ROOT/'solver/wave').iterdir():
        if pth.is_file() and pth.suffix in ('.h','.cpp','.f90','.f','.inc'):files.add(pth)
    files.add(ROOT/'braids_qgis/core/raster_surface.py')
    for pth in files:
        rel=pth.relative_to(ROOT)
        if any(part in ('runs','reports','resources','tests') for part in rel.parts):continue
        copy(pth,rel)
        for parent in rel.parents:
            if str(parent)=='.':continue
            init=ROOT/parent/'__init__.py'
            if init.is_file() and parent.parts[0]!='braids_qgis':copy(init,parent/'__init__.py')
    for pth in (ROOT/'braids_gpu/native_exporter_v1').iterdir():
        if pth.is_file():copy(pth,pth.relative_to(ROOT))
    for dep in deps:copy(Path('C:/Strawberry/c/bin')/dep,Path('braids_gpu/native_exporter_v1')/dep)
    for pth in (ROOT/'dist/windows-x86_64/braids_qgis/licenses').rglob('*'):
        if pth.is_file():copy(pth,Path('licenses')/pth.relative_to(ROOT/'dist/windows-x86_64/braids_qgis/licenses'))
    (dst/'PRECOMPILED.json').write_text(json.dumps(table,indent=2),encoding='utf-8')
    from braids_core.sediment.river_midpoint import METHOD
    manifest={'schema':1,'route':'combined_adaptive_river_v1','integrator':METHOD,
        'compiler_required':False,'source_checkout_required':False,'Delft3D_included':False,
        'files':{p.relative_to(dst).as_posix():sha(p) for p in dst.rglob('*') if p.is_file()}}
    (dst/'RUNTIME.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Packaged',len(manifest['files']),'files;',len(table),'native bindings',flush=True)

if __name__=='__main__':main()
