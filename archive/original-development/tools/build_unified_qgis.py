"""Release-only assembly of the frozen unified noncohesive QGIS product.

Record bindings by replaying accepted, strict project inputs. No physics is
added or selected here. The archive contains no inputs/results or Delft3D.
"""
import argparse
import ast
import configparser
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from braids_core.sediment.portable_river_runtime import sha,install,RECORDS,ENGINE_CLASSES


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def payload_files(base):
    """Imports during a staging check must never put bytecode in a release."""
    return [p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts]


def closure(seeds):
    files=set();pending=list(seeds)
    while pending:
        p=Path(pending.pop()).resolve()
        if p in files or not p.is_file() or not p.is_relative_to(ROOT):continue
        files.add(p)
        if p.suffix!='.py':continue
        parts=p.relative_to(ROOT).with_suffix('').parts
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base='.'.join(parts[:-node.level]) if node.level else ''
                base='.'.join(x for x in (base,node.module) if x)
                names=[base]+[base+'.'+a.name for a in node.names]
            for name in names:
                if name.split('.')[0] not in ('braids_core','braids_gpu','braids_qgis','validation','tools'):continue
                child=ROOT.joinpath(*name.split('.'))
                pending.extend([child.with_suffix('.py'),child/'__init__.py'])
        for ext in ('.h','.cu','.cpp','.f90','.inc'):
            pending.append(p.with_suffix(ext))
    return files


def record(evidence,projects):
    previous=json.loads((evidence/'RECORD.json').read_text()) if (evidence/'RECORD.json').exists() else None
    install()
    from braids_core import river_worker
    from braids_core.river_project import load_project
    if previous and previous['source_identity']!=river_worker.source_identity():raise ValueError('Cannot append different sources')
    if previous:RECORDS.update(json.loads((evidence/'BINDINGS.json').read_text()))
    accepted=[] if previous is None else previous['projects']
    offset=len(accepted)
    for i,path in enumerate(projects,offset):
        doc=load_project(path)
        if doc['model']['request']['execution']['backend']!='cpu':continue
        target=evidence/f'record_{i:03d}'
        prep=river_worker.prepare(path,target/'prepared')
        result=river_worker.run(prep,target/'run')
        if result['status']!='complete':raise ValueError(result)
        accepted.append(dict(project=str(path),sha256=sha(path),output=str(target/'run')))
        print('Recorded',path.name,len(RECORDS),'bindings',flush=True)
        # Retain completed release evidence if a later deck fails. Appending
        # still requires the same authenticated source identity above.
        write(evidence/'BINDINGS.json',RECORDS)
        seeds=[str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
               if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')
               and Path(m.__file__).resolve().is_relative_to(ROOT)]
        if previous:seeds=sorted(set(seeds+previous['seeds']))
        write(evidence/'RECORD.json',dict(projects=accepted,seeds=seeds,
              source_identity=river_worker.source_identity()))
    if not accepted:raise ValueError('No current CPU project inputs')
    write(evidence/'BINDINGS.json',RECORDS)
    seeds=[str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and str(m.__file__).endswith('.py')
           and Path(m.__file__).resolve().is_relative_to(ROOT)]
    if previous:seeds=sorted(set(seeds+previous['seeds']))
    write(evidence/'RECORD.json',dict(projects=accepted,seeds=seeds,
          source_identity=river_worker.source_identity()))


def merge_bindings(table, extra):
    """Merge recordings without replacing a previously qualified native binding."""
    def signature(row):
        result={k:v for k,v in row.items() if k not in ('libraries','native','library','identity','receipt')}
        result['libraries']={k:sha(p) for k,p in row.get('libraries',{}).items()}
        for key in ('native','library'):
            if key in row:result[key]=sha(row[key])
        return result
    merged=dict(table)
    for key,row in extra.items():
        if key in merged and signature(merged[key])!=signature(row):
            raise ValueError('Conflicting recorded native binding: '+key)
        if key not in merged:merged[key]=row
    return merged


def package(evidence,destination,supplements=(),python_updates_file=None):
    if destination.exists():raise FileExistsError(destination)
    from braids_core import river_worker
    recorded=json.loads((evidence/'RECORD.json').read_text())
    release_identity=river_worker.source_identity();loader_change=None;python_updates=[]
    if recorded['source_identity']!=release_identity:
        # Retain the recorded native bindings only for explicitly reviewed,
        # exact Python updates. Authenticate both endpoints and the complete
        # old source tree; any other change, including any kernel, fails closed.
        from unittest.mock import patch
        python_updates=json.loads((python_updates_file or ROOT/'tools/unified_python_updates.json').read_text())['updates']
        previous={}
        for row in python_updates:
            path=(ROOT/row['path']).resolve()
            if not path.is_relative_to(ROOT) or path.suffix!='.py' or sha(path)!=row['after']:
                raise ValueError('Unqualified Python update: '+row['path'])
            previous[path]=row['before']
        with patch.object(river_worker,'sha',lambda p:previous.get(Path(p).resolve()) or sha(p)):
            if recorded['source_identity']!=river_worker.source_identity():raise ValueError('Sources changed since recording')
        loader_change=python_updates[0]
    table=json.loads((evidence/'BINDINGS.json').read_text())
    supplemental_records=[];supplemental_seeds=[]
    for folder in supplements:
        rec=json.loads((folder/'RECORD.json').read_text())
        if rec['source_identity']!=release_identity:
            raise ValueError('Supplement must be recorded against current sources')
        table=merge_bindings(table,json.loads((folder/'BINDINGS.json').read_text()))
        supplemental_seeds.extend(rec['seeds'])
        supplemental_records.append(dict(directory=str(folder),source_identity=rec['source_identity'],
            record_sha256=sha(folder/'RECORD.json'),bindings_sha256=sha(folder/'BINDINGS.json'),
            projects=rec['projects']))
    plugin=destination/'braids_qgis';runtime=plugin/'river_runtime'
    def copy(source,rel,base=runtime):
        source=Path(source);target=base/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(source)!=sha(target):raise ValueError('Conflicting payload '+str(target))
        if not target.exists():shutil.copy2(source,target)
        return dict(file=target.relative_to(base).as_posix(),sha256=sha(target))
    deps=('libgcc_s_seh-1.dll','libgfortran-5.dll','libgomp-1.dll','libquadmath-0.dll','libwinpthread-1.dll','libdl.dll')
    def library(path,native=False):
        source=Path(path);folder=Path('precompiled')/sha(source)[:20]
        row=copy(source,folder/source.name)
        for name in deps:copy(Path('C:/Strawberry/c/bin')/name,folder/name)
        if native:
            for child in source.parent.glob('*.f90'):copy(child,folder/child.name)
        return row
    for row in table.values():
        for attr,path in list(row.get('libraries',{}).items()):row['libraries'][attr]=library(path)
        if 'native' in row:row['native']=library(row['native'],True)
        if 'library' in row:row['library']=library(row['library'])
    seeds=[Path(p) for p in recorded['seeds']+supplemental_seeds if not Path(p).is_relative_to(ROOT/'tests')]
    seeds += [ROOT/'braids_core/river_launcher.py',ROOT/'braids_core/river_worker.py',
              ROOT/'braids_core/river_folder.py',
              ROOT/'braids_core/boundaries/axis_aligned.py']
    # Dynamically archived GPU modules must be present even on a CPU-only build host.
    for folder in ('braids_gpu','validation/gpu_fwave'):
        seeds+=list((ROOT/folder).glob('*.py'))
    files=closure(seeds)
    for folder in ('braids_core/sediment','braids_gpu','validation/gpu_fwave',
                   'validation/sediment/river_morphodynamics','solver/wave'):
        files.update(p for p in (ROOT/folder).iterdir() if p.is_file() and
                     p.suffix in ('.h','.cu','.inc','.cpp','.f90','.f'))
    files.update((ROOT/'braids_core/sediment').glob('*.json'))
    files.add(ROOT/'braids_qgis/core/raster_surface.py')
    for p in sorted(files):
        rel=p.relative_to(ROOT)
        if any(s in ('runs','reports','resources','tests','build','dist') for s in rel.parts):continue
        if rel.parts[0]=='braids_qgis' and rel.as_posix()!='braids_qgis/core/raster_surface.py':continue
        copy(p,rel)
        for parent in rel.parents:
            init=ROOT/parent/'__init__.py'
            if str(parent)!='.' and parent.parts[0]!='braids_qgis' and init.is_file():copy(init,parent/'__init__.py')
    for p in (ROOT/'braids_gpu/native_exporter_v1').iterdir():
        if p.is_file():copy(p,p.relative_to(ROOT))
    for name in deps:copy(Path('C:/Strawberry/c/bin')/name,Path('braids_gpu/native_exporter_v1')/name)
    # Only the new public UI is shipped, not the retired dock/workspace.
    for p in closure([ROOT/'braids_qgis/__init__.py',ROOT/'braids_qgis/plugin.py']):
        if p.is_relative_to(ROOT/'braids_qgis'):copy(p,p.relative_to(ROOT/'braids_qgis'),plugin)
    for name in ('metadata.txt','BRAIDS_logo.png','FIRST_RELEASE.md','SURVEYED_BANKS.md','MODEL_SETUP.md'):
        copy(ROOT/'braids_qgis'/name,name,plugin)
    for p in (ROOT/'dist/windows-x86_64/braids_qgis/licenses').rglob('*'):
        if p.is_file():copy(p,Path('licenses')/p.relative_to(ROOT/'dist/windows-x86_64/braids_qgis/licenses'),plugin)
    write(runtime/'PRECOMPILED.json',table)
    write(runtime/'RUNTIME.json',dict(schema=1,route='unified_river_v1',
          compiler_required=False,cohesive_sediment=False,Delft3D_included=False,
          files={p.relative_to(runtime).as_posix():sha(p) for p in payload_files(runtime)}))
    config=configparser.ConfigParser();config.read(plugin/'metadata.txt')
    write(plugin/'MANIFEST.json',dict(schema='braids.unified-qgis-release.v1',
          version=config['general']['version'],source_identity=release_identity,
          recorded_source_identity=recorded['source_identity'],loader_only_update=loader_change,python_only_updates=python_updates,
          supplemental_records=supplemental_records,
          files={p.relative_to(plugin).as_posix():sha(p) for p in payload_files(plugin)}))
    verify(plugin)
    archive=destination/f"BRAIDS-{config['general']['version']}-QGIS-windows-x86_64.zip"
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload_files(plugin)):
            z.write(p,p.relative_to(destination))
    write(evidence/'PACKAGE.json',dict(plugin=str(plugin),archive=str(archive),sha256=sha(archive)))
    print(archive,flush=True)


def verify(plugin):
    manifest=json.loads((plugin/'MANIFEST.json').read_text())
    if manifest.get('schema')!='braids.unified-qgis-release.v1':raise ValueError('Not a unified release')
    files={p.relative_to(plugin).as_posix() for p in plugin.rglob('*') if p.is_file()
           and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
    if files!=set(manifest['files']):raise ValueError('Payload inventory differs')
    for rel,expected in manifest['files'].items():
        p=(plugin/rel).resolve()
        if not p.is_relative_to(plugin.resolve()) or sha(p)!=expected:raise ValueError('Payload differs '+rel)
    for banned in ('gui/dock.py','gui/adaptive_river.py','river_runtime/resources','river_runtime/braids_core/sediment/river_fines.py'):
        if (plugin/banned).exists():raise ValueError('Retired/research path in release '+banned)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('record','package','verify'))
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--destination',type=Path)
    parser.add_argument('--project-folder',type=Path,action='append',default=[])
    parser.add_argument('--project',type=Path,action='append',default=[])
    parser.add_argument('--supplement',type=Path,action='append',default=[])
    parser.add_argument('--python-updates',type=Path,help='Exact authenticated Python-only update manifest for this release')
    args=parser.parse_args()
    if args.action=='record':
        paths=list(args.project)
        for folder in args.project_folder:paths.extend(sorted(folder.glob('cpu_*.project.json')))
        record(args.evidence.resolve(),paths)
    elif args.action=='package':package(args.evidence.resolve(),args.destination.resolve(),[p.resolve() for p in args.supplement],args.python_updates)
    else:print(verify(args.destination.resolve())['version'])
