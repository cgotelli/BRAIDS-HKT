"""Build an isolated compiler-free runtime for the JC odd/even correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
OUT=ROOT/'validation/_checkerboard227'
TARGET=OUT/'candidate_v3'
SOURCES=(
    'braids_core/sediment/jiang_cao_coupled.py',
    'braids_core/sediment/jiang_cao_coupled.h',
    'braids_core/sediment/jiang_cao_exchange.py',
    'braids_core/sediment/jiang_cao_roughness.py',
    'braids_core/sediment/jiang_cao_wetdry.py',
)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        if sha(root/name)!=digest:raise ValueError(f'Runtime payload differs: {name}')
    return manifest


def main():
    base=verify(BASE)
    if TARGET.exists():raise FileExistsError(f'Preserve existing candidate: {TARGET}')
    for name in base['files']:
        dst=TARGET/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,dst)
    manifest=json.loads(json.dumps(base))
    for name in SOURCES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)

    sys.path[:0]=[str(ROOT)]
    from braids_core.sediment.jiang_cao_coupled import NAMES
    from braids_core.sediment.portable_river_runtime import _bindings
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    source=ROOT/'braids_core/sediment/jiang_cao_coupled.h'
    engine=WaveEngine('cpu',8,source=source,names=NAMES);row=_bindings(engine)
    table=json.loads((BASE/'PRECOMPILED.json').read_text(encoding='utf8'))
    existing=next(value for key,value in table.items()
        if key.startswith('validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'))
    rel=Path(next(iter(existing['libraries'].values()))['file'])
    deps=[p for p in (BASE/rel.parent).glob('*.dll') if p.name!=rel.name]
    added=[]
    for attr,path in list(row['libraries'].items()):
        library=Path(path);folder=TARGET/'precompiled'/sha(library)[:20];folder.mkdir(exist_ok=True)
        for item in [library,source,*deps]:
            dst=folder/item.name;shutil.copy2(item,dst)
            name=dst.relative_to(TARGET).as_posix();manifest['files'][name]=sha(dst);added.append(name)
        row['libraries'][attr]=dict(file=(folder/library.name).relative_to(TARGET).as_posix(),sha256=sha(library))
    key='validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(source)+':'+','.join(NAMES)
    if key in table:raise ValueError('Candidate binding unexpectedly exists in baseline')
    table[key]=row;save(TARGET/'PRECOMPILED.json',table)
    manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest);verify(TARGET)
    changed=[name for name in SOURCES if base['files'][name]!=manifest['files'][name]]
    record=dict(base_runtime_sha256=sha(BASE/'RUNTIME.json'),candidate_runtime_sha256=sha(TARGET/'RUNTIME.json'),
        changed=changed,new_binding=key,added=sorted(set(added)),installed_runtime_changed=False,
        physical_formulas_changed=False,
        gradient='aligned one-sided shared-face streamwise; forward preferred, backward at corners; centered LS transverse; affine-exact',
        capacity_state='flow-projected shared-face pair average with dynamic-location affine recovery; physical dry support preserved')
    save(OUT/'BUILD.json',record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
