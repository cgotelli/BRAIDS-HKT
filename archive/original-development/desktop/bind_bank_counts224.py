"""Add the omitted CPU block-history remap binding; preserve every old payload."""
import json,shutil,sys
from pathlib import Path
from stage_banks224 import ROOT,OUT,sha,save,verify

PARENT=OUT/'candidate_release'
TARGET=OUT/'candidate_packaged'

def main():
    assert sha(PARENT/'RUNTIME.json')=='a8da2e412d67f4bcfafa1e060dc4ad4199734616467c85bc86354aeef65b9021'
    old=verify(PARENT);assert not TARGET.exists()
    for n in old['files']:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT/n,dst)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from braids_core.sediment.portable_river_runtime import _bindings
    source=TARGET/'braids_core/sediment/river_bank_remap.h'
    assert sha(source)=='a68d9de6faee85b4ca98b5dcf09d64b687edb3d75559236f347c11056fe14f23'
    engine=WaveEngine('cpu',4,source=source,names=('bank_counts',))
    row=_bindings(engine);table=json.loads((PARENT/'PRECOMPILED.json').read_text())
    existing=next(v for k,v in table.items() if k.startswith('validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'))
    rel=Path(next(iter(existing['libraries'].values()))['file'])
    deps=[p for p in (PARENT/rel.parent).glob('*.dll') if p.name!=rel.name]
    manifest=json.loads(json.dumps(old))
    for attr,path in list(row['libraries'].items()):
        lib=Path(path);dest=TARGET/'precompiled'/sha(lib)[:20];dest.mkdir(exist_ok=True)
        for p in [lib,*deps]:
            dst=dest/p.name;shutil.copy2(p,dst)
            manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        row['libraries'][attr]=dict(file=(dest/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
    key='validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(source)+':bank_counts'
    assert key not in table;table[key]=row
    save(TARGET/'PRECOMPILED.json',table)
    manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest);verify(TARGET)
    changed=[n for n,h in old['files'].items() if manifest['files'][n]!=h]
    added=sorted(set(manifest['files'])-set(old['files']))
    assert changed==['PRECOMPILED.json'] and all(n.startswith('precompiled/') for n in added)
    before=json.loads((PARENT/'PRECOMPILED.json').read_text())
    assert all(table[k]==v for k,v in before.items())
    save(OUT/'BINDING_FIX.json',dict(parent_runtime=sha(PARENT/'RUNTIME.json'),runtime=sha(TARGET/'RUNTIME.json'),
        parent=str(PARENT),target=str(TARGET),changed=changed,added=added,new_binding=key,
        source_sha256=sha(source),existing_sources_and_bindings_unchanged=True,
        reason='Retained block AMR called the existing integer bank_counts routine, omitted from the compiler-free CPU package.'))
    print('Packaged missing bank_counts binding:',sha(TARGET/'RUNTIME.json'),flush=True)

if __name__=='__main__':main()
