"""Package only new native bindings; preserve every existing native payload."""
from stage_hkt4 import ROOT,OUT,TARGET,BASE,sha,save,verify
import json,shutil
from pathlib import Path


def main():
    m=verify(TARGET);table=json.loads((TARGET/'PRECOMPILED.json').read_text());record=json.loads((OUT/'BINDINGS.json').read_text())
    original=json.loads((BASE/'PRECOMPILED.json').read_text());keys=sorted(set(record)-set(original))
    assert {'braids_core.sediment.river_turbine_flow.Engine:','braids_core.sediment.river_hydrokinetic_stress.FullStressEngine:','braids_core.sediment.river_turbine_flow.instrumented_native'}<=set(keys)
    deps=['libgcc_s_seh-1.dll','libgfortran-5.dll','libgomp-1.dll','libquadmath-0.dll','libwinpthread-1.dll','libdl.dll']
    added=[]
    def copy(source,relative):
        p=TARGET/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==sha(source),relative
        else:shutil.copy2(source,p);added.append(relative)
        m['files'][relative]=sha(p);return dict(file=relative,sha256=sha(p))
    def library(path,native=False):
        src=Path(path);folder=Path('precompiled')/sha(src)[:20];row=copy(src,(folder/src.name).as_posix())
        for name in deps:copy(Path('C:/Strawberry/c/bin')/name,(folder/name).as_posix())
        if native:
            for p in src.parent.glob('*.f90'):copy(p,(folder/p.name).as_posix())
        return row
    for key in keys:
        assert key in record,key;row=record[key]
        for name,path in list(row.get('libraries',{}).items()):row['libraries'][name]=library(path)
        if 'native' in row:row['native']=library(row['native'],True)
        table[key]=row
    assert all(table[k]==v for k,v in original.items())
    save(TARGET/'PRECOMPILED.json',table);m['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json');save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'PACKAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),new_bindings=keys,added=added,prior_bindings_unchanged=True))
    print('Packaged',sha(TARGET/'RUNTIME.json'))


if __name__=='__main__':main()
