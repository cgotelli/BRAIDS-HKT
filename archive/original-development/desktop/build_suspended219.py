"""Add only the S6 shared kernel binding; retain every existing binary."""
import sys,json,shutil
from pathlib import Path
from stage_suspended219 import ROOT,BASE,OUT,TARGET,sha,save,verify
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_core.sediment import river_classic_suspension as gp
from braids_core.sediment.portable_river_runtime import _bindings

def main():
    table=json.loads((BASE/'PRECOMPILED.json').read_text());m=verify(TARGET)
    op=gp.ClassicSuspension('cpu',4,gp.parameters(gp.VR84,dict(ds=.0001,d50=.0002,d90=.0004))).engine
    path=Path(gp.__file__).with_suffix('.h');row=_bindings(op)
    for attr,original in list(row['libraries'].items()):
        lib=Path(original);package=TARGET/'precompiled'/sha(lib)[:20];package.mkdir(exist_ok=True)
        dependencies=list(lib.parent.glob('*.dll'))
        # Runtime dependencies already shipped for the same local compiler.
        dependencies+=list((BASE/'precompiled').glob('*/libgomp-1.dll'))[:1]
        dependencies+=list((BASE/'precompiled').glob('*/libgcc_s_seh-1.dll'))[:1]
        dependencies+=list((BASE/'precompiled').glob('*/libstdc++-6.dll'))[:1]
        dependencies+=list((BASE/'precompiled').glob('*/libwinpthread-1.dll'))[:1]
        dependencies+=list((BASE/'precompiled').glob('*/libdl.dll'))[:1]
        for p in [lib,path,*dependencies]:
            dst=package/p.name;shutil.copy2(p,dst);m['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        row['libraries'][attr]=dict(file=(package/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
    key='validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(path)+':'+','.join(gp.NAMES)
    table[key]=row
    base_files=json.loads((BASE/'RUNTIME.json').read_text())['files']
    new_dirs={str(Path(v['file']).parent).replace('\\','/')+'/' for v in row['libraries'].values()}
    # Exclude superseded trial binaries from the authenticated release payload.
    # Keep their actual files in staging as diagnostic evidence.
    for n in list(m['files']):
        if n.startswith('precompiled/') and n not in base_files and not any(n.startswith(d) for d in new_dirs):
            del m['files'][n]
    save(TARGET/'PRECOMPILED.json',table);m['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'BUILD.json',dict(key=key,manifest=sha(TARGET/'RUNTIME.json'),existing_bindings_unchanged=True))
    print('Built S6 classic closures; existing native bindings unchanged',flush=True)

if __name__=='__main__':main()

