"""Freeze the weir/clock candidate and its two shared OpenMP entry packages."""
import hashlib,json,shutil,sys,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_boundaries160';TARGET=OUT/'candidate3'
FILES=('braids_core/river_open_boundary.py','braids_core/sediment/river_exterior_outlet.py','braids_core/sediment/river_exterior_outlet.h',
       'braids_core/sediment/adaptive_river_worker.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if TARGET.exists():raise FileExistsError(TARGET)
    shutil.copytree(ROOT/'desktop/runtime',TARGET)
    manifest=json.loads((TARGET/'RUNTIME.json').read_text())
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from validation.sediment.river_morphodynamics import sediment_wave
    from braids_core.sediment.portable_river_runtime import _bindings,_key
    table=json.loads((TARGET/'PRECOMPILED.json').read_text())
    source=TARGET/'braids_core/sediment/river_exterior_outlet.h'
    for names in (('exterior_critical_outlet',),('exterior_free_weir',)):
        engine=sediment_wave.WaveEngine('cpu',2,source=source,names=names)
        bound=inspect.signature(sediment_wave.WaveEngine.__init__).bind(engine,'cpu',2,source=source,names=names);bound.apply_defaults()
        key=_key(sediment_wave,'WaveEngine',bound.arguments);row=_bindings(engine)
        for name,file in row['libraries'].items():
            src=Path(file);folder=TARGET/'precompiled'/sha(src)[:20];folder.mkdir(parents=True,exist_ok=True)
            for path in (src,*[Path(shutil.which('g++')).parent/n for n in ('libgomp-1.dll','libwinpthread-1.dll','libgcc_s_seh-1.dll','libdl.dll')]):
                target=folder/path.name
                if not target.exists():shutil.copy2(path,target)
                else:assert sha(target)==sha(path)
                manifest['files'][target.relative_to(TARGET).as_posix()]=sha(target)
            row['libraries'][name]=dict(file=(folder/src.name).relative_to(TARGET).as_posix(),sha256=sha(src))
        table[key]=row
    (TARGET/'PRECOMPILED.json').write_text(json.dumps(table,indent=2)+'\n');manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n');print(sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
