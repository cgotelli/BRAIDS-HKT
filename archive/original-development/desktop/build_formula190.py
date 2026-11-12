"""Compile capacity/wave sources only; hydraulics and Delft3D untouched."""
import json,os,sys,shutil
from pathlib import Path
from stage_formula190 import ROOT,BASE,OUT,TARGET,sha
from stage_secondary185 import save
from build_smart183 import sources,key
def main():
    baseline='--baseline' in sys.argv
    runtime=BASE if baseline else TARGET
    sys.path[:0]=[str(runtime),str(ROOT)]
    rows=sources(True)
    if baseline:
        save(OUT/'BASE_KEYS.json',{k:key(*v) for k,v in rows.items()});return
    from braids_core.sediment import river_capacity as cap,class_wave_cache as cache
    code=cap.wave_source(physical_drag=True)
    rows['ancey']=(code,('wave_state_gradients','matrix_bed_faces'))
    rows['ancey_cached']=(cache.source(code),cache.NAMES)
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from braids_core.sediment.portable_river_runtime import _bindings
    old=json.loads((OUT/'BASE_KEYS.json').read_text());table=json.loads((BASE/'PRECOMPILED.json').read_text())
    manifest=json.loads((TARGET/'RUNTIME.json').read_text());changed=[]
    librel=Path(next(iter(table[old['capacity']]['libraries'].values()))['file'])
    deps=[p for p in (BASE/librel.parent).glob('*.dll') if p.name!=librel.name]
    for label,(code,names) in rows.items():
        path=OUT/'build'/label/'source.h';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
        op=WaveEngine('cpu',4,source=path,names=names);row=_bindings(op)
        for attr,original in list(row['libraries'].items()):
            lib=Path(original);package=TARGET/'precompiled'/sha(lib)[:20];package.mkdir(exist_ok=True)
            for p in [lib,path,*deps]:
                dst=package/p.name
                if not dst.exists():shutil.copy2(p,dst)
                assert sha(p)==sha(dst)
                manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
            row['libraries'][attr]=dict(file=(package/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
        if label in old and old[label] in table:del table[old[label]]
        table[key(code,names)]=row;changed.append(dict(label=label,new=key(code,names),identity=op.identity))
        print('Built',label,flush=True)
    save(TARGET/'PRECOMPILED.json',table);manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest);save(OUT/'BUILD.json',dict(changed=changed,manifest=sha(TARGET/'RUNTIME.json')))
if __name__=='__main__':main()
