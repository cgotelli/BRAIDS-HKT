"""Compiler-free package bindings, preserving hydraulic and reference engines."""
import json,sys,shutil
from pathlib import Path
from stage_suspended216 import ROOT,BASE,OUT,TARGET,sha,save,verify
from build_smart183 import sources,key

def main():
    baseline='--baseline' in sys.argv
    sys.path[:0]=[str(BASE if baseline else TARGET),str(ROOT)]
    from braids_core.sediment import river_capacity as cap,class_wave_cache as cache,river_suspension as sus
    from braids_core.sediment.river_vr1993 import RiverVR1993
    from braids_core.sediment import river_vr1993 as vr
    rows=sources(True)
    code=cap.wave_source(physical_drag=True)
    rows['physical']=(code,('wave_state_gradients','matrix_bed_faces'))
    rows['physical_cached']=(cache.source(code),cache.NAMES)
    rows['suspension_base']=(Path(sus.__file__).with_suffix('.h').read_text(),sus.NAMES)
    rows['vr1993_producer']=(Path(vr.__file__).with_suffix('.h').read_text(),('vr1993_producer',))
    if baseline:save(OUT/'BASE_KEYS.json',{k:key(*v) for k,v in rows.items()});return
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from validation.sediment.river_morphodynamics import sediment_wave as wave
    from braids_core.sediment.portable_river_runtime import _bindings
    old=json.loads((OUT/'BASE_KEYS.json').read_text());table=json.loads((BASE/'PRECOMPILED.json').read_text())
    m=verify(TARGET);changed=[]
    librel=Path(next(iter(table[old['capacity']]['libraries'].values()))['file'])
    deps=[p for p in (BASE/librel.parent).glob('*.dll') if p.name!=librel.name]
    for label,(code,names) in rows.items():
        path=OUT/'build'/label/'source.h';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
        if label=='vr1993_producer':path=Path(vr.__file__).with_suffix('.h')
        if label=='suspension_base':path=Path(sus.__file__).with_suffix('.h')
        # Keep literal-header builds separate from historical text-normalized
        # caches: source identity includes line endings, even when code does not.
        previous_root=wave.ROOT
        if label=='suspension_base':wave.ROOT=OUT/'literal_native_cache'
        try:op=WaveEngine('cpu',4,source=path,names=names)
        finally:wave.ROOT=previous_root
        row=_bindings(op)
        for attr,original in list(row['libraries'].items()):
            lib=Path(original);package=TARGET/'precompiled'/sha(lib)[:20];package.mkdir(exist_ok=True)
            for p in [lib,path,*deps]:
                dst=package/p.name;shutil.copy2(p,dst);m['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
            row['libraries'][attr]=dict(file=(package/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
        newkey='validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(path)+':'+','.join(names)
        table.pop(old.get(label),None);table[newkey]=row
        changed.append(dict(label=label,key=newkey));print('Built',label,flush=True)
    save(TARGET/'PRECOMPILED.json',table);m['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',m);save(OUT/'BUILD.json',dict(changed=changed,manifest=sha(TARGET/'RUNTIME.json'),hydraulics_changed=False))

if __name__=='__main__':main()
