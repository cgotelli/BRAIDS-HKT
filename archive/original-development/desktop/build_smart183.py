"""Precompile the modified shared sediment operators, preserving hydraulic DLLs."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
from stage_smart183 import ROOT,BASE,TARGET,OUT,sha,save,BASE_HASH

def sources(smart):
    from braids_core.sediment import river_capacity as cap,class_wave_cache as cache
    result={'capacity':(cap.capacity_source(),cap.NAMES)}
    for label,options in [('scalar',{}),('intrinsic',dict(intrinsic=True)),
                          ('vector',dict(vector=True)),('vr1993',dict(vr1993=True)),
                          ('total',dict(total=True)),('total_vector',dict(total=True,vector=True)),
                          ('total_vr1993',dict(total=True,vr1993=True))]+([('smart',dict(smart=True))] if smart else []):
        code=cap.wave_source(**options)
        result[label]=(code,('wave_state_gradients','matrix_bed_faces'))
        result[label+'_cached']=(cache.source(code),cache.NAMES)
    return result

def key(code,names):
    return 'validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+hashlib.sha256(code.replace('\n',os.linesep).encode()).hexdigest()+':'+','.join(names)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',action='store_true');args=ap.parse_args()
    runtime=BASE if args.baseline else TARGET
    assert sha(BASE/'RUNTIME.json')==BASE_HASH,'Build against the retained preview28 baseline, never reinterpret the installed replacement'
    sys.path[:0]=[str(runtime),str(ROOT)]
    if args.baseline:
        save(OUT/'SOURCE_KEYS.json',{k:key(*v) for k,v in sources(False).items()});return
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from braids_core.sediment.portable_river_runtime import _bindings
    old=json.loads((OUT/'SOURCE_KEYS.json').read_text());table=json.loads((BASE/'PRECOMPILED.json').read_text())
    manifest=json.loads((TARGET/'RUNTIME.json').read_text());changed=[]
    oldcap=table[old['capacity']];librel=Path(next(iter(oldcap['libraries'].values()))['file'])
    deps=[p for p in (BASE/librel.parent).glob('*.dll') if p.name!=librel.name]
    for label,(code,names) in sources(True).items():
        path=OUT/'build'/label/'source.h';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(code,encoding='utf-8')
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
        table[key(code,names)]=row
        changed.append(dict(label=label,old=old.get(label),new=key(code,names),identity=op.identity))
        print('Compiled',label,flush=True)
    save(TARGET/'PRECOMPILED.json',table);manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest)
    save(OUT/'BUILD.json',dict(changed=changed,manifest=sha(TARGET/'RUNTIME.json'),hydraulics_changed=False))
if __name__=='__main__':main()
