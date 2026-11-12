"""Stage the secondary clock correction on the retained preview29 runtime."""
import hashlib,json,shutil,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'desktop/runtime'
OUT=ROOT/'validation/_secondary185';TARGET=OUT/'candidate'
BASE_HASH='fddbf4c03a89f4d19aa72e89dc3fb02967470289d0621c1c47d7ab9a73f71ae3'
FILES=('braids_core/sediment/river_explicit.py','braids_core/sediment/flow2dh_fixed_bed.py',
       'braids_core/sediment/shoreline_solver.py',
       'braids_core/sediment/river_secondary_clock.py','braids_core/sediment/river_secondary_clock.h')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for name,h in manifest['files'].items():assert sha(BASE/name)==h,name
    if not TARGET.exists():shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    else:
        snapshot=OUT/'source_snapshots'/sha(TARGET/'RUNTIME.json')
        for name in (*FILES,'RUNTIME.json','PRECOMPILED.json'):
            if (TARGET/name).exists() and not (snapshot/name).exists():
                (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(TARGET/name,snapshot/name)
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    sys.path[:0]=[str(TARGET),str(ROOT)]
    from braids_core.sediment.river_secondary_clock import NAMES
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
    from braids_core.sediment.portable_river_runtime import _bindings
    path=TARGET/'braids_core/sediment/river_secondary_clock.h'
    op=WaveEngine('cpu',4,source=path,names=NAMES);row=_bindings(op)
    table=json.loads((BASE/'PRECOMPILED.json').read_text())
    # Reuse the shipped OpenMP dependencies without compiling any hydraulics.
    existing=next(r for k,r in table.items() if k.startswith('validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'))
    librel=Path(next(iter(existing['libraries'].values()))['file'])
    deps=[p for p in (BASE/librel.parent).glob('*.dll') if p.name!=librel.name]
    for attr,original in list(row['libraries'].items()):
        lib=Path(original);package=TARGET/'precompiled'/sha(lib)[:20];package.mkdir(exist_ok=True)
        for p in [lib,*deps]:
            dst=package/p.name
            if not dst.exists():shutil.copy2(p,dst)
            assert sha(p)==sha(dst)
            manifest['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
        row['libraries'][attr]=dict(file=(package/lib.name).relative_to(TARGET).as_posix(),sha256=sha(lib))
    key='validation.sediment.river_morphodynamics.sediment_wave.WaveEngine:'+sha(path)+':'+','.join(NAMES)
    table[key]=row;save(TARGET/'PRECOMPILED.json',table)
    manifest['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',manifest)
    save(OUT/'STAGE.json',dict(baseline=BASE_HASH,candidate=sha(TARGET/'RUNTIME.json'),files=FILES))
    print('Staged',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
