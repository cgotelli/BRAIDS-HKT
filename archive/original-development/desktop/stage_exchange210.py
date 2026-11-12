"""Build a compiler-free research candidate; installed Desktop stays untouched."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_shallow209/candidate_v3';TARGET=ROOT/'validation/_exchange210/candidate'
FILES=('braids_core/sediment/jiang_cao_exchange.h','braids_core/sediment/jiang_cao_exchange.py','braids_core/sediment/jiang_cao_wetdry.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def main():
    assert sha(BASE/'RUNTIME.json')=='b7b9958a312979cdb2b10fa3e8ab33552ded636ecc6f6d1fd3183df4d4ebb38e'
    assert not TARGET.exists()
    m=json.loads((BASE/'RUNTIME.json').read_text())
    for n,h in m['files'].items():
        assert sha(BASE/n)==h,n
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:shutil.copy2(ROOT/n,TARGET/n);m['files'][n]=sha(TARGET/n)
    name='braids_core/river_jiang_cao.py';p=TARGET/name
    p.write_text(p.read_text().replace('bundle.jiang_cao_p15_grain_front209_v2','bundle.jiang_cao_p15_finite_stock210_v1'),encoding='utf-8')
    m['files'][name]=sha(p)
    from braids_core.sediment.portable_river_runtime import install,RECORDS
    install()
    from tests.test_jiang_cao196 import MAT
    from braids_core.sediment.jiang_cao_exchange import JiangCaoExchange
    JiangCaoExchange('cpu',MAT)
    save(TARGET.parent/'NATIVE_RECORDS.json',RECORDS)
    table=json.loads((BASE/'PRECOMPILED.json').read_text());added=[]
    deps={p.name:p for p in (BASE/'precompiled').rglob('*.dll') if p.name in ('libdl.dll','libgomp-1.dll','libgcc_s_seh-1.dll','libwinpthread-1.dll','libstdc++-6.dll')}
    for key,row in RECORDS.items():
        if key in table:continue
        for attr,name in row['libraries'].items():
            src=Path(name);dest=TARGET/'precompiled'/sha(src)[:20];dest.mkdir(parents=True,exist_ok=True)
            for file in [src,*deps.values()]:
                dst=dest/file.name;shutil.copy2(file,dst);m['files'][dst.relative_to(TARGET).as_posix()]=sha(dst)
            row['libraries'][attr]=dict(file=(dest/src.name).relative_to(TARGET).as_posix(),sha256=sha(src))
        table[key]=row;added.append(key)
    save(TARGET/'PRECOMPILED.json',table);m['files']['PRECOMPILED.json']=sha(TARGET/'PRECOMPILED.json')
    save(TARGET/'RUNTIME.json',m)
    for n,h in m['files'].items():assert sha(TARGET/n)==h,n
    record=dict(runtime=sha(TARGET/'RUNTIME.json'),new_bindings=added,changed=FILES)
    save(TARGET.parent/'STAGE.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
