"""Authenticated preview36 from qualified210; no physics changes at packaging."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_delivery211';TARGET=OUT/'candidate'
BASE=ROOT/'validation/_exchange210/candidate';BASE_HASH='b240c08d228679e9c89e33b52e111d2edef8265bb3d6de2115bd46bbc91b9a34'
FILES=('braids_core/river_jiang_cao.py','braids_core/sediment/jiang_cao_desktop.py','braids_desktop/__init__.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def verify(root):
    root=Path(root).resolve();m=json.loads((root/'RUNTIME.json').read_text())
    for n,h in m['files'].items():assert (root/n).resolve().is_relative_to(root) and sha(root/n)==h,n
    return m
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH and not TARGET.exists()
    m=verify(BASE)
    for n in m['files']:
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:shutil.copy2(ROOT/n,TARGET/n);m['files'][n]=sha(TARGET/n)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES))
    print('Staged preview36',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
