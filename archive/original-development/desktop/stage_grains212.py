"""Authenticated UI-only preview37; every numerical payload remains unchanged."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_grains212';TARGET=OUT/'candidate'
BASE=ROOT/'desktop/runtime'
BASE_HASH='045c4da141a6e8d4dd8d1820590b39896df7e61699efa4fafdee8669190c0d20'
FILES=('braids_desktop/__init__.py','braids_desktop/bedload_catalog.py',
       'braids_desktop/river_physics.py','braids_desktop/sediment_workflow.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def verify(root):
    root=Path(root).resolve();m=json.loads((root/'RUNTIME.json').read_text())
    for n,h in m['files'].items():assert (root/n).resolve().is_relative_to(root) and sha(root/n)==h,n
    return m
def main():
    assert sha(BASE/'RUNTIME.json')==BASE_HASH
    m=verify(BASE)
    if '--refresh-ui' in sys.argv:
        current=verify(TARGET)
        assert set(current['files'])==set(m['files'])
        assert all(h==current['files'][n] for n,h in m['files'].items() if n not in FILES)
    else:
        assert not TARGET.exists()
        for n in m['files']:
            dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    for n in FILES:shutil.copy2(ROOT/n,TARGET/n);m['files'][n]=sha(TARGET/n)
    save(TARGET/'RUNTIME.json',m);verify(TARGET)
    save(OUT/'STAGE.json',dict(runtime=sha(TARGET/'RUNTIME.json'),base=BASE_HASH,changed=FILES,
        numerical_payloads_unchanged=True))
    print('Staged preview37',sha(TARGET/'RUNTIME.json'),flush=True)
if __name__=='__main__':main()
