"""Install only the two tested refinement sources, retaining native solvers."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'desktop/runtime';STAGED=ROOT/'validation/_ui182/runtime'
FILES=('braids_core/sediment/river_field_policy.py','braids_core/sediment/river_terrain_refinement.py')
OLD='8bb07b4e4877411e0035060842a47feecc41b1d5e04f14cb389407c38c9349c9'
NEW='c0e931896a988945d71f963c2e13e2a9e860f8a3293fb2a7be1726401eb371d5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():assert sha(root/name)==digest,name
    return manifest
def main():
    assert sha(TARGET/'RUNTIME.json')==OLD
    assert sha(STAGED/'RUNTIME.json')==NEW
    old=verify(TARGET);new=verify(STAGED)
    assert set(old['files'])==set(new['files'])
    assert {name for name in old['files'] if old['files'][name]!=new['files'][name]}==set(FILES)
    for name in FILES:
        assert sha(ROOT/name)==sha(STAGED/name)
        assert sha(ROOT/'validation/_ui182/preview27_files'/name)==old['files'][name]
    for name in (*FILES,'RUNTIME.json'):shutil.copy2(STAGED/name,TARGET/name)
    verify(TARGET)
    receipt=dict(previous=OLD,installed=sha(TARGET/'RUNTIME.json'),changed_sources=FILES,
                 unchanged_native_solver=True,files_verified=len(new['files']))
    (ROOT/'validation/_ui182/INSTALL.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()
