"""Isolated compiler-free candidate; installed preview35 remains immutable."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime';TARGET=ROOT/'validation/_startup208/candidate'
FILES=('braids_core/sediment/jiang_cao_adjustment.py','braids_core/sediment/jiang_cao_desktop.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(BASE/'RUNTIME.json')=='f88145dcf039c589557e9bb1acf7c6dacaf8b1afedec7952e3a533aa9f6ccb6b'
    assert not TARGET.exists()
    m=json.loads((BASE/'RUNTIME.json').read_text())
    for n,h in m['files'].items():
        assert sha(BASE/n)==h,n
        dst=TARGET/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    changes={}
    for n in FILES:
        old=m['files'][n];shutil.copy2(ROOT/n,TARGET/n);m['files'][n]=sha(TARGET/n)
        changes[n]=dict(old=old,new=m['files'][n])
    (TARGET/'RUNTIME.json').write_text(json.dumps(m,indent=2)+'\n')
    (TARGET.parent/'STAGE.json').write_text(json.dumps(dict(runtime=sha(TARGET/'RUNTIME.json'),changes=changes,installed_unchanged=True),indent=2))
    print(sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
