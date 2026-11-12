"""Stage identified research candidate without changing installed Desktop."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_startup208/candidate';TARGET=ROOT/'validation/_shallow209/candidate_v3'
FILES=('braids_core/sediment/jiang_cao_wetdry.py','braids_core/sediment/jiang_cao_desktop.py','braids_core/river_jiang_cao.py')
SOURCE=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_shallow209/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(TARGET));a=p.parse_args();target=Path(a.output).resolve()
    assert sha(BASE/'RUNTIME.json')=='fce57aef51d2a856e21bde2722f5448fe36002222a7b77b454fbe5dcb64e0818'
    assert not target.exists()
    m=json.loads((BASE/'RUNTIME.json').read_text())
    for n,h in m['files'].items():
        assert sha(BASE/n)==h,n
        dst=target/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,dst)
    changes={}
    for n in FILES:
        old=m['files'][n];shutil.copy2(SOURCE/n,target/n);m['files'][n]=sha(target/n)
        changes[n]=dict(old=old,new=m['files'][n])
    (target/'RUNTIME.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(runtime=sha(target/'RUNTIME.json'),changes=changes)),flush=True)
if __name__=='__main__':main()
