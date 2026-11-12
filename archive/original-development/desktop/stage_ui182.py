"""Stage the explicit domain-border refinement option, without solver changes."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime';TARGET=ROOT/'validation/_ui182/runtime'
FILES=('braids_core/sediment/river_field_policy.py','braids_core/sediment/river_terrain_refinement.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if TARGET.exists():raise FileExistsError(TARGET)
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():assert sha(BASE/name)==digest,name
    shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    backup=ROOT/'validation/_ui182/preview27_files';backup.mkdir()
    for name in (*FILES,'RUNTIME.json'):
        p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,p)
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Staged',sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
