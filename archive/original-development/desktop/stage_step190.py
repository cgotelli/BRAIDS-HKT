"""Isolated Python controller qualification; retain the installed preview30."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_wall188/desktop_candidate'
TARGET=ROOT/'validation/_step190/candidate'
FILES=('braids_core/sediment/river_sediment_clock.py',
       'braids_core/sediment/river_field_policy.py',
       'braids_core/sediment/adaptive_river_worker.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(BASE/'RUNTIME.json')=='b171942a93b28a2d5ba13b7e16ba22da09a1424c45f9a632d7707fac455d020f'
    manifest=json.loads((BASE/'RUNTIME.json').read_text())
    for name,h in manifest['files'].items():assert sha(BASE/name)==h,name
    if not TARGET.exists():shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    for name in FILES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    (TARGET/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(sha(TARGET/'RUNTIME.json'))
if __name__=='__main__':main()
