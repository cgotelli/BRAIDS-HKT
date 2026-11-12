"""Recover missing packaged source only, with a verifiable provenance receipt.

Historical research fixtures/reports are not in the package and are NOT recovered.
Existing files, old simulations and the installed runtime are never overwritten.
"""
from pathlib import Path
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
OUT=ROOT/'desktop/qualification/supplied220'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((BASE/'RUNTIME.json').read_text()); restored=[]; existing=[]
    for name,expected in manifest['files'].items():
        if not name.startswith('validation/') or Path(name).suffix not in ('.py','.h','.cu','.f90','.inc'):
            continue
        src=BASE/name;dst=ROOT/name
        assert sha(src)==expected,name
        if dst.exists():existing.append(name);continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as target:target.write(src.read_bytes())
        assert sha(dst)==expected
        restored.append(dict(path=name,sha256=expected))
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=dict(source=str(BASE),runtime_sha256=sha(BASE/'RUNTIME.json'),restored=restored,
        existing_preserved=existing,historical_research_sources_recovered=False)
    with (OUT/'RECOVERY.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print('Recovered missing packaged source files:',len(restored))

if __name__=='__main__':main()
