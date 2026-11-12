"""Explicit bounded alpha131 deployment inputs, preserving alpha129 originals."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'validation/_amr131/release_projects'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for source in sorted((ROOT/'validation/_inlet129/release_projects_final').glob('check_*.json')):
        raw=source.read_bytes();doc=json.loads(raw)
        before=doc['settings']['controls']['cfl']
        hydro=doc['model']['physics']['sediment'] is None
        after=1. if hydro else .5
        doc['settings']['controls']['cfl']=after
        target=out/source.name
        target.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
        rows.append(dict(original=str(source),original_sha256=hashlib.sha256(raw).hexdigest(),
            input=str(target),input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            changes=[dict(path='settings.controls.cfl',before=before,after=after)],
            purpose='Compiler-free public CPU/GPU dynamic/static hydro/bedload delivery, not new physical validation'))
    (out/'PROVENANCE.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
