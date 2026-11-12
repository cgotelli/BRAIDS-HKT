"""Build alpha131 using authenticated unchanged native alpha129 bindings."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_unified_qgis import package, merge_bindings, sha, verify, write


def main():
    evidence=ROOT/'validation/_amr131/release_evidence'
    if evidence.exists():raise FileExistsError(evidence)
    evidence.mkdir(parents=True)
    old=ROOT/'dist/unified-alpha129/braids_qgis'
    manifest=verify(old)
    sources=[ROOT/'validation/_inlet129'/name for name in (
        'package_evidence','package_evidence_canonical','package_evidence_1m','package_evidence_5m')]
    records=[json.loads((p/'RECORD.json').read_text()) for p in sources]
    if any(r['source_identity']!=manifest['source_identity'] for r in records):
        raise ValueError('Original source recordings differ')
    bindings={}
    for p in sources:
        bindings=merge_bindings(bindings,json.loads((p/'BINDINGS.json').read_text()))
    write(evidence/'BINDINGS.json',bindings)
    write(evidence/'RECORD.json',dict(source_identity=manifest['source_identity'],
        seeds=sorted(set(s for r in records for s in r['seeds'])),
        projects=[p for r in records for p in r['projects']],
        provenance='Preserved alpha129 recordings, not new simulations',
        recordings=[dict(path=str(p),sha256=sha(p/'RECORD.json')) for p in sources]))
    updates=[]
    for relative in json.loads((old/'river_runtime/RUNTIME.json').read_text())['files']:
        previous=old/'river_runtime'/relative;current=ROOT/relative
        if previous.suffix not in ('.py','.h','.cu','.cpp','.f90','.f','.inc') or not current.is_file():continue
        if sha(previous)==sha(current):continue
        if previous.suffix!='.py':raise ValueError('Native source changed: '+relative)
        updates.append(dict(path=relative,before=sha(previous),after=sha(current),
            reason='Milestones130/131 qualified Python scheduling, geometry, I/O and explicit hydro-only CFL admission; native arithmetic unchanged'))
    write(evidence/'PYTHON_UPDATES.json',dict(updates=updates))
    # The builder reconstructs the complete recorded source identity using
    # these before hashes. Any unlisted/additional source change fails closed.
    package(evidence,ROOT/'dist/unified-alpha131',python_updates_file=evidence/'PYTHON_UPDATES.json')


if __name__=='__main__':main()
