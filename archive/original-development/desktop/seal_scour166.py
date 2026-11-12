"""Seal the bounded attribution evidence without touching either solver."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'validation/sediment/river_morphodynamics/reports/desktop_scour166'
RAW = ROOT/'validation/_scour166'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target = OUT/'MANIFEST.json'
    if target.exists():
        raise FileExistsError(target)
    audit = json.loads((OUT/'AUDIT.json').read_text())
    runtime = ROOT/'desktop/runtime'
    assert sha(runtime/'RUNTIME.json') == audit['runtime_manifest_sha256']
    for name, digest in json.loads((runtime/'RUNTIME.json').read_text())['files'].items():
        assert sha(runtime/name) == digest, name
    for name, digest in audit['native_sources_read_only'].items():
        assert sha(ROOT/name) == digest, name
    tests = RAW/'checks_final.xml'
    suite = ET.parse(tests).getroot().find('testsuite')
    assert int(suite.attrib['tests']) == 16
    assert all(int(suite.attrib[k]) == 0 for k in ('errors', 'failures', 'skipped'))
    files = ['desktop/replay_scour166.py', 'desktop/review_scour166.py',
             'desktop/seal_scour166.py', 'tests/test_desktop_scour166.py',
             'validation/RIVER_MORPHODYNAMICS_STATUS.md']
    for name in files:
        dest = OUT/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    receipts = {}
    for path in (RAW/'v3').glob('*/AUDIT.json'):
        row = json.loads(path.read_text())
        for item in row['rows']:
            assert sha(path.parent/item['file']) == item['sha256']
        receipts[str(path.relative_to(ROOT))] = sha(path)
    assert len(receipts) == 6
    payload = dict(installed_runtime_sha256=audit['runtime_manifest_sha256'],
        installed_runtime_changed=False, Delft3D_modified=False,
        scope='Frozen-state attribution, not a new coupled hydraulic correction or release',
        tests=dict(passed=16, failures=0, skipped=0, receipt=str(tests.relative_to(ROOT)), sha256=sha(tests)),
        probe_receipts=receipts, sources_sha256={name: sha(ROOT/name) for name in files},
        artifacts_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    target.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print('Desktop166 evidence sealed; installed preview26 unchanged.')


if __name__ == '__main__':
    main()
