"""Install the qualified exact-wall Desktop runtime, retaining preview26 intact."""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'desktop'))
from stage_wall172 import sha, save


def verify(base, expected):
    assert sha(base / 'RUNTIME.json') == expected, base
    manifest = json.loads((base / 'RUNTIME.json').read_text())
    for name, digest in manifest['files'].items():
        path = (base / name).resolve()
        assert path.is_relative_to(base.resolve()), name
        assert sha(path) == digest, name
    return manifest


def main():
    folder = ROOT / 'validation/_delivery172'
    stage = json.loads((folder / 'STAGE.json').read_text())
    base, candidate = ROOT / 'desktop/runtime', folder / 'candidate'
    old = verify(base, stage['previous_runtime_sha256'])
    new = verify(candidate, stage['runtime_sha256'])
    verify(folder / 'preview26_runtime', stage['previous_runtime_sha256'])
    for name, count in (('REGRESSIONS.xml', 376), ('PACKAGE.xml', 141)):
        suites = ET.parse(folder / name).getroot().findall('testsuite')
        assert sum(int(s.attrib['tests']) for s in suites) == count
        assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
    audit = json.loads((ROOT / 'validation/sediment/river_morphodynamics/reports/desktop_delivery172/AUDIT.json').read_text())
    assert audit['stage']['runtime_sha256'] == stage['runtime_sha256']
    assert len(audit['runs']) == 16 and len(audit['boundaries']) == 32 and len(audit['laws']) == 3
    for row in audit['runs'].values():
        assert row['check']['comparison']['exact'] and all(row['check']['restart']['exact'].values())
    for name, digest in audit['source_reference'].items():
        assert sha(ROOT / name) == digest
    backup = folder / 'preview26_ui'
    if backup.exists():
        raise FileExistsError(backup)
    for name in ('desktop/PROVENANCE.json', 'desktop/README.md', 'braids_desktop/__init__.py'):
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    changed = [name for name, digest in new['files'].items() if old['files'].get(name) != digest]
    # The complete old runtime is retained; no old output or source is deleted.
    # Validate all targets before replacing any payload. Publish manifest last.
    targets = [(candidate / name, (base / name).resolve()) for name in changed + ['RUNTIME.json']]
    assert all(target.is_relative_to(base.resolve()) for _, target in targets)
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verify(base, stage['runtime_sha256'])
    receipt = dict(previous_runtime=stage['previous_runtime_sha256'], installed_runtime=stage['runtime_sha256'],
                   changed=changed, runtime_backup=str(folder / 'preview26_runtime'), metadata_backup=str(backup),
                   QGIS_modified=False, Delft3D_modified=False, user_projects_modified=False,
                   qualification='validation/sediment/river_morphodynamics/reports/desktop_delivery172/REPORT.md')
    save(ROOT / 'desktop/evidence/wall172/DELIVERY.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
