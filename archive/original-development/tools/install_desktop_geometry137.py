"""Install qualified geometry reuse in Desktop only, with a recoverable backup."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'validation/_geometry137'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest=json.loads((root/'RUNTIME.json').read_text())
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:
            raise ValueError('Runtime artifact changed: '+name)
    return manifest


def main():
    for name in ('UNIT_LOCAL.xml','STAGED_FINAL.xml','DESKTOP_FINAL.xml'):
        tree=ET.parse(EVIDENCE/name)
        if not list(tree.iter('testcase')) or list(tree.iter('failure')) or list(tree.iter('error')) or list(tree.iter('skipped')):
            raise ValueError('Qualification did not pass: '+name)
    if not json.loads((EVIDENCE/'abba_final/STRICT_COMPARISON.json').read_text())['all_pass']:
        raise ValueError('Exact before/after qualification failed')
    source=EVIDENCE/'candidate3';manifest=verify(source)
    desktop=(ROOT/'desktop').resolve()
    target=desktop/'runtime'
    backup=desktop/'_reference/runtime_alpha136_before_geometry137'
    pending=desktop/'_reference/runtime_geometry137_pending'
    for path in (target,backup,pending):
        if not path.resolve().is_relative_to(desktop) or path.is_symlink():
            raise ValueError('Desktop runtime path escapes its installation')
    if backup.exists() or pending.exists():raise FileExistsError('Preserve prior install evidence')
    old=verify(target)
    staging=json.loads((EVIDENCE/'candidate3_STAGING.json').read_text())
    if sha(target/'RUNTIME.json')!=staging['base_manifest_sha256']:
        raise ValueError('Desktop release changed since qualification')
    pending.mkdir(parents=True)
    for name in (*manifest['files'],'RUNTIME.json'):
        dest=pending/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,dest)
    assert verify(pending)==manifest
    target.rename(backup)
    try:pending.rename(target)
    except BaseException:
        backup.rename(target)
        raise
    assert verify(target)==manifest and verify(backup)==old
    receipt=dict(runtime=str(target),backup=str(backup),
        files=len(manifest['files']),manifest_sha256=sha(target/'RUNTIME.json'),
        previous_manifest_sha256=sha(backup/'RUNTIME.json'),
        qgis_modified=False,source=str(source),native_artifacts_unchanged=True)
    (EVIDENCE/'INSTALLATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
