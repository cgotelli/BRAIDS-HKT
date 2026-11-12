"""Promote the qualified leaf-mesh handoff to Desktop, preserving preview2."""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.install_desktop_geometry137 import sha,verify


def main():
    evidence=ROOT/'validation/_mesh138'
    for name in ('UNIT_LOCAL.xml','STAGED_FINAL.xml','DESKTOP_FINAL.xml'):
        tree=ET.parse(evidence/name)
        if not list(tree.iter('testcase')) or any(list(tree.iter(k)) for k in ('failure','error','skipped')):
            raise ValueError('Qualification did not pass: '+name)
    if not json.loads((evidence/'abba_final/STRICT_COMPARISON.json').read_text())['all_pass']:
        raise ValueError('Exact before/after qualification failed')
    source=evidence/'candidate2';manifest=verify(source)
    desktop=(ROOT/'desktop').resolve();target=desktop/'runtime'
    backup=desktop/'_reference/runtime_preview2_before_mesh138'
    pending=desktop/'_reference/runtime_mesh138_pending'
    for path in (target,backup,pending):
        if not path.resolve().is_relative_to(desktop) or path.is_symlink():
            raise ValueError('Runtime path escapes its installation')
    if backup.exists() or pending.exists():raise FileExistsError('Preserve existing install evidence')
    old=verify(target);staging=json.loads((evidence/'candidate2_STAGING.json').read_text())
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
    receipt=dict(runtime=str(target),backup=str(backup),files=len(manifest['files']),
        manifest_sha256=sha(target/'RUNTIME.json'),previous_manifest_sha256=sha(backup/'RUNTIME.json'),
        qgis_modified=False,source=str(source),native_artifacts_unchanged=True)
    (evidence/'INSTALLATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
