"""Version the graded GPU source archive after common runner updates.

Earlier archives remain immutable. The graded CUDA and generated physics
modules are copied byte-for-byte from the authenticated parent archive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'braids_gpu/frozen_sources/graded_bedload_v6'
DESTINATION = ROOT/'braids_gpu/frozen_sources/graded_bedload_v7'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    original = SOURCE/'runtime_manifest.json'
    manifest = json.loads(original.read_text(encoding='utf-8'))
    files = manifest['source_files']
    actual = {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob('*')
              if path.is_file()}
    if actual != set(files)|{'runtime_manifest.json'} or any(
            digest(SOURCE/name) != expected for name, expected in files.items()):
        raise ValueError('The parent graded source inventory differs')
    shutil.copytree(SOURCE, DESTINATION)
    transaction = ROOT/'braids_gpu/a1_full_array_transaction.py'
    member = 'transaction_python/a1_full_array_transaction.py'
    shutil.copy2(transaction, DESTINATION/member)
    manifest['source_files'][member] = digest(transaction)
    manifest['transaction_sources']['a1_full_array_transaction.py'] = digest(transaction)
    manifest['transaction_sha256'] = digest(transaction)
    manifest['runner_sha256'] = digest(ROOT/'braids_gpu/runner.py')
    manifest['source_archive_parent_sha256'] = digest(original)
    (DESTINATION/'runtime_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if digest(transaction) != manifest['transaction_sha256'] or (
            digest(ROOT/'braids_gpu/runner.py') != manifest['runner_sha256']):
        raise ValueError('Shared GPU source changed during archive construction')
    return {'archive': str(DESTINATION.relative_to(ROOT)),
            'manifest_sha256': digest(DESTINATION/'runtime_manifest.json'),
            'parent_manifest_sha256': manifest['source_archive_parent_sha256']}


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
