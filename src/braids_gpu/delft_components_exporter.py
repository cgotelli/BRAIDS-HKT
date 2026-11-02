"""Authenticated zero-step initializer for current Delft3D component choices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HISTORICAL_EXPORTER_V13_SHA256=(
    '1828a447ea08e5311828128cca005581f44d14b88dc409a63ff9f48faacbacb8')
HISTORICAL_EXPORTER_V13_GENERATION='native_exporter_v13'
HISTORICAL_EXPORTER_V14_SHA256=(
    'd803c052cfc3db3de71ef387dc7e77f1458c1d11a691a1b44b429af4ba7f699f')
HISTORICAL_EXPORTER_V14_GENERATION='native_exporter_v14'
HISTORICAL_EXPORTER_V15_SHA256=(
    '9ccdd9cb457140eda9ad418a2843047666c40ea1d8fe6031d52a02a4da88fbd8')
HISTORICAL_EXPORTER_V15_GENERATION='native_exporter_v15'
EXPORTER_V16_SHA256=(
    '4e76d84516ba9fc24cf66154de35b548d8214932fa785cc365a818692e4cb266')
EXPORTER_V16_GENERATION='native_exporter_v16'
EXPORTER_V17_SHA256=(
    '75faae530e2d8a19144b62aa8e18aff7620b5f0b158c0e60a19e769770a2ae7d')
EXPORTER_V17_GENERATION='native_exporter_v17'
EXPORTER_V18_SHA256=(
    'fb6c9778e1b21400ed6a8bbce0acce4fcecbf3babda55f1dd17262b322f8f467')
EXPORTER_V18_GENERATION='native_exporter_v18'
# Keep the longstanding unqualified v15 identity for callers preparing a
# standalone v55 deck.  The runner selects repaired v17 explicitly and only
# for v56; v16 remains an authenticated historical snapshot.
GENERATION=HISTORICAL_EXPORTER_V15_GENERATION
EXPORTER_NAME='xbraids_delft_export.exe'
EXPORTER_RELATIVE='braids_gpu/'+GENERATION+'/'+EXPORTER_NAME
EXPORTER_SHA256=HISTORICAL_EXPORTER_V15_SHA256
EXPORTER_GENERATIONS={
    HISTORICAL_EXPORTER_V13_SHA256: HISTORICAL_EXPORTER_V13_GENERATION,
    HISTORICAL_EXPORTER_V14_SHA256: HISTORICAL_EXPORTER_V14_GENERATION,
    EXPORTER_SHA256: GENERATION,
    EXPORTER_V16_SHA256: EXPORTER_V16_GENERATION,
    EXPORTER_V17_SHA256: EXPORTER_V17_GENERATION,
    EXPORTER_V18_SHA256: EXPORTER_V18_GENERATION,
}


def verify_source_exporter(root=None, exporter_sha256=EXPORTER_SHA256):
    """Verify any pinned generation without rewriting frozen receipts."""
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    generation=EXPORTER_GENERATIONS.get(exporter_sha256)
    if generation is None:
        raise ValueError('Unknown Delft component initializer identity')
    runtime=root/'braids_gpu'/generation
    executable=runtime/EXPORTER_NAME
    build=json.loads((runtime/'build.json').read_text(encoding='utf-8'))
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (runtime.is_symlink() or executable.is_symlink() or
        build.get('generation')!=generation or
        build.get('executable_sha256')!=exporter_sha256 or
        sha(executable)!=exporter_sha256 or
        build.get('native_solver_steps_performed')!=0 or
        build.get('unconditional_initialization_exit') is not True or
        build.get('numerical_source_unchanged_by_instrumentation') is not True or
        build.get('Delft3D_source_modified') is not False):
        raise ValueError('Delft component initializer identity differs')
    for name,expected in build['snapshot_source_sha256'].items():
        path=runtime/name
        if path.is_symlink() or not path.is_file() or sha(path)!=expected:
            raise ValueError('Delft component initializer snapshot differs: '+name)
    return dict(executable=str(executable),runtime=str(runtime),build=build,
                executable_sha256=exporter_sha256,generation=generation)
