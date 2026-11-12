"""Build the isolated preview49+outer-CFL-predictor candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'desktop/runtime'
OUT=ROOT/'validation/_experiment229'
TARGET=OUT/'candidate_v2'
SOURCES=(
    'braids_core/sediment/jiang_cao_desktop.py',
    'braids_core/sediment/jiang_cao_coupled.py',
    'braids_core/sediment/jiang_cao_wetdry.py',
)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def verify(root):
    value=json.loads((root/'RUNTIME.json').read_text())
    for name,digest in value['files'].items():
        if sha(root/name)!=digest:raise ValueError('Runtime payload differs: '+name)
    return value


def main():
    base=verify(BASE)
    if TARGET.exists():raise FileExistsError(f'Preserve candidate: {TARGET}')
    for name in base['files']:
        target=TARGET/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,target)
    manifest=json.loads(json.dumps(base))
    for name in SOURCES:
        shutil.copy2(ROOT/name,TARGET/name);manifest['files'][name]=sha(TARGET/name)
    save(TARGET/'RUNTIME.json',manifest);verify(TARGET)
    record=dict(base_runtime_sha256=sha(BASE/'RUNTIME.json'),
        candidate_runtime_sha256=sha(TARGET/'RUNTIME.json'),changed=list(SOURCES),
        physical_formulas_changed=False,precompiled_bindings_changed=False,
        controller='stage-wave outer CFL predictor; accepted hint saved in JC checkpoint last diagnostics')
    save(OUT/'BUILD.json',record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
