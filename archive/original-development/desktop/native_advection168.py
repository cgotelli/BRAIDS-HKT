"""Output-only Cyclic diagnostic using the locked, unmodified FLOW engine."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop'), str(ROOT/'validation/sediment/delft3d_current')]
from check_bedload161 import save, sha, replace
import runtime


def main():
    parent = ROOT/'validation/_groin163/delft_mpm_cyclic'
    out = ROOT/'validation/_advection168/native_cyclic'
    out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((parent/'INPUTS.json').read_text())
    for name, digest in receipt['inputs'].items():
        assert sha(parent/name) == digest
        shutil.copy2(parent/name, out/name)
    changes = dict(MomTrm='#Y#', Flmap='39.9 0.01 40.1')
    path = out/'sim_P057.mdf'
    code = path.read_text()
    for key, value in changes.items():
        code = replace(code, key, value)
    path.write_text(code)
    save(out/'INPUTS.json', dict(parent=str(parent), changes=changes,
         parent_manifest_sha256=sha(parent/'INPUTS.json'),
         source_modified=False, runtime_modified=False,
         inputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out, configuration='config_flow2d3d.xml', timeout_s=180)), flush=True)


if __name__ == '__main__':
    main()
