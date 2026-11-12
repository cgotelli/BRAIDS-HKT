"""Run the isolated 2-D compact Jiang--Cao candidate on the paper flume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = ROOT / "validation/_transverse230/candidate_2d_compact"
OUT = ROOT / "validation/_transverse230/candidate_run60"
EXPECTED = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve existing candidate trajectory")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8",
        OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        QT_QPA_PLATFORM="offscreen",
    )
    if hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Candidate runtime differs")
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install

    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core import river_worker

    prepared = prepare(
        PROJECT / "braids_project.json",
        OUT / "prepared",
        asset_cache=PROJECT / "cache",
    )
    result = river_worker.run(
        prepared,
        OUT / "run",
        until=60.0,
        checkpoint_dir=OUT / "checkpoints",
    )
    (OUT / "RESULT.json").write_text(
        json.dumps(
            {
                "runtime_sha256": EXPECTED,
                "scope": "isolated candidate; not installed",
                "result": result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
