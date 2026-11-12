"""Continue the isolated 2-D compact candidate from 60 to 300 seconds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "validation/_transverse230/candidate_2d_compact"
BASE = ROOT / "validation/_transverse230/candidate_run60"
OUT = ROOT / "validation/_transverse230/candidate_restart300"
EXPECTED = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve existing 300 s candidate continuation")
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
    from braids_core import river_worker

    restart = BASE / "checkpoints/t_60_step_00003760.npz"
    result = river_worker.run(
        BASE / "prepared",
        OUT / "run",
        restart=restart,
        until=300.0,
        checkpoint_dir=OUT / "checkpoints",
    )
    (OUT / "RESULT.json").write_text(
        json.dumps(
            {
                "runtime_sha256": EXPECTED,
                "restart": str(restart),
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
