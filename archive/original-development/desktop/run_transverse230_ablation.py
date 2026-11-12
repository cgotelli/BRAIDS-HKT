"""Isolate the source of the Jiang--Cao transverse row mode.

These are diagnostic compositions only.  They do not change the Desktop
project, installed runtime, public method identity, or saved checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUNTIME = PROJECT / "runtime"
OUT = ROOT / "validation/_transverse230/ablations"
MANIFEST = "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve the existing transverse ablations")
    sys.path[:0] = [str(RUNTIME), str(ROOT)]
    os.environ.update(
        OMP_NUM_THREADS="8",
        OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        QT_QPA_PLATFORM="offscreen",
    )
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install

    verify()
    install(json.loads((RUNTIME / "PRECOMPILED.json").read_text()), RUNTIME)
    if hashlib.sha256((RUNTIME / "RUNTIME.json").read_bytes()).hexdigest() != MANIFEST:
        raise ValueError("Frozen runtime differs")

    from braids_core import river_worker
    from braids_core.sediment.jiang_cao_coupled import JiangCaoCoupled
    from braids_core.sediment.jiang_cao_wetdry import JiangCaoShoreline

    original_relative = JiangCaoShoreline.relative_discharge
    original_momentum = JiangCaoShoreline.momentum_correction

    def no_momentum(self, state, rate):
        # Diagnostic ablation: leave the published/profile-derived bedload
        # steering active while removing only covariance momentum feedback.
        self.dispersion_flux = self.xp.zeros((2, self.nf))

    def gravity_only(self, state, capacity):
        # Diagnostic ablation: retain the common gravitational bed-slope drift
        # and remove both secondary steering and its covariance tensor.
        fields = self.xp.zeros((9, self.n))
        fields[7:] = JiangCaoCoupled.relative_discharge(self, state, capacity)
        self.secondary_fields = fields
        return fields[7:].copy()

    prepared = PROJECT / "prepared/experiment229"
    results = {}
    try:
        JiangCaoShoreline.momentum_correction = no_momentum
        results["no_secondary_momentum"] = river_worker.run(
            prepared,
            OUT / "runs/no_secondary_momentum_60",
            until=60.0,
            checkpoint_dir=OUT / "checkpoints/no_secondary_momentum_60",
        )

        JiangCaoShoreline.relative_discharge = gravity_only
        results["no_secondary_momentum_or_steering"] = river_worker.run(
            prepared,
            OUT / "runs/no_secondary_all_60",
            until=60.0,
            checkpoint_dir=OUT / "checkpoints/no_secondary_all_60",
        )
    finally:
        JiangCaoShoreline.relative_discharge = original_relative
        JiangCaoShoreline.momentum_correction = original_momentum

    save(
        OUT / "RESULTS.json",
        {
            "scope": "isolated diagnostic; not an admitted model or Desktop change",
            "frozen_runtime_sha256": MANIFEST,
            "time_s": 60.0,
            "results": results,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
