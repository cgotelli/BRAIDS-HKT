"""Seal the 300 s gate and install project-local long-run controls."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUN = PROJECT / "runs/experiment229_gate_300s"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def main() -> None:
    index = json.loads((RUN / "frames/INDEX.json").read_text(encoding="utf8"))
    final = index["frames"][-1]
    if float(final["time_s"]) != 300.0:
        raise ValueError("The exact 300 s frame is required")
    checkpoint = PROJECT / "checkpoints/experiment229_gate_300s" / (
        f't_300_step_{int(final["steps"]):08d}.npz')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    preserved = RUN / "SUPERVISED_INTERRUPT_PROGRESS.json"
    if not preserved.exists():
        original = json.loads((RUN / "PROGRESS.json").read_text(encoding="utf8"))
        save(preserved, original)
    status = {
        "status": "stopped",
        "error": None,
        "time_s": 300.0,
        "steps": int(final["steps"]),
        "backend": "cpu",
        "method": "bundle.jiang_cao_p15_bulk_diameters213_v1",
        "reason": "Supervised morphology gate sealed after exact frame and checkpoint",
        "restart_checkpoint": str(checkpoint),
        "cfl": .5,
        "dt_max_s": .03,
    }
    save(RUN / "STATUS.json", status)
    save(RUN / "PROGRESS.json", dict(status, percent=100., phase="supervised gate complete"))
    save(PROJECT / "SUPERVISED_GATE_300S.json", status)
    shutil.copy2(ROOT / "desktop/continue_experiment229.py",
                 PROJECT / "continue_experiment.py")
    python = ROOT / "desktop/.venv/Scripts/python.exe"
    (PROJECT / "Check long run.cmd").write_text(
        '@echo off\n"' + str(python) + '" -X utf8 "%~dp0continue_experiment.py" status\n',
        encoding="utf8")
    (PROJECT / "Stop long run safely.cmd").write_text(
        '@echo off\ntype nul > "%~dp0STOP_18000.request"\n'
        'echo Stop requested. BRAIDS will seal a checkpoint before exiting.\npause\n',
        encoding="utf8")
    sys.path.insert(0, str(ROOT / "desktop"))
    from braids_desktop import services
    services.update_project_session(PROJECT,
        prepared="prepared/experiment229",
        output="runs/experiment229_gate_300s",
        log="runs/experiment229_gate_300s/STATUS.json")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
