"""Read and format the packaged BRAIDS morphodynamics qualification."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


QUALIFICATION_FORMAT = "braids-phase2-qualification-v1"
QUALIFICATION_FILE = Path(__file__).resolve().parents[1] / "phase2_qualification.json"
PHASE3_QUALIFICATION_FORMAT = "braids-phase3-qualification-v1"
PHASE3_QUALIFICATION_FILE = Path(__file__).resolve().parents[1] / "phase3_qualification.json"


def load_phase2_qualification(path: str | Path | None = None) -> dict[str, Any]:
    source = QUALIFICATION_FILE if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != QUALIFICATION_FORMAT:
        raise ValueError("Unsupported BRAIDS Phase 2 qualification format")
    if payload.get("phase") != 2 or payload.get("steps_total") != 9:
        raise ValueError("Incomplete BRAIDS Phase 2 qualification scope")
    passed = payload.get("gates_passed")
    total = payload.get("gates_total")
    if not isinstance(passed, int) or not isinstance(total, int) or total <= 0 or not 0 <= passed <= total:
        raise ValueError("Invalid BRAIDS Phase 2 gate totals")
    milestones = payload.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("BRAIDS Phase 2 qualification has no milestone evidence")
    covered: list[int] = []
    for row in milestones:
        covered.extend(row.get("steps", []))
        if row.get("status") != "PASS" or row.get("gates_passed") != row.get("gates_total"):
            raise ValueError(f"Unqualified BRAIDS Phase 2 milestone: {row.get('case_id', '?')}")
        for key in ("metrics_sha256", "report_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))):
                raise ValueError(f"Invalid milestone evidence hash: {row.get('case_id', '?')}")
    if sorted(covered) != list(range(1, 10)):
        raise ValueError("BRAIDS Phase 2 milestone coverage is incomplete or duplicated")
    if payload.get("status") != "PASS" or passed != total or payload.get("steps_passed") != 9:
        raise ValueError("BRAIDS Phase 2 qualification is not passing")
    return payload


def format_phase2_qualification(payload: dict[str, Any]) -> str:
    results = payload["key_results"]
    lines = [
        f"{payload['title']}: {payload['status']}",
        f"Steps {payload['steps_passed']}/{payload['steps_total']} · Gates {payload['gates_passed']}/{payload['gates_total']}",
        f"Qualified {payload['qualified_on']} against " + " and ".join(payload["comparison_basis"]) + ".",
        "",
        "Key diagnostics",
        f"• Maximum BRAIDS total sediment-budget residual: {results['maximum_reported_braids_total_sediment_budget_relative_residual']:.3e}",
        f"• Fixed/AMR bed-profile L1: {100*results['fixed_amr_profile_l1']:.3f}% (correlation {results['fixed_amr_profile_correlation']:.6f})",
        f"• AMR average leaves / uniform: {100*results['amr_average_leaf_fraction']:.1f}%",
        f"• Spatial solution-distance contraction: {results['grid_pair_contraction_ratio']:.4f}",
        f"• Timestep Cauchy envelope: {results['time_cauchy_envelope_profile_l1']:.3e}",
        f"• One/eight-thread state difference: {results['thread_maximum_state_difference']:.1e}",
        f"• Restart state difference: {results['restart_maximum_state_difference']:.1e}; checkpoint binary identity: {'yes' if results['restart_checkpoint_binary_identical'] else 'no'}",
        "",
        "Milestones",
    ]
    for row in payload["milestones"]:
        steps = ",".join(map(str, row["steps"]))
        lines.append(f"• Step {steps}: {row['title']} — {row['status']} ({row['gates_passed']}/{row['gates_total']}) · evidence {row['metrics_sha256'][:12]}…")
    lines += ["", "Qualified scope", *[f"• {item}" for item in payload["scope"]],
              "", "Limitations", *[f"• {item}" for item in payload["limitations"]]]
    return "\n".join(lines)


def load_phase3_qualification(path: str | Path | None = None) -> dict[str, Any]:
    """Load the packaged Phase 3 manifest and reject incomplete evidence."""
    source = PHASE3_QUALIFICATION_FILE if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != PHASE3_QUALIFICATION_FORMAT:
        raise ValueError("Unsupported BRAIDS Phase 3 qualification format")
    if payload.get("phase") != 3 or payload.get("milestones_total") != 3:
        raise ValueError("Incomplete BRAIDS Phase 3 qualification scope")
    passed = payload.get("gates_passed"); total = payload.get("gates_total")
    if not isinstance(passed, int) or not isinstance(total, int) or total <= 0 or not 0 <= passed <= total:
        raise ValueError("Invalid BRAIDS Phase 3 gate totals")
    milestones = payload.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("BRAIDS Phase 3 qualification has no milestone evidence")
    covered: list[int] = []
    for row in milestones:
        covered.append(row.get("milestone"))
        if row.get("status") != "PASS" or row.get("gates_passed") != row.get("gates_total"):
            raise ValueError(f"Unqualified BRAIDS Phase 3 milestone: {row.get('case_id', '?')}")
        for key in ("metrics_sha256", "report_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))):
                raise ValueError(f"Invalid milestone evidence hash: {row.get('case_id', '?')}")
    if covered != [1, 2, 3]:
        raise ValueError("BRAIDS Phase 3 milestone coverage is incomplete or duplicated")
    if payload.get("status") != "PASS" or passed != total or payload.get("milestones_passed") != 3:
        raise ValueError("BRAIDS Phase 3 qualification is not passing")
    results = payload.get("key_results", {})
    rejection = results.get("vertical_diffusion_rejection", {})
    if rejection.get("success") is not False or not rejection.get("negative_layer_mass_signature"):
        raise ValueError("BRAIDS Phase 3 rejected-mode evidence is incomplete")
    return payload


def format_phase3_qualification(payload: dict[str, Any]) -> str:
    """Render the Phase 3 scientific status for the QGIS diagnostics panel."""
    results = payload["key_results"]; decision = results["bed_composition_mode_decision"]
    rejection = results["vertical_diffusion_rejection"]
    lines = [
        f"{payload['title']}: {payload['status']}",
        f"Milestones {payload['milestones_passed']}/{payload['milestones_total']} - Gates {payload['gates_passed']}/{payload['gates_total']}",
        f"Qualified {payload['qualified_on']} against " + " and ".join(payload["comparison_basis"]) + ".",
        "", "Key diagnostics",
        f"- Minimum surface-sorting correlation: {results['minimum_surface_sorting_correlation']:.6f}",
        f"- Minimum graded bed-profile correlation: {results['minimum_graded_bed_profile_correlation']:.6f}",
        f"- Buried-pulse bed-profile correlation: {results['stratigraphy_bed_change_profile_correlation']:.6f}",
        f"- Depth-scaled active-layer MAE: {1000*results['depth_scaled_active_layer_profile_mae_m']:.6f} mm (correlation {results['depth_scaled_active_layer_profile_correlation']:.6f})",
        f"- Maximum BRAIDS sediment-budget residual: {results['maximum_braids_total_sediment_budget_relative_residual']:.3e}",
        f"- One/eight-thread state difference: {results['one_eight_thread_maximum_state_difference']:.1e}",
        f"- Rejected vertical diffusion: exit {rejection['return_code']} at timestep {rejection['failure_timestep']} (negative layer mass and hydraulic blow-up)",
        "", "Milestones",
    ]
    for row in payload["milestones"]:
        lines.append(f"- {row['milestone']}: {row['title']} - {row['status']} ({row['gates_passed']}/{row['gates_total']}) - evidence {row['metrics_sha256'][:12]}...")
    lines += ["", "Retained bed-composition modes", *[f"- {item}" for item in decision["retain"]],
              "", "Excluded modes", *[f"- {item}" for item in decision["exclude"]],
              "", "Qualified scope", *[f"- {item}" for item in payload["scope"]],
              "", "Limitations", *[f"- {item}" for item in payload["limitations"]]]
    return "\n".join(lines)
