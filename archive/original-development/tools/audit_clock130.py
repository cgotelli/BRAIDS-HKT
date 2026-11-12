"""Read sealed alpha129 evidence; separate physical dt from clock remainders."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def summary(path):
    data = json.loads(path.read_text())
    steps = data["steps"]
    diag = [x["diagnostics"] for x in steps]
    events = json.loads((path.parent / "run" / "ADAPTATION.json").read_text())
    times = {round(e["time_s"], 10) for e in events if e["changed"]}
    rejects = [r for d in diag for r in d["hydraulic_rejections"]]
    count = sum(d["hydraulic_substeps"] for d in diag)
    remainder = sum(d["hydraulic_endpoint_remainders"] for d in diag)
    fwave = sum(d["fwave_evaluations"] for d in diag)
    return dict(case=path.parent.name, source=str(path.relative_to(ROOT)),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), runtime=data["runtime"],
        duration_s=data["status"]["time_s"], outer_transactions=len(steps),
        accepted_hydraulic_substeps=count, endpoint_remainders=remainder,
        remainder_step_fraction=remainder/count,
        minimum_substep_s=min(d["hydraulic_minimum_dt"] for d in diag),
        maximum_substep_s=max(d["hydraulic_maximum_dt"] for d in diag),
        late_next_hint_s=diag[-1]["hydraulic_next_substep_hint_s"],
        mean_substep_s=data["status"]["time_s"]/count,
        hard_accepted_cfl=max(d["accepted_cfl_max"] for d in diag),
        changed_amr_events=len(times), total_rejections=len(rejects),
        rejections_immediately_after_amr=sum(round(r["time"], 10) in times for r in rejects),
        startup_rejections=sum(r["time"] == 0 for r in rejects),
        fwave_evaluations=fwave, extra_fwave_evaluations=fwave-2*count,
        note="A remainder is a real accepted step, not a physical stability limit; later hints remain unconstrained by it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"validation/_speed130/CLOCK_AUDIT.json")
    args = parser.parse_args()
    rows = [summary(p) for p in sorted((ROOT/"validation/_inlet129").glob("natural*m_final/*/QUALIFICATION.json"))]
    examples = []
    for physical_hint in (.07, .19):
        for safety in (.9, .98):
            hint = physical_hint*safety/.9
            for outer in (.5, 1., 2.):
                examples.append(dict(original_hint_s=physical_hint,safety=safety,outer_s=outer,
                    estimated_hint_s=hint, steps_per_simulated_second=math.ceil(outer/hint)/outer))
    document = dict(scope="read_only_alpha129_clock_audit", cases=rows,
        fixed_frequency_clock_examples=examples,
        example_warning="Illustrative ceiling arithmetic, not measured speed or proof of stability; AMR/forcing/output clocks still clip outer steps.",
        conclusions=[
            "The remaining 2 m baseline has only two startup retries; reducing retries cannot explain its main speed gap.",
            "Nearly every dynamic 5 m owner replacement loses the accepted hint and retries dt_max once.",
            "A single minimum timestep can be an output/outer-clock remainder; use the accepted distribution and late unconstrained hint.",
            "The 0.9 predictor is distinct from the unchanged hard stage CFL cap; testing 0.98 retains explicit stage rejection.",
            "Large outer chunks and near-limit hints can combine nonlinearly through ceiling/fragmentation; changing only one may save no hydraulic stages.",
            "No wave-frequency predictor is separately evaluated: the CFL kernel reduces already-computed face speeds after each actual spatial residual.",
        ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),cases=len(rows)),indent=2))


if __name__ == "__main__":
    main()
