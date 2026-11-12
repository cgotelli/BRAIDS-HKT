#!/usr/bin/env python3
"""Compare G4 unsplit/split physical cells independently of patch ordering."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools import g4_hydraulics as native
from validation.hydraulics.channels.run_verification import read_native, cell_arrays


def run():
    native.base.configure_runtime()
    target = native.REPORTS / "partition_comparisons.json"
    native.write_json(target, {"status": "RUNNING", "pass": False})
    report = {"scope": "G4 same-resolution unsplit versus split native cell states; not AMR",
              "method": "Match every output time and cell coordinate; sort by x then y; require bit-identical h, hu, hv and bed arrays. Ledger bytes differ across patch layouts and are not compared here.",
              "reproduce": ".venv/bin/python tools/compare_g4_partitions.py",
              "native_matrix_sha256": native.sha256(native.REPORTS / "native_results.json"),
              "pass": True, "cases": {}, "source_sha256": {
                  str(path.relative_to(ROOT)): native.sha256(path) for path in (
                      Path(__file__), native.VALIDATION / "run_verification.py",
                      ROOT / "validation/hydraulics/inlets/verify_native.py")}}
    for shape, experiment in ((shape, experiment) for shape in ("trapezoid", "semicircle")
                              for experiment in ("friction", "fill_wet", "fill_dry")):
        split_name = f"{shape}_fill_checkpoint" if experiment == "fill_wet" else f"{shape}_{experiment}_split_1t"
        paths = [native.RUNS / name / "Wave" for name in
                 (f"{shape}_{experiment}_dx0p25", split_name)]
        duration_s = 20. if experiment == "friction" else 60.
        first, second = (read_native(path) for path in paths)
        assert [frame[0] for frame in first] == [frame[0] for frame in second] == list(np.arange(0., duration_s+.1, 2.))
        frames = []
        for fa, fb in zip(first, second):
            arrays = [cell_arrays(frame[3]) for frame in (fa, fb)]
            arrays = [{key: values[np.lexsort((cells["y"], cells["x"]))]
                       for key, values in cells.items()} for cells in arrays]
            differences = {key: float(np.max(abs(arrays[0][key]-arrays[1][key])))
                           for key in ("x", "y", "h", "hu", "hv", "bed", "area")}
            matches = {key: arrays[0][key].shape == arrays[1][key].shape and
                       arrays[0][key].dtype == arrays[1][key].dtype and
                       arrays[0][key].tobytes() == arrays[1][key].tobytes() for key in differences}
            frames.append({"time_s": fa[0], "maximum_absolute_difference": differences,
                           "arrays_identical": matches, "pass": all(matches.values())})
        item = {"pass": all(frame["pass"] for frame in frames), "frames": frames,
                "execution_sha256": {str(path.relative_to(ROOT)): native.sha256(path / "g4_execution.json")
                                     for path in paths},
                "native_artifact_sha256": {str(path.relative_to(ROOT)): native.artifact_hashes(path)
                                           for path in paths}}
        report["cases"][shape if experiment == "friction" else f"{shape}_{experiment}"] = item
        report["pass"] &= item["pass"]
    assert all(native.sha256(ROOT / name) == digest for name, digest in report["source_sha256"].items())
    assert native.sha256(native.REPORTS / "native_results.json") == report["native_matrix_sha256"]
    report["status"] = "PASS" if report["pass"] else "FAIL"
    native.write_json(target, report)
    print(f"G4 unsplit/split cell comparison: {'PASS' if report['pass'] else 'FAIL'}: {target}")
    return report


if __name__ == "__main__":
    if not run()["pass"]:
        raise SystemExit(1)
